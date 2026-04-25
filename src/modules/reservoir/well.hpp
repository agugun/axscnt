#pragma once
#include "lib/interfaces.hpp"
#include <vector>
#include <memory>
#include <cmath>
#include <functional>

namespace mod {
using namespace top;

/**
 * @brief Base class for Grid-based Wells implementing ISourceSink.
 */
class IWell : public ISourceSink {
public:
    int i, j, k_min, k_max;
    double q_total;
    bool is_injector;

    IWell(int i_v, int j_v, int kmin, int kmax, double q, bool inj)
        : i(i_v), j(j_v), k_min(kmin), k_max(kmax), q_total(q), is_injector(inj) {}
    
    virtual double get_q_water(const top::IState& st) const = 0;
};

/**
 * @brief ConstantRateWell for single-phase flow.
 */
class ConstantRateWell : public IWell {
public:
    using IndexFunc = std::function<int(int, int, int)>;
    IndexFunc get_idx;
    double scale_factor;

    ConstantRateWell(int i_v, int j_v, int kmin, int kmax, double q, double scale, IndexFunc idx)
        : IWell(i_v, j_v, kmin, kmax, q, false), scale_factor(scale), get_idx(idx) {}

    void assemble_terms(const IState& st, SparseMatrix& J, Vector& R) const override {
        int num_layers = k_max - k_min + 1;
        double q_scaled = (q_total * scale_factor) / num_layers;

        for (int k = k_min; k <= k_max; ++k) {
            int c = get_idx(i, j, k);
            R[c] -= q_scaled; // Wells as Source term (Residual R = F - Q)
        }
    }

    double get_q_water(const top::IState& st) const override { return 0.0; }
};

/**
 * @brief Base class for Dual-Phase Wells with Water/Oil physics.
 */
class DualPhaseWell : public IWell {
public:
    using RelPermFunc = std::function<void(double, double&, double&)>;
    using IndexFunc = std::function<int(int, int)>;
    using SwFunc = std::function<double(const top::IState&, int, int)>;
    
    RelPermFunc get_rel_perm;
    IndexFunc get_idx;
    SwFunc get_sw; 
    double mu_w, mu_o;

    DualPhaseWell(int i_v, int j_v, double q, bool inj, RelPermFunc rp, IndexFunc idx, SwFunc sw_f, double mw, double mo)
        : IWell(i_v, j_v, 0, 0, q, inj), get_rel_perm(rp), get_idx(idx), get_sw(sw_f), mu_w(mw), mu_o(mo) {}

    void assemble_terms(const IState& st, SparseMatrix& J, Vector& R) const override {}
    
    double get_q_water(const top::IState& st) const override {
        if (is_injector) return q_total;
        
        double krw, kro;
        get_rel_perm(get_sw(st, i, j), krw, kro);
        double fw = (krw / mu_w) / ((krw / mu_w) + (kro / mu_o));
        return q_total * fw;
    }
};

/**
 * @brief Specialized well for 2D Dual Phase Reservoir (Fully Implicit support).
 */
class ReservoirWellDual2D : public DualPhaseWell {
public:
    ReservoirWellDual2D(int i_v, int j_v, double q, bool inj, RelPermFunc rp, IndexFunc idx, SwFunc sw_f, double mw, double mo)
        : DualPhaseWell(i_v, j_v, q, inj, rp, idx, sw_f, mw, mo) {}

    void assemble_terms(const IState& st, SparseMatrix& J, Vector& R) const override {
        int c = get_idx(i, j);
        int r_w = 2 * c;
        int r_o = 2 * c + 1;

        if (is_injector) {
            R[r_w] -= q_total; // Inject water
        } else {
            double krw, kro;
            double sw_val = get_sw(st, i, j);
            get_rel_perm(sw_val, krw, kro);
            double lam_w = krw / mu_w;
            double lam_o = kro / mu_o;
            double lam_t = lam_w + lam_o;
            double fw = lam_w / lam_t;
            
            R[r_w] -= q_total * fw;
            R[r_o] -= q_total * (1.0 - fw);

            // Jacobian terms for Wells (Implicit Sw dependence)
            double sw_res = 0.2, so_res = 0.2; 
            double den_sw = 1.0 - sw_res - so_res;
            double swe = (sw_val - sw_res) / den_sw;
            double dkrw_dsw = (sw_val > sw_res && sw_val < (1.0 - so_res)) ? (2.0 * swe / den_sw) : 0.0;
            double dkro_dsw = (sw_val > sw_res && sw_val < (1.0 - so_res)) ? (-2.0 * (1.0 - swe) / den_sw) : 0.0;
            
            double dlamw_dsw = dkrw_dsw / mu_w;
            double dlamo_dsw = dkro_dsw / mu_o;
            double dfw_dsw = (dlamw_dsw * lam_o - lam_w * dlamo_dsw) / (lam_t * lam_t);
            
            J.triplets.push_back({r_w, r_w + 1, -q_total * dfw_dsw});
            J.triplets.push_back({r_o, r_w + 1, q_total * dfw_dsw});
        }
    }
};

/**
 * @brief Specialized well for 2D Black Oil Reservoir (3-Phase support).
 */
class ReservoirWellBlackOil2D : public IWell {
public:
    using IndexFunc = std::function<int(int, int)>;
    using VarFunc = std::function<void(const top::IState&, int, int, double&, double&, double&)>;
    using RelPerm3Func = std::function<void(double, double, double&, double&, double&)>;

    IndexFunc get_idx;
    VarFunc get_vars;
    RelPerm3Func get_rel_perm;
    double mu_w, mu_o, mu_g;

    ReservoirWellBlackOil2D(int i_v, int j_v, double q, bool inj, RelPerm3Func rp, IndexFunc idx, VarFunc var_f, double mw, double mo, double mg)
        : IWell(i_v, j_v, 0, 0, q, inj), get_rel_perm(rp), get_idx(idx), get_vars(var_f), mu_w(mw), mu_o(mo), mu_g(mg) {}

    void assemble_terms(const IState& st, SparseMatrix& J, Vector& R) const override {
        int c = get_idx(i, j);
        int r_p = 3 * c;
        int r_sw = 3 * c + 1;
        int r_sg = 3 * c + 2;

        if (is_injector) {
            R[r_p] -= q_total;
        } else {
            double p, sw, sg;
            get_vars(st, i, j, p, sw, sg);
            double krw, kro, krg;
            get_rel_perm(sw, sg, krw, kro, krg);
            
            double mob_w = krw/mu_w, mob_o = kro/mu_o, mob_g = krg/mu_g;
            double mob_t = mob_w + mob_o + mob_g;
            
            R[r_p]   -= q_total * (mob_w / mob_t);
            R[r_sw]  -= q_total * (mob_o / mob_t);
            R[r_sg]  -= q_total * (mob_g / mob_t);
        }
    }
    
    double get_q_water(const top::IState& st) const override { return is_injector ? q_total : 0.0; }
};

/**
 * @brief Specialized well for 3D Black Oil Reservoir (3-Phase support).
 */
class ReservoirWellBlackOil3D : public IWell {
public:
    using IndexFunc = std::function<int(int, int, int)>;
    using VarFunc = std::function<void(const top::IState&, int, int, int, double&, double&, double&)>;
    using RelPerm3Func = std::function<void(double, double, double&, double&, double&)>;

    IndexFunc get_idx;
    VarFunc get_vars;
    RelPerm3Func get_rel_perm;
    double mu_w, mu_o, mu_g;

    ReservoirWellBlackOil3D(int i_v, int j_v, int kmin, int kmax, double q, bool inj, RelPerm3Func rp, IndexFunc idx, VarFunc var_f, double mw, double mo, double mg)
        : IWell(i_v, j_v, kmin, kmax, q, inj), get_rel_perm(rp), get_idx(idx), get_vars(var_f), mu_w(mw), mu_o(mo), mu_g(mg) {}

    void assemble_terms(const IState& st, SparseMatrix& J, Vector& R) const override {
        int num_layers = k_max - k_min + 1;
        double q_layer = q_total / num_layers;

        for (int k = k_min; k <= k_max; ++k) {
            int c = get_idx(i, j, k);
            int r_p = 3 * c;
            int r_sw = 3 * c + 1;
            int r_sg = 3 * c + 2;

            if (is_injector) {
                R[r_p] -= q_layer;
            } else {
                double p, sw, sg;
                get_vars(st, i, j, k, p, sw, sg);
                double krw, kro, krg;
                get_rel_perm(sw, sg, krw, kro, krg);
                
                double mob_w = krw/mu_w, mob_o = kro/mu_o, mob_g = krg/mu_g;
                double mob_t = mob_w + mob_o + mob_g;
                
                R[r_p]   -= q_layer * (mob_w / mob_t);
                R[r_sw]  -= q_layer * (mob_o / mob_t);
                R[r_sg]  -= q_layer * (mob_g / mob_t);
            }
        }
    }
    
    double get_q_water(const top::IState& st) const override { return is_injector ? q_total : 0.0; }
};

} // namespace mod
