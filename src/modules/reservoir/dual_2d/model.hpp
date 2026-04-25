#pragma once
#include "lib/interfaces.hpp"
#include "state.hpp"
#include "lib/discretization.hpp"
#include <vector>

namespace mod::reservoir {
using namespace top;

/**
 * @brief 2D Dual-Phase (Oil-Water) Reservoir Physical Model (Properties).
 */
class DualPhase2DModel : public IModel {
public:
    double mu_w, mu_o; // viscosities [cP]
    std::vector<std::shared_ptr<ISourceSink>> wells;
    double sw_res;     // residual water saturation
    double so_res;     // residual oil saturation
    std::shared_ptr<num::discretization::Conductance2D> rock_cond;
    double pore_vol_per_cell; 

    DualPhase2DModel(std::shared_ptr<num::discretization::Conductance2D> cond, double pv, double mw, double mo, 
                     const std::vector<std::shared_ptr<ISourceSink>>& wells_val)
        : rock_cond(cond), pore_vol_per_cell(pv), mu_w(mw), mu_o(mo), wells(wells_val), 
          sw_res(0.2), so_res(0.2) {}

    double get_tolerance() const override { return 1e-4; }

    void get_rel_perm(double sw, double& krw, double& kro) const {
        double swe = (sw - sw_res) / (1.0 - sw_res - so_res);
        swe = std::max(0.0, std::min(1.0, swe));
        krw = swe * swe;
        kro = (1.0 - swe) * (1.0 - swe);
    }

    Vector get_accumulation_weights(const IGrid& grd, const IState& st) const override {
        size_t n = st.to_vector().size();
        return Vector(n, pore_vol_per_cell);
    }
};

/**
 * @brief 2D Dual-Phase FVM Discretizer (Numerical Assembly).
 */
class DualPhase2DDiscretizer : public IDiscretizer {
public:
    void build_jacobian(const IGrid& grd, const IModel& mdl, const IState& st, SparseMatrix& J) const override {
        const auto& m = static_cast<const DualPhase2DModel&>(mdl);
        const auto& s = static_cast<const ReservoirDualPhase2DState&>(st);
        int nx = (int)s.spatial->nx;
        int ny = (int)s.spatial->ny;
        int n = nx * ny;

        if (J.rows != 2 * n) J = SparseMatrix(2 * n, 2 * n);
        J.triplets.clear();

        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int c = s.spatial->idx(i, j);
                
                // Diagonal accumulation sensitivities
                J.triplets.push_back({2 * c, 2 * c, 1e-4}); // Compressibility
                J.triplets.push_back({2 * c, 2 * c + 1, 1.0}); // dRw/dSw
                J.triplets.push_back({2 * c + 1, 2 * c + 1, -1.0}); // dRo/dSw

                auto add_flux_jac = [&](int n_idx, double t_rock) {
                    int up = (s.p(c) > s.p(n_idx)) ? c : n_idx;
                    double krw, kro;
                    m.get_rel_perm(s.sw(up), krw, kro);
                    
                    double Tw = t_rock * (krw / m.mu_w);
                    double To = t_rock * (kro / m.mu_o);

                    J.triplets.push_back({2 * c, 2 * c, Tw});
                    J.triplets.push_back({2 * c, 2 * n_idx, -Tw});
                    J.triplets.push_back({2 * c + 1, 2 * c, To});
                    J.triplets.push_back({2 * c + 1, 2 * n_idx, -To});
                };

                if (i > 0)      add_flux_jac(s.spatial->idx(i - 1, j), m.rock_cond->Tx[j * (nx - 1) + i - 1]);
                if (i < nx - 1) add_flux_jac(s.spatial->idx(i + 1, j), m.rock_cond->Tx[j * (nx - 1) + i]);
                if (j > 0)      add_flux_jac(s.spatial->idx(i, j - 1), m.rock_cond->Ty[(j - 1) * nx + i]);
                if (j < ny - 1) add_flux_jac(s.spatial->idx(i, j + 1), m.rock_cond->Ty[j * nx + i]);
            }
        }
    }

    void build_residual(const IGrid& grd, const IModel& mdl, const IState& st, Vector& R) const override {
        const auto& m = static_cast<const DualPhase2DModel&>(mdl);
        const auto& s = static_cast<const ReservoirDualPhase2DState&>(st);
        int nx = (int)s.spatial->nx;
        int ny = (int)s.spatial->ny;

        #pragma omp parallel for collapse(2)
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int c = s.spatial->idx(i, j);
                double p_c = s.p(c);
                double net_w = 0, net_o = 0;

                auto add_flux_res = [&](int n_idx, double t_rock) {
                    double p_n = s.p(n_idx);
                    int up = (p_c > p_n) ? c : n_idx;
                    double krw_u, kro_u;
                    m.get_rel_perm(s.sw(up), krw_u, kro_u);

                    net_w += t_rock * (krw_u / m.mu_w) * (p_n - p_c);
                    net_o += t_rock * (kro_u / m.mu_o) * (p_n - p_c);
                };

                if (i > 0)      add_flux_res(s.spatial->idx(i - 1, j), m.rock_cond->Tx[j * (nx - 1) + i - 1]);
                if (i < nx - 1) add_flux_res(s.spatial->idx(i + 1, j), m.rock_cond->Tx[j * (nx - 1) + i]);
                if (j > 0)      add_flux_res(s.spatial->idx(i, j - 1), m.rock_cond->Ty[(j - 1) * nx + i]);
                if (j < ny - 1) add_flux_res(s.spatial->idx(i, j + 1), m.rock_cond->Ty[j * nx + i]);

                R[2*c] = -net_w;
                R[2*c+1] = -net_o;
            }
        }
    }

    void apply_bc(const IGrid& grd, const IModel& mdl, const IState& st, SparseMatrix& J, Vector& R) const override {
        // No-flow is default.
    }
};

} // namespace mod::reservoir
