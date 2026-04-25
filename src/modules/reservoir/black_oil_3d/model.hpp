#pragma once
#include "lib/interfaces.hpp"
#include "state.hpp"
#include "../pvt.hpp"
#include "lib/discretization.hpp"
#include <vector>
 
namespace mod::reservoir {
using namespace top;

/**
 * @brief 3D Black Oil Physical Model (Properties and Mobilities).
 */
class BlackOil3DModel : public IModel {
public:
    BlackOilPVT pvt;
    std::shared_ptr<num::discretization::Conductance3D> rock_cond;
    double pore_vol_per_cell; 
    std::vector<std::shared_ptr<ISourceSink>> wells;

    BlackOil3DModel(std::shared_ptr<num::discretization::Conductance3D> cond, double pv, 
                    const std::vector<std::shared_ptr<ISourceSink>>& wells_val)
        : rock_cond(cond), pore_vol_per_cell(pv), wells(wells_val) {}

    double get_tolerance() const override { return 1e-4; }

    void get_rel_perm(double sw, double sg, double& krw, double& krog, double& krg) const {
        double swr = 0.2, sorg = 0.1, sgr = 0.05, swc = 0.2;
        double swe = (sw - swr) / (1.0 - swr - sorg);
        krw = std::pow(std::max(0.0, std::min(1.0, swe)), 2.0);
        double sge = (sg - sgr) / (1.0 - swc - sgr);
        krg = std::pow(std::max(0.0, std::min(1.0, sge)), 2.0);
        double so = 1.0 - sw - sg;
        double soe = (so - sorg) / (1.0 - swr - sorg);
        krog = std::pow(std::max(0.0, std::min(1.0, soe)), 3.0);
    }

    Vector get_accumulation_weights(const IGrid& grd, const IState& st) const override {
        size_t n = grd.get_total_cells();
        return Vector(3 * n, pore_vol_per_cell);
    }

    const std::vector<std::shared_ptr<ISourceSink>>& get_sources() const {
        return wells;
    }
};

/**
 * @brief 3D Black Oil FVM Discretizer (FIM implementation).
 */
class BlackOil3DDiscretizer : public IDiscretizer {
public:
    void build_jacobian(const IGrid& grd, const IModel& mdl, const IState& st, SparseMatrix& J) const override {
        const auto& m = static_cast<const BlackOil3DModel&>(mdl);
        const auto& s = static_cast<const ReservoirBlackOil3DState&>(st);
        int nx = (int)s.spatial->nx;
        int ny = (int)s.spatial->ny;
        int nz = (int)s.spatial->nz;
        int n = nx * ny * nz;

        if (J.rows != 3 * n) J = SparseMatrix(3 * n, 3 * n);
        J.triplets.clear();

        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    int c = s.spatial->idx(i, j, k);
                    double p = s.p(c), sw = s.sw(c), sg = s.sg(c);
                    double bw = m.pvt.get_bw(p), bo = m.pvt.get_bo(p, m.pvt.get_rs(p));
                    
                    // Diagonal block accumulation coupling
                    J.triplets.push_back({3 * c, 3 * c + 1, 1.0 / bw}); 
                    J.triplets.push_back({3 * c + 1, 3 * c + 1, -1.0 / bo});
                    J.triplets.push_back({3 * c + 1, 3 * c + 2, -1.0 / bo});

                    auto add_flux_jac = [&](int n_idx, double t_rock) {
                        int up = (s.p(c) > s.p(n_idx)) ? c : n_idx;
                        double krw, krog, krg;
                        m.get_rel_perm(s.sw(up), s.sg(up), krw, krog, krg);
                        
                        double p_up = s.p(up);
                        double rs_u = m.pvt.get_rs(p_up);
                        double bwu = m.pvt.get_bw(p_up), bou = m.pvt.get_bo(p_up, rs_u), bgu = m.pvt.get_bg(p_up);
                        double muw = m.pvt.get_mu_w(p_up), muo = m.pvt.get_mu_o(p_up, rs_u), mug = m.pvt.get_mu_g(p_up);

                        double Tw = t_rock * krw / (muw * bwu);
                        double To = t_rock * krog / (muo * bou);
                        double Tg = t_rock * (krg / (mug * bgu) + rs_u * krog / (muo * bou));

                        J.triplets.push_back({3 * c, 3 * c, Tw});
                        J.triplets.push_back({3 * c, 3 * n_idx, -Tw});
                        J.triplets.push_back({3 * c + 1, 3 * c, To});
                        J.triplets.push_back({3 * c + 1, 3 * n_idx, -To});
                        J.triplets.push_back({3 * c + 2, 3 * c, Tg});
                        J.triplets.push_back({3 * c + 2, 3 * n_idx, -Tg});
                    };

                    if (i > 0)      add_flux_jac(s.spatial->idx(i - 1, j, k), m.rock_cond->Tx[k * (nx - 1) * ny + j * (nx - 1) + i - 1]);
                    if (i < nx - 1) add_flux_jac(s.spatial->idx(i + 1, j, k), m.rock_cond->Tx[k * (nx - 1) * ny + j * (nx - 1) + i]);
                    if (j > 0)      add_flux_jac(s.spatial->idx(i, j - 1, k), m.rock_cond->Ty[k * nx * (ny - 1) + (j - 1) * nx + i]);
                    if (j < ny - 1) add_flux_jac(s.spatial->idx(i, j + 1, k), m.rock_cond->Ty[k * nx * (ny - 1) + j * nx + i]);
                    if (k > 0)      add_flux_jac(s.spatial->idx(i, j, k - 1), m.rock_cond->Tz[(k - 1) * nx * ny + j * nx + i]);
                    if (k < nz - 1) add_flux_jac(s.spatial->idx(i, j, k + 1), m.rock_cond->Tz[k * nx * ny + j * nx + i]);
                }
            }
        }
    }

    void build_residual(const IGrid& grd, const IModel& mdl, const IState& st, Vector& R) const override {
        const auto& m = static_cast<const BlackOil3DModel&>(mdl);
        const auto& s = static_cast<const ReservoirBlackOil3DState&>(st);
        int nx = (int)s.spatial->nx;
        int ny = (int)s.spatial->ny;
        int nz = (int)s.spatial->nz;

        #pragma omp parallel for collapse(3)
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    int c = s.spatial->idx(i, j, k);
                    double p_c = s.p(c);
                    double net_w = 0, net_o = 0, net_g = 0;

                    auto add_flux_res = [&](int n_idx, double t_rock) {
                        double p_n = s.p(n_idx);
                        int up = (p_c > p_n) ? c : n_idx;
                        double krw_u, krog_u, krg_u;
                        m.get_rel_perm(s.sw(up), s.sg(up), krw_u, krog_u, krg_u);
                        
                        double p_u = s.p(up);
                        double rs_u = m.pvt.get_rs(p_u);
                        double lam_w = krw_u / (m.pvt.get_mu_w(p_u) * m.pvt.get_bw(p_u));
                        double lam_o = krog_u / (m.pvt.get_mu_o(p_u, rs_u) * m.pvt.get_bo(p_u, rs_u));
                        double lam_g = krg_u / (m.pvt.get_mu_g(p_u) * m.pvt.get_bg(p_u)) + rs_u * lam_o;

                        net_w += t_rock * lam_w * (p_n - p_c);
                        net_o += t_rock * lam_o * (p_n - p_c);
                        net_g += t_rock * lam_g * (p_n - p_c);
                    };

                    if (i > 0)      add_flux_res(s.spatial->idx(i - 1, j, k), m.rock_cond->Tx[k * (nx - 1) * ny + j * (nx - 1) + i - 1]);
                    if (i < nx - 1) add_flux_res(s.spatial->idx(i + 1, j, k), m.rock_cond->Tx[k * (nx - 1) * ny + j * (nx - 1) + i]);
                    if (j > 0)      add_flux_res(s.spatial->idx(i, j - 1, k), m.rock_cond->Ty[k * nx * (ny - 1) + (j - 1) * nx + i]);
                    if (j < ny - 1) add_flux_res(s.spatial->idx(i, j + 1, k), m.rock_cond->Ty[k * nx * (ny - 1) + j * nx + i]);
                    if (k > 0)      add_flux_res(s.spatial->idx(i, j, k - 1), m.rock_cond->Tz[(k - 1) * nx * ny + j * nx + i]);
                    if (k < nz - 1) add_flux_res(s.spatial->idx(i, j, k + 1), m.rock_cond->Tz[k * nx * ny + j * nx + i]);

                    R[3*c] = -net_w;
                    R[3*c+1] = -net_o;
                    R[3*c+2] = -net_g;
                }
            }
        }
    }

    void apply_bc(const IGrid& grd, const IModel& mdl, const IState& st, SparseMatrix& J, Vector& R) const override {
        // No-flow is default.
    }
};

} // namespace mod::reservoir
