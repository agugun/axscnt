#pragma once
#include "lib/interfaces.hpp"
#include "state.hpp"
#include "lib/discretization.hpp"
 
namespace mod::wave {
using namespace top;

/**
 * @brief 2D Wave Physical Model (Properties).
 */
class Wave2DModel : public IModel {
public:
    std::shared_ptr<num::discretization::Conductance2D> cond;
    Vector storage_coeff;

    Wave2DModel(std::shared_ptr<num::discretization::Conductance2D> c, const Vector& storage) 
        : cond(c), storage_coeff(storage) {}

    double get_tolerance() const override { return 1e-4; }

    Vector get_accumulation_weights(const IGrid& grd, const IState& st) const override {
        size_t n = storage_coeff.size();
        Vector weights(2 * n);
        for (size_t i = 0; i < n; ++i) {
            weights[i] = 1.0;          // u accumulation weight
            weights[i + n] = storage_coeff[i]; // v accumulation weight (mass)
        }
        return weights;
    }
};

/**
 * @brief 2D Wave FVM Discretizer (Numerical Assembly).
 * Assembles the first-order system:
 *   du/dt = v
 *   m*dv/dt = Sum(Flux)
 */
class Wave2DDiscretizer : public IDiscretizer {
public:
    void build_jacobian(const IGrid& grd, const IModel& mdl, const IState& st, SparseMatrix& J) const override {
        const auto& w_model = static_cast<const Wave2DModel&>(mdl);
        const auto& w_state = static_cast<const Wave2DState&>(st);
        int n = (int)w_state.u.size();
        int nx = (int)w_state.spatial->nx;
        int ny = (int)w_state.spatial->ny;

        if (J.rows != 2 * n) J = SparseMatrix(2 * n, 2 * n);
        J.triplets.clear();

        // 1. du/dt = v part -> J_uv = -I
        for (int i = 0; i < n; ++i) {
            J.triplets.push_back({i, i + n, -1.0});
        }

        // 2. m*dv/dt = Laplacian(u) part -> J_vu = -Laplacian
        for (int j = 1; j < ny - 1; ++j) {
            for (int i = 1; i < nx - 1; ++i) {
                int cur = w_state.spatial->idx(i, j);
                int cur_v = cur + n;
                
                double tx_prev = w_model.cond->Tx[j*(nx-1) + i-1];
                double tx_next = w_model.cond->Tx[j*(nx-1) + i];
                double ty_prev = w_model.cond->Ty[(j-1)*nx + i];
                double ty_next = w_model.cond->Ty[j*nx + i];

                J.triplets.push_back({cur_v, (int)w_state.spatial->idx(i-1, j), -tx_prev});
                J.triplets.push_back({cur_v, (int)w_state.spatial->idx(i+1, j), -tx_next});
                J.triplets.push_back({cur_v, (int)w_state.spatial->idx(i, j-1), -ty_prev});
                J.triplets.push_back({cur_v, (int)w_state.spatial->idx(i, j+1), -ty_next});
                J.triplets.push_back({cur_v, cur, tx_prev + tx_next + ty_prev + ty_next});
            }
        }
    }

    void build_residual(const IGrid& grd, const IModel& mdl, const IState& st, Vector& R) const override {
        const auto& w_model = static_cast<const Wave2DModel&>(mdl);
        const auto& w_state = static_cast<const Wave2DState&>(st);
        int n = (int)w_state.u.size();
        int nx = (int)w_state.spatial->nx;
        int ny = (int)w_state.spatial->ny;

        // 1. R_u = -v
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            R[i] = -w_state.v[i];
        }

        // 2. R_v = -NetFlux(u)
        #pragma omp parallel for collapse(2)
        for (int j = 1; j < ny - 1; ++j) {
            for (int i = 1; i < nx - 1; ++i) {
                int cur = w_state.spatial->idx(i, j);
                double net_flux = 
                    w_model.cond->Tx[j*(nx-1) + i-1] * (w_state.u[w_state.spatial->idx(i-1,j)] - w_state.u[cur]) +
                    w_model.cond->Tx[j*(nx-1) + i]   * (w_state.u[w_state.spatial->idx(i+1,j)] - w_state.u[cur]) +
                    w_model.cond->Ty[(j-1)*nx + i]   * (w_state.u[w_state.spatial->idx(i,j-1)] - w_state.u[cur]) +
                    w_model.cond->Ty[j*nx + i]       * (w_state.u[w_state.spatial->idx(i,j+1)] - w_state.u[cur]);
                
                R[cur + n] = -net_flux;
            }
        }
    }

    void apply_bc(const IGrid& grd, const IModel& mdl, const IState& st, SparseMatrix& J, Vector& R) const override {
        const auto& w_state = static_cast<const Wave2DState&>(st);
        int n = (int)w_state.u.size();
        int nx = (int)w_state.spatial->nx;
        int ny = (int)w_state.spatial->ny;

        // Dirichlet BCs for u and fixed v at boundaries
        for (int i = 0; i < nx; ++i) {
            int b_idx = w_state.spatial->idx(i, 0);
            int t_idx = w_state.spatial->idx(i, ny - 1);
            
            R[b_idx] = w_state.u[b_idx]; J.triplets.push_back({b_idx, b_idx, 1.0});
            R[t_idx] = w_state.u[t_idx]; J.triplets.push_back({t_idx, t_idx, 1.0});
            R[b_idx+n] = w_state.v[b_idx]; J.triplets.push_back({b_idx+n, b_idx+n, 1.0});
            R[t_idx+n] = w_state.v[t_idx]; J.triplets.push_back({t_idx+n, t_idx+n, 1.0});
        }
        for (int j = 0; j < ny; ++j) {
            int l_idx = w_state.spatial->idx(0, j);
            int r_idx = w_state.spatial->idx(nx - 1, j);

            R[l_idx] = w_state.u[l_idx]; J.triplets.push_back({l_idx, l_idx, 1.0});
            R[r_idx] = w_state.u[r_idx]; J.triplets.push_back({r_idx, r_idx, 1.0});
            R[l_idx+n] = w_state.v[l_idx]; J.triplets.push_back({l_idx+n, l_idx+n, 1.0});
            R[r_idx+n] = w_state.v[r_idx]; J.triplets.push_back({r_idx+n, r_idx+n, 1.0});
        }
    }
};

} // namespace mod::wave
