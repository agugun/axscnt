#pragma once
#include "lib/interfaces.hpp"
#include "state.hpp"
#include "lib/discretization.hpp"
 
namespace mod {
using namespace top;

/**
 * @brief 2D Heat Physical Model (Properties).
 */
class Heat2DModel : public IModel {
public:
    double T_top, T_bottom, T_left, T_right;
    std::shared_ptr<num::discretization::Conductance2D> cond;
    Vector storage_coeff;

    Heat2DModel(std::shared_ptr<num::discretization::Conductance2D> c, const Vector& storage,
                double top, double bottom, double left, double right)
        : T_top(top), T_bottom(bottom), T_left(left), T_right(right), cond(c), storage_coeff(storage) {}

    double get_tolerance() const override { return 1e-6; }

    Vector get_accumulation_weights(const IGrid& grd, const IState& st) const override {
        return storage_coeff;
    }
};

/**
 * @brief 2D Heat FVM Discretizer (Numerical Assembly).
 * Uses a standard 5-point stencil for the Laplacian.
 */
class Heat2DDiscretizer : public IDiscretizer {
public:
    void build_jacobian(const IGrid& grd, const IModel& mdl, const IState& st, SparseMatrix& J) const override {
        const auto& h_model = static_cast<const Heat2DModel&>(mdl);
        const auto& h_state = static_cast<const Heat2DImplicitState&>(st);
        int nx = (int)h_state.spatial->nx;
        int ny = (int)h_state.spatial->ny;

        if (J.rows != nx * ny) J = SparseMatrix(nx * ny, nx * ny);
        J.triplets.clear();

        // 5-point stencil assembly for interior cells
        for (int j = 1; j < ny - 1; ++j) {
            for (int i = 1; i < nx - 1; ++i) {
                int cur = h_state.spatial->idx(i, j);
                double tx_prev = h_model.cond->Tx[j*(nx-1) + i-1];
                double tx_next = h_model.cond->Tx[j*(nx-1) + i];
                double ty_prev = h_model.cond->Ty[(j-1)*nx + i];
                double ty_next = h_model.cond->Ty[j*nx + i];

                J.triplets.push_back({cur, (int)h_state.spatial->idx(i-1, j), -tx_prev});
                J.triplets.push_back({cur, (int)h_state.spatial->idx(i+1, j), -tx_next});
                J.triplets.push_back({cur, (int)h_state.spatial->idx(i, j-1), -ty_prev});
                J.triplets.push_back({cur, (int)h_state.spatial->idx(i, j+1), -ty_next});
                J.triplets.push_back({cur, cur, tx_prev + tx_next + ty_prev + ty_next});
            }
        }
    }

    void build_residual(const IGrid& grd, const IModel& mdl, const IState& st, Vector& R) const override {
        const auto& h_model = static_cast<const Heat2DModel&>(mdl);
        const auto& h_state = static_cast<const Heat2DImplicitState&>(st);
        int nx = (int)h_state.spatial->nx;
        int ny = (int)h_state.spatial->ny;

        #pragma omp parallel for collapse(2)
        for (int j = 1; j < ny - 1; ++j) {
            for (int i = 1; i < nx - 1; ++i) {
                int cur = h_state.spatial->idx(i, j);
                double net_flux = 
                    h_model.cond->Tx[j*(nx-1) + i-1] * (h_state.temperatures[h_state.spatial->idx(i-1,j)] - h_state.temperatures[cur]) +
                    h_model.cond->Tx[j*(nx-1) + i]   * (h_state.temperatures[h_state.spatial->idx(i+1,j)] - h_state.temperatures[cur]) +
                    h_model.cond->Ty[(j-1)*nx + i]   * (h_state.temperatures[h_state.spatial->idx(i,j-1)] - h_state.temperatures[cur]) +
                    h_model.cond->Ty[j*nx + i]       * (h_state.temperatures[h_state.spatial->idx(i,j+1)] - h_state.temperatures[cur]);
                
                R[cur] = -net_flux;
            }
        }
    }

    void apply_bc(const IGrid& grd, const IModel& mdl, const IState& st, SparseMatrix& J, Vector& R) const override {
        const auto& h_model = static_cast<const Heat2DModel&>(mdl);
        const auto& h_state = static_cast<const Heat2DImplicitState&>(st);
        int nx = (int)h_state.spatial->nx;
        int ny = (int)h_state.spatial->ny;

        // Dirichlet BCs (Residual and Jacobian)
        for (int i = 0; i < nx; ++i) {
            // Bottom boundary
            int b_idx = (int)h_state.spatial->idx(i, 0);
            R[b_idx] = h_state.temperatures[b_idx] - h_model.T_bottom;
            J.triplets.push_back({b_idx, b_idx, 1.0});
            // Top boundary
            int t_idx = (int)h_state.spatial->idx(i, ny - 1);
            R[t_idx] = h_state.temperatures[t_idx] - h_model.T_top;
            J.triplets.push_back({t_idx, t_idx, 1.0});
        }
        for (int j = 0; j < ny; ++j) {
            // Left boundary
            int l_idx = (int)h_state.spatial->idx(0, j);
            R[l_idx] = h_state.temperatures[l_idx] - h_model.T_left;
            J.triplets.push_back({l_idx, l_idx, 1.0});
            // Right boundary
            int r_idx = (int)h_state.spatial->idx(nx - 1, j);
            R[r_idx] = h_state.temperatures[r_idx] - h_model.T_right;
            J.triplets.push_back({r_idx, r_idx, 1.0});
        }
    }
};

} // namespace mod
