#pragma once
#include "lib/interfaces.hpp"
#include "state.hpp"
#include "lib/discretization.hpp"
 
namespace mod {
using namespace top;
namespace heat {

/**
 * @brief 3D Heat Physical Model (Properties).
 */
class Heat3DModel : public IModel {
public:
    double T_front, T_back, T_top, T_bottom, T_left, T_right;
    std::shared_ptr<num::discretization::Conductance3D> cond;
    Vector storage_coeff;

    Heat3DModel(std::shared_ptr<num::discretization::Conductance3D> c, const Vector& storage,
                double front, double back, double top, double bottom, double left, double right)
        : T_front(front), T_back(back), T_top(top), T_bottom(bottom), T_left(left), T_right(right), 
          cond(c), storage_coeff(storage) {}

    double get_tolerance() const override { return 1e-6; }

    Vector get_accumulation_weights(const IGrid& grd, const IState& st) const override {
        return storage_coeff;
    }
};

/**
 * @brief 3D Heat FVM Discretizer (Numerical Assembly).
 * Uses a standard 7-point stencil for the 3D Laplacian.
 */
class Heat3DDiscretizer : public IDiscretizer {
public:
    void build_jacobian(const IGrid& grd, const IModel& mdl, const IState& st, SparseMatrix& J) const override {
        const auto& h_model = static_cast<const Heat3DModel&>(mdl);
        const auto& h_state = static_cast<const Heat3DImplicitState&>(st);
        int nx = (int)h_state.spatial->nx;
        int ny = (int)h_state.spatial->ny;
        int nz = (int)h_state.spatial->nz;

        if (J.rows != nx * ny * nz) J = SparseMatrix(nx * ny * nz, nx * ny * nz);
        J.triplets.clear();

        // 7-point stencil assembly for interior cells
        #pragma omp parallel for collapse(3)
        for (int k = 1; k < nz - 1; ++k) {
            for (int j = 1; j < ny - 1; ++j) {
                for (int i = 1; i < nx - 1; ++i) {
                    int cur = h_state.spatial->idx(i, j, k);
                    
                    double tx_prev = h_model.cond->Tx[(k * ny + j) * (nx - 1) + i - 1];
                    double tx_next = h_model.cond->Tx[(k * ny + j) * (nx - 1) + i];
                    double ty_prev = h_model.cond->Ty[(k * (ny - 1) + j - 1) * nx + i];
                    double ty_next = h_model.cond->Ty[(k * (ny - 1) + j) * nx + i];
                    double tz_prev = h_model.cond->Tz[((k - 1) * ny + j) * nx + i];
                    double tz_next = h_model.cond->Tz[(k * ny + j) * nx + i];

                    J.triplets.push_back({cur, (int)h_state.spatial->idx(i - 1, j, k), -tx_prev});
                    J.triplets.push_back({cur, (int)h_state.spatial->idx(i + 1, j, k), -tx_next});
                    J.triplets.push_back({cur, (int)h_state.spatial->idx(i, j - 1, k), -ty_prev});
                    J.triplets.push_back({cur, (int)h_state.spatial->idx(i, j + 1, k), -ty_next});
                    J.triplets.push_back({cur, (int)h_state.spatial->idx(i, j, k - 1), -tz_prev});
                    J.triplets.push_back({cur, (int)h_state.spatial->idx(i, j, k + 1), -tz_next});
                    J.triplets.push_back({cur, cur, tx_prev + tx_next + ty_prev + ty_next + tz_prev + tz_next});
                }
            }
        }
    }

    void build_residual(const IGrid& grd, const IModel& mdl, const IState& st, Vector& R) const override {
        const auto& h_model = static_cast<const Heat3DModel&>(mdl);
        const auto& h_state = static_cast<const Heat3DImplicitState&>(st);
        int nx = (int)h_state.spatial->nx;
        int ny = (int)h_state.spatial->ny;
        int nz = (int)h_state.spatial->nz;

        #pragma omp parallel for collapse(3)
        for (int k = 1; k < nz - 1; ++k) {
            for (int j = 1; j < ny - 1; ++j) {
                for (int i = 1; i < nx - 1; ++i) {
                    int cur = h_state.spatial->idx(i, j, k);
                    double net_flux = 
                        h_model.cond->Tx[(k * ny + j) * (nx - 1) + i - 1] * (h_state.temperatures[h_state.spatial->idx(i - 1, j, k)] - h_state.temperatures[cur]) +
                        h_model.cond->Tx[(k * ny + j) * (nx - 1) + i]     * (h_state.temperatures[h_state.spatial->idx(i + 1, j, k)] - h_state.temperatures[cur]) +
                        h_model.cond->Ty[(k * (ny - 1) + j - 1) * nx + i] * (h_state.temperatures[h_state.spatial->idx(i, j - 1, k)] - h_state.temperatures[cur]) +
                        h_model.cond->Ty[(k * (ny - 1) + j) * nx + i]     * (h_state.temperatures[h_state.spatial->idx(i, j + 1, k)] - h_state.temperatures[cur]) +
                        h_model.cond->Tz[((k - 1) * ny + j) * nx + i]     * (h_state.temperatures[h_state.spatial->idx(i, j, k - 1)] - h_state.temperatures[cur]) +
                        h_model.cond->Tz[(k * ny + j) * nx + i]           * (h_state.temperatures[h_state.spatial->idx(i, j, k + 1)] - h_state.temperatures[cur]);
                    
                    R[cur] = -net_flux;
                }
            }
        }
    }

    void apply_bc(const IGrid& grd, const IModel& mdl, const IState& st, SparseMatrix& J, Vector& R) const override {
        const auto& h_model = static_cast<const Heat3DModel&>(mdl);
        const auto& h_state = static_cast<const Heat3DImplicitState&>(st);
        int nx = (int)h_state.spatial->nx;
        int ny = (int)h_state.spatial->ny;
        int nz = (int)h_state.spatial->nz;

        // Dirichlet BCs for all 6 faces
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                // Left
                int l_idx = (int)h_state.spatial->idx(0, j, k);
                R[l_idx] = h_state.temperatures[l_idx] - h_model.T_left;
                J.triplets.push_back({l_idx, l_idx, 1.0});
                // Right
                int r_idx = (int)h_state.spatial->idx(nx - 1, j, k);
                R[r_idx] = h_state.temperatures[r_idx] - h_model.T_right;
                J.triplets.push_back({r_idx, r_idx, 1.0});
            }
        }
        for (int k = 0; k < nz; ++k) {
            for (int i = 0; i < nx; ++i) {
                // Bottom
                int b_idx = (int)h_state.spatial->idx(i, 0, k);
                R[b_idx] = h_state.temperatures[b_idx] - h_model.T_bottom;
                J.triplets.push_back({b_idx, b_idx, 1.0});
                // Top
                int t_idx = (int)h_state.spatial->idx(i, ny - 1, k);
                R[t_idx] = h_state.temperatures[t_idx] - h_model.T_top;
                J.triplets.push_back({t_idx, t_idx, 1.0});
            }
        }
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                // Front
                int f_idx = (int)h_state.spatial->idx(i, j, 0);
                R[f_idx] = h_state.temperatures[f_idx] - h_model.T_front;
                J.triplets.push_back({f_idx, f_idx, 1.0});
                // Back
                int bk_idx = (int)h_state.spatial->idx(i, j, nz - 1);
                R[bk_idx] = h_state.temperatures[bk_idx] - h_model.T_back;
                J.triplets.push_back({bk_idx, bk_idx, 1.0});
            }
        }
    }
};

} // namespace heat
} // namespace mod
