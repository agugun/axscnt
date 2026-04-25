#pragma once
#include "lib/interfaces.hpp"
#include "state.hpp"
#include "lib/discretization.hpp"
#include "modules/reservoir/well.hpp"
 
namespace mod {
using namespace top;
namespace reservoir {

/**
 * @brief 3D Reservoir Physical Model (Properties).
 */
class Reservoir3DModel : public IModel {
public:
    std::shared_ptr<num::discretization::Conductance3D> cond;
    Vector storage_coeff;
    std::vector<std::shared_ptr<ISourceSink>> wells;

    Reservoir3DModel(std::shared_ptr<num::discretization::Conductance3D> c, const Vector& storage, 
                     const std::vector<std::shared_ptr<ISourceSink>>& wells_val)
        : cond(c), storage_coeff(storage), wells(wells_val) {}

    double get_tolerance() const override { return 1e-4; }

    Vector get_accumulation_weights(const IGrid& grd, const IState& st) const override {
        return storage_coeff;
    }

    const std::vector<std::shared_ptr<ISourceSink>>& get_sources() const {
        return wells;
    }
};

/**
 * @brief 3D Reservoir FVM Discretizer.
 */
class Reservoir3DDiscretizer : public IDiscretizer {
public:
    void build_jacobian(const IGrid& grd, const IModel& mdl, const IState& st, SparseMatrix& J) const override {
        const auto& r_model = static_cast<const Reservoir3DModel&>(mdl);
        const auto& r_state = static_cast<const Reservoir3DState&>(st);
        int nx = (int)r_state.spatial->nx;
        int ny = (int)r_state.spatial->ny;
        int nz = (int)r_state.spatial->nz;
        int n = nx * ny * nz;

        if (J.rows != n) J = SparseMatrix(n, n);
        J.triplets.clear();

        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    int cur = r_state.spatial->idx(i, j, k);
                    double diag = 0.0;
                    
                    if (i > 0) {
                        double t = r_model.cond->Tx[(k * ny + j) * (nx - 1) + i - 1];
                        diag += t;
                        J.triplets.push_back({cur, (int)r_state.spatial->idx(i - 1, j, k), -t});
                    }
                    if (i < nx - 1) {
                        double t = r_model.cond->Tx[(k * ny + j) * (nx - 1) + i];
                        diag += t;
                        J.triplets.push_back({cur, (int)r_state.spatial->idx(i + 1, j, k), -t});
                    }
                    if (j > 0) {
                        double t = r_model.cond->Ty[(k * (ny - 1) + j - 1) * nx + i];
                        diag += t;
                        J.triplets.push_back({cur, (int)r_state.spatial->idx(i, j - 1, k), -t});
                    }
                    if (j < ny - 1) {
                        double t = r_model.cond->Ty[(k * (ny - 1) + j) * nx + i];
                        diag += t;
                        J.triplets.push_back({cur, (int)r_state.spatial->idx(i, j + 1, k), -t});
                    }
                    if (k > 0) {
                        double t = r_model.cond->Tz[((k - 1) * ny + j) * nx + i];
                        diag += t;
                        J.triplets.push_back({cur, (int)r_state.spatial->idx(i, j, k - 1), -t});
                    }
                    if (k < nz - 1) {
                        double t = r_model.cond->Tz[(k * ny + j) * nx + i];
                        diag += t;
                        J.triplets.push_back({cur, (int)r_state.spatial->idx(i, j, k + 1), -t});
                    }
                    
                    J.triplets.push_back({cur, cur, diag});
                }
            }
        }
    }

    void build_residual(const IGrid& grd, const IModel& mdl, const IState& st, Vector& R) const override {
        const auto& r_model = static_cast<const Reservoir3DModel&>(mdl);
        const auto& r_state = static_cast<const Reservoir3DState&>(st);
        int nx = (int)r_state.spatial->nx;
        int ny = (int)r_state.spatial->ny;
        int nz = (int)r_state.spatial->nz;

        #pragma omp parallel for collapse(3)
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    int cur = r_state.spatial->idx(i, j, k);
                    double net_flux = 0.0;
                    
                    if (i > 0) net_flux += r_model.cond->Tx[(k * ny + j) * (nx - 1) + i - 1] * (r_state.pressures[r_state.spatial->idx(i - 1, j, k)] - r_state.pressures[cur]);
                    if (i < nx - 1) net_flux += r_model.cond->Tx[(k * ny + j) * (nx - 1) + i]   * (r_state.pressures[r_state.spatial->idx(i + 1, j, k)] - r_state.pressures[cur]);
                    if (j > 0) net_flux += r_model.cond->Ty[(k * (ny - 1) + j - 1) * nx + i] * (r_state.pressures[r_state.spatial->idx(i, j - 1, k)] - r_state.pressures[cur]);
                    if (j < ny - 1) net_flux += r_model.cond->Ty[(k * (ny - 1) + j) * nx + i]     * (r_state.pressures[r_state.spatial->idx(i, j + 1, k)] - r_state.pressures[cur]);
                    if (k > 0) net_flux += r_model.cond->Tz[((k - 1) * ny + j) * nx + i]     * (r_state.pressures[r_state.spatial->idx(i, j, k - 1)] - r_state.pressures[cur]);
                    if (k < nz - 1) net_flux += r_model.cond->Tz[(k * ny + j) * nx + i]           * (r_state.pressures[r_state.spatial->idx(i, j, k + 1)] - r_state.pressures[cur]);
                    
                    R[cur] = -net_flux;
                }
            }
        }
    }

    void apply_bc(const IGrid& grd, const IModel& mdl, const IState& st, SparseMatrix& J, Vector& R) const override {
        // No-flow is default.
    }
};

} // namespace reservoir
} // namespace mod
