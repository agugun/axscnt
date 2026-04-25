#pragma once
#include "lib/interfaces.hpp"
#include "state.hpp"
#include "lib/discretization.hpp"
#include "lib/spatial.hpp"
 
namespace mod {
using namespace top;

/**
 * @brief 1D Heat Physical Model (Properties).
 */
class Heat1DModel : public IModel {
public:
    double T_left, T_right;
    std::shared_ptr<num::discretization::Conductance1D> cond;
    Vector storage_coeff;

    Heat1DModel(std::shared_ptr<num::discretization::Conductance1D> c, const Vector& storage, double TL, double TR) 
        : T_left(TL), T_right(TR), cond(c), storage_coeff(storage) {}

    double get_tolerance() const override { return 1e-6; }

    Vector get_accumulation_weights(const IGrid& grd, const IState& st) const override {
        return storage_coeff;
    }
};

/**
 * @brief 1D Heat FVM Discretizer (Numerical Assembly).
 */
class Heat1DDiscretizer : public IDiscretizer {
public:
    void build_jacobian(const IGrid& grd, const IModel& mdl, const IState& st, SparseMatrix& J) const override {
        const auto& h_model = static_cast<const Heat1DModel&>(mdl);
        const auto& h_state = static_cast<const Heat1DImplicitState&>(st);
        size_t n = h_state.temperatures.size();

        // Standard 1D Sparse Assembly
        if (J.rows != n) J = SparseMatrix(n, n);
        J.triplets.clear();

        for (int i = 1; i < (int)n - 1; ++i) {
            double t_prev = h_model.cond->T[i-1];
            double t_next = h_model.cond->T[i];
            
            // J = -dt * Grad(T) is handled by the TimeIntegrator and Discretizer collaboration.
            // Here we assemble ONLY the spatial part (Negative Laplacian).
            J.triplets.push_back({i, i-1, -t_prev});
            J.triplets.push_back({i, i, t_prev + t_next});
            J.triplets.push_back({i, i+1, -t_next});
        }
    }

    void build_residual(const IGrid& grd, const IModel& mdl, const IState& st, Vector& R) const override {
        const auto& h_model = static_cast<const Heat1DModel&>(mdl);
        const auto& h_state = static_cast<const Heat1DImplicitState&>(st);
        size_t n = h_state.temperatures.size();

        #pragma omp parallel for
        for (int i = 1; i < (int)n - 1; ++i) {
            double net_flux = h_model.cond->T[i-1] * (h_state.temperatures[i-1] - h_state.temperatures[i]) +
                              h_model.cond->T[i]   * (h_state.temperatures[i+1] - h_state.temperatures[i]);
            R[i] = -net_flux; // Residual is T(n+1) - T(n) - dt * Flux. Flux part is -net_flux.
        }
    }

    void apply_bc(const IGrid& grd, const IModel& mdl, const IState& st, SparseMatrix& J, Vector& R) const override {
        const auto& h_model = static_cast<const Heat1DModel&>(mdl);
        const auto& h_state = static_cast<const Heat1DImplicitState&>(st);
        size_t n = R.size();

        // Dirichlet Boundary Conditions
        R[0] = h_state.temperatures[0] - h_model.T_left;
        R[n-1] = h_state.temperatures[n-1] - h_model.T_right;

        // Jacobian rows for BCs
        J.triplets.push_back({0, 0, 1.0});
        J.triplets.push_back({(int)n-1, (int)n-1, 1.0});
    }
};

} // namespace mod
