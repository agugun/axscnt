#pragma once
#include "lib/interfaces.hpp"
#include <stdexcept>
#include <vector>

namespace num {
using namespace top;

/**
 * @brief Standard Implicit Euler (Backward Euler) Integrator.
 */
class ImplicitEulerIntegrator : public ITimeIntegrator {
public:
    double compute_dt(const IState& st, double t) const override {
        return 0.1; // Default
    }

    void apply_temporal(const IGrid& grd, const IModel& mdl, SparseMatrix& J, Vector& R, const IState& st_new, const IState& st_old, double dt) const override {
        Vector weights = mdl.get_accumulation_weights(grd, st_new);
        Vector u_new = st_new.to_vector();
        Vector u_old = st_old.to_vector();
        size_t n = R.size();
        
        #pragma omp parallel for
        for (int i = 0; i < (int)n; ++i) {
            double acc = weights[i] * (u_new[i] - u_old[i]) / dt;
            R[i] += acc;
        }

        // Add to Jacobian diagonal
        for (int i = 0; i < (int)n; ++i) {
            J.triplets.push_back({i, i, weights[i] / dt});
        }
    }
};

/**
 * @brief Forward Euler (Explicit) Integrator.
 */
class ForwardEulerIntegrator : public ITimeIntegrator {
public:
    double compute_dt(const IState& st, double t) const override { return 0.01; }
    
    void apply_temporal(const IGrid& grd, const IModel& mdl, SparseMatrix& J, Vector& R, const IState& st_new, const IState& st_old, double dt) const override {
        // Forward Euler in Newton-Raphson is not standard.
        // We throw to indicate this shouldn't be used with SimulationEngine's current linearizer.
        throw std::runtime_error("ForwardEulerIntegrator not supported in the current linearized framework.");
    }
};

/**
 * @brief Runge-Kutta 4 (Explicit) Integrator.
 */
class RungeKutta4Integrator : public ITimeIntegrator {
public:
    double compute_dt(const IState& st, double t) const override { return 0.01; }
    
    void apply_temporal(const IGrid& grd, const IModel& mdl, SparseMatrix& J, Vector& R, const IState& st_new, const IState& st_old, double dt) const override {
        throw std::runtime_error("RungeKutta4Integrator not supported in the current linearized framework.");
    }
};

/**
 * @brief Fully Implicit (Non-linear) Integrator.
 * Transitioned to use the same logic as ImplicitEuler for now, 
 * as the non-linearity is handled by the NewtonRaphson linearizer.
 */
class FullyImplicitIntegrator : public ImplicitEulerIntegrator {
};

} // namespace num
