#pragma once
#include "lib/interfaces.hpp"
#include "state.hpp"

namespace mod {

// Strong semantic typing
using Mass_kg = double;
using Damping_Ns_m = double;
using Stiffness_N_m = double;

/**
 * @brief Physics Model for the Harmonic Oscillator.
 * Only contains continuous physical properties and weight mappings.
 */
class OscillatorModel : public top::IModel {
public:
    Mass_kg m;
    Damping_Ns_m c;
    Stiffness_N_m k;

    OscillatorModel(Mass_kg mass, Damping_Ns_m damp, Stiffness_N_m stiff) 
        : m(mass), c(damp), k(stiff) {}

    double get_tolerance() const override { return 1e-6; }

    top::Vector get_accumulation_weights(const top::IGrid& grd, const top::IState& st) const override {
        // [Eq. 1 - Notebook] dx/dt = v  -> Weight = 1.0
        // [Eq. 2 - Notebook] m*dv/dt = ... -> Weight = m
        return {1.0, m};
    }
};

/**
 * @brief Discretizer for the Harmonic Oscillator.
 * Translates the analytical mdl rules into the Residual and Jacobian matrices.
 */
class OscillatorDiscretizer : public top::IDiscretizer {
public:
    void build_jacobian(const top::IGrid& grd, const top::IModel& base_model, const top::IState& base_state, top::SparseMatrix& J) const override {
        const auto& mdl = dynamic_cast<const OscillatorModel&>(base_model);
        
        // J = dR/du
        // Eq. 1: dx/dt - v = 0      --> dR0/dx = 0, dR0/dv = -1
        // Eq. 2: m*dv/dt + cx + kv = 0 --> dR1/dx = k, dR1/dv = c

        J.triplets.push_back({0, 1, -1.0});         // dR0 / dv
        J.triplets.push_back({1, 0, mdl.k});      // dR1 / dx
        J.triplets.push_back({1, 1, mdl.c});      // dR1 / dv
    }

    void build_residual(const top::IGrid& grd, const top::IModel& base_model, const top::IState& base_state, top::Vector& R) const override {
        const auto& mdl = dynamic_cast<const OscillatorModel&>(base_model);
        const auto& st = dynamic_cast<const OscillatorState&>(base_state);

        // Vectorized Math Semantics: The math reads exactly like the equations.
        // Eq. 1 [Notebook] Physics Residual for Position
        double r_pos = -st.v; 
        
        // Eq. 2 [Notebook] Physics Residual for Velocity 
        double r_vel = mdl.k * st.x + mdl.c * st.v;

        R[0] += r_pos;
        R[1] += r_vel;
    }

    void apply_bc(const top::IGrid& grd, const top::IModel& base_model, const top::IState& base_state, top::SparseMatrix& J, top::Vector& R) const override {
        // No boundaries for a 0D ODE problem.
    }
};

/**
 * @brief Dummy grd for the 2-variable ODE system.
 */
class OscillatorGrid : public top::IGrid {
public:
    size_t get_total_cells() const override { return 2; }
};

} // namespace mod
