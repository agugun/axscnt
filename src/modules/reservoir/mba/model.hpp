#pragma once
#include "lib/interfaces.hpp"
#include "state.hpp"

namespace mod::reservoir {
using namespace top;

/**
 * @brief Material Balance Model (Physical Properties).
 */
class MBAModel : public IModel {
public:
    double volume; 
    double ct;     
    double q;      

    MBAModel(double V, double ct_val, double q_val) : volume(V), ct(ct_val), q(q_val) {}

    double get_tolerance() const override { return 1e-4; }

    Vector get_accumulation_weights(const IGrid& grd, const IState& st) const override {
        return { volume * ct }; // Accumulation is volume * ct * dP
    }
};

/**
 * @brief MBA (0D) Discretizer (Numerical Assembly).
 * Implements d(V*ct*P)/dt = -q.
 */
class MBADiscretizer : public IDiscretizer {
public:
    void build_jacobian(const IGrid& grd, const IModel& mdl, const IState& st, SparseMatrix& J) const override {
        if (J.rows != 1) J = SparseMatrix(1, 1);
        J.triplets.clear();
        // For dP/dt = -q/(V*ct), J is 0 (rate is constant)
        // But for the linear system in NewtonRaphson: J is just the 0-matrix which is fine.
    }

    void build_residual(const IGrid& grd, const IModel& mdl, const IState& st, Vector& R) const override {
        const auto& m = static_cast<const MBAModel&>(mdl);
        R[0] = m.q; // Residual is Flux - Rate. Flux=0.
    }

    void apply_bc(const IGrid& grd, const IModel& mdl, const IState& st, SparseMatrix& J, Vector& R) const override {
        // No boundaries for 0D tank.
    }
};

} // namespace mod::reservoir
