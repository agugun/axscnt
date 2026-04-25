#pragma once
#include "lib/interfaces.hpp"
#include "state.hpp"
#include "lib/discretization.hpp"
 
namespace mod {
using namespace top;

/**
 * @brief 1D Pressure Physical Model (Properties).
 */
class Pressure1DModel : public IModel {
public:
    double p_left, p_right;
    std::shared_ptr<num::discretization::Conductance1D> cond;
    Vector storage_coeff;

    Pressure1DModel(std::shared_ptr<num::discretization::Conductance1D> c, const Vector& storage, double pl, double pr)
        : p_left(pl), p_right(pr), cond(c), storage_coeff(storage) {}

    double get_tolerance() const override { return 1e-6; }

    Vector get_accumulation_weights(const IGrid& grd, const IState& st) const override {
        return storage_coeff;
    }
};

/**
 * @brief 1D Pressure FVM Discretizer (Numerical Assembly).
 */
class Pressure1DDiscretizer : public IDiscretizer {
public:
    void build_jacobian(const IGrid& grd, const IModel& mdl, const IState& st, SparseMatrix& J) const override {
        const auto& p_model = static_cast<const Pressure1DModel&>(mdl);
        const auto& p_state = static_cast<const Pressure1DState&>(st);
        size_t n = p_state.pressures.size();

        if (J.rows != n) J = SparseMatrix(n, n);
        J.triplets.clear();

        for (int i = 1; i < (int)n - 1; ++i) {
            double t_prev = p_model.cond->T[i-1];
            double t_next = p_model.cond->T[i];
            
            J.triplets.push_back({i, i-1, -t_prev});
            J.triplets.push_back({i, i, t_prev + t_next});
            J.triplets.push_back({i, i+1, -t_next});
        }
    }

    void build_residual(const IGrid& grd, const IModel& mdl, const IState& st, Vector& R) const override {
        const auto& p_model = static_cast<const Pressure1DModel&>(mdl);
        const auto& p_state = static_cast<const Pressure1DState&>(st);
        size_t n = p_state.pressures.size();

        #pragma omp parallel for
        for (int i = 1; i < (int)n - 1; ++i) {
            double net_flux = p_model.cond->T[i-1] * (p_state.pressures[i-1] - p_state.pressures[i]) +
                              p_model.cond->T[i]   * (p_state.pressures[i+1] - p_state.pressures[i]);
            R[i] = -net_flux;
        }
    }

    void apply_bc(const IGrid& grd, const IModel& mdl, const IState& st, SparseMatrix& J, Vector& R) const override {
        const auto& p_model = static_cast<const Pressure1DModel&>(mdl);
        const auto& p_state = static_cast<const Pressure1DState&>(st);
        size_t n = R.size();

        // Dirichlet BCs
        R[0] = p_state.pressures[0] - p_model.p_left;
        R[n-1] = p_state.pressures[n-1] - p_model.p_right;

        J.triplets.push_back({0, 0, 1.0});
        J.triplets.push_back({(int)n-1, (int)n-1, 1.0});
    }
};

} // namespace mod
