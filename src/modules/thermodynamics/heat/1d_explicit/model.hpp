#pragma once
#include "lib/interfaces.hpp"
#include "state.hpp"

namespace mod {

class Heat1DExplicitModel : public top::IModel {
private:
    std::shared_ptr<num::discretization::Conductance1D> cond;
    std::vector<double> storage_coeff;

public:
    Heat1DExplicitModel(std::shared_ptr<num::discretization::Conductance1D> c, const std::vector<double>& storage) 
        : cond(c), storage_coeff(storage) {}

    void build_residual(const top::IState& st_new, const top::IState& st_old, double dt, std::vector<double>& R) const {
        const auto& s_new = static_cast<const Heat1DExplicitState&>(st_new);
        const auto& s_old = static_cast<const Heat1DExplicitState&>(st_old);
        size_t n = s_old.temperatures.size();

        #pragma omp parallel for
        for (int i = 0; i < (int)n; ++i) {
            double net_flux = 0.0;
            if (i > 0) net_flux += cond->T[i-1] * (s_old.temperatures[i-1] - s_old.temperatures[i]);
            if (i < (int)n - 1) net_flux += cond->T[i] * (s_old.temperatures[i+1] - s_old.temperatures[i]);
            
            R[i] = -(dt * net_flux) / storage_coeff[i];
        }
    }

    double get_tolerance() const override { return 1e-6; }
    
    top::Vector get_accumulation_weights(const top::IGrid& grd, const top::IState& st) const override {
        return storage_coeff;
    }
};

class Heat1DExplicitDiscretizer : public top::IDiscretizer {
public:
    void build_residual(const top::IGrid& grd, const top::IModel& mdl, const top::IState& st, top::Vector& R) const override {
        // For explicit, the mdl handles the RHS calculation.
    }

    void build_jacobian(const top::IGrid& grd, const top::IModel& mdl, const top::IState& st, top::SparseMatrix& J) const override {
        // Explicit Jacobian is not used in the same way.
    }

    void apply_bc(const top::IGrid& grd, const top::IModel& mdl, const top::IState& st, top::SparseMatrix& J, top::Vector& R) const override {
        // Boundary conditions are handled by the mdl.
    }
};

} // namespace mod