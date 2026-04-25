#pragma once
#include "lib/interfaces.hpp"
#include "state.hpp"
#include <vector>

namespace mod {
using namespace top;

/**
 * @brief 1D Burgers' Equation Model (Physical Properties).
 */
class BurgersModel : public IModel {
public:
    double nu; // Viscosity
    double dx;

    BurgersModel(double viscosity, double grid_spacing) : nu(viscosity), dx(grid_spacing) {}

    double get_tolerance() const override { return 1e-6; }

    Vector get_accumulation_weights(const IGrid& grd, const IState& st) const override {
        size_t n = st.to_vector().size();
        return Vector(n, 1.0); // u's mass matrix is identity in FDM
    }
};

/**
 * @brief 1D Burgers' FDM Discretizer (Numerical Assembly).
 * Implements du/dt + u*du/dx = nu*d^2u/dx^2.
 */
class BurgersDiscretizer : public IDiscretizer {
public:
    void build_jacobian(const IGrid& grd, const IModel& mdl, const IState& st, SparseMatrix& J) const override {
        const auto& m = static_cast<const BurgersModel&>(mdl);
        const auto& s = static_cast<const BurgersState&>(st);
        int n = (int)s.u.size();
        double dx = m.dx;
        double nu = m.nu;

        if (J.rows != n) J = SparseMatrix(n, n);
        J.triplets.clear();

        // 1. Internal Nodes (FDM Central/Upwind)
        for (int i = 1; i < n - 1; ++i) {
            double u_i = s.u[i];
            
            // Convection: u * du/dx (Upwind based on sign of u)
            if (u_i >= 0.0) {
                // u_i * (u_i - u_{i-1}) / dx
                // d/du_i = (2*u_i - u_{i-1}) / dx
                // d/du_{i-1} = -u_i / dx
                J.triplets.push_back({i, i, (2.0 * u_i - s.u[i-1]) / dx});
                J.triplets.push_back({i, i-1, -u_i / dx});
            } else {
                // u_i * (u_{i+1} - u_i) / dx
                // d/du_i = (u_{i+1} - 2*u_i) / dx
                // d/du_{i+1} = u_i / dx
                J.triplets.push_back({i, i, (s.u[i+1] - 2.0 * u_i) / dx});
                J.triplets.push_back({i, i+1, u_i / dx});
            }

            // Diffusion: -nu * (u_{i+1} - 2*u_i + u_{i-1}) / dx^2
            double d_coeff = nu / (dx * dx);
            J.triplets.push_back({i, i-1, -d_coeff});
            J.triplets.push_back({i, i, 2.0 * d_coeff});
            J.triplets.push_back({i, i+1, -d_coeff});
        }
    }

    void build_residual(const IGrid& grd, const IModel& mdl, const IState& st, Vector& R) const override {
        const auto& m = static_cast<const BurgersModel&>(mdl);
        const auto& s = static_cast<const BurgersState&>(st);
        int n = (int)s.u.size();
        double dx = m.dx;
        double nu = m.nu;

        #pragma omp parallel for
        for (int i = 1; i < n - 1; ++i) {
            double u_i = s.u[i];
            double c_flux = (u_i >= 0.0) ? u_i * (u_i - s.u[i-1]) / dx : u_i * (s.u[i+1] - u_i) / dx;
            double d_flux = nu * (s.u[i+1] - 2.0 * u_i + s.u[i-1]) / (dx * dx);
            
            R[i] = c_flux - d_flux;
        }
    }

    void apply_bc(const IGrid& grd, const IModel& mdl, const IState& st, SparseMatrix& J, Vector& R) const override {
        const auto& s = static_cast<const BurgersState&>(st);
        int n = (int)s.u.size();

        // Dirichlet BCs (u=0)
        R[0] = s.u[0] - 0.0;
        R[n-1] = s.u[n-1] - 0.0;
        J.triplets.push_back({0, 0, 1.0});
        J.triplets.push_back({n-1, n-1, 1.0});
    }
};

} // namespace mod
