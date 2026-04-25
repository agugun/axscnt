#pragma once
#include "lib/interfaces.hpp"
#include "state.hpp"
#include "lib/fem.hpp"
#include <vector>

namespace mod {
using namespace top;

/**
 * @brief 2D Incompressible Fluid Model (Stokes / Low-Re Navier-Stokes).
 */
class FluidModel : public IModel {
public:
    double mu; // Viscosity
    double rho; // Density
    std::shared_ptr<Mesh> mesh;
    std::vector<int> velocity_bc_nodes;
    std::vector<double> velocity_bc_values_u;
    std::vector<double> velocity_bc_values_v;

    FluidModel(std::shared_ptr<Mesh> m, double viscosity, double density) 
        : mu(viscosity), rho(density), mesh(m) {}

    void set_velocity_bc(int node_idx, double u_val, double v_val) {
        velocity_bc_nodes.push_back(node_idx);
        velocity_bc_values_u.push_back(u_val);
        velocity_bc_values_v.push_back(v_val);
    }

    double get_tolerance() const override { return 1e-6; }

    Vector get_accumulation_weights(const IGrid& grd, const IState& st) const override {
        // Steady st or semi-implicit: return zero weights for pressure and 
        // identity-like weights for momentum if transient is needed.
        // For Stokes, return zero for now.
        return Vector(3 * mesh->num_nodes(), 0.0);
    }
};

/**
 * @brief 2D Fluid FEM Discretizer (Stabilized P1-P1).
 */
class FluidDiscretizer : public IDiscretizer {
public:
    void build_jacobian(const IGrid& grd, const IModel& mdl, const IState& st, SparseMatrix& J) const override {
        const auto& m = static_cast<const FluidModel&>(mdl);
        size_t n = m.mesh->num_nodes();
        
        if (J.rows != 3 * n) J = SparseMatrix(3 * n, 3 * n);
        J.triplets.clear();

        std::vector<bool> is_bc_u(n, false);
        std::vector<bool> is_bc_v(n, false);
        for (int node : m.velocity_bc_nodes) {
            is_bc_u[node] = true;
            is_bc_v[node] = true;
        }

        for (const auto& el : m.mesh->elements) {
            auto data = LinearTriangle::compute_data(m.mesh->nodes[el.nodes[0]], 
                                                   m.mesh->nodes[el.nodes[1]], 
                                                   m.mesh->nodes[el.nodes[2]]);
            
            for (int i = 0; i < 3; ++i) {
                int row = el.nodes[i];
                for (int j = 0; j < 3; ++j) {
                    int col = el.nodes[j];
                    
                    double diff = m.mu * (data.dNdx[i] * data.dNdx[j] + data.dNdy[i] * data.dNdy[j]) * data.area;
                    
                    // Momentum rows
                    if (!is_bc_u[row]) {
                        J.triplets.push_back({ row, col, diff });
                        J.triplets.push_back({ row, col + 2 * (int)n, data.dNdx[i] * (data.area / 3.0) });
                    }
                    if (!is_bc_v[row]) {
                        J.triplets.push_back({ row + (int)n, col + (int)n, diff });
                        J.triplets.push_back({ row + (int)n, col + 2 * (int)n, data.dNdy[i] * (data.area / 3.0) });
                    }
                    
                    // Continuity (Pressure) row
                    J.triplets.push_back({ row + 2 * (int)n, col, data.dNdx[j] * (data.area / 3.0) });
                    J.triplets.push_back({ row + 2 * (int)n, col + (int)n, data.dNdy[j] * (data.area / 3.0) });

                    // PSPG Stabilization
                    double tau = (data.area) / (12.0 * m.mu);
                    J.triplets.push_back({ row + 2 * (int)n, col + 2 * (int)n, tau * (data.dNdx[i] * data.dNdx[j] + data.dNdy[i] * data.dNdy[j]) * data.area });
                }
            }
        }
    }

    void build_residual(const IGrid& grd, const IModel& mdl, const IState& st, Vector& R) const override {
        const auto& m = static_cast<const FluidModel&>(mdl);
        const auto& s = static_cast<const FluidState&>(st);
        size_t n = m.mesh->num_nodes();

        std::vector<bool> is_bc_u(n, false);
        std::vector<bool> is_bc_v(n, false);
        for (int node : m.velocity_bc_nodes) {
            is_bc_u[node] = true;
            is_bc_v[node] = true;
        }

        // Element Assembly
        for (const auto& el : m.mesh->elements) {
            auto data = LinearTriangle::compute_data(m.mesh->nodes[el.nodes[0]], 
                                                   m.mesh->nodes[el.nodes[1]], 
                                                   m.mesh->nodes[el.nodes[2]]);
            
            for (int i = 0; i < 3; ++i) {
                int row = el.nodes[i];
                for (int j = 0; j < 3; ++j) {
                    int col = el.nodes[j];
                    double diff = m.mu * (data.dNdx[i] * data.dNdx[j] + data.dNdy[i] * data.dNdy[j]) * data.area;
                    
                    if (!is_bc_u[row]) {
                        R[row] += diff * s.u[col] + data.dNdx[i] * (data.area / 3.0) * s.p[col];
                    }
                    if (!is_bc_v[row]) {
                        R[row + n] += diff * s.v[col] + data.dNdy[i] * (data.area / 3.0) * s.p[col];
                    }
                    
                    R[row + 2 * n] += (data.dNdx[j] * s.u[col] + data.dNdy[j] * s.v[col]) * (data.area / 3.0);
                    // PSPG Stability part in Residual
                    double tau = (data.area) / (12.0 * m.mu);
                    R[row + 2 * n] += tau * (data.dNdx[i] * data.dNdx[j] + data.dNdy[i] * data.dNdy[j]) * data.area * s.p[col];
                }
            }
        }
    }

    void apply_bc(const IGrid& grd, const IModel& mdl, const IState& st, SparseMatrix& J, Vector& R) const override {
        const auto& m = static_cast<const FluidModel&>(mdl);
        const auto& s = static_cast<const FluidState&>(st);
        size_t n = m.mesh->num_nodes();

        for (size_t i = 0; i < m.velocity_bc_nodes.size(); ++i) {
            int node = m.velocity_bc_nodes[i];
            R[node] = s.u[node] - m.velocity_bc_values_u[i];
            R[node + n] = s.v[node] - m.velocity_bc_values_v[i];
            J.triplets.push_back({ node, node, 1.0 });
            J.triplets.push_back({ node + (int)n, node + (int)n, 1.0 });
        }
        // Pressure Pinning
        R[2*n] = s.p[0] - 0.0;
        J.triplets.push_back({ 2 * (int)n, 2 * (int)n, 1.0 });
    }
};

} // namespace mod
