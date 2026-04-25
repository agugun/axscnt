#pragma once
#include "lib/fem.hpp"
#include "lib/interfaces.hpp"

namespace mod {
using namespace top;

/**
 * @brief State representing the incompressible velocity (u, v) and pressure (p) fields.
 * Interleaved or concatenated for the solver.
 */
class FluidState : public IState {
public:
    Vector u; // Velocity X
    Vector v; // Velocity Y
    Vector p; // Pressure
    std::shared_ptr<Mesh> mesh;

    FluidState(std::shared_ptr<Mesh> m) 
        : u(m->num_nodes(), 0.0), v(m->num_nodes(), 0.0), p(m->num_nodes(), 0.0), mesh(m) {}

    void update(const std::vector<double>& delta) override {
        size_t n = mesh->num_nodes();
        for (size_t i = 0; i < n; ++i) {
            u[i] += delta[i];
            v[i] += delta[i + n];
            p[i] += delta[i + 2 * n];
        }
    }

    std::vector<double> to_vector() const override {
        size_t n = mesh->num_nodes();
        Vector res(3 * n);
        for (size_t i = 0; i < n; ++i) {
            res[i] = u[i];
            res[i + n] = v[i];
            res[i + 2 * n] = p[i];
        }
        return res;
    }

    std::unique_ptr<IState> clone() const override {
        auto copy = std::make_unique<FluidState>(mesh);
        copy->u = this->u;
        copy->v = this->v;
        copy->p = this->p;
        return copy;
    }
};

} // namespace mod
