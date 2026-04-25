#pragma once
#include "lib/spatial.hpp"
#include "lib/interfaces.hpp"

namespace mod {
using namespace top;

class Wave1DState : public IState {
public:
    Vector u; // Displacement
    Vector v; // Velocity
    std::shared_ptr<Spatial1D> spatial;

    Wave1DState(std::shared_ptr<Spatial1D> s, double initial_displacement = 0.0) 
        : u(s->nx, initial_displacement), v(s->nx, 0.0), spatial(s) {}

    void update(const std::vector<double>& delta) override {
        size_t n = u.size();
        for (size_t i = 0; i < n; ++i) {
            u[i] += delta[i];
            v[i] += delta[i + n];
        }
    }

    std::vector<double> to_vector() const override {
        size_t n = u.size();
        Vector combined(2 * n);
        for (size_t i = 0; i < n; ++i) {
            combined[i] = u[i];
            combined[i + n] = v[i];
        }
        return combined;
    }

    std::unique_ptr<IState> clone() const override {
        auto copy = std::make_unique<Wave1DState>(spatial, 0.0);
        copy->u = this->u;
        copy->v = this->v;
        return copy;
    }
};

} // namespace mod
