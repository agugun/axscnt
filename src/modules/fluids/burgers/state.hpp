#pragma once
#include "lib/spatial.hpp"
#include "lib/interfaces.hpp"

namespace mod {
using namespace top;

/**
 * @brief State representing the scalar velocity field for Burgers' Equation.
 */
class BurgersState : public IState {
public:
    Vector u; // Velocity scalar field
    std::shared_ptr<Spatial1D> spatial;

    BurgersState(std::shared_ptr<Spatial1D> s, double initial_val = 0.0) 
        : u(s->nx, initial_val), spatial(s) {}

    void update(const std::vector<double>& delta) override {
        for (size_t i = 0; i < u.size(); ++i) {
            u[i] += delta[i];
        }
    }

    std::vector<double> to_vector() const override {
        return u;
    }

    std::unique_ptr<IState> clone() const override {
        auto copy = std::make_unique<BurgersState>(spatial, 0.0);
        copy->u = this->u;
        return copy;
    }
};

} // namespace mod
