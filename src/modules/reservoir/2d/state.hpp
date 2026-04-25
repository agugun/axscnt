#pragma once
#include "lib/spatial.hpp"
#include "lib/interfaces.hpp"

namespace mod {
using namespace top;

class Reservoir2DState : public IState {
public:
    Vector pressures; // [psi]
    std::shared_ptr<Spatial2D> spatial;

    Reservoir2DState(std::shared_ptr<Spatial2D> s, double initial_p)
        : pressures(s->total_size(), initial_p), spatial(s) {}

    void update(const std::vector<double>& delta) override {
        for (size_t i = 0; i < pressures.size(); ++i) {
            pressures[i] += delta[i];
        }
    }

    std::vector<double> to_vector() const override {
        return pressures;
    }

    std::unique_ptr<IState> clone() const override {
        auto copy = std::make_unique<Reservoir2DState>(spatial, 0.0);
        copy->pressures = this->pressures;
        return copy;
    }
};

} // namespace mod
