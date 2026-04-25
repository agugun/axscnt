#pragma once
#include "lib/spatial.hpp"
#include "lib/interfaces.hpp"

namespace mod {
using namespace top;

class Pressure1DState : public IState {
public:
    Vector pressures;
    std::shared_ptr<Spatial1D> spatial;

    Pressure1DState(std::shared_ptr<Spatial1D> s, double initial_press) 
        : pressures(s->nx, initial_press), spatial(s) {}

    void update(const std::vector<double>& delta) override {
        for (size_t i = 0; i < pressures.size(); ++i) {
            pressures[i] += delta[i];
        }
    }

    std::vector<double> to_vector() const override {
        return pressures;
    }

    std::unique_ptr<IState> clone() const override {
        auto copy = std::make_unique<Pressure1DState>(spatial, 0.0);
        copy->pressures = this->pressures;
        return copy;
    }
};

} // namespace mod
