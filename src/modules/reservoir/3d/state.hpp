#pragma once
#include "lib/spatial.hpp"
#include "lib/interfaces.hpp"
#include <vector>
#include <memory>

namespace mod::reservoir {
using namespace top;

/**
 * @brief State representing a 3D Single-Phase Reservoir (Pressure).
 */
class Reservoir3DState : public IState {
public:
    Vector pressures; // [psi]
    std::shared_ptr<Spatial3D> spatial;

    Reservoir3DState(std::shared_ptr<Spatial3D> s, double initial_p)
        : spatial(s), pressures(s->nx * s->ny * s->nz, initial_p) {}

    void update(const top::Vector& delta) override {
        for (size_t i = 0; i < pressures.size(); ++i) {
            pressures[i] += delta[i];
        }
    }

    top::Vector to_vector() const override {
        return pressures;
    }

    std::unique_ptr<IState> clone() const override {
        auto copy = std::make_unique<Reservoir3DState>(spatial, 0.0);
        copy->pressures = this->pressures;
        return copy;
    }
};

} // namespace mod::reservoir
