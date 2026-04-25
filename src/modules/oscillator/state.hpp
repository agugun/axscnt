#pragma once
#include "lib/interfaces.hpp"
#include <vector>
#include <memory>

namespace mod {

// Strong semantic typing for physical quantities
using Position_m = double;
using Velocity_ms = double;

/**
 * @brief State object for the Harmonic Oscillator.
 * Adheres to the notebook structure: State -> Properties -> Discretization -> Assembly
 */
class OscillatorState : public top::IState {
public:
    Position_m x;
    Velocity_ms v;

    OscillatorState(Position_m x0, Velocity_ms v0) : x(x0), v(v0) {}

    void update(const std::vector<double>& delta) override {
        // delta is the Newton-Raphson update or Explicit update
        x += delta[0];
        v += delta[1];
    }

    std::vector<double> to_vector() const override {
        return {x, v};
    }

    std::unique_ptr<top::IState> clone() const override {
        return std::make_unique<OscillatorState>(x, v);
    }
};

} // namespace mod
