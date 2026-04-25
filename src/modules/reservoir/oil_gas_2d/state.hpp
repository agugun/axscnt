#pragma once
#include "lib/spatial.hpp"
#include "lib/interfaces.hpp"
#include <vector>
#include <algorithm>

namespace mod::reservoir {
using namespace top;

/**
 * @brief State representing the Oil-Gas multiphase system (Pressure and Gas Saturation).
 */
class ReservoirOilGas2DState : public IState {
public:
    std::shared_ptr<Spatial2D> spatial;
    Vector variables; // Interleaved [P, Sg, P, Sg, ...]
    
    const double swc = 0.2; // Fixed Connate Water

    ReservoirOilGas2DState(std::shared_ptr<Spatial2D> s, double p_init, double s_init)
        : spatial(s), variables(2 * s->total_size(), 0.0) {
        for (int i = 0; i < (int)s->total_size(); ++i) {
            variables[2 * i] = p_init;
            variables[2 * i + 1] = s_init;
        }
    }

    void update(const std::vector<double>& delta) override {
        for (size_t i = 0; i < variables.size(); i += 2) {
            variables[i] += delta[i];
            variables[i + 1] += delta[i + 1];
            
            // Physical boundaries
            variables[i] = std::max(14.7, variables[i]);
            variables[i + 1] = std::max(0.0, std::min(1.0 - swc, variables[i + 1]));
        }
    }

    std::vector<double> to_vector() const override {
        return variables;
    }

    std::unique_ptr<IState> clone() const override {
        auto copy = std::make_unique<ReservoirOilGas2DState>(spatial, 2000.0, 0.0);
        copy->variables = this->variables;
        return copy;
    }

    // Helper accessors
    double p(int c) const { return variables[2 * c]; }
    double sg(int c) const { return variables[2 * c + 1]; }
    double so(int c) const { return 1.0 - sg(c) - swc; }
};

} // namespace mod::reservoir
