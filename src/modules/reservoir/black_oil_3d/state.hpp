#pragma once
#include "lib/spatial.hpp"
#include "lib/interfaces.hpp"
#include <vector>
#include <algorithm>

namespace mod::reservoir {
using namespace top;

/**
 * @brief ReservoirBlackOil3DState tracks 3 variables per cell in 3D.
 * Primary Variables: [P, Sw, Sg] interleved.
 */
class ReservoirBlackOil3DState : public IState {
public:
    std::shared_ptr<Spatial3D> spatial;
    Vector variables;

    ReservoirBlackOil3DState(std::shared_ptr<Spatial3D> s)
        : spatial(s), variables(3 * s->total_size(), 0.0) {
        
        for (int i = 0; i < (int)s->total_size(); ++i) {
            variables[3 * i]     = 2000.0; // Initial Pressure
            variables[3 * i + 1] = 0.2;    // Initial Water (Connate)
            variables[3 * i + 2] = 0.0;    // Initial Gas
        }
    }

    void update(const std::vector<double>& delta) override {
        for (size_t i = 0; i < variables.size(); i += 3) {
            variables[i]     += delta[i];
            variables[i + 1] += delta[i + 1];
            variables[i + 2] += delta[i + 2];
            
            // Physical boundaries
            variables[i]     = std::max(14.7, variables[i]);
            variables[i + 1] = std::max(0.0, std::min(1.0, variables[i + 1]));
            variables[i + 2] = std::max(0.0, std::min(1.0 - variables[i + 1], variables[i + 2]));
        }
    }

    std::vector<double> to_vector() const override {
        return variables;
    }

    std::unique_ptr<IState> clone() const override {
        auto copy = std::make_unique<ReservoirBlackOil3DState>(spatial);
        copy->variables = this->variables;
        return copy;
    }
    
    // Helper accessors
    double p(int c) const { return variables[3 * c]; }
    double sw(int c) const { return variables[3 * c + 1]; }
    double sg(int c) const { return variables[3 * c + 2]; }
};

} // namespace mod::reservoir
