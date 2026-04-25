#pragma once
#include "lib/interfaces.hpp"

namespace mod {
using namespace top;

/**
 * @brief Base class for all Heat Simulation states.
 */
class HeatState : public IState {
public:
    Vector temperatures;

    HeatState(size_t size, double initial_temp) : temperatures(size, initial_temp) {}
    
    virtual ~HeatState() = default;

    /**
     * @brief Update temperature values using a delta vector.
     */
    void apply_update(const std::vector<double>& delta) override {
        for (size_t i = 0; i < temperatures.size(); ++i) {
            temperatures[i] += delta[i];
        }
    }

    std::vector<double> to_vector() const override {
        return temperatures;
    }

    // apply_update remains pure virtual in IState, implemented here.
    // clone() remains pure virtual to be implemented by spatial subclasses.
};

} // namespace mod
