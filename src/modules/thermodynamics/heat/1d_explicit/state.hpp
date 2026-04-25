#pragma once
#include "lib/spatial.hpp"
#include "lib/interfaces.hpp"

namespace mod {

class Heat1DExplicitState : public top::IState {
public:
    top::Spatial1D spatial;
    std::vector<double> temperatures;

    Heat1DExplicitState(top::Spatial1D s, double initial_temp) 
        : spatial(s), temperatures(s.nx, initial_temp) {}

    void update(const std::vector<double>& delta) override {
        for (size_t i = 0; i < temperatures.size(); ++i) {
            temperatures[i] += delta[i];
        }
    }

    std::vector<double> to_vector() const override {
        return temperatures;
    }

    std::unique_ptr<IState> clone() const override {
        auto copy = std::make_unique<Heat1DExplicitState>(spatial, 0.0);
        copy->temperatures = this->temperatures;
        return copy;
    }
};

} // namespace mod