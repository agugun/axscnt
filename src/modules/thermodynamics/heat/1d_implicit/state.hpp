#pragma once
#include "lib/spatial.hpp"
#include "../heat_state.hpp"

namespace mod {
using namespace top;

class Heat1DImplicitState : public HeatState {
public:
    std::shared_ptr<Spatial1D> spatial;

    Heat1DImplicitState(std::shared_ptr<Spatial1D> s, double initial_temp) 
        : HeatState(s->nx, initial_temp), spatial(s) {}

    std::unique_ptr<IState> clone() const override {
        auto copy = std::make_unique<Heat1DImplicitState>(spatial, 0.0);
        copy->temperatures = this->temperatures;
        return copy;
    }
};

} // namespace mod
