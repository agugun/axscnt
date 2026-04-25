#pragma once
#include "lib/spatial.hpp"
#include "../heat_state.hpp"

namespace mod {
using namespace top;

class Heat2DImplicitState : public HeatState {
public:
    std::shared_ptr<Spatial2D> spatial;

    Heat2DImplicitState(std::shared_ptr<Spatial2D> s, double initial_temp = 0.0) 
        : HeatState(s->total_size(), initial_temp), spatial(s) {}

    std::unique_ptr<IState> clone() const override {
        auto copy = std::make_unique<Heat2DImplicitState>(spatial, 0.0);
        copy->temperatures = this->temperatures;
        return copy;
    }
};

} // namespace mod
