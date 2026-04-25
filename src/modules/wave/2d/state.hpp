#pragma once
#include "lib/spatial.hpp"
#include "lib/interfaces.hpp"

namespace mod::wave {
using namespace top;

class Wave2DState : public IState {
public:
    Vector u; // Displacement
    Vector v; // Velocity
    std::shared_ptr<Spatial2D> spatial;

    Wave2DState(std::shared_ptr<Spatial2D> s) 
        : u(s->total_size(), 0.0), v(s->total_size(), 0.0), spatial(s) {}

    void update(const std::vector<double>& delta) override {
        size_t n = spatial->total_size();
        for (size_t i = 0; i < n; ++i) {
            u[i] += delta[i];
            v[i] += delta[i + n];
        }
    }

    std::vector<double> to_vector() const override {
        size_t n = spatial->total_size();
        Vector combined(2 * n);
        for (size_t i = 0; i < n; ++i) {
            combined[i] = u[i];
            combined[i + n] = v[i];
        }
        return combined;
    }

    std::unique_ptr<IState> clone() const override {
        auto copy = std::make_unique<Wave2DState>(spatial);
        copy->u = this->u;
        copy->v = this->v;
        return copy;
    }
};

} // namespace mod::wave
