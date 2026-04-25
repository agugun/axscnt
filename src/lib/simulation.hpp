/**
 * @file simulation.hpp
 * @brief High-level Orchestrator for Time-Stepped Simulations.
 */
#pragma once
#include "interfaces.hpp"
#include <vector>
#include <memory>

namespace top {

class SimulationEngine {
private:
    std::shared_ptr<IGrid> grd;
    std::shared_ptr<IModel> mdl;
    std::shared_ptr<IDiscretizer> discretizer;
    std::shared_ptr<ITimeIntegrator> timer;
    std::shared_ptr<ILinearizer> linearizer;
    std::shared_ptr<ISolver> solver;
    std::shared_ptr<IParallelManager> parallel;
    std::vector<std::shared_ptr<ISourceSink>> sources;
    std::vector<std::shared_ptr<IObserver>> observers;

public:
    SimulationEngine(std::shared_ptr<IGrid> g,
                     std::shared_ptr<IModel> m,
                     std::shared_ptr<IDiscretizer> d,
                     std::shared_ptr<ITimeIntegrator> t,
                     std::shared_ptr<ILinearizer> l,
                     std::shared_ptr<ISolver> s,
                     std::shared_ptr<IParallelManager> p,
                     std::vector<std::shared_ptr<ISourceSink>> src = {})
        : grd(g), mdl(m), discretizer(d), timer(t), linearizer(l), 
          solver(s), parallel(p), sources(std::move(src)) {
        if (linearizer) linearizer->set_sources(sources);
    }

    void add_observer(std::shared_ptr<IObserver> ob) {
        observers.push_back(ob);
    }

    /**
     * @brief Perform a single time step and notify observers.
     * @return unique_ptr to the next state.
     */
    std::unique_ptr<IState> step(double t, double dt, const IState& st_curr, int step_count = 0) {
        auto st_next = linearizer->resolve(
            st_curr, dt, *grd, *mdl, *discretizer, *timer, *solver, *parallel
        );

        for (auto& obs : observers) {
            obs->on_step_complete(t + dt, step_count, *st_next);
        }

        return st_next;
    }

    /**
     * @brief Run the full simulation until t_max.
     */
    std::unique_ptr<IState> run(double t_max, double dt_initial, std::unique_ptr<IState> st_init) {
        auto st_n = std::move(st_init);
        double t = 0.0;
        int step_count = 0;

        for (auto& obs : observers) obs->on_simulation_start(*grd);
        for (auto& obs : observers) obs->on_step_complete(t, step_count, *st_n);

        while (t < t_max) {
            double dt = timer->compute_dt(*st_n, t);
            if (t + dt > t_max) dt = t_max - t;

            st_n = step(t, dt, *st_n, step_count + 1);
            
            t += dt;
            step_count++;
        }

        for (auto& obs : observers) obs->on_simulation_end();
        return st_n;
    }
};

} // namespace top
