#pragma once
#include "lib/simulation.hpp"
#include "lib/linearizers.hpp"
#include "lib/engine_infra.hpp"
#include "lib/utils/config_reader.hpp"
#include "lib/utils/logger.hpp"
#include "state.hpp"
#include "model.hpp"
#include "lib/solvers.hpp"
#include "lib/integrators.hpp"

namespace mod::reservoir {

class MBASimulationBuilder {
public:
    struct BuildResult {
        std::unique_ptr<top::SimulationEngine> engine;
        std::unique_ptr<top::IState> st_init;
        std::shared_ptr<utl::StandardLogger> logger;
    };

    static BuildResult build(const utl::ConfigReader& config) {
        double V = config.get("volume", 1e6);
        double ct = config.get("ct", 1e-5);
        double q = config.get("q", 500.0);
        double pi = config.get("p_init", 3000.0);
        
        // 1. Grid (None) and State
        auto st = std::make_unique<MBAState>(pi);
        
        // 2. Physics Model and Discretization
        auto mdl = std::make_shared<MBAModel>(V, ct, q);
        auto discretizer = std::make_shared<MBADiscretizer>();
        
        // 3. Engine Components
        auto timer = std::make_shared<num::ImplicitEulerIntegrator>();
        auto linearizer = std::make_shared<num::NewtonRaphson>(1e-4, 5, true);
        auto solver = std::make_shared<num::LinearTridiagonalSolver>(); // 1x1 is trivial
        auto pm = std::make_shared<top::SerialParallelManager>();

        auto engine = std::make_unique<top::SimulationEngine>(nullptr, mdl, discretizer, timer, linearizer, solver, pm);

        // 4. Logger / Observer setup
        auto logger = std::make_shared<utl::StandardLogger>(config);
        logger->add_field("Pressure", [](const top::IState& s) {
            return s.to_vector();
        });
        
        engine->add_observer(logger);

        return { std::move(engine), std::move(st), logger };
    }
};

} // namespace mod::reservoir
