#include "modules/reservoir/well.hpp"
#pragma once
#include "lib/simulation.hpp"
#include "lib/linearizers.hpp"
#include "lib/engine_infra.hpp"
#include "lib/discretization.hpp"
#include "lib/utils/config_reader.hpp"
#include "lib/utils/logger.hpp"
#include "state.hpp"
#include "model.hpp"
#include "lib/solvers.hpp"
#include "lib/integrators.hpp"

namespace mod::reservoir {

class Reservoir1DImplicitSimulation {
public:
    struct BuildResult {
        std::unique_ptr<top::SimulationEngine> engine;
        std::unique_ptr<top::IState> st_init;
        std::shared_ptr<utl::StandardLogger> logger;
    };

    static BuildResult build(const utl::ConfigReader& config) {
        size_t nx = config.get("nx", 100);
        double dx = config.get("dx", 10.0);
        double k = config.get("k", 50.0);
        double phi = config.get("phi", 0.2);
        double mu = config.get("mu", 1.0);
        double ct = config.get("ct", 1e-6);
        double area = config.get("area", 100.0);
        
        // 1. Grid and State
        auto spatial = std::make_shared<Spatial1D>(nx, dx);
        auto st = std::make_unique<Reservoir1DState>(spatial, config.get("p_initial", 2000.0));
        
        // 2. Physics Model and Discretization
        auto cond = num::discretization::pressure_cond_1d(nx, dx, k, mu, area);
        Vector storage = num::discretization::pressure_storage(nx, dx * area, phi, ct);
        
        // Wells
        std::vector<std::shared_ptr<ISourceSink>> wells;
        if (config.get("q_well", 0.0) != 0.0) {
            auto idx_func = [](int i, int j, int k) { return i; };
            wells.push_back(std::make_shared<ConstantRateWell>(
                (int)nx/2, 0, 0, 0, config.get("q_well", 0.0), 1.0, idx_func
            ));
        }

        auto mdl = std::make_shared<Reservoir1DModel>(cond, storage, wells);
        auto discretizer = std::make_shared<Reservoir1DDiscretizer>();
        
        // 3. Engine Components
        auto timer = std::make_shared<num::ImplicitEulerIntegrator>();
        auto linearizer = std::make_shared<num::NewtonRaphson>(1e-4, 12, true);
        auto solver = std::make_shared<num::LinearTridiagonalSolver>();
        auto pm = std::make_shared<top::SerialParallelManager>();

        auto engine = std::make_unique<top::SimulationEngine>(spatial, mdl, discretizer, timer, linearizer, solver, pm);

        // 4. Logger / Observer setup
        auto logger = std::make_shared<utl::StandardLogger>(config);
        logger->set_grid(nx, 1, 1, dx);
        logger->add_field("Pressure", [](const top::IState& s) {
            return s.to_vector();
        });
        
        engine->add_observer(logger);

        return { std::move(engine), std::move(st), logger };
    }
};

} // namespace mod::reservoir
