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

class Reservoir2DImplicitSimulation {
public:
    struct BuildResult {
        std::unique_ptr<top::SimulationEngine> engine;
        std::unique_ptr<top::IState> st_init;
        std::shared_ptr<utl::StandardLogger> logger;
    };

    static BuildResult build(const utl::ConfigReader& config) {
        size_t nx = config.get("nx", 50);
        size_t ny = config.get("ny", 50);
        double dx = config.get("dx", 50.0);
        double dy = config.get("dy", 50.0);
        double k = config.get("k", 100.0);
        double phi = config.get("phi", 0.2);
        double mu = config.get("mu", 1.0);
        double ct = config.get("ct", 1e-6);
        double area = config.get("area", 100.0);
        
        // 1. Grid and State
        auto spatial = std::make_shared<Spatial2D>(nx, ny, dx, dy);
        auto st = std::make_unique<Reservoir2DState>(spatial, config.get("p_initial", 3000.0));
        
        // 2. Physics Model and Discretization
        auto cond = num::discretization::pressure_cond_2d(nx, ny, dx, dy, k, mu, area);
        Vector storage = num::discretization::pressure_storage(nx * ny, dx * dy * area, phi, ct);
        
        // Wells
        std::vector<std::shared_ptr<ISourceSink>> wells;
        if (config.get("q_producer", 0.0) != 0.0) {
            auto idx_func = [nx, ny](int i, int j, int k) { return j * (int)nx + i; };
            // Producer at center
            wells.push_back(std::make_shared<ConstantRateWell>(
                (int)nx/2, (int)ny/2, 0, 0, -std::abs(config.get("q_producer", 1000.0)), 1.0, idx_func
            ));
        }

        auto mdl = std::make_shared<Reservoir2DModel>(cond, storage, wells);
        auto discretizer = std::make_shared<Reservoir2DDiscretizer>();
        
        // 3. Engine Components
        auto timer = std::make_shared<num::ImplicitEulerIntegrator>();
        auto linearizer = std::make_shared<num::NewtonRaphson>(1e-4, 12, true);
        
        // 2D Reservoir matrices can be large, use BiCGSTAB
        auto solver = std::make_shared<num::BiCGSTABSolver>();
        solver->verbose = false;
        
        auto pm = std::make_shared<top::SerialParallelManager>();

        auto engine = std::make_unique<top::SimulationEngine>(spatial, mdl, discretizer, timer, linearizer, solver, pm);

        // 4. Logger / Observer setup
        auto logger = std::make_shared<utl::StandardLogger>(config);
        logger->set_grid(nx, ny, 1, dx);
        logger->add_field("Pressure", [](const top::IState& s) {
            return s.to_vector();
        });
        
        engine->add_observer(logger);

        return { std::move(engine), std::move(st), logger };
    }
};

} // namespace mod::reservoir
