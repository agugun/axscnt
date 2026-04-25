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

class DualPhase2DImplicitSimulation {
public:
    struct BuildResult {
        std::unique_ptr<top::SimulationEngine> engine;
        std::unique_ptr<top::IState> st_init;
        std::shared_ptr<utl::StandardLogger> logger;
    };

    static BuildResult build(const utl::ConfigReader& config) {
        size_t nx = config.get("nx", 30);
        size_t ny = config.get("ny", 30);
        double dx = config.get("dx", 100.0);
        double dy = config.get("dy", 100.0);
        double k = config.get("k", 100.0);
        double phi = config.get("phi", 0.2);
        double area = config.get("area", 1.0);
        
        // 1. Grid and State
        auto spatial = std::make_shared<Spatial2D>(nx, ny, dx, dy);
        auto st = std::make_unique<ReservoirDualPhase2DState>(spatial, config.get("p_init", 2000.0), config.get("sw_init", 0.2));
        
        // 2. Physics Model and Discretization
        auto cond = num::discretization::pressure_cond_2d(nx, ny, dx, dy, k, 1.0, area); // mu=1 stub
        double pv = dx * dy * area * phi;
        
        // Wells
        std::vector<std::shared_ptr<ISourceSink>> wells;
        
        // Water Injector at Bottom-Left
        wells.push_back(std::make_shared<ConstantRateWell>(
            0, 0, 0, 0, std::abs(config.get("q_injector", 500.0)), 1.0, 
            [nx, ny](int i, int j, int k) { return j * (int)nx + i; }
        ));

        // Fluid Producer at Top-Right
        wells.push_back(std::make_shared<ConstantRateWell>(
            (int)nx-1, (int)ny-1, (int)nx-1, (int)ny-1, -std::abs(config.get("q_producer", 500.0)), 1.0, 
            [nx, ny](int i, int j, int k) { return j * (int)nx + i; }
        ));

        auto mdl = std::make_shared<DualPhase2DModel>(cond, pv, config.get("mu_w", 1.0), config.get("mu_o", 2.0), wells);
        auto discretizer = std::make_shared<DualPhase2DDiscretizer>();
        
        // 3. Engine Components
        auto timer = std::make_shared<num::ImplicitEulerIntegrator>();
        auto linearizer = std::make_shared<num::NewtonRaphson>(1e-4, 12, true);
        auto solver = std::make_shared<num::BiCGSTABSolver>();
        solver->verbose = false;
        
        auto pm = std::make_shared<top::SerialParallelManager>();

        auto engine = std::make_unique<top::SimulationEngine>(spatial, mdl, discretizer, timer, linearizer, solver, pm);

        // 4. Logger / Observer setup
        auto logger = std::make_shared<utl::StandardLogger>(config);
        logger->set_grid(nx, ny, 1, dx);
        logger->add_field("Pressure", [](const top::IState& s) {
            auto v = s.to_vector();
            Vector p(v.size() / 2);
            for (size_t i = 0; i < p.size(); ++i) p[i] = v[2 * i];
            return p;
        });
        logger->add_field("WaterSat", [](const top::IState& s) {
            auto v = s.to_vector();
            Vector sw(v.size() / 2);
            for (size_t i = 0; i < sw.size(); ++i) sw[i] = v[2 * i + 1];
            return sw;
        });
        
        engine->add_observer(logger);

        return { std::move(engine), std::move(st), logger };
    }
};

} // namespace mod::reservoir
