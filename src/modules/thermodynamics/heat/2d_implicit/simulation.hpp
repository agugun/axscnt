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

namespace mod::heat {

class Heat2DImplicitSimulation {
public:
    struct BuildResult {
        std::unique_ptr<top::SimulationEngine> engine;
        std::unique_ptr<top::IState> st_init;
        std::shared_ptr<utl::StandardLogger> logger;
    };

    static BuildResult build(const utl::ConfigReader& config) {
        size_t nx = config.get("nx", 50);
        size_t ny = config.get("ny", 50);
        double dx = config.get("dx", 0.02);
        double dy = config.get("dy", 0.02);
        double k = config.get("k", 0.1);
        double rho = config.get("rho", 1.0);
        double cp = config.get("cp", 1.0);
        double area = config.get("area", 1.0);
        
        // 1. Grid and State
        auto spatial = std::make_shared<Spatial2D>(nx, ny, dx, dy);
        auto st = std::make_unique<Heat2DImplicitState>(spatial, 0.0);
        
        // Initial condition: Hot spot in center
        for (size_t j = 0; j < ny; ++j) {
            for (size_t i = 0; i < nx; ++i) {
                double x = i * dx;
                double y = j * dy;
                double dist_sq = std::pow(x - 0.5, 2) + std::pow(y - 0.5, 2);
                st->temperatures[spatial->idx(i, j)] = std::exp(-dist_sq / 0.02);
            }
        }

        // 2. Physics Model and Discretization
        auto cond = num::discretization::heat_cond_2d(nx, ny, dx, dy, k, area);
        Vector storage = num::discretization::heat_storage(nx * ny, dx * dy * area, rho, cp);
        
        auto mdl = std::make_shared<Heat2DModel>(
            cond, storage, 
            config.get("t_top", 0.0), config.get("t_bottom", 0.0),
            config.get("t_left", 0.0), config.get("t_right", 0.0)
        );
        auto discretizer = std::make_shared<Heat2DDiscretizer>();
        
        // 3. Engine Components
        auto timer = std::make_shared<num::ImplicitEulerIntegrator>();
        auto linearizer = std::make_shared<num::NewtonRaphson>(1e-6, 12, true);
        
        // For 2D, we use BiCGSTAB as LU is too slow
        auto solver = std::make_shared<num::BiCGSTABSolver>();
        solver->verbose = false;
        
        auto pm = std::make_shared<top::SerialParallelManager>();

        auto engine = std::make_unique<top::SimulationEngine>(spatial, mdl, discretizer, timer, linearizer, solver, pm);

        // 4. Logger / Observer setup
        auto logger = std::make_shared<utl::StandardLogger>(config);
        logger->set_grid(nx, ny, 1, dx);
        logger->add_field("Temperature", [](const top::IState& s) {
            return s.to_vector();
        });
        
        engine->add_observer(logger);

        return { std::move(engine), std::move(st), logger };
    }
};

} // namespace mod::heat
