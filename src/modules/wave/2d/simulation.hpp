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

namespace mod::wave {

class Wave2DImplicitSimulation {
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
        double c = config.get("c", 1.0);
        double rho = config.get("rho", 1.0);
        double area = config.get("area", 1.0);
        
        // 1. Grid and State
        auto spatial = std::make_shared<Spatial2D>(nx, ny, dx, dy);
        auto st = std::make_unique<Wave2DState>(spatial);
        
        // Initial condition: Gaussian pulse in the center
        for (size_t j = 0; j < ny; ++j) {
            for (size_t i = 0; i < nx; ++i) {
                double x = i * dx;
                double y = j * dy;
                double dist_sq = std::pow(x - 0.5, 2) + std::pow(y - 0.5, 2);
                st->u[spatial->idx(i, j)] = std::exp(-dist_sq / 0.01);
            }
        }

        // 2. Physics Model and Discretization
        auto cond = num::discretization::heat_cond_2d(nx, ny, dx, dy, c * c, area); 
        Vector storage = num::discretization::heat_storage(nx * ny, dx * dy * area, rho, 1.0);
        
        auto mdl = std::make_shared<Wave2DModel>(cond, storage);
        auto discretizer = std::make_shared<Wave2DDiscretizer>();
        
        // 3. Engine Components
        auto timer = std::make_shared<num::ImplicitEulerIntegrator>();
        auto linearizer = std::make_shared<num::NewtonRaphson>(1e-4, 12, true);
        
        // System is 2*N and non-symmetric, use BiCGSTAB
        auto solver = std::make_shared<num::BiCGSTABSolver>();
        solver->verbose = false;
        
        auto pm = std::make_shared<top::SerialParallelManager>();

        auto engine = std::make_unique<top::SimulationEngine>(spatial, mdl, discretizer, timer, linearizer, solver, pm);

        // 4. Logger / Observer setup
        auto logger = std::make_shared<utl::StandardLogger>(config);
        logger->set_grid(nx, ny, 1, dx);
        logger->add_field("Displacement", [](const top::IState& s) {
            auto v = s.to_vector();
            return Vector(v.begin(), v.begin() + v.size() / 2);
        });
        
        engine->add_observer(logger);

        return { std::move(engine), std::move(st), logger };
    }
};

} // namespace mod::wave
