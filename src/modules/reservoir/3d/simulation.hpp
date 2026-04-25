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

class Reservoir3DImplicitSimulation {
public:
    struct BuildResult {
        std::unique_ptr<top::SimulationEngine> engine;
        std::unique_ptr<top::IState> st_init;
        std::shared_ptr<utl::StandardLogger> logger;
    };

    static BuildResult build(const utl::ConfigReader& config) {
        size_t nx = config.get("nx", 20);
        size_t ny = config.get("ny", 20);
        size_t nz = config.get("nz", 10);
        double dx = config.get("dx", 50.0);
        double dy = config.get("dy", 50.0);
        double dz = config.get("dz", 20.0);
        double k = config.get("k", 100.0);
        double phi = config.get("phi", 0.2);
        double mu = config.get("mu", 1.0);
        double ct = config.get("ct", 1e-6);
        double area = config.get("area", 1.0); // For 3D, area is embedded in dx*dy or dy*dz
        
        // 1. Grid and State
        auto spatial = std::make_shared<Spatial3D>(nx, ny, nz, dx, dy, dz);
        auto st = std::make_unique<Reservoir3DState>(spatial, config.get("p_initial", 3000.0));
        
        // 2. Physics Model and Discretization
        auto cond = num::discretization::pressure_cond_3d(nx, ny, nz, dx, dy, dz, k, area);
        Vector storage = num::discretization::pressure_storage(nx * ny * nz, dx * dy * dz * area, phi, ct);
        
        // Wells
        std::vector<std::shared_ptr<ISourceSink>> wells;
        auto idx_func = [nx, ny](int i, int j, int k) { return (k * (int)ny + j) * (int)nx + i; };
        
        // Producer at one corner (Top slice)
        wells.push_back(std::make_shared<ConstantRateWell>(
            0, 0, 0, 0, -std::abs(config.get("q_producer", 500.0)), 1.0, idx_func
        ));
        
        // Injector at opposite corner (Bottom slice)
        wells.push_back(std::make_shared<ConstantRateWell>(
            (int)nx-1, (int)ny-1, (int)nz-1, (int)nz-1, std::abs(config.get("q_injector", 500.0)), 1.0, idx_func
        ));

        auto mdl = std::make_shared<Reservoir3DModel>(cond, storage, wells);
        auto discretizer = std::make_shared<Reservoir3DDiscretizer>();
        
        // 3. Engine Components
        auto timer = std::make_shared<num::ImplicitEulerIntegrator>();
        auto linearizer = std::make_shared<num::NewtonRaphson>(1e-4, 12, true);
        
        // Mandartory BiCGSTAB for 3D
        auto solver = std::make_shared<num::BiCGSTABSolver>();
        solver->verbose = false;
        
        auto pm = std::make_shared<top::SerialParallelManager>();

        auto engine = std::make_unique<top::SimulationEngine>(spatial, mdl, discretizer, timer, linearizer, solver, pm);

        // 4. Logger / Observer setup
        auto logger = std::make_shared<utl::StandardLogger>(config);
        logger->set_grid(nx, ny, nz, dx);
        logger->add_field("Pressure", [](const top::IState& s) {
            return s.to_vector();
        });
        
        engine->add_observer(logger);

        return { std::move(engine), std::move(st), logger };
    }
};

} // namespace mod::reservoir
