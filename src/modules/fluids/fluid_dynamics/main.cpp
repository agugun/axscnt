#include <omp.h>
#include <iostream>
#include <cmath>
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

using namespace mod;
using namespace utl;
using namespace top;

struct BuildResult {
    std::unique_ptr<SimulationEngine> engine;
    std::unique_ptr<IState> st_init;
    std::shared_ptr<StandardLogger> logger;
};

BuildResult build_simulation(const ConfigReader& config) {
    double L = config.get("L", 1.0);
    double H = config.get("H", 1.0);
    int nx = config.get("nx", 10);
    int ny = config.get("ny", 10);
    double mu = config.get("mu", 0.01);
    double rho = config.get("rho", 1.0);
    
    // 1. Grid (Mesh) and State
    auto mesh = std::make_shared<Mesh>(Mesh::generate_quad_mesh(L, H, nx, ny));
    auto st = std::make_unique<FluidState>(mesh);
    
    // 2. Physics Model and Discretization
    auto mdl = std::make_shared<FluidModel>(mesh, mu, rho);
    auto discretizer = std::make_shared<FluidDiscretizer>();
    
    // Setup BCs (Lid-driven cavity example)
    for (int i = 0; i < mesh->num_nodes(); ++i) {
        double x = mesh->nodes[i].x;
        double y = mesh->nodes[i].y;
        
        // Top lid (y=H)
        if (std::abs(y - H) < 1e-6) {
            mdl->set_velocity_bc(i, 1.0, 0.0);
        }
        // Walls (x=0, x=L, y=0)
        else if (std::abs(x) < 1e-6 || std::abs(x - L) < 1e-6 || std::abs(y) < 1e-6) {
            mdl->set_velocity_bc(i, 0.0, 0.0);
        }
    }

    // 3. Engine Components
    auto timer = std::make_shared<num::ImplicitEulerIntegrator>();
    auto linearizer = std::make_shared<num::NewtonRaphson>(1e-4, 10, true);
    
    // FEM systems are usually non-symmetric and larger, BiCGSTAB is standard
    auto solver = std::make_shared<num::BiCGSTABSolver>();
    solver->verbose = false;
    
    auto pm = std::make_shared<SerialParallelManager>();

    auto engine = std::make_unique<SimulationEngine>(mesh, mdl, discretizer, timer, linearizer, solver, pm);

    // 4. Logger / Observer setup
    auto logger = std::make_shared<StandardLogger>(config);
    logger->set_grid(nx, ny, 1, L/nx);
    logger->add_field("U-Velocity", [](const IState& s) {
        const auto& fs = static_cast<const FluidState&>(s);
        return fs.u;
    });
    logger->add_field("P-Field", [](const IState& s) {
        const auto& fs = static_cast<const FluidState&>(s);
        return fs.p;
    });
    
    engine->add_observer(logger);

    return { std::move(engine), std::move(st), logger };
}

int main(int argc, char** argv) {
    std::string config_file = "input/fluid_fem.txt";
    for (int i = 1; i < argc; ++i) {
        if (argv[i][0] != '-') {
            config_file = argv[i];
            break;
        }
    }

    ConfigReader config;
    if (!config.load(config_file)) {
        std::cerr << "Failed to load config: " << config_file << "\n";
        return 1;
    }
    
    int num_threads = config.get("num_threads", 4);
    omp_set_num_threads(num_threads);

    auto [engine, st, logger] = build_simulation(config);

    double dt = config.get("dt", 0.1);
    double t_end = config.get("t_end", 5.0);

    std::cout << "Starting 2D Fluid Dynamics Simulation (FEM) [Refactored Architecture]...\n";
    engine->run(t_end, dt, std::move(st));

    std::cout << "Fluid Simulation Successfully Completed.\n";
    return 0;
}
