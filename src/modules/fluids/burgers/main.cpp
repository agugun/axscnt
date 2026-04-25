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
    size_t nx = config.get("nx", 100);
    double dx = config.get("dx", 0.01);
    double nu = config.get("nu", 0.01);
    
    // 1. Grid and State
    auto spatial = std::make_shared<Spatial1D>(nx, dx);
    auto st = std::make_unique<BurgersState>(spatial, 0.0);
    
    // Initial condition: Sine wave
    for (size_t i = 0; i < nx; ++i) {
        double x = i * dx;
        st->u[i] = std::sin(2.0 * M_PI * x);
    }

    // 2. Physics Model and Discretization
    auto mdl = std::make_shared<BurgersModel>(nu, dx);
    auto discretizer = std::make_shared<BurgersDiscretizer>();
    
    // 3. Engine Components
    auto timer = std::make_shared<num::ImplicitEulerIntegrator>();
    auto linearizer = std::make_shared<num::NewtonRaphson>(1e-6, 15, true);
    
    // 1D non-linear system, use Tridiagonal if possible or BiCGSTAB
    auto solver = std::make_shared<num::BiCGSTABSolver>();
    solver->verbose = false;
    
    auto pm = std::make_shared<SerialParallelManager>();

    auto engine = std::make_unique<SimulationEngine>(spatial, mdl, discretizer, timer, linearizer, solver, pm);

    // 4. Logger / Observer setup
    auto logger = std::make_shared<StandardLogger>(config);
    logger->set_grid(nx, 1, 1, dx);
    logger->add_field("Velocity", [](const IState& s) {
        return s.to_vector();
    });
    
    engine->add_observer(logger);

    return { std::move(engine), std::move(st), logger };
}

int main(int argc, char** argv) {
    std::string config_file = "input/burgers_fdm.txt";
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

    double dt = config.get("dt", 0.01);
    double t_end = config.get("t_end", 2.0);

    std::cout << "Starting 1D Burgers' Equation Simulation (FDM) [Refactored Architecture]...\n";
    engine->run(t_end, dt, std::move(st));

    std::cout << "Burgers Simulation Successfully Completed.\n";
    return 0;
}
