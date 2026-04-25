#include <omp.h>
#include <iostream>
#include <memory>
#include <cmath>
#include "lib/simulation.hpp"
#include "lib/linearizers.hpp"
#include "lib/engine_infra.hpp"
#include "lib/discretization.hpp"
#include "lib/utils/config_reader.hpp"
#include "lib/utils/logger.hpp"
#include "lib/solvers.hpp"
#include "lib/integrators.hpp"
#include "state.hpp"
#include "model.hpp"

using namespace top;
using namespace utl;
using namespace mod;

struct BuildResult {
    std::unique_ptr<SimulationEngine> engine;
    std::unique_ptr<IState> st_init;
    std::shared_ptr<StandardLogger> logger;
};

BuildResult build_simulation(const ConfigReader& config) {
    size_t nx = config.get("nx", 100);
    double dx = config.get("dx", 0.01);
    double k = config.get("k", config.get("alpha", 0.1));
    double rho = config.get("rho", 1.0);
    double cp = config.get("cp", 1.0);
    double area = config.get("area", 1.0);
    
    // 1. Grid and State
    auto spatial = std::make_shared<Spatial1D>(nx, dx);
    auto st = std::make_unique<Heat1DImplicitState>(spatial, 0.0);
    for (size_t i = 0; i < nx; ++i) {
        double x = i * dx;
        st->temperatures[i] = std::exp(-std::pow(x - 0.5 * (nx-1) * dx, 2) / 0.02);
    }

    // 2. Physics Model and Discretization
    auto cond = num::discretization::heat_cond_1d(nx, dx, k, area);
    Vector storage = num::discretization::heat_storage(nx, dx * area, rho, cp);
    auto mdl = std::make_shared<Heat1DModel>(cond, storage, config.get("t_left", 0.0), config.get("t_right", 0.0));
    auto discretizer = std::make_shared<Heat1DDiscretizer>();
    
    // 3. Engine Components
    auto timer = std::make_shared<num::ImplicitEulerIntegrator>();
    auto linearizer = std::make_shared<num::NewtonRaphson>(1e-6, 12, true);
    auto solver = std::make_shared<num::LinearTridiagonalSolver>();
    auto pm = std::make_shared<SerialParallelManager>();

    auto engine = std::make_unique<SimulationEngine>(spatial, mdl, discretizer, timer, linearizer, solver, pm);

    // 4. Logger / Observer setup
    auto logger = std::make_shared<StandardLogger>(config);
    logger->set_grid(nx, 1, 1, dx);
    logger->add_field("Temperature", [](const IState& s) {
        return s.to_vector();
    });
    
    engine->add_observer(logger);

    return { std::move(engine), std::move(st), logger };
}

int main(int argc, char** argv) {
    // 1. Controller: Configuration & Environment
    std::string config_file = "input/heat_1d_implicit.txt";
    for (int i = 1; i < argc; ++i) {
        if (argv[i][0] != '-') {
            config_file = argv[i];
            break;
        }
    }

    ConfigReader config;
    config.load(config_file);
    
    // Command line overrides
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--vtk") config.set("enable_vtk", 1);
        if (arg == "--csv") config.set("enable_csv", 1);
    }

    omp_set_num_threads(config.get("num_threads", 1));

    // 2. Construction: Simulation Engine & Logger
    auto [engine, st, logger] = build_simulation(config);

    // 3. Execution: Orchestration
    double t_end = config.get("t_end", 2.0);
    double dt = config.get("dt", 0.01);

    std::cout << "Starting Heat 1D Implicit Simulation\n";
    engine->run(t_end, dt, std::move(st));

    std::cout << "Simulation Successful.\n";
    return 0;
}
