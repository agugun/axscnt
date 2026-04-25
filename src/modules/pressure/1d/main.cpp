#include <omp.h>
#include <iostream>
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
    double dx = config.get("dx", 10.0);
    double k = config.get("k", 50.0); // mD
    double phi = config.get("phi", 0.2);
    double mu = config.get("mu", 1.0); // cP
    double ct = config.get("ct", 1e-6); // 1/psi
    double area = config.get("area", 100.0);
    
    // 1. Grid and State
    auto spatial = std::make_shared<Spatial1D>(nx, dx);
    auto st = std::make_unique<Pressure1DState>(spatial, config.get("p_initial", 2000.0));
    
    // 2. Physics Model and Discretization
    auto cond = num::discretization::pressure_cond_1d(nx, dx, k, mu, area);
    Vector storage = num::discretization::pressure_storage(nx, dx * area, phi, ct);
    
    auto mdl = std::make_shared<Pressure1DModel>(
        cond, storage, config.get("p_left", 3000.0), config.get("p_right", 1000.0)
    );
    auto discretizer = std::make_shared<Pressure1DDiscretizer>();
    
    // 3. Engine Components
    auto timer = std::make_shared<num::ImplicitEulerIntegrator>();
    auto linearizer = std::make_shared<num::NewtonRaphson>(1e-6, 12, true);
    auto solver = std::make_shared<num::LinearTridiagonalSolver>();
    auto pm = std::make_shared<SerialParallelManager>();

    auto engine = std::make_unique<SimulationEngine>(spatial, mdl, discretizer, timer, linearizer, solver, pm);

    // 4. Logger / Observer setup
    auto logger = std::make_shared<StandardLogger>(config);
    logger->set_grid(nx, 1, 1, dx);
    logger->add_field("Pressure", [](const IState& s) {
        return s.to_vector();
    });
    
    engine->add_observer(logger);

    return { std::move(engine), std::move(st), logger };
}

int main(int argc, char** argv) {
    std::string config_file = "input/pressure_1d.txt";
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

    std::cout << "Starting 1D Pressure Simulation [Standard Engine Architecture]...\n";
    engine->run(t_end, dt, std::move(st));

    std::cout << "Pressure Simulation Successfully Completed.\n";
    return 0;
}
