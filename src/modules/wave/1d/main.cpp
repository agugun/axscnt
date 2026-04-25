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
    size_t nx = config.get("nx", 200);
    double dx = config.get("dx", 0.01);
    double c = config.get("c", 1.0);
    double rho = config.get("rho", 1.0);
    double area = config.get("area", 1.0);
    
    // 1. Grid and State
    auto spatial = std::make_shared<Spatial1D>(nx, dx);
    auto st = std::make_unique<Wave1DState>(spatial, 0.0);
    
    // Initial condition: Gaussian pulse in displacement
    for (size_t i = 0; i < nx; ++i) {
        double x = i * dx;
        st->u[i] = std::exp(-std::pow(x - 0.5, 2) / 0.01);
    }

    // 2. Physics Model and Discretization
    auto cond = num::discretization::heat_cond_1d(nx, dx, c * c, area); 
    Vector storage = num::discretization::heat_storage(nx, dx * area, rho, 1.0);
    
    auto mdl = std::make_shared<Wave1DModel>(cond, storage);
    auto discretizer = std::make_shared<Wave1DDiscretizer>();
    
    // 3. Engine Components
    auto timer = std::make_shared<num::ImplicitEulerIntegrator>();
    auto linearizer = std::make_shared<num::NewtonRaphson>(1e-4, 12, true);
    
    // Use BiCGSTAB as the system is 2N and non-tridiagonal (due to u-v coupling)
    auto solver = std::make_shared<num::BiCGSTABSolver>();
    
    auto pm = std::make_shared<SerialParallelManager>();

    auto engine = std::make_unique<SimulationEngine>(spatial, mdl, discretizer, timer, linearizer, solver, pm);

    // 4. Logger / Observer setup
    auto logger = std::make_shared<StandardLogger>(config);
    logger->set_grid(nx, 1, 1, dx);
    logger->add_field("Displacement", [](const IState& s) {
        auto v = s.to_vector();
        return Vector(v.begin(), v.begin() + v.size() / 2);
    });
    
    engine->add_observer(logger);

    return { std::move(engine), std::move(st), logger };
}

int main(int argc, char** argv) {
    std::string config_file = "input/wave_1d.txt";
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

    double dt = config.get("dt", 0.05);
    double t_end = config.get("t_end", 10.0);

    std::cout << "Starting 1D Wave Simulation [Modular Engine Architecture]...\n";
    engine->run(t_end, dt, std::move(st));

    std::cout << "Wave Simulation Successfully Completed.\n";
    return 0;
}
