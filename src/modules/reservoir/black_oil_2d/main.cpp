#include <omp.h>
#include <iostream>
#include "simulation.hpp"
#include "lib/utils/config_reader.hpp"

using namespace mod::reservoir;
using namespace utl;

int main(int argc, char** argv) {
    std::string config_file = "input/reservoir_black_oil_2d.txt";
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

    auto [engine, st, logger] = BlackOil2DImplicitSimulation::build(config);

    double dt = config.get("dt", 0.01);
    double t_end = config.get("t_end", 1.0);

    std::cout << "Starting 2D Full Black Oil Simulation (3-Phase) [Modular Engine Architecture]...\n";
    engine->run(t_end, dt, std::move(st));

    std::cout << "Successfully completed Black Oil Simulation.\n";
    return 0;
}
