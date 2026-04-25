#include <iostream>
#include <fstream>
#include <memory>
#include "state.hpp"
#include "model.hpp"
#include "lib/simulation.hpp"
#include "lib/integrators.hpp"
#include "lib/linearizers.hpp"
#include "lib/solvers.hpp"
#include "lib/engine_infra.hpp"

using namespace mod;
using namespace top;
using namespace num;

/**
 * @brief Custom Integrator to enforce a specific dt.
 */
class FixedDtIntegrator : public num::ImplicitEulerIntegrator {
    double fixed_dt;
public:
    FixedDtIntegrator(double dt) : fixed_dt(dt) {}
    double compute_dt(const top::IState& st, double t) const override { 
        return fixed_dt; 
    }
};

/**
 * @brief Observer to write st variables to CSV.
 */
class CSVObserver : public IObserver {
    std::ofstream out;
public:
    CSVObserver(const std::string& filename) {
        out.open(filename);
        out << "Time,Position,Velocity\n";
    }
    void on_step_complete(double t, int step, const IState& base_state) override {
        const auto& st = dynamic_cast<const OscillatorState&>(base_state);
        out << t << "," << st.x << "," << st.v << "\n";
    }
};

int main() {
    std::cout << "Starting Harmonic Oscillator Simulation..." << std::endl;

    // 1. Initial State: x(0) = 1, v(0) = 0
    auto st_init = std::make_unique<OscillatorState>(1.0, 0.0);

    // 2. Model: m=1.0, c=0.0 (undamped to match SymPy phase 1), k=4.0 (omega=2.0)
    auto mdl = std::make_shared<OscillatorModel>(1.0, 0.0, 4.0);
    auto grd = std::make_shared<OscillatorGrid>();
    auto discretizer = std::make_shared<OscillatorDiscretizer>();

    // 3. Numerical Engine Components
    double dt = 0.05;
    auto timer = std::make_shared<FixedDtIntegrator>(dt);
    auto linearizer = std::make_shared<NewtonRaphson>(1e-6, 10, false);
    auto solver = std::make_shared<LUSolver>();
    auto pm = std::make_shared<SerialParallelManager>();

    SimulationEngine engine(grd, mdl, discretizer, timer, linearizer, solver, pm);

    auto observer = std::make_shared<CSVObserver>("exports/oscillator_output.csv");
    engine.add_observer(observer);

    // 4. Run Simulation
    double t_max = 10.0;
    
    // We add an epsilon to ensure the final step completes
    engine.run(t_max + 1e-4, dt, std::move(st_init));

    std::cout << "Simulation complete. Output saved to exports/oscillator_output.csv" << std::endl;
    return 0;
}
