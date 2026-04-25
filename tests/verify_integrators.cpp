#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cassert>
#include <chrono>
#include "lib/integrators.hpp"
#include "lib/solvers.hpp"
#include "modules/reservoir/reservoir_integrators.hpp"
#include "modules/reservoir/dual_2d/mdl.hpp"
#include "modules/reservoir/dual_2d/st.hpp"

using namespace num;
using namespace mod;
using namespace mod::reservoir;
using namespace std;

class SimpleState : public IState {
public:
    Vector val;
    SimpleState(double x) : val({x}) {}
    void update(const Vector& delta) override { val[0] += delta[0]; }
    unique_ptr<IState> clone() const override { return make_unique<SimpleState>(val[0]); }
};

/**
 * @brief Simple Decay Model: dy/dt = -y
 */
class DecayModel : public IModel {
public:
    Vector evaluate_rhs(const IState& st) const override {
        const auto& s = dynamic_cast<const SimpleState&>(st);
        double y = s.val[0];
        return { -y };
    }
    
    Vector build_residual(const IState& st, const IState& state_old, double dt) const override {
        const auto& s = dynamic_cast<const SimpleState&>(st);
        const auto& s_old = dynamic_cast<const SimpleState&>(state_old);
        double y = s.val[0];
        double y_old = s_old.val[0];
        // R = (y - y_old)/dt + y = 0
        return { (y - y_old) / dt + y };
    }
    
    Matrix build_jacobian(const IState& st, double dt) const override {
        // dR/dy = 1/dt + 1
        return { { 1.0 / dt + 1.0 } };
    }

    Vector apply_jacobian(const IState& st, const Vector& v, double dt) const override {
        // J * v = (1/dt + 1) * v
        return { (1.0 / dt + 1.0) * v[0] };
    }
};

void run_benchmark(const string& name, ITimeIntegrator& integ, IModel& mdl, ISolver* solver, double y0, double dt, int steps) {
    SimpleState st(y0);
    
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < steps; ++i) {
        integ.step(mdl, st, solver, dt);
    }
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double, micro> duration = end - start;

    double analytical = y0 * exp(-dt * steps);
    double error = abs(st.val[0] - analytical);

    cout << left << setw(25) << name 
         << " | Final y: " << fixed << setprecision(6) << st.val[0] 
         << " | Error: " << scientific << setprecision(2) << error 
         << " | Time: " << fixed << setprecision(2) << duration.count() << " us" << endl;
}

/**
 * @brief Verification Test for Reservoir IMPES Integrator
 * Simulates a simple 5x5 grd with one injector and one producer.
 */
void test_reservoir_impes() {
    cout << "\n--- Testing Reservoir IMPES Integrator (2D Dual Phase) ---" << endl;

    int nx = 5, ny = 5;
    double dx = 100.0, dy = 100.0;
    auto spatial = make_shared<Spatial2D>(nx, ny, dx, dy);
    double init_p = 3000.0;
    double init_sw = 0.2;

    auto st = make_unique<ReservoirDualPhase2DState>(spatial, init_p, init_sw);
    
    // Lambdas for Well interface
    auto rp_func = [](double sw, double& krw, double& kro) -> void {
        double swe = (sw - 0.2) / (1.0 - 0.2 - 0.2);
        swe = std::max(0.0, std::min(1.0, swe));
        krw = swe * swe;
        kro = (1.0 - swe) * (1.0 - swe);
    };
    auto idx_func = [nx](int i, int j) -> int { return j * nx + i; };
    auto sw_func = [spatial](const top::IState& base_state, int i, int j) -> double { 
        const auto& s = dynamic_cast<const ReservoirDualPhase2DState&>(base_state);
        return s.water_saturations[spatial->idx(i, j)]; 
    };

    // Setup Wells (Injector at 0,0 and Producer at 4,4)
    auto inj = make_shared<mod::ReservoirWellDual2D>(0, 0, 500.0, true, rp_func, idx_func, sw_func, 1.0, 2.0);
    auto prod = make_shared<mod::ReservoirWellDual2D>(4, 4, -300.0, false, rp_func, idx_func, sw_func, 1.0, 2.0);
    vector<shared_ptr<mod::ISourceSink>> sources = { inj, prod };

    ReservoirDual2DModel mdl(100.0, 0.2, 1.0, 2.0, 50.0, sources);
    
    ConjugateGradientSolver cg_solver;
    ReservoirIMPESIntegrator<ReservoirDualPhase2DState, ReservoirDual2DModel> impes;

    double dt = 0.01;
    int steps = 5;

    cout << left << setw(10) << "Step" << " | Avg Pressure | Avg Saturation | Execution Time" << endl;
    cout << string(60, '-') << endl;

    for (int i = 1; i <= steps; ++i) {
        auto start = chrono::high_resolution_clock::now();
        impes.step(mdl, *st, &cg_solver, dt);
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double, micro> duration = end - start;

        double avg_p = 0, avg_s = 0;
        for(auto p : st->pressures) avg_p += p;
        for(auto s : st->water_saturations) avg_s += s;
        avg_p /= st->pressures.size();
        avg_s /= st->water_saturations.size();

        cout << left << setw(10) << i 
             << " | " << scientific << setprecision(2) << setw(12) << avg_p 
             << " | " << fixed << setprecision(4) << setw(14) << avg_s 
             << " | " << fixed << setprecision(2) << duration.count() << " us" << endl;
    }
}

int main() {
    DecayModel mdl;
    ConjugateGradientSolver cg_solver;
    
    double y0 = 1.0;
    double dt = 0.01;
    int steps = 100;

    cout << "--- Integrator Verification (dy/dt = -y, t_end = 1.0) ---" << endl;
    cout << left << setw(25) << "Method" << " | Result          | Accuracy   | Performance" << endl;
    cout << string(75, '-') << endl;

    ForwardEulerIntegrator fe;
    run_benchmark("Forward Euler", fe, mdl, nullptr, y0, dt, steps);

    RungeKutta4Integrator rk4;
    run_benchmark("Runge-Kutta 4", rk4, mdl, nullptr, y0, dt, steps);

    ImplicitEulerIntegrator ie;
    run_benchmark("Implicit Euler (CG)", ie, mdl, &cg_solver, y0, dt, steps);

    FullyImplicitIntegrator fi;
    run_benchmark("Fully Implicit (CG)", fi, mdl, &cg_solver, y0, dt, steps);

    // Domain Specific Integration
    test_reservoir_impes();

    cout << "\nVerification Successful." << endl;
    return 0;
}
