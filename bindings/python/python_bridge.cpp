#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "modules/thermodynamics/heat/1d_implicit/mdl.hpp"
#include "modules/thermodynamics/heat/1d_implicit/st.hpp"
#include "modules/thermodynamics/heat/2d_implicit/mdl.hpp"
#include "modules/thermodynamics/heat/2d_implicit/st.hpp"
#include "modules/pressure/1d/mdl.hpp"
#include "modules/pressure/1d/st.hpp"
#include "modules/wave/1d/mdl.hpp"
#include "modules/wave/1d/st.hpp"
#include "modules/fluids/burgers/mdl.hpp"
#include "modules/fluids/burgers/st.hpp"
#include "modules/fluids/fluid_dynamics/mdl.hpp"
#include "modules/fluids/fluid_dynamics/st.hpp"
#include "modules/reservoir/1d/mdl.hpp"
#include "modules/reservoir/1d/st.hpp"
#include "modules/reservoir/2d/mdl.hpp"
#include "modules/reservoir/2d/st.hpp"
#include "lib/simulation.hpp"
#include "lib/integrators.hpp"
#include "lib/solvers.hpp"
#include "lib/linearizers.hpp"
#include "lib/engine_infra.hpp"
#include "lib/fem.hpp"

namespace py = pybind11;
using namespace num;
using namespace mod;
using namespace top;

/**
 * @brief Numerical Wrapper for Heat Simulations
 */
class HeatSimulationWrapper {
private:
    std::shared_ptr<Heat1DModel> mdl;
    std::shared_ptr<Heat1DImplicitState> st;
    std::unique_ptr<SimulationEngine> engine;
    double t = 0.0;
    int step_count = 0;

public:
    HeatSimulationWrapper(std::shared_ptr<num::discretization::Conductance1D> cond, const Vector& storage, double TL, double TR) {
        int nx = (int)storage.size();
        auto spatial = std::make_shared<Spatial1D>(nx, 1.0);
        st = std::make_shared<Heat1DImplicitState>(spatial, 0.0);
        mdl = std::make_shared<Heat1DModel>(cond, storage, TL, TR);
        
        auto discretizer = std::make_shared<Heat1DDiscretizer>();
        auto integrator = std::make_shared<ImplicitEulerIntegrator>();
        auto solver = std::make_shared<LinearTridiagonalSolver>();
        auto linearizer = std::make_shared<NewtonRaphson>(1e-6, 1, false);
        auto pm = std::make_shared<SerialParallelManager>();

        engine = std::make_unique<SimulationEngine>(spatial, mdl, discretizer, integrator, linearizer, solver, pm);
    }

    void set_initial_condition(const std::vector<double>& ic) {
        st->temperatures = ic;
    }

    void step(double dt) {
        auto next_state = engine->step(t, dt, *st, ++step_count);
        st = std::shared_ptr<Heat1DImplicitState>(static_cast<Heat1DImplicitState*>(next_state.release()));
        t += dt;
    }

    std::vector<double> get_values() const {
        return st->to_vector();
    }
};

/**
 * @brief Numerical Wrapper for 2D Heat Simulations
 */
class Heat2DSimulationWrapper {
private:
    std::shared_ptr<Heat2DModel> mdl;
    std::shared_ptr<Heat2DImplicitState> st;
    std::unique_ptr<SimulationEngine> engine;
    double t = 0.0;
    int step_count = 0;

public:
    Heat2DSimulationWrapper(int nx, int ny, double Lx, double Ly, double alpha) {
        auto spatial = std::make_shared<Spatial2D>(nx, ny, Lx, Ly);
        st = std::make_shared<Heat2DImplicitState>(spatial, 0.0);
        double dx = Lx / (nx - 1);
        double dy = Ly / (ny - 1);
        auto cond = num::discretization::heat_cond_2d(nx, ny, dx, dy, alpha, 1.0);
        Vector storage(nx * ny, 1.0);
        mdl = std::make_shared<Heat2DModel>(cond, storage, 0.0, 0.0, 0.0, 0.0);
        
        auto discretizer = std::make_shared<Heat2DDiscretizer>();
        auto integrator = std::make_shared<ImplicitEulerIntegrator>();
        auto solver = std::make_shared<BiCGSTABSolver>();
        auto linearizer = std::make_shared<NewtonRaphson>(1e-6, 1, false);
        auto pm = std::make_shared<SerialParallelManager>();

        engine = std::make_unique<SimulationEngine>(spatial, mdl, discretizer, integrator, linearizer, solver, pm);
    }

    void set_initial_condition(const std::vector<double>& ic) {
        st->temperatures = ic;
    }

    void step(double dt) {
        auto next_state = engine->step(t, dt, *st, ++step_count);
        st = std::shared_ptr<Heat2DImplicitState>(static_cast<Heat2DImplicitState*>(next_state.release()));
        t += dt;
    }

    std::vector<double> get_values() const {
        return st->to_vector();
    }
};

/**
 * @brief Numerical Wrapper for Pressure Simulations
 */
class PressureSimulationWrapper {
private:
    std::shared_ptr<Pressure1DModel> mdl;
    std::shared_ptr<Pressure1DState> st;
    std::unique_ptr<SimulationEngine> engine;
    double t = 0.0;
    int step_count = 0;

public:
    PressureSimulationWrapper(std::shared_ptr<num::discretization::Conductance1D> cond, const Vector& storage, double PL, double PR) {
        int nx = (int)storage.size();
        auto spatial = std::make_shared<Spatial1D>(nx, 1.0);
        st = std::make_shared<Pressure1DState>(spatial, 0.0);
        mdl = std::make_shared<Pressure1DModel>(cond, storage, PL, PR);
        
        auto discretizer = std::make_shared<Pressure1DDiscretizer>();
        auto integrator = std::make_shared<ImplicitEulerIntegrator>();
        auto solver = std::make_shared<LinearTridiagonalSolver>();
        auto linearizer = std::make_shared<NewtonRaphson>(1e-6, 1, false);
        auto pm = std::make_shared<SerialParallelManager>();

        engine = std::make_unique<SimulationEngine>(spatial, mdl, discretizer, integrator, linearizer, solver, pm);
    }

    void set_initial_condition(const std::vector<double>& ic) {
        st->pressures = ic;
    }

    void step(double dt) {
        auto next_state = engine->step(t, dt, *st, ++step_count);
        st = std::shared_ptr<Pressure1DState>(static_cast<Pressure1DState*>(next_state.release()));
        t += dt;
    }

    std::vector<double> get_values() const {
        return st->to_vector();
    }
};

/**
 * @brief Numerical Wrapper for Wave Simulations
 */
class WaveSimulationWrapper {
private:
    std::shared_ptr<Wave1DModel> mdl;
    std::shared_ptr<Wave1DState> st;
    std::unique_ptr<SimulationEngine> engine;
    double t = 0.0;
    int step_count = 0;

public:
    WaveSimulationWrapper(std::shared_ptr<num::discretization::Conductance1D> cond, const Vector& storage) {
        int nx = (int)storage.size();
        auto spatial = std::make_shared<Spatial1D>(nx, 1.0);
        st = std::make_shared<Wave1DState>(spatial, 0.0);
        mdl = std::make_shared<Wave1DModel>(cond, storage);
        
        auto discretizer = std::make_shared<Wave1DDiscretizer>();
        auto integrator = std::make_shared<ImplicitEulerIntegrator>();
        auto solver = std::make_shared<LinearTridiagonalSolver>();
        auto linearizer = std::make_shared<NewtonRaphson>(1e-6, 1, false);
        auto pm = std::make_shared<SerialParallelManager>();

        engine = std::make_unique<SimulationEngine>(spatial, mdl, discretizer, integrator, linearizer, solver, pm);
    }

    void set_initial_condition(const std::vector<double>& u, const std::vector<double>& v) {
        st->u = u;
        st->v = v;
    }

    void step(double dt) {
        auto next_state = engine->step(t, dt, *st, ++step_count);
        st = std::shared_ptr<Wave1DState>(static_cast<Wave1DState*>(next_state.release()));
        t += dt;
    }

    std::vector<double> get_values() const {
        return st->to_vector();
    }
};

/**
 * @brief Numerical Wrapper for Burgers Simulations
 */
class BurgersSimulationWrapper {
private:
    std::shared_ptr<BurgersModel> mdl;
    std::shared_ptr<BurgersState> st;
    std::unique_ptr<SimulationEngine> engine;
    double t = 0.0;
    int step_count = 0;

public:
    BurgersSimulationWrapper(double nu, double dx, const std::vector<double>& ic) {
        int nx = (int)ic.size();
        auto spatial = std::make_shared<Spatial1D>(nx, dx);
        st = std::make_shared<BurgersState>(spatial, 0.0);
        st->u = ic;
        mdl = std::make_shared<BurgersModel>(nu, dx);
        
        auto discretizer = std::make_shared<BurgersDiscretizer>();
        auto integrator = std::make_shared<ImplicitEulerIntegrator>();
        auto solver = std::make_shared<LinearTridiagonalSolver>();
        auto linearizer = std::make_shared<NewtonRaphson>(1e-6, 20, true);
        auto pm = std::make_shared<SerialParallelManager>();

        engine = std::make_unique<SimulationEngine>(spatial, mdl, discretizer, integrator, linearizer, solver, pm);
    }

    void step(double dt) {
        auto next_state = engine->step(t, dt, *st, ++step_count);
        st = std::shared_ptr<BurgersState>(static_cast<BurgersState*>(next_state.release()));
        t += dt;
    }

    std::vector<double> get_values() const {
        return st->to_vector();
    }
};

/**
 * @brief Numerical Wrapper for Stokes Simulations
 */
class StokesSimulationWrapper {
private:
    std::shared_ptr<FluidModel> mdl;
    std::shared_ptr<FluidState> st;
    std::unique_ptr<SimulationEngine> engine;
    double t = 0.0;
    int step_count = 0;

public:
    StokesSimulationWrapper(int nx, int ny, double Lx, double Ly, double mu) {
        auto mesh = std::make_shared<num::fem::Mesh>();
        mesh->generate_quad_mesh(nx, ny, Lx, Ly);
        st = std::make_shared<FluidState>(mesh);
        mdl = std::make_shared<FluidModel>(mesh, mu, 1.0);
        
        auto discretizer = std::make_shared<FluidDiscretizer>();
        auto integrator = std::make_shared<ImplicitEulerIntegrator>();
        auto solver = std::make_shared<BiCGSTABSolver>();
        auto linearizer = std::make_shared<NewtonRaphson>(1e-6, 1, false);
        auto pm = std::make_shared<SerialParallelManager>();

        engine = std::make_unique<SimulationEngine>(nullptr, mdl, discretizer, integrator, linearizer, solver, pm);
    }

    void set_boundary_condition(int node, double u, double v) {
        mdl->set_velocity_bc(node, u, v);
    }

    void solve() {
        auto next_state = engine->step(t, 1.0, *st, ++step_count);
        st = std::shared_ptr<FluidState>(static_cast<FluidState*>(next_state.release()));
        t += 1.0;
    }

    std::vector<double> get_u() const { return st->u; }
    std::vector<double> get_v() const { return st->v; }
    std::vector<double> get_p() const { return st->p; }
};

/**
 * @brief Numerical Wrapper for Reservoir Simulations
 */
class ReservoirSimulationWrapper {
private:
    std::shared_ptr<Reservoir1DModel> mdl;
    std::shared_ptr<Reservoir1DState> st;
    std::unique_ptr<SimulationEngine> engine;
    double t = 0.0;
    int step_count = 0;

public:
    ReservoirSimulationWrapper(std::shared_ptr<num::discretization::Conductance1D> cond, const Vector& storage) {
        int nx = (int)storage.size();
        auto spatial = std::make_shared<Spatial1D>(nx, 1.0);
        st = std::make_shared<Reservoir1DState>(spatial, 0.0);
        std::vector<std::shared_ptr<ISourceSink>> wells;
        mdl = std::make_shared<Reservoir1DModel>(cond, storage, wells);
        
        auto discretizer = std::make_shared<Reservoir1DDiscretizer>();
        auto integrator = std::make_shared<ImplicitEulerIntegrator>();
        auto solver = std::make_shared<LinearTridiagonalSolver>();
        auto linearizer = std::make_shared<NewtonRaphson>(1e-6, 1, false);
        auto pm = std::make_shared<SerialParallelManager>();

        engine = std::make_unique<SimulationEngine>(spatial, mdl, discretizer, integrator, linearizer, solver, pm);
    }

    void set_initial_condition(const std::vector<double>& ic) {
        st->pressures = ic;
    }

    void step(double dt) {
        auto next_state = engine->step(t, dt, *st, ++step_count);
        st = std::shared_ptr<Reservoir1DState>(static_cast<Reservoir1DState*>(next_state.release()));
        t += dt;
    }

    std::vector<double> get_values() const {
        return st->to_vector();
    }
};

/**
 * @brief Numerical Wrapper for 2D Reservoir Simulations
 */
class Reservoir2DSimulationWrapper {
private:
    std::shared_ptr<Reservoir2DModel> mdl;
    std::shared_ptr<Reservoir2DState> st;
    std::unique_ptr<SimulationEngine> engine;
    double t = 0.0;
    int step_count = 0;

public:
    Reservoir2DSimulationWrapper(int nx, int ny, double Lx, double Ly, double eta) {
        auto spatial = std::make_shared<Spatial2D>(nx, ny, Lx, Ly);
        st = std::make_shared<Reservoir2DState>(spatial, 0.0);
        double dx = Lx / (nx - 1);
        double dy = Ly / (ny - 1);
        auto cond = num::discretization::reservoir_cond_2d(nx, ny, dx, dy, 100.0, 1.0, 1.0, 100.0); // Simplified
        Vector storage(nx * ny, 0.001);

        std::vector<std::shared_ptr<ISourceSink>> wells;
        mdl = std::make_shared<Reservoir2DModel>(cond, storage, wells);
        
        auto discretizer = std::make_shared<Reservoir2DDiscretizer>();
        auto integrator = std::make_shared<ImplicitEulerIntegrator>();
        auto solver = std::make_shared<BiCGSTABSolver>();
        auto linearizer = std::make_shared<NewtonRaphson>(1e-6, 1, false);
        auto pm = std::make_shared<SerialParallelManager>();

        engine = std::make_unique<SimulationEngine>(spatial, mdl, discretizer, integrator, linearizer, solver, pm);
    }

    void set_initial_condition(const std::vector<double>& ic) {
        st->pressures = ic;
    }

    void step(double dt) {
        auto next_state = engine->step(t, dt, *st, ++step_count);
        st = std::shared_ptr<Reservoir2DState>(static_cast<Reservoir2DState*>(next_state.release()));
        t += dt;
    }

    std::vector<double> get_values() const {
        return st->to_vector();
    }
};

PYBIND11_MODULE(axcnt_cpp, m) {
    m.doc() = "NumPhys Core Python Bridge - Numerical Execution Engine";

    py::class_<num::discretization::Conductance1D, std::shared_ptr<num::discretization::Conductance1D>>(m, "Conductance1D")
        .def(py::init<size_t>())
        .def_readwrite("T", &num::discretization::Conductance1D::T);

    py::class_<HeatSimulationWrapper>(m, "Heat1D")
        .def(py::init([](const std::vector<double>& T_cond, const std::vector<double>& storage, double TL, double TR) {
            auto cond = std::make_shared<num::discretization::Conductance1D>(T_cond.size() + 1);
            cond->T = T_cond;
            return new HeatSimulationWrapper(cond, storage, TL, TR);
        }))
        .def("set_initial_condition", &HeatSimulationWrapper::set_initial_condition)
        .def("step", &HeatSimulationWrapper::step)
        .def("get_values", &HeatSimulationWrapper::get_values);

    py::class_<Heat2DSimulationWrapper>(m, "Heat2D")
        .def(py::init<int, int, double, double, double>())
        .def("set_initial_condition", &Heat2DSimulationWrapper::set_initial_condition)
        .def("step", &Heat2DSimulationWrapper::step)
        .def("get_values", &Heat2DSimulationWrapper::get_values);

    py::class_<PressureSimulationWrapper>(m, "Pressure1D")
        .def(py::init([](const std::vector<double>& T_cond, const std::vector<double>& storage, double PL, double PR) {
            auto cond = std::make_shared<num::discretization::Conductance1D>(T_cond.size() + 1);
            cond->T = T_cond;
            return new PressureSimulationWrapper(cond, storage, PL, PR);
        }))
        .def("set_initial_condition", &PressureSimulationWrapper::set_initial_condition)
        .def("step", &PressureSimulationWrapper::step)
        .def("get_values", &PressureSimulationWrapper::get_values);

    py::class_<WaveSimulationWrapper>(m, "Wave1D")
        .def(py::init([](const std::vector<double>& T_cond, const std::vector<double>& storage) {
            auto cond = std::make_shared<num::discretization::Conductance1D>(T_cond.size() + 1);
            cond->T = T_cond;
            return new WaveSimulationWrapper(cond, storage);
        }))
        .def("set_initial_condition", &WaveSimulationWrapper::set_initial_condition)
        .def("step", &WaveSimulationWrapper::step)
        .def("get_values", &WaveSimulationWrapper::get_values);

    py::class_<BurgersSimulationWrapper>(m, "Burgers1D")
        .def(py::init<double, double, const std::vector<double>&>())
        .def("step", &BurgersSimulationWrapper::step)
        .def("get_values", &BurgersSimulationWrapper::get_values);

    py::class_<StokesSimulationWrapper>(m, "Stokes2D")
        .def(py::init<int, int, double, double, double>())
        .def("set_boundary_condition", &StokesSimulationWrapper::set_boundary_condition)
        .def("solve", &StokesSimulationWrapper::solve)
        .def("get_u", &StokesSimulationWrapper::get_u)
        .def("get_v", &StokesSimulationWrapper::get_v)
        .def("get_p", &StokesSimulationWrapper::get_p);

    py::class_<ReservoirSimulationWrapper>(m, "Reservoir1D")
        .def(py::init([](const std::vector<double>& T_cond, const std::vector<double>& storage) {
            auto cond = std::make_shared<num::discretization::Conductance1D>(T_cond.size() + 1);
            cond->T = T_cond;
            return new ReservoirSimulationWrapper(cond, storage);
        }))
        .def("set_initial_condition", &ReservoirSimulationWrapper::set_initial_condition)
        .def("step", &ReservoirSimulationWrapper::step)
        .def("get_values", &ReservoirSimulationWrapper::get_values);

    py::class_<Reservoir2DSimulationWrapper>(m, "Reservoir2D")
        .def(py::init<int, int, double, double, double>())
        .def("set_initial_condition", &Reservoir2DSimulationWrapper::set_initial_condition)
        .def("step", &Reservoir2DSimulationWrapper::step)
        .def("get_values", &Reservoir2DSimulationWrapper::get_values);
}
