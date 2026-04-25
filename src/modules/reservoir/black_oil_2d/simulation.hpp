#pragma once
#include "lib/simulation.hpp"
#include "lib/linearizers.hpp"
#include "modules/reservoir/well.hpp"
#include "lib/engine_infra.hpp"
#include "lib/discretization.hpp"
#include "lib/utils/config_reader.hpp"
#include "lib/utils/logger.hpp"
#include "state.hpp"
#include "model.hpp"
#include "lib/solvers.hpp"
#include "lib/integrators.hpp"

namespace mod::reservoir {

class BlackOil2DImplicitSimulation {
public:
    struct BuildResult {
        std::unique_ptr<top::SimulationEngine> engine;
        std::unique_ptr<top::IState> st_init;
        std::shared_ptr<utl::StandardLogger> logger;
    };

    static BuildResult build(const utl::ConfigReader& config) {
        size_t nx = config.get("nx", 30);
        size_t ny = config.get("ny", 30);
        double dx = config.get("dx", 100.0);
        double dy = config.get("dy", 100.0);
        double k = config.get("k", 100.0);
        double phi = config.get("phi", 0.2);
        double area = config.get("area", 1.0);
        
        // 1. Grid and State
        auto spatial = std::make_shared<Spatial2D>(nx, ny, dx, dy);
        auto st = std::make_unique<ReservoirBlackOil2DState>(spatial);
        
        // 2. Physics Model and Discretization
        auto cond = num::discretization::pressure_cond_2d(nx, ny, dx, dy, k, 1.0, area); // mu=1 stub
        double pv = dx * dy * area * phi;
        
        // Wells
        std::vector<std::shared_ptr<ISourceSink>> wells;
        auto idx_func = [nx, ny](int i, int j) { return j * (int)nx + i; };
        auto var_func = [](const top::IState& s, int i, int j, double& p, double& sw, double& sg) {
            const auto& bo_s = static_cast<const ReservoirBlackOil2DState&>(s);
            int idx = bo_s.spatial->idx(i, j);
            p = bo_s.p(idx);
            sw = bo_s.sw(idx);
            sg = bo_s.sg(idx);
        };

        // Injector at corner (Bottom-Left)
        wells.push_back(std::make_shared<mod::ReservoirWellBlackOil2D>(
            0, 0, std::abs(config.get("q_injector", 1000.0)), true,
            [](double sw, double sg, double& krw, double& kro, double& krg) {
                krw = sw*sw; krg = sg*sg; kro = (1-sw-sg)*(1-sw-sg);
            },
            idx_func, var_func, 1.0, 2.0, 0.02
        ));

        auto mdl = std::make_shared<BlackOil2DModel>(cond, pv, wells);
        auto discretizer = std::make_shared<BlackOil2DDiscretizer>();
        
        // 3. Engine Components
        auto timer = std::make_shared<num::ImplicitEulerIntegrator>();
        auto linearizer = std::make_shared<num::NewtonRaphson>(1e-4, 15, true);
        
        // 3-Phase 2D system is 3*N, BiCGSTAB is mandatory
        auto solver = std::make_shared<num::BiCGSTABSolver>();
        solver->verbose = false;
        
        auto pm = std::make_shared<top::SerialParallelManager>();

        auto engine = std::make_unique<top::SimulationEngine>(spatial, mdl, discretizer, timer, linearizer, solver, pm);

        // 4. Logger / Observer setup
        auto logger = std::make_shared<utl::StandardLogger>(config);
        logger->set_grid(nx, ny, 1, dx);
        logger->add_field("Pressure", [](const top::IState& s) {
            auto v = s.to_vector();
            Vector p(v.size() / 3);
            for (size_t i = 0; i < p.size(); ++i) p[i] = v[3 * i];
            return p;
        });
        logger->add_field("Sw", [](const top::IState& s) {
            auto v = s.to_vector();
            Vector sw(v.size() / 3);
            for (size_t i = 0; i < sw.size(); ++i) sw[i] = v[3 * i + 1];
            return sw;
        });
        
        engine->add_observer(logger);

        return { std::move(engine), std::move(st), logger };
    }
};

} // namespace mod::reservoir
