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

class BlackOil3DImplicitSimulation {
public:
    struct BuildResult {
        std::unique_ptr<top::SimulationEngine> engine;
        std::unique_ptr<top::IState> st_init;
        std::shared_ptr<utl::StandardLogger> logger;
    };

    static BuildResult build(const utl::ConfigReader& config) {
        size_t nx = config.get("nx", 15);
        size_t ny = config.get("ny", 15);
        size_t nz = config.get("nz", 5);
        double dx = config.get("dx", 100.0);
        double dy = config.get("dy", 100.0);
        double dz = config.get("dz", 20.0);
        double k = config.get("k", 100.0);
        double phi = config.get("phi", 0.2);
        double area = config.get("area", 1.0);
        
        // 1. Grid and State
        auto spatial = std::make_shared<Spatial3D>(nx, ny, nz, dx, dy, dz);
        auto st = std::make_unique<ReservoirBlackOil3DState>(spatial);
        
        // 2. Physics Model and Discretization
        auto cond = num::discretization::pressure_cond_3d(nx, ny, nz, dx, dy, dz, k, area);
        double pv = dx * dy * dz * area * phi;
        
        // Wells
        std::vector<std::shared_ptr<ISourceSink>> wells;
        auto idx_func = [nx, ny, nz](int i, int j, int k) { return (int)((k * ny + j) * nx + i); };
        auto var_func = [](const top::IState& s, int i, int j, int k, double& p, double& sw, double& sg) {
            const auto& bo_s = static_cast<const ReservoirBlackOil3DState&>(s);
            int idx = bo_s.spatial->idx(i, j, k);
            p = bo_s.p(idx);
            sw = bo_s.sw(idx);
            sg = bo_s.sg(idx);
        };

        // Vertical injector at corner (Bottom to Top)
        wells.push_back(std::make_shared<mod::ReservoirWellBlackOil3D>(
            0, 0, 0, (int)nz-1, std::abs(config.get("q_injector", 2000.0)), true,
            [](double sw, double sg, double& krw, double& kro, double& krg) {
                krw=sw*sw; krg=sg*sg; kro=(1-sw-sg)*(1-sw-sg);
            },
            idx_func, var_func, 1.0, 2.0, 0.02
        ));

        auto mdl = std::make_shared<BlackOil3DModel>(cond, pv, wells);
        auto discretizer = std::make_shared<BlackOil3DDiscretizer>();
        
        // 3. Engine Components
        auto timer = std::make_shared<num::ImplicitEulerIntegrator>();
        auto linearizer = std::make_shared<num::NewtonRaphson>(1e-4, 15, true);
        
        // Large 3D Block System, BiCGSTAB is required
        auto solver = std::make_shared<num::BiCGSTABSolver>();
        solver->verbose = false;
        
        auto pm = std::make_shared<top::SerialParallelManager>();

        auto engine = std::make_unique<top::SimulationEngine>(spatial, mdl, discretizer, timer, linearizer, solver, pm);

        // 4. Logger / Observer setup
        auto logger = std::make_shared<utl::StandardLogger>(config);
        logger->set_grid(nx, ny, nz, dx);
        logger->add_field("Pressure", [](const top::IState& s) {
            auto v = s.to_vector();
            Vector p(v.size() / 3);
            for (size_t i = 0; i < p.size(); ++i) p[i] = v[3 * i];
            return p;
        });
        
        engine->add_observer(logger);

        return { std::move(engine), std::move(st), logger };
    }
};

} // namespace mod::reservoir
