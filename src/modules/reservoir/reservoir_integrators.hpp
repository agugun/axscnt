#pragma once
#include "lib/interfaces.hpp"
#include "lib/solvers.hpp"
#include "modules/reservoir/well.hpp"
#include <vector>
#include <algorithm>

namespace num {
using namespace top;
using namespace mod;

/**
 * @brief Reservoir IMPES Integrator (Implicit Pressure, Explicit Saturation).
 * 
 * Fits With:
 * - Two-phase or three-phase reservoir flow (e.g., Oil-Water, Black Oil).
 * - Decoupling pressure and saturation to reduce computational cost compared to Fully Implicit.
 * 
 * Best Case:
 * - Systems where pressure changes rapidly but saturation fronts move slowly.
 * - Pressure is solved implicitly (large dt stability), saturation is solved explicitly.
 * 
 * Worst Case:
 * - High flow rates or small grd cells where the explicit saturation update is limited by a strict CFL condition.
 * - Non-convergent pressure solutions if the Jacobian is not well-estimated.
 * 
 * Sample Input:
 * @tparam TState The specialized reservoir st class (e.g., Reservoir2DState).
 * @tparam TModel The specialized reservoir mdl class (e.g., DualPhaseModel).
 * @param solver Must be ConjugateGradientSolver for the implicit pressure stage.
 */
template<typename TState, typename TModel>
class ReservoirIMPESIntegrator : public ITimeIntegrator {
public:
    void step(const IModel& model_raw, IState& state_raw, ISolver* solver, double dt) override {
        auto& st = dynamic_cast<TState&>(state_raw);
        const auto& mdl = dynamic_cast<const TModel&>(model_raw);
        auto* cg_solver = dynamic_cast<ConjugateGradientSolver*>(solver);
        
        if (!cg_solver) throw std::runtime_error("IMPES requires ConjugateGradientSolver.");

        int nx = st.spatial.nx, ny = st.spatial.ny;
        double dx = st.spatial.dx, dy = st.spatial.dy;
        double unit_conv = 0.001127;
        
        // --- 1. SOLVE PRESSURE IMPLICITLY ---
        auto apply_A = [&](const Vector& v) {
            return mdl.apply_jacobian(st, v, dt);
        };
        
        Vector R_raw = mdl.build_residual(st, st, dt);
        Vector R_p(st.pressures.size(), 0.0);
        
        // Sum water and oil equations for pressure residual if coupled
        if (R_raw.size() == 2 * st.pressures.size()) {
            for (size_t i = 0; i < st.pressures.size(); ++i) {
                R_p[i] = R_raw[2 * i] + R_raw[2 * i + 1];
            }
        } else {
            R_p = R_raw;
        }

        Vector delta = cg_solver->solve_iterative(apply_A, scale(R_p, -1.0), Vector(R_p.size(), 0.0));
        st.update(delta);
        
        // --- 2. UPDATE SATURATION EXPLICITLY ---
        Vector sw_new = st.water_saturations;
        double pv = mdl.pore_vol_per_cell; // Now a single pre-calculated value

        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int cur = st.idx(i, j);
                double water_flux_sum = 0.0;
                
                auto add_water_flux = [&](int ni, int nj, double t_rock) {
                    int neighbor = st.idx(ni, nj);
                    double p_cur = st.pressures[cur];
                    double p_neigh = st.pressures[neighbor];
                    
                    double sw_up = (p_cur > p_neigh) ? st.water_saturations[cur] : st.water_saturations[neighbor];
                    double krw, kro;
                    mdl.get_rel_perm(sw_up, krw, kro);
                    
                    double lambda_w = krw / mdl.mu_w;
                    double Tw = t_rock * lambda_w;
                    water_flux_sum += Tw * (p_neigh - p_cur);
                };

                if (i > 0) add_water_flux(i - 1, j, mdl.rock_cond->Tx[j * (nx - 1) + i - 1]);
                if (i < nx - 1) add_water_flux(i + 1, j, mdl.rock_cond->Tx[j * (nx - 1) + i]);
                if (j > 0) add_water_flux(i, j - 1, mdl.rock_cond->Ty[(j - 1) * nx + i]);
                if (j < ny - 1) add_water_flux(i, j + 1, mdl.rock_cond->Ty[j * nx + i]);

                double qw = 0.0;
                for (const auto& s : mdl.wells) {
                    // Try to cast to ReservoirWellDual2D to get water rate
                    auto w = std::dynamic_pointer_cast<mod::ReservoirWellDual2D>(s);
                    if (w && w->i == i && w->j == j) {
                        qw = w->get_q_water(st);
                    }
                }

                double delta_sw = (dt * (water_flux_sum + qw)) / pv;
                sw_new[cur] += delta_sw;
                sw_new[cur] = std::max(0.0, std::min(1.0, sw_new[cur]));
            }
        }
        st.water_saturations = sw_new;
    }

private:
    Vector scale(const Vector& v, double s) {
        Vector res = v;
        for (auto& val : res) val *= s;
        return res;
    }
};

} // namespace num
