/**
 * @file interfaces.hpp
 * @brief Centralized Component-Based Architecture for Numerical Simulations.
 */
#pragma once
#include <vector>
#include <memory>
#include <string>
#include <functional>
#include "lib/sparse.hpp"

namespace top {

// Global types for consistency
using Vector = std::vector<double>;
using Matrix = std::vector<std::vector<double>>;
using SparseMatrix = num::SparseMatrix;

/**
 * @brief Handles topology and geometry.
 * Who is next to whom, and how big are they?
 */
class IGrid {
public:
    virtual ~IGrid() = default;
    virtual size_t get_total_cells() const = 0;
};

/**
 * @brief Handles variable memory and physical bounds.
 */
class IState {
public:
    virtual ~IState() = default;
    virtual void update(const std::vector<double>& delta) = 0;
    virtual std::vector<double> to_vector() const = 0;
    virtual std::unique_ptr<IState> clone() const = 0;
};

/**
 * @brief Generic property lookups (PVT, Rock, Thermal, etc).
 */
class IPropertyManager {
public:
    virtual ~IPropertyManager() = default;
};

/**
 * @brief Translates continuous physics into discrete terms.
 */
class IModel {
public:
    virtual ~IModel() = default;
    virtual double get_tolerance() const = 0;
    
    /**
     * @brief Returns the diagonal weights for the accumulation term (e.g., C in C*du/dt).
     */
    virtual Vector get_accumulation_weights(const IGrid& grd, const IState& st) const = 0;
};

/**
 * @brief Numerical Mapping: Translates Grid + Model into Sparse Matrices.
 * Covers FDM, FVM, and FEM assembly logic.
 */
class IDiscretizer {
public:
    virtual ~IDiscretizer() = default;
    virtual void build_jacobian(const IGrid& grd, const IModel& mdl, const IState& st, SparseMatrix& J) const = 0;
    virtual void build_residual(const IGrid& grd, const IModel& mdl, const IState& st, Vector& R) const = 0;
    virtual void apply_bc(const IGrid& grd, const IModel& mdl, const IState& st, SparseMatrix& J, Vector& R) const = 0;
};

/**
 * @brief External Source/Sink or Inner Boundary Terms.
 */
class ISourceSink {
public:
    virtual ~ISourceSink() = default;
    virtual void assemble_terms(const IState& st, SparseMatrix& J, Vector& R) const = 0;
};

/**
 * @brief Temporal Strategy: Manages accumulation terms (du/dt).
 */
class ITimeIntegrator {
public:
    virtual ~ITimeIntegrator() = default;
    virtual double compute_dt(const IState& st, double t) const = 0;
    virtual void apply_temporal(const IGrid& grd, const IModel& mdl, SparseMatrix& J, Vector& R, const IState& st_new, const IState& st_old, double dt) const = 0;
};

/**
 * @brief Pure Algebraic Solver (Ax = b).
 */
class ISolver {
public:
    virtual ~ISolver() = default;
    virtual Vector solve(const SparseMatrix& A, const Vector& b) = 0;
};

/**
 * @brief Parallel Orchestrator (MPI/Halo-sync and Global Reductions).
 */
class IParallelManager {
public:
    virtual ~IParallelManager() = default;
    virtual void sync_ghost_cells(IState& st) const = 0;
    virtual double get_global_norm(const Vector& r) const = 0;
};

/**
 * @brief Non-linear Orchestrator (Orchestrates assembly and solve steps).
 */
class ILinearizer {
public:
    virtual ~ILinearizer() = default;
    
    virtual void set_sources(const std::vector<std::shared_ptr<ISourceSink>>& sources) = 0;

    virtual std::unique_ptr<IState> resolve(
        const IState& st_n, double dt, 
        const IGrid& grd, const IModel& mdl, const IDiscretizer& discretizer,
        const ITimeIntegrator& timer, ISolver& solver, const IParallelManager& pm) = 0;
};

/**
 * @brief Telemetry and Output (Handles logging, VTK/CSV exporting, etc).
 */
class IObserver {
public:
    virtual ~IObserver() = default;
    virtual void on_simulation_start(const IGrid& grd) {}
    virtual void on_step_complete(double t, int step, const IState& st) = 0;
    virtual void on_simulation_end() {}
};

} // namespace top
