#pragma once
/**
 * @file spatial.hpp
 * @brief Geometric Language for Mesh-based Simulations.
 * 
 * Objective:
 * This file provides the standardized data structures for representing the
 * physical domain (1D, 2D, or 3D) and its discrete topology.
 * 
 * Architectural Rationale:
 * Segregating spatial metadata prevents "Geometric Drift" across the library. 
 * By isolating grid logic from both physical governing equations and 
 * solver algorithms, we ensure that domain topology remains an 
 * independent, reusable service.
 * 
 * Strategic Importance:
 * It serves as the common language of indexing. Any component—from a 
 * legacy VTK exporter to a 3D Black Oil model—uses these structures 
 * to understand the spatial context without being tightly coupled 
 * to a specific physics implementation.
 */
#include "interfaces.hpp"

namespace mod {

/**
 * @brief 1D Spatial Grid Metadata.
 */
class Spatial1D : public top::IGrid {
public:
    size_t nx;
    double dx;

    Spatial1D(size_t nx, double dx) : nx(nx), dx(dx) {}
    size_t total_size() const { return nx; }
    size_t get_total_cells() const override { return nx; }
};

/**
 * @brief 2D Spatial Grid Metadata.
 */
class Spatial2D : public top::IGrid {
public:
    size_t nx, ny;
    double dx, dy;

    Spatial2D(size_t nx, size_t ny, double dx, double dy) 
        : nx(nx), ny(ny), dx(dx), dy(dy) {}

    size_t total_size() const { return nx * ny; }
    size_t get_total_cells() const override { return nx * ny; }
    
    size_t idx(size_t i, size_t j) const { 
        return j * nx + i; 
    }
};

/**
 * @brief 3D Spatial Grid Metadata.
 */
class Spatial3D : public top::IGrid {
public:
    size_t nx, ny, nz;
    double dx, dy, dz;

    Spatial3D(size_t nx, size_t ny, size_t nz, double dx, double dy, double dz) 
        : nx(nx), ny(ny), nz(nz), dx(dx), dy(dy), dz(dz) {}

    size_t total_size() const { return nx * ny * nz; }
    size_t get_total_cells() const override { return nx * ny * nz; }

    size_t idx(size_t i, size_t j, size_t k) const { 
        return (k * ny + j) * nx + i; 
    }
};

} // namespace mod
