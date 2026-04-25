/**
 * @file fem.hpp
 * @brief Finite Element Method (FEM) Core Infrastructure.
 * 
 * Objective:
 * This file provides the standardized data structures for representing
 * unstructured meshes and performing element-level numerical integrations.
 * 
 * Strategic Importance:
 * While FDM and FVM are excellent for structured grids common in reservoirs,
 * FEM is the architecture of choice for complex geometries and vector-field 
 * physics like Fluid Dynamics. This file enables the project to handle 
 * unstructured domains.
 */
#pragma once
#include <vector>
#include <cmath>
#include <array>
#include "lib/sparse.hpp"

namespace mod {

/**
 * @brief 2D Point/Node in space.
 */
struct Node {
    double x, y;
};

/**
 * @brief Linear Triangular Element (3 nodes).
 */
struct Element {
    std::array<int, 3> nodes;
};

/**
 * @brief Unstructured 2D Mesh.
 */
struct Mesh : public top::IGrid {
    std::vector<Node> nodes;
    std::vector<Element> elements;
    
    size_t get_total_cells() const override { return nodes.size(); }

    static Mesh generate_quad_mesh(double L, double H, int nx, int ny) {
        Mesh mesh;
        double dx = L / (nx - 1);
        double dy = H / (ny - 1);
        
        // 1. Nodes
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                mesh.nodes.push_back({ i * dx, j * dy });
            }
        }
        
        // 2. Elements (Divide each quad into 2 triangles)
        for (int j = 0; j < ny - 1; ++j) {
            for (int i = 0; i < nx - 1; ++i) {
                int n0 = j * nx + i;
                int n1 = n0 + 1;
                int n2 = (j + 1) * nx + i;
                int n3 = n2 + 1;
                
                // Triangle 1
                mesh.elements.push_back({ {n0, n1, n3} });
                // Triangle 2
                mesh.elements.push_back({ {n0, n3, n2} });
            }
        }
        return mesh;
    }

    size_t num_nodes() const { return nodes.size(); }
    size_t num_elements() const { return elements.size(); }
};

/**
 * @brief Mathematical Toolbox for Linear Triangle Elements (P1).
 */
class LinearTriangle {
public:
    /**
     * @brief Computes the area and gradients of the shape functions.
     * 
     * Triangle defined by (x1, y1), (x2, y2), (x3, y3).
     * Shape functions N_i are linear. Grad(N_i) is constant over the element.
     */
    struct ElementData {
        double area;
        std::array<double, 3> dNdx;
        std::array<double, 3> dNdy;
    };

    static ElementData compute_data(const Node& n1, const Node& n2, const Node& n3) {
        double x1 = n1.x, y1 = n1.y;
        double x2 = n2.x, y2 = n2.y;
        double x3 = n3.x, y3 = n3.y;

        // Jacobian Matrix coefficients
        double b1 = y2 - y3;
        double b2 = y3 - y1;
        double b3 = y1 - y2;

        double c1 = x3 - x2;
        double c2 = x1 - x3;
        double c3 = x2 - x1;

        double detJ = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2);
        double area = 0.5 * std::abs(detJ);

        ElementData data;
        data.area = area;
        double inv_detJ = 1.0 / detJ;

        data.dNdx = { b1 * inv_detJ, b2 * inv_detJ, b3 * inv_detJ };
        data.dNdy = { c1 * inv_detJ, c2 * inv_detJ, c3 * inv_detJ };

        return data;
    }

    /**
     * @brief Build the local Stiffness Matrix for the Poisson operator: Integral( Grad(Ni) . Grad(Nj) )
     */
    static std::array<std::array<double, 3>, 3> build_poisson_ke(const ElementData& data, double conductivity) {
        std::array<std::array<double, 3>, 3> ke;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                ke[i][j] = conductivity * (data.dNdx[i] * data.dNdx[j] + data.dNdy[i] * data.dNdy[j]) * data.area;
            }
        }
        return ke;
    }
};

/**
 * @brief Global Assembly Helper.
 */
class Assembler {
public:
    /**
     * @brief Assemblies local element matrices into global sparse triplets.
     */
    static void add_to_triplets(std::vector<num::SparseMatrix::Entry>& triplets, 
                               const std::array<std::array<double, 3>, 3>& ke, 
                               const std::array<int, 3>& nodal_indices) {
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                triplets.push_back({ nodal_indices[i], nodal_indices[j], ke[i][j] });
            }
        }
    }
};

} // namespace mod
