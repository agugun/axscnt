/**
 * @file io.hpp
 * @brief Data Persistence & Visualization Layer.
 * 
 * Objective:
 * This file handles the output of simulation results into persistent formats 
 * compatible with external visualization tools (e.g., ParaView).
 * 
 * Architectural Rationale:
 * Segregating the IO layer enforces the "Single Responsibility Principle." By 
 * isolating data persistence from mathematical computation, we ensure that 
 * the core simulation "hot-path" remains clean of filesystem and serialization 
 * logic.
 * 
 * Strategic Importance:
 * It protects the simulation core from "Library Bloat." It ensures that 
 * changing or adding output formats (e.g., from Legacy VTK to modern HDF5) 
 * requires zero changes to the underlying physics or numerical code.
 */
#pragma once
#include <string>
#include <vector>
#include <fstream>
#include "lib/interfaces.hpp"

namespace utl {

struct VTKField {
    std::string name;
    std::vector<double> data;
};

class VTKExporter {
public:
    /**
     * @brief Exports a 3D scalar field to a Legacy VTK file.
     */
    static void export_structured_3d(const std::string& filename, 
                                     const std::vector<double>& data,
                                     int nx, int ny, int nz,
                                     double dx, double dy, double dz,
                                     const std::string& scalar_name = "Scalar") {
        std::ofstream file(filename);
        if (!file.is_open()) return;

        file << "# vtk DataFile Version 3.0\n";
        file << "NumPhys 3D Export\n";
        file << "ASCII\n";
        file << "DATASET STRUCTURED_POINTS\n";
        file << "DIMENSIONS " << nx << " " << ny << " " << nz << "\n";
        file << "ORIGIN 0 0 0\n";
        file << "SPACING " << dx << " " << dy << " " << dz << "\n";
        file << "POINT_DATA " << nx * ny * nz << "\n";
        file << "SCALARS " << scalar_name << " double\n";
        file << "LOOKUP_TABLE default\n";

        for (const auto& val : data) {
            file << val << "\n";
        }

        file.close();
    }

    /**
     * @brief Exports a 2D scalar field to a Legacy VTK file (as a Z=1 plane).
     */
    static void export_structured_2d(const std::string& filename,
                                     const std::vector<double>& data,
                                     int nx, int ny,
                                     double dx, double dy,
                                     const std::string& scalar_name = "Scalar") {
        export_structured_3d(filename, data, nx, ny, 1, dx, dy, 1.0, scalar_name);
    }

    /**
     * @brief Exports a 3D scalar field to a VTK XML Image Data (.vti) file.
     */
    static void export_vti_3d(const std::string& filename, 
                             const std::vector<double>& data,
                             int nx, int ny, int nz,
                             double dx, double dy, double dz,
                             const std::string& scalar_name = "Scalar") {
        std::ofstream file(filename);
        if (!file.is_open()) return;

        file << "<?xml version=\"1.0\"?>\n";
        file << "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
        file << "  <ImageData WholeExtent=\"0 " << nx-1 << " 0 " << ny-1 << " 0 " << nz-1 
             << "\" Origin=\"0 0 0\" Spacing=\"" << dx << " " << dy << " " << dz << "\">\n";
        file << "    <Piece Extent=\"0 " << nx-1 << " 0 " << ny-1 << " 0 " << nz-1 << "\">\n";
        file << "      <PointData Scalars=\"" << scalar_name << "\">\n";
        file << "        <DataArray type=\"Float64\" Name=\"" << scalar_name << "\" format=\"ascii\">\n";

        for (const auto& val : data) {
            file << val << " ";
        }

        file << "\n        </DataArray>\n";
        file << "      </PointData>\n";
        file << "    </Piece>\n";
        file << "  </ImageData>\n";
        file << "</VTKFile>\n";

        file.close();
    }

    /**
     * @brief Exports a 2D scalar field to a VTK XML Image Data (.vti) file.
     */
    static void export_vti_2d(const std::string& filename,
                             const std::vector<double>& data,
                             int nx, int ny,
                             double dx, double dy,
                             const std::string& scalar_name = "Scalar") {
        export_vti_3d(filename, data, nx, ny, 1, dx, dy, 1.0, scalar_name);
    }

    /**
     * @brief Exports a 1D scalar field to a VTK XML Image Data (.vti) file.
     */
    static void export_vti_1d(const std::string& filename,
                             const std::vector<double>& data,
                             int nx, double dx,
                             const std::string& scalar_name = "Scalar") {
        export_vti_3d(filename, data, nx, 1, 1, dx, 1.0, 1.0, scalar_name);
    }
    /**
     * @brief Exports multiple scalar fields to a VTK XML Image Data (.vti) file.
     */
    static void export_vti_multi_3d(const std::string& filename,
                                    int nx, int ny, int nz,
                                    double dx, double dy, double dz,
                                    const std::vector<VTKField>& fields) {
        if (fields.empty()) return;
        std::ofstream file(filename);
        if (!file.is_open()) return;

        file << "<?xml version=\"1.0\"?>\n";
        file << "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
        file << "  <ImageData WholeExtent=\"0 " << nx-1 << " 0 " << ny-1 << " 0 " << nz-1 
             << "\" Origin=\"0 0 0\" Spacing=\"" << dx << " " << dy << " " << dz << "\">\n";
        file << "    <Piece Extent=\"0 " << nx-1 << " 0 " << ny-1 << " 0 " << nz-1 << "\">\n";
        file << "      <PointData Scalars=\"" << fields[0].name << "\">\n";

        for (const auto& field : fields) {
            file << "        <DataArray type=\"Float64\" Name=\"" << field.name << "\" format=\"ascii\">\n";
            for (const auto& val : field.data) {
                file << val << " ";
            }
            file << "\n        </DataArray>\n";
        }

        file << "      </PointData>\n";
        file << "    </Piece>\n";
        file << "  </ImageData>\n";
        file << "</VTKFile>\n";

        file.close();
    }

    /**
     * @brief Exports multiple 2D scalar fields to a VTK XML Image Data (.vti) file.
     */
    static void export_vti_multi_2d(const std::string& filename,
                                    int nx, int ny,
                                    double dx, double dy,
                                    const std::vector<VTKField>& fields) {
        export_vti_multi_3d(filename, nx, ny, 1, dx, dy, 1.0, fields);
    }

    /**
     * @brief Exports multiple 1D scalar fields to a VTK XML Image Data (.vti) file.
     */
    static void export_vti_multi_1d(const std::string& filename,
                                    int nx, double dx,
                                    const std::vector<VTKField>& fields) {
        export_vti_multi_3d(filename, nx, 1, 1, dx, 1.0, 1.0, fields);
    }
};

/**
 * @brief High-performance Binary and JSON Exporter for scientific exercising.
 */
class BinaryExporter {
public:
    static void export_snapshot(const std::string& base_filename,
                                int nx, int ny, int nz,
                                double dx, double dy, double dz,
                                const std::vector<VTKField>& fields) {
        // 1. Export JSON Metadata
        std::string json_file = base_filename + ".json";
        std::ofstream jf(json_file);
        if (jf.is_open()) {
            jf << "{\n";
            jf << "  \"grid\": { \"nx\": " << nx << ", \"ny\": " << ny << ", \"nz\": " << nz << " },\n";
            jf << "  \"spacing\": { \"dx\": " << dx << ", \"dy\": " << dy << ", \"dz\": " << dz << " },\n";
            jf << "  \"fields\": [\n";
            for (size_t i = 0; i < fields.size(); ++i) {
                jf << "    \"" << fields[i].name << "\"" << (i == fields.size() - 1 ? "" : ",") << "\n";
            }
            jf << "  ]\n";
            jf << "}\n";
            jf.close();
        }

        // 2. Export Raw Binary Data (all fields concatenated)
        std::string bin_file = base_filename + ".bin";
        std::ofstream bf(bin_file, std::ios::binary);
        if (bf.is_open()) {
            for (const auto& field : fields) {
                bf.write(reinterpret_cast<const char*>(field.data.data()), 
                         field.data.size() * sizeof(double));
            }
            bf.close();
        }
    }
};

/**
 * @brief Standard CSV Exporter for tabulated analysis.
 */
class CSVExporter {
public:
    static void export_snapshot(const std::string& filename,
                                int nx, int ny, int nz,
                                double dx, double dy, double dz,
                                const std::vector<VTKField>& fields) {
        std::ofstream f(filename);
        if (!f.is_open()) return;

        // Headers
        f << "i,j,k,x,y,z";
        for (const auto& field : fields) f << "," << field.name;
        f << "\n";

        // Body
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    int c = k * (nx * ny) + j * nx + i;
                    f << i << "," << j << "," << k << ","
                      << i * dx << "," << j * dy << "," << k * dz;
                    for (const auto& field : fields) f << "," << field.data[c];
                    f << "\n";
                }
            }
        }
        f.close();
    }
};

} // namespace utl
