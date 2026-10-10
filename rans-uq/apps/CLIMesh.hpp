#pragma once
#include "Mesh.hpp"

namespace rans_cli {
// Keep the demonstration geometry in one place so its physical patch layout
// can be verified independently of the expensive nonlinear solve.
// Why not makePlate2D: its single-wall stretching would also change the grid.
// Retag this existing grid to isolate the corrected upper-boundary semantics.
inline Mesh flatPlateMesh(int nx, int ny, double length, double height, double reynolds) {
    Mesh mesh = Mesh::makeChannel2D(nx, ny, length, height, reynolds);
    mesh.setPatchType("top", "freestream");
    return mesh;
}
} // namespace rans_cli
