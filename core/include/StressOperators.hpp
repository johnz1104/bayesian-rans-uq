#pragma once

#include "Mesh.hpp"
#include "Field.hpp"
#include <vector>

// Explicit stress-divergence pieces the componentwise implicit diffusion
// operator div(coeff grad U_i) does not represent. The full (Favre/Boussinesq)
// deviatoric stress divergence is
//   div(coeff (grad U + grad U^T)) - (2/3) grad(coeff div U)   [+ -(2/3) grad(rho k)]
// and the implicit operator supplies only the first (untransposed) half, so
// the remainder enters the momentum source as a deferred correction evaluated
// at the current iterate.

// Transpose-stress source, div(coeff (grad U)^T), per cell (NOT divided by
// volume). Face-based and dimension-general:
//   source_i[c] = sum_f coeff_f (dU_j/dx_i)_f n_j A_f   (outward n per cell)
// with cell Green-Gauss gradients linearly interpolated to faces and owner
// extrapolation on boundary faces (at a resolved wall the coefficient, an
// eddy viscosity, vanishes, so the boundary flux is negligible). The term is
// identically zero in parallel shear flows (fully-developed channel/Couette)
// and for constant coefficient with solenoidal U; it acts wherever the
// coefficient varies along the flow (separated shear layers).
// wallCoeffOwner (optional, indexed by owner cell): coefficient to use at
// WALL faces instead of the owner-extrapolated interior coefficient. The
// physically correct wall value differs from the owner cell's: an eddy
// viscosity vanishes AT the wall even though the owner-cell value does not
// (incompressible passes zeros), and a compressible muEff keeps only its
// molecular part there (turbulent part zero). Non-wall boundary faces keep
// owner extrapolation.
inline std::vector<double>
transposeStressSource(const Mesh &mesh, const ScalarField &coeff, const VectorField &U,
                      int component, const std::vector<double> *wallCoeffOwner = nullptr) {
    VelocityGradients vg = computeVelocityGradients(U);
    // column `component` of grad U at a cell:
    //   (dU_x/dx_i, dU_y/dx_i, dU_z/dx_i) for i = component
    auto gcol = [&](int ci) -> Vec3 {
        if (component == 0)
            return Vec3(vg.dudx[ci].x, vg.dvdx[ci].x, vg.dwdx[ci].x);
        if (component == 1)
            return Vec3(vg.dudx[ci].y, vg.dvdx[ci].y, vg.dwdx[ci].y);
        return Vec3(vg.dudx[ci].z, vg.dvdx[ci].z, vg.dwdx[ci].z);
    };

    std::vector<double> src(mesh.nCells(), 0.0);
    int nIF = mesh.nInternalFaces();
    for (int fi = 0; fi < nIF; ++fi) {
        const Face &face = mesh.face(fi);
        int o = face.owner, n = face.neighbor;
        double w = face.weight;
        double cf = w * coeff[o] + (1.0 - w) * coeff[n];
        Vec3 g = gcol(o) * w + gcol(n) * (1.0 - w);
        double flux = cf * g.dot(face.normal) * face.area;
        src[o] += flux; // face normal is outward for the owner
        src[n] -= flux; // and inward for the neighbor
    }
    if (wallCoeffOwner == nullptr) {
        for (int fi = nIF; fi < mesh.nFaces(); ++fi) {
            const Face &face = mesh.face(fi);
            int o = face.owner;
            double flux = coeff[o] * gcol(o).dot(face.normal) * face.area;
            src[o] += flux;
        }
        return src;
    }
    // wall-aware boundary treatment: iterate patches so wall faces get the
    // supplied wall coefficient and every other boundary keeps owner
    // extrapolation
    for (int pi = 0; pi < mesh.nPatches(); ++pi) {
        const Patch &pat = mesh.patch(pi);
        const bool isWall = (pat.type == "wall");
        for (FaceID fi : pat.faces) {
            const Face &face = mesh.face(fi);
            int o = face.owner;
            double cf = isWall ? (*wallCoeffOwner)[o] : coeff[o];
            double flux = cf * gcol(o).dot(face.normal) * face.area;
            src[o] += flux;
        }
    }
    return src;
}
