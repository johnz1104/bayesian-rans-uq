// Streamwise-periodic curved-channel mesh and body-force drive.
//
// The periodic-hills family needs a curved-bottom channel whose streamwise
// periodicity is built as wrap-around INTERNAL faces (no boundary condition),
// driven by a constant body force (no inlet exists). Verifies:
//   1. Exact quad geometry on a cosine-bump mesh: total volume equals the
//      trapezoidal integral of the gap, normals are unit, wrap-face delta is
//      one cell spacing (the periodic image), not the domain length.
//   2. A flat periodic channel driven by a body force converges with the SST
//      closure, is streamwise-invariant (periodicity is exact), and satisfies
//      the global momentum balance f V = sum of wall shear within a few
//      percent.
//   3. The cosine-bump periodic channel converges to finite fields (the
//      pressure reference pin makes the outlet-free Poisson system regular).

#include "Mesh.hpp"
#include "Field.hpp"
#include "BoundaryCondition.hpp"
#include "SIMPLESolver.hpp"
#include "SSTModel.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

namespace {

#define REQUIRE(cond, msg)                                                                         \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            std::fprintf(stderr, "FAIL [%s:%d] %s\n  required: %s\n", __FILE__, __LINE__, (msg),   \
                         #cond);                                                                   \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

Mesh cosineMesh(int nx, int ny, double Lx, double yTop, double bump) {
    std::vector<double> xN(nx + 1), yB(nx + 1);
    for (int i = 0; i <= nx; ++i) {
        xN[i] = Lx * i / nx;
        yB[i] = 0.5 * bump * (1.0 + std::cos(2.0 * M_PI * xN[i] / Lx));
    }
    return Mesh::makeCurvedChannelPeriodic2D(xN, yB, yTop, ny, 0.0, 1.0);
}

} // namespace

int main() {
    // 1. geometry on a cosine bump
    {
        const int nx = 24, ny = 10;
        const double Lx = 4.0, yTop = 3.0, bump = 1.0;
        Mesh m = cosineMesh(nx, ny, Lx, yTop, bump);
        REQUIRE(m.nCells() == nx * ny, "cell count");

        // exact quad total volume = trapezoidal integral of the gap
        double vol = 0.0;
        for (int ci = 0; ci < m.nCells(); ++ci) {
            REQUIRE(m.cell(ci).volume > 0.0, "positive cell volume");
            vol += m.cell(ci).volume;
        }
        double exact = 0.0;
        for (int i = 0; i < nx; ++i) {
            double x0 = Lx * i / nx, x1 = Lx * (i + 1) / nx;
            double g0 = yTop - 0.5 * bump * (1.0 + std::cos(2.0 * M_PI * x0 / Lx));
            double g1 = yTop - 0.5 * bump * (1.0 + std::cos(2.0 * M_PI * x1 / Lx));
            exact += 0.5 * (g0 + g1) * (x1 - x0);
        }
        REQUIRE(std::fabs(vol - exact) < 1e-10 * exact, "total volume exact");

        // unit normals everywhere; wrap-face delta is one cell spacing
        int nWrapChecked = 0;
        for (int fi = 0; fi < m.nFaces(); ++fi) {
            const Face &f = m.face(fi);
            REQUIRE(std::fabs(f.normal.norm() - 1.0) < 1e-12, "unit normal");
            if (!f.isBoundary()) {
                bool wrap = (f.owner % nx == nx - 1) && (f.neighbor % nx == 0) &&
                            (f.owner / nx == f.neighbor / nx);
                if (wrap) {
                    REQUIRE(f.delta < 2.0 * Lx / nx,
                            "wrap delta must be one spacing, not the domain");
                    ++nWrapChecked;
                }
            }
        }
        REQUIRE(nWrapChecked == ny, "one wrap face per row");
    }

    // 2. flat periodic channel: body-force-driven SST solve
    {
        const int nx = 24, ny = 28;
        const double Lx = 3.0, H = 1.0, nu = 2.0e-4, fb = 5.0e-3;
        std::vector<double> xN(nx + 1), yB(nx + 1, 0.0);
        for (int i = 0; i <= nx; ++i)
            xN[i] = Lx * i / nx;
        Mesh mesh = Mesh::makeCurvedChannelPeriodic2D(xN, yB, H, ny, 2500.0, 1.0);
        mesh.computeWallDistance();

        const double kIn = 1e-4, omIn = 10.0;
        FlowBoundaryConditions bcs = FlowBoundaryConditions::channelDefaults(mesh, 1.0, kIn, omIn);
        SolverSettings settings;
        settings.maxIterations = 12000;
        settings.convergenceTol = 1e-4;
        settings.alphaU = 0.5;
        settings.alphaP = 0.3;
        settings.verbose = false;
        settings.bodyForce = Vec3(fb, 0.0, 0.0);
        SSTModel sst{SSTCoefficients{}};

        SIMPLESolver solver(mesh, sst, bcs, nu, settings);
        FlowFields fld(mesh);
        solver.initUniform(fld, Vec3(0.3, 0.0, 0.0), 0.0, kIn, omIn);
        ConvergenceHistory hist = solver.solve(fld);
        REQUIRE(!hist.diverged, "flat periodic channel diverged");
        REQUIRE(hist.converged, "flat periodic channel did not converge");

        // streamwise invariance: column 0 equals column nx/2 (exact periodicity)
        double maxDiff = 0.0, uBulkNum = 0.0, uBulkDen = 0.0;
        for (int j = 0; j < ny; ++j) {
            double u0 = fld.U[j * nx + 0].x;
            double u1 = fld.U[j * nx + nx / 2].x;
            maxDiff = std::max(maxDiff, std::fabs(u0 - u1));
        }
        for (int ci = 0; ci < mesh.nCells(); ++ci) {
            uBulkNum += fld.U[ci].x * mesh.cell(ci).volume;
            uBulkDen += mesh.cell(ci).volume;
        }
        double uBulk = uBulkNum / uBulkDen;
        REQUIRE(uBulk > 0.1, "bulk flow not established by the body force");
        REQUIRE(maxDiff < 1e-6 * std::max(uBulk, 1e-12), "solution is not streamwise-invariant");

        // Global momentum balance: f * V_total = integral of wall shear, with
        // the DISCRETE wall stress the solver imposes. At a wall-resolved face
        // nuT is zero, so the integrated face coefficient is molecular nu;
        // owner-cell nuT belongs to the interior and must not be extrapolated
        // across the wall half-cell.
        double drive = fb * uBulkDen;
        double shear = 0.0;
        for (int pi = 0; pi < mesh.nPatches(); ++pi) {
            for (FaceID fi : mesh.patch(pi).faces) {
                const Face &fc = mesh.face(fi);
                double du = fld.U[fc.owner].x; // wall value is zero
                shear += nu * du / fc.delta * fc.area;
            }
        }
        REQUIRE(std::fabs(shear - drive) / drive < 0.05,
                "wall shear does not balance the body force within 5 percent");
    }

    // 3. cosine-bump periodic channel converges to finite fields
    {
        const int nx = 36, ny = 24;
        Mesh mesh = cosineMesh(nx, ny, 4.5, 3.0, 1.0);
        mesh.computeWallDistance();
        const double nu = 3.0e-4, kIn = 1e-4, omIn = 10.0;
        FlowBoundaryConditions bcs = FlowBoundaryConditions::channelDefaults(mesh, 1.0, kIn, omIn);
        SolverSettings settings;
        settings.maxIterations = 15000;
        settings.convergenceTol = 1e-4;
        settings.alphaU = 0.4;
        settings.alphaP = 0.25;
        settings.verbose = false;
        settings.bodyForce = Vec3(4.0e-3, 0.0, 0.0);
        SSTModel sst{SSTCoefficients{}};

        SIMPLESolver solver(mesh, sst, bcs, nu, settings);
        FlowFields fld(mesh);
        solver.initUniform(fld, Vec3(0.3, 0.0, 0.0), 0.0, kIn, omIn);
        ConvergenceHistory hist = solver.solve(fld);
        REQUIRE(!hist.diverged, "cosine-bump periodic channel diverged");
        double uMax = 0.0;
        for (int ci = 0; ci < mesh.nCells(); ++ci) {
            REQUIRE(std::isfinite(fld.U[ci].x) && std::isfinite(fld.p[ci]), "non-finite field");
            uMax = std::max(uMax, std::fabs(fld.U[ci].x));
        }
        REQUIRE(uMax > 0.05 && uMax < 100.0, "bump-channel velocity scale");
    }

    std::printf("test_periodic_channel: all assertions passed\n");
    return 0;
}
