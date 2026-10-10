// Rhie-Chow on every mesh, with stored face fluxes and a coefficient that
// does not depend on under-relaxation.
//
// The face-flux dissipation used to be applied only where the mesh has no
// outlet; bounded meshes ran on centrally interpolated fluxes. Where it was
// applied, its coefficient used the under-relaxed momentum diagonal (the
// unrelaxed copy was made and never used), so the converged solution
// depended on alpha_U; and the momentum, k and omega convection recomputed
// face fluxes from interpolated cell velocities, which do not satisfy the
// continuity equation the pressure correction enforces.
//
// This test pins:
//   1. production Couette (bounded mesh with an outlet): the converged
//      solution has no spurious streamwise pressure gradient, so both walls
//      carry the same shear. Without the term on this mesh the pressure
//      settles into a staircase across the four columns whose gradient is
//      a uniform dp/dx of ~1.75e-6, a 0.12% wall-shear asymmetry.
//   2. curved periodic channel: the converged field does not depend on the
//      momentum under-relaxation factor;
//   3. the stored face fluxes satisfy discrete continuity at convergence on a
//      bounded and on a periodic mesh;
//   4. the directional odd-even diagnostic detects a one-directional
//      checkerboard that the two-colour mode misses.

#include "BoundaryCondition.hpp"
#include "Field.hpp"
#include "FlowFields.hpp"
#include "Mesh.hpp"
#include "SIMPLESolver.hpp"
#include "SSTModel.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
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

double wallShear(const Mesh &mesh, const FlowFields &f, double nu, const std::string &patch) {
    const Patch &p = mesh.patch(mesh.patchByName(patch));
    double ta = 0.0, a = 0.0;
    for (FaceID fi : p.faces) {
        const Face &fc = mesh.face(fi);
        double du = f.U[fc.owner].x - f.U.bface(fi).x;
        ta += nu * std::fabs(du) / fc.delta * fc.area;
        a += fc.area;
    }
    return ta / a;
}

// worst cell imbalance of a face-flux vector, relative to the mean |flux|
double continityDefect(const Mesh &mesh, const std::vector<double> &phi) {
    std::vector<double> net(mesh.nCells(), 0.0);
    double sumAbs = 0.0;
    for (int fi = 0; fi < mesh.nFaces(); ++fi) {
        const Face &f = mesh.face(fi);
        net[f.owner] += phi[fi];
        if (!f.isBoundary())
            net[f.neighbor] -= phi[fi];
        sumAbs += std::fabs(phi[fi]);
    }
    double worst = 0.0;
    for (double v : net)
        worst = std::max(worst, std::fabs(v));
    return worst / (sumAbs / mesh.nFaces());
}

Mesh cosineMesh(int nx, int ny) {
    const double Lx = 4.5, yTop = 3.0;
    std::vector<double> xN(nx + 1), yB(nx + 1);
    for (int i = 0; i <= nx; ++i) {
        xN[i] = Lx * i / nx;
        yB[i] = 0.5 * (1.0 + std::cos(2.0 * M_PI * xN[i] / Lx));
    }
    Mesh m = Mesh::makeCurvedChannelPeriodic2D(xN, yB, yTop, ny, 0.0, 1.0);
    m.computeWallDistance();
    return m;
}

} // namespace

int main() {
    // 1. production Couette: no spurious streamwise pressure gradient
    {
        const double nu = 0.0003060182410114865, Uwall = 2.0, Ub = 1.0, tol = 1e-6;
        Mesh mesh = Mesh::makeChannel2D(4, 96, 1.5, 2.0, Uwall / nu, 0.3);
        mesh.computeWallDistance();
        const double kIn = 1.5 * (Ub * 0.05) * (Ub * 0.05), omIn = kIn / (nu * 100.0);
        FlowBoundaryConditions bcs =
            FlowBoundaryConditions::couetteDefaults(mesh, Uwall, kIn, omIn);
        SolverSettings s;
        s.maxIterations = 60000;
        s.convergenceTol = tol;
        s.alphaU = 0.7;
        s.alphaP = 0.5;
        s.verbose = false;
        SSTModel sst{SSTCoefficients{}};
        SIMPLESolver solver(mesh, sst, bcs, nu, s);
        solver.setConvergenceMonitor([&](const FlowFields &g) {
            return std::vector<double>{wallShear(mesh, g, nu, "bottom"),
                                       wallShear(mesh, g, nu, "top")};
        });
        FlowFields f(mesh);
        solver.initUniform(f, Vec3(Ub, 0.0, 0.0), 0.0, kIn, omIn);
        ConvergenceHistory h = solver.solve(f);
        REQUIRE(h.converged, "Couette must converge");
        double tb = wallShear(mesh, f, nu, "bottom"), tt = wallShear(mesh, f, nu, "top");
        double dpx = 0.0; // largest streamwise pressure range in a row
        for (int j = 0; j < 96; ++j) {
            double lo = f.p[4 * j], hi = f.p[4 * j];
            for (int i = 1; i < 4; ++i) {
                lo = std::min(lo, f.p[4 * j + i]);
                hi = std::max(hi, f.p[4 * j + i]);
            }
            dpx = std::max(dpx, hi - lo);
        }
        std::printf("  Couette: converged at %d, tau_top/tau_bottom - 1 = %.3e, "
                    "streamwise p range %.3e\n",
                    h.finalIter, tt / tb - 1.0, dpx);
        REQUIRE(std::fabs(tt / tb - 1.0) < 1e-4,
                "converged Couette must carry the same shear on both walls");
        REQUIRE(std::fabs(continityDefect(mesh, solver.faceFluxes())) < 1e-6,
                "3. stored fluxes must satisfy continuity (bounded mesh)");
    }

    // 2. converged field independent of the momentum under-relaxation
    {
        Mesh mesh = cosineMesh(36, 24);
        const double nu = 3.0e-4, kIn = 1e-4, omIn = 10.0;
        FlowBoundaryConditions bcs = FlowBoundaryConditions::channelDefaults(mesh, 1.0, kIn, omIn);
        SSTModel sst{SSTCoefficients{}};
        auto solveAt = [&](double alphaU, std::vector<double> &phiOut) {
            SolverSettings s;
            s.maxIterations = 60000;
            s.convergenceTol = 1e-8;
            s.alphaU = alphaU;
            s.alphaP = 0.25;
            s.verbose = false;
            s.bodyForce = Vec3(4.0e-3, 0.0, 0.0);
            // upwind convection isolates the pressure-velocity coupling and
            // reaches 1e-8 on this coarse separated case; the limited
            // second-order scheme contracts too slowly for a 1e-8 test here
            s.momentumConvection = "upwind";
            s.turbulenceConvection = "upwind";
            SIMPLESolver solver(mesh, sst, bcs, nu, s);
            FlowFields f(mesh);
            solver.initUniform(f, Vec3(0.3, 0.0, 0.0), 0.0, kIn, omIn);
            ConvergenceHistory h = solver.solve(f);
            REQUIRE(h.converged, "cosine-bump channel must converge tightly");
            phiOut = solver.faceFluxes();
            return f;
        };
        std::vector<double> phiA, phiB;
        FlowFields a = solveAt(0.4, phiA);
        FlowFields b = solveAt(0.6, phiB);
        double du = 0.0, uMax = 0.0;
        for (int ci = 0; ci < mesh.nCells(); ++ci) {
            du = std::max(du, (a.U[ci] - b.U[ci]).norm());
            uMax = std::max(uMax, a.U[ci].norm());
        }
        std::printf("  cosine bump: max|U(alpha_U=0.4) - U(0.6)| / max|U| = %.3e\n", du / uMax);
        REQUIRE(du / uMax < 1e-6, "the converged field must not depend on alpha_U");
        REQUIRE(continityDefect(mesh, phiA) < 1e-6,
                "3. stored fluxes must satisfy continuity (periodic mesh)");
    }

    // 4. directional odd-even diagnostic
    {
        Mesh mesh = Mesh::makeChannel2D(12, 10, 2.0, 1.0);
        ScalarField smooth(mesh, "s"), xOnly(mesh, "x"), both(mesh, "b");
        for (int ci = 0; ci < mesh.nCells(); ++ci) {
            const Vec3 &c = mesh.cell(ci).center;
            int i = ci % 12, j = ci / 12;
            smooth[ci] = std::sin(c.x) + c.y * c.y;
            xOnly[ci] = c.y + ((i % 2 == 0) ? 0.3 : -0.3); // alternates in x only
            both[ci] = ((i + j) % 2 == 0) ? 1.0 : -1.0;
        }
        OddEvenRatios rs = oddEvenEnergyRatios(mesh, smooth);
        OddEvenRatios rx = oddEvenEnergyRatios(mesh, xOnly);
        OddEvenRatios rb = oddEvenEnergyRatios(mesh, both);
        std::printf("  odd-even (two-colour, x, y): smooth %.3f %.3f %.3f | x-only %.3f %.3f %.3f "
                    "| checkerboard %.3f %.3f %.3f\n",
                    rs.twoColour, rs.x, rs.y, rx.twoColour, rx.x, rx.y, rb.twoColour, rb.x, rb.y);
        REQUIRE(rs.twoColour < 0.15 && rs.x < 0.15 && rs.y < 0.15, "smooth field scores low");
        REQUIRE(rx.x > 4.0 * rs.x, "the x mode must discriminate");
        REQUIRE(rx.x > 0.5 && rx.twoColour < 0.05,
                "x-only alternation: seen by x mode, missed by two-colour");
        REQUIRE(rb.twoColour > 0.99, "full checkerboard: two-colour mode");
    }

    std::printf("test_rhie_chow_policy: all checks passed\n");
    return 0;
}
