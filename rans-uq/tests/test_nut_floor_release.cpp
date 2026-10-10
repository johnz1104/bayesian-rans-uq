// Startup-only eddy-viscosity floor and the molecular wall-stress observable.
//
// A permanent nuT >= 0.1*nu floor changes the converged transport solution.
// Including cell-centre nuT in resolved-wall stress also biases the observable.
// This body-force-driven periodic-channel test checks that the floor releases
// after startup and that wall stress uses molecular viscosity:
//
//   1. In-window: while iter < turbStartIter + nuTFloorIters the floor binds,
//      so a short run exits with every wall-adjacent nuT at or above 0.1*nu
//      (the startup guard still exists).
//   2. Released: a full solve converges AFTER the window, and the converged
//      wall-adjacent nuT sits far below 0.1*nu (SST asymptotics restored; the
//      floor is genuinely gone from the converged state, and releasing it did
//      not destabilize the solve).
//   3. The skin-friction observable is MOLECULAR: scaling the stored nuT field
//      leaves the observed Cf bit-identical (a nu+nuT observable would move
//      by the scaling), so no numerical nuT bound can bias the observation.

#include "Mesh.hpp"
#include "Field.hpp"
#include "FlowFields.hpp"
#include "BoundaryCondition.hpp"
#include "SIMPLESolver.hpp"
#include "SSTModel.hpp"
#include "ObservationOperator.hpp"
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

double minWallAdjacentNuT(const FlowFields &fld, int nx, int ny) {
    double mn = 1e300;
    for (int i = 0; i < nx; ++i) {
        mn = std::min(mn, fld.nuT[0 * nx + i]);        // bottom row
        mn = std::min(mn, fld.nuT[(ny - 1) * nx + i]); // top row
    }
    return mn;
}

} // namespace

int main() {
    // the flat periodic channel of test_periodic_channel, body-force driven
    const int nx = 24, ny = 28;
    const double Lx = 3.0, H = 1.0, nu = 2.0e-4, fb = 5.0e-3;
    std::vector<double> xN(nx + 1), yB(nx + 1, 0.0);
    for (int i = 0; i <= nx; ++i)
        xN[i] = Lx * i / nx;
    Mesh mesh = Mesh::makeCurvedChannelPeriodic2D(xN, yB, H, ny, 2500.0, 1.0);
    mesh.computeWallDistance();

    const double kIn = 1e-4, omIn = 10.0;
    FlowBoundaryConditions bcs = FlowBoundaryConditions::channelDefaults(mesh, 1.0, kIn, omIn);
    SSTModel sst{SSTCoefficients{}};

    SolverSettings settings;
    settings.convergenceTol = 1e-4;
    settings.alphaU = 0.5;
    settings.alphaP = 0.3;
    settings.verbose = false;
    settings.bodyForce = Vec3(fb, 0.0, 0.0);

    // 1. in-window: the startup floor binds
    {
        SolverSettings s = settings;
        s.maxIterations = 200; // < turbStartIter + nuTFloorIters (default 505)
        SIMPLESolver solver(mesh, sst, bcs, nu, s);
        FlowFields fld(mesh);
        solver.initUniform(fld, Vec3(0.3, 0.0, 0.0), 0.0, kIn, omIn);
        solver.solve(fld);
        double mn = minWallAdjacentNuT(fld, nx, ny);
        REQUIRE(mn >= 0.1 * nu * (1.0 - 1e-12), "startup window must keep the 0.1*nu floor active");
    }

    // 2. released: converged wall-adjacent nuT is far below the old floor
    FlowFields converged(mesh);
    {
        SolverSettings s = settings;
        s.maxIterations = 12000;
        SIMPLESolver solver(mesh, sst, bcs, nu, s);
        solver.initUniform(converged, Vec3(0.3, 0.0, 0.0), 0.0, kIn, omIn);
        ConvergenceHistory hist = solver.solve(converged);
        REQUIRE(!hist.diverged, "floor release must not destabilize the solve");
        REQUIRE(hist.converged, "channel must converge with the startup-only floor");
        double mn = minWallAdjacentNuT(converged, nx, ny);
        REQUIRE(mn < 0.02 * nu,
                "converged wall-adjacent nuT must sit far below the old 0.1*nu floor");
    }

    // 3. the Cf observable ignores the stored nuT entirely (molecular stress)
    {
        ObservationOperator obs;
        obs.addSkinFriction("bottom_wall", Vec3(1.5, 0.0, 0.0), 0.0, 1.0, 1.0);
        double cf0 = obs.evaluate(mesh, converged, nu)[0];
        REQUIRE(cf0 > 0.0, "channel wall Cf must be positive");

        FlowFields scaled = converged;
        for (int ci = 0; ci < mesh.nCells(); ++ci)
            scaled.nuT[ci] = 5.0 * nu; // absurd eddy viscosity everywhere
        double cf1 = obs.evaluate(mesh, scaled, nu)[0];
        REQUIRE(cf0 == cf1, "skin-friction observable must be independent of the nuT field");
    }

    std::printf("test_nut_floor_release: all checks passed\n");
    return 0;
}
