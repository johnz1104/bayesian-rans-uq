// Convergence accounting under a turbulence update interval > 1.
//
// Zeroing k/omega change norms on non-update iterations can declare
// convergence between turbulence updates while those fields are still moving.
// The solver must carry the last-computed norms across iterations. This test
// uses an update interval of two and checks:
//
//   1. after the first turbulence update, no recorded k/omega entry is ever
//      zero (the false-zero channel is gone);
//   2. on non-update iterations the recorded norms EQUAL the previous
//      iteration's (carried, not recomputed or zeroed);
//   3. the converged final entry satisfies the tolerance in the TURBULENCE
//      norms too, not only momentum and pressure (turbulence genuinely
//      finished moving when convergence was declared).

#include "Mesh.hpp"
#include "Field.hpp"
#include "FlowFields.hpp"
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

} // namespace

int main() {
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
    settings.turbStartIter = 30;
    settings.turbUpdateInterval = 3; // exercise the carried-norm path hard
    SSTModel sst{SSTCoefficients{}};

    SIMPLESolver solver(mesh, sst, bcs, nu, settings);
    FlowFields fld(mesh);
    solver.initUniform(fld, Vec3(0.3, 0.0, 0.0), 0.0, kIn, omIn);
    ConvergenceHistory hist = solver.solve(fld);
    REQUIRE(hist.converged, "interval-3 channel must converge");

    const int start = settings.turbStartIter;
    const int interval = settings.turbUpdateInterval;
    int carriedChecked = 0;
    for (size_t i = 0; i < hist.entries.size(); ++i) {
        const ResidualEntry &e = hist.entries[i];
        if (e.iteration < start)
            continue;
        // 1. never a zero turbulence norm once turbulence is active
        REQUIRE(e.k > 0.0 && e.omega > 0.0, "turbulence norms must never be zero once active");
        // 2. non-update iterations carry the previous values exactly
        bool isUpdate = ((e.iteration - start) % interval) == 0;
        if (!isUpdate && i > 0) {
            const ResidualEntry &prev = hist.entries[i - 1];
            REQUIRE(e.k == prev.k && e.omega == prev.omega,
                    "non-update iterations must carry the last norms");
            ++carriedChecked;
        }
    }
    REQUIRE(carriedChecked > 10, "test must actually exercise carried entries");

    // 3. at the declared convergence the turbulence norms satisfy the tol
    const ResidualEntry &last = hist.entries.back();
    REQUIRE(last.k < settings.convergenceTol && last.omega < settings.convergenceTol,
            "convergence must include the turbulence norms");

    std::printf("test_convergence_accounting: all checks passed "
                "(%d carried entries verified)\n",
                carriedChecked);
    return 0;
}
