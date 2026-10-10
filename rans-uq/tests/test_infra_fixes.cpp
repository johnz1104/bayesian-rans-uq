// Infrastructure contracts, each pinned by a discriminating check:
//
//   1. GaussSeidelSolver performs a TRUE in-place sweep: after ONE sweep of a
//      3-cell chain the hand-computed Gauss-Seidel values appear (the old
//      implementation produced the Jacobi values, which differ in cells 1
//      and 2 because they must see updated upstream unknowns).
//   2. InferenceParameterSet::inBounds treats a size-mismatched theta as
//      out of bounds instead of reading past the end (was UB).
//   3. WarmStartCache::findNearest copies the entry UNDER THE LOCK: the copy
//      remains valid and equal after the source entry is evicted.
//   4. oddEvenEnergyRatio separates a smooth field from a checkerboard by
//      orders of magnitude (the diagnostic behind the Rhie-Chow probe flag).
//   5. Rhie-Chow policy: the bounded (outlet) channel converges both with
//      the legacy gate (no face-flux dissipation on outlet meshes) and with
//      the default stored Rhie-Chow fluxes, and the two solutions stay close
//      (the term changes stabilization, not the physics).

#include "Mesh.hpp"
#include "Field.hpp"
#include "FlowFields.hpp"
#include "BoundaryCondition.hpp"
#include "LinearSolver.hpp"
#include "InferenceParameters.hpp"
#include "SIMPLESolver.hpp"
#include "SSTModel.hpp"
#include "ForwardModel.hpp"
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
    // 1. true Gauss-Seidel sweep (vs the Jacobi it used to be)
    {
        LinearSystem A;
        A.nCells = 3;
        A.nIF = 2;
        A.diag = {2.0, 2.0, 2.0};
        A.upper = {-1.0, -1.0};
        A.lower = {-1.0, -1.0};
        A.own = {0, 1};
        A.nbr = {1, 2};
        A.source = {1.0, 0.0, 1.0};
        std::vector<double> x = {0.0, 0.0, 0.0};
        GaussSeidelSolver gs;
        gs.solve(A, x, /*maxIter=*/1, /*tol=*/1e-30);
        // one in-place ascending sweep: x0 = 0.5, x1 = (0 + x0)/2 = 0.25,
        // x2 = (1 + x1)/2 = 0.625; Jacobi would give 0.5, 0.0, 0.5
        REQUIRE(std::fabs(x[0] - 0.5) < 1e-15, "GS x0");
        REQUIRE(std::fabs(x[1] - 0.25) < 1e-15, "GS x1 must see updated x0");
        REQUIRE(std::fabs(x[2] - 0.625) < 1e-15, "GS x2 must see updated x1");
    }

    // 2. inBounds size guard
    {
        InferenceParameterSet ps = InferenceParameterSet::a1_betaStar();
        REQUIRE(ps.inBounds({0.31, 0.09}), "nominal theta must be in bounds");
        REQUIRE(!ps.inBounds({0.31}), "short theta must be out of bounds, not UB");
        REQUIRE(!ps.inBounds({0.31, 0.09, 0.5}), "long theta must be out of bounds");
        REQUIRE(!ps.inBounds({}), "empty theta must be out of bounds");
        // live10 covers exactly the equations' coefficients (kappa excluded)
        REQUIRE(InferenceParameterSet::live10().nActive() == 10, "live10 size");
    }

    // 3. warm-start cache copies under the lock (survives eviction)
    {
        Mesh mesh = Mesh::makeChannel2D(4, 3, 1.0, 1.0);
        mesh.computeWallDistance();
        WarmStartCache cache(/*maxSize=*/1, /*threshold=*/10.0);
        FlowFields f1(mesh);
        f1.p.setUniform(3.5);
        cache.store({0.3, 0.09}, f1);

        WarmStartCache::CacheEntry hit;
        REQUIRE(cache.findNearest({0.31, 0.09}, hit), "stored entry must be found");
        // evict the source entry (capacity 1)
        FlowFields f2(mesh);
        f2.p.setUniform(-7.0);
        cache.store({0.9, 0.19}, f2);
        REQUIRE(std::fabs(hit.fields.p[0] - 3.5) < 1e-15,
                "the copied entry must survive eviction of its source");
    }

    // 4. odd-even diagnostic discrimination
    {
        Mesh mesh = Mesh::makeChannel2D(12, 10, 2.0, 1.0);
        mesh.computeWallDistance();
        ScalarField smooth(mesh, "s"), checker(mesh, "c");
        for (int ci = 0; ci < mesh.nCells(); ++ci) {
            const Vec3 &c = mesh.cell(ci).center;
            smooth[ci] = 0.3 * c.x + 0.1 * c.y;
            int i = ci % 12, j = ci / 12;
            checker[ci] = ((i + j) % 2 == 0) ? 1.0 : -1.0;
        }
        double rSmooth = oddEvenEnergyRatio(mesh, smooth);
        double rCheck = oddEvenEnergyRatio(mesh, checker);
        REQUIRE(rSmooth < 0.05, "smooth field must score low");
        REQUIRE(rCheck > 0.99, "checkerboard must project almost exactly");
        REQUIRE(rCheck > 20.0 * rSmooth, "diagnostic must discriminate");
    }

    // 5. the Rhie-Chow probe flag on a bounded (outlet) channel
    {
        const int nx = 16, ny = 12;
        Mesh mesh = Mesh::makeChannel2D(nx, ny, 4.0, 1.0, 5000.0, 1.0);
        mesh.computeWallDistance();
        const double nu = 2.0e-4, kIn = 1e-4, omIn = 10.0;
        FlowBoundaryConditions bcs = FlowBoundaryConditions::channelDefaults(mesh, 1.0, kIn, omIn);
        SSTModel sst{SSTCoefficients{}};

        struct RcResult {
            double ub;
            double chk;
            double pSpan;
        };
        auto solveWith = [&](bool rcAll) -> RcResult {
            SolverSettings s;
            s.maxIterations = 20000;
            s.convergenceTol = 1e-3;
            s.alphaU = 0.5;
            s.alphaP = 0.3;
            s.verbose = false;
            s.legacyRhieChowGate = !rcAll;
            SIMPLESolver solver(mesh, sst, bcs, nu, s);
            FlowFields f(mesh);
            solver.initUniform(f, Vec3(1.0, 0.0, 0.0), 0.0, kIn, omIn);
            ConvergenceHistory h = solver.solve(f);
            REQUIRE(!h.diverged, "bounded channel must not diverge");
            REQUIRE(h.converged, "bounded channel must genuinely converge");
            double ub = 0.0, vol = 0.0;
            for (int ci = 0; ci < mesh.nCells(); ++ci) {
                ub += f.U[ci].x * mesh.cell(ci).volume;
                vol += mesh.cell(ci).volume;
            }
            // the decisive measurement: odd-even energy of the SOLVED
            // pressure field itself, not of a synthetic checkerboard
            double pMin = f.p[0], pMax = f.p[0];
            for (int ci = 1; ci < mesh.nCells(); ++ci) {
                pMin = std::min(pMin, f.p[ci]);
                pMax = std::max(pMax, f.p[ci]);
            }
            return {ub / vol, oddEvenEnergyRatio(mesh, f.p), pMax - pMin};
        };
        RcResult off = solveWith(false);
        RcResult on = solveWith(true);
        std::printf("  Rhie-Chow bounded probe: off(chk=%.6g, Ub=%.6g, "
                    "dp=%.6g) on(chk=%.6g, Ub=%.6g, dp=%.6g)\n",
                    off.chk, off.ub, off.pSpan, on.chk, on.ub, on.pSpan);
        // the default (gated-off) bounded solve must itself be checkerboard
        // free at the diagnostic level: this is the empirical evidence the
        // gate adjudication rests on, asserted rather than assumed
        REQUIRE(off.chk < 0.2, "solved bounded-channel pressure must not be checkerboarded");
        // the probe must not increase decoupling and must not change physics
        REQUIRE(on.chk < 0.2, "probe run must stay checkerboard free");
        REQUIRE(std::fabs(on.ub - off.ub) < 0.01 * std::fabs(off.ub),
                "the probe dissipation must not change the bulk physics");
    }

    std::printf("test_infra_fixes: all checks passed\n");
    return 0;
}
