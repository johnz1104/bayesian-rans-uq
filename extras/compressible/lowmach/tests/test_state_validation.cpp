// Discriminating test for the direct per-cell state validation.
//
// The defect this pins: chained max/min reductions cannot detect a NaN,
// because std::max(a, NaN) evaluates the comparison as false and returns a,
// so an aggregate norm built by reduction stays finite while the field is
// corrupt. The test first DEMONSTRATES that masking on a plain reduction,
// then verifies stateIsValid catches a single corrupted cell in every solved
// field, plus each positivity violation, none of which the old
// aggregate-based check could see through a masked reduction.
#include "Mesh.hpp"
#include "SSTModel.hpp"
#include "CompressibleFlowFields.hpp"
#include "CompressibleBCs.hpp"
#include "CompressibleSIMPLESolver.hpp"
#include "IdealGasEOS.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <algorithm>

static void REQUIRE(bool ok, const char *msg) {
    if (!ok) {
        std::printf("FAIL: %s\n", msg);
        std::exit(1);
    }
}

int main() {
    const double NaN = std::numeric_limits<double>::quiet_NaN();

    // 1. The masking defect is operand-order dependent. A max-reduction over
    //    a NaN-corrupted vector supplies the running finite aggregate first
    //    and therefore reports a finite result. Conversely, the temperature
    //    clamp supplies NaN first and preserves it for stateIsValid.
    {
        double vals[4] = {1.0, NaN, 2.0, 0.5};
        double agg = 0.0;
        for (double v : vals)
            agg = std::max(agg, std::abs(v));
        REQUIRE(std::isfinite(agg), "premise: max-reduction must mask the NaN (else the old check "
                                    "was sufficient and this test is vacuous)");
        REQUIRE(std::isnan(std::max(NaN, 1.0)),
                "NaN as the first std::max operand must be preserved");
    }

    IdealGasEOS eos;
    const double T_in = 300.0, p_ref = 101325.0, Uin = 30.0;
    Mesh mesh = Mesh::makeChannel2D(8, 6, 4.0, 1.0, 1e5, 1.0);
    mesh.computeWallDistance();
    auto bcs = CompressibleBoundaryConditions::channelDefaults(mesh, Uin, T_in, p_ref, 1e-3, 10.0);
    SSTModel sst;
    SolverSettings settings;
    CompressibleSIMPLESolver solver(mesh, sst, bcs, eos, settings);

    CompressibleFlowFields f(mesh);
    solver.initUniform(f, Vec3(Uin, 0, 0), p_ref, T_in, 1e-3, 10.0);
    REQUIRE(solver.stateIsValid(f), "freshly initialized state must be valid");

    const int probe = mesh.nCells() / 2;

    // 2. a single NaN in any solved field is caught, including the spanwise
    //    velocity component the aggregate norms never saw
    {
        CompressibleFlowFields g = f;
        g.U[probe].x = NaN;
        REQUIRE(!solver.stateIsValid(g), "NaN in Ux must invalidate");
    }
    {
        CompressibleFlowFields g = f;
        g.U[probe].y = NaN;
        REQUIRE(!solver.stateIsValid(g), "NaN in Uy must invalidate");
    }
    {
        CompressibleFlowFields g = f;
        g.U[probe].z = NaN;
        REQUIRE(!solver.stateIsValid(g), "NaN in Uz must invalidate");
    }
    {
        CompressibleFlowFields g = f;
        g.p[probe] = NaN;
        REQUIRE(!solver.stateIsValid(g), "NaN in p must invalidate");
    }
    {
        CompressibleFlowFields g = f;
        g.T[probe] = NaN;
        REQUIRE(!solver.stateIsValid(g), "NaN in T must invalidate");
    }
    {
        CompressibleFlowFields g = f;
        g.rho[probe] = NaN;
        REQUIRE(!solver.stateIsValid(g), "NaN in rho must invalidate");
    }
    {
        CompressibleFlowFields g = f;
        g.k[probe] = NaN;
        REQUIRE(!solver.stateIsValid(g), "NaN in k must invalidate");
    }
    {
        CompressibleFlowFields g = f;
        g.omega[probe] = NaN;
        REQUIRE(!solver.stateIsValid(g), "NaN in omega must invalidate");
    }

    // 3. positivity violations in the thermodynamic state
    {
        CompressibleFlowFields g = f;
        g.T[probe] = 0.0;
        REQUIRE(!solver.stateIsValid(g), "T <= 0 must invalidate");
    }
    {
        CompressibleFlowFields g = f;
        g.rho[probe] = -1.0;
        REQUIRE(!solver.stateIsValid(g), "rho <= 0 must invalidate");
    }
    {
        CompressibleFlowFields g = f;
        g.p[probe] = 0.0;
        REQUIRE(!solver.stateIsValid(g), "p <= 0 must invalidate");
    }
    {
        CompressibleFlowFields g = f;
        g.p[probe] = 1.0;
        g.rho[probe] = 1.0;
        g.k[probe] = 3.0;
        REQUIRE(!solver.stateIsValid(g),
                "positive mechanical p with negative thermodynamic p must invalidate");
    }

    // 4. an infinity is as invalid as a NaN
    {
        CompressibleFlowFields g = f;
        g.U[probe].y = std::numeric_limits<double>::infinity();
        REQUIRE(!solver.stateIsValid(g), "inf in Uy must invalidate");
    }

    std::printf("test_state_validation: all checks passed\n");
    return 0;
}
