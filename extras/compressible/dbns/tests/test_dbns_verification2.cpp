// Verification ladder (part 2): 2D inviscid shock reflection and the
// compressible wall observation operator.
//
//   1. Oblique-shock reflection -> match the analytic oblique-shock states in
//      the incident-shock region and the post-reflection region (Yee 1985).
//   2. Wall observation operator -> on a prescribed linear (Couette + linear-T)
//      field the operator must return the exact wall shear (Cf) and wall heat
//      flux (q_w), and a consistent Stanton number.
//
// Viscous flat-plate and compression-corner cases are NOT asserted here: the
// explicit pseudo-time march does not converge viscous-dominated steady states
// on the required near-wall meshes (they need implicit LU-SGS integration,
// covered by test_dbns_implicit).  The viscous operator and the
// heat-flux observation are exercised directly instead (case 2 and the conduction
// field), and the inviscid shock-capturing core is verified in part 1.

#include "Mesh.hpp"
#include "DBNSSolver.hpp"
#include "DBNSObservation.hpp"
#include "IdealGasEOS.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace dbns;

namespace {
#define REQUIRE(cond, msg)                                                                         \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            std::fprintf(stderr, "FAIL [%s:%d] %s\n  required: %s\n", __FILE__, __LINE__, (msg),   \
                         #cond);                                                                   \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

Primitive sampleNearest(const DBNSSolver &s, double x, double y) {
    const Mesh &m = s.mesh();
    int best = 0;
    double bd = 1e30;
    for (int ci = 0; ci < m.nCells(); ++ci) {
        double dx = m.cell(ci).center.x - x, dy = m.cell(ci).center.y - y;
        double d = dx * dx + dy * dy;
        if (d < bd) {
            bd = d;
            best = ci;
        }
    }
    return s.primitive(best);
}
} // namespace

static void test_oblique_shock() {
    // Yee, Warming, Harten (1985) Mach-2.9 shock reflection on [0,4]x[0,1].
    IdealGasEOS eos;
    eos.gamma = 1.4;
    eos.R = 1.0;
    Mesh mesh = Mesh::makeChannel2D(120, 30, 4.0, 1.0);

    SSTCoefficients sst;
    DBNSBoundaryConditions bcs;
    BoundarySpec inflow;
    inflow.kind = BoundaryKind::SupersonicInflow;
    inflow.freestream = {1.0, 2.9, 0.0, 0.71429, 0, 0};
    BoundarySpec top;
    top.kind = BoundaryKind::FixedState;
    top.freestream = {1.69997, 2.61934, -0.50633, 1.52819, 0, 0};
    BoundarySpec slip;
    slip.kind = BoundaryKind::SlipWall;
    BoundarySpec ext;
    ext.kind = BoundaryKind::Extrapolate;
    bcs.set("inlet", inflow);
    bcs.set("top", top);
    bcs.set("bottom", slip);
    bcs.set("outlet", ext);

    DBNSSettings st;
    st.timeMode = TimeMode::Steady;
    st.cfl = 0.5;
    st.viscous = false;
    st.turbulent = false;
    st.reconstructOrder = 2;
    st.limitReconstruction = true;
    st.convergenceTol = 1e-6;
    st.maxIterations = 5000;

    DBNSSolver solver(mesh, eos, sst, bcs, st);
    solver.initUniform({1.0, 2.9, 0.0, 0.71429, 0, 0});
    SolveReport rep = solver.solve();
    REQUIRE(rep.status != EvaluationStatus::Diverged, "oblique shock diverged");

    // region 2 (between incident and reflected shocks)
    Primitive r2 = sampleNearest(solver, 2.4, 0.85);
    REQUIRE(std::abs(r2.rho - 1.69997) / 1.69997 < 0.04, "region 2 density off");
    REQUIRE(std::abs(r2.p - 1.52819) / 1.52819 < 0.05, "region 2 pressure off");
    // region 3 (post-reflection, near the wall downstream)
    Primitive r3 = sampleNearest(solver, 3.5, 0.15);
    REQUIRE(std::abs(r3.rho - 2.68757) / 2.68757 < 0.05, "region 3 density off");
    REQUIRE(std::abs(r3.p - 2.93407) / 2.93407 < 0.05, "region 3 pressure off");
    REQUIRE(std::abs(r3.v) < 0.06, "region 3 should be parallel to the wall");
    std::printf("  [oblique] r2 rho=%.4f p=%.4f  r3 rho=%.4f p=%.4f (iters=%d)\n", r2.rho, r2.p,
                r3.rho, r3.p, rep.iterations);
}

static void test_wall_observation() {
    // Prescribe a linear Couette velocity and a linear temperature field; the
    // observation operator must return the exact wall shear and heat flux.
    IdealGasEOS eos;
    double H = 0.01, Lx = 0.02, U = 80.0, p0 = 101325.0;
    double Tw = 300.0, dT = 120.0; // T(y) = Tw + dT*(y/H)
    double mu = 0.005;
    Mesh mesh = Mesh::makeChannel2D(6, 24, Lx, H);
    mesh.computeWallDistance();

    SSTCoefficients sst;
    DBNSBoundaryConditions bcs; // BCs irrelevant here (no time advance)
    BoundarySpec ext;
    ext.kind = BoundaryKind::Extrapolate;
    bcs.set("inlet", ext);
    bcs.set("outlet", ext);
    bcs.set("top", ext);
    bcs.set("bottom", ext);

    DBNSSettings st;
    st.viscous = true;
    st.turbulent = false;
    st.constMu = mu;
    DBNSSolver solver(mesh, eos, sst, bcs, st);

    std::vector<Primitive> field(mesh.nCells());
    for (int ci = 0; ci < mesh.nCells(); ++ci) {
        double y = mesh.cell(ci).center.y;
        double T = Tw + dT * (y / H);
        field[ci] = {p0 / (eos.R * T), U * (y / H), 0.0, p0, 0.0, 0.0};
    }
    solver.initField(field);
    solver.prepareProperties();

    ReferenceState ref;
    ref.rho = p0 / (eos.R * Tw);
    ref.U = U;
    ref.T = Tw;
    ref.p = p0;
    DBNSObservation obs(solver, ref);
    WallRecord w = obs.wall("bottom", Tw);

    // exact wall shear and heat flux for the linear field
    double tau_exact = mu * U / H;
    double lam = eos.Cp() * mu / eos.Pr;
    double qw_exact = lam * dT / H;
    double dynP = 0.5 * ref.rho * U * U;
    double Cf_exact = tau_exact / dynP;

    int mid = (int)w.Cf.size() / 2;
    double Cf = w.Cf[mid], qw = w.qw[mid], St = w.St[mid];
    std::printf("  [wall obs] Cf=%.5e (exact %.5e)  qw=%.1f (exact %.1f)  St=%.4e\n", Cf, Cf_exact,
                qw, qw_exact, St);
    REQUIRE(std::abs(Cf - Cf_exact) / Cf_exact < 0.02, "wall Cf off from mu U/H");
    REQUIRE(std::abs(qw - qw_exact) / qw_exact < 0.02, "wall q_w off from lambda dT/H");

    // Stanton consistency: St = q_w / (rho_inf U_inf Cp (T_aw - T_w))
    double Minf = U / std::sqrt(eos.gamma * eos.R * Tw);
    double r = std::sqrt(eos.Pr);
    double Taw = Tw * (1.0 + r * 0.5 * (eos.gamma - 1.0) * Minf * Minf);
    double St_exact = qw_exact / (ref.rho * U * eos.Cp() * (Taw - Tw));
    REQUIRE(std::abs(St - St_exact) / std::abs(St_exact) < 0.02, "Stanton number inconsistent");
}

int main() {
    test_oblique_shock();
    test_wall_observation();
    std::printf("test_dbns_verification2: all checks passed\n");
    return 0;
}
