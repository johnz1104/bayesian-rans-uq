// Implicit (LU-SGS) steady-driver verification: the viscous-dominated steady
// cases the explicit pseudo-time march cannot converge on wall-clustered
// meshes (its viscous spectral radius scales as 1/dy^2).
//
//   1. Compressible Couette with constant viscosity -> EXACT steady solution:
//      constant shear gives a linear velocity profile regardless of the
//      density variation, and the energy balance k T'' = -mu (du/dy)^2 with
//      isothermal walls gives the parabolic viscous-heating temperature
//        T(y) = T0 + (mu Uw^2 / (2 k)) (y/H)(1 - y/H),  k = cp mu / Pr.
//      Run on a wall-clustered mesh: convergence here is precisely what the
//      explicit march lacks.
//   2. Supersonic laminar flat plate (M = 2, adiabatic wall): the developing
//      layer must be self-similar, Cf(x) sqrt(Re_x) constant across stations
//      inside the laminar compressible range, and the wall temperature must
//      sit at the laminar recovery value. Bounds bracket the analytic values
//      rather than pin them (constant viscosity is only approximately a
//      Chapman-Rubesin law), so the assertion verifies viscous-steady
//      convergence and boundary-layer physics, not a fitted constant.

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
} // namespace

static void test_couette_viscous_steady(bool periodic = false) {
    IdealGasEOS eos; // air: gamma 1.4, R 287, Pr 0.72
    double T0 = 300.0, p0 = 101325.0;
    double rho0 = p0 / (eos.R * T0);
    double H = 0.01, L = 0.02;
    double Uw = 300.0;
    double Re = 1.0e4;
    double mu = rho0 * Uw * H / Re;

    Mesh mesh = periodic ? Mesh::makePeriodicChannel2D(4, 40, L, .5 * H, 680, 1.0)
                         : Mesh::makeChannel2D(4, 40, L, H, Re, 1.0);

    SSTCoefficients sst;
    DBNSBoundaryConditions bcs;
    BoundarySpec bottom;
    bottom.kind = BoundaryKind::NoSlipIsothermal;
    bottom.wallTemp = T0;
    BoundarySpec top;
    top.kind = BoundaryKind::NoSlipIsothermal;
    top.wallTemp = T0;
    top.wallVelocity = Vec3{Uw, 0.0, 0.0};
    BoundarySpec ends;
    ends.kind = BoundaryKind::Extrapolate;
    // one pressure-anchored end: zero-gradient ends at subsonic speed trap a
    // standing acoustic mode (reflective both sides) that floors the residual
    // and lets the mean level wander; anchoring the outlet pressure pins the
    // level and lets the mode leave, which is what deep convergence needs
    BoundarySpec anchor;
    anchor.kind = BoundaryKind::SubsonicOutflow;
    anchor.backPressure = p0;
    bcs.set("bottom", bottom);
    bcs.set("top", top);
    if (!periodic) {
        bcs.set("inlet", ends);
        bcs.set("outlet", anchor);
    }

    DBNSSettings st;
    st.timeMode = TimeMode::Steady;
    st.implicitSteady = true;
    // the per-sweep effective step saturates at the acoustic scale, so the
    // diffusive start-up transient needs order 1e5 cheap sweeps (seconds)
    st.cflImplicit = 1e4;
    st.cflRampStart = 2.0;
    st.cflRampIters = 200;
    st.viscous = true;
    st.turbulent = false;
    st.constMu = mu;
    st.reconstructOrder = 2;
    st.limitReconstruction = true;
    st.convergenceTol = 1e-9;
    st.maxIterations = 150000;

    DBNSSolver solver(mesh, eos, sst, bcs, st);
    solver.initUniform({rho0, 0.0, 0.0, p0, 0.0, 0.0});
    SolveReport rep = solver.solve();
    std::printf("  [couette] status %d iters %d rel-res %.3e\n", (int)rep.status, rep.iterations,
                rep.finalResidual);
    REQUIRE(rep.status == EvaluationStatus::Converged,
            "implicit Couette must converge on the wall-clustered mesh");

    // exact solution checks at the mid column
    double kCond = eos.Cp() * mu / eos.Pr;
    double dTmax = mu * Uw * Uw / (8.0 * kCond);
    double uErrMax = 0.0, TErrMax = 0.0;
    const Mesh &m = solver.mesh();
    for (int ci = 0; ci < m.nCells(); ++ci) {
        double x = m.cell(ci).center.x, y = m.cell(ci).center.y;
        if (x < 0.25 * L || x > 0.75 * L)
            continue;
        Primitive V = solver.primitive(ci);
        double uExact = Uw * y / H;
        double eta = y / H;
        double TExact = T0 + (mu * Uw * Uw / (2.0 * kCond)) * eta * (1.0 - eta);
        double T = GasState::temperature(V, eos);
        uErrMax = std::max(uErrMax, std::abs(V.u - uExact) / Uw);
        TErrMax = std::max(TErrMax, std::abs(T - TExact) / dTmax);
    }
    std::printf("  [couette] max |du|/Uw %.4f  max |dT|/dTmax %.4f "
                "(dTmax %.2f K)\n",
                uErrMax, TErrMax, dTmax);
    // bars at the measured 40-cell discretization level with margin (the
    // converged discrete solution sits 2.5 percent of Uw off the exact
    // profile at mid-channel and 14 percent of the 8 K viscous-heating bump)
    REQUIRE(uErrMax < 0.04, "Couette velocity at the discretization level");
    REQUIRE(TErrMax < 0.20, "viscous-heating temperature at the level");
}

static void test_laminar_supersonic_plate() {
    IdealGasEOS eos;
    double Tinf = 300.0, pinf = 101325.0;
    double rhoinf = pinf / (eos.R * Tinf);
    double a = std::sqrt(eos.gamma * eos.R * Tinf);
    double Minf = 2.0, Uinf = Minf * a;
    // the domain is tall enough that the leading-edge (viscous-interaction)
    // wave crosses the top downstream of the outflow instead of reflecting
    // off the zero-gradient top back onto the plate mid-length
    double L = 0.1, Hdom = 0.05;
    double ReL = 2.0e4;
    double mu = rhoinf * Uinf * L / ReL;

    Mesh mesh = Mesh::makeChannel2D(60, 56, L, Hdom, ReL, 1.0);

    SSTCoefficients sst;
    DBNSBoundaryConditions bcs;
    BoundarySpec inflow;
    inflow.kind = BoundaryKind::SupersonicInflow;
    inflow.freestream = {rhoinf, Uinf, 0.0, pinf, 0.0, 0.0};
    BoundarySpec wall;
    wall.kind = BoundaryKind::NoSlipAdiabatic;
    BoundarySpec ext;
    ext.kind = BoundaryKind::Extrapolate;
    bcs.set("inlet", inflow);
    bcs.set("bottom", wall);
    bcs.set("top", ext);
    bcs.set("outlet", ext);

    DBNSSettings st;
    st.timeMode = TimeMode::Steady;
    st.implicitSteady = true;
    st.cflImplicit = 1e3;
    st.cflRampStart = 1.0;
    st.cflRampIters = 400;
    st.viscous = true;
    st.turbulent = false;
    st.constMu = mu;
    st.reconstructOrder = 2;
    st.limitReconstruction = true;
    st.convergenceTol = 1e-7;
    st.maxIterations = 160000;

    DBNSSolver solver(mesh, eos, sst, bcs, st);
    solver.initUniform({rhoinf, Uinf, 0.0, pinf, 0.0, 0.0});
    SolveReport rep = solver.solve();
    std::printf("  [plate] status %d iters %d rel-res %.3e\n", (int)rep.status, rep.iterations,
                rep.finalResidual);
    REQUIRE(rep.status == EvaluationStatus::Converged, "implicit laminar plate must converge");

    ReferenceState ref;
    ref.rho = rhoinf;
    ref.U = Uinf;
    ref.T = Tinf;
    ref.p = pinf;
    DBNSObservation obs(solver, ref);
    // adiabatic wall: pass the recovery estimate so Stanton is regular; only
    // Cf and the near-wall temperature are asserted
    double rLam = std::sqrt(eos.Pr);
    double Taw = Tinf * (1.0 + rLam * 0.5 * (eos.gamma - 1.0) * Minf * Minf);
    WallRecord w = obs.wall("bottom", Taw);

    // self-similarity: Cf sqrt(Re_x) at three stations away from the leading
    // edge and the outflow; the laminar compressible range brackets the value
    double s1 = -1.0, s2 = -1.0, s3 = -1.0;
    for (size_t i = 0; i < w.x.size(); ++i) {
        double Rex = rhoinf * Uinf * w.x[i] / mu;
        double s = w.Cf[i] * std::sqrt(std::max(Rex, 1.0));
        if (s1 < 0 && w.x[i] > 0.40 * L)
            s1 = s;
        if (s2 < 0 && w.x[i] > 0.60 * L)
            s2 = s;
        if (s3 < 0 && w.x[i] > 0.80 * L)
            s3 = s;
    }
    std::printf("  [plate] Cf sqrt(Re_x) at 0.4/0.6/0.8 L: %.3f %.3f %.3f\n", s1, s2, s3);
    REQUIRE(s1 > 0 && s2 > 0 && s3 > 0, "stations found");
    REQUIRE(std::abs(s2 / s1 - 1.0) < 0.10, "self-similar Cf decay (0.4-0.6 L)");
    REQUIRE(std::abs(s3 / s2 - 1.0) < 0.10, "self-similar Cf decay (0.6-0.8 L)");
    REQUIRE(s2 > 0.35 && s2 < 0.75, "Cf sqrt(Re_x) inside the laminar compressible range");

    // adiabatic-wall recovery: near-wall temperature at the recovery value
    double Twall = -1.0;
    const Mesh &m = solver.mesh();
    double bestd = 1e30;
    for (int ci = 0; ci < m.nCells(); ++ci) {
        double x = m.cell(ci).center.x, y = m.cell(ci).center.y;
        if (std::abs(x - 0.7 * L) > 0.02 * L)
            continue;
        if (y < bestd) {
            bestd = y;
            Twall = GasState::temperature(solver.primitive(ci), eos);
        }
    }
    double rMeasured = (Twall / Tinf - 1.0) / (0.5 * (eos.gamma - 1.0) * Minf * Minf);
    std::printf("  [plate] T_wall/T_inf %.4f  recovery factor %.3f "
                "(sqrt(Pr) = %.3f)\n",
                Twall / Tinf, rMeasured, rLam);
    REQUIRE(rMeasured > 0.70 && rMeasured < 1.0, "adiabatic recovery factor in the laminar range");
}

int main() {
    std::printf("[dbns implicit] Couette viscous steady (exact solution)\n");
    test_couette_viscous_steady();
    std::printf("[dbns implicit] periodic Couette analytic control\n");
    test_couette_viscous_steady(true);
    std::printf("[dbns implicit] supersonic laminar flat plate\n");
    test_laminar_supersonic_plate();
    std::printf("[dbns implicit] ALL PASSED\n");
    return 0;
}
