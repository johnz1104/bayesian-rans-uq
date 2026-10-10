// Model-form injection verification for the density-based solver: the
// deferred-correction coupling with the heat-flux (energy-equation) reach.
//
//   1. Zero-correction identity: a zero STORED discrepancy must reproduce
//      the baseline fields exactly, laminar AND turbulent. The turbulent
//      case is the discriminating one: a target-minus-solver-Boussinesq
//      formulation breaks this identity at any nonzero mu_t through the
//      operator difference between the conditioning-side gradients and the
//      solver's Green-Gauss ones; the stored-discrepancy form makes db = 0
//      exactly zero flux at any state.
//   2. Heat-flux reach directionality: a constant positive wall-normal
//      heat-flux correction transports energy upward, so the converged
//      temperature field must become asymmetric (upper half warmer than the
//      symmetric baseline) by the physically expected sign.
//   3. Realizability recording: a target outside the barycentric set is
//      recorded in the diagnostics (violation magnitude, not silently
//      projected); a realizable target records clean.
//
// The turbulent (stress-reach) coupled validation at the interaction Mach
// runs with the study's attached baseline (gate A of the pre-registered
// scheme), where the SST inflow state exists; this file verifies the
// injection machinery itself.

#include "Mesh.hpp"
#include "DBNSSolver.hpp"
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

struct CouetteCase {
    IdealGasEOS eos;
    double T0 = 300.0, p0 = 101325.0, H = 0.01, L = 0.02, Uw = 300.0;
    double rho0, mu;
    Mesh mesh;
    DBNSBoundaryConditions bcs;
    DBNSSettings st;

    CouetteCase() : mesh(Mesh::makeChannel2D(4, 32, 0.02, 0.01, 1.0e4, 1.0)) {
        rho0 = p0 / (eos.R * T0);
        mu = rho0 * Uw * H / 1.0e4;
        BoundarySpec bottom;
        bottom.kind = BoundaryKind::NoSlipIsothermal;
        bottom.wallTemp = T0;
        BoundarySpec top;
        top.kind = BoundaryKind::NoSlipIsothermal;
        top.wallTemp = T0;
        top.wallVelocity = Vec3{Uw, 0.0, 0.0};
        BoundarySpec ends;
        ends.kind = BoundaryKind::Extrapolate;
        BoundarySpec anchor;
        anchor.kind = BoundaryKind::SubsonicOutflow;
        anchor.backPressure = p0;
        bcs.set("bottom", bottom);
        bcs.set("top", top);
        bcs.set("inlet", ends);
        bcs.set("outlet", anchor);

        st.timeMode = TimeMode::Steady;
        st.implicitSteady = true;
        st.cflImplicit = 1e4;
        st.cflRampStart = 2.0;
        st.cflRampIters = 200;
        st.viscous = true;
        st.turbulent = false;
        st.constMu = mu;
        st.reconstructOrder = 2;
        st.limitReconstruction = true;
        st.convergenceTol = 1e-8;
        st.maxIterations = 120000;
    }
};

// mean temperature of the upper minus the lower quarter at mid length
double thermalAsymmetry(const DBNSSolver &s, const CouetteCase &c) {
    const Mesh &m = s.mesh();
    double hi = 0.0, lo = 0.0;
    int nhi = 0, nlo = 0;
    for (int ci = 0; ci < m.nCells(); ++ci) {
        double x = m.cell(ci).center.x, y = m.cell(ci).center.y;
        if (x < 0.25 * c.L || x > 0.75 * c.L)
            continue;
        double T = GasState::temperature(s.primitive(ci), c.eos);
        if (y > 0.75 * c.H) {
            hi += T;
            ++nhi;
        }
        if (y < 0.25 * c.H) {
            lo += T;
            ++nlo;
        }
    }
    return hi / nhi - lo / nlo;
}
} // namespace

static void test_zero_correction_identity() {
    CouetteCase c;
    DBNSSolver base(c.mesh, c.eos, SSTCoefficients{}, c.bcs, c.st);
    base.initUniform({c.rho0, 0.0, 0.0, c.p0, 0.0, 0.0});
    SolveReport rb = base.solve();
    REQUIRE(rb.status == EvaluationStatus::Converged, "baseline converges");

    DBNSSolver inj(c.mesh, c.eos, SSTCoefficients{}, c.bcs, c.st);
    inj.initUniform({c.rho0, 0.0, 0.0, c.p0, 0.0, 0.0});
    int nc = c.mesh.nCells();
    std::vector<double> b6(6 * nc, 0.0), dq2(2 * nc, 0.0);
    inj.setTargetCorrection(b6, b6, dq2, true);
    SolveReport ri = inj.solve();
    REQUIRE(ri.status == EvaluationStatus::Converged, "injected converges");

    double maxdiff = 0.0;
    for (int ci = 0; ci < nc; ++ci) {
        Primitive Vb = base.primitive(ci), Vi = inj.primitive(ci);
        maxdiff = std::max(maxdiff, std::abs(Vb.u - Vi.u) / c.Uw);
        maxdiff = std::max(maxdiff, std::abs(Vb.p - Vi.p) / c.p0);
    }
    std::printf("  [identity] max relative field difference %.3e\n", maxdiff);
    REQUIRE(maxdiff < 1e-10, "zero correction reproduces the baseline");
    REQUIRE(inj.injectionDiagnostics().active, "diagnostics active");
    REQUIRE(inj.injectionDiagnostics().allRealizable, "zero anisotropy is realizable");
    REQUIRE(inj.injectionDiagnostics().checkedIters > 0,
            "realizability re-checked in the running solve");
}

static void test_heat_flux_reach_direction() {
    CouetteCase c;
    DBNSSolver base(c.mesh, c.eos, SSTCoefficients{}, c.bcs, c.st);
    base.initUniform({c.rho0, 0.0, 0.0, c.p0, 0.0, 0.0});
    REQUIRE(base.solve().status == EvaluationStatus::Converged, "baseline converges");
    double asymBase = thermalAsymmetry(base, c);

    // an upward turbulent heat-flux correction with an interior divergence
    // (dq_y ~ sin(pi y/H)): a divergence-free constant flux would
    // short-circuit into the isothermal walls through the thin wall cells
    // and leave only a milli-Kelvin trace, so the test drives the interior,
    // where div(rho cp dq) cools the lower half and warms the upper half.
    // b = 0 stays inert on this laminar flow (the stress part carries k).
    DBNSSolver inj(c.mesh, c.eos, SSTCoefficients{}, c.bcs, c.st);
    inj.initUniform({c.rho0, 0.0, 0.0, c.p0, 0.0, 0.0});
    int nc = c.mesh.nCells();
    std::vector<double> b6(6 * nc, 0.0), dq2(2 * nc, 0.0);
    for (int ci = 0; ci < nc; ++ci) {
        double y = c.mesh.cell(ci).center.y;
        dq2[2 * ci + 1] = 1.0 * std::sin(M_PI * y / c.H); // dq_y [m/s K]
    }
    inj.setTargetCorrection(b6, b6, dq2, true);
    REQUIRE(inj.solve().status == EvaluationStatus::Converged, "injected converges");
    double asymInj = thermalAsymmetry(inj, c);

    std::printf("  [heat reach] upper-lower asymmetry: baseline %.3f K, "
                "injected %.3f K\n",
                asymBase, asymInj);
    // the baseline viscous-heating profile is close to symmetric between
    // isothermal walls (a sub-Kelvin outlet-anchor bias is tolerated); the
    // upward flux correction must warm the upper quarter against it
    REQUIRE(std::abs(asymBase) < 1.0, "baseline near-symmetric");
    REQUIRE(asymInj > asymBase + 0.5, "upward heat-flux reach warms the top");

    // the energy reach off: the same targets must leave the flow unchanged
    // on this laminar case (the stress part is identically zero)
    DBNSSolver noReach(c.mesh, c.eos, SSTCoefficients{}, c.bcs, c.st);
    noReach.initUniform({c.rho0, 0.0, 0.0, c.p0, 0.0, 0.0});
    noReach.setTargetCorrection(b6, b6, dq2, false);
    REQUIRE(noReach.solve().status == EvaluationStatus::Converged,
            "anisotropy-only variant converges");
    double asymOff = thermalAsymmetry(noReach, c);
    std::printf("  [heat reach] energy reach off: asymmetry %.3f K\n", asymOff);
    REQUIRE(std::abs(asymOff - asymBase) < 0.2,
            "anisotropy-only variant carries no heat-flux correction");
}

static void test_realizability_recording() {
    // the diagnostic is on the EFFECTIVE RUNNING anisotropy
    // b_eff(W) = b_B(W) + db_stored. On this laminar flow mu_t = 0, so
    // b_eff = db exactly, and the running k = 0 makes the injected flux
    // -div(2 rho k db) identically zero: the recording is exercised without
    // a destabilizing force, and the state is never projected or clipped.
    CouetteCase c;
    c.st.maxIterations = 300; // a short march suffices for the recording
    c.st.convergenceTol = 1e-30;
    DBNSSolver s(c.mesh, c.eos, SSTCoefficients{}, c.bcs, c.st);
    s.initUniform({c.rho0, 0.0, 0.0, c.p0, 0.0, 0.0});
    int nc = c.mesh.nCells();
    std::vector<double> db6(6 * nc, 0.0), dq2;
    for (int ci = 0; ci < nc; ++ci) {
        db6[6 * ci + 0] = 0.9; // b_xx beyond the one-component corner
        db6[6 * ci + 1] = -0.45;
        db6[6 * ci + 2] = -0.45;
    }
    std::vector<double> bDiag(6 * nc, 0.0);
    s.setTargetCorrection(db6, bDiag, dq2, true);
    s.solve();
    const InjectionDiagnostics &d = s.injectionDiagnostics();
    std::printf("  [realizability] checked %d iters, all realizable %d, "
                "min margin %.3e at iter %d cell %d, max violation %.3e\n",
                d.checkedIters, (int)d.allRealizable, d.minMargin, d.minMarginIter, d.minMarginCell,
                d.maxViolation);
    REQUIRE(d.checkedIters > 0, "check ran");
    REQUIRE(!d.allRealizable, "violation recorded, not silently projected");
    REQUIRE(d.maxViolation > 0.0, "violation magnitude recorded");
    REQUIRE(d.minMargin < 0.0, "worst margin recorded");
    REQUIRE(d.minMarginIter > 0, "iteration of the worst margin recorded");
    REQUIRE(d.minMarginCell >= 0 && d.minMarginCell < nc, "cell of the worst margin recorded");
    REQUIRE(d.maxViolationIter > 0 && d.maxViolationCell >= 0,
            "violation extreme carries iteration and cell");

    // a realizable stored discrepancy on the same laminar state records a
    // clean margin (b_eff = db inside the triangle)
    DBNSSolver s2(c.mesh, c.eos, SSTCoefficients{}, c.bcs, c.st);
    s2.initUniform({c.rho0, 0.0, 0.0, c.p0, 0.0, 0.0});
    std::vector<double> small(6 * nc, 0.0);
    for (int ci = 0; ci < nc; ++ci)
        small[6 * ci + 3] = -0.05;
    s2.setTargetCorrection(small, bDiag, dq2, true);
    s2.solve();
    REQUIRE(s2.injectionDiagnostics().allRealizable,
            "realizable effective anisotropy records clean");
    REQUIRE(s2.injectionDiagnostics().minMargin >= 0.0, "clean margin is nonnegative");

    s.clearTargetCorrection();
    REQUIRE(!s.injectionDiagnostics().active, "clear resets the diagnostics");
}

static void test_injection_conservation_sign_and_work() {
    // One explicit forward-Euler step from an identical state, with and
    // without the injection: the difference isolates the injected fluxes
    // exactly. A uniform stored discrepancy makes the injected stress
    // uniform, so interior cells feel no net force (the face fluxes
    // telescope) and only the rows adjacent to the walls (whose boundary
    // faces carry no injected flux) respond, with opposite signs; the
    // global sums vanish (conservative internal-face assembly).
    CouetteCase c;
    c.st.turbulent = true; // running rho k scales the injection
    c.st.timeMode = TimeMode::Unsteady;
    c.st.rkStages = 1;
    c.st.tEnd = 1e-12; // ONE capped global step: the step must
                       // stay below the explicit stability limit
                       // so exactly one Euler update runs and
                       // the A/B difference is the injected
                       // flux alone (a longer horizon takes
                       // several steps and couples nonlinearly)
    c.st.maxIterations = 10;
    int nc = c.mesh.nCells();
    Primitive init{c.rho0, 0.5 * c.Uw, 0.0, c.p0, 1.0, 50.0};

    auto oneStep = [&](const std::vector<double> &db6, const std::vector<double> &dq2, bool reach) {
        DBNSSolver s(c.mesh, c.eos, SSTCoefficients{}, c.bcs, c.st);
        s.initUniform(init);
        std::vector<double> bDiag(6 * nc, 0.0);
        s.setTargetCorrection(db6, bDiag, dq2, reach);
        s.solve();
        std::vector<StateVec> W(nc);
        for (int ci = 0; ci < nc; ++ci)
            W[ci] = s.conserved(ci);
        return W;
    };

    std::vector<double> zero6(6 * nc, 0.0), zero2;
    std::vector<double> db6(6 * nc, 0.0);
    for (int ci = 0; ci < nc; ++ci)
        db6[6 * ci + 3] = -0.15; // db_xy
    std::vector<double> dq2(2 * nc, 0.0);
    for (int ci = 0; ci < nc; ++ci)
        dq2[2 * ci + 1] = 1.0; // dq_y

    auto WA = oneStep(zero6, zero2, true); // no injection
    auto WB = oneStep(db6, zero2, true);   // stress + work reach
    auto WC = oneStep(db6, zero2, false);  // stress, reach off
    auto WD = oneStep(zero6, dq2, true);   // heat flux only

    // per-unit-depth cell volumes (divergence theorem, the solver's own
    // construction): the global step divides the residual by the cell
    // volume, so the conservation sums must be volume-weighted
    std::vector<double> vol(nc, 0.0);
    for (int ci = 0; ci < nc; ++ci) {
        double sv = 0.0;
        for (FaceID fi : c.mesh.cell(ci).faces) {
            const Face &f = c.mesh.face(fi);
            double sgn = (f.owner == ci) ? 1.0 : -1.0;
            sv += (f.center.x * f.normal.x + f.center.y * f.normal.y) * sgn * f.area;
        }
        vol[ci] = 0.5 * sv;
    }

    double yLo = 1e30, yHi = -1e30;
    for (int ci = 0; ci < nc; ++ci) {
        double y = c.mesh.cell(ci).center.y;
        yLo = std::min(yLo, y);
        yHi = std::max(yHi, y);
    }
    double sumU = 0.0, sumAbsU = 0.0, sumE = 0.0, sumAbsE = 0.0;
    double loU = 0.0, hiU = 0.0, edgeMag = 0.0;
    for (int ci = 0; ci < nc; ++ci) {
        double dU = vol[ci] * (WB[ci][I_RHOU] - WA[ci][I_RHOU]);
        double dE = vol[ci] * (WB[ci][I_RHOE] - WA[ci][I_RHOE]);
        sumU += dU;
        sumAbsU += std::abs(dU);
        sumE += dE;
        sumAbsE += std::abs(dE);
        edgeMag = std::max(edgeMag, std::abs(dU));
        double y = c.mesh.cell(ci).center.y;
        if (std::abs(y - yLo) < 1e-15)
            loU += dU;
        if (std::abs(y - yHi) < 1e-15)
            hiU += dU;
    }
    std::printf("  [conservation] sum dU %.3e (abs %.3e), sum dE %.3e "
                "(abs %.3e), wall rows %.3e / %.3e\n",
                sumU, sumAbsU, sumE, sumAbsE, loU, hiU);
    // tolerance floor: the positivity clamp's decode-encode roundtrip adds
    // ulp-level state noise per cell, so the telescoped sums land at that
    // floor rather than at exact zero (measured order 1e-7 relative at this
    // step size; a non-conservative assembly would read order one)
    REQUIRE(sumAbsU > 0.0, "the injection moved momentum");
    REQUIRE(std::abs(sumU) < 1e-6 * sumAbsU, "global momentum change telescopes to zero");
    REQUIRE(std::abs(sumE) < 1e-6 * std::max(sumAbsE, 1e-300),
            "global energy change telescopes to zero");
    // sign: tau_inj_xy = -2 rho k db_xy > 0 for db_xy < 0 transports
    // x-momentum toward the lower wall row
    REQUIRE(loU > 0.0, "lower wall row gains streamwise momentum");
    REQUIRE(hiU < 0.0, "upper wall row loses streamwise momentum");
    // interior cells of a uniform correction feel no net force
    for (int ci = 0; ci < nc; ++ci) {
        double y = c.mesh.cell(ci).center.y;
        if (std::abs(y - yLo) < 1e-15 || std::abs(y - yHi) < 1e-15)
            continue;
        double dU = vol[ci] * (WB[ci][I_RHOU] - WA[ci][I_RHOU]);
        REQUIRE(std::abs(dU) < 1e-9 * edgeMag, "uniform correction is force-free in the interior");
    }
    // energy reach off: identical momentum rows, untouched energy row (the
    // energy comparison tolerates the positivity clamp's decode-encode
    // roundtrip, whose arithmetic differs once the momentum rows differ)
    for (int ci = 0; ci < nc; ++ci) {
        REQUIRE(WC[ci][I_RHOU] == WB[ci][I_RHOU], "reach flag does not alter the momentum flux");
        REQUIRE(std::abs(WC[ci][I_RHOE] - WA[ci][I_RHOE]) <= 1e-12 * std::abs(WA[ci][I_RHOE]),
                "reach off leaves the energy row at baseline");
    }
    // heat-flux-only injection: momentum untouched, energy conservative
    double sumEq = 0.0, sumAbsEq = 0.0;
    for (int ci = 0; ci < nc; ++ci) {
        REQUIRE(WD[ci][I_RHOU] == WA[ci][I_RHOU], "dq carries no momentum flux");
        double dE = vol[ci] * (WD[ci][I_RHOE] - WA[ci][I_RHOE]);
        sumEq += dE;
        sumAbsEq += std::abs(dE);
    }
    REQUIRE(sumAbsEq > 0.0, "the heat-flux correction moved energy");
    REQUIRE(std::abs(sumEq) < 1e-6 * sumAbsEq, "global heat-flux energy change telescopes to zero");
}

static void test_frozen_mean_transport_mode() {
    // the registered gate-B fallback baseline: the primitive mean is pinned
    // every iteration and only k and omega march
    CouetteCase c;
    c.st.turbulent = true;
    c.st.frozenMeanFlow = true;
    c.st.maxIterations = 2000;
    c.st.convergenceTol = 1e-4;
    DBNSSolver s(c.mesh, c.eos, SSTCoefficients{}, c.bcs, c.st);
    Primitive init{c.rho0, 0.5 * c.Uw, 0.0, c.p0, 1.0, 50.0};
    s.initUniform(init);
    SolveReport rep = s.solve();
    std::printf("  [frozen mean] status %d iters %d\n", (int)rep.status, rep.iterations);
    double meanDrift = 0.0, kMove = 0.0, omMove = 0.0;
    for (int ci = 0; ci < c.mesh.nCells(); ++ci) {
        Primitive V = s.primitive(ci);
        meanDrift = std::max({meanDrift, std::abs(V.rho - init.rho) / init.rho,
                              std::abs(V.u - init.u) / c.Uw, std::abs(V.v) / c.Uw,
                              std::abs(V.p - init.p) / init.p});
        kMove = std::max(kMove, std::abs(V.k - init.k));
        omMove = std::max(omMove, std::abs(V.omega - init.omega));
    }
    std::printf("  [frozen mean] mean drift %.3e, k moved %.3e, "
                "omega moved %.3e\n",
                meanDrift, kMove, omMove);
    REQUIRE(meanDrift < 1e-10, "the mean state is pinned");
    REQUIRE(kMove > 1e-3, "k transport marched");
    REQUIRE(omMove > 1.0, "omega transport marched");

    // the explicit path refuses the mode (implicit steady only)
    CouetteCase c2;
    c2.st.turbulent = true;
    c2.st.frozenMeanFlow = true;
    c2.st.implicitSteady = false;
    DBNSSolver s2(c2.mesh, c2.eos, SSTCoefficients{}, c2.bcs, c2.st);
    s2.initUniform(init);
    bool threw = false;
    try {
        s2.solve();
    } catch (const std::exception &) {
        threw = true;
    }
    REQUIRE(threw, "frozen mean without the implicit driver refuses");
}

static void test_turbulent_zero_discrepancy_identity() {
    // SST active (k, mu_t nonzero): the laminar identity can never catch a
    // Boussinesq-subtraction operator mismatch, this one exists to
    CouetteCase c;
    c.st.turbulent = true;
    c.st.maxIterations = 400; // a short march discriminates fully
    c.st.convergenceTol = 1e-30;
    auto init = [&](DBNSSolver &s) {
        s.initUniform({c.rho0, 0.5 * c.Uw, 0.0, c.p0, 1.0, 50.0}); // nonzero k, omega
    };
    DBNSSolver base(c.mesh, c.eos, SSTCoefficients{}, c.bcs, c.st);
    init(base);
    base.solve();

    DBNSSolver inj(c.mesh, c.eos, SSTCoefficients{}, c.bcs, c.st);
    init(inj);
    int nc = c.mesh.nCells();
    std::vector<double> db6(6 * nc, 0.0), bDiag(6 * nc, 0.0), dq2;
    // a NONZERO diagnostic target: only the stored discrepancy may act
    for (int ci = 0; ci < nc; ++ci)
        bDiag[6 * ci + 3] = -0.15;
    inj.setTargetCorrection(db6, bDiag, dq2, true);
    inj.solve();

    double maxDiff = 0.0;
    for (int ci = 0; ci < nc; ++ci) {
        Primitive Vb = base.primitive(ci), Vi = inj.primitive(ci);
        maxDiff =
            std::max({maxDiff, std::abs(Vb.u - Vi.u), std::abs(Vb.v - Vi.v), std::abs(Vb.p - Vi.p),
                      std::abs(Vb.k - Vi.k), std::abs(Vb.omega - Vi.omega)});
    }
    std::printf("  [turbulent identity] max primitive difference %.3e\n", maxDiff);
    REQUIRE(maxDiff == 0.0, "zero stored discrepancy is bit-identical at a turbulent state");
}

static void test_frozen_mean_sweep_isolation() {
    // sweep contamination: the LU-SGS neighbor terms
    // consume the full six-component increment of already-swept cells, so
    // phantom mean-flow increments would contaminate the k and omega
    // coupling even though the final mean is re-pinned. Discriminator: an
    // artificial MEAN-ROW source (manufactured source on rho, momentum and
    // energy only) changes the mean residual and hence any phantom mean
    // increment, but under the corrected sweep (mean rows zeroed the
    // moment they exist) the turbulence march must be BIT-IDENTICAL.
    CouetteCase c;
    c.st.turbulent = true;
    c.st.frozenMeanFlow = true;
    c.st.maxIterations = 300;
    c.st.convergenceTol = 1e-30;
    Primitive init{c.rho0, 0.5 * c.Uw, 0.0, c.p0, 1.0, 50.0};

    DBNSSolver a(c.mesh, c.eos, SSTCoefficients{}, c.bcs, c.st);
    a.initUniform(init);
    a.solve();

    DBNSSolver b(c.mesh, c.eos, SSTCoefficients{}, c.bcs, c.st);
    b.initUniform(init);
    int nc = c.mesh.nCells();
    std::vector<StateVec> src(nc, StateVec{});
    for (int ci = 0; ci < nc; ++ci) {
        double y = c.mesh.cell(ci).center.y;
        src[ci][I_RHO] = 1.0;
        src[ci][I_RHOU] = 1e3 * std::sin(M_PI * y / c.H);
        src[ci][I_RHOE] = 1e5;
    }
    b.setManufacturedSource(src);
    b.solve();

    double kDiff = 0.0, omDiff = 0.0;
    for (int ci = 0; ci < nc; ++ci) {
        Primitive Va = a.primitive(ci), Vb = b.primitive(ci);
        kDiff = std::max(kDiff, std::abs(Va.k - Vb.k));
        omDiff = std::max(omDiff, std::abs(Va.omega - Vb.omega));
    }
    std::printf("  [sweep isolation] k diff %.3e, omega diff %.3e under a "
                "mean-row source\n",
                kDiff, omDiff);
    REQUIRE(kDiff == 0.0 && omDiff == 0.0,
            "mean-row residuals cannot touch the frozen-mean turbulence "
            "march");
}

static void test_checkpoint_roundtrip_zero_identity() {
    // the persisted-checkpoint contract of the member solves: reload a
    // CONVERGED state with its reconstruction-limiter checkpoint restored,
    // inject a zero correction, and the solve must classify Converged
    // (the quiescence route) while the state stays at the checkpoint;
    // without the limiter restore the fresh limiters re-open the
    // reconstruction and the restart drifts.
    CouetteCase c;
    // This test restores the explicitly historical frozen reconstruction map.
    c.st.legacyAutoFreezeLimiter = true;
    DBNSSolver a(c.mesh, c.eos, SSTCoefficients{}, c.bcs, c.st);
    a.initUniform({c.rho0, 0.0, 0.0, c.p0, 0.0, 0.0});
    REQUIRE(a.solve().status == EvaluationStatus::Converged, "reference converges");
    int nc = c.mesh.nCells();
    std::vector<Primitive> checkpoint(nc);
    for (int ci = 0; ci < nc; ++ci)
        checkpoint[ci] = a.primitive(ci);
    std::vector<double> lims = a.reconstructionLimiter();

    auto drift = [&](const DBNSSolver &s) {
        double d = 0.0;
        for (int ci = 0; ci < nc; ++ci) {
            Primitive V = s.primitive(ci);
            d = std::max(d, std::abs(V.u - checkpoint[ci].u) / c.Uw);
            d = std::max(d, std::abs(V.p - checkpoint[ci].p) / c.p0);
        }
        return d;
    };

    std::vector<double> zero6(6 * nc, 0.0), dq2(2 * nc, 0.0);
    DBNSSolver rest(c.mesh, c.eos, SSTCoefficients{}, c.bcs, c.st);
    rest.initField(checkpoint);
    rest.setFrozenReconstructionLimiter(lims);
    rest.setTargetCorrection(zero6, zero6, dq2, true);
    SolveReport rr = rest.solve();
    double dRestored = drift(rest);

    DBNSSolver fresh(c.mesh, c.eos, SSTCoefficients{}, c.bcs, c.st);
    fresh.initField(checkpoint);
    fresh.setTargetCorrection(zero6, zero6, dq2, true);
    SolveReport rf = fresh.solve();
    double dFresh = drift(fresh);

    std::printf("  [checkpoint] restored: %d iters %d, drift %.3e | fresh "
                "limiters: %d iters %d, drift %.3e\n",
                (int)rr.status, rr.iterations, dRestored, (int)rf.status, rf.iterations, dFresh);
    REQUIRE(rr.status == EvaluationStatus::Converged, "restored checkpoint classifies converged");
    // the restored restart continues the baseline's remaining decay toward
    // the true fixed point (a relative-decay criterion leaves that content
    // behind by construction); the contract is that the OPERATOR is the
    // checkpointed one, so the drift is the small continued-decay term and
    // never the order-larger limiter-refresh re-transient of a fresh
    // restart (measured here about two orders apart)
    REQUIRE(dRestored < 1e-3, "restored drift is bounded by the remaining-decay scale");
    REQUIRE(dRestored <= 0.1 * dFresh,
            "the limiter restore removes the limiter-refresh re-transient");
}

static void test_turbulent_checkpoint_and_forced_restart() {
    // blocker-1/2 gates: (a) the TURBULENT round-trip (mu_t live, so the
    // restart's first-evaluation eddy viscosity matters): a zero-correction
    // restored restart classifies via the quiescence route and holds the
    // state; (b) a NONZERO stored discrepancy is a fresh force, so its
    // restart must NEVER carry the quiescence tag whatever its status.
    CouetteCase c;
    // This test restores the explicitly historical frozen reconstruction map.
    c.st.legacyAutoFreezeLimiter = true;
    c.st.turbulent = true;
    c.st.convergenceTol = 1e-5;
    c.st.maxIterations = 60000;
    DBNSSolver a(c.mesh, c.eos, SSTCoefficients{}, c.bcs, c.st);
    a.initUniform({c.rho0, 0.5 * c.Uw, 0.0, c.p0, 1.0, 5000.0});
    SolveReport ra = a.solve();
    std::printf("  [turb checkpoint] reference status %d iters %d\n", (int)ra.status,
                ra.iterations);
    REQUIRE(ra.status == EvaluationStatus::Converged, "turbulent reference converges");
    int nc = c.mesh.nCells();
    std::vector<Primitive> chk(nc);
    for (int ci = 0; ci < nc; ++ci)
        chk[ci] = a.primitive(ci);
    std::vector<double> lims = a.reconstructionLimiter();
    std::vector<double> zero6(6 * nc, 0.0), dq2(2 * nc, 0.0);

    DBNSSolver rest(c.mesh, c.eos, SSTCoefficients{}, c.bcs, c.st);
    rest.initField(chk);
    rest.setFrozenReconstructionLimiter(lims);
    rest.setTargetCorrection(zero6, zero6, dq2, true);
    SolveReport rr = rest.solve();
    double drift = 0.0;
    for (int ci = 0; ci < nc; ++ci) {
        Primitive V = rest.primitive(ci);
        drift = std::max(drift, std::abs(V.u - chk[ci].u) / c.Uw);
        drift = std::max(drift, std::abs(V.p - chk[ci].p) / c.p0);
    }
    std::printf("  [turb checkpoint] zero-db restart status %d iters %d "
                "quiescent %d drift %.3e\n",
                (int)rr.status, rr.iterations, (int)rr.quiescent, drift);
    REQUIRE(rr.status == EvaluationStatus::Converged,
            "turbulent zero-db restart classifies converged");
    REQUIRE(rr.quiescent, "zero-db restored restart rides quiescence");
    REQUIRE(drift < 1e-2, "turbulent restart holds the checkpoint");

    // a forced restart with a small stored discrepancy: whatever the
    // classification, the quiescence tag must be false (the gate)
    std::vector<double> db6(6 * nc, 0.0);
    for (int ci = 0; ci < nc; ++ci)
        db6[6 * ci + 3] = -1e-4;
    DBNSSolver forced(c.mesh, c.eos, SSTCoefficients{}, c.bcs, c.st);
    forced.initField(chk);
    forced.setFrozenReconstructionLimiter(lims);
    forced.setTargetCorrection(db6, zero6, dq2, true);
    SolveReport rf = forced.solve();
    std::printf("  [turb checkpoint] forced restart status %d iters %d "
                "quiescent %d\n",
                (int)rf.status, rf.iterations, (int)rf.quiescent);
    REQUIRE(!rf.quiescent, "a nonzero stored discrepancy never rides quiescence");
}

int main() {
    std::printf("[dbns injection] zero-correction identity\n");
    test_zero_correction_identity();
    std::printf("[dbns injection] turbulent zero-discrepancy identity\n");
    test_turbulent_zero_discrepancy_identity();
    std::printf("[dbns injection] heat-flux reach directionality\n");
    test_heat_flux_reach_direction();
    std::printf("[dbns injection] effective-running realizability recording\n");
    test_realizability_recording();
    std::printf("[dbns injection] conservation, sign and energy work\n");
    test_injection_conservation_sign_and_work();
    std::printf("[dbns injection] frozen-mean transport mode\n");
    test_frozen_mean_transport_mode();
    std::printf("[dbns injection] frozen-mean sweep isolation\n");
    test_frozen_mean_sweep_isolation();
    std::printf("[dbns injection] checkpoint round-trip zero identity\n");
    test_checkpoint_roundtrip_zero_identity();
    std::printf("[dbns injection] turbulent checkpoint and forced restart\n");
    test_turbulent_checkpoint_and_forced_restart();
    std::printf("[dbns injection] ALL PASSED\n");
    return 0;
}
