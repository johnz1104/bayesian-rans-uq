// SST omega wall anchoring: an imposed turbulent boundary layer must SUSTAIN
// its turbulence. Without the viscous-sublayer omega value at the wall the
// layer laminarizes and the skin friction decays like a laminar plate from
// the inflow (Cf 1.2e-3 falling to 0.6e-3 where the reference data holds
// 2.6e-3).
//
// Configuration: supersonic plate at M = 2 with a crude log-law turbulent
// inflow profile (u+ = ln(y+)/kappa + B, k and omega from the equilibrium
// log-layer relations). Assertions: the skin friction at the check station
// stays at the turbulent level (several times the laminar value at that
// Reynolds number), and the near-wall omega sits on the viscous-sublayer
// analytic floor.

#include "Mesh.hpp"
#include "DBNSSolver.hpp"
#include "DBNSObservation.hpp"
#include "IdealGasEOS.hpp"
#include <algorithm>
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

int main() {
    IdealGasEOS eos;
    double Tinf = 300.0, pinf = 101325.0;
    double rhoinf = pinf / (eos.R * Tinf);
    double a = std::sqrt(eos.gamma * eos.R * Tinf);
    double Minf = 2.0, Uinf = Minf * a;
    double L = 0.30, H = 0.06;
    double mu = eos.viscosity(Tinf);

    // an incoming layer of thickness delta with u_tau from a nominal
    // turbulent skin friction (the log-law construction below)
    double delta = 0.01;
    double cf0 = 2.6e-3;
    double Taw = Tinf * (1.0 + std::cbrt(eos.Pr) * 0.5 * (eos.gamma - 1.0) * Minf * Minf);
    double rho_w = rhoinf * Tinf / Taw; // adiabatic wall density
    double u_tau = Uinf * std::sqrt(0.5 * cf0 * rhoinf / rho_w);
    double nu_w = mu / rho_w;

    double ReL = rhoinf * Uinf * L / mu;
    Mesh mesh = Mesh::makePlate2D(120, 72, L, H, ReL, 1.0);

    SSTCoefficients sst;
    DBNSBoundaryConditions bcs;
    BoundarySpec inflow;
    inflow.kind = BoundaryKind::SupersonicInflow;
    inflow.freestream = {rhoinf, Uinf, 0.0, pinf, 1e-4 * Uinf * Uinf, 5.0 * Uinf / delta};

    // per-face log-law inflow profile: u+ = ln(y+)/kappa + B capped at the
    // free stream; k from the equilibrium log layer (u_tau^2/sqrt(beta*)),
    // decaying above the layer; omega = u_tau/(sqrt(beta*) kappa y) inside,
    // ambient above
    const double kappa = 0.41, B = 5.0, bStar = 0.09;
    std::vector<double> yc;
    {
        // unique wall distances of the wall-clustered mesh's cell rows
        std::vector<double> ys;
        for (int ci = 0; ci < mesh.nCells(); ++ci) {
            double y = mesh.cell(ci).center.y;
            bool seen = false;
            for (double v : ys)
                if (std::abs(v - y) < 1e-12) {
                    seen = true;
                    break;
                }
            if (!seen)
                ys.push_back(y);
        }
        std::sort(ys.begin(), ys.end());
        yc = ys;
    }
    inflow.profile.resize(yc.size());
    for (size_t j = 0; j < yc.size(); ++j) {
        double y = yc[j];
        double yplus = y * u_tau / nu_w;
        double uplus = std::min(std::log(std::max(yplus, 1.0)) / kappa + B, Uinf / u_tau);
        double u = std::min(uplus * u_tau, Uinf);
        double inside = (y < delta) ? 1.0 : std::exp(-(y - delta) / delta);
        double k = std::max(u_tau * u_tau / std::sqrt(bStar) * inside, 1e-4 * Uinf * Uinf);
        double omega = (y < delta) ? u_tau / (std::sqrt(bStar) * kappa * std::max(y, 1e-8))
                                   : 5.0 * Uinf / delta;
        Primitive V;
        V.rho = rhoinf;
        V.u = u;
        V.v = 0.0;
        V.p = pinf;
        V.k = k;
        V.omega = omega;
        inflow.profile[j] = V;
    }

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
    st.cflImplicit = 200.0;
    st.cflRampStart = 1.0;
    st.cflRampIters = 300;
    st.viscous = true;
    st.turbulent = true;
    st.reconstructOrder = 2;
    st.limitReconstruction = true;
    st.convergenceTol = 1e-5;
    st.maxIterations = 40000;

    DBNSSolver solver(mesh, eos, sst, bcs, st);
    solver.initUniform({rhoinf, Uinf, 0.0, pinf, 1e-4 * Uinf * Uinf, 5.0 * Uinf / delta});
    SolveReport rep = solver.solve();
    std::printf("  [sst wall] status %d iters %d rel-res %.3e\n", (int)rep.status, rep.iterations,
                rep.finalResidual);
    REQUIRE(rep.status != EvaluationStatus::Diverged, "turbulent plate must not diverge");

    ReferenceState ref;
    ref.rho = rhoinf;
    ref.U = Uinf;
    ref.T = Tinf;
    ref.p = pinf;
    DBNSObservation obs(solver, ref);
    WallRecord w = obs.wall("bottom", Taw);
    double cf_mid = -1.0, cf_late = -1.0;
    for (size_t i = 0; i < w.x.size(); ++i) {
        if (cf_mid < 0 && w.x[i] > 0.45 * L)
            cf_mid = w.Cf[i];
        if (cf_late < 0 && w.x[i] > 0.75 * L)
            cf_late = w.Cf[i];
    }
    // laminar level at these stations is about 0.5e-3; a sustained
    // turbulent layer holds several times that
    double Rex = rhoinf * Uinf * 0.75 * L / mu;
    double cf_lam = 0.664 / std::sqrt(Rex);
    std::printf("  [sst wall] Cf mid %.3e late %.3e (laminar %.3e)\n", cf_mid, cf_late, cf_lam);
    REQUIRE(cf_late > 3.0 * cf_lam, "turbulence sustained: Cf stays several times laminar");
    REQUIRE(cf_late > 0.6 * cf_mid, "no laminar-like streamwise collapse of Cf");

    // the near-wall omega sits on the viscous-sublayer analytic floor
    const Mesh &m = solver.mesh();
    int ci_wall = -1;
    double ybest = 1e30;
    for (int ci = 0; ci < m.nCells(); ++ci) {
        double x = m.cell(ci).center.x, y = m.cell(ci).center.y;
        if (std::abs(x - 0.5 * L) > 0.02 * L)
            continue;
        if (y < ybest) {
            ybest = y;
            ci_wall = ci;
        }
    }
    Primitive Vw = solver.primitive(ci_wall);
    double nu_loc = solver.laminarViscosity(ci_wall) / Vw.rho;
    double omegaVis = 6.0 * nu_loc / (sst.beta1 * ybest * ybest);
    std::printf("  [sst wall] near-wall omega %.3e vs sublayer %.3e\n", Vw.omega, omegaVis);
    REQUIRE(Vw.omega > 0.9 * omegaVis, "omega anchored at the wall");

    std::printf("[dbns sst wall] ALL PASSED\n");
    return 0;
}
