// Smoke test: pressure-based compressible SIMPLE solver at Ma=0.1.
//
// Purpose: lock in the validated Ma=0.1 channel baseline so future changes to
// the low-Mach solver cannot silently break its only validated case.
//
// Acceptance: the test passes only if
//   - the solver GENUINELY converges under the full-residual criterion
//     (Ux, Uy, p, T, and the carried k/omega change norms; MaxIter fails),
//   - density and temperature stay positive everywhere,
//   - the computed Cf is within coarse-mesh headroom [0.3x, 3x] of Dean
//     (the 40x30 regression suite carries the quantitative guard),
//   - the maximum Mach number remains subsonic (< 0.8).
//
// Mesh is intentionally small (20x16) so the test runs in a few seconds even
// in Debug builds.  The former 16x12 grid is below the demonstrated resolution
// threshold for this Re=2.2e6 wall-resolved case and enters a turbulence
// fixed-point limit cycle after the startup nuT floor is released.  Tighter
// quantitative regression tests belong in a dedicated suite.

#include "Mesh.hpp"
#include "SSTModel.hpp"
#include "IdealGasEOS.hpp"
#include "CompressibleBCs.hpp"
#include "CompressibleSIMPLESolver.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>

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
    // Small mesh for fast turnaround; same physics as the CompressibleMain.cpp channel.
    const int nx = 20, ny = 16;
    const double Lx = 10.0, H = 1.0;

    IdealGasEOS eos;
    const double T_in = 300.0;
    const double p_ref = 101325.0;
    const double rho_in = eos.density(p_ref, T_in);
    const double mu_in = eos.viscosity(T_in);
    const double a_in = eos.soundSpeed(T_in);
    const double Ma_in = 0.1;
    const double Uin = Ma_in * a_in;
    const double nu_in = mu_in / rho_in;
    const double Re = rho_in * Uin * H / mu_in;

    Mesh mesh = Mesh::makeChannel2D(nx, ny, Lx, H, Re, 1.0);
    mesh.computeWallDistance();

    const double Tu = 0.05;
    const double kIn = 1.5 * (Uin * Tu) * (Uin * Tu);
    const double omIn = kIn / (nu_in * 100.0);

    auto bcs = CompressibleBoundaryConditions::channelDefaults(mesh, Uin, T_in, p_ref, kIn, omIn);

    SolverSettings settings;
    // budget sized for the complete convergence criterion (absolute Ux, Uy
    // and p thresholds plus dimensionless field-change gates on the velocity
    // field, T, and the carried k/omega norms; Uy has the slowest raw
    // residual): genuine convergence lands near iteration 3600 on this
    // coarse case, whereas a Ux/p-only criterion stops prematurely near 1500
    settings.maxIterations = 5000;
    settings.convergenceTol = 1e-3;
    settings.divergenceLimit = 1e10;
    settings.alphaU = 0.5;
    settings.alphaP = 0.2;
    settings.alphaT = 0.7;
    settings.alphaK = 0.4;
    settings.alphaOmega = 0.4;
    settings.innerIterations = 200;
    settings.innerTolerance = 1e-4;
    settings.turbStartIter = 50;
    settings.turbUpdateInterval = 2;
    settings.verbose = false;
    settings.reportInterval = 200;

    SSTModel sst;
    CompressibleSIMPLESolver solver(mesh, sst, bcs, eos, settings);
    CompressibleFlowFields fields(mesh);
    solver.initUniform(fields, Vec3(Uin, 0, 0), p_ref, T_in, kIn, omIn);

    const auto hist = solver.solve(fields);

    if (!hist.entries.empty()) {
        const auto &e = hist.entries.back();
        std::printf("  final residuals at %d: Ux=%.6g Uy=%.6g p=%.6g "
                    "T=%.6g k=%.6g omega=%.6g\n",
                    hist.finalIter, e.Ux, e.Uy, e.p, e.T, e.k, e.omega);
    }

    REQUIRE(!hist.diverged, "compressible solver diverged at Ma=0.1");
    // a smoke pass requires a GENUINE converged classification (MaxIter
    // does not pass), under the full-residual criterion incl. Uy, T and the
    // carried k/omega norms
    REQUIRE(hist.converged, "compressible solver did not converge at Ma=0.1");
    REQUIRE(!hist.entries.empty(), "no residual history recorded");
    // the temperature norm is genuinely recorded:
    // finite everywhere, and strictly positive somewhere during the transient
    // (a hardcoded zero would pass a plain >= 0 check)
    bool anyT = false;
    for (const auto &e : hist.entries) {
        REQUIRE(std::isfinite(e.T), "temperature norm must be finite");
        if (e.T > 0.0)
            anyT = true;
    }
    REQUIRE(anyT, "temperature norm was never positive; not actually recorded");

    // Positivity and EOS invariants. The stored pressure is mechanical; the
    // recovered thermodynamic pressure must be positive and must equal rho RT
    // on the RETURNED state, including a final iteration that updated k.
    double rho_min = 1e30, T_min = 1e30, p_min = 1e30;
    double eos_rel_max = 0.0;
    double Umax = 0.0, Tmax = 0.0;
    for (int ci = 0; ci < mesh.nCells(); ++ci) {
        rho_min = std::min(rho_min, fields.rho[ci]);
        T_min = std::min(T_min, fields.T[ci]);
        const double pThermo =
            fields.p[ci] - (2.0 / 3.0) * fields.rho[ci] * std::max(fields.k[ci], 0.0);
        p_min = std::min(p_min, pThermo);
        const double pEOS = fields.rho[ci] * eos.R * fields.T[ci];
        eos_rel_max =
            std::max(eos_rel_max, std::fabs(pThermo - pEOS) / std::max(std::fabs(pEOS), 1.0));
        Tmax = std::max(Tmax, fields.T[ci]);
        const double um =
            std::sqrt(fields.U[ci].x * fields.U[ci].x + fields.U[ci].y * fields.U[ci].y);
        Umax = std::max(Umax, um);
    }
    REQUIRE(rho_min > 0.0, "negative or zero density encountered");
    REQUIRE(T_min > 0.0, "negative or zero temperature encountered");
    REQUIRE(p_min > 0.0, "negative or zero thermodynamic pressure encountered");
    REQUIRE(eos_rel_max < 1e-12, "returned two-pressure state is not EOS-consistent");

    // The prescribed outlet is thermodynamic. Its stored mechanical boundary
    // value must use the final owner rho/k, not a state from before energy or
    // turbulence was updated.
    for (int pi = 0; pi < mesh.nPatches(); ++pi) {
        const Patch &pat = mesh.patch(pi);
        if (pat.type != "outlet")
            continue;
        for (FaceID fi : pat.faces) {
            int o = mesh.face(fi).owner;
            double pThermoFace =
                fields.p.bface(fi) - (2.0 / 3.0) * fields.rho[o] * std::max(fields.k[o], 0.0);
            REQUIRE(std::fabs(pThermoFace - p_ref) < 1e-12 * p_ref,
                    "outlet mechanical pressure is stale against final rho/k");
        }
    }

    const double Ma_max = Umax / eos.soundSpeed(Tmax);
    REQUIRE(Ma_max < 0.8, "max Mach exceeded subsonic regime; solver invalid");

    // Loose Cf check: integrate dp/dx and compare to the Dean correlation.
    double p_in_avg = 0, p_out_avg = 0;
    int n_in = 0, n_out = 0;
    for (int ci = 0; ci < mesh.nCells(); ++ci) {
        const double cx = mesh.cell(ci).center.x;
        if (cx < Lx / nx) {
            p_in_avg += fields.p[ci];
            ++n_in;
        }
        if (cx > Lx - Lx / nx) {
            p_out_avg += fields.p[ci];
            ++n_out;
        }
    }
    REQUIRE(n_in > 0 && n_out > 0, "could not sample inlet/outlet cell strips");
    p_in_avg /= n_in;
    p_out_avg /= n_out;
    const double dp_dx = (p_in_avg - p_out_avg) / Lx;
    const double tau_w = dp_dx * H / 2.0;
    const double Cf = 2.0 * tau_w / (rho_in * Uin * Uin);
    const double Re_D = Re * 2.0;
    const double Cf_dean = 0.073 * std::pow(Re_D, -0.25);

    std::printf("  compressible Ma=%.2f  Re=%.0f  iters=%d  status=%s\n", Ma_in, Re, hist.finalIter,
                hist.converged  ? "Converged"
                : hist.diverged ? "Diverged"
                                : "MaxIter");
    std::printf("  rho_min=%.4f  T_min=%.2f  p_min=%.0f  Ma_max=%.4f  "
                "Cf=%.5f  Cf_dean=%.5f\n",
                rho_min, T_min, p_min, Ma_max, Cf, Cf_dean);

    // a factor-of-ten band would be no guard at all. On this
    // deliberately coarse 20x16 developing channel the dp/dx-integral Cf at
    // genuine convergence measures 1.28x Dean. Sixteen cells across a
    // Re=2.2e6 layer under-resolve the log region and the integral spans the
    // entry length, so the smoke band is [0.3x, 3x]: it catches a broken
    // momentum/pressure path while the 40x30 station-sampled regression
    // suite carries the quantitative few-percent guard.
    REQUIRE(std::isfinite(Cf) && Cf > 0.0, "developed-flow Cf not measurable");
    const double ratio = Cf / Cf_dean;
    REQUIRE(ratio > 0.3 && ratio < 3.0,
            "Cf inconsistent with the Dean correlation beyond coarse-mesh headroom");

    std::printf("test_compressible_smoke: all checks passed\n");
    return 0;
}
