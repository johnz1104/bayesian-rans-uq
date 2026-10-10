// Verification ladder (part 1) for the density-based solver: cases with an
// analytic reference, run as a self-checking program (exit non-zero on failure).
//
//   1. Sod shock tube      -> match the exact Riemann solution (Toro).
//   2. Lax strong shock    -> match the exact Riemann solution, stay positive.
//   3. Method of manufactured solutions (inviscid Euler) -> recover the design
//      order of accuracy of the HLLC + MUSCL convective discretization.
//
// All cases run on rectangular makeChannel2D meshes (no new mesh factory), so
// the frozen core Mesh is untouched.

#include "Mesh.hpp"
#include "DBNSSolver.hpp"
#include "ExactRiemann.hpp"
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

// Run a 1D Riemann problem on an nx x 1 strip and return the L1 error in
// (rho, u, p) against the exact self-similar solution at time tEnd.
static void run_shock_tube(const char *name, RiemannState1D L, RiemannState1D R, double x0,
                           double tEnd, double nx, double L1tol) {
    IdealGasEOS eos;
    eos.gamma = 1.4;
    double Lx = 1.0, Ly = 0.02;
    Mesh mesh = Mesh::makeChannel2D((int)nx, 1, Lx, Ly);
    mesh.computeGeometry();

    SSTCoefficients sst;
    DBNSBoundaryConditions bcs;
    BoundarySpec ext;
    ext.kind = BoundaryKind::Extrapolate;
    BoundarySpec slip;
    slip.kind = BoundaryKind::SlipWall;
    bcs.set("inlet", ext);
    bcs.set("outlet", ext);
    bcs.set("top", slip);
    bcs.set("bottom", slip);

    DBNSSettings st;
    st.timeMode = TimeMode::Unsteady;
    st.tEnd = tEnd;
    st.cfl = 0.4;
    st.viscous = false;
    st.turbulent = false;
    st.reconstructOrder = 2;
    st.limitReconstruction = true;
    st.maxIterations = 100000;

    DBNSSolver solver(mesh, eos, sst, bcs, st);
    // initial condition: left/right of the diaphragm
    std::vector<Primitive> init(mesh.nCells());
    for (int ci = 0; ci < mesh.nCells(); ++ci) {
        double x = mesh.cell(ci).center.x;
        const RiemannState1D &s = (x < x0) ? L : R;
        init[ci] = {s.rho, s.u, 0.0, s.p, 0.0, 0.0};
    }
    solver.initField(init);
    SolveReport rep = solver.solve();
    REQUIRE(rep.status == EvaluationStatus::Converged, "shock tube did not finish");

    ExactRiemann ex(1.4);
    double e_rho = 0, e_u = 0, e_p = 0;
    int n = mesh.nCells();
    for (int ci = 0; ci < n; ++ci) {
        double x = mesh.cell(ci).center.x;
        RiemannState1D exact = ex.sample(L, R, (x - x0) / tEnd);
        Primitive V = solver.primitive(ci);
        e_rho += std::abs(V.rho - exact.rho);
        e_u += std::abs(V.u - exact.u);
        e_p += std::abs(V.p - exact.p);
        REQUIRE(V.rho > 0.0 && V.p > 0.0, "shock tube produced a non-physical state");
    }
    e_rho /= n;
    e_u /= n;
    e_p /= n;
    std::printf("  [%s] L1 errors  rho=%.4f  u=%.4f  p=%.4f  (iters=%d)\n", name, e_rho, e_u, e_p,
                rep.iterations);
    REQUIRE(e_rho < L1tol, "shock-tube density L1 error too large");
    REQUIRE(e_p < L1tol, "shock-tube pressure L1 error too large");
}

// ---- Method of manufactured solutions (inviscid Euler) --------------------
// Manufactured steady solution: constant velocity, sinusoidal rho and p.
//   theta = kx*x + ky*y
//   rho = rho0 (1 + A sin theta),  p = p0 (1 + A sin theta),  u=u0, v=v0
struct MMSField {
    double rho0 = 1.2, p0 = 1.0e5, u0 = 60.0, v0 = 40.0, A = 0.2;
    double kx, ky, gamma = 1.4;
    MMSField(double Lx, double Ly) : kx(2.0 * M_PI / Lx), ky(2.0 * M_PI / Ly) {}

    Primitive exact(double x, double y) const {
        double s = std::sin(kx * x + ky * y);
        return {rho0 * (1 + A * s), u0, v0, p0 * (1 + A * s), 0.0, 0.0};
    }
    // Analytic source S = div(F_exact) of the steady Euler flux at (x,y).
    StateVec source(double x, double y) const {
        double c = std::cos(kx * x + ky * y);
        double drdx = rho0 * A * kx * c, drdy = rho0 * A * ky * c; // d rho
        double dpdx = p0 * A * kx * c, dpdy = p0 * A * ky * c;     // d p
        double u = u0, v = v0, q2 = u * u + v * v;
        double g = gamma / (gamma - 1.0);
        StateVec S{};
        S[I_RHO] = u * drdx + v * drdy;
        S[I_RHOU] = u * u * drdx + dpdx + u * v * drdy;
        S[I_RHOV] = u * v * drdx + v * v * drdy + dpdy;
        // energy: d/dx[(rhoE+p)u] + d/dy[(rhoE+p)v], rhoE+p = g p + 0.5 rho q2
        S[I_RHOE] = u * (g * dpdx + 0.5 * q2 * drdx) + v * (g * dpdy + 0.5 * q2 * drdy);
        S[I_RHOK] = 0.0;
        S[I_RHOW] = 0.0;
        return S;
    }
};

static double run_mms(int nxy, int order) {
    IdealGasEOS eos;
    eos.gamma = 1.4;
    double Lx = 1.0, Ly = 1.0;
    Mesh mesh = Mesh::makeChannel2D(nxy, nxy, Lx, Ly);
    mesh.computeGeometry();
    MMSField mms(Lx, Ly);

    SSTCoefficients sst;
    DBNSBoundaryConditions bcs;
    BoundarySpec fixed;
    fixed.kind = BoundaryKind::FixedState;
    bcs.set("inlet", fixed);
    bcs.set("outlet", fixed);
    bcs.set("top", fixed);
    bcs.set("bottom", fixed);

    DBNSSettings st;
    st.timeMode = TimeMode::Steady;
    st.cfl = 0.6;
    st.viscous = false;
    st.turbulent = false;
    st.reconstructOrder = order;
    st.limitReconstruction = false; // unlimited: measure smooth design order
    st.convergenceTol = 1e-9;
    st.maxIterations = 30000;

    DBNSSolver solver(mesh, eos, sst, bcs, st);
    std::vector<Primitive> init(mesh.nCells());
    std::vector<StateVec> src(mesh.nCells());
    for (int ci = 0; ci < mesh.nCells(); ++ci) {
        const Vec3 &c = mesh.cell(ci).center;
        init[ci] = mms.exact(c.x, c.y); // start from the exact field
        src[ci] = mms.source(c.x, c.y);
    }
    solver.initField(init);
    solver.setManufacturedSource(src);

    // Dirichlet boundary: exact solution at each boundary-face centre.
    int nIF = mesh.nInternalFaces();
    std::vector<Primitive> ghost(mesh.nFaces() - nIF);
    for (int fi = nIF; fi < mesh.nFaces(); ++fi) {
        const Vec3 &fc = mesh.face(fi).center;
        ghost[fi - nIF] = mms.exact(fc.x, fc.y);
    }
    solver.setBoundaryOverride(ghost);

    // perturb the interior away from exact so convergence is a real test
    std::vector<Primitive> pert = init;
    for (int ci = 0; ci < mesh.nCells(); ++ci) {
        pert[ci].rho *= 1.05;
        pert[ci].p *= 0.97;
    }
    solver.initField(pert);

    SolveReport rep = solver.solve();
    REQUIRE(rep.status == EvaluationStatus::Converged, "MMS did not reach steady convergence");
    REQUIRE(std::isfinite(rep.finalResidual) && rep.finalResidual <= st.convergenceTol,
            "MMS final joint residual exceeds its declared tolerance");
    std::printf("  [MMS %d order=%d] steady steps %d residual %.3e\n", nxy, order, rep.iterations,
                rep.finalResidual);

    double err = 0.0;
    for (int ci = 0; ci < mesh.nCells(); ++ci) {
        const Vec3 &c = mesh.cell(ci).center;
        Primitive ex = mms.exact(c.x, c.y);
        Primitive V = solver.primitive(ci);
        double d = V.rho - ex.rho;
        err += d * d * mesh.cell(ci).volume;
    }
    return std::sqrt(err); // L2 density error (volume-weighted)
}

static void test_mms_order() {
    // second-order reconstruction: observed order should approach 2.
    double e1 = run_mms(16, 2);
    double e2 = run_mms(32, 2);
    double e3 = run_mms(64, 2);
    double p1 = std::log(e1 / e2) / std::log(2.0);
    double p2 = std::log(e2 / e3) / std::log(2.0);
    std::printf("  [MMS order=2] errors %.3e %.3e %.3e  observed order %.2f %.2f\n", e1, e2, e3, p1,
                p2);
    REQUIRE(p2 > 1.6, "second-order MMS did not reach design order");

    // first-order reconstruction: observed order should be near 1, and clearly
    // below the second-order result (the discretizations are distinct).
    double f1 = run_mms(16, 1);
    double f2 = run_mms(32, 1);
    double q1 = std::log(f1 / f2) / std::log(2.0);
    std::printf("  [MMS order=1] errors %.3e %.3e  observed order %.2f\n", f1, f2, q1);
    REQUIRE(q1 > 0.7 && q1 < 1.5, "first-order MMS order out of expected range");
}

int main() {
    // Sod problem (Toro): tolerance reflects 200-cell TVD resolution.
    run_shock_tube("Sod", {1.0, 0.0, 1.0}, {0.125, 0.0, 0.1}, 0.5, 0.2, 200, 0.02);
    // Lax problem: stronger shock and contact.
    run_shock_tube("Lax", {0.445, 0.698, 3.528}, {0.5, 0.0, 0.571}, 0.5, 0.13, 200, 0.08);
    test_mms_order();
    std::printf("test_dbns_verification: all checks passed\n");
    return 0;
}
