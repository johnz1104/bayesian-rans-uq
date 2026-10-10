// The canonical channel for fully developed comparisons: periodic, body-force
// driven, at a prescribed Re_tau.
//
// An inlet/outlet channel includes entrance effects. This periodic,
// body-force-driven fixture instead enforces a global wall-force balance
// at a prescribed friction Reynolds number.
//
// In u_tau = 1 units (nu = h / Re_tau, body force 1/h) the global balance
// f V = integral of wall shear fixes Re_tau exactly. This test pins, on
// makePeriodicChannel2D at the Lee-Moser Re_tau 182.088:
//   1. the converged wall shear is 1 on both walls (Re_tau as prescribed),
//      to ten times the solver tolerance;
//   2. the observation operator resolves "bottom" and returns, through the
//      interpolating probes, the viscous-sublayer profile U+ = y+ at the
//      first DNS station and a log-region U+ near y+ = 100;
//   3. the flow is streamwise invariant.

#include "BoundaryCondition.hpp"
#include "Field.hpp"
#include "FlowFields.hpp"
#include "Mesh.hpp"
#include "ObservationOperator.hpp"
#include "SIMPLESolver.hpp"
#include "SSTModel.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
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

double wallShear(const Mesh &mesh, const FlowFields &f, double nu, const std::string &patch) {
    const Patch &p = mesh.patch(mesh.patchByName(patch));
    double ta = 0.0, a = 0.0;
    for (FaceID fi : p.faces) {
        const Face &fc = mesh.face(fi);
        ta += nu * std::fabs(f.U[fc.owner].x - f.U.bface(fi).x) / fc.delta * fc.area;
        a += fc.area;
    }
    return ta / a;
}

} // namespace

int main() {
    const double reTau = 182.088, h = 1.0, nu = h / reTau, tol = 1e-6;
    const int nx = 4, ny = 56;
    Mesh mesh = Mesh::makePeriodicChannel2D(nx, ny, 1.0, h, reTau, 0.5);
    mesh.computeWallDistance();
    const double kIn = 1.0, omIn = 10.0;
    FlowBoundaryConditions bcs = FlowBoundaryConditions::channelDefaults(mesh, 1.0, kIn, omIn);
    SolverSettings s;
    s.maxIterations = 60000;
    s.convergenceTol = tol;
    s.alphaU = 0.5;
    s.alphaP = 0.3;
    s.verbose = false;
    s.bodyForce = Vec3(1.0 / h, 0.0, 0.0);
    SSTModel sst{SSTCoefficients{}};
    SIMPLESolver solver(mesh, sst, bcs, nu, s);
    solver.setConvergenceMonitor([&](const FlowFields &g) {
        return std::vector<double>{wallShear(mesh, g, nu, "bottom"), wallShear(mesh, g, nu, "top")};
    });
    FlowFields f(mesh);
    solver.initUniform(f, Vec3(15.0, 0.0, 0.0), 0.0, kIn, omIn);
    ConvergenceHistory hist = solver.solve(f);
    REQUIRE(hist.converged, "the periodic channel must converge");

    // 1. Re_tau as prescribed
    double tb = wallShear(mesh, f, nu, "bottom"), tt = wallShear(mesh, f, nu, "top");
    std::printf("  converged at %d: tau_w bottom %.8f top %.8f, Re_tau %.4f (prescribed %.4f)\n",
                hist.finalIter, tb, tt, std::sqrt(tb) * h / nu, reTau);
    REQUIRE(std::fabs(0.5 * (tb + tt) - 1.0) < 10.0 * tol,
            "body force and wall shear must balance");
    REQUIRE(std::fabs(tb / tt - 1.0) < 10.0 * tol, "both walls carry the same shear");

    // 2. observations: "bottom" resolves, probes see the sublayer and log region
    ObservationOperator obs;
    obs.addSkinFriction("bottom", Vec3(0.5, 0.0, 0.0), 0.0, 1.0, 1.0);
    const double yd1 = 0.0054122475544012838, yd100 = 100.0 / reTau;
    obs.addVelocityProfile(Vec3(0.5, yd1 * h, 0.0), 0, 0.0, 1.0);
    obs.addVelocityProfile(Vec3(0.5, yd100 * h, 0.0), 0, 0.0, 1.0);
    std::vector<double> pred = obs.evaluate(mesh, f, nu);
    double yplus1 = yd1 * reTau;
    std::printf("  Cf(U_ref = u_tau) %.6f; U+ at y+ %.3f: %.4f; U+ at y+ 100: %.3f\n", pred[0],
                yplus1, pred[1], pred[2]);
    REQUIRE(std::fabs(pred[0] - 2.0 * tb) < 1e-12, "Cf on the canonical channel's bottom wall");
    REQUIRE(std::fabs(pred[1] / yplus1 - 1.0) < 0.02,
            "viscous sublayer: U+ = y+ at the first station");
    REQUIRE(pred[2] > 14.0 && pred[2] < 19.0, "log region near y+ = 100");

    // 3. streamwise invariance
    double maxDiff = 0.0;
    for (int j = 0; j < ny; ++j)
        for (int i = 1; i < nx; ++i)
            maxDiff = std::max(maxDiff, std::fabs(f.U[j * nx + i].x - f.U[j * nx].x));
    REQUIRE(maxDiff < 1e-8 * 15.0, "the solution must be streamwise invariant");

    std::printf("test_periodic_channel_retau: all checks passed\n");
    return 0;
}
