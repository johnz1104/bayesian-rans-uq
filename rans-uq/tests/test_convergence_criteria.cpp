// A declared convergence must mean the discrete solution has stopped moving.
//
// The legacy SIMPLE criterion divides each residual by its value at
// iteration 0. That reference depends on the initial state, not on the
// problem: a cold start from a uniform field has a huge iteration-0 residual,
// so a fine mesh can appear converged while still on a slow transient. A
// restart from a converged state instead has a tiny initial residual, making
// the same normalized criterion too strict.
//
// These checks use force balance, a discrete Couette reference, and restart
// consistency to assess convergence independently of the initial residual:
//   1. periodic channel: the body force balances the integrated wall shear
//      (the cell residuals sum to f V - sum tau_w A);
//   2. moving-wall Couette through the forward model: the reported Cf matches
//      a tightly converged reference for the same discrete setup;
//   3. a solve started from its own converged state reports convergence.
// The tolerance on 1 and 2 is ten times the solver tolerance.

#include "BoundaryCondition.hpp"
#include "Field.hpp"
#include "FlowFields.hpp"
#include "ForwardModel.hpp"
#include "InferenceParameters.hpp"
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

// area-averaged molecular wall shear on one patch
double wallShear(const Mesh &mesh, const FlowFields &f, double nu, const std::string &patch) {
    const Patch &p = mesh.patch(mesh.patchByName(patch));
    double ta = 0.0, a = 0.0;
    for (FaceID fi : p.faces) {
        const Face &fc = mesh.face(fi);
        double du = f.U[fc.owner].x - f.U.bface(fi).x;
        ta += nu * std::fabs(du) / fc.delta * fc.area;
        a += fc.area;
    }
    return ta / a;
}

} // namespace

int main() {
    const double tol = 1e-4;

    // 1. Periodic body-force channel: global wall-force balance.
    {
        const int nx = 24, ny = 28;
        const double Lx = 3.0, H = 1.0, nu = 2.0e-4, fb = 5.0e-3;
        std::vector<double> xN(nx + 1), yB(nx + 1, 0.0);
        for (int i = 0; i <= nx; ++i)
            xN[i] = Lx * i / nx;
        Mesh mesh = Mesh::makeCurvedChannelPeriodic2D(xN, yB, H, ny, 2500.0, 1.0);
        mesh.computeWallDistance();
        const double kIn = 1e-4, omIn = 10.0;
        FlowBoundaryConditions bcs = FlowBoundaryConditions::channelDefaults(mesh, 1.0, kIn, omIn);
        SolverSettings s;
        s.maxIterations = 30000;
        s.convergenceTol = tol;
        s.alphaU = 0.5;
        s.alphaP = 0.3;
        s.verbose = false;
        s.bodyForce = Vec3(fb, 0.0, 0.0);
        SSTModel sst{SSTCoefficients{}};
        SIMPLESolver solver(mesh, sst, bcs, nu, s);
        FlowFields f(mesh);
        solver.initUniform(f, Vec3(0.3, 0.0, 0.0), 0.0, kIn, omIn);
        ConvergenceHistory h = solver.solve(f);
        REQUIRE(h.converged, "periodic channel must converge");
        double shear =
            (wallShear(mesh, f, nu, "bottom_wall") + wallShear(mesh, f, nu, "top_wall")) * Lx;
        double drive = fb * Lx * H;
        std::printf("  periodic channel: converged at %d, shear/drive - 1 = %.3e\n", h.finalIter,
                    shear / drive - 1.0);
        REQUIRE(std::fabs(shear / drive - 1.0) < 10.0 * tol,
                "declared convergence must balance body force and wall shear");

        // 3. restart from the converged state: must report convergence
        SIMPLESolver again(mesh, sst, bcs, nu, s);
        FlowFields g = f; // turbEstablished carries over
        ConvergenceHistory h2 = again.solve(g);
        double maxDu = 0.0;
        for (int ci = 0; ci < mesh.nCells(); ++ci)
            maxDu = std::max(maxDu, std::fabs(g.U[ci].x - f.U[ci].x));
        std::printf("  restart from converged state: %s at %d, max|dU| = %.3e\n",
                    h2.converged ? "converged" : (h2.diverged ? "diverged" : "not converged"),
                    h2.finalIter, maxDu);
        REQUIRE(h2.converged, "a solve started at its converged state must report convergence");
        REQUIRE(h2.finalIter < 500, "and must do so promptly");
    }

    // 2. Couette fixture: Re_tau 171, ny=96, conv_tol 1e-4, 12000-iteration cap.
    //    Reported Cf must match a tightly converged reference of this same
    //    discrete problem at conv_tol 1e-9: 2 tau_w = 2 * 2.905677e-3.
    //    That reference required 29480 iterations, with contraction rate
    //    0.99938 per iteration. A spurious streamwise pressure mode can split
    //    the two wall shears; Rhie-Chow face fluxes suppress that mode.
    {
        const double nu = 0.0003060182410114865, Uwall = 2.0, Ub = 1.0;
        const double cfRef = 2.0 * 2.905677e-3;
        Mesh mesh = Mesh::makeChannel2D(4, 96, 1.5, 2.0, Uwall / nu, 0.3);
        mesh.computeWallDistance();
        const double kIn = 1.5 * (Ub * 0.05) * (Ub * 0.05), omIn = kIn / (nu * 100.0);
        FlowBoundaryConditions bcs =
            FlowBoundaryConditions::couetteDefaults(mesh, Uwall, kIn, omIn);
        ObservationOperator obs;
        obs.addSkinFriction("bottom", Vec3(0.75, 0.0, 0.0), 6.0e-3, 1.0e-3, Ub);
        SolverSettings s;
        s.maxIterations = 12000;
        s.convergenceTol = tol;
        s.alphaU = 0.7;
        s.alphaP = 0.5;
        s.verbose = false;
        InferenceParameterSet ps = InferenceParameterSet::a1_betaStar();
        ForwardModel fm(mesh, ps, obs, bcs, nu, s, Vec3(Ub, 0.0, 0.0), 0.0, kIn, omIn);
        EvaluationResult r = fm.evaluate(ps.pack(SSTCoefficients{}));
        std::printf("  Couette ny=96 forward model: %s after %d iterations, Cf %.6e "
                    "(converged %.6e, %+.3e)\n",
                    statusString(r.status), r.simpleIters, r.predictions[0], cfRef,
                    r.predictions[0] / cfRef - 1.0);
        REQUIRE(r.status == EvaluationStatus::Converged,
                "the production Couette solve must converge");
        REQUIRE(std::fabs(r.predictions[0] / cfRef - 1.0) < 10.0 * tol,
                "the Cf reported as converged must match the converged discrete Cf");
    }

    std::printf("test_convergence_criteria: all checks passed\n");
    return 0;
}
