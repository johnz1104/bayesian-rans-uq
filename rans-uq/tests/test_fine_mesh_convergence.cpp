// Fine-mesh regression for the convergence criterion (slow: run with
// tests/tools/check.sh --all).
//
// Moving-wall Couette at ny=384 resolves a slow transient on which an
// iteration-0-normalized criterion can declare convergence prematurely.
// The explicit setup below uses a ForwardModel skin-friction observation
// and conv_tol 1e-4. A solve reporting convergence must lie within 0.5% of
// the tightly converged bottom-wall Re_tau of the same discrete problem.
// REF_RE_TAU below records the reference conditions.

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

namespace {

#define REQUIRE(cond, msg)                                                                         \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            std::fprintf(stderr, "FAIL [%s:%d] %s\n  required: %s\n", __FILE__, __LINE__, (msg),   \
                         #cond);                                                                   \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

// Discrete reference for the 4x384 outlet mesh below, with Rhie-Chow fluxes:
// bottom-wall Re_tau 176.7183 and top-wall Re_tau 176.7187 at conv_tol 1e-7
// with a wall-shear monitor. The solve required 157220 iterations and had
// contraction rate 0.999959. The reference below uses their mean. Wall-shear
// agreement also guards against a spurious streamwise pressure mode.
const double REF_RE_TAU = 176.7185;

} // namespace

int main() {
    const double tol = 1e-4;
    const double nu = 0.0003060182410114865, Uwall = 2.0, Ub = 1.0;
    Mesh mesh = Mesh::makeChannel2D(4, 384, 1.5, 2.0, Uwall / nu, 0.3);
    mesh.computeWallDistance();
    const double kIn = 1.5 * (Ub * 0.05) * (Ub * 0.05), omIn = kIn / (nu * 100.0);
    FlowBoundaryConditions bcs = FlowBoundaryConditions::couetteDefaults(mesh, Uwall, kIn, omIn);
    ObservationOperator obs;
    obs.addSkinFriction("bottom", Vec3(0.75, 0.0, 0.0), 6.0e-3, 1.0e-3, Ub);
    SolverSettings s;
    s.maxIterations = 120000;
    s.convergenceTol = tol;
    s.alphaU = 0.7;
    s.alphaP = 0.5;
    s.verbose = false;
    InferenceParameterSet ps = InferenceParameterSet::a1_betaStar();
    ForwardModel fm(mesh, ps, obs, bcs, nu, s, Vec3(Ub, 0.0, 0.0), 0.0, kIn, omIn);
    EvaluationResult r = fm.evaluate(ps.pack(SSTCoefficients{}));

    // Cf = tau_w / (0.5 Ub^2) and Re_tau = sqrt(tau_w) / nu (h = 1)
    double reTau = std::sqrt(0.5 * r.predictions[0]) / nu;
    std::printf("  Couette ny=384 forward model: %s after %d iterations, Re_tau %.4f "
                "(converged %.4f, %+.3f%%)\n",
                statusString(r.status), r.simpleIters, reTau, REF_RE_TAU,
                100.0 * (reTau / REF_RE_TAU - 1.0));
    REQUIRE(r.status == EvaluationStatus::Converged,
            "the fine-mesh Couette solve must converge within the iteration budget");
    REQUIRE(std::fabs(reTau / REF_RE_TAU - 1.0) < 0.005,
            "a converged fine-mesh solve must land within 0.5% of the converged Re_tau");
    std::printf("test_fine_mesh_convergence: all checks passed\n");
    return 0;
}
