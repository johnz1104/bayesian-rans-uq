#pragma once

#include "Mesh.hpp"
#include "Field.hpp"
#include <vector>
#include <string>
#include <cmath>
#include <limits>
#include <stdexcept>

// Primary solved flow variables and all turbulence model fields, stored cell-centred.
struct FlowFields {
    VectorField U;     // velocity
    ScalarField p;     // pressure
    ScalarField k;     // turbulent kinetic energy
    ScalarField omega; // specific dissipation rate

    ScalarField nuT;    // eddy viscosity
    ScalarField F1, F2; // SST blending functions
    ScalarField Pk;     // production of k
    ScalarField CDkw;   // cross-diffusion term

    // Turbulence-establishment marker carried WITH the state: false after a
    // cold (uniform) init, set true once a solve passes the startup floor
    // window. The startup-only nuT floor consults this so a WARM restart
    // (fields from a converged cache) never re-engages the floor at iter 0,
    // which would otherwise contaminate warm-FD gradients and warm-started
    // ensembles with a floor the base state does not carry.
    bool turbEstablished = false;

    FlowFields() = default;
    explicit FlowFields(const Mesh &mesh)
        : U(mesh, "U"), p(mesh, "p"), k(mesh, "k"), omega(mesh, "omega"), nuT(mesh, "nuT"),
          F1(mesh, "F1"), F2(mesh, "F2"), Pk(mesh, "Pk"), CDkw(mesh, "CDkw") {}
};

// Per-iteration residual norms for all solved equations.
// Ux..p are the raw linear-system residual norms; k and omega are relative
// field changes. The scaled and change fields below are what the default
// convergence criterion judges (see SolverSettings::legacyConvergence).
struct ResidualEntry {
    int iteration = 0;
    double Ux = 0, Uy = 0, Uz = 0;
    double p = 0;
    double k = 0, omega = 0;
    double momentumScaled = 0;   // ||b - A U||_1 / sum_i a_P,i |U_i|, worst component
    double continuityScaled = 0; // sum_cells |net mass flux| / sum_faces |mass flux|
    double changeU = 0;          // max_i |U_i - U_i,old| / max_i |U_i|
    double remainingU = 0;       // extrapolated remaining change of U (inf if not contracting)
};

// Full residual history from one SIMPLE solve.
struct ConvergenceHistory {
    std::vector<ResidualEntry> entries;
    bool converged = false;
    bool diverged = false;
    // maxIterations reached while the measured contraction rate was >= 1:
    // the iteration had stopped making progress, not merely run out of time
    bool stalled = false;
    int finalIter = 0;
    // at exit: the largest of the judged quantities (scaled residuals, the
    // relative changes and their extrapolated remainders), and the
    // per-iteration contraction rate of U measured over the last window
    double finalWorstMeasure = 0;
    double contractionRateU = 0;
};

// Configuration for the SIMPLE solver: iteration limits, tolerances,
// relaxation factors, linear solver names, and turbulence scheduling.
struct SolverSettings {
    int maxIterations = 500;
    double convergenceTol = 1e-5;
    double divergenceLimit = 1e6;

    // Convergence criterion. By default a solve is converged only when
    //   (a) the momentum and continuity residuals, scaled by magnitudes of the
    //       current state (not by their iteration-0 values, which depend on
    //       the initial guess), are below convergenceTol, and
    //   (b) the relative per-iteration change of U, k, omega, and of the
    //       monitored QoIs when a monitor is set, is below convergenceTol,
    //       and so is the remaining change extrapolated from the contraction
    //       rate measured over the last convergenceWindow samples.
    // (b) is what stops a fine mesh declaring convergence on a slow
    // transient: there the per-iteration change is tiny but the contraction
    // rate is close to 1, so the extrapolated remainder is not.
    // legacyConvergence = true restores the iteration-0 normalisation, only
    // to reproduce historical numbers.
    bool legacyConvergence = false;
    int convergenceWindow = 25; // samples per contraction-rate window
    int monitorInterval = 10;   // iterations between QoI monitor samples

    double alphaU = 0.7;
    double alphaP = 0.3;
    double alphaK = 0.5;
    double alphaOmega = 0.5;
    double alphaT = 0.7; // temperature under-relaxation (compressible solver)

    int innerIterations = 200;
    double innerTolerance = 1e-3;

    int turbStartIter = 5;
    int turbUpdateInterval = 1;

    // Startup-only eddy-viscosity floor window: the 0.1*nu floor that guards
    // against early k-omega collapse stays active only while
    // iter < turbStartIter + nuTFloorIters, then releases to a plain
    // non-negativity clamp. SST requires nuT -> 0 at a resolved wall, so a
    // permanent floor would bias the converged near-wall solution and the
    // wall stress (about +10 percent where it binds); the converged state
    // must be floor-free. Widen per-case in config if a startup needs it.
    int nuTFloorIters = 500;

    // Rhie-Chow face fluxes. By default every mesh uses them: the face flux
    // is predicted with the Rhie-Chow pressure term plus the
    // under-relaxation correction (1 - alphaU)(phi_old - U_f,old . S), which
    // makes the converged flux, and so the solution, independent of alphaU;
    // it is corrected with the pressure-correction coefficients, so it is
    // discretely divergence-free, and U, k and omega are convected with it.
    // legacyRhieChowGate = true restores the old discretisation exactly:
    // the term only on outlet-free meshes, with the relaxed coefficient and
    // no correction, and convection with interpolated cell velocities. This
    // option is retained for legacy reproduction: outlet meshes can retain a
    // spurious streamwise pressure mode, and periodic solutions can depend
    // on alphaU even after convergence.
    bool legacyRhieChowGate = false;

    // Historical solve-then-overwrite omega changes neighboring transport
    // equations and leaves extrapolated corner faces stale. Keep it only for
    // source-bound reproduction; new solves constrain wall cells in the matrix.
    // Steady sensitivities require the constrained wall treatment.
    bool legacyOmegaWallPinning = false;

    // Convection schemes, applied as a deferred correction on the implicit
    // first-order upwind matrix (so the matrix and its diagonal dominance are
    // unchanged and only the converged answer moves):
    //   "upwind"              first order; the scheme the solver always had
    //   "linearUpwind"        phi_f = phi_U + grad(phi)_U . (x_f - x_U):
    //                         second order, unlimited and smooth in the state,
    //                         so matrix-free tangents stay differentiable
    //   "limitedLinearUpwind" phi_f = phi_U + psi(r)/2 (phi_D - phi_U) with the
    //                         van Albada limiter: second order where smooth,
    //                         bounded, so k and omega cannot be driven negative
    // First-order upwind truncation error can dominate separated-flow comparisons.
    // Production solves default to the bounded scheme for every equation:
    // unlimited linear upwind for U does not settle on coarse separated flows
    // (on the cosine-bump channel it oscillates at 1-3% per iteration even
    // at alphaU 0.2); use it where the flow is smooth or a differentiable
    // operator matters (matrix-free tangents).
    std::string momentumConvection = "limitedLinearUpwind";
    std::string turbulenceConvection = "limitedLinearUpwind";

    // Under-relaxation of the explicit Reynolds-stress-injection source (the
    // deferred-correction body force is blended across outer iterations at
    // this factor; 1.0 disables blending). The explicit correction cancels a
    // first-order part of the implicit nuT stabilizer, so unrelaxed feedback
    // leaves a residual floor near the tolerance.
    double alphaInjection = 0.3;

    // Constant body force per unit volume (momentum source f_i * V per cell).
    // A streamwise-PERIODIC domain has no inlet to drive it: the mean pressure
    // gradient is represented by this force (e.g. bodyForce.x tuned so the
    // solved bulk velocity matches the target Reynolds number).
    Vec3 bodyForce = Vec3(0.0, 0.0, 0.0);

    double kMin = 1e-10;
    double kMax = 1e10;
    double omegaMin = 1e-6;

    int reportInterval = 10;
    bool verbose = true;

    std::string pressureSolver = "AMG";
    std::string momentumSolver = "BiCGSTAB";
    std::string turbulenceSolver = "BiCGSTAB";

    // Wall treatment. Resolved SST walls require near-wall resolution.
    // Incompressible SIMPLE uses ω = 6ν/(β1 y²) with a singular diffusion
    // balance on qualifying orthogonal periodic channels; other geometries
    // and the low-Mach solver retain the coefficient 60.
    // Wall-function mode blends the Menter value ω_res = 60ν/(β1 y²) with
    // the log-law form
    //   ω_log = u_τ / (√β* κ y)
    // via   ω_w = √(ω_res² + ω_log²)
    // Coarse-mesh accuracy requires verification for the chosen case.
    bool useWallFunctions = false;
    double vonKarman = 0.41; // κ
    double wallFnE = 9.0;    // log-law coefficient

    // Closure selection.
    // Maps to SSTVariant: 0 = Full (standard SST), 1 = NoLimiter (drop Bradshaw
    // shear-stress limiter), 2 = KOmega (force F1=1, baseline k-w).  Stored as int
    // to keep FlowFields.hpp free of an SSTModel.hpp dependency.
    int sstVariant = 0;

    // Both SIMPLE drivers use these counters and denominators. Validate at
    // construction, including settings exposed directly through pybind11.
    void validate() const {
        if (sstVariant < 0 || sstVariant > 2)
            throw std::invalid_argument(
                "unknown SST variant: expected Full (0), NoLimiter (1), or KOmega (2)");
        if (maxIterations < 0 || innerIterations <= 0 || convergenceWindow < 2 ||
            monitorInterval <= 0 || reportInterval <= 0 || turbUpdateInterval <= 0 ||
            turbStartIter < 0 || nuTFloorIters < 0 ||
            turbStartIter > std::numeric_limits<int>::max() - nuTFloorIters)
            throw std::invalid_argument("invalid SIMPLE iteration count or cadence");
        if (!std::isfinite(divergenceLimit) || divergenceLimit < 0.0)
            throw std::invalid_argument("SIMPLE divergence limit must be finite and nonnegative");
        // A zero threshold requests fixed iterations; the strict convergence
        // comparisons cannot certify any nonnegative residual against it.
        if (!std::isfinite(convergenceTol) || convergenceTol < 0.0)
            throw std::invalid_argument(
                "SIMPLE convergence tolerance must be finite and nonnegative");
        for (double value : {innerTolerance, alphaU, alphaK, alphaOmega, alphaT})
            if (!std::isfinite(value) || value <= 0.0)
                throw std::invalid_argument(
                    "SIMPLE tolerances and relaxation denominators must be finite and positive");
        // Zero disables these updates; values above one are explicit
        // extrapolation choices, not silently clipped under-relaxation.
        for (double value : {alphaP, alphaInjection})
            if (!std::isfinite(value) || value < 0.)
                throw std::invalid_argument(
                    "SIMPLE pressure/source relaxation must be finite and nonnegative");
        if (!std::isfinite(bodyForce.x) || !std::isfinite(bodyForce.y) ||
            !std::isfinite(bodyForce.z))
            throw std::invalid_argument("SIMPLE body force must be finite");
        for (double value : {kMin, kMax, omegaMin})
            if (!std::isfinite(value) || value < 0.)
                throw std::invalid_argument(
                    "SIMPLE turbulence limits must be finite and nonnegative");
        if (kMax < kMin)
            throw std::invalid_argument("SIMPLE kMax must be at least kMin");
        for (double value : {vonKarman, wallFnE})
            if (!std::isfinite(value) || value <= 0.)
                throw std::invalid_argument(
                    "SIMPLE wall-function constants must be finite and positive");
        for (const auto *scheme : {&momentumConvection, &turbulenceConvection})
            if (*scheme != "upwind" && *scheme != "linearUpwind" &&
                *scheme != "limitedLinearUpwind")
                throw std::invalid_argument("unknown SIMPLE convection scheme: " + *scheme);
    }
};
