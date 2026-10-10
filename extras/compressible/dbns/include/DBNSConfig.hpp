#pragma once

#include "Mesh.hpp" // Vec3
#include "DBNSTypes.hpp"
#include "Limiters.hpp"
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------
// Configuration for the density-based shock-capturing solver: numerical
// settings and boundary-condition specification.
// ---------------------------------------------------------------------------

namespace dbns {

// Pseudo-time integration mode.
//   Steady:   local time stepping, iterate to residual convergence.
//   Unsteady: global (min) time step, march to settings.tEnd (e.g. shock tube).
enum class TimeMode { Steady, Unsteady };

// Compressibility (dilatational dissipation) correction model.
//   None:   incompressible SST dissipation.
//   Sarkar: eps_total = eps_s (1 + xi* M_t^2)              (Sarkar 1991)
//   Zeman:  eps_total = eps_s (1 + xi* F(M_t)), M_t0 cutoff (Wilcox/Zeman form)
enum class CompressibilityModel { None, Sarkar, Zeman };

struct DBNSSettings {
    TimeMode timeMode = TimeMode::Steady;
    double cfl = 0.5;             // Courant number (explicit stability)
    int maxIterations = 20000;    // steady iteration cap
    double tEnd = 0.2;            // unsteady end time
    double convergenceTol = 1e-6; // explicit: max scaled steady residual; implicit: relative decay
    double divergenceLimit = 1e12;

    // Spatial reconstruction: 1 = first order (constant), 2 = MUSCL linear.
    // limitReconstruction toggles the sharp Barth-Jespersen gradient limiter;
    // turn it OFF only for smooth MMS order-of-accuracy verification.
    int reconstructOrder = 2;
    bool limitReconstruction = true;
    double venkatK = 5.0; // reserved legacy field; the active sharp limiter does not use it

    bool viscous = true;    // include viscous + heat fluxes
    bool turbulent = false; // integrate SST k-omega transport
    double constMu = -1.0;  // if >= 0, use this constant dynamic viscosity
                            // (verification: constant mu and lambda)
    CompressibilityModel compressibility = CompressibilityModel::None;
    double turbMachCutoff = 0.25; // M_t0 for the Zeman form
    double comprXiStar = 1.0;     // xi* scaling of dilatational dissipation

    int rkStages = 3; // SSP-RK stages (3 = SSP-RK3)
    int reportInterval = 1000;
    // fail-fast for warm perturbation solves: when > 0, a solve whose joint
    // convergence measure still exceeds earlyAbortRelMax at earlyAbortIter
    // returns Unconverged immediately instead of burning the full budget
    int earlyAbortIter = 0;
    double earlyAbortRelMax = 0.0;
    // injected-correction ramp: the deferred-correction fluxes scale by
    // min(1, iter/injectionRampIters) so a warm perturbation solve reaches
    // the full force gradually; 0 applies the full force from iteration one
    int injectionRampIters = 0;
    // frozen-k correction scale: the injected stress difference uses the
    // rho k captured when the targets were set (the baseline state) instead
    // of the running rho k, removing the runaway feedback in which the
    // force grows with the k it produces; default keeps the registered
    // running-k form
    bool injectionFrozenK = false;
    // frozen-mean transport mode (the registered gate-B fallback baseline):
    // the mean-flow primitive state (rho, u, v, p) captured at solve start
    // is held fixed every iteration and only the k and omega transport
    // equations march to steadiness, the constitutive-error-at-matched-mean
    // convention. Implicit steady driver only; convergence gates on the
    // turbulence field-change norms alone.
    bool frozenMeanFlow = false;
    bool verbose = false;

    double kFloor = 1e-12; // turbulence positivity floors
    double omegaFloor = 1e-3;

    // Implicit (LU-SGS) steady driver: matrix-free backward-Euler pseudo-time
    // with lower-upper symmetric Gauss-Seidel sweeps (Yoon and Jameson 1988).
    // The fix for the explicit march's viscous stiffness: the viscous spectral
    // radius scales as 1/dy^2 on wall-clustered meshes, so explicit local time
    // stepping cannot converge viscous-dominated steady states (documented in
    // the verification suite); the implicit diagonal absorbs that stiffness.
    // Only the Steady time mode consults these.
    bool implicitSteady = false;
    // Historical acceleration freezes the reconstruction after a residual
    // drop, changing the nonlinear operator. Opt in only for reproduction.
    bool legacyAutoFreezeLimiter = false;
    double cflImplicit = 100.0; // target implicit CFL (orders above explicit)
    double cflRampStart = 2.0;  // CFL at the first iteration
    int cflRampIters = 200;     // linear ramp length to cflImplicit

    void validate() const {
        auto positive = [](double value, const char *name) {
            if (!std::isfinite(value) || value <= 0)
                throw std::invalid_argument(std::string("DBNS ") + name +
                                            " must be finite and positive");
        };
        auto nonnegative = [](double value, const char *name) {
            if (!std::isfinite(value) || value < 0)
                throw std::invalid_argument(std::string("DBNS ") + name +
                                            " must be finite and nonnegative");
        };
        positive(cfl, "cfl");
        positive(kFloor, "kFloor");
        positive(omegaFloor, "omegaFloor");
        positive(cflImplicit, "cflImplicit");
        positive(cflRampStart, "cflRampStart");
        nonnegative(tEnd, "tEnd");
        nonnegative(convergenceTol, "convergenceTol");
        nonnegative(divergenceLimit, "divergenceLimit");
        nonnegative(turbMachCutoff, "turbMachCutoff");
        nonnegative(comprXiStar, "comprXiStar");
        nonnegative(earlyAbortRelMax, "earlyAbortRelMax");
        if (!std::isfinite(constMu))
            throw std::invalid_argument(
                "DBNS constMu must be finite; a negative value selects Sutherland");
        if (maxIterations < 0 || earlyAbortIter < 0 || injectionRampIters < 0 || cflRampIters < 0)
            throw std::invalid_argument("DBNS iteration counts must be nonnegative");
        if (reportInterval <= 0)
            throw std::invalid_argument("DBNS reportInterval must be positive");
        if (reconstructOrder != 1 && reconstructOrder != 2)
            throw std::invalid_argument("DBNS reconstructOrder must be 1 or 2");
        if (rkStages != 1 && rkStages != 3)
            throw std::invalid_argument("DBNS rkStages must be 1 (Euler) or 3 (SSPRK3)");
        if (timeMode != TimeMode::Steady && timeMode != TimeMode::Unsteady)
            throw std::invalid_argument("DBNS timeMode is not recognized");
        if (compressibility != CompressibilityModel::None &&
            compressibility != CompressibilityModel::Sarkar &&
            compressibility != CompressibilityModel::Zeman)
            throw std::invalid_argument("DBNS compressibility model is not recognized");
    }
};

// Boundary-condition kinds.  Wall thermal condition is split into adiabatic and
// isothermal as the task requires.
enum class BoundaryKind {
    SupersonicInflow, // all variables prescribed from the freestream
    Extrapolate,      // zero-gradient (supersonic outflow / far-field exit)
    SubsonicInflow,   // prescribe velocity + temperature, extrapolate pressure
    SubsonicOutflow,  // prescribe back pressure, extrapolate the rest
    SlipWall,         // inviscid wall / symmetry (reflect normal velocity)
    NoSlipAdiabatic,  // viscous wall, zero wall heat flux
    NoSlipIsothermal, // viscous wall, fixed wall temperature
    FixedState        // Dirichlet to a per-patch uniform state (MMS / imposed shock)
};

// Per-patch boundary specification.  The prescribed values are interpreted
// according to kind; unused fields are ignored.
struct BoundarySpec {
    BoundaryKind kind = BoundaryKind::Extrapolate;
    Primitive freestream;           // for inflow / fixed-state / imposed-shock
    double wallTemp = 300;          // for NoSlipIsothermal
    double backPressure = 101325.0; // for SubsonicOutflow
    Vec3 wallVelocity{};            // tangential wall velocity (moving wall / Couette)
    // Optional spatially varying prescribed state for SupersonicInflow and
    // FixedState patches (e.g. a measured incoming boundary-layer profile, or
    // an imposed-shock top boundary with pre and post-shock segments), one
    // Primitive per patch face in the patch's own face order. Empty means the
    // uniform `freestream` state applies.
    std::vector<Primitive> profile;
    // Optional spatially varying wall temperature for NoSlipIsothermal
    // patches, one value per patch face in the patch's own face order (the
    // interaction data's measured wall-temperature row: recovery upstream of
    // the thermal switch, the imposed s-condition downstream). Empty means
    // the uniform `wallTemp` applies.
    std::vector<double> wallTempProfile;
};

// Container mapping patch names to their boundary specs.
struct DBNSBoundaryConditions {
    std::unordered_map<std::string, BoundarySpec> specs;

    void set(const std::string &patch, const BoundarySpec &s) { specs[patch] = s; }

    const BoundarySpec &get(const std::string &patch) const {
        auto it = specs.find(patch);
        if (it == specs.end())
            throw std::runtime_error("DBNSBoundaryConditions: no spec for patch '" + patch + "'");
        return it->second;
    }

    bool has(const std::string &patch) const { return specs.count(patch) > 0; }
};

} // namespace dbns
