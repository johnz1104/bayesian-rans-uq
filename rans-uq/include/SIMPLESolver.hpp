#pragma once

#include "Mesh.hpp"
#include "Field.hpp"
#include "FlowFields.hpp"
#include "BoundaryCondition.hpp"
#include "LinearSolver.hpp"
#include "SSTModel.hpp"
#include <vector>
#include <string>
#include <memory>
#include <functional>

// SIMPLE = Semi-Implicit Method for Pressure-Linked Equations
// Iteratively solves momentum and pressure-correction equations to enforce
// continuity and converge the steady incompressible RANS flow field.
class SIMPLESolver {
  public:
    SIMPLESolver(const Mesh &mesh, const SSTModel &sst, const FlowBoundaryConditions &bcs,
                 double nu, const SolverSettings &settings = {});

    // Numerical storage only. Every solve resets history, guesses and source memory.
    struct Workspace {
        const Mesh *mesh;
        Mesh::Stamp stamp;
        bool active = false;
        LinearSystem momentum, pressure, k, omega;
        ScalarField pressureCorrection, strain;
        SSTWorkspace closure;
        std::vector<double> residual, previousObservations, solveValues, diagonal, oldK, oldOmega,
            wallOmega, interpolatedFlux, predictedFlux;
        std::vector<Vec3> oldVelocity;
        explicit Workspace(const Mesh &m)
            : mesh(&m), stamp(m.stamp()), momentum(makeSystem(m)), pressure(makeSystem(m)),
              k(makeSystem(m)), omega(makeSystem(m)), pressureCorrection(m, "p'"), strain(m, "|S|"),
              closure(m), residual(m.nCells()), solveValues(m.nCells()), diagonal(m.nCells()),
              oldVelocity(m.nCells()) {}
    };
    ConvergenceHistory solve(FlowFields &fields);
    ConvergenceHistory solve(FlowFields &fields, Workspace &workspace);

    // Native research policies change only production, not SST's constitutive
    // fields or discretization. Context references are valid for the call only.
    struct ProductionContext {
        const Mesh &mesh;
        const FlowFields &fields;
        const SSTCoefficients &coefficients;
        const ScalarField &strainMagnitude;
        const std::vector<double> &targetAnisotropy; // xx,yy,zz,xy,xz,yz
        double nu;
    };
    class ProductionPolicy {
      public:
        virtual ~ProductionPolicy() = default;
        // Called after pre-momentum SST/floor updates (and for fresh diagnostic
        // closure). Write nCells finite entries; no implicit clipping is added.
        virtual void kProduction(const ProductionContext &context,
                                 std::vector<double> &output) const = 0;
        // Called after the k solve. Return production before alpha(F1)*volume;
        // fields.Pk still carries the pre-momentum policy output.
        virtual double omegaProduction(const ProductionContext &context, int cell) const = 0;
    };
    void setProductionPolicy(std::shared_ptr<const ProductionPolicy> policy);
    bool hasProductionPolicy() const { return static_cast<bool>(productionPolicy_); }

    // Scale the whole fresh integrated injection force before source-memory
    // blending. Default 1 preserves historical injection. A custom production
    // policy supports endpoint scales only: 0 bypasses production, 1 enables it.
    void setInjectionScale(double scale);
    double injectionScale() const { return injectionScale_; }

    enum class SourceStage { PreMomentumX };
    struct SourceSnapshotView {
        int iteration;
        const FlowFields &fields;              // all cell and boundary values, before x solve
        const std::vector<double> &faceFluxes; // active owner-outward convection
        const std::vector<Vec3> &freshIntegratedForce;   // -sum_faces(D.n)*area
        const std::vector<Vec3> &appliedIntegratedForce; // after source blending
        SourceStage stage = SourceStage::PreMomentumX;
    };
    using SourceObserver = std::function<void(const SourceSnapshotView &)>;
    // Output-only native callback, once per outer iteration after x assembly,
    // before x solve. References expire on return; copy explicit snapshots to
    // retain them. Never called by fixed-state diagnostics. Null costs no copies.
    void setSourceObserver(SourceObserver observer) { sourceObserver_ = std::move(observer); }

    enum class ClosureMode { Stored, Recomputed };
    struct MomentumDiagnostics {
        FlowFields fields;          // owned arrays; borrows the same mesh as the solver
        std::vector<Vec3> residual; // all three integrated b-A*U components
        std::vector<Vec3> freshIntegratedForce;
    };
    // Explicit saved flux, no steady-flux reconstruction or source memory.
    // Work is isolated from this solver, including when called by an observer.
    MomentumDiagnostics diagnoseMomentum(const FlowFields &state, const SSTCoefficients &theta,
                                         const std::vector<double> &savedFaceFluxes,
                                         ClosureMode closure) const;

    void initUniform(FlowFields &f, const Vec3 &Uinit, double pInit, double kInit,
                     double omegaInit);

    // Interior resolved-wall value used by this solver, including its geometry
    // qualification. Wall functions and historical pinning retain their own
    // treatment. Exposed so sensitivity operators use the identical condition.
    double resolvedWallOmega(double wallDistance) const;

    // Quantities of interest whose convergence the solve must also certify
    // (e.g. the observation vector of a forward model). Evaluated every
    // settings.monitorInterval iterations; the default criterion then
    // requires each component's change, and its extrapolated remainder, below
    // convergenceTol, measured relative to max(|q_i|, floor_i). The floors
    // stop a quantity that sits near zero (a pressure tap near the reference
    // level, Cf near reattachment) from being judged relative to a vanishing
    // value; a forward model passes its observation standard deviations.
    // Without floors, 1e-6 of the largest |q| is used. Ignored by the legacy
    // criterion.
    using ConvergenceMonitor = std::function<std::vector<double>(const FlowFields &)>;
    void setConvergenceMonitor(ConvergenceMonitor fn, std::vector<double> floors = {}) {
        monitor_ = std::move(fn);
        monitorFloors_ = std::move(floors);
    }

    // Fixed primary/boundary state; recompute closure and exactly two flux
    // reconstruction passes. Transport uses b-A*q; continuity is outward
    // flux sum. This diagnostic algorithm is NOT the steady solved map.
    struct DiagnosticResidual {
        std::vector<double> transport;  // [Ux, Uy, k, omega], 4*nCells
        std::vector<double> continuity; // nCells
        std::vector<double> faceFluxes; // owner-outward, nFaces
        std::vector<double> equations() const {
            auto result = transport;
            result.insert(result.end(), continuity.begin(), continuity.end());
            return result;
        }
    };
    struct DiagnosticWorkspace {
        const Mesh *mesh;
        Mesh::Stamp stamp;
        bool active = false;
        FlowFields fields;
        ScalarField strain;
        SSTWorkspace closure;
        LinearSystem system;
        std::vector<double> diagonal, values, residual;
        explicit DiagnosticWorkspace(const Mesh &m)
            : mesh(&m), stamp(m.stamp()), fields(m), strain(m, "|S|"), closure(m),
              system(makeSystem(m)), diagonal(m.nCells()), values(m.nCells()),
              residual(m.nCells()) {}
    };
    DiagnosticResidual diagnosticResidual(const FlowFields &state,
                                          const SSTCoefficients &theta) const;
    void diagnosticResidual(const FlowFields &state, const SSTCoefficients &theta,
                            DiagnosticResidual &output, DiagnosticWorkspace &workspace) const;

    // Eleven columns with the same transport/continuity/flux layout. Ordinary
    // branch derivatives: users must qualify smoothness by perturbation tests.
    std::vector<DiagnosticResidual> diagnosticParameterPartials(const FlowFields &state,
                                                                const SSTCoefficients &theta) const;

    struct SteadyResidual {
        std::vector<double> equations;  // [transport, continuity/gauge, free-flux consistency]
        std::vector<double> continuity; // includes the replaced periodic row
        std::vector<double> faceFluxes;
        double removedContinuity = 0;
        bool pressureGauge = false;
    };
    std::vector<int> freeFluxFaces() const;
    SteadyResidual steadyResidual(const FlowFields &state, const SSTCoefficients &theta,
                                  const std::vector<double> &freeFluxes) const;

    // Historical four-block APIs. Corrected full-transpose supported calls
    // delegate to the diagnostic partials. Reduced/legacy calls remain
    // experimental compatibility paths, not steady sensitivities.
    std::vector<double> assembleResidual(const FlowFields &state, const SSTCoefficients &theta);
    std::vector<std::vector<double>> assembleResidualSensitivity(const FlowFields &state,
                                                                 const SSTCoefficients &theta,
                                                                 bool includeTransposeTheta = true);

    // Assemble the (Picard) segregated block operators at a FIXED state with the closure from
    // `theta`: the momentum-x system A_mom (its relaxed diagonal returned in aP), the k system
    // A_k, and the ω system A_om.  These are used ONLY as a physics-based PRECONDITIONER for the
    // matrix-free coupled tangent (full within-block solves capture stiff spatial coupling that
    // a Jacobi diagonal misses); they are not an analytic state Jacobian (no cross-block
    // coupling, no transpose; A ≈ −∂R_block/∂block per block).  Reuses assembleMomentum/K/Omega.
    void assemblePreconBlocks(const FlowFields &state, const SSTCoefficients &theta,
                              LinearSystem &Amom, LinearSystem &Ak, LinearSystem &Aom,
                              std::vector<double> &aP);

    int residualSize() const { return 4 * mesh_.nCells(); } // length of one residual vector

    // Diagnostic two-pass continuity, not a claim of steady flux consistency.
    std::vector<double> continuityResidual(const FlowFields &state, const SSTCoefficients &theta);

    // Face fluxes (one per face, owner-outward) of the last solve's final
    // iteration: the Rhie-Chow fluxes the pressure correction made
    // discretely divergence-free and that U, k and omega were convected with.
    const std::vector<double> &faceFluxes() const {
        mesh_.requireStamp(meshStamp_);
        return phi_;
    }

    // ---- A-POSTERIORI REYNOLDS-STRESS INJECTION (explicit deferred-correction) ----
    // Inject a prescribed per-cell target anisotropy b_target into the momentum
    // equation. The turbulent stress the baseline applies is the implicit
    // Boussinesq diffusion div((nu+nuT) grad U); the injection adds the explicit
    // body force
    //     f_inj = -div( 2 k b_target + 2 nuT dev(S) )  =  -div( 2 k (b_target - b_B) )
    // with b_B = -(nuT/k) dev(S) the solver's own Boussinesq anisotropy, k the
    // RUNNING turbulent kinetic energy, and nuT kept implicit as the stabilizer.
    // The force vanishes identically when b_target equals b_B (the baseline solve
    // is recovered exactly), and the injected difference from the baseline is
    // exactly -div(2 k db). This avoids the ill-conditioning of full explicit
    // Reynolds-stress substitution (Wu et al. 2019).
    //
    // b6 points at nCells*6 values ordered xx, yy, zz, xy, xz, yz per cell; the
    // caller keeps the storage alive across solve(). Realizability of b_target is
    // re-checked every outer iteration (barycentric margin, AnisotropyTools) and
    // recorded in the diagnostics; the check is separate from the (Python-side)
    // Galilean-invariant feature construction.
    struct InjectionDiagnostics {
        bool active = false;       // a target was set
        int checkedIters = 0;      // outer iterations the check ran on
        bool allRealizable = true; // barycentric margin >= -tol everywhere, always
        double maxViolation = 0.0; // worst barycentric violation seen (>= 0)
    };
    void setTargetAnisotropy(const std::vector<double> *b6);

    // Per-cell momentum source per unit volume (added as s_i V_i), for
    // manufactured-solution verification. Why not bodyForce: it is uniform.
    // The caller keeps the storage alive across solve().
    void setMomentumSource(const std::vector<Vec3> *source);
    const InjectionDiagnostics &injectionDiagnostics() const { return injDiag_; }

  private:
    std::shared_ptr<const ProductionPolicy> productionPolicy_;
    double injectionScale_ = 1.0;
    SourceObserver sourceObserver_;
    std::vector<Vec3> freshInjection_;
    bool sourceObservationActive_ = false;
    bool productionPolicyActive() const {
        return productionPolicy_ && bTarget6_ && injectionScale_ != 0.0;
    }
    void applyProductionPolicy(FlowFields &fields, const ScalarField &strain);
    const std::vector<double> *bTarget6_ = nullptr;
    const std::vector<Vec3> *momSource_ = nullptr;

    // Deferred correction F_f (phi_f^HO - phi_U) for one transported cell
    // field: subtracted from the upwind cell's equation and added to the
    // downwind one, with F_f the convective flux of the assembly.
    void addConvectionCorrection(LinearSystem &sys, const FlowFields &f, const ScalarField &phi,
                                 const std::string &scheme) const;
    InjectionDiagnostics injDiag_;

    // Under-relaxed injection source, persistent across the outer iterations of
    // one solve: q_blend <- (1 - alpha) q_blend + alpha q(U^n, k^n). The
    // explicit correction cancels a first-order part of the implicit nuT
    // stabilizer, and feeding it back unrelaxed leaves a residual oscillation
    // floor at the tolerance level (the Wu et al. 2019 explicit-treatment
    // ill-conditioning); the blend damps the feedback and leaves the fixed
    // point unchanged. Fresh (unblended) evaluation is used outside solve()
    // (assembleResidual), where the source must be state-consistent.
    struct InjectionWorkspace {
        VelocityGradientWorkspace components;
        VelocityGradients gradients;
        std::vector<double> tensor;
        std::vector<Vec3> force;
        explicit InjectionWorkspace(const Mesh &mesh)
            : components(mesh),
              gradients{VectorField(mesh, "grad(Ux)"), VectorField(mesh, "grad(Uy)"),
                        VectorField(mesh, "grad(Uz)")},
              tensor(6 * mesh.nCells()), force(mesh.nCells()) {}
    };
    std::unique_ptr<InjectionWorkspace> injectionWorkspace_;
    std::vector<Vec3> injSrcBlend_;
    double injRelax_ = 1.0; // solve() sets the blend factor; 1 = fresh

    // Adds the explicit deferred-correction source for `component` to sys and,
    // on the x-component pass, re-checks realizability of b_target.
    void addInjectionSource(LinearSystem &sys, const FlowFields &f, int component);

  public:
    // Reduced-operator switch for the FROZEN-PRESSURE tangent: when set, the
    // momentum assembly omits the transpose-stress deferred correction, so the
    // matrix-free (dR/dU) application matches the Picard operator the solver
    // actually inverts per outer iteration (deferred corrections are lagged,
    // i.e. frozen in U, like the pressure in this reduced tangent;
    // a frozen source is constant in U and drops out of the FD difference).
    // Without this the reduced 4-block operator carries the full-stress
    // rigid-rotation near-null mode that only pressure coupling removes, and
    // BiCGSTAB stagnates on it. The coupled tangent keeps the full operator.
    void setFreezeTransposeStress(bool v) { freezeTransposeStress_ = v; }

    // Cumulative time by solver stage for the last
    // solve() call (momentum assembly+solves, pressure assembly, pressure
    // solve, turbulence updates), printed at solve exit when verbose; the
    // AMG setup/iterate split lives on the AMG solver instance.
    double momSeconds() const { return momSeconds_; }
    double pAssembleSeconds() const { return pAssembleSeconds_; }
    double pSolveSeconds() const { return pSolveSeconds_; }
    double turbSeconds() const { return turbSeconds_; }
    void reportTimers() const;

  private:
    bool freezeTransposeStress_ = false;
    double momSeconds_ = 0.0, pAssembleSeconds_ = 0.0, pSolveSeconds_ = 0.0, turbSeconds_ = 0.0;

    const Mesh &mesh_;
    Mesh::Stamp meshStamp_ = mesh_.stamp();
    SSTModel sst_;
    FlowBoundaryConditions bcs_;
    double nu_;
    SolverSettings settings_;

    // Geometry-only singular extension for orthogonal periodic parallel-wall
    // meshes. Empty elsewhere; no planar extension is applied to curved walls.
    std::vector<double> omegaSingularity_, omegaNormalGradient_, omegaLaplacian_;
    void preparePlanarOmegaBalance();

    std::unique_ptr<ILinearSolver> pSolver_;
    std::unique_ptr<ILinearSolver> mSolver_;
    std::unique_ptr<ILinearSolver> tSolver_;

    // Diagonal momentum coefficients retained for Rhie-Chow interpolation.
    // aP_ is the relaxed diagonal (aP_raw/alphaU) used for velocity correction;
    // aPunrelaxed_ is the raw diagonal used for the pressure Laplacian.
    std::vector<double> aP_;
    std::vector<double> aPunrelaxed_;

    // Iter-0 residual norms for normalisation. Equations with negligible iter-0
    // residual (e.g. Uy in 2D) are skipped via safeNorm to avoid false divergence.
    // Since the scaled criterion became the default these drive only the
    // divergence check and the legacy criterion.
    double normUx0_ = 1, normUy0_ = 1, normP0_ = 1;

    ConvergenceMonitor monitor_;
    std::vector<double> monitorFloors_;
    // sum of |face mass flux| over all faces from the last pressure-correction
    // assembly: the scale of the continuity residual
    double massFluxScale_ = 0.0;

    // Stored face fluxes (see faceFluxes) and the flux the assemblies convect
    // with: when convFlux_ points at a flux vector the assemblies use it,
    // otherwise they interpolate the cell velocities (legacy behaviour).
    std::vector<double> phi_;
    const std::vector<double> *convFlux_ = nullptr;
    // pressure-correction Laplacian coefficients (V/a_P)_f |S| / delta of the
    // last assembly, per internal face, and per boundary face on outlets
    std::vector<double> pCoeff_;

    double interpolatedFlux(const FlowFields &f, int fi) const;
    double convectiveFlux(const FlowFields &f, int fi) const;
    // U_f . S_f: interpolated on internal faces, the boundary value on
    // boundary faces (one entry per face)
    void interpolatedFluxes(const FlowFields &f, std::vector<double> &phi) const;
    // phi_f -= D_f [ (p_N - p_O)|S|/delta - grad(p)_f . e |S| ] on internal
    // faces, with D_f the distance-weighted interpolation of V / aP
    void addRhieChow(const FlowFields &f, const std::vector<double> &aP,
                     std::vector<double> &phi) const;
    // the steady flux phi(U,p) of a fixed state (unrelaxed coefficient, two
    // fixed-point passes on the momentum diagonal); legacy: interpolated
    void steadyFluxes(const FlowFields &f, std::vector<double> &phi);
    bool legacyRhieChowActive() const; // the old gate: outlet-free meshes only

    void assembleMomentum(LinearSystem &sys, const FlowFields &f, int component,
                          std::vector<double> &aP);
    // predicted: the predicted face fluxes whose divergence the correction
    // removes (one per face); nullptr = legacy (interpolated, Rhie-Chow only
    // on outlet-free meshes, computed inside)
    void assemblePressureCorrection(LinearSystem &sys, const FlowFields &f,
                                    const std::vector<double> &aP, ScalarField &pPrime,
                                    const std::vector<double> *predicted = nullptr);
    void assembleKEquation(LinearSystem &sys, const FlowFields &f);
    // Smag is frozen from pre-correction computeFields to prevent pressure-correction
    // oscillations from amplifying omega production and causing blow-up.
    void assembleOmegaEquation(LinearSystem &sys, const FlowFields &f, const ScalarField &Smag);

    void omegaWallValues(const FlowFields &fields, std::vector<double> &values) const;
    void constrainOmega(LinearSystem &system, const std::vector<double> &values) const;

    void correctVelocity(FlowFields &f, const ScalarField &pPrime, const std::vector<double> &aP);
    void correctPressure(FlowFields &f, const ScalarField &pPrime);

    // ADJOINT GROUNDWORK helper: recompute the SST closure fields of `work` from the
    // current sst_.coeffs (computeFields + 0.1ν nuT floor) and the frozen strain |S|,
    // exactly as the SIMPLE loop does on a turbulence-update iteration.
    DiagnosticResidual diagnosticResidualImpl(const FlowFields &state, const SSTCoefficients &theta,
                                              const std::vector<double> *suppliedFlux = nullptr);
    void diagnosticResidualInto(const FlowFields &state, const SSTCoefficients &theta,
                                DiagnosticResidual &output, DiagnosticWorkspace &workspace,
                                const std::vector<double> *suppliedFlux = nullptr);
    void requireDiagnosticSupport(const FlowFields &state) const;
    std::vector<std::vector<double>> transportClosurePartials(const FlowFields &state,
                                                              const SSTCoefficients &theta,
                                                              bool includeTransposeTheta);
    std::vector<std::vector<double>>
    diagnosticFluxPartials(const FlowFields &work,
                           const std::vector<SSTClosureSensitivity> &closure,
                           std::vector<double> &flux);
    void recomputeClosure(FlowFields &work, ScalarField &Smag);
};
