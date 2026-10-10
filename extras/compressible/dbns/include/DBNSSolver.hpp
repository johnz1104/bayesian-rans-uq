#pragma once

#include "Mesh.hpp"
#include "DBNSTypes.hpp"
#include "DBNSConfig.hpp"
#include "SSTModel.hpp"
#include "EvaluationTypes.hpp"
#include <array>
#include <cstdint>
#include <vector>

// ---------------------------------------------------------------------------
// Density-based, shock-capturing finite-volume RANS solver carrying the
// Favre-averaged SST k-omega closure.
//
// Discretization:
//   - Convective flux: HLLC approximate Riemann solver (HLLCFlux).
//   - Reconstruction: MUSCL linear with optional slope limiting; switchable to
//     first order or unlimited. Smooth unlimited MMS tests check design order
//     on their stated fixtures; they do not establish a general TVD guarantee.
//   - Viscous flux: corrected face-gradient stresses and heat conduction.
//     Full-SST, wall-work and nonorthogonal-mesh accuracy is verified only on
//     the tested fixtures, not in general.
//   - Turbulence:      compressible SST sources with optional dilatational
//     (Sarkar/Zeman) dissipation; turbulent Prandtl number set by the EOS.
//   - Time march:      explicit SSP Runge-Kutta with local time stepping for
//     steady problems and a global step for unsteady (shock tube) problems.
//
// The solver classifies outcomes (Converged / Unconverged / Diverged) instead
// of throwing, matching the repo's forward-model discipline.
// ---------------------------------------------------------------------------

namespace dbns {

struct SolveReport {
    EvaluationStatus status = EvaluationStatus::Unknown;
    int iterations = 0;
    double finalResidual = 0.0;
    double tFinal = 0.0;
    // true when the quiescence route classified the convergence (restored
    // checkpoint, zero injected correction, full-CFL stationarity); false
    // for the registered relative-decay route, so consumers can assert a
    // forced member never rode quiescence
    bool quiescent = false;
    std::vector<double> residualHistory; // density-equation L2 residual per report
};

// Diagnostics of the model-form injection (mirrors the incompressible
// solver's record). The realizability re-check is on the EFFECTIVE RUNNING
// anisotropy b_eff(W) = b_B(W) + db_stored, where b_B is the solver's own
// Boussinesq anisotropy at the current state (-mu_t/(rho k) dev(S), the
// running gradients and eddy viscosity the momentum fluxes actually carry)
// and db_stored is the injected discrepancy. Recorded per residual
// evaluation, diagnostic only: b_eff is never projected or clipped during
// the run. The extremes carry the residual-evaluation ordinal and the cell.
struct InjectionDiagnostics {
    bool active = false;
    int checkedIters = 0;
    bool allRealizable = true;
    double minMargin = 1e30;   // worst barycentric margin of b_eff seen
    int minMarginIter = -1;    // residual-evaluation ordinal of that worst
    int minMarginCell = -1;    // cell index of that worst
    double maxViolation = 0.0; // max(0, -margin) over the run
    int maxViolationIter = -1;
    int maxViolationCell = -1;
    double maxDb = 0.0; // max |db_stored| component magnitude
    double maxDq = 0.0; // max |dq_target| component magnitude
};

class DBNSSolver {
  public:
    DBNSSolver(const Mesh &mesh, const IdealGasEOS &eos, const SSTCoefficients &sst,
               const DBNSBoundaryConditions &bcs, const DBNSSettings &settings = {});

    // Initialise every cell to a uniform primitive state.
    void initUniform(const Primitive &V);

    // Initialise from a per-cell primitive field (e.g. an MMS field).
    void initField(const std::vector<Primitive> &V);

    // Advance the solution (steady iteration or unsteady march).
    SolveReport solve();

    // Compute derived properties (gradients, laminar/eddy viscosity) for the
    // current state without advancing it.  Lets the observation operator be
    // exercised on a prescribed field independent of solver convergence.
    void prepareProperties() {
        mesh_.requireStamp(meshStamp_);
        updateLaminarProperties();
        computeGradients();
        updateTurbulentProperties();
    }

    // --- accessors for observation / extraction ---
    int nCells() const {
        mesh_.requireStamp(meshStamp_);
        return mesh_.nCells();
    }
    Primitive primitive(int ci) const {
        mesh_.requireStamp(meshStamp_);
        return GasState::toPrimitive(W_.at(static_cast<size_t>(ci)), eos_);
    }
    const StateVec &conserved(int ci) const {
        mesh_.requireStamp(meshStamp_);
        return W_.at(static_cast<size_t>(ci));
    }
    double eddyViscosity(int ci) const {
        mesh_.requireStamp(meshStamp_);
        return muT_.at(static_cast<size_t>(ci));
    } // mu_t [Pa s]
    double laminarViscosity(int ci) const {
        mesh_.requireStamp(meshStamp_);
        return muLam_.at(static_cast<size_t>(ci));
    } // mu [Pa s] (honours constMu)
    const Mesh &mesh() const {
        mesh_.requireStamp(meshStamp_);
        return mesh_;
    }
    const IdealGasEOS &eos() const { return eos_; }

    // --- model-form injection (the deferred-correction coupling) ---
    // Set the sampled closure correction: per-cell STORED anisotropy
    // discrepancy db (nCells x 6, ordered xx, yy, zz, xy, xz, yz) and
    // per-cell turbulent heat-flux correction dq (nCells x 2, x and y
    // components, SOLVER units of a Favre temperature flux,
    // <rho u_i''T''>/<rho> [m/s K]). The stored discrepancy is the operative
    // input: the injected flux is -div(2 rho k db), so db = 0 is exactly
    // zero flux at any state (the discrete zero-correction contract). With
    // energyReach the consistent stress work and the injected heat flux
    // (against the implicit gradient-diffusion term) enter the energy
    // equation, the heat-flux reach of the pre-registered scheme. Values are
    // COPIED (no caller lifetime coupling). bDiag6 (the caller's absolute
    // target) is retained as a provenance record only: the running
    // realizability diagnostic checks b_eff(W) = b_B(W) + db at every
    // residual evaluation (see InjectionDiagnostics), recording rather than
    // projecting; bDiag6 has no diagnostic consumer since that replacement.
    // The caller projects its absolute targets before setting.
    void setTargetCorrection(const std::vector<double> &db6, const std::vector<double> &bDiag6,
                             const std::vector<double> &dq2, bool energyReach = true,
                             const std::vector<std::uint8_t> &mask = {});
    void clearTargetCorrection();
    // per-cell omega-production limiter activation of the LAST residual
    // evaluation (the solver's own branch decision, so activation maps are
    // exact rather than a python recomputation). This is the SST source
    // limiter, distinct from the Venkatakrishnan reconstruction limiter
    // below.
    const std::vector<std::uint8_t> &limiterActive() const {
        mesh_.requireStamp(meshStamp_);
        return limiterActive_;
    }
    // reconstruction-limiter checkpointing: export the Venkatakrishnan
    // limiter values of the last evaluation (nCells x 6, primitive order
    // rho, u, v, p, k, omega) and restore them as a FROZEN limiter for a
    // warm restart, so a reloaded converged state resumes the exact
    // discrete operator it converged under and carries no limiter-refresh
    // transient (the checkpoint-restart semantics of the member solves).
    // Implicit steady driver only; the explicit path recomputes limiters.
    std::vector<double> reconstructionLimiter() const;
    void setFrozenReconstructionLimiter(const std::vector<double> &lim6);
    const InjectionDiagnostics &injectionDiagnostics() const { return injDiag_; }

    // --- MMS support ---
    // Inject a constant-in-time per-cell source S so the discrete residual
    // balances a manufactured solution: dW/dt = -(Res - S*V)/V.
    // Nonfinite sources are permitted for diagnostic fault injection; solve()
    // reports Diverged instead of treating them as successful physical input.
    // A laminar solve carries no turbulence, so its k and omega rows must be zero.
    void setManufacturedSource(const std::vector<StateVec> &src);
    // Override the boundary ghost state per boundary face (for spatially varying
    // Dirichlet, e.g. the MMS exact solution).  Indexed by boundary-face index
    // (global face id - nInternalFaces). States are validated like initial states.
    void setBoundaryOverride(const std::vector<Primitive> &ghost);

  private:
    const Mesh &mesh_;
    Mesh::Stamp meshStamp_ = mesh_.stamp();
    IdealGasEOS eos_;
    SSTCoefficients sstCoeffs_;
    SSTModel sst_; // for F1/F2 blending formulas
    DBNSBoundaryConditions bcs_;
    DBNSSettings settings_;

    std::vector<StateVec> W_;            // conserved state per cell
    std::vector<double> muLam_;          // laminar dynamic viscosity per cell
    std::vector<double> muT_;            // turbulent (eddy) dynamic viscosity
    std::vector<StateVec> mmsSource_;    // optional MMS source (per cell)
    std::vector<Primitive> bndOverride_; // optional per-boundary-face ghost

    // gradients of primitive variables (rho,u,v,p,k,omega) per cell
    std::vector<std::array<Vec3, 6>> grad_;
    std::vector<std::array<double, 6>> limiter_; // Venkatakrishnan scalar per var
    bool startLimiterFrozen_ = false;            // restored-checkpoint limiter state

    // workspace
    std::vector<StateVec> res_;  // residual accumulator
    std::vector<double> dtCell_; // local time step

    // patch-local ordinal of each boundary face (per-face boundary profiles)
    std::vector<int> patchLocalIdx_;

    // wall-adjacent cells (owners of no-slip faces) and their center-to-face
    // distances, for the per-iteration Menter omega re-pin
    std::vector<int> wallAdjCell_;
    std::vector<double> wallAdjDelta_;

    // wall distance measured to the patches the BOUNDARY CONDITIONS mark as
    // no-slip walls (the mesh's own lazily-computed distance is empty unless
    // a caller populates it, and its patch types are the generator's guess;
    // the turbulent path needs the true viscous walls). Computed once in the
    // constructor; 1e10 everywhere when no no-slip patch exists.
    std::vector<double> wallDist_;

    // scale-aware positivity floors, captured at initialisation so the
    // rescue values are sane at any unit scale (nondimensional shock tubes
    // and dimensional wall flows share this solver)
    double rhoFloor_ = 1e-10;
    double pFloor_ = 1e-10;

    // true when the state came from initField (a restart or a prescribed
    // nonuniform field): the march then computes the discrete gradients
    // before its first property evaluation, so the first eddy viscosity is
    // not formed at zero strain. Uniform cold initialisations skip it BY
    // CONSTRUCTION, keeping their established trajectories bit-identical
    // (with a prescribed inflow profile the boundary ghosts make even a
    // uniform field's Green-Gauss gradients nonzero at boundary cells, so
    // an unconditional pre-march gradient pass would shift cold paths).
    bool initFromField_ = false;

    // pipeline stages
    void updateProperties(); // muLam, muT, SST blending
    void updateLaminarProperties();
    void updateTurbulentProperties();
    void computeGradients();
    void computeLimiters();
    Primitive ghostState(const Primitive &interior, int faceId, const BoundarySpec &spec,
                         int boundaryIdx) const;
    double wallOmegaGhost(const Primitive &interior, int faceId) const;
    void computeResidual();          // fills res_
    void addViscousFace(int faceId); // internal-face viscous contribution
    void addBoundaryFlux(int faceId, int patchIdx);
    void addTurbulenceSources();
    void computeTimeStep(double cfl, bool includeViscous);
    void clampPositivity();
    void clampPositivityCore();
    double rhoResidualNorm() const;
    // per-equation volume-scaled RMS residual norms (all NVAR conservative
    // equations); the completed implicit-steady convergence criterion gates
    // on EVERY live equation's relative decay, not density alone (a member
    // could otherwise classify converged while momentum, energy or the
    // turbulence equations are still moving, the same false-convergence
    // class the pressure-based solvers guard against)
    std::array<double, NVAR> equationResidualNorms() const;
    // direct per-cell conservative-state validation: every component finite
    // and the decoded thermodynamic state admissible (max/min reductions
    // drop NaN operands, so only an explicit sweep can prove integrity)
    bool stateIsValid() const;

    // implicit (LU-SGS) steady march: matrix-free backward-Euler pseudo-time
    // with lower-upper symmetric Gauss-Seidel sweeps on the spectral-radius-
    // split first-order Jacobian, point-implicit SST destruction diagonal
    SolveReport solveImplicitSteady();
    void computeFaceSpectralRadii(); // fills lamFace_
    std::vector<double> lamFace_;    // face spectral radius (conv + visc)
    std::vector<StateVec> dW_;       // LU-SGS state-increment workspace

    // model-form injection state (values owned by the solver)
    void addInjectionFluxes(); // internal-face deferred correction
    // Recomputed every residual call; retained only as private numeric storage.
    std::vector<double> correctionXX_, correctionYY_, correctionXY_, correctionQx_, correctionQy_;
    std::vector<double> dbTarget6_;           // nCells x 6 stored discrepancy
    std::vector<double> bTarget6_;            // nCells x 6 target anisotropy (diagnostics)
    std::vector<std::uint8_t> injMask_;       // per-cell injection activity
    double injRampScale_ = 1.0;               // current ramp factor on the fluxes
    std::vector<double> injRhoK0_;            // rho k at target-set time (frozen-k)
    std::vector<std::uint8_t> limiterActive_; // omega-limiter branch record
    std::vector<double> dqTarget2_;           // nCells x 2 heat-flux correction
    bool injectEnergyReach_ = true;
    InjectionDiagnostics injDiag_;

    // reconstruct primitive at a face from cell ci toward face center xf
    Primitive reconstruct(int ci, const Vec3 &xf) const;
};

} // namespace dbns
