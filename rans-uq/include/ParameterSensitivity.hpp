#pragma once

#include "Mesh.hpp"
#include "Field.hpp"
#include "FlowFields.hpp"
#include "SSTModel.hpp"
#include "SIMPLESolver.hpp"
#include "ObservationOperator.hpp"
#include "BoundaryCondition.hpp"
#include "InferenceParameters.hpp"
#include "EvaluationTypes.hpp"
#include "MatrixFreeSolver.hpp"
#include <vector>
#include <array>
#include <string>

// Fixed-state coefficient partials and total finite-difference sensitivities.
// solveState must converge before any state-dependent operation is available.
// Current observations read U/p and molecular viscosity: at fixed state their
// coefficient partials are exactly zero. Residual partials/tangents of the
// corrected Rhie-Chow discretization remain experimental.
// No analytic state-adjoint implementation is supplied by this interface.

struct TangentGradientResult {
    std::vector<std::vector<double>> dObsDTheta; // nObs × 11  (∂η/∂θ)
    std::vector<double> logLikGradient; // 11 — ∂logL/∂θ (Gaussian); warm-FD fills this
    std::vector<int> krylovIters;       // 11 — Krylov/outer iterations per coefficient
    std::vector<double> krylovRelRes;   // 11 — final ‖r‖/‖b‖ (or continuity) per coeff
    std::vector<int> krylovConverged;   // 11 — 1/0 (int for a clean binding)
    std::vector<double> finiteDifferenceSteps;    // actual central half-width, 11
    std::vector<double> perturbedSolveTolerances; // actual tolerance used, 11
    std::vector<EvaluationStatus> positiveStatus, negativeStatus;
    int nResidualEvals = 0; // total residual assemblies / solve iterations
    int nColors = 0;        // colours used for the colored-FD diagonal
};

// Fixed user-declared nondimensional scales; never chosen from a derivative.
struct SteadySensitivityOptions {
    std::array<double, 6> stateScales{{1, 1, 1, 1, 1, 1}}; // Ux,Uy,k,omega,p,flux
    std::array<double, 6> equationScales{{1, 1, 1, 1, 1, 1}};
    double actionStep = 1e-5;
    double parameterRelativeStep = 1e-5;
    double krylovTolerance = 1e-8;
    int restart = 60;
    int maxIterations = 600;
};
struct SteadySensitivityBase {
    FlowFields fields;
    std::vector<double> correctedFluxes;
    ConvergenceHistory history;
    std::vector<double> coefficients;
    SolverSettings settings;
    FlowBoundaryConditions boundaries;
    double molecularViscosity;
    std::string sourceDigest;
};
struct SteadyTangentResult {
    TangentGradientResult derivatives;
    std::vector<std::vector<double>> stateTangents; // eleven augmented columns
    std::vector<int> branchQualified;
    double normalizedBaseResidual = 0;
    double removedContinuity = 0;
    size_t krylovBasisBytes = 0;
    bool experimental = true; // external refinement/dense/total-FD gates still required
    std::string qualification;
};

class ParameterSensitivity {
  public:
    ParameterSensitivity(const Mesh &mesh, const ObservationOperator &obs,
                         const FlowBoundaryConditions &bcs, double nu,
                         const SolverSettings &settings = {}, const Vec3 &Uinit = {1, 0, 0},
                         double pInit = 0.0, double kInit = 1e-4, double omegaInit = 1.0);

    // Solve once at θ (full 11-vector) and store the converged state as the fixed point.
    EvaluationStatus solveState(const std::vector<double> &theta11);

    // Corrected Rhie-Chow state/parameter residual consistency has not yet
    // been verified. These methods remain explicitly experimental until the
    // full directional-derivative gates pass. Warm FD does not need this opt-in.
    void setAllowExperimentalTangents(bool allow) { allowExperimentalTangents_ = allow; }
    void setCoupledKrylovRestart(int restart) {
        if (restart < 1)
            throw std::invalid_argument("Coupled Krylov restart must be positive");
        coupledKrylovRestart_ = restart;
    }
    int coupledKrylovRestart() const { return coupledKrylovRestart_; }
    size_t coupledKrylovBasisBytes() const {
        return FGMRESWorkspace::basisBytes(5 * static_cast<size_t>(mesh_.nCells()),
                                           coupledKrylovRestart_);
    }

    bool hasState() const { return hasState_; }
    int nState() const { return 4 * mesh_.nCells(); } // [Ux|Uy|k|ω] blocks
    int nCells() const { return mesh_.nCells(); }
    int nObs() const { return obs_.nObs(); }

    // R(q*, θ): stacked discrete residual at the fixed state, closure recomputed from θ.
    std::vector<double> residual(const std::vector<double> &theta11);
    // ∂R/∂θ: 11 residual-shaped vectors (analytic).
    std::vector<std::vector<double>> dResidualDTheta(const std::vector<double> &theta11);

    // g(q*, θ): observation predictions with the SST closure recomputed from θ.
    std::vector<double> observe(const std::vector<double> &theta11);
    // Scalar Gaussian log-likelihood at the fixed state (closure recomputed from θ).  With a
    // preceding solveState(θ) this is logL at the converged state — the WarmFDForwardModel loglik.
    double logLik(const std::vector<double> &theta11);
    // ∂g/∂θ: nObs × 11, zero for all current molecular/state-only observations.
    std::vector<std::vector<double>> dObsDTheta(const std::vector<double> &theta11);

    // Experimental frozen-pressure/Picard-reduced tangent. It is not the total
    // derivative of a fully coupled forward solve. Requires explicit opt-in,
    // a converged state at theta11 and reports each linear convergence flag.
    TangentGradientResult etaJacobianTangent(const std::vector<double> &theta11,
                                             double krylovTol = 1e-8, int maxIter = 3000,
                                             double fdStep = 1e-6);

    // Experimental five-variable tangent with FGMRES and explicit linear
    // residual verification. Corrected Rhie-Chow parameter/continuity coupling
    // has not passed total-derivative validation; linear convergence alone does
    // not certify that derivative. Basis workspace is reused across columns.
    TangentGradientResult etaJacobianTangentCoupled(const std::vector<double> &theta11,
                                                    double krylovTol = 1e-8, int maxIter = 3000,
                                                    double fdStep = 1e-6);

    // Central finite difference of fully re-converged solves, starting each
    // perturbation from the same converged base. Failed columns contain NaN
    // and convergence flag 0. krylovIters is total SIMPLE iterations for +/-;
    // krylovRelRes is the worse final convergence measure. Zero overrides use
    // configured limits, with default tolerance tightened relative to each actual relative step.
    // absoluteSteps overrides all eleven absolute half-widths when supplied.
    // Step-size and solver-tolerance refinement remain the caller's accuracy gate.
    TangentGradientResult etaJacobianWarmFD(const std::vector<double> &theta11, double hRel = 5e-4,
                                            double hFloor = 1e-7, int warmMaxIter = 0,
                                            double warmTol = 0.0,
                                            const std::vector<double> &absoluteSteps = {});

    // Separate augmented steady operator; no change to historical tangents.
    SteadySensitivityBase steadyBase() const;
    std::vector<double> steadyStateVector() const;
    SIMPLESolver::SteadyResidual steadyResidualAt(const std::vector<double> &state,
                                                  const std::vector<double> &theta11) const;
    std::vector<double> steadyStateAction(const std::vector<double> &direction,
                                          double step = 1e-5) const;
    // Qualify actual independently re-solved perturbations, not only the
    // linearized tangent prediction. Throws on a branch or free-flux failure.
    void requireSteadyPair(const std::vector<double> &plus, const std::vector<double> &minus,
                           const std::vector<double> &thetaPlus,
                           const std::vector<double> &thetaMinus) const;
    size_t steadyKrylovBasisBytes(int restart) const;
    SteadyTangentResult etaJacobianSteady(const SteadySensitivityOptions &options = {});

  private:
    const Mesh &mesh_;
    Mesh::Stamp meshStamp_ = mesh_.stamp();
    ObservationOperator obs_;
    FlowBoundaryConditions bcs_;
    double nu_;
    SolverSettings settings_;
    Vec3 Uinit_;
    double pInit_, kInit_, omegaInit_;

    FlowFields state_;
    bool hasState_ = false;
    std::vector<double> stateTheta_;
    std::vector<double> stateFluxes_;
    ConvergenceHistory stateHistory_;
    FlowFields mappedSteadyFields(const std::vector<double> &state,
                                  const std::vector<double> &theta11) const;
    std::vector<int> steadyBranches(const std::vector<double> &state,
                                    const std::vector<double> &theta11) const;
    int coupledKrylovRestart_ = 150;
    bool allowExperimentalTangents_ = false;
    void requireState() const;
    void requireMatchingState(const std::vector<double> &theta11) const;
    void requireExperimentalTangent() const;

    SSTModel makeModel(const std::vector<double> &theta11) const;

    // Greedy distance-1 colouring of the cell face-adjacency graph: two face-neighbour
    // cells never share a colour, so a single perturbation of all same-colour cells reads
    // the exact tangent-operator diagonal ∂R_(i,f)/∂x_(i,f) at every probed cell (the
    // residual stencil is distance-1).  Used to build the Jacobi preconditioner.
    std::vector<int> cellColoring(int &nColors) const;

    // Per-cell discrete continuity residual Σ_f (U_f·n) S_f (the SIMPLE mass-flux divergence,
    // no Rhie-Chow).  `homogeneous` true ⇒ tangent boundary fluxes (fixed-velocity walls/inlet
    // contribute 0; outlet uses zero-gradient = owner increment); false ⇒ use the state's
    // boundary face velocities.  Operates on the supplied interior velocity components.
    std::vector<double> massFluxDivergence(const std::vector<double> &ux,
                                           const std::vector<double> &uy, bool homogeneous) const;

    // Assemble the SIMPLE pressure-Poisson Laplacian ∇·((V/aP)∇p') with outlet-Dirichlet
    // (p'=0), matching assemblePressureCorrection — the Schur operator of the SIMPLE
    // preconditioner.  Matrix only (zero source); `aP` is the momentum diagonal.
    LinearSystem assemblePressurePoisson(const std::vector<double> &aP) const;
    // Recompute the closure of `work` from θ (computeFields + non-negativity clamp); if `cs` is
    // non-null also fill the per-cell pointwise closure sensitivities (and |S|).
    void recompute(const std::vector<double> &theta11, FlowFields &work,
                   std::vector<SSTClosureSensitivity> *cs) const;
};
