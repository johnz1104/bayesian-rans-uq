#include "CompressibleForwardModel.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>

CompressibleForwardModel::CompressibleForwardModel(
    const Mesh &mesh, const InferenceParameterSet &paramSet, const ObservationOperator &obsOp,
    const CompressibleBoundaryConditions &bcs, const IdealGasEOS &eos,
    const SolverSettings &settings, const Vec3 &uInit, double pInit, double TInit, double kInit,
    double omegaInit)
    : mesh_(mesh), paramSet_(paramSet), obsOp_(obsOp), bcs_(bcs), eos_(eos), settings_(settings),
      uInit_(uInit), pInit_(pInit), TInit_(TInit), kInit_(kInit), omegaInit_(omegaInit) {
    eos_.validate();
    settings_.validate();
    validateCompressibleInitialState(uInit_, pInit_, TInit_, kInit_, omegaInit_, eos_);
    bcs_.validate(mesh_);
    // The shared observation adapter receives a FlowFields view with no
    // density or local dynamic viscosity. Its drag operator is therefore
    // dimensionally valid only for the incompressible kinematic-pressure
    // convention. Reject the unsupported combination at the typed boundary
    // instead of leaving a comment on a silently executable mixed-units path.
    for (const Observable &obs : obsOp_.observables()) {
        if (obs.type == ObsType::Drag) {
            throw std::invalid_argument("CompressibleForwardModel does not support generic Drag "
                                        "observations; use the DBNS wall-observation path");
        }
    }
}

EvaluationResult CompressibleForwardModel::evaluate(const std::vector<double> &theta) {
    EvaluationResult result;
    result.status = EvaluationStatus::Unknown;
    // Retained diagnostics belong to the current call, including early failures.
    hasFields_ = false;
    lastFields_ = CompressibleFlowFields{};

    // Validate parameters
    if (!paramSet_.inBounds(theta)) {
        result.status = EvaluationStatus::InvalidParameters;
        result.loglik = -1e30;
        return result;
    }

    // Build SST coefficients from theta
    SSTCoefficients coeffs = paramSet_.unpack(theta);
    SSTModel sst(coeffs);
    sst.variant = static_cast<SSTVariant>(settings_.sstVariant);

    // Build and initialise solver
    CompressibleSIMPLESolver solver(mesh_, sst, bcs_, eos_, settings_);
    CompressibleFlowFields fields(mesh_);
    solver.initUniform(fields, uInit_, pInit_, TInit_, kInit_, omegaInit_);

    // Solve
    CompressibleConvergenceHistory hist = solver.solve(fields);
    result.simpleIters = hist.finalIter;

    // Always retain the latest fields (even on divergence) so callers can do
    // post-mortem analysis instead of having to re-solve from scratch.
    lastFields_ = fields;
    hasFields_ = true;

    if (hist.diverged) {
        result.status = EvaluationStatus::Diverged;
        result.loglik = -1e30;
        return result;
    }

    // Map compressible fields to FlowFields for ObservationOperator
    // (reuses existing incompressible observation machinery)
    FlowFields ff(mesh_);
    ff.U = fields.U;
    // observables see the THERMODYNAMIC pressure: the solver's p is the
    // mechanical pressure with the (2/3) rho k turbulent normal stress
    // absorbed (see the assembly note), and a pressure tap or drag integral
    // must not report the bookkeeping term
    ff.p = fields.p;
    for (int ci = 0; ci < mesh_.nCells(); ++ci)
        ff.p[ci] = fields.p[ci] - (2.0 / 3.0) * fields.rho[ci] * std::max(fields.k[ci], 0.0);
    ff.k = fields.k;
    ff.omega = fields.omega;
    ff.nuT = fields.nuT;
    ff.F1 = fields.F1;
    ff.F2 = fields.F2;
    ff.Pk = fields.Pk;
    ff.CDkw = fields.CDkw;

    // Effective kinematic viscosity (μ/ρ at first interior cell)
    double mu0 = eos_.viscosity(fields.T[0]);
    double rho0 = std::max(fields.rho[0], 1e-30);
    double nu_eff = mu0 / rho0;

    // Evaluate observations and build result
    result.predictions = obsOp_.evaluate(mesh_, ff, nu_eff);
    result.loglik = obsOp_.logLikelihood(mesh_, ff, nu_eff);
    result.status = hist.converged ? EvaluationStatus::Converged : EvaluationStatus::Unconverged;
    result.simpleIters = hist.finalIter;

    return result;
}

double CompressibleForwardModel::penalizedLogLikelihood(const std::vector<double> &theta) {
    auto result = evaluate(theta);
    if (!std::isfinite(result.loglik) || result.status != EvaluationStatus::Converged)
        return -1e30;
    return result.loglik;
}
