#include "ForwardModel.hpp"
#include "SIMPLEValidation.hpp"
#include <iostream>
#include <stdexcept>

// Constructor
ForwardModel::ForwardModel(const Mesh &mesh, const InferenceParameterSet &paramSet,
                           const ObservationOperator &obsOp, const FlowBoundaryConditions &bcs,
                           double nu, const SolverSettings &settings, const Vec3 &Uinit,
                           double pInit, double kInit, double omegaInit)
    // initializer list
    : mesh_(mesh), paramSet_(paramSet), obsOp_(obsOp), preparedObservations_(obsOp_.prepare(mesh)),
      workspace_(mesh), bcs_(bcs), nu_(nu), settings_(settings), Uinit_(Uinit), pInit_(pInit),
      kInit_(kInit), omegaInit_(omegaInit), cache_(50, 0.5) {
    simple_admission::problem(mesh_, bcs_, nu_, settings_);
    simple_admission::initial(Uinit_, pInit_, kInit_, omegaInit_);
    paramSet_.validate();
}

// set the injected per-cell target anisotropy (validated length, stored by value)
void ForwardModel::setTargetAnisotropy(std::vector<double> b6) {
    mesh_.requireStamp(meshStamp_);
    if (b6.size() != static_cast<size_t>(6 * mesh_.nCells()))
        throw std::runtime_error("setTargetAnisotropy: expected nCells*6 values");
    if (!std::all_of(b6.begin(), b6.end(), [](double value) { return std::isfinite(value); }))
        throw std::invalid_argument("setTargetAnisotropy: target values must be finite");
    bTarget6_ = std::move(b6);
}

// evaluate: θ -> EvaluationResult
EvaluationResult ForwardModel::evaluate(const std::vector<double> &theta) {
    return evaluateWithDiagnostics(theta).result;
}

ForwardEvaluationResult ForwardModel::evaluateWithDiagnostics(const std::vector<double> &theta,
                                                              bool snapshot) {
    mesh_.requireStamp(meshStamp_);
    ForwardEvaluationResult evaluation;
    EvaluationResult &result = evaluation.result;
    hasLastFields_ = false;
    lastFields_ = FlowFields{};
    lastInjection_ = {};

    // bounds check
    if (!paramSet_.inBounds(theta)) {
        result.status = EvaluationStatus::InvalidParameters;
        result.loglik = -1e30;
        return evaluation;
    }

    try {
        // unpack θ -> SST coefficients
        SSTCoefficients coeffs = paramSet_.unpack(theta);
        SSTModel sst(coeffs);
        // Apply the selected closure variant (Full / NoLimiter / KOmega).
        sst.variant = static_cast<SSTVariant>(settings_.sstVariant);

        // set up solver
        SIMPLESolver solver(mesh_, sst, bcs_, nu_, settings_);
        if (!bTarget6_.empty())
            solver.setTargetAnisotropy(&bTarget6_);
        // the predictions are what the likelihood consumes, so convergence
        // must certify them too, not only the fields
        if (obsOp_.nObs() > 0) {
            std::vector<double> floors;
            for (const Observable &ob : obsOp_.observables())
                floors.push_back(std::sqrt(ob.meas.totalVariance()));
            solver.setConvergenceMonitor(
                [this](const FlowFields &ff) { return preparedObservations_.evaluate(ff, nu_); },
                floors);
        }
        FlowFields fields(mesh_);

        // warm start: check cache for nearby solution. Injected solves are
        // fully cache-independent (no lookup, no store): a warm start from a
        // nearby state makes the iter-0 residual normalisation, and with it
        // the convergence classification, depend on what ran before, and an
        // ensemble study needs every member reproducible in isolation.
        evaluation.usedWarmStart = bTarget6_.empty() && cache_.findNearest(theta, fields);
        if (!evaluation.usedWarmStart) {
            solver.initUniform(fields, Uinit_, pInit_, kInit_, omegaInit_);
        }

        // solve
        ConvergenceHistory &hist = evaluation.convergence;
        hist = solver.solve(fields, workspace_);
        result.simpleIters = hist.finalIter;
        lastInjection_ = solver.injectionDiagnostics();
        evaluation.injection = lastInjection_;

        // classify convergence
        if (hist.diverged) {
            result.status = EvaluationStatus::Diverged;
            result.loglik = -1e6;
            // INVALIDATE any previously retained fields: a diverged solve
            // leaves no valid state, and keeping the previous run's fields
            // visible let downstream QoI extraction silently inherit them
            // (two committed hills member records duplicated their
            // predecessor exactly this way)
            hasLastFields_ = false;
            return evaluation;
        } else if (hist.converged) {
            result.status = EvaluationStatus::Converged;
        } else {
            result.status = EvaluationStatus::Unconverged;
            // Keep this solve's observations for diagnostics. The posterior
            // wrapper below rejects it because a capped state is not a CFD likelihood.
        }

        // retain fields for visualization / post-processing
        // extract observables and log-likelihood
        result.predictions = preparedObservations_.evaluate(fields, nu_);
        result.loglik = obsOp_.logLikelihood(result.predictions);
        // A converged field is reusable only after its observations and
        // likelihood also pass validation. Failed evaluations do not seed a cache.
        if (result.status == EvaluationStatus::Converged && bTarget6_.empty())
            cache_.store(theta, fields);
        if (snapshot)
            evaluation.snapshot.emplace(fields, solver.faceFluxes());
        if (retainLastFields_) {
            lastFields_ = std::move(fields);
            hasLastFields_ = true;
        }

    } catch (const std::bad_alloc &) {
        throw; // Allocation failure aborts a batch after its workers join.
    } catch (const std::exception &e) {
        result.status = EvaluationStatus::Unknown;
        result.loglik = -1e6;
        result.predictions.clear();
        evaluation.snapshot.reset();
        evaluation.error = e.what();
        if (settings_.verbose)
            std::cerr << "ForwardModel exception: " << e.what() << "\n";
    }
    return evaluation;
}

// MCMC friendly wrapper
double ForwardModel::penalizedLogLikelihood(const std::vector<double> &theta) {
    EvaluationResult res = evaluate(theta);
    return res.status == EvaluationStatus::Converged ? res.loglik : -1e30;
}
