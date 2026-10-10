#include "ForwardModel.hpp"
#include "ParameterSensitivity.hpp"
#include <iostream>
#include <limits>
#include <future>

static int observationDerivativeTest() {
    Mesh mesh = Mesh::makeChannel2D(4, 8, 2.0, 1.0, 10.0, 1.0);
    mesh.computeWallDistance();
    auto bcs = FlowBoundaryConditions::couetteDefaults(mesh, 1.0, 1e-4, 1.0);
    SolverSettings settings;
    settings.maxIterations = 12000;
    settings.convergenceTol = 1e-7;
    settings.nuTFloorIters = 0;
    settings.verbose = false;
    ObservationOperator obs;
    obs.addSkinFriction("top", Vec3(1.0, 1.0, 0), 0, 1, 1);
    obs.addDrag("top", 0, 1, 1, 1);
    obs.addSeparationPoint("bottom", 0, 1);
    obs.addReattachmentLength("bottom", 0, 1);
    obs.addVelocityProfile(Vec3(1.0, 0.5, 0), 0, 0, 1);
    obs.addPressureTap(Vec3(1.0, 0.5, 0), 0, 1, 1);
    ParameterSensitivity sensitivity(mesh, obs, bcs, 0.1, settings);
    auto theta = InferenceParameterSet::toVector(SSTCoefficients{});
    if (sensitivity.solveState(theta) != EvaluationStatus::Converged) {
        std::cerr << "FAIL: observation derivative fixture did not converge\n";
        return 1;
    }
    bool experimentalRejected = false;
    try {
        sensitivity.etaJacobianTangentCoupled(theta, 1e-6, 1);
    } catch (const std::logic_error &) {
        experimentalRejected = true;
    }
    if (!experimentalRejected) {
        std::cerr << "FAIL: unsupported tangent needs explicit opt-in\n";
        return 1;
    }
    sensitivity.setCoupledKrylovRestart(7);
    if (sensitivity.coupledKrylovRestart() != 7 ||
        sensitivity.coupledKrylovBasisBytes() != size_t{15 * 5 * 32 * 8})
        return 1;
    auto other = theta;
    other[0] *= 1.01;
    bool mismatchRejected = false;
    try {
        sensitivity.etaJacobianWarmFD(other);
    } catch (const std::invalid_argument &) {
        mismatchRejected = true;
    }
    if (!mismatchRejected)
        return 1;
    // Strict one-iteration perturbations fail convergence and must not masquerade as zero
    // derivatives.
    auto failedFD = sensitivity.etaJacobianWarmFD(theta, 5e-4, 1e-7, 1, 1e-14);
    for (int j = 0; j < 11; ++j) {
        if (failedFD.krylovConverged[j] != 0 || !std::isnan(failedFD.logLikGradient[j]))
            return 1;
        for (const auto &row : failedFD.dObsDTheta)
            if (!std::isnan(row[j]))
                return 1;
    }
    // The betaStar/a1 caps must control the actual perturbed solve tolerance,
    // and both failures must remain visible even when the column is NaN.
    for (int j = 0; j < 11; ++j) {
        if (failedFD.positiveStatus[j] == EvaluationStatus::Converged ||
            failedFD.negativeStatus[j] == EvaluationStatus::Converged ||
            failedFD.finiteDifferenceSteps[j] <= 0)
            return 1;
    }
    const auto actual = sensitivity.etaJacobianWarmFD(theta, 5e-4, 1e-7, 1);
    for (int j = 0; j < 11; ++j) {
        const double expected = std::min(
            settings.convergenceTol, 0.01 * actual.finiteDifferenceSteps[j] / std::abs(theta[j]));
        if (std::abs(actual.perturbedSolveTolerances[j] - expected) > 1e-20)
            return 1;
    }
    std::vector<double> explicitSteps(11, 2e-6);
    const auto explicitFD =
        sensitivity.etaJacobianWarmFD(theta, 5e-4, 1e-7, 1, 1e-12, explicitSteps);
    for (int j = 0; j < 11; ++j)
        if (std::abs(explicitFD.finiteDifferenceSteps[j] - explicitSteps[j]) > 1e-16 ||
            explicitFD.perturbedSolveTolerances[j] != 1e-12)
            return 1;
    bool badSteps = false;
    try {
        sensitivity.etaJacobianWarmFD(theta, 5e-4, 1e-7, 1, 0, {1e-5});
    } catch (const std::invalid_argument &) {
        badSteps = true;
    }
    if (!badSteps)
        return 1;
    // Force the Bradshaw limiter active at the fixed state. The historical
    // analytic observable derivative incorrectly differentiated nuT even though
    // the actual wall observable uses only molecular viscosity.
    theta[9] = 1e-10;
    auto plus = theta, minus = theta;
    plus[9] *= 1.01;
    minus[9] *= 0.99;
    const double fd =
        (sensitivity.observe(plus)[0] - sensitivity.observe(minus)[0]) / (plus[9] - minus[9]);
    const double analytic = sensitivity.dObsDTheta(theta)[0][9];
    if (fd != 0.0 || analytic != 0.0) {
        std::cerr << "FAIL: fixed-state molecular wall derivative: FD=" << fd
                  << " analytic=" << analytic << '\n';
        return 1;
    }
    const auto analyticAll = sensitivity.dObsDTheta(theta);
    const auto gp = sensitivity.observe(plus), gm = sensitivity.observe(minus);
    for (size_t i = 0; i < gp.size(); ++i) {
        if (gp[i] != gm[i])
            return 1;
        for (double derivative : analyticAll[i])
            if (derivative != 0.0)
                return 1;
    }
    auto invalid = InferenceParameterSet::toVector(SSTCoefficients{});
    invalid[0] = std::numeric_limits<double>::quiet_NaN();
    if (sensitivity.solveState(invalid) != EvaluationStatus::InvalidParameters ||
        sensitivity.hasState())
        return 1;
    return 0;
}

int main(int argc, char **) {
    if (argc > 1)
        return observationDerivativeTest();
    int failures = 0;
    auto check = [&](bool ok, const char *message) {
        if (!ok) {
            std::cerr << "FAIL: " << message << '\n';
            ++failures;
        }
    };
    Mesh mesh = Mesh::makeChannel2D(4, 4, 2.0, 1.0, 1e5, 1.0);
    mesh.computeWallDistance();
    auto bcs = FlowBoundaryConditions::channelDefaults(mesh, 1.0, 1e-4, 1.0);
    SolverSettings settings;
    settings.maxIterations = 1;
    settings.verbose = false;
    ObservationOperator obs;
    auto params = InferenceParameterSet::a1_betaStar();
    auto theta = params.pack(SSTCoefficients{});
    ForwardModel model(mesh, params, obs, bcs, 1e-3, settings);
    auto first = model.evaluate(theta);
    check(first.status == EvaluationStatus::Unconverged, "fixture must hit its iteration cap");
    check(model.hasLastFields(),
          "successful finite capped solve exposes its own diagnostic fields");
    check(model.penalizedLogLikelihood(theta) == -1e30,
          "MCMC penalty wrapper must reject capped CFD likelihoods");
    auto invalid = model.evaluate({-1.0, theta[1]});
    check(invalid.status == EvaluationStatus::InvalidParameters, "invalid proposal rejected");
    check(!model.hasLastFields(), "invalid proposal clears previous fields");

    ParameterSensitivity sensitivity(mesh, obs, bcs, 1e-3, settings);
    auto theta11 = InferenceParameterSet::toVector(SSTCoefficients{});
    check(sensitivity.solveState(theta11) == EvaluationStatus::Unconverged,
          "sensitivity fixture must be unconverged");
    check(!sensitivity.hasState(), "unconverged base is not a sensitivity fixed point");
    bool rejected = false;
    try {
        sensitivity.observe(theta11);
    } catch (const std::exception &) {
        rejected = true;
    }
    check(rejected, "sensitivity operations reject an unconverged base");
    SolverSettings divergentSettings = settings;
    divergentSettings.divergenceLimit = 1e-100;
    ParameterSensitivity divergent(mesh, obs, bcs, 1e-3, divergentSettings);
    check(divergent.solveState(theta11) == EvaluationStatus::Diverged && !divergent.hasState(),
          "diverged sensitivity base is unavailable");

    WarmStartCache cache(2, 0.5);
    FlowFields fields(mesh);
    fields.p.setUniform(1.0);
    cache.store({0.0}, fields);
    fields.p.setUniform(2.0);
    cache.store({0.0}, fields);
    check(cache.size() == 1, "same parameters replace an existing cached state");
    WarmStartCache::CacheEntry copied;
    check(!cache.findNearest({0.0, 0.0}, copied), "different parameter dimensions cannot match");
    check(cache.findNearest({0.0}, copied) && copied.fields.p[0] == 2.0,
          "retrieval uses replacement state");
    cache.clear();
    check(copied.fields.p[0] == 2.0, "retrieved value survives cache eviction");

    const size_t oneEntry = WarmStartCache::entryBytes({0.0}, fields);
    cache.setByteBudget(oneEntry);
    cache.store({0.0}, fields);
    fields.p.setUniform(3.0);
    cache.store({2.0}, fields);
    FlowFields retrieved(mesh);
    check(cache.retainedBytes() == oneEntry && cache.size() == 1,
          "byte budget evicts the oldest state");
    check(!cache.findNearest({0.0}, retrieved) && cache.findNearest({2.0}, retrieved),
          "eviction chooses old key");
    cache.setByteBudget(oneEntry - 1);
    cache.store({3.0}, fields);
    check(cache.size() == 0 && cache.retainedBytes() == 0, "oversized state is not retained");
    check(retrieved.p[0] == 3.0, "direct retrieved state survives eviction");

    // Separate workers share only immutable mesh/observation/config inputs.
    obs.addVelocityProfile(Vec3(0.7, 0.5, 0), 0, 1.0, 0.1);
    auto runWorker = [&]() {
        ForwardModel worker(mesh, params, obs, bcs, 1e-3, settings);
        worker.setRetainLastFields(false);
        auto evaluated = worker.evaluateWithDiagnostics(theta, true);
        if (worker.hasLastFields())
            throw std::runtime_error("worker retained optional legacy fields");
        return evaluated;
    };
    auto pending = std::async(std::launch::async, runWorker);
    auto workerA = runWorker();
    auto workerB = pending.get();
    check(workerA.result.status == workerB.result.status &&
              workerA.result.predictions == workerB.result.predictions,
          "independent worker execution matches serial results");
    check(workerA.snapshot.has_value() && workerB.snapshot.has_value(),
          "per-evaluation snapshots owned after worker deletion");
    check(workerA.snapshot->p == workerB.snapshot->p &&
              workerA.snapshot->UBoundary.size() ==
                  static_cast<size_t>(mesh.nFaces() - mesh.nInternalFaces()) &&
              workerA.snapshot->faceFluxes.size() == static_cast<size_t>(mesh.nFaces()),
          "complete snapshot includes boundaries and final face fluxes");
    workerA.snapshot->p[0] += 100;
    check(workerA.snapshot->p != workerB.snapshot->p, "snapshots share no writable storage");
    check(obs.logLikelihood(workerB.result.predictions) == workerB.result.loglik,
          "likelihood reuses the exact returned predictions");
    bool wrongCountRejected = false;
    try {
        obs.logLikelihood(std::vector<double>{});
    } catch (const std::invalid_argument &) {
        wrongCountRejected = true;
    }
    check(wrongCountRejected, "prediction likelihood validates shape");

    // The cache lock protects copies against concurrent mutation and eviction.
    cache.setByteBudget(oneEntry);
    auto writer = std::async(std::launch::async, [&]() {
        for (int i = 0; i < 100; ++i)
            cache.store({static_cast<double>(i)}, fields);
    });
    for (int i = 0; i < 100; ++i) {
        FlowFields candidate(mesh);
        if (cache.findNearest({static_cast<double>(i)}, candidate))
            check(candidate.p[0] == 3.0, "concurrent cache retrieval returns a complete value");
    }
    writer.get();
    return failures == 0 ? observationDerivativeTest() : 1;
}
