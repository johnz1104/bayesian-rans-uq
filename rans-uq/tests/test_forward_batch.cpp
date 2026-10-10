#include "ForwardBatch.hpp"
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <new>
static std::atomic<long> failAfter{-1};
static std::atomic<bool> allocationFailed{false};
void *operator new(std::size_t n) {
    if (failAfter.load() >= 0 && failAfter.fetch_sub(1) == 0) {
        failAfter = -1;
        allocationFailed = true;
        throw std::bad_alloc();
    }
    if (void *p = std::malloc(n ? n : 1))
        return p;
    throw std::bad_alloc();
}
void operator delete(void *p) noexcept {
    std::free(p);
}
void *operator new[](std::size_t n) {
    return ::operator new(n);
}
void operator delete[](void *p) noexcept {
    ::operator delete(p);
}
static void require(bool b) {
    if (!b)
        throw std::runtime_error("forward batch assertion failed");
}
static bool equal(double a, double b) {
    return std::memcmp(&a, &b, sizeof(double)) == 0;
}
int main() {
    auto mesh = Mesh::makeChannel2D(3, 4, 1, 1);
    mesh.computeWallDistance();
    const auto parameters = InferenceParameterSet::a1_betaStar();
    ObservationOperator obs;
    obs.addVelocityProfile({.5, .4, 0}, 0, 0, 1);
    const auto bc = FlowBoundaryConditions::channelDefaults(mesh, 1, .02, 3);
    SolverSettings settings;
    settings.verbose = false;
    settings.maxIterations = 8;
    ForwardBatch batch(mesh, parameters, obs, bc, .01, settings);
    const std::vector<std::vector<double>> theta{{.31, .09}, {.35, .1}, {-1, .1}, {.3, .08}};
    const auto one = batch.evaluate(theta), two = batch.evaluate(theta, 2);
    require(one.rows.size() == theta.size() && one.nObservations == 1);
    for (size_t i = 0; i < theta.size(); ++i) {
        ForwardModel cold(mesh, parameters, obs, bc, .01, settings);
        cold.cache().setByteBudget(0);
        const auto reference = cold.evaluateWithDiagnostics(theta[i]);
        for (const auto *output : {&one, &two}) {
            const auto &row = output->rows[i];
            require(row.result.status == reference.result.status &&
                    row.result.simpleIters == reference.result.simpleIters &&
                    row.result.loglik == reference.result.loglik);
            if (reference.result.predictions.empty()) {
                for (double value : row.result.predictions)
                    require(std::isnan(value));
            } else
                require(row.result.predictions == reference.result.predictions);
            require(equal(row.finalWorstMeasure, reference.convergence.finalWorstMeasure) &&
                    equal(row.contractionRateU, reference.convergence.contractionRateU));
            require(static_cast<bool>(output->converged[i]) ==
                    (reference.result.status == EvaluationStatus::Converged));
        }
    }
    require(!one.rows[2].error.empty());
    require(batch.evaluate({}, 2).rows.empty());
    bool rejected = false;
    try {
        batch.evaluate({{.31}}, 2);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    require(rejected);
    rejected = false;
    try {
        batch.evaluate(theta, 0);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    require(rejected);
    // Original inputs can change without invalidating the batch-owned copies.
    mesh.computeGeometry();
    obs.addVelocityProfile({.1, .2, 0}, 0, 0, 1);
    const auto independent = batch.evaluate(theta, 2);
    require(independent.nObservations == 1 &&
            independent.rows[0].result.predictions == one.rows[0].result.predictions);
    int failures = 0;
    for (int nth = 0; nth < 400; nth += 13) {
        allocationFailed = false;
        failAfter = nth;
        bool propagated = false;
        try {
            (void)batch.evaluate(theta, 2);
        } catch (const std::bad_alloc &) {
            ++failures;
            propagated = true;
        }
        failAfter = -1;
        require(!allocationFailed ||
                propagated); // Resource exhaustion must never become an ordinary row error.
        const auto recovered = batch.evaluate({theta[0]}, 1);
        require(recovered.rows[0].result.predictions == one.rows[0].result.predictions);
    }
    require(failures > 0);
    std::cout << "serial-cold/one/two-worker equality, ordering, input ownership, allocation "
                 "failure and recovery passed\n";
}
