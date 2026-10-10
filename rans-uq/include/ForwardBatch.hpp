#pragma once
#include "ForwardModel.hpp"
#include "SIMPLEValidation.hpp"
#include <atomic>
#include <exception>
#include <thread>

struct ForwardBatchRow {
    EvaluationResult result;
    double finalWorstMeasure = std::numeric_limits<double>::quiet_NaN();
    double contractionRateU = std::numeric_limits<double>::quiet_NaN();
    bool stalled = false;
    std::string error;
};
struct ForwardBatchResult {
    int nObservations = 0;
    std::vector<ForwardBatchRow> rows;
    std::vector<std::uint8_t> converged;
};

// A synchronous coefficient-only batch. All problem inputs are owned values;
// each worker owns its model and scratch, with no warm cache or field retention.
class ForwardBatch {
    const Mesh mesh_;
    const InferenceParameterSet parameters_;
    const ObservationOperator observations_;
    const FlowBoundaryConditions boundaries_;
    const double nu_;
    const SolverSettings settings_;
    const Vec3 initialU_;
    const double initialP_, initialK_, initialOmega_;

  public:
    ForwardBatch(const Mesh &mesh, const InferenceParameterSet &parameters,
                 const ObservationOperator &observations, const FlowBoundaryConditions &boundaries,
                 double nu, const SolverSettings &settings = {}, const Vec3 &u = {1, 0, 0},
                 double p = 0, double k = 1e-4, double omega = 1)
        : mesh_(mesh), parameters_(parameters), observations_(observations),
          boundaries_(boundaries), nu_(nu), settings_(settings), initialU_(u), initialP_(p),
          initialK_(k), initialOmega_(omega) {
        simple_admission::problem(mesh_, boundaries_, nu_, settings_);
        simple_admission::initial(initialU_, initialP_, initialK_, initialOmega_);
        parameters_.validate();
        if (!mesh_.wallDistanceValid())
            throw std::invalid_argument(
                "ForwardBatch needs compute_wall_distance before construction");
        (void)observations_.prepare(
            mesh_); // Validate geometry-dependent definitions before launching workers.
    }
    int nParameters() const { return parameters_.nActive(); }
    int nObservations() const { return observations_.nObs(); }
    ForwardBatchResult evaluate(const std::vector<std::vector<double>> &theta,
                                int workers = 1) const {
        if (workers < 1)
            throw std::invalid_argument("workers must be positive");
        for (const auto &row : theta)
            if (row.size() != static_cast<size_t>(parameters_.nActive()))
                throw std::invalid_argument("batch rows must all have n_parameters entries");
        ForwardBatchResult result;
        result.nObservations = observations_.nObs();
        result.rows.resize(theta.size());
        result.converged.resize(theta.size());
        if (theta.empty())
            return result;
        std::atomic<size_t> next{0};
        std::atomic<bool> stop{false};
        std::exception_ptr fatal;
        std::mutex fatalMutex;
        auto execute = [&] {
            try {
                ForwardModel model(mesh_, parameters_, observations_, boundaries_, nu_, settings_,
                                   initialU_, initialP_, initialK_, initialOmega_);
                model.cache().setByteBudget(0);
                model.setRetainLastFields(false);
                while (!stop.load()) {
                    const auto index = next.fetch_add(1);
                    if (index >= theta.size())
                        break;
                    auto &row = result.rows[index];
                    try {
                        auto evaluation = model.evaluateWithDiagnostics(theta[index], false);
                        row.result = std::move(evaluation.result);
                        row.error = std::move(evaluation.error);
                        row.finalWorstMeasure = evaluation.convergence.finalWorstMeasure;
                        row.contractionRateU = evaluation.convergence.contractionRateU;
                        row.stalled = evaluation.convergence.stalled;
                    } catch (const std::bad_alloc &) {
                        throw;
                    } catch (const std::exception &error) {
                        row.error = error.what();
                    }
                    if (row.result.predictions.empty())
                        row.result.predictions.assign(observations_.nObs(),
                                                      std::numeric_limits<double>::quiet_NaN());
                    if (row.result.status == EvaluationStatus::InvalidParameters &&
                        row.error.empty())
                        row.error = "coefficient row is nonfinite or out of bounds";
                    result.converged[index] = row.result.status == EvaluationStatus::Converged;
                }
            } catch (...) {
                std::lock_guard<std::mutex> guard(fatalMutex);
                if (!fatal)
                    fatal = std::current_exception();
                stop = true;
            }
        };
        std::vector<std::thread> threads;
        const auto count = std::min(static_cast<size_t>(workers), theta.size());
        threads.reserve(count);
        try {
            for (size_t i = 0; i < count; ++i)
                threads.emplace_back(execute);
        } catch (...) {
            stop = true;
            for (auto &thread : threads)
                thread.join();
            throw;
        }
        for (auto &thread : threads)
            thread.join();
        if (fatal)
            std::rethrow_exception(fatal);
        return result;
    }
};
