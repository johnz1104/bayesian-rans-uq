#pragma once

#include "Mesh.hpp"
#include "Field.hpp"
#include "SSTModel.hpp"
#include "SIMPLESolver.hpp"
#include "InferenceParameters.hpp"
#include "ObservationOperator.hpp"
#include "EvaluationTypes.hpp" // EvaluationStatus, EvaluationResult (shared with compressible)
#include <vector>
#include <string>
#include <deque>
#include <mutex>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>

// Stores recent CFD solutions and finds nearest cached solution vector in parameter space to use as
// initial conditions
class WarmStartCache {
  public:
    // container for solution of the flow for parameter vector θ
    struct CacheEntry {
        std::vector<double> theta; // active inference parameter vector
        FlowFields fields;         // converged solution
        size_t bytes = 0;
    };

    // defines max number of stored solutions and max allowed parameter distance for reuse
    explicit WarmStartCache(int maxSize = 50, double threshold = 0.5,
                            size_t byteBudget = 64 * 1024 * 1024)
        : maxSize_(maxSize), threshold_(threshold), byteBudget_(byteBudget) {
        if (maxSize < 0 || !std::isfinite(threshold) || threshold < 0)
            throw std::invalid_argument("Invalid warm-start cache limits");
    }

    // Budget counts copied numeric payload (interiors, boundaries and parameters),
    // excluding allocator/container/name overhead. It is not a process-RSS limit.
    static size_t entryBytes(const std::vector<double> &theta, const FlowFields &f) {
        f.U.requireValid();
        size_t bytes = theta.size() * sizeof(double);
        bytes += (static_cast<size_t>(f.U.size()) + f.U.nBnd()) * sizeof(Vec3);
        for (const ScalarField *field :
             {&f.p, &f.k, &f.omega, &f.nuT, &f.F1, &f.F2, &f.Pk, &f.CDkw})
            bytes += (static_cast<size_t>(field->size()) + field->nBnd()) * sizeof(double);
        return bytes;
    }

    // Copy the closest cached solution into `out` while holding the lock and
    // report whether one was found. The owned copy remains valid after unlock,
    // even if a concurrent store() evicts the cached entry.
    bool findNearest(const std::vector<double> &theta, CacheEntry &out) const {
        std::lock_guard<std::mutex> lock(mtx_);
        const CacheEntry *best = nearest(theta);
        if (!best)
            return false;
        out = *best; // deep copy under the lock
        return true;
    }

    // The solver uses this overload: one deep copy straight into working fields.
    bool findNearest(const std::vector<double> &theta, FlowFields &out) const {
        std::lock_guard<std::mutex> lock(mtx_);
        const CacheEntry *best = nearest(theta);
        if (!best)
            return false;
        out = best->fields;
        return true;
    }

    // store a converged solution
    void store(const std::vector<double> &theta, const FlowFields &fields) {
        std::lock_guard<std::mutex> lock(mtx_);
        for (double value : theta)
            if (!std::isfinite(value))
                return;
        for (auto it = cache_.begin(); it != cache_.end(); ++it) {
            if (it->theta == theta) {
                retainedBytes_ -= it->bytes;
                cache_.erase(it);
                break;
            }
        }
        const size_t bytes = entryBytes(theta, fields);
        if (maxSize_ == 0 || bytes > byteBudget_)
            return;
        // Evict before allocating the new state to bound retained payload peaks.
        while (!cache_.empty() && (cache_.size() >= static_cast<size_t>(maxSize_) ||
                                   retainedBytes_ > byteBudget_ - bytes))
            evictFront();
        cache_.push_back({theta, fields, bytes});
        retainedBytes_ += bytes;
    }

    int size() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return (int)cache_.size();
    }
    void clear() {
        std::lock_guard<std::mutex> lock(mtx_);
        cache_.clear();
        retainedBytes_ = 0;
    }
    size_t retainedBytes() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return retainedBytes_;
    }
    size_t byteBudget() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return byteBudget_;
    }
    void setByteBudget(size_t bytes) {
        std::lock_guard<std::mutex> lock(mtx_);
        byteBudget_ = bytes;
        while (!cache_.empty() && retainedBytes_ > byteBudget_)
            evictFront();
    }

  private:
    int maxSize_;
    double threshold_;
    size_t byteBudget_;
    size_t retainedBytes_ = 0;
    mutable std::mutex mtx_;
    std::deque<CacheEntry> cache_;

    void evictFront() {
        retainedBytes_ -= cache_.front().bytes;
        cache_.pop_front();
    }
    const CacheEntry *nearest(const std::vector<double> &theta) const {
        const CacheEntry *best = nullptr;
        double bestDist = threshold_;
        for (const auto &entry : cache_) {
            entry.fields.U.requireValid();
            const double distance = euclidean(theta, entry.theta);
            if (distance < bestDist) {
                bestDist = distance;
                best = &entry;
            }
        }
        return best;
    }

    // computes Euclidean distance
    static double euclidean(const std::vector<double> &a, const std::vector<double> &b) {
        if (a.size() != b.size())
            return std::numeric_limits<double>::infinity();
        double s = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            double d = a[i] - b[i];
            s += d * d;
        }
        return std::sqrt(s);
    }
};

// Owned state arrays for export, independent of solver AND mesh lifetimes.
// Restart import still requires matching mesh/configuration identity; this is
// a complete numeric snapshot, not a serialized standalone solver definition.
struct OwnedFlowSnapshot {
    std::vector<Vec3> U;
    std::vector<double> p, k, omega, nuT, F1, F2, Pk, CDkw;
    std::vector<Vec3> UBoundary;
    std::vector<double> pBoundary, kBoundary, omegaBoundary, nuTBoundary, F1Boundary, F2Boundary,
        PkBoundary, CDkwBoundary;
    std::vector<double> faceFluxes;
    bool turbEstablished;

    template <typename T> static std::vector<T> boundaryCopy(const Field<T> &field) {
        std::vector<T> values(static_cast<size_t>(field.nBnd()));
        for (int i = 0; i < field.nBnd(); ++i)
            values[static_cast<size_t>(i)] = field.bnd(i);
        return values;
    }
    OwnedFlowSnapshot(const FlowFields &fields, const std::vector<double> &fluxes)
        : U(fields.U.data()), p(fields.p.data()), k(fields.k.data()), omega(fields.omega.data()),
          nuT(fields.nuT.data()), F1(fields.F1.data()), F2(fields.F2.data()), Pk(fields.Pk.data()),
          CDkw(fields.CDkw.data()), UBoundary(boundaryCopy(fields.U)),
          pBoundary(boundaryCopy(fields.p)), kBoundary(boundaryCopy(fields.k)),
          omegaBoundary(boundaryCopy(fields.omega)), nuTBoundary(boundaryCopy(fields.nuT)),
          F1Boundary(boundaryCopy(fields.F1)), F2Boundary(boundaryCopy(fields.F2)),
          PkBoundary(boundaryCopy(fields.Pk)), CDkwBoundary(boundaryCopy(fields.CDkw)),
          faceFluxes(fluxes), turbEstablished(fields.turbEstablished) {}
};

struct ForwardEvaluationResult {
    EvaluationResult result;
    ConvergenceHistory convergence;
    SIMPLESolver::InjectionDiagnostics injection;
    std::optional<OwnedFlowSnapshot> snapshot;
    std::string error;
    bool usedWarmStart = false;
};

// One instance per worker: the cache mutex does NOT make evaluations of this
// object concurrent-safe. Immutable mesh storage may be shared between workers.
class ForwardModel {
  public:
    // constructor
    ForwardModel(
        const Mesh &mesh, // mesh (cells, faces, owner-neighbor connectivity, patches, geometry)
        const InferenceParameterSet
            &paramSet, // paramSet (defines inferred parameters and mapping to SST coefficients)
        const ObservationOperator &obsOp,  // contains the experimental/prior data
        const FlowBoundaryConditions &bcs, // boundary conditions
        double nu, // kinematic viscosity (used in momentum, turbulence, skin friction)
        const SolverSettings &settings = {}, // contains solver settings

        // fallback initial fields (for when no warm-start is avaiable)
        const Vec3 &Uinit = {1, 0, 0}, double pInit = 0.0, double kInit = 1e-4,
        double omegaInit = 1.0);

    // core forward model evaluation
    // input: θ = inferred parameters
    // output: EvaluationResult
    EvaluationResult evaluate(const std::vector<double> &theta);
    ForwardEvaluationResult evaluateWithDiagnostics(const std::vector<double> &theta,
                                                    bool snapshot = false);
    void setRetainLastFields(bool retain) {
        retainLastFields_ = retain;
        if (!retain) {
            lastFields_ = FlowFields{};
            hasLastFields_ = false;
        }
    }

    // safe wrapper for MCMC
    // Accepts only converged solves; every other status returns -1e30.
    // Use evaluateWithDiagnostics to inspect capped states without accepting their likelihood.
    double penalizedLogLikelihood(const std::vector<double> &theta);

    // access functions
    WarmStartCache &cache() { return cache_; } // access warm-start cache (inspect, clear, resize)
    const InferenceParameterSet &paramSet() const { return paramSet_; }

    // last solved flow fields (retained for visualization / post-processing)
    const FlowFields &lastFields() const {
        mesh_.requireStamp(meshStamp_);
        if (!hasLastFields_)
            throw std::logic_error("No fields retained for the last evaluation");
        return lastFields_;
    }
    bool hasLastFields() const { return hasLastFields_; }

    // ---- a-posteriori Reynolds-stress injection ----
    // Set a per-cell target anisotropy (nCells*6 values, xx yy zz xy xz yz per
    // cell); every subsequent evaluate() solves with the explicit
    // deferred-correction source (see SIMPLESolver::addInjectionSource) until
    // cleared. The vector is stored by value (no external lifetime to manage).
    // Injected solves skip the warm-start cache store, so the plain theta ->
    // solution cache is not polluted by closure-perturbed states.
    void setTargetAnisotropy(std::vector<double> b6);
    void clearTargetAnisotropy() { bTarget6_.clear(); }
    bool hasTargetAnisotropy() const { return !bTarget6_.empty(); }
    const SIMPLESolver::InjectionDiagnostics &injectionDiagnostics() const {
        return lastInjection_;
    }

  private:
    const Mesh &mesh_;
    Mesh::Stamp meshStamp_ = mesh_.stamp();
    InferenceParameterSet paramSet_;
    ObservationOperator obsOp_;
    PreparedObservation preparedObservations_;
    SIMPLESolver::Workspace workspace_;
    FlowBoundaryConditions bcs_;
    double nu_;
    SolverSettings settings_;
    Vec3 Uinit_;
    double pInit_, kInit_, omegaInit_;

    WarmStartCache cache_;
    FlowFields lastFields_;
    bool hasLastFields_ = false;
    bool retainLastFields_ = true;

    std::vector<double> bTarget6_;
    SIMPLESolver::InjectionDiagnostics lastInjection_;
};
