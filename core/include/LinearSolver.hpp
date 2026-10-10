#pragma once

#include "Mesh.hpp"
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <numeric>
#include <stdexcept>

// LinearSystem - stores Ax = source in a LDU (Lower-Diagonal-Upper) matrix
// upper - owner-neighbor fluxes
// diag - sum of magnitude of all flux coefficients
// lower - neighbor-owner fluxes (upper but negative)
// assumes one equation per cell, cells only interact with cells that it shares a face with, and
// internal faces connect to exactly two cells
struct LinearSystem {
    // nCells = number of control volumes (unknowns)
    // nIF    = number of internal faces (owner–neighbour connections)
    int nCells = 0, nIF = 0;

    // Diagonal entries:
    // diag[i] = A_ii - self influence on cell i
    // size = nCells
    std::vector<double> diag;

    // Off-diagonal entries stored per internal face f:
    // upper[f] = A_ij - neighbor j effects on owner i
    // lower[f] = A_ji - owner i effects on neighbor j
    // size = nIF
    std::vector<double> upper;
    std::vector<double> lower;

    // Right-hand side vector:
    // source[i] = b_i  in A*x = b
    // size = nCells
    std::vector<double> source;

    // Mesh connectivity for each internal face f:
    // own[f] = owner cell index
    // nbr[f] = neighbour cell index
    // size = nIF
    std::vector<int> own;
    std::vector<int> nbr;

    LinearSystem() = default;

    // Constructor: (allocates values to match mesh topology)
    // diag -> size nCells
    // upper/lower → size nInternalFaces
    // source → size nCells
    // own/nbr → size nInternalFaces
    LinearSystem(int nc, int nif)
        : nCells(nc), nIF(nif), diag(nc, 0.0), upper(nif, 0.0), lower(nif, 0.0), source(nc, 0.0),
          own(nif), nbr(nif) {}

    // Solvers admit public mutable matrices before reading indexed storage.
    // Inner matvec/residual kernels require this validated layout and sized,
    // nonaliasing input/output vectors; they do not repeat the whole scan.
    void validate() const {
        if (nCells < 0 || nIF < 0 || diag.size() != static_cast<size_t>(nCells) ||
            source.size() != static_cast<size_t>(nCells) ||
            upper.size() != static_cast<size_t>(nIF) || lower.size() != static_cast<size_t>(nIF) ||
            own.size() != static_cast<size_t>(nIF) || nbr.size() != static_cast<size_t>(nIF))
            throw std::invalid_argument("linear system storage must match its declared dimensions");
        for (int f = 0; f < nIF; ++f)
            if (own[f] < 0 || own[f] >= nCells || nbr[f] < 0 || nbr[f] >= nCells)
                throw std::invalid_argument("linear system connectivity index is out of range");
        for (const auto *values : {&diag, &source, &upper, &lower})
            for (double value : *values)
                if (!std::isfinite(value))
                    throw std::invalid_argument(
                        "linear system coefficients and source must be finite");
    }

    void validateSolve(const std::vector<double> &x, int maxIter, double tol) const {
        validate();
        if (x.size() != static_cast<size_t>(nCells) || maxIter < 0 || !std::isfinite(tol) ||
            tol <= 0.)
            throw std::invalid_argument("linear solve requires matching x, nonnegative iteration "
                                        "cap and positive finite tolerance");
        for (double value : x)
            if (!std::isfinite(value))
                throw std::invalid_argument("linear solve initial values must be finite");
    }

    // resets matrix and RHS to zero
    void zero() {
        std::fill(diag.begin(), diag.end(), 0.0);
        std::fill(upper.begin(), upper.end(), 0.0);
        std::fill(lower.begin(), lower.end(), 0.0);
        std::fill(source.begin(), source.end(), 0.0);
    }

    // y = A * x
    void matvec(const std::vector<double> &x, std::vector<double> &y) const {
        for (int i = 0; i < nCells; ++i)
            y[i] = diag[i] * x[i]; // y_i = A_ii * x_i

        // sparse matrix multiply
        for (int f = 0; f < nIF; ++f) {        // off diag terms
            y[own[f]] += upper[f] * x[nbr[f]]; // y_i+ = A_ij * x_j
            y[nbr[f]] += lower[f] * x[own[f]]; // y_j+ = A_ji * x_i
        }
    }

    // residual r = source - A*x
    void residual(const std::vector<double> &x, std::vector<double> &r) const {
        matvec(x, r);
        for (int i = 0; i < nCells; ++i)
            r[i] = source[i] - r[i];
    }
};

// creates linear system with mesh topology
inline LinearSystem makeSystem(const Mesh &m) {
    LinearSystem A(m.nCells(), m.nInternalFaces());
    const auto &ol = m.ownerList();
    const auto &nl = m.neighborList();
    for (int f = 0; f < m.nInternalFaces(); ++f) {
        A.own[f] = ol[f];
        A.nbr[f] = nl[f];
    }
    return A;
}

// result container
// initialRes tracks the absolute equation imbalance (r0 = |b - Ax0|) before the linear solve
// this is used for SIMPLE outer-loop convergence instead of finalRes (= rn/r0), which measures
// only the relative drop within a single linear solve
// with uniform initial conditions the assembled systems have near-zero RHS,
// so finalRes drops below tolerance immediately at iter 0, causing false convergence.
// initialRes avoids this by measuring the actual equation residual
struct SolverResult {
    int iterations = 0;
    double finalRes = 0.0; // relative residual (rn/r0) — used internally by linear solver
    double initialRes =
        0.0; // absolute initial residual (r0 = |b - Ax0|) — used for outer convergence
    bool converged = false;
};

// interface linear solver
// functions all linear solver must implement
class ILinearSolver {
  public:
    virtual ~ILinearSolver() = default;
    virtual SolverResult solve(const LinearSystem &A, std::vector<double> &x, int maxIter,
                               double tol) = 0;
    virtual std::string name() const = 0;
};

// helpers
namespace linalg {
// dot product
inline double dot(const std::vector<double> &a, const std::vector<double> &b) {
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
        s += a[i] * b[i];
    return s;
}
// norm
inline double norm(const std::vector<double> &v) {
    return std::sqrt(dot(v, v));
}
// Jacobi Preconditioner
// args:
// A - Matrix system
// r - residual vector (difference between observed and predicted data)
// z - output vector storing preconditioned residual (z = diag(A)^-1 r)
inline void jacobiPrecond(const LinearSystem &A, const std::vector<double> &r,
                          std::vector<double> &z) {
    for (int i = 0; i < A.nCells; ++i)
        z[i] = (std::abs(A.diag[i]) > 1e-30) ? r[i] / A.diag[i] : r[i];
}
} // namespace linalg

// Solvers

// Preconditioned Conjugate Gradient
// used for large spare symmetric positive definite matrices (pressure Poisson)
class PCGSolver : public ILinearSolver {
    std::vector<double> r_, z_, p_, Ap_;

  public:
    size_t workspaceCapacityBytes() const {
        return sizeof(double) * (r_.capacity() + z_.capacity() + p_.capacity() + Ap_.capacity());
    }
    SolverResult solve(const LinearSystem &A, std::vector<double> &x, int maxIter = 500,
                       double tol = 1e-6) override {
        A.validateSolve(x, maxIter, tol);
        int n = A.nCells;
        r_.resize(n);
        z_.resize(n);
        p_.resize(n);
        Ap_.resize(n);
        auto &r = r_;
        auto &z = z_;
        auto &p = p_;
        auto &Ap = Ap_;
        SolverResult res;

        A.residual(x, r);
        double r0 = linalg::norm(r);
        res.initialRes = r0;
        if (r0 < 1e-30) {
            res.converged = true;
            return res;
        }

        linalg::jacobiPrecond(A, r, z);
        p = z;
        double rz = linalg::dot(r, z);

        for (int it = 0; it < maxIter; ++it) {
            A.matvec(p, Ap);
            double pAp = linalg::dot(p, Ap);
            if (std::abs(pAp) < 1e-30)
                break;
            double alpha = rz / pAp;
            for (int i = 0; i < n; ++i) {
                x[i] += alpha * p[i];
                r[i] -= alpha * Ap[i];
            }

            double rn = linalg::norm(r);
            res.iterations = it + 1;
            res.finalRes = rn / r0;
            if (res.finalRes < tol) {
                res.converged = true;
                return res;
            }

            linalg::jacobiPrecond(A, r, z);
            double rzNew = linalg::dot(r, z);
            double beta = rzNew / (rz + 1e-30);
            for (int i = 0; i < n; ++i)
                p[i] = z[i] + beta * p[i];
            rz = rzNew;
        }
        return res;
    }
    std::string name() const override { return "PCG"; }
};

// BiConjugate Gradient Stabilized
// iterative Krylov subspace method
// used for large sparse asymmetric matrices (momentum, turbulence)
class BiCGSTABSolver : public ILinearSolver {
    std::vector<double> r_, rh_, p_, v_, s_, t_, ph_, sh_;

  public:
    size_t workspaceCapacityBytes() const {
        return sizeof(double) * (r_.capacity() + rh_.capacity() + p_.capacity() + v_.capacity() +
                                 s_.capacity() + t_.capacity() + ph_.capacity() + sh_.capacity());
    }
    SolverResult solve(const LinearSystem &A, std::vector<double> &x, int maxIter = 500,
                       double tol = 1e-6) override {
        A.validateSolve(x, maxIter, tol);
        int n = A.nCells;
        r_.resize(n);
        rh_.resize(n);
        p_.resize(n);
        v_.resize(n);
        s_.resize(n);
        t_.resize(n);
        ph_.resize(n);
        sh_.resize(n);
        auto &r = r_;
        auto &rh = rh_;
        auto &p = p_;
        auto &v = v_;
        auto &s = s_;
        auto &t = t_;
        auto &ph = ph_;
        auto &sh = sh_;
        // These two recurrence vectors are read before being overwritten.
        // Reset them even after exact starts, zero caps or breakdowns.
        std::fill(p.begin(), p.end(), 0.0);
        std::fill(v.begin(), v.end(), 0.0);
        SolverResult res;

        A.residual(x, r);
        double r0 = linalg::norm(r);
        res.initialRes = r0;
        if (r0 < 1e-30) {
            res.converged = true;
            return res;
        }

        rh = r;
        double rho = 1, alpha = 1, omega = 1;

        for (int it = 0; it < maxIter; ++it) {
            double rhoN = linalg::dot(rh, r);
            if (std::abs(rhoN) < 1e-30)
                break;
            double beta = (rhoN / (rho + 1e-30)) * (alpha / (omega + 1e-30));
            for (int i = 0; i < n; ++i)
                p[i] = r[i] + beta * (p[i] - omega * v[i]);

            linalg::jacobiPrecond(A, p, ph);
            A.matvec(ph, v);
            double rv = linalg::dot(rh, v);
            if (std::abs(rv) < 1e-30)
                break;
            alpha = rhoN / rv;

            for (int i = 0; i < n; ++i)
                s[i] = r[i] - alpha * v[i];
            double sn = linalg::norm(s);
            if (sn / r0 < tol) {
                for (int i = 0; i < n; ++i)
                    x[i] += alpha * ph[i];
                res = {it + 1, sn / r0, r0, true};
                return res;
            }

            linalg::jacobiPrecond(A, s, sh);
            A.matvec(sh, t);
            double tt = linalg::dot(t, t);
            omega = (tt > 1e-30) ? linalg::dot(t, s) / tt : 0.0;

            for (int i = 0; i < n; ++i)
                x[i] += alpha * ph[i] + omega * sh[i];
            for (int i = 0; i < n; ++i)
                r[i] = s[i] - omega * t[i];

            double rn = linalg::norm(r);
            res.iterations = it + 1;
            res.finalRes = rn / r0;
            if (res.finalRes < tol) {
                res.converged = true;
                return res;
            }
            if (std::abs(omega) < 1e-30)
                break;
            rho = rhoN;
        }
        return res;
    }
    std::string name() const override { return "BiCGSTAB"; }
};

// Gauss-Seidel iterative solver: each unknown is updated in place using the
// latest values of its neighbours within the same sweep. The face-based LDU
// storage is converted to per-cell adjacency once per solve so the sweep
// can read updated neighbours.
class GaussSeidelSolver : public ILinearSolver {
  public:
    SolverResult solve(const LinearSystem &A, std::vector<double> &x, int maxIter = 500,
                       double tol = 1e-6) override {
        A.validateSolve(x, maxIter, tol);
        int n = A.nCells;
        std::vector<double> r(n);
        SolverResult res;
        double b0 = linalg::norm(A.source);
        // initialRes is |b - A x0| like every other solver (it was |b|, which
        // is the residual only for a zero initial guess)
        A.residual(x, r);
        res.initialRes = linalg::norm(r);
        if (b0 < 1e-30)
            b0 = 1.0;

        // per-cell adjacency (neighbour index, off-diagonal coefficient)
        std::vector<int> deg(n, 0);
        for (int f = 0; f < A.nIF; ++f) {
            ++deg[A.own[f]];
            ++deg[A.nbr[f]];
        }
        std::vector<int> off(n + 1, 0);
        for (int i = 0; i < n; ++i)
            off[i + 1] = off[i] + deg[i];
        std::vector<int> adjIdx(off[n]);
        std::vector<double> adjCoef(off[n]);
        std::vector<int> cursor(off.begin(), off.end() - 1);
        for (int f = 0; f < A.nIF; ++f) {
            int o = A.own[f], nb = A.nbr[f];
            adjIdx[cursor[o]] = nb;
            adjCoef[cursor[o]++] = A.upper[f];
            adjIdx[cursor[nb]] = o;
            adjCoef[cursor[nb]++] = A.lower[f];
        }

        for (int it = 0; it < maxIter; ++it) {
            for (int i = 0; i < n; ++i) {
                if (std::abs(A.diag[i]) <= 1e-30)
                    continue;
                double s = A.source[i];
                for (int a = off[i]; a < off[i + 1]; ++a)
                    s -= adjCoef[a] * x[adjIdx[a]]; // latest in-sweep values
                x[i] = s / A.diag[i];
            }
            A.residual(x, r);
            res.iterations = it + 1;
            res.finalRes = linalg::norm(r) / b0;
            if (res.finalRes < tol) {
                res.converged = true;
                return res;
            }
        }
        return res;
    }
    std::string name() const override { return "GaussSeidel"; }
};

// Algebraic Multigrid (AMG) preconditioned PCG solver
// general sparse matrix in CSR format for coarse AMG levels
struct CSRMatrix {
    int nRows = 0;
    std::vector<int> rowPtr;
    std::vector<int> colIdx;
    std::vector<double> values;
    std::vector<double> diag;

    void matvec(const std::vector<double> &x, std::vector<double> &y) const {
        for (int i = 0; i < nRows; ++i) {
            double s = 0.0;
            for (int j = rowPtr[i]; j < rowPtr[i + 1]; ++j)
                s += values[j] * x[colIdx[j]];
            y[i] = s;
        }
    }

    void residual(const std::vector<double> &x, const std::vector<double> &b,
                  std::vector<double> &r) const {
        for (int i = 0; i < nRows; ++i) {
            double s = 0.0;
            for (int j = rowPtr[i]; j < rowPtr[i + 1]; ++j)
                s += values[j] * x[colIdx[j]];
            r[i] = b[i] - s;
        }
    }

    void gaussSeidelSweep(std::vector<double> &x, const std::vector<double> &b) const {
        for (int i = 0; i < nRows; ++i) {
            double s = b[i];
            for (int j = rowPtr[i]; j < rowPtr[i + 1]; ++j) {
                if (colIdx[j] != i)
                    s -= values[j] * x[colIdx[j]];
            }
            if (std::abs(diag[i]) > 1e-30)
                x[i] = s / diag[i];
        }
    }

    // the same sweep in reverse row order: forward pre-smoothing followed by
    // backward post-smoothing makes the V-cycle a symmetric operator
    void gaussSeidelSweepBackward(std::vector<double> &x, const std::vector<double> &b) const {
        for (int i = nRows - 1; i >= 0; --i) {
            double s = b[i];
            for (int j = rowPtr[i]; j < rowPtr[i + 1]; ++j) {
                if (colIdx[j] != i)
                    s -= values[j] * x[colIdx[j]];
            }
            if (std::abs(diag[i]) > 1e-30)
                x[i] = s / diag[i];
        }
    }

    // position of column c in row r (rows are sorted), -1 if absent
    int find(int r, int c) const {
        auto first = colIdx.begin() + rowPtr[r], last = colIdx.begin() + rowPtr[r + 1];
        auto it = std::lower_bound(first, last, c);
        return (it != last && *it == c) ? static_cast<int>(it - colIdx.begin()) : -1;
    }
};

// Convert LDU LinearSystem to CSR format
inline CSRMatrix lduToCSR(const LinearSystem &A) {
    int n = A.nCells;
    // Count entries per row (diagonal + off-diagonals)
    std::vector<int> count(n, 1); // diagonal
    for (int f = 0; f < A.nIF; ++f) {
        count[A.own[f]]++;
        count[A.nbr[f]]++;
    }

    CSRMatrix csr;
    csr.nRows = n;
    csr.rowPtr.resize(n + 1, 0);
    for (int i = 0; i < n; ++i)
        csr.rowPtr[i + 1] = csr.rowPtr[i] + count[i];
    int nnz = csr.rowPtr[n];
    csr.colIdx.resize(nnz);
    csr.values.resize(nnz);
    csr.diag.resize(n);

    // build unsorted entries per row using fill pointers
    std::vector<int> pos(n);
    for (int i = 0; i < n; ++i)
        pos[i] = csr.rowPtr[i];

    // diagonal first
    for (int i = 0; i < n; ++i) {
        csr.colIdx[pos[i]] = i;
        csr.values[pos[i]] = A.diag[i];
        csr.diag[i] = A.diag[i];
        pos[i]++;
    }
    // off-diagonals
    for (int f = 0; f < A.nIF; ++f) {
        int o = A.own[f], nb = A.nbr[f];
        csr.colIdx[pos[o]] = nb;
        csr.values[pos[o]] = A.upper[f];
        pos[o]++;
        csr.colIdx[pos[nb]] = o;
        csr.values[pos[nb]] = A.lower[f];
        pos[nb]++;
    }

    // sort each row by column index for consistent access
    for (int i = 0; i < n; ++i) {
        int start = csr.rowPtr[i], end = csr.rowPtr[i + 1];
        // simple insertion sort (rows are small)
        for (int a = start + 1; a < end; ++a) {
            int key_c = csr.colIdx[a];
            double key_v = csr.values[a];
            int b = a - 1;
            while (b >= start && csr.colIdx[b] > key_c) {
                csr.colIdx[b + 1] = csr.colIdx[b];
                csr.values[b + 1] = csr.values[b];
                b--;
            }
            csr.colIdx[b + 1] = key_c;
            csr.values[b + 1] = key_v;
        }
    }
    return csr;
}

// greedy aggregation coarsening with strength-of-connection filter
inline void buildAggregates(const CSRMatrix &A, std::vector<int> &map, int &nCoarse) {
    int n = A.nRows;
    map.assign(n, -1);
    nCoarse = 0;

    // Pass 1: seed aggregates from unaggregated cells
    for (int i = 0; i < n; ++i) {
        if (map[i] >= 0)
            continue;
        int agg = nCoarse++;
        map[i] = agg;
        double aii = std::abs(A.diag[i]);
        for (int j = A.rowPtr[i]; j < A.rowPtr[i + 1]; ++j) {
            int c = A.colIdx[j];
            if (c == i || map[c] >= 0)
                continue;
            double ajj = std::abs(A.diag[c]);
            double strength = std::abs(A.values[j]) / std::sqrt(aii * ajj + 1e-30);
            if (strength > 0.25)
                map[c] = agg;
        }
    }

    // Pass 2: assign orphans to strongest neighbor's aggregate
    for (int i = 0; i < n; ++i) {
        if (map[i] >= 0)
            continue;
        double bestStr = -1.0;
        int bestAgg = 0;
        for (int j = A.rowPtr[i]; j < A.rowPtr[i + 1]; ++j) {
            int c = A.colIdx[j];
            if (c == i && map[c] < 0)
                continue;
            if (map[c] >= 0 && std::abs(A.values[j]) > bestStr) {
                bestStr = std::abs(A.values[j]);
                bestAgg = map[c];
            }
        }
        map[i] = bestAgg;
    }
}

// Galerkin coarse matrix: A_c = R * A * P (piecewise-constant prolongation)
inline CSRMatrix galerkinProduct(const CSRMatrix &A, const std::vector<int> &map, int nCoarse) {
    // Since P[i, map[i]] = 1: A_c[I,J] = sum of A[i,j] for i in I, j in J
    // Use map to accumulate
    std::vector<std::unordered_map<int, double>> rows(nCoarse);
    for (int i = 0; i < A.nRows; ++i) {
        int I = map[i];
        for (int j = A.rowPtr[i]; j < A.rowPtr[i + 1]; ++j) {
            int J = map[A.colIdx[j]];
            rows[I][J] += A.values[j];
        }
    }

    // convert to CSR
    CSRMatrix Ac;
    Ac.nRows = nCoarse;
    Ac.rowPtr.resize(nCoarse + 1, 0);
    Ac.diag.resize(nCoarse, 0.0);

    int nnz = 0;
    for (int I = 0; I < nCoarse; ++I)
        nnz += (int)rows[I].size();
    Ac.colIdx.resize(nnz);
    Ac.values.resize(nnz);

    int pos = 0;
    for (int I = 0; I < nCoarse; ++I) {
        Ac.rowPtr[I] = pos;
        // collect and sort entries
        std::vector<std::pair<int, double>> entries(rows[I].begin(), rows[I].end());
        std::sort(entries.begin(), entries.end());
        for (auto &[col, val] : entries) {
            Ac.colIdx[pos] = col;
            Ac.values[pos] = val;
            if (col == I)
                Ac.diag[I] = val;
            pos++;
        }
    }
    Ac.rowPtr[nCoarse] = pos;
    return Ac;
}

// AMG level data
struct AMGLevel {
    CSRMatrix A;
    std::vector<int> aggregateMap;
    int nCoarse = 0;
    // galerkinIdx[k]: position in the next level's CSR that fine entry k sums
    // into (P is piecewise constant, so A_c[I,J] = sum of A[i,j] over the
    // aggregates). Fixed by the aggregation, so a coefficient change is an
    // O(nnz) re-summation instead of a rebuild.
    std::vector<int> galerkinIdx;
    std::vector<int> diagPos; // position of (i,i) in this level's CSR
    // working vectors
    std::vector<double> x, b, r;
};

// AMG-preconditioned PCG solver
//
// Setup is done once per sparsity pattern and kept across solves. Between
// SIMPLE iterations only the pressure-matrix coefficients change (through
// V/a_P), so later solves refresh the values of the frozen hierarchy in
// O(nnz): a scatter into the cached level-0 CSR and a re-summation of each
// Galerkin product. The aggregation, which depends on the coefficients only
// through a strength threshold, is rebuilt when the PCG iteration count of a
// refreshed solve exceeds 1.5x (+2) the count right after the last full
// setup, so a hierarchy that has drifted away from the matrix is replaced
// rather than kept. The V-cycle uses forward Gauss-Seidel before and backward
// Gauss-Seidel after the coarse correction, which makes the preconditioner
// symmetric as CG requires (forward sweeps on both sides did not), and the
// coarsest level of at most 400 unknowns is solved exactly by dense LU.
class AMGSolver : public ILinearSolver {
  public:
    // cumulative wall-clock split between hierarchy construction (full setups
    // and value refreshes) and the preconditioned iteration, across every
    // solve this instance performed
    double setupSeconds = 0.0;
    double iterateSeconds = 0.0;
    int fullSetups = 0;
    int refreshes = 0;

    // z = M^-1 r for the current hierarchy (exposed so tests can check that
    // the V-cycle is symmetric)
    void applyPreconditioner(const std::vector<double> &r, std::vector<double> &z) {
        precondition(r, z);
    }

  private:
    static constexpr int maxLevels_ = 20;
    static constexpr int minCoarseSize_ = 50;
    static constexpr int maxDirectSize_ = 400;
    static constexpr int nPreSmooth_ = 2;
    static constexpr int nPostSmooth_ = 2;
    static constexpr int coarseSweeps_ = 50;

    std::vector<AMGLevel> levels_;

    // cached LDU -> level-0 CSR scatter map, valid for one sparsity pattern
    std::vector<int> patternOwn_, patternNbr_;
    std::vector<int> upperPos_, lowerPos_;
    bool mappable_ = false; // false if a row holds a repeated column
    bool needFullSetup_ = true;
    int itersAfterSetup_ = -1;

    // dense LU of the coarsest level (partial pivoting), when it is small
    std::vector<double> lu_;
    std::vector<int> piv_;
    bool coarseDirect_ = false;

    bool samePattern(const LinearSystem &A) const {
        return !levels_.empty() && levels_[0].A.nRows == A.nCells && patternOwn_ == A.own &&
               patternNbr_ == A.nbr;
    }

    static void computeDiagPos(AMGLevel &lev) {
        lev.diagPos.assign(lev.A.nRows, -1);
        for (int i = 0; i < lev.A.nRows; ++i)
            lev.diagPos[i] = lev.A.find(i, i);
    }

    // coarse values from the fine values through the fixed Galerkin map
    static void sumCoarse(const AMGLevel &fine, AMGLevel &coarse) {
        std::fill(coarse.A.values.begin(), coarse.A.values.end(), 0.0);
        const int nnz = static_cast<int>(fine.A.values.size());
        for (int k = 0; k < nnz; ++k)
            coarse.A.values[fine.galerkinIdx[k]] += fine.A.values[k];
        for (int I = 0; I < coarse.A.nRows; ++I)
            coarse.A.diag[I] = coarse.diagPos[I] >= 0 ? coarse.A.values[coarse.diagPos[I]] : 0.0;
    }

    void factorCoarsest() {
        const AMGLevel &lev = levels_.back();
        const int n = lev.A.nRows;
        coarseDirect_ = n <= maxDirectSize_;
        if (!coarseDirect_)
            return;
        lu_.assign(static_cast<size_t>(n) * n, 0.0);
        piv_.resize(n);
        for (int i = 0; i < n; ++i)
            for (int j = lev.A.rowPtr[i]; j < lev.A.rowPtr[i + 1]; ++j)
                lu_[static_cast<size_t>(i) * n + lev.A.colIdx[j]] += lev.A.values[j];
        for (int k = 0; k < n; ++k) {
            int p = k;
            for (int i = k + 1; i < n; ++i)
                if (std::abs(lu_[static_cast<size_t>(i) * n + k]) >
                    std::abs(lu_[static_cast<size_t>(p) * n + k]))
                    p = i;
            piv_[k] = p;
            if (std::abs(lu_[static_cast<size_t>(p) * n + k]) < 1e-300) {
                coarseDirect_ = false;
                return;
            }
            if (p != k)
                for (int j = 0; j < n; ++j)
                    std::swap(lu_[static_cast<size_t>(k) * n + j],
                              lu_[static_cast<size_t>(p) * n + j]);
            const double inv = 1.0 / lu_[static_cast<size_t>(k) * n + k];
            for (int i = k + 1; i < n; ++i) {
                double &lik = lu_[static_cast<size_t>(i) * n + k];
                lik *= inv;
                if (lik != 0.0)
                    for (int j = k + 1; j < n; ++j)
                        lu_[static_cast<size_t>(i) * n + j] -=
                            lik * lu_[static_cast<size_t>(k) * n + j];
            }
        }
    }

    void solveCoarsest(AMGLevel &lev) {
        const int n = lev.A.nRows;
        if (!coarseDirect_) {
            // symmetric pairs of sweeps keep the V-cycle symmetric
            for (int s = 0; s < coarseSweeps_ / 2; ++s) {
                lev.A.gaussSeidelSweep(lev.x, lev.b);
                lev.A.gaussSeidelSweepBackward(lev.x, lev.b);
            }
            return;
        }
        std::vector<double> &y = lev.x;
        y = lev.b;
        for (int k = 0; k < n; ++k)
            std::swap(y[k], y[piv_[k]]);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < i; ++j)
                y[i] -= lu_[static_cast<size_t>(i) * n + j] * y[j];
        for (int i = n - 1; i >= 0; --i) {
            for (int j = i + 1; j < n; ++j)
                y[i] -= lu_[static_cast<size_t>(i) * n + j] * y[j];
            y[i] /= lu_[static_cast<size_t>(i) * n + i];
        }
    }

    void fullSetup(const LinearSystem &A) {
        levels_.clear();

        // Level 0: finest level from LDU, plus the scatter map for refreshes
        AMGLevel lev0;
        lev0.A = lduToCSR(A);
        computeDiagPos(lev0);
        upperPos_.assign(A.nIF, -1);
        lowerPos_.assign(A.nIF, -1);
        mappable_ = true;
        for (int i = 0; i < lev0.A.nRows && mappable_; ++i)
            for (int j = lev0.A.rowPtr[i] + 1; j < lev0.A.rowPtr[i + 1]; ++j)
                if (lev0.A.colIdx[j] == lev0.A.colIdx[j - 1]) {
                    mappable_ = false;
                    break;
                }
        if (mappable_)
            for (int f = 0; f < A.nIF; ++f) {
                upperPos_[f] = lev0.A.find(A.own[f], A.nbr[f]);
                lowerPos_[f] = lev0.A.find(A.nbr[f], A.own[f]);
            }
        patternOwn_ = A.own;
        patternNbr_ = A.nbr;
        levels_.push_back(std::move(lev0));

        // coarsen until small enough
        for (int l = 0; l < maxLevels_; ++l) {
            auto &fine = levels_[l];
            if (fine.A.nRows <= minCoarseSize_)
                break;

            std::vector<int> map;
            int nc;
            buildAggregates(fine.A, map, nc);

            // stop if coarsening stalls
            if (nc == 0 || nc >= fine.A.nRows)
                break;

            fine.aggregateMap = std::move(map);
            fine.nCoarse = nc;

            AMGLevel coarse;
            coarse.A = galerkinProduct(fine.A, fine.aggregateMap, fine.nCoarse);
            computeDiagPos(coarse);
            fine.galerkinIdx.resize(fine.A.values.size());
            for (int i = 0; i < fine.A.nRows; ++i)
                for (int j = fine.A.rowPtr[i]; j < fine.A.rowPtr[i + 1]; ++j)
                    fine.galerkinIdx[j] =
                        coarse.A.find(fine.aggregateMap[i], fine.aggregateMap[fine.A.colIdx[j]]);
            sumCoarse(fine, coarse); // same summation order as a refresh
            levels_.push_back(std::move(coarse));
        }

        // allocate working vectors
        for (auto &lev : levels_) {
            int n = lev.A.nRows;
            lev.x.assign(n, 0.0);
            lev.b.assign(n, 0.0);
            lev.r.assign(n, 0.0);
        }
        factorCoarsest();
        ++fullSetups;
        needFullSetup_ = false;
        itersAfterSetup_ = -1;
    }

    void refreshValues(const LinearSystem &A) {
        AMGLevel &lev0 = levels_[0];
        for (int i = 0; i < A.nCells; ++i) {
            lev0.A.values[lev0.diagPos[i]] = A.diag[i];
            lev0.A.diag[i] = A.diag[i];
        }
        for (int f = 0; f < A.nIF; ++f) {
            lev0.A.values[upperPos_[f]] = A.upper[f];
            lev0.A.values[lowerPos_[f]] = A.lower[f];
        }
        for (size_t l = 0; l + 1 < levels_.size(); ++l)
            sumCoarse(levels_[l], levels_[l + 1]);
        factorCoarsest();
        ++refreshes;
    }

    void vcycle(int level) {
        auto &lev = levels_[level];
        int n = lev.A.nRows;

        if (level == (int)levels_.size() - 1) {
            solveCoarsest(lev);
            return;
        }

        // pre-smooth (forward)
        for (int s = 0; s < nPreSmooth_; ++s)
            lev.A.gaussSeidelSweep(lev.x, lev.b);

        // compute residual
        lev.A.residual(lev.x, lev.b, lev.r);

        // restrict residual to coarse level (R = P^T, piecewise-constant)
        auto &coarse = levels_[level + 1];
        std::fill(coarse.b.begin(), coarse.b.end(), 0.0);
        std::fill(coarse.x.begin(), coarse.x.end(), 0.0);
        for (int i = 0; i < n; ++i)
            coarse.b[lev.aggregateMap[i]] += lev.r[i];

        // recurse
        vcycle(level + 1);

        // prolongate correction and add (P is piecewise-constant)
        for (int i = 0; i < n; ++i)
            lev.x[i] += coarse.x[lev.aggregateMap[i]];

        // post-smooth (backward: the adjoint of the pre-smoother)
        for (int s = 0; s < nPostSmooth_; ++s)
            lev.A.gaussSeidelSweepBackward(lev.x, lev.b);
    }

    // apply AMG as preconditioner: solve M*z ≈ r
    void precondition(const std::vector<double> &r, std::vector<double> &z) {
        auto &lev0 = levels_[0];
        lev0.b = r;
        std::fill(lev0.x.begin(), lev0.x.end(), 0.0);
        vcycle(0);
        z = lev0.x;
    }

  public:
    // Prepare the hierarchy for A without solving: a full setup for a new
    // pattern (or when the last refreshed solve degraded), else a refresh.
    void prepare(const LinearSystem &A) {
        A.validate();
        if (needFullSetup_ || !mappable_ || !samePattern(A))
            fullSetup(A);
        else
            refreshValues(A);
    }

    SolverResult solve(const LinearSystem &A, std::vector<double> &x, int maxIter = 500,
                       double tol = 1e-6) override {
        A.validateSolve(x, maxIter, tol);
        const auto tSetup0 = std::chrono::steady_clock::now();
        const bool wasFull = needFullSetup_ || !mappable_ || !samePattern(A);
        prepare(A);
        const auto tSetup1 = std::chrono::steady_clock::now();
        setupSeconds += std::chrono::duration<double>(tSetup1 - tSetup0).count();

        int n = A.nCells;
        std::vector<double> r(n), z(n), p(n), Ap(n);
        SolverResult res;

        // everything after setup is the iteration phase; the destructor
        // closes the account on every return path and judges whether the
        // frozen hierarchy still serves
        struct IterTimer {
            AMGSolver *s;
            bool full;
            const SolverResult &res;
            std::chrono::steady_clock::time_point t0;
            IterTimer(AMGSolver *self, bool f, const SolverResult &r)
                : s(self), full(f), res(r), t0(std::chrono::steady_clock::now()) {}
            ~IterTimer() {
                s->iterateSeconds +=
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                // the reference count is the first nontrivial solve after a
                // full setup; later refreshed solves are judged against it
                if (full || s->itersAfterSetup_ <= 0)
                    s->itersAfterSetup_ = res.iterations;
                else if (!res.converged || res.iterations > 1.5 * s->itersAfterSetup_ + 2)
                    s->needFullSetup_ = true;
            }
        } iterTimer(this, wasFull, res);

        A.residual(x, r);
        double r0 = linalg::norm(r);
        res.initialRes = r0;
        if (r0 < 1e-30) {
            res.converged = true;
            return res;
        }

        precondition(r, z);
        p = z;
        double rz = linalg::dot(r, z);

        for (int it = 0; it < maxIter; ++it) {
            A.matvec(p, Ap);
            double pAp = linalg::dot(p, Ap);
            if (std::abs(pAp) < 1e-30)
                break;
            double alpha = rz / pAp;
            for (int i = 0; i < n; ++i) {
                x[i] += alpha * p[i];
                r[i] -= alpha * Ap[i];
            }

            double rn = linalg::norm(r);
            res.iterations = it + 1;
            res.finalRes = rn / r0;
            if (res.finalRes < tol) {
                res.converged = true;
                return res;
            }

            precondition(r, z);
            double rzNew = linalg::dot(r, z);
            double beta = rzNew / (rz + 1e-30);
            for (int i = 0; i < n; ++i)
                p[i] = z[i] + beta * p[i];
            rz = rzNew;
        }
        return res;
    }

    std::string name() const override { return "AMG"; }
};

// Factory Function
inline std::unique_ptr<ILinearSolver> makeSolver(const std::string &name) {
    if (name == "PCG")
        return std::make_unique<PCGSolver>();
    if (name == "BiCGSTAB")
        return std::make_unique<BiCGSTABSolver>();
    if (name == "GaussSeidel")
        return std::make_unique<GaussSeidelSolver>();
    if (name == "AMG")
        return std::make_unique<AMGSolver>();
    throw std::invalid_argument("unknown linear solver: '" + name + "'");
}
