#pragma once

#include "LinearSolver.hpp" // linalg::dot / linalg::norm
#include <vector>
#include <functional>
#include <cmath>
#include <stdexcept>
#include <limits>

// Matrix-free Krylov solvers access A only through a callback computing y = A·x.
// BiCGSTAB uses a supplied inverse diagonal for Jacobi preconditioning; FGMRES
// accepts a right-preconditioner callback. The solution vector is both the initial
// guess and the output, so callers may use a cold start or retain a warm start.
// Returned convergence is certified from the true residual of that output.
//
// A tangent application can solve (∂R/∂U) w_j = −∂R/∂θ_j with a directional
// residual difference Jv ≈ [R(U*+εv) − R(U*−εv)]/(2ε). The Krylov solvers do not
// assemble an analytic state Jacobian or adjoint.

struct MatrixFreeResult {
    int iterations = 0;
    double finalRes = 0.0;   // ‖b − A·x‖ / ‖b‖ at return (unit scale for zero b)
    double initialRes = 0.0; // ‖b − A·x_initial‖
    bool converged = false;
    int matvecs = 0; // operator applications (cost accounting)
};

// MatVec: callable void(const std::vector<double>& x, std::vector<double>& y) writing y = A·x.
// invDiag: length-n inverse-diagonal Jacobi preconditioner (z_i = invDiag_i · r_i).  Pass a
// vector of 1.0 for the unpreconditioned solve.
// x is the initial guess AND the output: pass an all-zero x for a cold start (the initial
// matvec is skipped), or a previous solution to WARM-START (the outer SIMPLE-tangent loop
// re-solves with only the RHS changed, so warm-starting from the last increment is cheap).
inline MatrixFreeResult bicgstabMatrixFree(
    const std::function<void(const std::vector<double> &, std::vector<double> &)> &matvec,
    const std::vector<double> &b, std::vector<double> &x, const std::vector<double> &invDiag,
    int maxIter = 1000, double tol = 1e-8) {
    if (b.size() > static_cast<size_t>(std::numeric_limits<int>::max()) || x.size() != b.size() ||
        invDiag.size() != b.size())
        throw std::invalid_argument("matrix-free BiCGSTAB requires matching RHS, initial-vector "
                                    "and inverse-diagonal sizes");
    if (maxIter < 0 || !std::isfinite(tol) || tol <= 0.)
        throw std::invalid_argument(
            "matrix-free BiCGSTAB requires a nonnegative cap and positive finite tolerance");
    const int n = static_cast<int>(b.size());
    MatrixFreeResult res;
    std::vector<double> r(n), rh(n), p(n, 0.0), v(n, 0.0);
    std::vector<double> s(n), t(n), ph(n), sh(n);

    auto precond = [&](const std::vector<double> &in, std::vector<double> &out) {
        for (int i = 0; i < n; ++i)
            out[i] = invDiag[i] * in[i];
    };

    // r0 = b − A·x0.  Cold start (x0 ≈ 0) skips the matvec; warm start pays one.
    if (linalg::norm(x) < 1e-300) {
        r = b;
    } else {
        matvec(x, r);
        ++res.matvecs;
        for (int i = 0; i < n; ++i)
            r[i] = b[i] - r[i];
    }
    double r0 = linalg::norm(r);
    res.initialRes = r0;
    const double bnorm = linalg::norm(b);
    const double scale = bnorm > 1e-300 ? bnorm : 1.;
    const auto certify = [&]() {
        matvec(x, t);
        ++res.matvecs;
        for (int i = 0; i < n; ++i)
            t[i] = b[i] - t[i];
        res.finalRes = linalg::norm(t) / scale;
        res.converged = std::isfinite(res.finalRes) && res.finalRes < tol;
        return res;
    };
    if (r0 < 1e-300) {
        res.converged = true;
        return certify();
    }

    rh = r;
    double rho = 1, alpha = 1, omega = 1;

    for (int it = 0; it < maxIter; ++it) {
        double rhoN = linalg::dot(rh, r);
        if (std::abs(rhoN) < 1e-300)
            break;
        double beta = (rhoN / (rho + 1e-300)) * (alpha / (omega + 1e-300));
        for (int i = 0; i < n; ++i)
            p[i] = r[i] + beta * (p[i] - omega * v[i]);

        precond(p, ph);
        matvec(ph, v);
        ++res.matvecs;
        double rv = linalg::dot(rh, v);
        if (std::abs(rv) < 1e-300)
            break;
        alpha = rhoN / rv;

        for (int i = 0; i < n; ++i)
            s[i] = r[i] - alpha * v[i];
        double sn = linalg::norm(s);
        if (sn / r0 < tol) {
            for (int i = 0; i < n; ++i)
                x[i] += alpha * ph[i];
            res.iterations = it + 1;
            res.finalRes = sn / r0;
            res.converged = true;
            return certify();
        }

        precond(s, sh);
        matvec(sh, t);
        ++res.matvecs;
        double tt = linalg::dot(t, t);
        omega = (tt > 1e-300) ? linalg::dot(t, s) / tt : 0.0;

        for (int i = 0; i < n; ++i)
            x[i] += alpha * ph[i] + omega * sh[i];
        for (int i = 0; i < n; ++i)
            r[i] = s[i] - omega * t[i];

        double rn = linalg::norm(r);
        res.iterations = it + 1;
        res.finalRes = rn / r0;
        if (res.finalRes < tol) {
            res.converged = true;
            return certify();
        }
        if (std::abs(omega) < 1e-300)
            break;
        rho = rhoN;
    }
    return certify();
}

// Reusable storage for sequential RHS columns; never share a workspace between workers.
// basisBytes is the basis payload only, excluding H, scalar work, vector objects and allocator
// overhead.
struct FGMRESWorkspace {
    std::vector<std::vector<double>> V, Z, H;
    std::vector<double> Ax, w, r, cs, sn, g, y;

    static size_t basisBytes(size_t n, int restart) {
        if (restart < 1)
            throw std::invalid_argument("FGMRES restart must be positive");
        const size_t vectors = 2 * static_cast<size_t>(restart) + 1;
        if (n > std::numeric_limits<size_t>::max() / sizeof(double) / vectors)
            throw std::overflow_error("FGMRES basis byte estimate overflows");
        return n * vectors * sizeof(double);
    }
    void resize(size_t n, int restart) {
        const size_t m = static_cast<size_t>(restart);
        V.resize(m + 1);
        Z.resize(m);
        H.resize(m + 1);
        for (auto &v : V)
            v.resize(n);
        for (auto &z : Z)
            z.resize(n);
        for (auto &h : H)
            h.resize(m);
        Ax.resize(n);
        w.resize(n);
        r.resize(n);
        cs.resize(m);
        sn.resize(m);
        g.resize(m + 1);
        y.resize(m);
    }
};

// Flexible restarted FGMRES(m), matrix-free + RIGHT preconditioned.  Stores the preconditioned
// Krylov vectors Z[k] = M⁻¹·V[k] and updates x += Σ Z[k]·y[k], so the preconditioner M⁻¹ may
// VARY between applications (here it runs inner iterative block solves).  Right preconditioning
// minimises the TRUE residual ‖b − A·x‖ (so a singular/weak M⁻¹ cannot collapse the RHS into a
// false-converged zero — the GMRES failure mode).  `matvec` = A (the σ-scaled Jv); `precond` =
// M⁻¹ (the physics-based block-solve SIMPLE preconditioner).  x is the initial guess + output.
inline MatrixFreeResult fgmresMatrixFree(
    const std::function<void(const std::vector<double> &, std::vector<double> &)> &matvec,
    const std::function<void(const std::vector<double> &, std::vector<double> &)> &precond,
    const std::vector<double> &b, std::vector<double> &x, int restart = 60, int maxIter = 1500,
    double tol = 1e-8, FGMRESWorkspace *workspace = nullptr) {
    const int n = (int)b.size();
    if (restart < 1 || maxIter < 0 || !std::isfinite(tol) || tol <= 0 || x.size() != b.size())
        throw std::invalid_argument(
            "FGMRES requires positive restart/tolerance, nonnegative cap and matching vectors");
    const int m = restart;
    MatrixFreeResult res;
    FGMRESWorkspace local;
    FGMRESWorkspace &ws = workspace ? *workspace : local;
    ws.resize(b.size(), m);
    auto &Ax = ws.Ax;
    auto &w = ws.w;
    auto &r = ws.r;
    auto &V = ws.V;
    auto &Z = ws.Z;
    auto &H = ws.H;
    auto &cs = ws.cs;
    auto &sn = ws.sn;
    auto &g = ws.g;
    auto &y = ws.y;

    const double bnorm = linalg::norm(b);
    const double scale = bnorm > 1e-300 ? bnorm : 1.0;
    auto trueResidual = [&]() {
        matvec(x, Ax);
        ++res.matvecs;
        for (int i = 0; i < n; ++i)
            r[i] = b[i] - Ax[i];
        res.finalRes = linalg::norm(r) / scale;
        res.converged = std::isfinite(res.finalRes) && res.finalRes < tol;
    };
    trueResidual();
    res.initialRes = linalg::norm(r);
    if (res.converged || !std::isfinite(res.finalRes))
        return res;

    int total = 0;
    while (total < maxIter) {
        double beta = linalg::norm(r);

        for (int i = 0; i < n; ++i)
            V[0][i] = r[i] / beta;
        std::fill(g.begin(), g.end(), 0.0);
        for (auto &h : H)
            std::fill(h.begin(), h.end(), 0.0);
        g[0] = beta;

        int k = 0;
        bool breakdown = false;
        for (k = 0; k < m && total < maxIter; ++k) {
            precond(V[k], Z[k]); // Z[k] = M⁻¹ V[k]  (flexible)
            matvec(Z[k], w);
            ++res.matvecs; // w = A·Z[k]
            ++total;
            for (int i = 0; i <= k; ++i) {
                H[i][k] = linalg::dot(w, V[i]);
                for (int p = 0; p < n; ++p)
                    w[p] -= H[i][k] * V[i][p];
            }
            H[k + 1][k] = linalg::norm(w);
            breakdown = H[k + 1][k] <= 1e-300;
            if (H[k + 1][k] > 1e-300)
                for (int p = 0; p < n; ++p)
                    V[k + 1][p] = w[p] / H[k + 1][k];
            for (int i = 0; i < k; ++i) {
                double t = cs[i] * H[i][k] + sn[i] * H[i + 1][k];
                H[i + 1][k] = -sn[i] * H[i][k] + cs[i] * H[i + 1][k];
                H[i][k] = t;
            }
            double denom = std::hypot(H[k][k], H[k + 1][k]);
            cs[k] = (denom > 1e-300) ? H[k][k] / denom : 1.0;
            sn[k] = (denom > 1e-300) ? H[k + 1][k] / denom : 0.0;
            H[k][k] = cs[k] * H[k][k] + sn[k] * H[k + 1][k];
            H[k + 1][k] = 0.0;
            g[k + 1] = -sn[k] * g[k];
            g[k] = cs[k] * g[k];
            res.iterations = total;
            if (std::abs(g[k + 1]) / scale < tol || breakdown) {
                ++k;
                break;
            }
        }
        std::fill(y.begin(), y.end(), 0.0);
        for (int i = k - 1; i >= 0; --i) {
            double s = g[i];
            for (int j = i + 1; j < k; ++j)
                s -= H[i][j] * y[j];
            y[i] = (std::abs(H[i][i]) > 1e-300) ? s / H[i][i] : 0.0;
        }
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < k; ++j)
                x[i] += Z[j][i] * y[j]; // flexible: use Z, not V
        // The projected residual is only a stopping candidate. Certify the actual
        // returned x, including happy/unhappy Arnoldi breakdown and iteration caps.
        trueResidual();
        if (res.converged || breakdown || !std::isfinite(res.finalRes))
            return res;
    }
    return res;
}

// GMRES on the supplied operator, with the same true-residual certificate as
// FGMRES. A caller-supplied left-preconditioned operator/RHS is still supported;
// finalRes certifies that supplied system. An identity right preconditioner
// makes the flexible kernel ordinary GMRES while preserving caller warm starts.
inline MatrixFreeResult gmresMatrixFree(
    const std::function<void(const std::vector<double> &, std::vector<double> &)> &matvec,
    const std::vector<double> &b, std::vector<double> &x, int restart = 60, int maxIter = 2000,
    double tol = 1e-8) {
    const auto identity = [](const std::vector<double> &in, std::vector<double> &out) { out = in; };
    return fgmresMatrixFree(matvec, identity, b, x, restart, maxIter, tol);
}
