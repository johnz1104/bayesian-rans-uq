#pragma once

#include <array>
#include <cmath>
#include <algorithm>
#include <functional> // std::greater (libstdc++ does not include it transitively)

// ---------------------------------------------------------------------------
// Barycentric realizability projection for the Reynolds-stress anisotropy.
//
// Realizability and Galilean invariance are kept as two separate operations.
// This module enforces realizability only, by projecting the anisotropy
// eigenvalues into the barycentric triangle; Galilean invariance belongs to the
// integrity-basis construction elsewhere. Conflating the two can destabilize
// the solver, so the projection touches only the eigenvalues and leaves the
// (invariant) eigenvector frame untouched.
//
// Anisotropy:        b_ij = R_ij/(2k) - delta_ij/3,   k = 0.5 R_ii
// Barycentric map (Banerjee et al. 2007) from sorted eigenvalues l1>=l2>=l3:
//   C1c = l1 - l2            (one-component limit)
//   C2c = 2 (l2 - l3)        (two-component limit)
//   C3c = 3 l3 + 1           (three-component / isotropic limit)
//   C1c + C2c + C3c = 1, and the realizable states are exactly the simplex
//   {C_i >= 0, sum C_i = 1}.  Projection = Euclidean projection of (C1c,C2c,C3c)
//   onto that simplex, then reconstruct eigenvalues and rebuild b.
// ---------------------------------------------------------------------------

namespace dbns {

// Symmetric 3x3 tensor stored by its six independent components.
struct Sym3 {
    double xx = 0, yy = 0, zz = 0, xy = 0, xz = 0, yz = 0;

    double trace() const { return xx + yy + zz; }
};

struct RealizabilityProjection {
    // Jacobi eigenvalue algorithm for a symmetric 3x3 matrix.  Returns the
    // three eigenvalues (unsorted) in eval and the corresponding orthonormal
    // eigenvectors as columns of evec (evec[i][j] = component i of eigvec j).
    static void jacobiEigen(const Sym3 &A, double eval[3], double evec[3][3]) {
        double a[3][3] = {{A.xx, A.xy, A.xz}, {A.xy, A.yy, A.yz}, {A.xz, A.yz, A.zz}};
        // initialise eigenvectors to identity
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                evec[i][j] = (i == j) ? 1.0 : 0.0;

        for (int sweep = 0; sweep < 50; ++sweep) {
            // find largest off-diagonal magnitude
            double off = std::abs(a[0][1]) + std::abs(a[0][2]) + std::abs(a[1][2]);
            if (off < 1e-300)
                break;
            for (int p = 0; p < 2; ++p) {
                for (int q = p + 1; q < 3; ++q) {
                    if (std::abs(a[p][q]) < 1e-300)
                        continue;
                    // Rotation angle that zeroes a[p][q] (Golub and Van Loan).
                    double tau = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                    double t =
                        (tau >= 0.0 ? 1.0 : -1.0) / (std::abs(tau) + std::sqrt(1.0 + tau * tau));
                    double c = 1.0 / std::sqrt(1.0 + t * t);
                    double s = t * c;
                    // apply rotation to A
                    for (int i = 0; i < 3; ++i) {
                        double aip = a[i][p], aiq = a[i][q];
                        a[i][p] = c * aip - s * aiq;
                        a[i][q] = s * aip + c * aiq;
                    }
                    for (int i = 0; i < 3; ++i) {
                        double api = a[p][i], aqi = a[q][i];
                        a[p][i] = c * api - s * aqi;
                        a[q][i] = s * api + c * aqi;
                    }
                    // accumulate into eigenvectors
                    for (int i = 0; i < 3; ++i) {
                        double vip = evec[i][p], viq = evec[i][q];
                        evec[i][p] = c * vip - s * viq;
                        evec[i][q] = s * vip + c * viq;
                    }
                }
            }
        }
        eval[0] = a[0][0];
        eval[1] = a[1][1];
        eval[2] = a[2][2];
    }

    // Barycentric coordinates from sorted eigenvalues l1 >= l2 >= l3.
    static std::array<double, 3> barycentric(double l1, double l2, double l3) {
        return {l1 - l2, 2.0 * (l2 - l3), 3.0 * l3 + 1.0};
    }

    // Euclidean projection of a 3-vector onto the probability simplex
    // {x_i >= 0, sum x_i = 1} (Wang and Carreira-Perpinan 2013).
    static std::array<double, 3> projectSimplex(std::array<double, 3> c) {
        std::array<double, 3> u = c;
        std::sort(u.begin(), u.end(), std::greater<double>());
        double cssv = 0.0, theta = 0.0;
        for (int j = 0; j < 3; ++j) {
            cssv += u[j];
            double t = (cssv - 1.0) / (j + 1);
            if (u[j] - t > 0.0) {
                theta = t;
            }
        }
        std::array<double, 3> out{};
        for (int i = 0; i < 3; ++i)
            out[i] = std::max(0.0, c[i] - theta);
        return out;
    }

    // True if the anisotropy eigenvalues lie inside the barycentric triangle
    // (finite, traceless and all barycentric coordinates non-negative within
    // an absolute dimensionless tolerance).
    static bool isRealizable(const Sym3 &b, double tol = 1e-9) {
        if (!std::isfinite(tol) || tol < 0)
            return false;
        for (double v : {b.xx, b.yy, b.zz, b.xy, b.xz, b.yz})
            if (!std::isfinite(v))
                return false;
        if (!std::isfinite(b.trace()) || std::abs(b.trace()) > tol)
            return false;
        double ev[3], evec[3][3];
        jacobiEigen(b, ev, evec);
        for (double v : ev)
            if (!std::isfinite(v))
                return false;
        std::sort(ev, ev + 3, std::greater<double>());
        auto c = barycentric(ev[0], ev[1], ev[2]);
        for (double v : c)
            if (!std::isfinite(v) || v < -tol)
                return false;
        return true;
    }

    // Project an anisotropy tensor b (symmetric, ideally traceless) into the
    // realizable set.  Eigenvectors are preserved; only the eigenvalues move.
    // Returns the projected b; sets distance to the L2 eigenvalue shift.
    static Sym3 projectAnisotropy(const Sym3 &b, double *distance = nullptr) {
        double ev[3], evec[3][3];
        jacobiEigen(b, ev, evec);
        // sort eigenvalues descending, carrying eigenvector columns along
        int idx[3] = {0, 1, 2};
        std::sort(idx, idx + 3, [&](int i, int j) { return ev[i] > ev[j]; });
        double l[3] = {ev[idx[0]], ev[idx[1]], ev[idx[2]]};

        auto c = barycentric(l[0], l[1], l[2]);
        auto cp = projectSimplex(c);

        // reconstruct eigenvalues from projected barycentric coordinates
        double l3 = (cp[2] - 1.0) / 3.0;
        double l2 = l3 + 0.5 * cp[1];
        double l1 = l2 + cp[0];
        double lp[3] = {l1, l2, l3};

        if (distance) {
            double d = 0.0;
            for (int i = 0; i < 3; ++i)
                d += (lp[i] - l[i]) * (lp[i] - l[i]);
            *distance = std::sqrt(d);
        }

        // rebuild b = sum_i lp_i v_i v_i^T using the sorted eigenvectors
        Sym3 out{};
        for (int m = 0; m < 3; ++m) {
            int col = idx[m];
            double vx = evec[0][col], vy = evec[1][col], vz = evec[2][col];
            out.xx += lp[m] * vx * vx;
            out.yy += lp[m] * vy * vy;
            out.zz += lp[m] * vz * vz;
            out.xy += lp[m] * vx * vy;
            out.xz += lp[m] * vx * vz;
            out.yz += lp[m] * vy * vz;
        }
        return out;
    }

    // Convenience: project a full Reynolds-stress tensor R (symmetric, positive
    // semidefinite target) given its trace 2k.  Forms b, projects, rebuilds R.
    static Sym3 projectReynoldsStress(const Sym3 &R, double *distance = nullptr) {
        double twoK = R.trace();
        if (twoK <= 1e-30)
            return R; // no turbulence energy, nothing to do
        double invTwoK = 1.0 / twoK;
        Sym3 b;
        b.xx = R.xx * invTwoK - 1.0 / 3.0;
        b.yy = R.yy * invTwoK - 1.0 / 3.0;
        b.zz = R.zz * invTwoK - 1.0 / 3.0;
        b.xy = R.xy * invTwoK;
        b.xz = R.xz * invTwoK;
        b.yz = R.yz * invTwoK;
        Sym3 bp = projectAnisotropy(b, distance);
        Sym3 out;
        out.xx = twoK * (bp.xx + 1.0 / 3.0);
        out.yy = twoK * (bp.yy + 1.0 / 3.0);
        out.zz = twoK * (bp.zz + 1.0 / 3.0);
        out.xy = twoK * bp.xy;
        out.xz = twoK * bp.xz;
        out.yz = twoK * bp.yz;
        return out;
    }
};

} // namespace dbns
