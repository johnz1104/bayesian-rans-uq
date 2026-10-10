#pragma once

#include <cmath>
#include <algorithm>
#include <limits>

// Small analytic tools for the Reynolds-stress anisotropy b_ij, used by the
// a-posteriori Reynolds-stress injection to verify realizability of the target
// anisotropy in the running solve. This is the realizability constraint only;
// the Galilean-invariant feature construction is a separate concern (Python
// UQ.discrepancy) and is deliberately not represented here.
//
// Barycentric map (Banerjee et al. 2007) from the sorted eigenvalues
// l1 >= l2 >= l3 of b:
//   C1c = l1 - l2,   C2c = 2 (l2 - l3),   C3c = 3 l3 + 1,   sum = 1
// For finite, traceless b the realizable states are exactly {C_i >= 0}.

namespace aniso {

// Eigenvalues of a symmetric 3x3 tensor, descending, by the trigonometric
// closed form (Smith 1961): for A = p Q + q I the eigenvalues follow from the
// characteristic equation of the deviator via acos of its normalised
// determinant. Components ordered xx, yy, zz, xy, xz, yz.
inline void sym3Eigenvalues(const double b[6], double lam[3]) {
    const double xx = b[0], yy = b[1], zz = b[2];
    const double xy = b[3], xz = b[4], yz = b[5];

    const double q = (xx + yy + zz) / 3.0;                 // mean eigenvalue
    const double dxx = xx - q, dyy = yy - q, dzz = zz - q; // deviator diagonal
    const double p2 =
        (dxx * dxx + dyy * dyy + dzz * dzz + 2.0 * (xy * xy + xz * xz + yz * yz)) / 6.0;
    const double p = std::sqrt(std::max(p2, 0.0));

    if (p < 1e-30) { // isotropic: triple eigenvalue q
        lam[0] = lam[1] = lam[2] = q;
        return;
    }

    // r = det(deviator/p) / 2, clamped into [-1, 1] against round-off
    const double inv = 1.0 / p;
    const double m00 = dxx * inv, m11 = dyy * inv, m22 = dzz * inv;
    const double m01 = xy * inv, m02 = xz * inv, m12 = yz * inv;
    double r = 0.5 * (m00 * (m11 * m22 - m12 * m12) - m01 * (m01 * m22 - m12 * m02) +
                      m02 * (m01 * m12 - m11 * m02));
    r = std::max(-1.0, std::min(1.0, r));

    const double twoPiOver3 = 2.0943951023931953;
    const double phi = std::acos(r) / 3.0;
    // eigenvalues of A: q + 2 p cos(phi + 2 pi k / 3), k = 0, 1, 2 (descending)
    lam[0] = q + 2.0 * p * std::cos(phi);
    lam[2] = q + 2.0 * p * std::cos(phi + twoPiOver3);
    lam[1] = 3.0 * q - lam[0] - lam[2]; // trace identity
}

// Most negative barycentric coordinate. Membership also requires trace(b)=0.
// Nonfinite inputs or an overflowing eigensolve have margin -infinity.
inline double barycentricMargin(const double b[6]) {
    for (int i = 0; i < 6; ++i)
        if (!std::isfinite(b[i]))
            return -std::numeric_limits<double>::infinity();
    double lam[3];
    sym3Eigenvalues(b, lam);
    for (double v : lam)
        if (!std::isfinite(v))
            return -std::numeric_limits<double>::infinity();
    const double c1 = lam[0] - lam[1];
    const double c2 = 2.0 * (lam[1] - lam[2]);
    const double c3 = 3.0 * lam[2] + 1.0;
    return std::min(c1, std::min(c2, c3));
}

// Absolute, dimensionless violation of trace zero and barycentric positivity.
// Keep the barycentric margin separate so existing interior margins retain
// their meaning; no tensor is projected or modified by this diagnostic.
inline double realizabilityViolation(const double b[6]) {
    const double margin = barycentricMargin(b);
    const double trace = b[0] + b[1] + b[2];
    if (!std::isfinite(margin) || !std::isfinite(trace))
        return std::numeric_limits<double>::infinity();
    return std::max(std::abs(trace), std::max(0.0, -margin));
}

// The tolerance must sit above the trigonometric solver's conditioning floor
// at DEGENERATE eigenvalues: exactly at a barycentric corner the acos argument
// r is 1 and d(acos)/dr is infinite there, so a one-ulp rounding of r (a few
// 1e-16) perturbs the eigenvalues by ~ p sqrt(2 eps) ~ 1e-8. Realizable states
// projected ONTO the simplex boundary (the corners the eigenspace baseline
// uses) therefore carry margins as low as about -2e-8 in exact arithmetic
// terms; 1e-6 clears that floor by two orders while staying far below any
// physically meaningful margin. (Verified against LAPACK eigenvalues, which
// give the corner margin at 1e-16; the floor is a property of the closed-form
// trig method, not of the state.)
inline bool isRealizable(const double b[6], double tol = 1e-6) {
    return std::isfinite(tol) && tol >= 0 && realizabilityViolation(b) <= tol;
}

} // namespace aniso
