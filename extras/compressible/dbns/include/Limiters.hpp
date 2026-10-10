#pragma once

#include <algorithm>
#include <cmath>

// ---------------------------------------------------------------------------
// Slope limiters for MUSCL reconstruction.
//
// Two families are provided:
//   1. Classical scalar TVD limiter functions psi(r) of the consecutive-slope
//      ratio r, used for one-dimensional / structured reconstruction and for
//      the limiter-property unit tests (symmetry psi(r)/r = psi(1/r), the
//      Sweby TVD region, psi(r<=0)=0).
//   2. The Barth-Jespersen / Venkatakrishnan multidimensional limiter scalar,
//      used by the cell-gradient reconstruction. Its sharp branch enforces
//      local neighbour bounds; positive smoothing relaxes those bounds.
//
// References: Sweby (1984); van Leer (1979); Barth and Jespersen (1989);
// Venkatakrishnan (1993).
// ---------------------------------------------------------------------------

namespace dbns {

enum class LimiterKind { None, MinMod, VanLeer, VanAlbada, Superbee };

struct Limiters {
    // psi(r): scalar TVD flux/slope limiter.  r is the ratio of the upwind to
    // local solution gradient. Except for None, these return 0 for r <= 0 and
    // satisfy the symmetry property psi(r) = r*psi(1/r) for positive r.
    static double psi(LimiterKind kind, double r) {
        switch (kind) {
        case LimiterKind::None:
            return 1.0;
        case LimiterKind::MinMod:
            return std::max(0.0, std::min(1.0, r));
        case LimiterKind::VanLeer: {
            // van Leer: (r + |r|) / (1 + |r|)
            double ar = std::abs(r);
            return (r + ar) / (1.0 + ar);
        }
        case LimiterKind::VanAlbada: {
            // van Albada: (r^2 + r) / (r^2 + 1), clipped at 0 for r <= 0
            if (r <= 0.0)
                return 0.0;
            return (r * r + r) / (r * r + 1.0);
        }
        case LimiterKind::Superbee:
            return std::max({0.0, std::min(2.0 * r, 1.0), std::min(r, 2.0)});
        }
        return 1.0;
    }

    // Venkatakrishnan smooth limiter scalar for one cell face, given the
    // unlimited reconstruction increment d = grad.(x_f - x_c), the cell value
    // value, the neighbour max/min (qMax, qMin), and a smoothing length eps2
    // (= (K*h)^3 style term; pass 0 for the sharp Barth-Jespersen limit).
    // phi is in [0,1] for consistent finite bounds. Only the sharp branch
    // guarantees that value + phi*d stays within [qMin,qMax].
    static double venkat(double d, double value, double qMax, double qMin, double eps2) {
        // a = signed headroom toward the bound the face value moves into.
        double a;
        if (d > 1e-300)
            a = qMax - value; // moving up -> ceiling
        else if (d < -1e-300)
            a = qMin - value; // moving down -> floor
        else
            return 1.0; // no increment, no limiting
        return phiVenkat(a, d, eps2);
    }

  private:
    // Venkatakrishnan limiter scalar (Venkatakrishnan 1993, eq 10) in terms of
    // the headroom a = Delta+ and the increment d = Delta-:
    //   phi = (a^2 + 2 a d + eps2) / (a^2 + a d + 2 d^2 + eps2)
    // The eps2 <= 0 branch explicitly selects the different, sharp
    // Barth-Jespersen min(1, a/d) formula.
    static double phiVenkat(double a, double d, double eps2) {
        if (eps2 <= 0.0)
            return std::min(1.0, a / d); // a, d same sign -> a/d >= 0
        double num = a * a + 2.0 * a * d + eps2;
        double den = a * a + a * d + 2.0 * d * d + eps2;
        double phi = num / den;
        return std::min(1.0, std::max(0.0, phi));
    }
};

} // namespace dbns
