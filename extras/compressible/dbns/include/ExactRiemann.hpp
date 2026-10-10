#pragma once

#include <cmath>
#include <algorithm>
#include <stdexcept>

// ---------------------------------------------------------------------------
// Exact Riemann solver for the 1D Euler equations of an ideal (calorically
// perfect) gas.  This is a VERIFICATION reference only: the shock-capturing
// solver is checked against the self-similar solution this produces; it is not
// used inside the solver itself.
//
// Algorithm follows Toro, "Riemann Solvers and Numerical Methods for Fluid
// Dynamics" (3rd ed.), chapter 4: Newton iteration on the star-region pressure
// via the pressure functions f_L, f_R, then self-similar sampling at speed
// S = x/t.
// ---------------------------------------------------------------------------

namespace dbns {

struct RiemannState1D {
    double rho = 1.0;
    double u = 0.0;
    double p = 1.0;
};

struct ExactRiemann {
    double gamma = 1.4;

    explicit ExactRiemann(double g = 1.4) : gamma(g) {}

    double soundSpeed(const RiemannState1D &s) const { return std::sqrt(gamma * s.p / s.rho); }

    // Pressure function f_K(p) for state K (Toro 4.6-4.7).
    double fK(double p, const RiemannState1D &K, double aK) const {
        if (p > K.p) {
            // shock branch
            double A = 2.0 / ((gamma + 1.0) * K.rho);
            double B = (gamma - 1.0) / (gamma + 1.0) * K.p;
            return (p - K.p) * std::sqrt(A / (p + B));
        }
        // rarefaction branch
        double pw = std::pow(p / K.p, (gamma - 1.0) / (2.0 * gamma));
        return 2.0 * aK / (gamma - 1.0) * (pw - 1.0);
    }

    // Derivative f_K'(p) (Toro 4.37).
    double dfK(double p, const RiemannState1D &K, double aK) const {
        if (p > K.p) {
            double A = 2.0 / ((gamma + 1.0) * K.rho);
            double B = (gamma - 1.0) / (gamma + 1.0) * K.p;
            return std::sqrt(A / (B + p)) * (1.0 - 0.5 * (p - K.p) / (B + p));
        }
        double pw = std::pow(p / K.p, -(gamma + 1.0) / (2.0 * gamma));
        return pw / (K.rho * aK);
    }

    // Solve for the star-region pressure p* and velocity u*.
    void starState(const RiemannState1D &L, const RiemannState1D &R, double &pStar,
                   double &uStar) const {
        if (!std::isfinite(gamma) || gamma <= 1)
            throw std::invalid_argument("ExactRiemann requires finite gamma > 1");
        for (const auto &s : {L, R})
            if (!std::isfinite(s.rho) || s.rho <= 0 || !std::isfinite(s.p) || s.p <= 0 ||
                !std::isfinite(s.u))
                throw std::invalid_argument("ExactRiemann requires finite positive gas states");
        double aL = soundSpeed(L), aR = soundSpeed(R);
        double du = R.u - L.u;
        const double velocityScale = aL + aR + std::abs(du);
        if (!std::isfinite(velocityScale) || velocityScale <= 0)
            throw std::runtime_error("ExactRiemann acoustic scale is not representable");
        if (du >= 2 * (aL + aR) / (gamma - 1))
            throw std::domain_error("ExactRiemann vacuum solution is unsupported");

        // Two-rarefaction (PVRS-style) initial guess, floored positive.
        double pPV = 0.5 * (L.p + R.p) - 0.125 * du * (L.rho + R.rho) * (aL + aR);
        double p = std::max(1e-8, pPV);

        // Newton iteration (Toro 4.44); converges in a handful of steps.
        for (int it = 0; it < 100; ++it) {
            double f = fK(p, L, aL) + fK(p, R, aR) + du;
            double df = dfK(p, L, aL) + dfK(p, R, aR);
            double pNew = p - f / df;
            if (pNew < 1e-12)
                pNew = 1e-12; // keep pressure positive
            double change = 2.0 * std::abs(pNew - p) / (pNew + p);
            p = pNew;
            if (change < 1e-12)
                break;
        }
        // A small Newton update can mean stagnation at the pressure floor.
        // Certify the actual scalar equation before exposing an oracle value.
        const double residual = fK(p, L, aL) + fK(p, R, aR) + du;
        if (!std::isfinite(p) || p <= 0 || !std::isfinite(residual) ||
            std::abs(residual) / velocityScale > 1e-12)
            throw std::runtime_error("ExactRiemann star pressure did not satisfy its equation");
        const double velocity = 0.5 * (L.u + R.u) + 0.5 * (fK(p, R, aR) - fK(p, L, aL));
        if (!std::isfinite(velocity))
            throw std::runtime_error("ExactRiemann star velocity is not representable");
        pStar = p;
        uStar = velocity;
    }

    // Sample the self-similar solution at speed S = x/t.
    RiemannState1D sample(const RiemannState1D &L, const RiemannState1D &R, double S) const {
        if (!std::isfinite(S))
            throw std::invalid_argument("ExactRiemann requires a finite sampling speed");
        double pStar, uStar;
        starState(L, R, pStar, uStar);
        double aL = soundSpeed(L), aR = soundSpeed(R);
        double g1 = (gamma - 1.0) / (2.0 * gamma);
        double g2 = (gamma + 1.0) / (2.0 * gamma);

        if (S <= uStar) {
            // sampling point is left of the contact discontinuity
            if (pStar > L.p) {
                // left shock
                double SL = L.u - aL * std::sqrt(g2 * pStar / L.p + g1);
                if (S <= SL)
                    return L;
                double rhoStarL = L.rho * (pStar / L.p + (gamma - 1.0) / (gamma + 1.0)) /
                                  ((gamma - 1.0) / (gamma + 1.0) * pStar / L.p + 1.0);
                return {rhoStarL, uStar, pStar};
            } else {
                // left rarefaction fan
                double SHL = L.u - aL;
                double aStarL = aL * std::pow(pStar / L.p, g1);
                double STL = uStar - aStarL;
                if (S <= SHL)
                    return L;
                if (S >= STL) {
                    double rhoStarL = L.rho * std::pow(pStar / L.p, 1.0 / gamma);
                    return {rhoStarL, uStar, pStar};
                }
                // inside the fan
                double c = 2.0 / (gamma + 1.0) + (gamma - 1.0) / ((gamma + 1.0) * aL) * (L.u - S);
                double rho = L.rho * std::pow(c, 2.0 / (gamma - 1.0));
                double u = 2.0 / (gamma + 1.0) * (aL + (gamma - 1.0) / 2.0 * L.u + S);
                double p = L.p * std::pow(c, 2.0 * gamma / (gamma - 1.0));
                return {rho, u, p};
            }
        } else {
            // sampling point is right of the contact discontinuity
            if (pStar > R.p) {
                // right shock
                double SR = R.u + aR * std::sqrt(g2 * pStar / R.p + g1);
                if (S >= SR)
                    return R;
                double rhoStarR = R.rho * (pStar / R.p + (gamma - 1.0) / (gamma + 1.0)) /
                                  ((gamma - 1.0) / (gamma + 1.0) * pStar / R.p + 1.0);
                return {rhoStarR, uStar, pStar};
            } else {
                // right rarefaction fan
                double SHR = R.u + aR;
                double aStarR = aR * std::pow(pStar / R.p, g1);
                double STR = uStar + aStarR;
                if (S >= SHR)
                    return R;
                if (S <= STR) {
                    double rhoStarR = R.rho * std::pow(pStar / R.p, 1.0 / gamma);
                    return {rhoStarR, uStar, pStar};
                }
                double c = 2.0 / (gamma + 1.0) - (gamma - 1.0) / ((gamma + 1.0) * aR) * (R.u - S);
                double rho = R.rho * std::pow(c, 2.0 / (gamma - 1.0));
                double u = 2.0 / (gamma + 1.0) * (-aR + (gamma - 1.0) / 2.0 * R.u + S);
                double p = R.p * std::pow(c, 2.0 * gamma / (gamma - 1.0));
                return {rho, u, p};
            }
        }
    }
};

} // namespace dbns
