#pragma once

#include "DBNSTypes.hpp"
#include <cmath>
#include <algorithm>

// ---------------------------------------------------------------------------
// HLLC approximate Riemann solver (Toro, Spruce and Speares 1994; Toro 2009,
// chapter 10) for the multidimensional Favre-averaged RANS-SST system.
//
// HLLC restores the contact and shear waves omitted by the two-wave HLL model.
// The turbulence scalars (k, omega) are passive across the acoustic waves, so
// the star state carries their side values unchanged across those waves.
//
// Wave speeds combine the side acoustic speeds with Roe averages and Toro's
// contact speed. The shock/contact tests cover their stated finite states;
// near-vacuum behavior and positivity of the complete reconstructed, viscous,
// forced time update are not established by this flux formula alone.
// ---------------------------------------------------------------------------

namespace dbns {

struct HLLCFlux {
    // Compute the HLLC numerical flux through a face with unit normal (nx, ny)
    // given the reconstructed left and right primitive states.
    static StateVec flux(const Primitive &L, const Primitive &R, double nx, double ny,
                         const IdealGasEOS &eos) {
        double gamma = eos.gamma;

        double unL = L.u * nx + L.v * ny;
        double unR = R.u * nx + R.v * ny;
        double aL = GasState::soundSpeed(L, eos);
        double aR = GasState::soundSpeed(R, eos);

        // total energy (incl TKE) and total enthalpy for each side
        double EL = L.p / ((gamma - 1.0) * L.rho) + 0.5 * (L.u * L.u + L.v * L.v) + L.k;
        double ER = R.p / ((gamma - 1.0) * R.rho) + 0.5 * (R.u * R.u + R.v * R.v) + R.k;
        double HL = EL + L.p / L.rho;
        double HR = ER + R.p / R.rho;

        // Roe averages for the Einfeldt wave-speed bounds.
        double srL = std::sqrt(L.rho), srR = std::sqrt(R.rho);
        double inv = 1.0 / (srL + srR);
        double uT = (srL * L.u + srR * R.u) * inv;
        double vT = (srL * L.v + srR * R.v) * inv;
        double HT = (srL * HL + srR * HR) * inv;
        double unT = uT * nx + vT * ny;
        double q2T = uT * uT + vT * vT;
        // k is advected with the contact and is excluded by the pressure EOS.
        // Remove its Roe average from the acoustic enthalpy as well.
        double kT = (srL * L.k + srR * R.k) * inv;
        double aT2 = (gamma - 1.0) * (HT - 0.5 * q2T - kT);
        double aT = std::sqrt(std::max(aT2, 1e-30));

        // Einfeldt SL, SR (bounded by both the face states and the Roe average).
        double SL = std::min(unL - aL, unT - aT);
        double SR = std::max(unR + aR, unT + aT);

        // Contact wave speed S* (Toro 10.37).
        double mL = L.rho * (SL - unL);
        double mR = R.rho * (SR - unR);
        double Sstar = (R.p - L.p + unL * mL - unR * mR) / (mL - mR);

        StateVec FL = GasState::normalFlux(L, nx, ny, eos);
        if (SL >= 0.0)
            return FL;

        StateVec FR = GasState::normalFlux(R, nx, ny, eos);
        if (SR <= 0.0)
            return FR;

        // Star states U*_K (Toro 10.73), then F*_K = F_K + S_K (U*_K - U_K).
        if (Sstar >= 0.0) {
            StateVec UL = GasState::toConserved(L, eos);
            StateVec Us = starState(L, unL, SL, Sstar, EL, nx, ny);
            StateVec F{};
            for (int i = 0; i < NVAR; ++i)
                F[i] = FL[i] + SL * (Us[i] - UL[i]);
            return F;
        } else {
            StateVec UR = GasState::toConserved(R, eos);
            StateVec Us = starState(R, unR, SR, Sstar, ER, nx, ny);
            StateVec F{};
            for (int i = 0; i < NVAR; ++i)
                F[i] = FR[i] + SR * (Us[i] - UR[i]);
            return F;
        }
    }

  private:
    // HLLC star (intermediate) conserved state for one side K.
    static StateVec starState(const Primitive &K, double unK, double SK, double Sstar, double EK,
                              double nx, double ny) {
        double factor = K.rho * (SK - unK) / (SK - Sstar);
        double dvn = Sstar - unK;
        StateVec Us{};
        Us[I_RHO] = factor;
        Us[I_RHOU] = factor * (K.u + dvn * nx);
        Us[I_RHOV] = factor * (K.v + dvn * ny);
        Us[I_RHOE] = factor * (EK + dvn * (Sstar + K.p / (K.rho * (SK - unK))));
        Us[I_RHOK] = factor * K.k; // scalars ride the contact unchanged
        Us[I_RHOW] = factor * K.omega;
        return Us;
    }
};

} // namespace dbns
