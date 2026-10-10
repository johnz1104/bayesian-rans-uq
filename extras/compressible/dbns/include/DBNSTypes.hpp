#pragma once

#include "IdealGasEOS.hpp"
#include <array>
#include <cmath>

// ---------------------------------------------------------------------------
// Density-based Navier-Stokes (DBNS) state types for the shock-capturing
// compressible RANS solver.
//
// This is the additive, density-based path: a finite-volume, conservative,
// shock-capturing solver carrying the Favre-averaged SST k-omega closure.  It
// is intentionally separate from the existing low-Mach pressure-based
// CompressibleSIMPLESolver (which stays untouched).
//
// The solver is 2D (the mesh generators are 2D); the z-momentum is zero.  The
// conserved state carries six variables:
//   W = [ rho, rho*u, rho*v, rho*E, rho*k, rho*omega ]
// with the Favre-averaged total energy INCLUDING the turbulent kinetic energy
//   rho*E = p/(gamma-1) + 0.5*rho*(u^2+v^2) + rho*k          (Wilcox 2006, eq 5.x)
// so that for laminar runs (k = 0) this reduces to the standard Euler/NS total
// energy.  Laminar runs carry no turbulence: the solver rejects any laminar
// input with nonzero k or omega.
// ---------------------------------------------------------------------------

namespace dbns {

// Number of conserved variables (2D Favre-averaged RANS-SST).
constexpr int NVAR = 6;

// Conserved-variable index map (documented for readers of the flux kernels).
enum ConsIndex { I_RHO = 0, I_RHOU = 1, I_RHOV = 2, I_RHOE = 3, I_RHOK = 4, I_RHOW = 5 };

using StateVec = std::array<double, NVAR>;

// Primitive variables: rho, u, v, p (and the turbulence scalars k, omega).
// T is derived from p and rho via the ideal-gas EOS, so it is not stored here.
struct Primitive {
    double rho = 1.0;
    double u = 0.0;
    double v = 0.0;
    double p = 101325.0;
    double k = 0.0;
    double omega = 0.0;
};

// Thermodynamic and conserved/primitive helpers.  All static (no state of its
// own); the gas model is passed in so the same kernels serve any ideal gas.
struct GasState {
    // Conserved -> primitive.  Pressure follows from the internal energy with
    // the TKE removed:  e_internal = E - 0.5|u|^2 - k,  p = (gamma-1)*rho*e_int.
    static Primitive toPrimitive(const StateVec &W, const IdealGasEOS &eos) {
        Primitive V;
        double rho = W[I_RHO];
        V.rho = rho;
        double invRho = 1.0 / rho;
        V.u = W[I_RHOU] * invRho;
        V.v = W[I_RHOV] * invRho;
        V.k = W[I_RHOK] * invRho;
        V.omega = W[I_RHOW] * invRho;
        double E = W[I_RHOE] * invRho;
        double eInternal = E - 0.5 * (V.u * V.u + V.v * V.v) - V.k;
        V.p = (eos.gamma - 1.0) * rho * eInternal;
        return V;
    }

    // Primitive -> conserved.
    static StateVec toConserved(const Primitive &V, const IdealGasEOS &eos) {
        StateVec W{};
        double eInternal = V.p / ((eos.gamma - 1.0) * V.rho);
        double E = eInternal + 0.5 * (V.u * V.u + V.v * V.v) + V.k;
        W[I_RHO] = V.rho;
        W[I_RHOU] = V.rho * V.u;
        W[I_RHOV] = V.rho * V.v;
        W[I_RHOE] = V.rho * E;
        W[I_RHOK] = V.rho * V.k;
        W[I_RHOW] = V.rho * V.omega;
        return W;
    }

    // Frozen sound speed a = sqrt(gamma p / rho).
    static double soundSpeed(const Primitive &V, const IdealGasEOS &eos) {
        return std::sqrt(eos.gamma * V.p / V.rho);
    }

    static double temperature(const Primitive &V, const IdealGasEOS &eos) {
        return V.p / (V.rho * eos.R);
    }

    // Inviscid (Euler) flux projected onto a face normal n = (nx, ny).
    // F.n = [ rho*un, rho*u*un + p*nx, rho*v*un + p*ny,
    //         (rho*E + p)*un, rho*k*un, rho*omega*un ]
    // where un = u*nx + v*ny is the contravariant (normal) velocity.
    static StateVec normalFlux(const Primitive &V, double nx, double ny, const IdealGasEOS &eos) {
        double un = V.u * nx + V.v * ny;
        double eInternal = V.p / ((eos.gamma - 1.0) * V.rho);
        double E = eInternal + 0.5 * (V.u * V.u + V.v * V.v) + V.k;
        double rhoun = V.rho * un;
        StateVec F{};
        F[I_RHO] = rhoun;
        F[I_RHOU] = rhoun * V.u + V.p * nx;
        F[I_RHOV] = rhoun * V.v + V.p * ny;
        F[I_RHOE] = (V.rho * E + V.p) * un;
        F[I_RHOK] = rhoun * V.k;
        F[I_RHOW] = rhoun * V.omega;
        return F;
    }

    // Physical admissibility of a conserved state (positive density, finite,
    // positive pressure).  Used by the solver to classify divergence rather
    // than to throw, mirroring the EvaluationStatus discipline of the repo.
    static bool admissible(const StateVec &W, const IdealGasEOS &eos) {
        if (!(W[I_RHO] > 0.0))
            return false;
        for (int i = 0; i < NVAR; ++i)
            if (!std::isfinite(W[i]))
                return false;
        Primitive V = toPrimitive(W, eos);
        return V.p > 0.0;
    }
};

} // namespace dbns
