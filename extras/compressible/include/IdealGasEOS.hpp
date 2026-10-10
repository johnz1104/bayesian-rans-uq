#pragma once
#include <cmath>
#include <stdexcept>

// Thermodynamic properties and equation of state for a calorically perfect ideal gas.
// All units are SI (Pa, K, kg/m³, J/kg).
struct IdealGasEOS {
    double gamma = 1.4; // ratio of specific heats (air: 1.4)
    double R = 287.0;   // specific gas constant [J/(kg·K)]  (air: 287)
    double Pr = 0.72;   // laminar Prandtl number
    double Pr_T = 0.9;  // turbulent Prandtl number

    double Cp() const { return gamma * R / (gamma - 1.0); }
    double Cv() const { return R / (gamma - 1.0); }
    double mu_ref = 1.716e-5; // dynamic viscosity at T_ref [Pa·s]
    double T_ref = 273.15;    // reference temperature for Sutherland [K]
    double S_suth = 110.4;    // Sutherland constant [K]

    // Solvers validate their copied model before evaluating thermodynamics.
    // Standalone users can validate a mutable parameter set the same way.
    void validate() const {
        if (!std::isfinite(gamma) || gamma <= 1.0)
            throw std::invalid_argument("IdealGasEOS requires finite gamma > 1");
        if (!std::isfinite(R) || R <= 0.0)
            throw std::invalid_argument("IdealGasEOS requires finite R > 0");
        if (!std::isfinite(Pr) || Pr <= 0.0 || !std::isfinite(Pr_T) || Pr_T <= 0.0)
            throw std::invalid_argument("IdealGasEOS requires finite positive Pr and Pr_T");
        if (!std::isfinite(mu_ref) || mu_ref < 0.0)
            throw std::invalid_argument("IdealGasEOS requires finite mu_ref >= 0");
        if (!std::isfinite(T_ref) || T_ref <= 0.0)
            throw std::invalid_argument("IdealGasEOS requires finite T_ref > 0");
        if (!std::isfinite(S_suth) || S_suth < 0.0)
            throw std::invalid_argument("IdealGasEOS requires finite S_suth >= 0");
    }

    // Sutherland viscosity law: μ(T) = μ_ref * (T/T_ref)^(3/2) * (T_ref+S)/(T+S)
    double viscosity(double T) const {
        double ratio = T / T_ref;
        return mu_ref * ratio * std::sqrt(ratio) * (T_ref + S_suth) / (T + S_suth);
    }

    // Effective thermal conductivity (laminar + turbulent)
    double conductivity(double mu, double muT) const { return mu * Cp() / Pr + muT * Cp() / Pr_T; }

    // Ideal gas EOS: ρ = p / (R T)
    double density(double p, double T) const { return p / (R * T); }
    double temperature(double p, double rho) const { return p / (rho * R); }
    double pressure(double rho, double T) const { return rho * R * T; }

    double soundSpeed(double T) const { return std::sqrt(gamma * R * T); }
    double machNumber(double Umag, double T) const { return Umag / soundSpeed(T); }

    // Total enthalpy: H = Cp*T + 0.5*|U|²
    double totalEnthalpy(double T, double Umag2) const { return Cp() * T + 0.5 * Umag2; }
};
