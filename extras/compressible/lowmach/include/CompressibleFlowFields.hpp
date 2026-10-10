#pragma once
#include "Mesh.hpp"
#include "Field.hpp"
#include "IdealGasEOS.hpp"
#include "FlowFields.hpp" // reuse SolverSettings, ConvergenceHistory

// Initialization is shared by the direct solver and its forward-model boundary.
inline void validateCompressibleInitialState(const Vec3 &velocity, double pressure,
                                             double temperature, double k, double omega,
                                             const IdealGasEOS &eos) {
    if (!std::isfinite(velocity.x) || !std::isfinite(velocity.y) || !std::isfinite(velocity.z) ||
        !std::isfinite(pressure) || !std::isfinite(temperature) || !std::isfinite(k) ||
        !std::isfinite(omega) || pressure <= 0 || temperature <= 0 || k < 0 || omega < 0)
        throw std::invalid_argument("compressible initial velocity must be finite; pressure and "
                                    "temperature positive; k and omega nonnegative");
    const double density = eos.density(pressure, temperature);
    const double viscosity = eos.viscosity(temperature);
    const double mechanicalPressure = pressure + (2.0 / 3.0) * density * k;
    if (!std::isfinite(density) || density <= 0 || !std::isfinite(viscosity) || viscosity < 0 ||
        !std::isfinite(mechanicalPressure))
        throw std::invalid_argument(
            "compressible initial state exceeds the finite thermodynamic range");
}

// Compressible RANS flow fields: extends incompressible set with density and temperature.
struct CompressibleFlowFields {
    VectorField U;   // velocity [m/s]
    ScalarField p;   // MECHANICAL working pressure [Pa]: p_mech = p_thermo
                     // + (2/3) rho k (the Favre trace absorbed into the
                     // momentum working variable). Thermodynamic statics
                     // are recovered as p_thermo = p - (2/3) rho k; the
                     // observation shim and the python field export do
                     // this, and boundary/initialization inputs specified
                     // as thermodynamic values are converted on entry.
    ScalarField T;   // static temperature [K]
    ScalarField rho; // density [kg/m³]  (derived: ρ = p_mech/(R T + 2k/3))

    // SST turbulence model fields (same as incompressible)
    ScalarField k;      // turbulent kinetic energy [m²/s²]
    ScalarField omega;  // specific dissipation rate [1/s]
    ScalarField nuT;    // eddy viscosity [m²/s]
    ScalarField F1, F2; // SST blending functions
    ScalarField Pk;     // turbulence production
    ScalarField CDkw;   // cross-diffusion

    // Turbulence-establishment marker (see FlowFields::turbEstablished): the
    // startup-only nuT floor applies until this flips, and warm restarts carry
    // it so the floor never re-engages on an established state.
    bool turbEstablished = false;

    void requireLayout(const Mesh &mesh) const {
        requireFieldStorage(mesh, U);
        for (const ScalarField *field : {&p, &T, &rho, &k, &omega, &nuT, &F1, &F2, &Pk, &CDkw})
            requireFieldStorage(mesh, *field);
    }

    CompressibleFlowFields() = default;
    explicit CompressibleFlowFields(const Mesh &mesh)
        : U(mesh, "U"), p(mesh, "p"), T(mesh, "T"), rho(mesh, "rho"), k(mesh, "k"),
          omega(mesh, "omega"), nuT(mesh, "nuT"), F1(mesh, "F1"), F2(mesh, "F2"), Pk(mesh, "Pk"),
          CDkw(mesh, "CDkw") {}
};

// Per-iteration residuals for the compressible solver.
struct CompressibleResidualEntry {
    int iteration = 0;
    double Ux = 0, Uy = 0, Uz = 0;
    double p = 0;
    double T = 0;
    double k = 0, omega = 0;
};

struct CompressibleConvergenceHistory {
    std::vector<CompressibleResidualEntry> entries;
    bool converged = false;
    bool diverged = false;
    int finalIter = 0;
};
