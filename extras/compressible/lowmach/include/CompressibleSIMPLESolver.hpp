#pragma once
#include "Mesh.hpp"
#include "Field.hpp"
#include "LinearSolver.hpp"
#include "SSTModel.hpp"
#include "FlowFields.hpp" // SolverSettings
#include "CompressibleFlowFields.hpp"
#include "CompressibleBCs.hpp"
#include "IdealGasEOS.hpp"
#include <memory>
#include <vector>

// Pressure-based LOW-MACH compressible SIMPLE solver for steady RANS with an
// ideal gas: a variable-density, temperature-transport approximation, NOT a
// general subsonic compressible scheme. What is actually implemented:
//
//   1. Update density from EOS:  ρ = p/(R T)
//   2. Update dynamic viscosity via Sutherland's law
//   3. Update SST turbulence model (nuT, F1, F2, Pk)
//   4. Solve momentum equations for U* (upwind convection, central diffusion,
//      density-weighted face fluxes)
//   5. Solve the pressure-correction equation with density-weighted
//      coefficients. There is NO Rhie-Chow face-flux treatment on bounded
//      compressible meshes and NO rho' = p'/(RT) convective compressibility
//      term: the Poisson operator is the incompressible form with
//      density-weighted fluxes.
//   6. Correct U, p, ρ
//   7. Solve energy as SENSIBLE-ENTHALPY TRANSPORT ONLY,
//      ∇·(ρ U Cp T) = ∇·(λ_eff ∇T): pressure work, kinetic energy, and
//      viscous/turbulent dissipation are NOT modeled.
//   8. Update T from energy, update ρ from EOS
//   9. Solve turbulence (k, ω) with density-scaled production; because k is
//      part of the two-pressure EOS, refresh rho and the mechanical outlet
//      pressure after a k update before convergence can be declared.
//
// Evidence and applicability: the COMMITTED validation is the Ma 0.1 channel
// regression; Ma ~0.5 is the INTENDED applicability ceiling implied by the
// second-order-in-Mach omissions above, not yet demonstrated by a Mach
// ladder. Higher Mach, shocks, and genuine compressible energy coupling
// belong to the density-based DBNS solver, not to extensions of this one.
class CompressibleSIMPLESolver {
  public:
    CompressibleSIMPLESolver(const Mesh &mesh, const SSTModel &sst,
                             const CompressibleBoundaryConditions &bcs, const IdealGasEOS &eos,
                             const SolverSettings &settings = {});

    CompressibleConvergenceHistory solve(CompressibleFlowFields &fields);

    // Direct per-cell state validation: every component of every solved field
    // (U including the spanwise component, p, T, rho, k, omega) is checked
    // with std::isfinite, plus positivity of T, rho, mechanical p and the
    // recovered thermodynamic p. Aggregate max/min
    // reductions can never prove this: std::max(a, NaN) evaluates the
    // comparison as false and KEEPS a, so a NaN entering a chained reduction
    // is silently dropped. Public so a unit test can check that divergence
    // detection is not masked this way.
    bool stateIsValid(const CompressibleFlowFields &f) const;

    void initUniform(CompressibleFlowFields &f, const Vec3 &Uinit, double p_init, double T_init,
                     double kInit, double omegaInit);

  private:
    const Mesh &mesh_;
    Mesh::Stamp meshStamp_ = mesh_.stamp();
    SSTModel sst_;
    CompressibleBoundaryConditions bcs_;
    IdealGasEOS eos_;
    SolverSettings settings_;

    std::unique_ptr<ILinearSolver> pSolver_;
    std::unique_ptr<ILinearSolver> mSolver_;
    std::unique_ptr<ILinearSolver> tSolver_;
    std::unique_ptr<ILinearSolver> eSolver_;

    // Momentum diagonal coefficients for Rhie-Chow
    std::vector<double> aP_;
    std::vector<double> aPunrelaxed_;
    // Cell-centred dynamic viscosity (from Sutherland at current T)
    std::vector<double> mu_;

    void updateViscosity(const CompressibleFlowFields &f);
    void updateDensity(CompressibleFlowFields &f);
    // The prescribed outlet pressure is a THERMODYNAMIC static value; the
    // field carries the mechanical working pressure, so every BC application
    // is followed by this conversion of the outlet boundary faces,
    // p_mech,b = p_out + (2/3) rho_o k_o with the owner-cell state standing
    // in for the boundary turbulence energy.
    void mechanicalizeOutletPressure(CompressibleFlowFields &f) const;

    void assembleMomentum(LinearSystem &sys, const CompressibleFlowFields &f, int component,
                          std::vector<double> &aP);
    void assemblePressureCorrection(LinearSystem &sys, const CompressibleFlowFields &f,
                                    const std::vector<double> &aP, ScalarField &pPrime);
    void assembleEnergy(LinearSystem &sys, const CompressibleFlowFields &f);
    void assembleKEquation(LinearSystem &sys, const CompressibleFlowFields &f);
    void assembleOmegaEquation(LinearSystem &sys, const CompressibleFlowFields &f,
                               const ScalarField &Smag);

    void correctVelocity(CompressibleFlowFields &f, const ScalarField &pPrime,
                         const std::vector<double> &aP);
    void correctPressure(CompressibleFlowFields &f, const ScalarField &pPrime);

    double computeResidual(const CompressibleFlowFields &f, int component);
    double computeScalarResidual(const ScalarField &phi_old, const ScalarField &phi_new);
};
