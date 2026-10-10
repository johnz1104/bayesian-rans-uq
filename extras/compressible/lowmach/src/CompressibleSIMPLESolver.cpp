#include "CompressibleSIMPLESolver.hpp"
#include "LinearSolver.hpp"
#include "BoundaryCondition.hpp"
#include "StressOperators.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>

// ─── Constructor ──────────────────────────────────────────────────────────────

CompressibleSIMPLESolver::CompressibleSIMPLESolver(const Mesh &mesh, const SSTModel &sst,
                                                   const CompressibleBoundaryConditions &bcs,
                                                   const IdealGasEOS &eos,
                                                   const SolverSettings &settings)
    : mesh_(mesh), sst_(sst), bcs_(bcs), eos_(eos), settings_(settings), aP_(mesh.nCells(), 0.0),
      aPunrelaxed_(mesh.nCells(), 0.0), mu_(mesh.nCells(), eos.mu_ref) {
    settings_.validate();
    eos_.validate();
    sst_.validate();
    bcs_.validate(mesh_);
    pSolver_ = makeSolver(settings_.pressureSolver);
    mSolver_ = makeSolver(settings_.momentumSolver);
    tSolver_ = makeSolver(settings_.turbulenceSolver);
    eSolver_ = makeSolver("BiCGSTAB"); // energy equation
}

// ─── Initialisation ───────────────────────────────────────────────────────────

void CompressibleSIMPLESolver::initUniform(CompressibleFlowFields &f, const Vec3 &Uinit,
                                           double p_init, double T_init, double kInit,
                                           double omegaInit) {
    mesh_.requireStamp(meshStamp_);
    f.requireLayout(mesh_);
    validateCompressibleInitialState(Uinit, p_init, T_init, kInit, omegaInit, eos_);
    const auto &wd = mesh_.wallDistance(); // preflight before changing caller fields
    f.U.setUniform(Uinit);
    // p_init is the THERMODYNAMIC initialization pressure; the field carries
    // the MECHANICAL working pressure p_mech = p_thermo + (2/3) rho k, so the
    // stored value converts with the exact closed form (rho0 solves the plain
    // EOS at (p_init, T_init), and with p_mech0 below, updateDensity's
    // two-pressure inversion returns exactly rho0 at iteration zero)
    f.T.setUniform(T_init);
    f.k.setUniform(kInit);
    f.omega.setUniform(omegaInit);
    f.F1.setUniform(1.0);
    f.F2.setUniform(1.0);
    f.Pk.setUniform(0.0);
    f.CDkw.setUniform(0.0);
    f.turbEstablished = false; // cold start: the startup nuT floor window applies

    // Density from EOS at the thermodynamic state, then the mechanical field
    const double rho0 = eos_.density(p_init, T_init);
    f.rho.setUniform(rho0);
    f.p.setUniform(p_init + (2.0 / 3.0) * rho0 * std::max(kInit, 0.0));

    // Sutherland viscosity at T_init
    const double mu0 = eos_.viscosity(T_init);
    for (int ci = 0; ci < mesh_.nCells(); ++ci)
        mu_[ci] = mu0;

    // Wall-distance omega profile (same logic as incompressible)
    for (int ci = 0; ci < mesh_.nCells(); ++ci) {
        double y = std::max(wd[ci], 1e-20);
        double nu0 = mu0 / rho0;
        double omWall = 60.0 * nu0 / (sst_.coeffs.beta1 * y * y);
        f.omega[ci] = std::max(omWall, omegaInit);
    }

    // Initial nuT
    for (int ci = 0; ci < mesh_.nCells(); ++ci)
        f.nuT[ci] = sst_.coeffs.a1 * kInit / std::max(sst_.coeffs.a1 * f.omega[ci], 1e-20);

    // Apply BCs
    double nu0 = mu0 / rho0;
    applyAllCompressibleBCs(f.U, f.p, f.T, f.k, f.omega, mesh_, bcs_, nu0);
    mechanicalizeOutletPressure(f);
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

void CompressibleSIMPLESolver::updateViscosity(const CompressibleFlowFields &f) {
    for (int ci = 0; ci < mesh_.nCells(); ++ci)
        mu_[ci] = eos_.viscosity(f.T[ci]);
}

void CompressibleSIMPLESolver::mechanicalizeOutletPressure(CompressibleFlowFields &f) const {
    for (int pi = 0; pi < mesh_.nPatches(); ++pi) {
        const Patch &pat = mesh_.patch(pi);
        if (pat.type != "outlet")
            continue;
        const PatchBC &bc = bcs_.pressureBC[pi];
        if (bc.type != BCType::OutletPressure && bc.type != BCType::Dirichlet)
            continue;
        for (FaceID fi : pat.faces) {
            int o = mesh_.face(fi).owner;
            f.p.bface(fi) = bc.value + (2.0 / 3.0) * f.rho[o] * std::max(f.k[o], 0.0);
        }
    }
}

bool CompressibleSIMPLESolver::stateIsValid(const CompressibleFlowFields &f) const {
    mesh_.requireStamp(meshStamp_);
    f.requireLayout(mesh_);
    // explicit per-cell sweep; see the header note on why max/min reductions
    // cannot substitute (they drop NaN operands)
    for (int ci = 0; ci < mesh_.nCells(); ++ci) {
        const double pThermo = f.p[ci] - (2.0 / 3.0) * f.rho[ci] * std::max(f.k[ci], 0.0);
        const bool finite =
            std::isfinite(f.U[ci].x) && std::isfinite(f.U[ci].y) && std::isfinite(f.U[ci].z) &&
            std::isfinite(f.p[ci]) && std::isfinite(f.T[ci]) && std::isfinite(f.rho[ci]) &&
            std::isfinite(f.k[ci]) && std::isfinite(f.omega[ci]) && std::isfinite(pThermo);
        // The stored pressure is mechanical. Check BOTH it and the recovered
        // thermodynamic pressure: positivity of the larger mechanical value
        // alone cannot validate an arbitrarily corrupted rho/k state.
        if (!finite || f.T[ci] <= 0.0 || f.rho[ci] <= 0.0 || f.p[ci] <= 0.0 || pThermo <= 0.0)
            return false;
    }
    return true;
}

void CompressibleSIMPLESolver::updateDensity(CompressibleFlowFields &f) {
    // f.p is the MECHANICAL pressure (thermodynamic + (2/3) rho k, the
    // absorbed turbulent normal stress); the thermodynamic density follows
    // from rho R T = p_mech - (2/3) rho k exactly:
    //   rho = p_mech / (R T + (2/3) k)
    for (int ci = 0; ci < mesh_.nCells(); ++ci)
        f.rho[ci] =
            f.p[ci] / (eos_.R * std::max(f.T[ci], 1.0) + (2.0 / 3.0) * std::max(f.k[ci], 0.0));
    // The prescribed outlet is thermodynamic while the stored face value is
    // mechanical. Density changes after pressure, energy, or k updates, so
    // keep that crossing point synchronized with every EOS refresh.
    mechanicalizeOutletPressure(f);
}

// ─── Momentum equation ────────────────────────────────────────────────────────
// ∇·(ρ U U) = -∇p + ∇·(μ_eff ∇U)
// Convection uses density-weighted face flux: ṁ = ρ_f (U_f·S_f)

void CompressibleSIMPLESolver::assembleMomentum(LinearSystem &sys, const CompressibleFlowFields &f,
                                                int component, std::vector<double> &aP) {
    sys.zero();
    int nIF = mesh_.nInternalFaces();

    for (int fi = 0; fi < nIF; ++fi) {
        const Face &face = mesh_.face(fi);
        int o = face.owner, n = face.neighbor;
        double Sf = face.area;
        double delta = std::max(face.delta, 1e-20);

        // Density and dynamic viscosity at face (linear interpolation)
        double rho_f = face.weight * f.rho[o] + (1.0 - face.weight) * f.rho[n];
        double mu_f = face.weight * mu_[o] + (1.0 - face.weight) * mu_[n];
        double nuT_f = face.weight * f.nuT[o] + (1.0 - face.weight) * f.nuT[n];
        double muEff_f = mu_f + rho_f * nuT_f; // total effective dynamic viscosity

        // Diffusion coefficient (units: Pa·s · m / m = Pa = N/m²)
        double Df = muEff_f * Sf / delta;

        // Density-weighted convective mass flux ṁ = ρ_f (U_f·S_f)
        Vec3 Uf = f.U[o] * face.weight + f.U[n] * (1.0 - face.weight);
        double mFlux =
            rho_f * (Uf.x * face.normal.x + Uf.y * face.normal.y + Uf.z * face.normal.z) * Sf;

        double cPos = std::max(mFlux, 0.0);
        double cNeg = std::max(-mFlux, 0.0);

        sys.diag[o] += Df + cPos;
        sys.diag[n] += Df + cNeg;
        sys.upper[fi] = -(Df + cNeg);
        sys.lower[fi] = -(Df + cPos);
    }

    // Boundary faces. At RESOLVED walls the turbulent viscosity vanishes on
    // the face, so the implicit wall diffusion is MOLECULAR mu(T) only (the
    // same convention as the transpose correction and the wall-stress
    // observation); owner-cell extrapolation of nuT overestimates the
    // discrete wall shear. Under wall functions the owner value stays.
    std::vector<char> isWallM(mesh_.nFaces(), 0);
    if (!settings_.useWallFunctions) {
        for (int pi = 0; pi < mesh_.nPatches(); ++pi) {
            const Patch &pat = mesh_.patch(pi);
            if (pat.type != "wall")
                continue;
            for (FaceID wfi : pat.faces)
                isWallM[wfi] = 1;
        }
    }
    for (int fi = nIF; fi < mesh_.nFaces(); ++fi) {
        const Face &face = mesh_.face(fi);
        int o = face.owner;
        double Sf = face.area, delta = std::max(face.delta, 1e-20);
        double mu_b = mu_[o];
        double nuT_b = f.nuT[o];
        double rho_b = f.rho[o];
        double muEff = isWallM[fi] ? mu_b : mu_b + rho_b * nuT_b;
        double Db = muEff * Sf / delta;

        double Ub_comp;
        if (component == 0)
            Ub_comp = f.U.bface(fi).x;
        else if (component == 1)
            Ub_comp = f.U.bface(fi).y;
        else
            Ub_comp = f.U.bface(fi).z;

        Vec3 Ubv = f.U.bface(fi);
        double mFlux =
            rho_b * (Ubv.x * face.normal.x + Ubv.y * face.normal.y + Ubv.z * face.normal.z) * Sf;

        if (mFlux >= 0) {
            sys.diag[o] += mFlux + Db;
            sys.source[o] += Db * Ub_comp;
        } else {
            sys.diag[o] += Db;
            sys.source[o] += (Db - mFlux) * Ub_comp;
        }
    }

    // Pressure gradient source: -∂p/∂x_k · V
    VectorField gradP = greenGaussGrad(f.p);
    for (int ci = 0; ci < mesh_.nCells(); ++ci) {
        double vol = mesh_.cell(ci).volume;
        double dpdx = (component == 0) ? gradP[ci].x : (component == 1) ? gradP[ci].y : gradP[ci].z;
        sys.source[ci] -= dpdx * vol;
    }

    // Favre-stress completion (explicit deferred correction): the implicit
    // diffusion above is the componentwise Laplacian div(muEff grad U_i); the
    // full modeled stress (TMR Favre-averaged form) additionally carries
    //   div(muEff (grad U)^T) - (2/3) d/dx_i(muEff div U) - (2/3) d/dx_i(rho k)
    // with muEff = mu + rho nuT covering the viscous and turbulent parts.
    // TWO-PRESSURE CONVENTION for the isotropic -(2/3) rho k part: the near-
    // wall balance p + (2/3) rho k ~ const makes that term and the pressure
    // gradient a stiff cancelling pair, and applying it as an explicit source
    // diverges the startup even blended and ramped, so the MOMENTUM equation
    // solves in the MECHANICAL pressure p_mech = p_thermo + (2/3) rho k (the
    // pair absorbed, stable), while the EOS and the observation path use the
    // THERMODYNAMIC pressure: updateDensity evaluates the exact closed form
    // rho = p_mech / (R T + (2/3) k), which is identically
    // rho = p_thermo / (R T), and the forward-model observation shim maps
    // p_thermo = p_mech - (2/3) rho k. Density is therefore never mispriced
    // by the turbulence bookkeeping term. At wall faces the
    // transpose coefficient keeps only the MOLECULAR viscosity (the
    // turbulent part vanishes at a no-slip wall; owner extrapolation of
    // muEff would overweight the wall flux).
    {
        ScalarField muEff(mesh_, "muEff");
        for (int ci = 0; ci < mesh_.nCells(); ++ci)
            muEff[ci] = mu_[ci] + f.rho[ci] * f.nuT[ci];
        std::vector<double> tsrc = transposeStressSource(mesh_, muEff, f.U, component, &mu_);

        VelocityGradients vg = computeVelocityGradients(f.U);
        ScalarField muDivU(mesh_, "muDivU");
        for (int ci = 0; ci < mesh_.nCells(); ++ci) {
            double divU = vg.dudx[ci].x + vg.dvdx[ci].y + vg.dwdx[ci].z;
            muDivU[ci] = muEff[ci] * divU;
        }
        VectorField gTr = greenGaussGrad(muDivU);
        for (int ci = 0; ci < mesh_.nCells(); ++ci) {
            double vol = mesh_.cell(ci).volume;
            double gi = (component == 0) ? gTr[ci].x : (component == 1) ? gTr[ci].y : gTr[ci].z;
            sys.source[ci] += tsrc[ci] - (2.0 / 3.0) * gi * vol;
        }
    }

    // Under-relaxation
    const double alphaU = settings_.alphaU;
    for (int ci = 0; ci < mesh_.nCells(); ++ci) {
        double phi_old = (component == 0) ? f.U[ci].x : (component == 1) ? f.U[ci].y : f.U[ci].z;
        aPunrelaxed_[ci] = sys.diag[ci];
        sys.source[ci] += (1.0 - alphaU) / alphaU * sys.diag[ci] * phi_old;
        sys.diag[ci] /= alphaU;
        aP[ci] = sys.diag[ci];
    }
}

// ─── Pressure-correction equation ─────────────────────────────────────────────
// Continuity: ∑_f (ρ_f U_f · S_f) = 0
// Correction: ρ = p/(RT) → ρ' ≈ p'/(RT) at constant T
// Velocity:   U' = -(V/aP) ∇p'
// → Poisson:  ∑_f (ρ_f V_f / aP_f) / delta * (p'_P - p'_F) = -∑_f ṁ*_f

void CompressibleSIMPLESolver::assemblePressureCorrection(LinearSystem &sys,
                                                          const CompressibleFlowFields &f,
                                                          const std::vector<double> &aP,
                                                          ScalarField &pPrime) {
    sys.zero();
    pPrime.setUniform(0.0);
    int nIF = mesh_.nInternalFaces();

    for (int fi = 0; fi < nIF; ++fi) {
        const Face &face = mesh_.face(fi);
        int o = face.owner, n = face.neighbor;
        double Sf = face.area;
        double delta = std::max(face.delta, 1e-20);

        double Vo = mesh_.cell(o).volume;
        double Vn = mesh_.cell(n).volume;
        double rho_f = face.weight * f.rho[o] + (1.0 - face.weight) * f.rho[n];

        double dP_o = (std::abs(aP[o]) > 1e-30) ? f.rho[o] * Vo / aP[o] : 0.0;
        double dP_n = (std::abs(aP[n]) > 1e-30) ? f.rho[n] * Vn / aP[n] : 0.0;
        double dP_f = face.weight * dP_o + (1.0 - face.weight) * dP_n;

        double coeff = dP_f * Sf / delta;

        sys.diag[o] += coeff;
        sys.diag[n] += coeff;
        sys.upper[fi] = -coeff;
        sys.lower[fi] = -coeff;

        // Density-weighted mass flux source -∑ ρ_f U*·S_f
        Vec3 Uf = f.U[o] * face.weight + f.U[n] * (1.0 - face.weight);
        double mFlux =
            rho_f * (Uf.x * face.normal.x + Uf.y * face.normal.y + Uf.z * face.normal.z) * Sf;
        sys.source[o] -= mFlux;
        sys.source[n] += mFlux;
    }

    for (int pi = 0; pi < mesh_.nPatches(); ++pi) {
        const Patch &pat = mesh_.patch(pi);
        bool isOutlet = (pat.type == "outlet");
        for (FaceID fi : pat.faces) {
            const Face &face = mesh_.face(fi);
            int o = face.owner;
            double Sf = face.area;
            double delta = std::max(face.delta, 1e-20);

            Vec3 Ub = f.U.bface(fi);
            double mFlux = f.rho[o] *
                           (Ub.x * face.normal.x + Ub.y * face.normal.y + Ub.z * face.normal.z) *
                           Sf;
            sys.source[o] -= mFlux;

            if (isOutlet) {
                double Vo = mesh_.cell(o).volume;
                double dP_o = (std::abs(aP[o]) > 1e-30) ? f.rho[o] * Vo / aP[o] : 0.0;
                double coeff = dP_o * Sf / delta;
                sys.diag[o] += coeff;
            }
        }
    }
}

// ─── Energy equation ──────────────────────────────────────────────────────────
// Steady: ∇·(ρ U Cp T) = ∇·(λ_eff ∇T)
// λ_eff = μ Cp/Pr + μ_T Cp/Pr_T

void CompressibleSIMPLESolver::assembleEnergy(LinearSystem &sys, const CompressibleFlowFields &f) {
    sys.zero();
    int nIF = mesh_.nInternalFaces();

    const double Cp = eos_.Cp();

    for (int fi = 0; fi < nIF; ++fi) {
        const Face &face = mesh_.face(fi);
        int o = face.owner, n = face.neighbor;
        double Sf = face.area;
        double delta = std::max(face.delta, 1e-20);

        double rho_f = face.weight * f.rho[o] + (1.0 - face.weight) * f.rho[n];
        double mu_f = face.weight * mu_[o] + (1.0 - face.weight) * mu_[n];
        double nuT_f = face.weight * f.nuT[o] + (1.0 - face.weight) * f.nuT[n];
        double muT_f = rho_f * nuT_f;

        double lambda_f = eos_.conductivity(mu_f, muT_f);

        // Diffusion coefficient: λ_eff * Sf / delta
        double Df = lambda_f * Sf / delta;

        // Convection mass flux (same density-weighted flux as momentum)
        Vec3 Uf = f.U[o] * face.weight + f.U[n] * (1.0 - face.weight);
        double mFlux =
            rho_f * Cp * (Uf.x * face.normal.x + Uf.y * face.normal.y + Uf.z * face.normal.z) * Sf;

        double cPos = std::max(mFlux, 0.0);
        double cNeg = std::max(-mFlux, 0.0);

        sys.diag[o] += Df + cPos;
        sys.diag[n] += Df + cNeg;
        sys.upper[fi] = -(Df + cNeg);
        sys.lower[fi] = -(Df + cPos);
    }

    for (int fi = nIF; fi < mesh_.nFaces(); ++fi) {
        const Face &face = mesh_.face(fi);
        int o = face.owner;
        double Sf = face.area, delta = std::max(face.delta, 1e-20);

        double mu_o = mu_[o];
        double muT_o = f.rho[o] * f.nuT[o];
        double lam = eos_.conductivity(mu_o, muT_o);
        double Db = lam * Sf / delta;

        double Tb = f.T.bface(fi);
        Vec3 Ubv = f.U.bface(fi);
        double mFlux = f.rho[o] * Cp *
                       (Ubv.x * face.normal.x + Ubv.y * face.normal.y + Ubv.z * face.normal.z) * Sf;

        if (mFlux >= 0) {
            sys.diag[o] += mFlux + Db;
            sys.source[o] += Db * Tb;
        } else {
            sys.diag[o] += Db;
            sys.source[o] += (Db - mFlux) * Tb;
        }
    }

    // Under-relaxation for temperature (dedicated alphaT, default 0.7)
    const double alphaT = settings_.alphaT;
    for (int ci = 0; ci < mesh_.nCells(); ++ci) {
        sys.source[ci] += (1.0 - alphaT) / alphaT * sys.diag[ci] * f.T[ci];
        sys.diag[ci] /= alphaT;
    }
}

// ─── Turbulence: k equation ───────────────────────────────────────────────────
// ∇·(ρ U k) = ∇·((μ + ρ σ_k νT) ∇k) + ρ Pk − ρ β* ω k

void CompressibleSIMPLESolver::assembleKEquation(LinearSystem &sys,
                                                 const CompressibleFlowFields &f) {
    sys.zero();
    int nIF = mesh_.nInternalFaces();

    for (int fi = 0; fi < nIF; ++fi) {
        const Face &face = mesh_.face(fi);
        int o = face.owner, n = face.neighbor;
        double Sf = face.area;
        double delta = std::max(face.delta, 1e-20);

        double F1_f = face.weight * f.F1[o] + (1.0 - face.weight) * f.F1[n];
        double sk = sst_.coeffs.sigma_k(F1_f);
        double rho_f = face.weight * f.rho[o] + (1.0 - face.weight) * f.rho[n];
        double mu_f = face.weight * mu_[o] + (1.0 - face.weight) * mu_[n];
        double nuT_f = face.weight * f.nuT[o] + (1.0 - face.weight) * f.nuT[n];
        double muEff = mu_f + rho_f * sk * nuT_f;
        double Df = muEff * Sf / delta;

        Vec3 Uf = f.U[o] * face.weight + f.U[n] * (1.0 - face.weight);
        double mFlux =
            rho_f * (Uf.x * face.normal.x + Uf.y * face.normal.y + Uf.z * face.normal.z) * Sf;
        double cPos = std::max(mFlux, 0.0);
        double cNeg = std::max(-mFlux, 0.0);

        sys.diag[o] += Df + cPos;
        sys.diag[n] += Df + cNeg;
        sys.upper[fi] = -(Df + cNeg);
        sys.lower[fi] = -(Df + cPos);
    }

    for (int fi = nIF; fi < mesh_.nFaces(); ++fi) {
        const Face &face = mesh_.face(fi);
        int o = face.owner;
        double Sf = face.area, delta = std::max(face.delta, 1e-20);
        double sk = sst_.coeffs.sigma_k(f.F1[o]);
        double muEff = mu_[o] + f.rho[o] * sk * f.nuT[o];
        double Db = muEff * Sf / delta;

        double kb = f.k.bface(fi);
        Vec3 Ubv = f.U.bface(fi);
        double mFlux =
            f.rho[o] * (Ubv.x * face.normal.x + Ubv.y * face.normal.y + Ubv.z * face.normal.z) * Sf;
        if (mFlux >= 0) {
            sys.diag[o] += mFlux + Db;
            sys.source[o] += Db * kb;
        } else {
            sys.diag[o] += Db;
            sys.source[o] += (Db - mFlux) * kb;
        }
    }

    for (int ci = 0; ci < mesh_.nCells(); ++ci) {
        double vol = mesh_.cell(ci).volume;
        double rhoC = f.rho[ci];
        sys.source[ci] += rhoC * f.Pk[ci] * vol;
        double dest = rhoC * sst_.coeffs.betaStar * std::max(f.omega[ci], 1e-20);
        sys.diag[ci] += dest * vol;
    }

    const double alphaK = settings_.alphaK;
    for (int ci = 0; ci < mesh_.nCells(); ++ci) {
        sys.source[ci] += (1.0 - alphaK) / alphaK * sys.diag[ci] * f.k[ci];
        sys.diag[ci] /= alphaK;
    }
}

// ─── Turbulence: omega equation ───────────────────────────────────────────────

void CompressibleSIMPLESolver::assembleOmegaEquation(LinearSystem &sys,
                                                     const CompressibleFlowFields &f,
                                                     const ScalarField &Smag) {
    sys.zero();
    int nIF = mesh_.nInternalFaces();

    for (int fi = 0; fi < nIF; ++fi) {
        const Face &face = mesh_.face(fi);
        int o = face.owner, n = face.neighbor;
        double Sf = face.area;
        double delta = std::max(face.delta, 1e-20);

        double F1_f = face.weight * f.F1[o] + (1.0 - face.weight) * f.F1[n];
        double sw = sst_.coeffs.sigma_w(F1_f);
        double rho_f = face.weight * f.rho[o] + (1.0 - face.weight) * f.rho[n];
        double mu_f = face.weight * mu_[o] + (1.0 - face.weight) * mu_[n];
        double nuT_f = face.weight * f.nuT[o] + (1.0 - face.weight) * f.nuT[n];
        double muEff = mu_f + rho_f * sw * nuT_f;
        double Df = muEff * Sf / delta;

        Vec3 Uf = f.U[o] * face.weight + f.U[n] * (1.0 - face.weight);
        double mFlux =
            rho_f * (Uf.x * face.normal.x + Uf.y * face.normal.y + Uf.z * face.normal.z) * Sf;
        double cPos = std::max(mFlux, 0.0);
        double cNeg = std::max(-mFlux, 0.0);

        sys.diag[o] += Df + cPos;
        sys.diag[n] += Df + cNeg;
        sys.upper[fi] = -(Df + cNeg);
        sys.lower[fi] = -(Df + cPos);
    }

    for (int fi = nIF; fi < mesh_.nFaces(); ++fi) {
        const Face &face = mesh_.face(fi);
        int o = face.owner;
        double Sf = face.area, delta = std::max(face.delta, 1e-20);
        double sw = sst_.coeffs.sigma_w(f.F1[o]);
        double muEff = mu_[o] + f.rho[o] * sw * f.nuT[o];
        double Db = muEff * Sf / delta;

        double wb = f.omega.bface(fi);
        Vec3 Ubv = f.U.bface(fi);
        double mFlux =
            f.rho[o] * (Ubv.x * face.normal.x + Ubv.y * face.normal.y + Ubv.z * face.normal.z) * Sf;
        if (mFlux >= 0) {
            sys.diag[o] += mFlux + Db;
            sys.source[o] += Db * wb;
        } else {
            sys.diag[o] += Db;
            sys.source[o] += (Db - mFlux) * wb;
        }
    }

    for (int ci = 0; ci < mesh_.nCells(); ++ci) {
        double vol = mesh_.cell(ci).volume;
        double F1 = f.F1[ci];
        double rhoC = f.rho[ci];
        double omC = std::max(f.omega[ci], 1e-20);

        // production: rho * alpha * Pk_limited/nuT (SST-2003 corrected form, bounded
        // above by the rho*alpha*S^2 misprint; see incompressible assembly note)
        double alphaB = sst_.coeffs.alpha(F1);
        double S = Smag[ci];
        sys.source[ci] +=
            rhoC * alphaB * sst_.productionOmega(f.nuT[ci], S, f.k[ci], f.omega[ci]) * vol;

        double betaB = sst_.coeffs.beta(F1);
        sys.diag[ci] += rhoC * betaB * omC * vol;

        // cross-diffusion: rho-weighted like the other omega sources (f.CDkw is the
        // kinematic 2*sigma_w2/omega gradK.gradOmega), UNCLIPPED per SST-2003
        sys.source[ci] += rhoC * (1.0 - F1) * f.CDkw[ci] * vol;
    }

    const double alphaW = settings_.alphaOmega;
    for (int ci = 0; ci < mesh_.nCells(); ++ci) {
        sys.source[ci] += (1.0 - alphaW) / alphaW * sys.diag[ci] * f.omega[ci];
        sys.diag[ci] /= alphaW;
    }
}

// ─── Correction steps ─────────────────────────────────────────────────────────

void CompressibleSIMPLESolver::correctVelocity(CompressibleFlowFields &f, const ScalarField &pPrime,
                                               const std::vector<double> &aP) {
    VectorField gradPp = greenGaussGrad(pPrime);
    for (int ci = 0; ci < mesh_.nCells(); ++ci) {
        double rAP = (std::abs(aP[ci]) > 1e-30) ? 1.0 / aP[ci] : 0.0;
        double vol = mesh_.cell(ci).volume;
        f.U[ci].x -= rAP * gradPp[ci].x * vol;
        f.U[ci].y -= rAP * gradPp[ci].y * vol;
        f.U[ci].z -= rAP * gradPp[ci].z * vol;
    }
    // Re-apply BCs via the FlowBoundaryConditions form
    FlowBoundaryConditions fbc;
    fbc.velocityBC = bcs_.velocityBC;
    fbc.pressureBC = bcs_.pressureBC;
    fbc.kBC = bcs_.kBC;
    fbc.omegaBC = bcs_.omegaBC;
    applyVelocityBC(f.U, mesh_, fbc);
}

void CompressibleSIMPLESolver::correctPressure(CompressibleFlowFields &f,
                                               const ScalarField &pPrime) {
    for (int ci = 0; ci < mesh_.nCells(); ++ci)
        f.p[ci] += settings_.alphaP * pPrime[ci];

    FlowBoundaryConditions fbc;
    fbc.velocityBC = bcs_.velocityBC;
    fbc.pressureBC = bcs_.pressureBC;
    fbc.kBC = bcs_.kBC;
    fbc.omegaBC = bcs_.omegaBC;
    applyPressureBC(f.p, mesh_, fbc);
    mechanicalizeOutletPressure(f);
}

double CompressibleSIMPLESolver::computeResidual(const CompressibleFlowFields &, int) {
    return 0.0; // tracked via solver residuals in the main loop
}

double CompressibleSIMPLESolver::computeScalarResidual(const ScalarField &old_phi,
                                                       const ScalarField &new_phi) {
    double maxVal = 1e-30, maxDiff = 0.0;
    for (int ci = 0; ci < mesh_.nCells(); ++ci) {
        maxVal = std::max(maxVal, std::abs(new_phi[ci]));
        maxDiff = std::max(maxDiff, std::abs(new_phi[ci] - old_phi[ci]));
    }
    return maxDiff / maxVal;
}

// ─── Main SIMPLE loop ─────────────────────────────────────────────────────────

CompressibleConvergenceHistory CompressibleSIMPLESolver::solve(CompressibleFlowFields &f) {
    mesh_.requireStamp(meshStamp_);
    f.requireLayout(mesh_);
    CompressibleConvergenceHistory hist;

    LinearSystem momSys = makeSystem(mesh_);
    LinearSystem pSys = makeSystem(mesh_);
    LinearSystem kSys = makeSystem(mesh_);
    LinearSystem omSys = makeSystem(mesh_);
    LinearSystem eSys = makeSystem(mesh_);
    ScalarField pPrime(mesh_, "p'");
    ScalarField SmagFrozen(mesh_, "Smag");

    // carried turbulence change norms (initialised to 1 so nothing can claim
    // turbulence convergence before the first update)
    double lastKChange = 1.0, lastOmChange = 1.0;

    VectorField Uprev(mesh_, "Uprev");
    for (int iter = 0; iter < settings_.maxIterations; ++iter) {
        for (int ci = 0; ci < mesh_.nCells(); ++ci)
            Uprev[ci] = f.U[ci];

        // 1. Update thermodynamic state: ρ, μ from current p, T
        updateDensity(f);
        updateViscosity(f);

        // 2. SST turbulence fields
        bool turbActive = (iter >= settings_.turbStartIter);
        bool turbUpdate =
            turbActive && ((iter - settings_.turbStartIter) % settings_.turbUpdateInterval == 0);

        if (turbUpdate) {
            // Compute kinematic eddy viscosity nuT = μT/ρ for SST model input
            // (SSTModel expects kinematic quantities)
            ScalarField nuT_k(mesh_, "nuT_k");
            for (int ci = 0; ci < mesh_.nCells(); ++ci)
                nuT_k[ci] = f.nuT[ci]; // already kinematic in our storage

            ScalarField k_scaled(mesh_, "k_s");
            for (int ci = 0; ci < mesh_.nCells(); ++ci)
                k_scaled[ci] = f.k[ci];

            double nu0 = mu_[0] / std::max(f.rho[0], 1e-30);
            // LOCAL kinematic viscosity mu(T)/rho per cell for the F1/F2
            // blending arguments (the inlet-cell nu0 remains only as the
            // steady startup-floor value below)
            ScalarField nuLocal(mesh_, "nuLocal");
            for (int ci = 0; ci < mesh_.nCells(); ++ci)
                nuLocal[ci] = mu_[ci] / std::max(f.rho[ci], 1e-30);
            sst_.computeFields(mesh_, k_scaled, f.omega, f.U, nuLocal, f.nuT, f.F1, f.F2, f.Pk,
                               f.CDkw);

            SmagFrozen = strainRateMagnitude(computeVelocityGradients(f.U));

            // startup-only floor (see SolverSettings::nuTFloorIters); releases
            // to a non-negativity clamp afterwards, and warm restarts
            // (turbEstablished carried in the fields) never re-engage it. The
            // floor VALUE stays the steady inlet-cell nu0: a per-cell
            // mu(T)/rho floor jitters with the startup pressure transient
            // (rho = p/RT locally) and destabilizes marginal developing
            // channels; a startup-only numerical guard wants a constant, and
            // the local-viscosity treatment belongs to the blending
            // functions, not the floor.
            if (!f.turbEstablished && iter >= settings_.turbStartIter + settings_.nuTFloorIters)
                f.turbEstablished = true;
            const double nuTMin = f.turbEstablished ? 0.0 : 0.1 * nu0;
            for (int ci = 0; ci < mesh_.nCells(); ++ci)
                f.nuT[ci] = std::max(f.nuT[ci], nuTMin);
        }

        // 3. Momentum equations
        assembleMomentum(momSys, f, 0, aP_);
        std::vector<double> Ux(mesh_.nCells());
        for (int ci = 0; ci < mesh_.nCells(); ++ci)
            Ux[ci] = f.U[ci].x;
        SolverResult resUx =
            mSolver_->solve(momSys, Ux, settings_.innerIterations, settings_.innerTolerance);
        for (int ci = 0; ci < mesh_.nCells(); ++ci)
            f.U[ci].x = Ux[ci];

        std::vector<double> aPstore = aP_;

        assembleMomentum(momSys, f, 1, aP_);
        std::vector<double> Uy(mesh_.nCells());
        for (int ci = 0; ci < mesh_.nCells(); ++ci)
            Uy[ci] = f.U[ci].y;
        SolverResult resUy =
            mSolver_->solve(momSys, Uy, settings_.innerIterations, settings_.innerTolerance);
        for (int ci = 0; ci < mesh_.nCells(); ++ci)
            f.U[ci].y = Uy[ci];

        assembleMomentum(momSys, f, 2, aP_);
        std::vector<double> Uz(mesh_.nCells());
        for (int ci = 0; ci < mesh_.nCells(); ++ci)
            Uz[ci] = f.U[ci].z;
        SolverResult resUz =
            mSolver_->solve(momSys, Uz, settings_.innerIterations, settings_.innerTolerance);
        for (int ci = 0; ci < mesh_.nCells(); ++ci)
            f.U[ci].z = Uz[ci];

        // Re-apply velocity BCs
        FlowBoundaryConditions fbc_tmp;
        fbc_tmp.velocityBC = bcs_.velocityBC;
        fbc_tmp.pressureBC = bcs_.pressureBC;
        fbc_tmp.kBC = bcs_.kBC;
        fbc_tmp.omegaBC = bcs_.omegaBC;
        applyVelocityBC(f.U, mesh_, fbc_tmp);

        // 4. Pressure correction (with density weighting)
        assemblePressureCorrection(pSys, f, aPstore, pPrime);
        std::vector<double> ppVec(mesh_.nCells(), 0.0);
        SolverResult resP =
            pSolver_->solve(pSys, ppVec, settings_.innerIterations, settings_.innerTolerance);
        for (int ci = 0; ci < mesh_.nCells(); ++ci)
            pPrime[ci] = ppVec[ci];

        // p' boundary conditions
        for (int pi = 0; pi < mesh_.nPatches(); ++pi) {
            const Patch &pat = mesh_.patch(pi);
            if (pat.type == "outlet")
                continue;
            for (FaceID fi : pat.faces)
                pPrime.bface(fi) = pPrime[mesh_.face(fi).owner];
        }

        correctVelocity(f, pPrime, aPstore);
        correctPressure(f, pPrime);

        // velocity field-change norm over the full iteration (momentum solve
        // plus correction): the dimensionless signal that covers BOTH
        // components, in particular the wall-normal one whose raw equation
        // imbalance is not commensurate with the absolute Ux/p thresholds
        double uChange = 0.0;
        {
            double uMaxVal = 1e-30, uMaxDiff = 0.0;
            for (int ci = 0; ci < mesh_.nCells(); ++ci) {
                double mag = std::sqrt(f.U[ci].x * f.U[ci].x + f.U[ci].y * f.U[ci].y +
                                       f.U[ci].z * f.U[ci].z);
                double dx = f.U[ci].x - Uprev[ci].x;
                double dy = f.U[ci].y - Uprev[ci].y;
                double dz = f.U[ci].z - Uprev[ci].z;
                uMaxVal = std::max(uMaxVal, mag);
                uMaxDiff = std::max(uMaxDiff, std::sqrt(dx * dx + dy * dy + dz * dz));
            }
            uChange = uMaxDiff / uMaxVal;
        }

        // 5. Energy equation for T. The convergence signal is the FIELD-CHANGE
        // norm ||T_new - T_old||_inf / ||T||_inf (the same convention as k and
        // omega): at low Mach the temperature is nearly unforced, so its
        // dimensional equation residual decays too slowly for a reduction
        // criterion while the field itself stopped changing long before.
        double tChange = 0.0;
        SolverResult resT = {};
        {
            ScalarField T_old(mesh_, "T_old");
            for (int ci = 0; ci < mesh_.nCells(); ++ci)
                T_old[ci] = f.T[ci];

            assembleEnergy(eSys, f);
            std::vector<double> Tvec(mesh_.nCells());
            for (int ci = 0; ci < mesh_.nCells(); ++ci)
                Tvec[ci] = f.T[ci];
            resT = eSolver_->solve(eSys, Tvec, settings_.innerIterations, settings_.innerTolerance);
            for (int ci = 0; ci < mesh_.nCells(); ++ci) {
                // Prevent non-physical T <= 0. With NaN as the FIRST
                // argument, std::max(NaN, 1.0) returns that first argument,
                // so this clamp preserves a NaN for stateIsValid to detect.
                // We also retain the linear-solver result below so a
                // non-finite residual is diagnosed directly rather than only
                // through its eventual effect on the field.
                f.T[ci] = std::max(Tvec[ci], 1.0);
            }
            applyTemperatureBC(f.T, mesh_, bcs_);

            double tMaxVal = 1e-30, tMaxDiff = 0.0;
            for (int ci = 0; ci < mesh_.nCells(); ++ci) {
                tMaxVal = std::max(tMaxVal, std::abs(f.T[ci]));
                tMaxDiff = std::max(tMaxDiff, std::abs(f.T[ci] - T_old[ci]));
            }
            tChange = tMaxDiff / tMaxVal;

            // Update density from updated p, T
            updateDensity(f);
        }

        // 6. Turbulence equations
        SolverResult resK = {}, resOm = {};
        if (turbUpdate) {
            // pre-solve fields for the change-norm convergence metric (the
            // wall re-pinning below keeps the omega EQUATION imbalance
            // permanently nonzero, so field change is the honest signal,
            // exactly as the incompressible solver tracks it)
            std::vector<double> kOld = f.k.data();
            std::vector<double> omOld = f.omega.data();

            assembleKEquation(kSys, f);
            std::vector<double> kVec = f.k.data();
            resK = tSolver_->solve(kSys, kVec, settings_.innerIterations, settings_.innerTolerance);
            for (int ci = 0; ci < mesh_.nCells(); ++ci)
                f.k[ci] = kVec[ci];
            f.k.clamp(settings_.kMin, settings_.kMax);
            applyKBC(f.k, mesh_, fbc_tmp);

            assembleOmegaEquation(omSys, f, SmagFrozen);
            std::vector<double> omVec = f.omega.data();
            resOm =
                tSolver_->solve(omSys, omVec, settings_.innerIterations, settings_.innerTolerance);
            for (int ci = 0; ci < mesh_.nCells(); ++ci)
                f.omega[ci] = omVec[ci];
            f.omega.clamp(settings_.omegaMin, 1e15);
            // boundary-face omega with the LOCAL owner-cell viscosity, not
            // the inlet value (matching the wall-anchor treatment)
            ScalarField nuLocalBC(mesh_, "nuLocalBC");
            for (int ci = 0; ci < mesh_.nCells(); ++ci)
                nuLocalBC[ci] = mu_[ci] / std::max(f.rho[ci], 1e-30);
            applyOmegaBC(f.omega, mesh_, fbc_tmp, nuLocalBC, sst_.coeffs.beta1);

            // Re-pin near-wall omega. When settings_.useWallFunctions is on,
            // blend the resolved-wall form with the log-law form so the first
            // cell may sit in the log layer (y+ >= 30).
            const double betaStar = sst_.coeffs.betaStar;
            const double kappa = settings_.vonKarman;
            for (int pi = 0; pi < mesh_.nPatches(); ++pi) {
                const Patch &pat = mesh_.patch(pi);
                if (pat.type != "wall")
                    continue;
                for (FaceID fi : pat.faces) {
                    const Face &face = mesh_.face(fi);
                    int o = face.owner;
                    double y1 = std::max(face.delta, 1e-20);
                    // wall anchor with the LOCAL owner-cell viscosity
                    double nuO = mu_[o] / std::max(f.rho[o], 1e-30);
                    double omRes = 60.0 * nuO / (sst_.coeffs.beta1 * y1 * y1);
                    if (settings_.useWallFunctions) {
                        double k_p = std::max(f.k[o], 1e-30);
                        double uTau = std::sqrt(std::sqrt(betaStar) * k_p);
                        double omLog = uTau / (std::sqrt(betaStar) * kappa * y1);
                        f.omega[o] = std::sqrt(omRes * omRes + omLog * omLog);
                    } else {
                        f.omega[o] = omRes;
                    }
                }
            }

            // field-change norms ||new - old||_inf / ||new||_inf, CARRIED
            // between updates (never fresh zeros)
            double kMaxVal = 1e-30, omMaxVal = 1e-30;
            double kMaxDiff = 0.0, omMaxDiff = 0.0;
            for (int ci = 0; ci < mesh_.nCells(); ++ci) {
                kMaxVal = std::max(kMaxVal, std::abs(f.k[ci]));
                omMaxVal = std::max(omMaxVal, std::abs(f.omega[ci]));
                kMaxDiff = std::max(kMaxDiff, std::abs(f.k[ci] - kOld[ci]));
                omMaxDiff = std::max(omMaxDiff, std::abs(f.omega[ci] - omOld[ci]));
            }
            lastKChange = kMaxDiff / kMaxVal;
            lastOmChange = omMaxDiff / omMaxVal;
        }

        // 7. Convergence and divergence check. Turbulence change norms are
        // CARRIED between updates (never fresh zeros), the temperature
        // residual is recorded, and convergence judges EVERY solved
        // equation, not only Ux and p.
        // resP is the pressure-correction (continuity) imbalance, so mass
        // conservation is inside the criterion through nP.
        CompressibleResidualEntry entry;
        entry.iteration = iter + 1;
        entry.Ux = resUx.initialRes;
        entry.Uy = resUy.initialRes;
        entry.p = resP.initialRes;
        entry.T = tChange;
        entry.k = turbActive ? lastKChange : 0.0;
        entry.omega = turbActive ? lastOmChange : 0.0;
        hist.entries.push_back(entry);

        // Check for divergence. Every recorded residual/norm and both
        // residuals returned by every linear solve must be finite. The state
        // itself is also validated DIRECTLY per cell (stateIsValid): a
        // reduction such as std::max(finite, NaN) keeps its finite first
        // argument, so aggregate norms alone cannot prove field integrity.
        // The amplitude limit then runs on a state known finite.
        auto solverResultFinite = [](const SolverResult &r) {
            return std::isfinite(r.initialRes) && std::isfinite(r.finalRes);
        };
        bool diverged =
            !std::isfinite(entry.Ux) || !std::isfinite(entry.Uy) || !std::isfinite(entry.p) ||
            !std::isfinite(entry.T) || !std::isfinite(entry.k) || !std::isfinite(entry.omega) ||
            !std::isfinite(uChange) || !std::isfinite(tChange) || !solverResultFinite(resUx) ||
            !solverResultFinite(resUy) || !solverResultFinite(resUz) || !solverResultFinite(resP) ||
            !solverResultFinite(resT) ||
            (turbUpdate && (!solverResultFinite(resK) || !solverResultFinite(resOm))) ||
            !stateIsValid(f);
        if (!diverged) {
            double pMax = 0.0;
            for (int ci = 0; ci < mesh_.nCells(); ++ci)
                pMax = std::max(pMax, std::abs(f.p[ci]));
            diverged = (pMax > settings_.divergenceLimit);
        }
        if (diverged) {
            hist.diverged = true;
            hist.finalIter = iter + 1;
            if (settings_.verbose)
                std::cout << "  DIVERGED at iteration " << iter + 1 << "\n";
            return hist;
        }

        if (settings_.verbose && ((iter + 1) % settings_.reportInterval == 0)) {
            std::cout << "  iter " << iter + 1 << "  Ux=" << entry.Ux << "  Uy=" << entry.Uy
                      << "  p=" << entry.p << "  T=" << entry.T << "  k=" << entry.k
                      << "  om=" << entry.omega << "\n";
        }

        // Check convergence over EVERY solved quantity. Ux and p keep this
        // solver's established ABSOLUTE-threshold semantics for
        // convergenceTol (the regression baselines are defined with it); the
        // velocity FIELD (both components, in particular the wall-normal one
        // whose raw equation imbalance is not commensurate with those
        // thresholds), the temperature, and the carried k/omega all enter as
        // dimensionless field-change norms. No solved quantity is ignored,
        // skipped turbulence updates can never inject a false zero, and
        // convergence is withheld until the startup nuT floor has released
        // so no run freezes a floored near-wall state.
        double tol = settings_.convergenceTol;
        double maxRes = std::max({entry.Ux, entry.Uy, entry.p, uChange, tChange});
        if (turbActive)
            maxRes = std::max({maxRes, lastKChange, lastOmChange});
        // convergence requires that scheduled turbulence has actually run and
        // its startup floor released: without the turbScheduled gate a solve
        // could declare convergence BEFORE its first turbulence update, on a
        // laminar transient the criterion never sees
        const bool turbScheduled = settings_.turbStartIter < settings_.maxIterations;
        bool converged =
            (maxRes < tol) && (iter > 0) && (!turbScheduled || (turbActive && f.turbEstablished));
        if (converged) {
            // k participates in the two-pressure EOS but is intentionally
            // lagged during the segregated turbulence sweep. Refresh the
            // algebraic thermodynamic state once at the accepted fixed point
            // so returned rho and the outlet face correspond to final k.
            updateDensity(f);
            if (!stateIsValid(f)) {
                hist.diverged = true;
                hist.finalIter = iter + 1;
                return hist;
            }
            hist.converged = true;
            hist.finalIter = iter + 1;
            if (settings_.verbose)
                std::cout << "  Converged at iteration " << iter + 1 << "\n";
            return hist;
        }
    }

    hist.finalIter = settings_.maxIterations;
    return hist;
}
