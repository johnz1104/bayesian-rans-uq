#pragma once

#include "Mesh.hpp"
#include "Field.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

// Container for SST k-omega turbulence model constants
struct SSTCoefficients {
    // k-omega model coefficients (near-wall layer)
    double sigma_k1 = 0.85;
    double sigma_w1 = 0.5;
    double beta1 = 0.075;
    double alpha1 = 5.0 / 9.0; // ~0.5556

    // k-epsilon model coefficients (expressed in k-omega form) (away-from-wall layer)
    double sigma_k2 = 1.0;
    double sigma_w2 = 0.856;
    double beta2 = 0.0828;
    double alpha2 = 0.44;

    // general SST constants
    double betaStar = 0.09;
    double a1 = 0.31;    // Bradshaw structural parameter
    double kappa = 0.41; // von Karman (usually fixed)

    void validate() const {
        const std::pair<const char *, double> constants[] = {
            {"sigma_k1", sigma_k1}, {"sigma_w1", sigma_w1},
            {"beta1", beta1},       {"alpha1", alpha1},
            {"sigma_k2", sigma_k2}, {"sigma_w2", sigma_w2},
            {"beta2", beta2},       {"alpha2", alpha2},
            {"betaStar", betaStar}, {"a1", a1},
            {"kappa", kappa}};
        for (const auto &entry : constants)
            if (!std::isfinite(entry.second) || entry.second < 0.)
                throw std::invalid_argument(std::string("SST ") + entry.first +
                                            " must be finite and nonnegative");
        // These coefficients occur in wall/blending denominators. Other
        // nonnegative term coefficients may be zero to disable that term.
        if (beta1 == 0. || betaStar == 0.)
            throw std::invalid_argument("SST beta1 and betaStar must be positive");
    }

    // blended coefficient: phi = F1*phi1 + (1-F1)*phi2
    // F1 = 1: pure k-omega
    // F1 = 0: pure k-epsilon
    double blend(double phi1, // k-omega coefficient
                 double phi2, // k-epsilon coefficient
                 double F1)   // blending function
        const {
        return F1 * phi1 + (1.0 - F1) * phi2;
    }
    double sigma_k(double F1) const { return blend(sigma_k1, sigma_k2, F1); }
    double sigma_w(double F1) const { return blend(sigma_w1, sigma_w2, F1); }
    double beta(double F1) const { return blend(beta1, beta2, F1); }
    double alpha(double F1) const { return blend(alpha1, alpha2, F1); }
};

// Term-toggled closure variants for model comparisons.
// Competing models are toggles in the SAME SST model, never new solvers:
//   Full      — standard Menter SST.
//   NoLimiter — drop the Bradshaw shear-stress limiter: nuT = a1 k/max(a1 w, S F2)
//               -> nuT = k/w  (standard k-w eddy viscosity).
//   KOmega    — force F1 = 1 everywhere (baseline k-w, no k-e blending).
enum class SSTVariant { Full = 0, NoLimiter = 1, KOmega = 2 };

// Pointwise analytic sensitivities of the closure quantities that enter the discrete
// residual, w.r.t. each of the 11 SST coefficients θ, at FIXED primary state
// (k, ω, U). These coefficient partials do not provide an analytic state Jacobian
// or adjoint. Indexing matches InferenceParameterSet:
//   0 σ_k1  1 σ_w1  2 β1  3 α1  4 σ_k2  5 σ_w2  6 β2  7 α2  8 β*  9 a1  10 κ
// Only nuT (floored, for diffusion), Pk, F1 and CDkw enter the assembly directly; F2
// is upstream of nuT and so appears only inside dnuT (the β* path).
struct SSTClosureSensitivity {
    double dnuT[11] = {0};  // ∂(floored nuT)/∂θ_j   — momentum/k/ω diffusion
    double dPk[11] = {0};   // ∂Pk/∂θ_j              — k production (uses UNfloored nuT)
    double dF1[11] = {0};   // ∂F1/∂θ_j              — σ_k/σ_w/α/β blends + cross-diff
    double dCDkw[11] = {0}; // ∂CDkw/∂θ_j (raw)      — ω cross-diffusion
};

// Reusable geometric-field storage, with no cached field values or gradients.
struct SSTWorkspace {
    VelocityGradientWorkspace components;
    VelocityGradients velocity;
    VectorField gradK, gradOmega;
    explicit SSTWorkspace(const Mesh &mesh)
        : components(mesh), velocity{VectorField(mesh, "grad(Ux)"), VectorField(mesh, "grad(Uy)"),
                                     VectorField(mesh, "grad(Uz)")},
          gradK(mesh, "grad(k)"), gradOmega(mesh, "grad(omega)") {}
};

// SSTModel
// Computes F1, F2, nuT, Pk, and source terms
class SSTModel {
  public:
    SSTCoefficients coeffs;
    SSTVariant variant = SSTVariant::Full; // closure variant

    // Constructors
    SSTModel() = default;
    explicit SSTModel(const SSTCoefficients &c) : coeffs(c) { validate(); }
    SSTModel(const SSTCoefficients &c, SSTVariant v) : coeffs(c), variant(v) { validate(); }

    // Public coefficients and variant may be mutated after construction.
    void validate() const {
        coeffs.validate();
        if (variant != SSTVariant::Full && variant != SSTVariant::NoLimiter &&
            variant != SSTVariant::KOmega)
            throw std::invalid_argument("unknown SST model variant");
    }

    // Pointwise functions - operate on one control volume (single cell)
    double computeF1(double k, double omega, double y, double nu, double CDkw_pos)
        const; // primary blending function (detects if cell is in boundary layer)
    double computeF2(double k, double omega, double y,
                     double nu) const; // second blending function (used in eddy viscosity limiter)
    double crossDiffusion(const Vec3 &gradK, const Vec3 &gradOmega,
                          double omega) const; // computes cross-diffusion term in omega equation
    double eddyViscosity(double k, double omega, double S,
                         double F2) const; // computes turbulent viscosity
    double production(double nuT, double S, double k,
                      double omega) const; // computes turbulence production term
    double productionOmega(double nuT, double S, double k, double omega)
        const; // specific omega production Pk_limited/nuT (SST-2003 corrected form)
    double sourceK(double Pk, double k, double omega) const; // computes source term in k equation
    double sourceOmega(double S, double nuT, double k, double omega, double F1,
                       double CDkw)
        const; // reference omega-equation RHS (corrected 2003 form, unclipped cross-diffusion)

    // ADJOINT GROUNDWORK — pointwise analytic ∂(closure)/∂θ at one cell, holding the
    // primary state (k, ω, U) fixed.  Re-derives the same branch decisions computeFields
    // makes (eddy-viscosity Bradshaw branch, F2/F1 min/max winners, Pk Menter limiter,
    // nuT floor) so the analytic derivative matches a FD of computeFields exactly.
    //   k,omega          — cell k and ω (raw; safeguarded internally as in computeFields)
    //   S                — strain-rate magnitude |S|
    //   y,nu             — wall distance, molecular viscosity
    //   CDkw             — raw cross-diffusion value (signed) for this cell
    //   nuTFloor         — eddy-viscosity floor the solver applies (0.1·ν); dnuT=0 below it
    SSTClosureSensitivity closureSensitivity(double k, double omega, double S, double y, double nu,
                                             double CDkw, double nuTFloor) const;

    // Computes for the entire mesh
    void computeFields(const Mesh &mesh, const ScalarField &k, const ScalarField &omega,
                       const VectorField &U, double nu, ScalarField &nuT, ScalarField &F1field,
                       ScalarField &F2field, ScalarField &Pk, ScalarField &CDkwField) const;

    // Also retain the pre-correction strain used by this closure evaluation.
    // All fields must have matching mesh identity and interior/boundary storage.
    // strainOut must be distinct from k, omega and every scalar closure output;
    // invalid storage/aliasing is rejected before any field is changed.
    // Its interior is overwritten and its boundary values are reset to zero.
    void computeFields(const Mesh &mesh, const ScalarField &k, const ScalarField &omega,
                       const VectorField &U, double nu, ScalarField &nuT, ScalarField &F1field,
                       ScalarField &F2field, ScalarField &Pk, ScalarField &CDkwField,
                       ScalarField &strainOut) const;
    void computeFields(const Mesh &mesh, const ScalarField &k, const ScalarField &omega,
                       const VectorField &U, double nu, ScalarField &nuT, ScalarField &F1field,
                       ScalarField &F2field, ScalarField &Pk, ScalarField &CDkwField,
                       ScalarField &strainOut, SSTWorkspace &workspace) const;

    // Per-cell-viscosity overload: identical closure evaluation with the
    // LOCAL kinematic viscosity mu(T)/rho entering the F1/F2 blending
    // arguments, replacing the low-Mach shortcut of a single inlet-cell value.
    void computeFields(const Mesh &mesh, const ScalarField &k, const ScalarField &omega,
                       const VectorField &U, const ScalarField &nuLocal, ScalarField &nuT,
                       ScalarField &F1field, ScalarField &F2field, ScalarField &Pk,
                       ScalarField &CDkwField) const;
};
