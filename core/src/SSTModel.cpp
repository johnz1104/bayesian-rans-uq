#include "SSTModel.hpp"
#include <cmath>
#include <algorithm>

namespace {
// SST-2003 blending uses this floor only in the F1 denominator. The omega
// source retains the signed, unfloored cross-diffusion value.
constexpr double crossDiffusionFloor = 1e-10;
} // namespace

// Pointwise functions

// F1 blending: switches between inner (k-omega) and outer (k-epsilon) model
// detects whether cell is near a wall or in free flow
double SSTModel::computeF1(double k, double omega, double y, double nu, double CDkw_pos) const {
    validate();
    // numeric safeguards
    double sqrtK = std::sqrt(std::max(k, 0.0));
    double omSafe = std::max(omega, 1e-20);
    double ySafe = std::max(y, 1e-20);
    double y2s = ySafe * ySafe;

    double term1 = sqrtK / (coeffs.betaStar * omSafe * ySafe);
    double term2 = 500.0 * nu / (y2s * omSafe);
    double CDpos = std::max(CDkw_pos, crossDiffusionFloor);
    double term3 = 4.0 * coeffs.sigma_w2 * std::max(k, 0.0) / (CDpos * y2s);

    // arg1 = min( max(sqrt(k)/(betaStar*omega*y), 500*nu/(y^2*omega)), 4*sigma_w2*k / (CDkw_pos *
    // y^2)) F1 = tanh(arg1^4)
    double arg1 = std::min(std::max(term1, term2), term3);
    double a4 = arg1 * arg1 * arg1 * arg1;
    return std::tanh(a4);
}

// F2 blending: used in the eddy viscosity limiter
// detects how close cell is to a wall to limit turbulent eddy viscosity near walls
double SSTModel::computeF2(double k, double omega, double y, double nu) const {
    validate();
    // numeric safeguards
    double sqrtK = std::sqrt(std::max(k, 0.0));
    double omSafe = std::max(omega, 1e-20);
    double ySafe = std::max(y, 1e-20);
    double y2 = ySafe * ySafe;

    double term1 = 2.0 * sqrtK / (coeffs.betaStar * omSafe * ySafe);
    double term2 = 500.0 * nu / (y2 * omSafe);

    // arg2 = max(2*sqrt(k)/(betaStar*omega*y), 500*nu/(y^2*omega))
    // F2 = tanh(arg2^2)
    double arg2 = std::max(term1, term2);
    return std::tanh(arg2 * arg2);
}

// Cross-diffusion term: CDkw = 2*sigma_w2/omega * (gradK . gradOmega)
double SSTModel::crossDiffusion(const Vec3 &gradK, const Vec3 &gradOmega, double omega) const {
    validate();
    double omSafe = std::max(omega, 1e-20);
    return 2.0 * coeffs.sigma_w2 / omSafe * gradK.dot(gradOmega);
}

// Eddy viscosity with Bradshaw limiter:
// ensures turbulent shear stress scales with turbulent kinetic energy
// prevents excessuve eddy viscosity in high-strain regions (e.g. near walls)
// nuT = a1*k / max(a1*omega, S*F2)   (Full SST)
// NoLimiter variant drops the Bradshaw limiter: nuT = k/omega.
double SSTModel::eddyViscosity(double k, double omega, double S, double F2) const {
    validate();
    double kSafe = std::max(k, 0.0);
    if (variant == SSTVariant::NoLimiter) {
        return kSafe / std::max(omega, 1e-20);
    }
    double denom = std::max(coeffs.a1 * omega, S * F2);
    denom = std::max(denom, 1e-20);
    return coeffs.a1 * kSafe / denom;
}

// Production with Menter limiter:
// prevents overprediction of turbulent kinetic energy at stagnation points (fluid velocity = 0)
// Pk = min(nuT * S^2, 10 * betaStar * k * omega)
double SSTModel::production(double nuT, double S, double k, double omega) const {
    validate();
    double Pk_raw = nuT * S * S;
    double Pk_lim = 10.0 * coeffs.betaStar * std::max(k, 0.0) * std::max(omega, 1e-20);
    return std::min(Pk_raw, Pk_lim);
}

// Specific omega production (SST-2003 corrected form):
// P_omega/alpha = Pk_limited/nuT = min(nuT*S^2, 10*betaStar*k*omega)/nuT
//               = min(S^2, 10*betaStar*k*omega/nuT)
// The 2003 paper prints alpha*S^2; the specification (NASA TMR SST page) corrects it to
// the limited form. The min form is singular-safe: as nuT -> 0 with k > 0 the limiter
// branch blows up and the min selects S^2, which is the correct limit of Pk/nuT there.
// At k = 0 exactly the specification ratio is 0/0 (Pk_limited = 0 and nuT = 0); this
// returns 0, the k-equation-consistent choice (no k production means no omega
// production), erring on the reducing side; solvers seed k > 0 at initialization so the
// state does not persist. Because min(S^2, lim) <= S^2 pointwise, this term is bounded
// above by the S^2 misprint: the corrected production can never exceed the former term
// at the same state (a statement about this term, not about every coupled path in the
// model); the two coincide wherever the k-production limiter is inactive
// (the equilibrium log layer of attached flows; limiter-ACTIVE states, including
// near-wall and startup states of attached runs, shift toward less production).
double SSTModel::productionOmega(double nuT, double S, double k, double omega) const {
    validate();
    double S2 = S * S;
    double lim =
        10.0 * coeffs.betaStar * std::max(k, 0.0) * std::max(omega, 1e-20) / std::max(nuT, 1e-30);
    return std::min(S2, lim);
}

// Source term for k-equation:
// Sk = Pk - betaStar * omega * k
// production minus dissipation of turbulent kinetic energy
double SSTModel::sourceK(double Pk, double k, double omega) const {
    validate();
    return Pk - coeffs.betaStar * std::max(omega, 1e-20) * std::max(k, 0.0);
}

// Source term of omega-equation (reference form; solvers assemble the same terms with
// the destruction kept implicit on the matrix diagonal)
// Sw = alpha*(Pk_limited/nuT) - beta*omega^2 + (1-F1)*CDkw
// The cross-diffusion term enters the omega equation UNCLIPPED; only the CDkw quantity
// inside the F1 argument is floored at 1e-10 by computeF1, matching SST-2003.
double SSTModel::sourceOmega(double S, double nuT, double k, double omega, double F1,
                             double CDkw) const {
    validate();
    double alphaB = coeffs.alpha(F1);
    double betaB = coeffs.beta(F1);
    return alphaB * productionOmega(nuT, S, k, omega) - betaB * omega * omega + (1.0 - F1) * CDkw;
}

// Field function
// computes
void SSTModel::computeFields(const Mesh &mesh, const ScalarField &k, const ScalarField &omega,
                             const VectorField &U, double nu,
                             ScalarField &nuT,      // turbulent eddy viscosity
                             ScalarField &F1field,  // iner/outer model blending
                             ScalarField &F2field,  // eddy viscosity limiter
                             ScalarField &Pk,       // turbulence production
                             ScalarField &CDkwField // cross-diffusion term
) const {
    validate();

    ScalarField Smag(mesh, "|S|");
    computeFields(mesh, k, omega, U, nu, nuT, F1field, F2field, Pk, CDkwField, Smag);
}

void SSTModel::computeFields(const Mesh &mesh, const ScalarField &k, const ScalarField &omega,
                             const VectorField &U, double nu, ScalarField &nuT,
                             ScalarField &F1field, ScalarField &F2field, ScalarField &Pk,
                             ScalarField &CDkwField, ScalarField &strainOut) const {
    validate();
    SSTWorkspace workspace(mesh);
    computeFields(mesh, k, omega, U, nu, nuT, F1field, F2field, Pk, CDkwField, strainOut,
                  workspace);
}

void SSTModel::computeFields(const Mesh &mesh, const ScalarField &k, const ScalarField &omega,
                             const VectorField &U, double nu, ScalarField &nuT,
                             ScalarField &F1field, ScalarField &F2field, ScalarField &Pk,
                             ScalarField &CDkwField, ScalarField &strainOut,
                             SSTWorkspace &workspace) const {
    validate();

    const auto matches = [&mesh](const auto &field) {
        return field.meshPtr() == &mesh && field.size() == mesh.nCells() &&
               field.nBnd() == mesh.nFaces() - mesh.nInternalFaces();
    };
    if (!matches(U) || !matches(strainOut))
        throw std::invalid_argument("SST computeFields requires matching mesh and field storage");
    const ScalarField *fields[] = {&k, &omega, &nuT, &F1field, &F2field, &Pk, &CDkwField};
    for (const ScalarField *field : fields) {
        if (field == &strainOut)
            throw std::invalid_argument("SST strain output must not alias another scalar field");
        if (!matches(*field))
            throw std::invalid_argument(
                "SST computeFields requires matching mesh and field storage");
    }

    U.requireValid();
    strainOut.requireValid();
    for (const ScalarField *field : fields)
        field->requireValid();

    const ScalarField *outputs[] = {&nuT, &F1field, &F2field, &Pk, &CDkwField, &strainOut};
    for (size_t i = 0; i < 6; ++i) {
        if (outputs[i] == &k || outputs[i] == &omega)
            throw std::invalid_argument("SST output cannot alias an input");
        for (size_t j = 0; j < i; ++j)
            if (outputs[i] == outputs[j])
                throw std::invalid_argument("SST outputs must be distinct");
    }
    for (const auto *field :
         {&workspace.components.Ux, &workspace.components.Uy, &workspace.components.Uz}) {
        requireFieldStorage(mesh, *field);
        if (field == &k || field == &omega)
            throw std::invalid_argument("SST scratch cannot alias input");
        for (const auto *output : outputs)
            if (field == output)
                throw std::invalid_argument("SST scratch cannot alias output");
    }
    for (const auto *field : {&workspace.velocity.dudx, &workspace.velocity.dvdx,
                              &workspace.velocity.dwdx, &workspace.gradK, &workspace.gradOmega}) {
        requireFieldStorage(mesh, *field);
        if (field == &U)
            throw std::invalid_argument("SST scratch cannot alias velocity input");
    }
    (void)mesh.wallDistance(); // Reject unprepared geometry before changing outputs.
    computeVelocityGradients(U, workspace.velocity, workspace.components);
    strainRateMagnitude(workspace.velocity, strainOut);
    greenGaussGrad(k, workspace.gradK);
    greenGaussGrad(omega, workspace.gradOmega);
    const auto &gradK = workspace.gradK;
    const auto &gradOmega = workspace.gradOmega;

    // computes distance to the nearest wall
    const auto &wd = mesh.wallDistance();

    // loops over cells
    for (int ci = 0; ci < mesh.nCells(); ++ci) {
        double kc = std::max(k[ci], 0.0);
        double wc = std::max(omega[ci], 1e-20);
        double y = std::max(wd[ci], 1e-20);
        double Sc = strainOut[ci];

        // cross-diffusion
        double CDkw = crossDiffusion(gradK[ci], gradOmega[ci], wc);
        CDkwField[ci] = CDkw;

        // blending functions
        // KOmega variant forces F1 = 1 (pure k-w; no k-e blending of coefficients
        // and the (1-F1) cross-diffusion term in the omega equation vanishes).
        F1field[ci] = (variant == SSTVariant::KOmega) ? 1.0 : computeF1(kc, wc, y, nu, CDkw);
        F2field[ci] = computeF2(kc, wc, y, nu);

        // eddy viscosity
        nuT[ci] = eddyViscosity(kc, wc, Sc, F2field[ci]);

        // production
        Pk[ci] = production(nuT[ci], Sc, kc, wc);
    }
}

// Per-cell-viscosity overload (see header): same loop with nuLocal[ci] in the
// blending functions.
void SSTModel::computeFields(const Mesh &mesh, const ScalarField &k, const ScalarField &omega,
                             const VectorField &U, const ScalarField &nuLocal, ScalarField &nuT,
                             ScalarField &F1field, ScalarField &F2field, ScalarField &Pk,
                             ScalarField &CDkwField) const {
    validate();

    requireFieldStorage(mesh, U);
    const ScalarField *inputs[] = {&k, &omega, &nuLocal};
    const ScalarField *outputs[] = {&nuT, &F1field, &F2field, &Pk, &CDkwField};
    for (const auto *input : inputs)
        requireFieldStorage(mesh, *input);
    for (size_t i = 0; i < 5; ++i) {
        requireFieldStorage(mesh, *outputs[i]);
        for (const auto *input : inputs)
            if (outputs[i] == input)
                throw std::invalid_argument("SST output cannot alias an input");
        for (size_t j = 0; j < i; ++j)
            if (outputs[i] == outputs[j])
                throw std::invalid_argument("SST outputs must be distinct");
    }
    (void)mesh.wallDistance();

    VelocityGradients vg = computeVelocityGradients(U);
    ScalarField Smag = strainRateMagnitude(vg);
    VectorField gradK = greenGaussGrad(k);
    VectorField gradOmega = greenGaussGrad(omega);
    const auto &wd = mesh.wallDistance();

    for (int ci = 0; ci < mesh.nCells(); ++ci) {
        double kc = std::max(k[ci], 0.0);
        double wc = std::max(omega[ci], 1e-20);
        double y = std::max(wd[ci], 1e-20);
        double Sc = Smag[ci];
        double nuC = nuLocal[ci];

        double CDkw = crossDiffusion(gradK[ci], gradOmega[ci], wc);
        CDkwField[ci] = CDkw;

        F1field[ci] = (variant == SSTVariant::KOmega) ? 1.0 : computeF1(kc, wc, y, nuC, CDkw);
        F2field[ci] = computeF2(kc, wc, y, nuC);

        nuT[ci] = eddyViscosity(kc, wc, Sc, F2field[ci]);
        Pk[ci] = production(nuT[ci], Sc, kc, wc);
    }
}

// Pointwise analytic coefficient partials ∂(closure)/∂θ at fixed primary state.
// Mirrors computeF1/computeF2/eddyViscosity/production exactly, then differentiates the
// ACTIVE branch w.r.t. each coefficient.  No derivative is ever taken w.r.t. the fields
// (k, ω, U): these are partials for the 11 coefficients, not a state Jacobian.
SSTClosureSensitivity SSTModel::closureSensitivity(double k, double omega, double S, double y,
                                                   double nu, double CDkw, double nuTFloor) const {
    validate();
    SSTClosureSensitivity d; // zero-initialised: untouched coefficients stay 0

    // --- safeguarded primitives (identical to computeF1/F2/computeFields) -----------
    const double kc = std::max(k, 0.0);
    const double sqrtK = std::sqrt(kc);
    const double wc = std::max(omega, 1e-20);
    const double ys = std::max(y, 1e-20);
    const double y2 = ys * ys;
    const double bStar = coeffs.betaStar;

    // ============================ F2 (β* only) ======================================
    // arg2 = max(2√k/(β* ω y), 500ν/(y²ω));  F2 = tanh(arg2²)
    const double t1F2 = 2.0 * sqrtK / (bStar * wc * ys);
    const double t2F2 = 500.0 * nu / (y2 * wc);
    const double arg2 = std::max(t1F2, t2F2);
    const double F2 = std::tanh(arg2 * arg2);
    double dF2_dbStar = 0.0;
    if (t1F2 >= t2F2 && bStar > 0.0) {
        // d(arg2)/dβ* = d t1F2/dβ* = −t1F2/β*;  dF2 = (1−F2²)·2·arg2·d(arg2)
        dF2_dbStar = (1.0 - F2 * F2) * 2.0 * arg2 * (-t1F2 / bStar);
    }

    // ============================ F1 (β* and floor-active σ_w2) =====================
    // arg1 = min( max(√k/(β* ω y), 500ν/(y²ω)), 4σ_w2 k/(CDpos y²) );  F1 = tanh(arg1⁴)
    // KOmega forces F1≡1 (constant) ⇒ dF1 ≡ 0.
    if (variant != SSTVariant::KOmega) {
        const double t1 = sqrtK / (bStar * wc * ys);
        const double t2 = 500.0 * nu / (y2 * wc);
        const double CDp = std::max(CDkw, crossDiffusionFloor);
        const double t3 = 4.0 * coeffs.sigma_w2 * kc / (CDp * y2);
        const double m12 = std::max(t1, t2);
        const double arg1 = std::min(m12, t3);
        const double F1 = std::tanh(arg1 * arg1 * arg1 * arg1);
        const double dF1_darg1 = (1.0 - F1 * F1) * 4.0 * arg1 * arg1 * arg1;
        // β* enters only through t1. Above the CD floor, the σ_w2 factors
        // in t3 cancel. Below it, the denominator is constant and the
        // numerator retains its σ_w2 derivative when t3 wins.
        if (arg1 == m12 && t1 >= t2 && bStar > 0.0) {
            d.dF1[8] = dF1_darg1 * (-t1 / bStar); // β* = index 8
        }
        if (t3 < m12 && CDkw < crossDiffusionFloor)
            d.dF1[5] = dF1_darg1 * 4.0 * kc / (CDp * y2);
    }

    // ============================ nuT (a1, β* via F2) ===============================
    // Full/KOmega: nuT_raw = a1 k / max(a1 ω, S F2);  NoLimiter: nuT_raw = k/ω.
    // Pk uses the UNfloored nuT_raw; the diffusion coefficients use max(nuT_raw,floor).
    double dnuTraw_da1 = 0.0, dnuTraw_dbStar = 0.0, nuTraw = 0.0;
    if (variant == SSTVariant::NoLimiter) {
        nuTraw = kc / wc; // independent of θ ⇒ derivs 0
    } else {
        const double a1 = coeffs.a1;
        const double cA = a1 * wc; // candidate: a1·ω
        const double cB = S * F2;  // candidate: S·F2
        double denom = std::max(cA, cB);
        denom = std::max(denom, 1e-20);
        nuTraw = a1 * kc / denom;
        if (denom == cB && cB >= cA && cB >= 1e-20) {
            // limiter active (branch B): nuT_raw = a1 k/(S F2)
            dnuTraw_da1 = kc / (S * F2);
            dnuTraw_dbStar = -a1 * kc / (S * F2 * F2) * dF2_dbStar;
        } else if (denom == cA) {
            // branch A: nuT_raw = k/ω, a1 cancels ⇒ both derivs 0
        } else {
            // degenerate 1e-20 floor on denom (essentially never for real fields)
            dnuTraw_da1 = kc / denom;
        }
    }
    // Apply the eddy-viscosity floor to the DIFFUSION derivative only.
    const double nuT = std::max(nuTraw, nuTFloor);
    if (nuT > nuTFloor) {           // above floor ⇒ floored nuT == nuT_raw
        d.dnuT[9] = dnuTraw_da1;    // a1 = index 9
        d.dnuT[8] = dnuTraw_dbStar; // β* = index 8
    } // at/below floor ⇒ floored nuT constant ⇒ dnuT 0

    // ============================ Pk (a1, β*; Menter limiter) =======================
    // Pk = min(nuT_raw·S², 10 β* k ω)   (production() is fed the UNfloored nuT_raw)
    const double PkRaw = nuTraw * S * S;
    const double PkLim = 10.0 * bStar * kc * wc;
    if (PkRaw <= PkLim) {
        d.dPk[9] = S * S * dnuTraw_da1;    // a1
        d.dPk[8] = S * S * dnuTraw_dbStar; // β* (via nuT_raw → F2)
    } else {
        d.dPk[8] = 10.0 * kc * wc; // β* (limiter branch)
    }

    // ============================ CDkw (σ_w2 only) ==================================
    // CDkw = 2 σ_w2/ω · (∇k·∇ω) ⇒ ∂CDkw/∂σ_w2 = CDkw/σ_w2  (linear in σ_w2).
    if (coeffs.sigma_w2 != 0.0)
        d.dCDkw[5] = CDkw / coeffs.sigma_w2; // σ_w2 = index 5

    return d;
}
