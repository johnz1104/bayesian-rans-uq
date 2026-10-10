#include "SIMPLESolver.hpp"
#include "AnisotropyTools.hpp"
#include "StressOperators.hpp"
#include "DiagnosticKernels.hpp"
#include "SIMPLEValidation.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace {

// Diagnostic parameter substitution is temporary, including when a field,
// callback, or matrix validation throws before the operation completes.
struct CoefficientRestore {
    SSTCoefficients &destination;
    SSTCoefficients saved;
    explicit CoefficientRestore(SSTCoefficients &coefficients)
        : destination(coefficients), saved(coefficients) {}
    ~CoefficientRestore() { destination = saved; }
};

// Remaining-change estimate for a linearly contracting fixed-point iteration.
// Samples are per-iteration (or per-monitor-interval) relative changes d_n.
// If d_n ~ C rho^n, the largest change in the last window of m samples over
// the largest in the window before it is rho^m, and the change still to come
// after sample n is d_n rho / (1 - rho). Taking window maxima keeps the
// estimate conservative when the decay is not monotone. A change already
// below `negligible` counts as done whatever the rate; otherwise a rate >= 1,
// or too little history, gives infinity.
class ContractionEstimate {
  public:
    explicit ContractionEstimate(int window) : m_(std::max(window, 2)) {}

    void push(double d) { hist_.push_back(d); }
    double last() const {
        return hist_.empty() ? std::numeric_limits<double>::infinity() : hist_.back();
    }

    double rate() const {
        const size_t n = hist_.size(), m = static_cast<size_t>(m_);
        if (n < 2 * m)
            return std::numeric_limits<double>::quiet_NaN();
        double dNew = 0.0, dOld = 0.0;
        for (size_t i = n - m; i < n; ++i)
            dNew = std::max(dNew, hist_[i]);
        for (size_t i = n - 2 * m; i < n - m; ++i)
            dOld = std::max(dOld, hist_[i]);
        if (dOld <= 0.0)
            return dNew > 0.0 ? std::numeric_limits<double>::infinity() : 0.0;
        return std::pow(dNew / dOld, 1.0 / m_);
    }

    double remaining(double negligible) const {
        const double inf = std::numeric_limits<double>::infinity();
        const size_t n = hist_.size(), m = static_cast<size_t>(m_);
        if (n < 2 * m)
            return inf;
        double dNew = 0.0;
        for (size_t i = n - m; i < n; ++i)
            dNew = std::max(dNew, hist_[i]);
        if (dNew <= negligible)
            return dNew;
        const double r = rate();
        if (!(r < 1.0))
            return inf;
        return dNew * r / (1.0 - r);
    }

  private:
    int m_;
    std::vector<double> hist_;
};

} // namespace

// Constructor
// Returns initialized SIMPLESolver object
SIMPLESolver::SIMPLESolver(const Mesh &mesh, const SSTModel &sst, const FlowBoundaryConditions &bcs,
                           double nu, const SolverSettings &settings)
    : mesh_(mesh), sst_(sst), bcs_(bcs), nu_(nu), settings_(settings), aP_(mesh.nCells(), 0.0),
      aPunrelaxed_(mesh.nCells(), 0.0) {
    simple_admission::problem(mesh_, bcs_, nu_, settings_);
    sst_.validate();
    pSolver_ = makeSolver(settings_.pressureSolver);
    mSolver_ = makeSolver(settings_.momentumSolver);
    tSolver_ = makeSolver(settings_.turbulenceSolver);
    preparePlanarOmegaBalance();
}

void SIMPLESolver::preparePlanarOmegaBalance() {
    if (settings_.useWallFunctions || settings_.legacyOmegaWallPinning ||
        mesh_.nFaces() == mesh_.nInternalFaces())
        return;
    const auto &first = mesh_.face(mesh_.nInternalFaces());
    const Vec3 inward = first.normal * -1.0;
    const double origin = first.center.dot(inward);
    double height = 0;
    bool periodic = false;
    // Qualify the complete boundary, not patch names. The extension is valid
    // for two parallel planes; wall-distance minima alone cannot distinguish a
    // plane from a step corner or curved surface.
    for (int fi = mesh_.nInternalFaces(); fi < mesh_.nFaces(); ++fi) {
        const auto &face = mesh_.face(fi);
        const int patch = face.patchID;
        if (mesh_.patch(patch).type != "wall" || bcs_.omegaBC[patch].type != BCType::WallKOmega ||
            std::abs(std::abs(face.normal.dot(inward)) - 1.0) > 1e-12)
            return;
        height = std::max(height, face.center.dot(inward) - origin);
    }
    if (!(height > 0))
        return;
    const double tolerance = 1e-12 * height;
    for (int fi = 0; fi < mesh_.nFaces(); ++fi) {
        const auto &face = mesh_.face(fi);
        // Clustered-cell polygon centroids lose a few relative digits in the
        // thin wall-normal separation. This admits geometric roundoff, not a
        // nonorthogonal discretization (a finite skew is excluded).
        if ((face.d - face.normal * face.delta).norm() > 1e-9 * face.delta ||
            std::abs(face.shift.dot(inward)) > tolerance)
            return;
        periodic = periodic || face.shift.norm2() > 0;
        if (face.isBoundary()) {
            const double y = face.center.dot(inward) - origin;
            if (std::min(std::abs(y), std::abs(y - height)) > tolerance)
                return;
            if ((y < height / 2 ? -1.0 : 1.0) * face.normal.dot(inward) < 1 - 1e-12)
                return;
        }
    }
    if (!periodic)
        return;
    std::vector<double> singular(mesh_.nCells()), laplacian(mesh_.nCells());
    for (int cell = 0; cell < mesh_.nCells(); ++cell) {
        const double y = mesh_.cell(cell).center.dot(inward) - origin;
        if (!(y > 0 && y < height))
            return;
        singular[cell] = 1 / (y * y) + 1 / ((height - y) * (height - y));
        laplacian[cell] = 6 * (std::pow(y, -4) + std::pow(height - y, -4));
    }
    std::vector<double> gradient(mesh_.nInternalFaces());
    for (int fi = 0; fi < mesh_.nInternalFaces(); ++fi) {
        const auto &face = mesh_.face(fi);
        const double y = face.center.dot(inward) - origin;
        if (!(y > 0 && y < height))
            return;
        gradient[fi] = 2 * (std::pow(height - y, -3) - std::pow(y, -3)) * face.normal.dot(inward);
    }
    omegaSingularity_ = std::move(singular);
    omegaNormalGradient_ = std::move(gradient);
    omegaLaplacian_ = std::move(laplacian);
}

double SIMPLESolver::resolvedWallOmega(double wallDistance) const {
    mesh_.requireStamp(meshStamp_);
    if (!std::isfinite(wallDistance) || wallDistance <= 0)
        throw std::invalid_argument("resolved wall distance must be finite and positive");
    const double coefficient = omegaSingularity_.empty() ? 60.0 : 6.0;
    return coefficient * nu_ / (sst_.coeffs.beta1 * wallDistance * wallDistance);
}

void SIMPLESolver::setTargetAnisotropy(const std::vector<double> *b6) {
    if (b6 && (b6->size() != static_cast<size_t>(6 * mesh_.nCells()) ||
               !std::all_of(b6->begin(), b6->end(), [](double x) { return std::isfinite(x); })))
        throw std::invalid_argument("target anisotropy needs six finite values per cell");
    bTarget6_ = b6;
}

void SIMPLESolver::setMomentumSource(const std::vector<Vec3> *source) {
    if (source && (source->size() != static_cast<size_t>(mesh_.nCells()) ||
                   !std::all_of(source->begin(), source->end(), simple_admission::finite)))
        throw std::invalid_argument("momentum source needs one finite vector per cell");
    momSource_ = source;
}

void SIMPLESolver::setInjectionScale(double scale) {
    if (!std::isfinite(scale) || scale < 0.0 || scale > 1.0)
        throw std::invalid_argument("injection scale must be finite and in [0,1]");
    if (productionPolicy_ && scale != 0.0 && scale != 1.0)
        throw std::invalid_argument("custom production supports injection scale endpoints only");
    injectionScale_ = scale;
}

void SIMPLESolver::setProductionPolicy(std::shared_ptr<const ProductionPolicy> policy) {
    if (policy && injectionScale_ != 0.0 && injectionScale_ != 1.0)
        throw std::invalid_argument("custom production supports injection scale endpoints only");
    productionPolicy_ = std::move(policy);
}

void SIMPLESolver::applyProductionPolicy(FlowFields &f, const ScalarField &strain) {
    if (!productionPolicyActive())
        return;
    setTargetAnisotropy(bTarget6_);
    // Separate output keeps the context immutable and validates before commit.
    auto output = f.Pk.data();
    productionPolicy_->kProduction({mesh_, f, sst_.coeffs, strain, *bTarget6_, nu_}, output);
    if (output.size() != static_cast<size_t>(mesh_.nCells()) ||
        !std::all_of(output.begin(), output.end(), [](double x) { return std::isfinite(x); }))
        throw std::runtime_error("production policy returned invalid k production");
    f.Pk.data() = std::move(output);
}

SIMPLESolver::MomentumDiagnostics
SIMPLESolver::diagnoseMomentum(const FlowFields &state, const SSTCoefficients &theta,
                               const std::vector<double> &savedFaceFluxes,
                               ClosureMode closure) const {
    mesh_.requireStamp(meshStamp_);
    auto validField = [&](const auto &field) {
        return field.size() == mesh_.nCells() &&
               field.nBnd() == mesh_.nFaces() - mesh_.nInternalFaces() && &field.mesh() == &mesh_;
    };
    if (!validField(state.U) || !validField(state.p) || !validField(state.k) ||
        !validField(state.omega) || !validField(state.nuT) || !validField(state.F1) ||
        !validField(state.F2) || !validField(state.Pk) || !validField(state.CDkw))
        throw std::invalid_argument(
            "diagnostic fields must match this solver's mesh and dimensions");
    if (savedFaceFluxes.size() != static_cast<size_t>(mesh_.nFaces()) ||
        !std::all_of(savedFaceFluxes.begin(), savedFaceFluxes.end(),
                     [](double x) { return std::isfinite(x); }))
        throw std::invalid_argument("diagnostics need one finite saved flux per face");
    if (closure != ClosureMode::Stored && closure != ClosureMode::Recomputed)
        throw std::invalid_argument("unknown diagnostic closure mode");
    // A fresh solver owns ALL mutable assembly state. In particular, diagnostic
    // calls from a source observer cannot alter the live blend or flux pointer.
    SIMPLESolver diagnostic(mesh_, SSTModel(theta), bcs_, nu_, settings_);
    diagnostic.setTargetAnisotropy(bTarget6_);
    diagnostic.setInjectionScale(injectionScale_);
    diagnostic.setProductionPolicy(productionPolicy_);
    diagnostic.setMomentumSource(momSource_);
    diagnostic.freezeTransposeStress_ = freezeTransposeStress_;
    diagnostic.convFlux_ = &savedFaceFluxes;
    diagnostic.injRelax_ = 1.0;
    MomentumDiagnostics result{state, std::vector<Vec3>(mesh_.nCells()),
                               std::vector<Vec3>(mesh_.nCells())};
    if (closure == ClosureMode::Recomputed) {
        ScalarField strain(mesh_, "diagnostic_strain");
        diagnostic.recomputeClosure(result.fields, strain);
    }
    auto system = makeSystem(mesh_);
    std::vector<double> diagonal(mesh_.nCells()), velocity(mesh_.nCells()),
        residual(mesh_.nCells());
    for (int component = 0; component < 3; ++component) {
        diagnostic.assembleMomentum(system, result.fields, component, diagonal);
        for (int i = 0; i < mesh_.nCells(); ++i) {
            const auto &u = result.fields.U[i];
            velocity[i] = component == 0 ? u.x : component == 1 ? u.y : u.z;
        }
        system.residual(velocity, residual);
        for (int i = 0; i < mesh_.nCells(); ++i) {
            if (component == 0)
                result.residual[i].x = residual[i];
            else if (component == 1)
                result.residual[i].y = residual[i];
            else
                result.residual[i].z = residual[i];
        }
    }
    if (bTarget6_)
        result.freshIntegratedForce = std::move(diagnostic.injSrcBlend_);
    return result;
}

// Sets uniform initial conditions for velocity, pressure, and turbulence variables (k, omega)
// initializes derived SST fields (nuT, F1, F2, Pk, CDkw), and applies boundary conditions before
// iteration
void SIMPLESolver::initUniform(FlowFields &f, const Vec3 &Uinit, double pInit, double kInit,
                               double omegaInit) {
    mesh_.requireStamp(meshStamp_);
    simple_admission::initial(Uinit, pInit, kInit, omegaInit);
    simple_admission::fields(mesh_, f);
    f.U.setUniform(Uinit);
    f.p.setUniform(pInit);
    f.k.setUniform(kInit);
    f.omega.setUniform(omegaInit);
    f.F1.setUniform(1.0);
    f.F2.setUniform(1.0);
    f.Pk.setUniform(0.0);
    f.CDkw.setUniform(0.0);
    f.turbEstablished = false; // cold start: the startup nuT floor window applies

    // Start with the solver's positive near-wall profile. A uniform omega
    // creates a large jump to the constrained first cells on the first
    // turbulence update. The planar balanced discretization uses the interior
    // asymptote; other geometries retain their established wall approximation.
    const auto &wd = mesh_.wallDistance();
    for (int ci = 0; ci < mesh_.nCells(); ++ci) {
        double y = std::max(wd[ci], 1e-20);
        double omWall = resolvedWallOmega(y);
        f.omega[ci] = std::max(omWall, omegaInit);
    }

    // nuT from the updated omega profile
    for (int ci = 0; ci < mesh_.nCells(); ++ci)
        f.nuT[ci] = sst_.coeffs.a1 * kInit / std::max(sst_.coeffs.a1 * f.omega[ci], 1e-20);

    applyAllBCs(f.U, f.p, f.k, f.omega, mesh_, bcs_, nu_);
}

// Momentum equation assembly (Ux, Uy, or Uz)
// constructs a linear system for finite-volume discretization of momentum equation
// builds A_U * U_component = b (where A_U is in LDU and b is source)
void SIMPLESolver::assembleMomentum(LinearSystem &sys, const FlowFields &f, int component,
                                    std::vector<double> &aP) {
    setMomentumSource(momSource_);
    sys.zero(); // resets linear system
    int nIF = mesh_.nInternalFaces();

    // internal faces: convection + diffusion
    for (int fi = 0; fi < nIF; ++fi) {

        // face geometry
        const Face &face = mesh_.face(fi);
        int o = face.owner;
        int n = face.neighbor;
        double Sf = face.area;
        double delta = std::max(face.delta, 1e-20);

        // effective viscosity at face (molecular + eddy)
        double nuEff_o = nu_ + f.nuT[o]; // molecular viscosity
        double nuEff_n = nu_ + f.nuT[n]; // eddy viscosity
        double nuEff_f = face.weight * nuEff_o + (1.0 - face.weight) * nuEff_n; // interpolation

        // diffusion coefficient
        double Df = nuEff_f * Sf / delta;

        // mass flux through face (first-order upwind)
        // flux = rho * U_f . S_f   (rho = 1 for incompressible)
        double mFlux = convectiveFlux(f, fi);

        // upwind convection coefficients
        double cPos = std::max(mFlux, 0.0);  // flux from owner to neighbor
        double cNeg = std::max(-mFlux, 0.0); // flux from neighbor to owner

        // owner equation: aP += Df + cPos,  aN = -(Df + cNeg)
        sys.diag[o] += Df + cPos;
        sys.diag[n] += Df + cNeg;
        sys.upper[fi] = -(Df + cNeg); // off-diag contribution to owner from neighbor
        sys.lower[fi] = -(Df + cPos); // off-diag contribution to neighbor from owner
    }

    // boundary faces. At RESOLVED walls the eddy viscosity vanishes on the
    // face (SST asymptotics), so the implicit wall diffusion uses the
    // MOLECULAR viscosity only: extrapolating the owner-cell nuT (nonzero at
    // y+ ~ 1) overestimates the discrete wall shear the momentum equation
    // carries, inconsistently with the molecular wall-stress observation and
    // the wall-zero transpose treatment. Under wall functions the owner
    // value stays (the modeled stress carrier). Non-wall boundaries keep
    // owner extrapolation.
    std::vector<char> isWall(mesh_.nFaces(), 0);
    for (int pi = 0; pi < mesh_.nPatches(); ++pi) {
        const Patch &pat = mesh_.patch(pi);
        if (pat.type != "wall")
            continue;
        for (FaceID wfi : pat.faces)
            isWall[wfi] = 1;
    }
    for (int fi = nIF; fi < mesh_.nFaces(); ++fi) {

        // face geometry
        const Face &face = mesh_.face(fi);
        int o = face.owner;
        double Sf = face.area;
        double delta = std::max(face.delta, 1e-20);
        double nuEff = (isWall[fi] && !settings_.useWallFunctions) ? nu_ : nu_ + f.nuT[o];

        // diffusion term
        double Db = nuEff * Sf / delta;

        // computes boundary velocity component
        double Ub;
        if (component == 0)
            Ub = f.U.bface(fi).x;
        else if (component == 1)
            Ub = f.U.bface(fi).y;
        else
            Ub = f.U.bface(fi).z;

        // boundary mass flux
        double mFlux = convectiveFlux(f, fi);

        if (mFlux >= 0) {
            // outflow: convection from owner
            sys.diag[o] += mFlux + Db;
            sys.source[o] += Db * Ub;
        } else {
            // inflow: fixed boundary value
            sys.diag[o] += Db;
            sys.source[o] += (Db - mFlux) * Ub;
        }
    }

    // pressure gradient source (-dP/dx_component * V)
    VectorField gradP = greenGaussGrad(f.p);
    for (int ci = 0; ci < mesh_.nCells(); ++ci) {
        double vol = mesh_.cell(ci).volume;
        double dpdx;
        if (component == 0)
            dpdx = gradP[ci].x;
        else if (component == 1)
            dpdx = gradP[ci].y;
        else
            dpdx = gradP[ci].z;
        sys.source[ci] -= dpdx * vol;
    }

    // constant body force (drives streamwise-periodic domains, which have no
    // inlet; represents the mean pressure gradient)
    const double fb = (component == 0)   ? settings_.bodyForce.x
                      : (component == 1) ? settings_.bodyForce.y
                                         : settings_.bodyForce.z;
    if (fb != 0.0)
        for (int ci = 0; ci < mesh_.nCells(); ++ci)
            sys.source[ci] += fb * mesh_.cell(ci).volume;

    // per-cell source (manufactured solutions)
    if (momSource_)
        for (int ci = 0; ci < mesh_.nCells(); ++ci) {
            const Vec3 &sv = (*momSource_)[ci];
            sys.source[ci] += (component == 0   ? sv.x
                               : component == 1 ? sv.y
                                                : sv.z) *
                              mesh_.cell(ci).volume;
        }

    // variable-viscosity transpose stress div(nuT (grad U)^T), the explicit
    // half completing the Boussinesq deviatoric stress divergence (the
    // implicit diffusion above is only the componentwise Laplacian); the
    // constant-nu molecular transpose is analytically zero for solenoidal U,
    // so only nuT enters, and the -2/3 k I normal stress stays absorbed in
    // the working pressure. The
    // freeze switch is the frozen-pressure tangent's reduced operator (the
    // source is lagged there, matching the solver's own Picard operator).
    if (!freezeTransposeStress_) {
        // wall faces carry ZERO transpose coefficient: the eddy viscosity
        // vanishes AT a no-slip wall even though the owner-cell value does
        // not, so owner extrapolation would overweight the wall flux.
        std::vector<double> zeroWall(mesh_.nCells(), 0.0);
        std::vector<double> tsrc = transposeStressSource(mesh_, f.nuT, f.U, component, &zeroWall);
        for (int ci = 0; ci < mesh_.nCells(); ++ci)
            sys.source[ci] += tsrc[ci];
    }

    // a-posteriori Reynolds-stress injection (explicit deferred-correction)
    if (bTarget6_)
        addInjectionSource(sys, f, component);

    // higher-order convection as a deferred correction on the upwind matrix
    if (settings_.momentumConvection != "upwind") {
        ScalarField uc(mesh_, "Uc");
        auto pick = [component](const Vec3 &v) {
            return component == 0 ? v.x : component == 1 ? v.y : v.z;
        };
        for (int ci = 0; ci < mesh_.nCells(); ++ci)
            uc[ci] = pick(f.U[ci]);
        for (int fi = mesh_.nInternalFaces(); fi < mesh_.nFaces(); ++fi)
            uc.bface(fi) = pick(f.U.bface(fi));
        addConvectionCorrection(sys, f, uc, settings_.momentumConvection);
    }

    // In SIMPLE, the momentum equation is usually under-relaxed to improve stability.
    // under-relaxation: numerical stabilization technique - prevents solution from changing too
    // drastically between iterations makes small adjustments to allow for smooth convergence

    // under-relaxation:  aP_new = aP / alphaU,  source += (1-alphaU)/alphaU * aP * phi_old
    double alphaU = settings_.alphaU;
    for (int ci = 0; ci < mesh_.nCells(); ++ci) {
        double phi_old;
        if (component == 0)
            phi_old = f.U[ci].x;
        else if (component == 1)
            phi_old = f.U[ci].y;
        else
            phi_old = f.U[ci].z;

        aPunrelaxed_[ci] =
            sys.diag[ci]; // store UNRELAXED diagonal for Rhie-Chow pressure Laplacian
        sys.source[ci] += (1.0 - alphaU) / alphaU * sys.diag[ci] * phi_old;
        sys.diag[ci] /= alphaU;
        aP[ci] = sys.diag[ci]; // store RELAXED diagonal (after /alphaU) for velocity correction
    }
}

// Higher-order convection by deferred correction. The implicit matrix stays
// first-order upwind; the difference between the higher-order face value and
// the upwind one, times the face flux, moves to the right-hand side, so it
// vanishes from the converged equations only where the schemes agree.
//   linearUpwind:        phi_f = phi_U + grad(phi)_U . (x_f - x_U)
//   limitedLinearUpwind: phi_f = phi_U + psi(r)/2 (phi_D - phi_U),
//                        r = 2 grad(phi)_U . (x_D - x_U) / (phi_D - phi_U) - 1,
//                        psi = van Albada (r^2 + r)/(r^2 + 1) for r > 0, else 0
// The Green-Gauss gradient is lagged at the current state. Positions across a
// periodic wrap face are taken in the upwind cell's frame. Boundary faces
// stay upwind (an inflow carries its boundary value).
void SIMPLESolver::addConvectionCorrection(LinearSystem &sys, const FlowFields &f,
                                           const ScalarField &phi,
                                           const std::string &scheme) const {
    if (scheme == "upwind")
        return;
    const bool limited = (scheme == "limitedLinearUpwind");
    if (!limited && scheme != "linearUpwind")
        throw std::invalid_argument("unknown convection scheme: " + scheme);
    VectorField grad = greenGaussGrad(phi);
    for (int fi = 0; fi < mesh_.nInternalFaces(); ++fi) {
        const Face &face = mesh_.face(fi);
        const double F = convectiveFlux(f, fi);
        if (F == 0.0)
            continue;
        const int o = face.owner, n = face.neighbor;
        const bool ownerUp = F > 0.0;
        const int up = ownerUp ? o : n, dn = ownerUp ? n : o;
        const Vec3 &xU = mesh_.cell(up).center;
        double phiF;
        if (!limited) {
            const Vec3 xf = ownerUp ? face.center : face.center - face.shift;
            phiF = phi[up] + grad[up].dot(xf - xU);
        } else {
            const Vec3 xD =
                ownerUp ? mesh_.cell(n).center + face.shift : mesh_.cell(o).center - face.shift;
            const double dphi = phi[dn] - phi[up];
            double psi = 0.0;
            if (std::abs(dphi) > 1e-300) {
                const double r = 2.0 * grad[up].dot(xD - xU) / dphi - 1.0;
                psi = r > 0.0 ? (r * r + r) / (r * r + 1.0) : 0.0;
            }
            phiF = phi[up] + 0.5 * psi * dphi;
        }
        const double corr = F * (phiF - phi[up]);
        sys.source[o] -= corr;
        sys.source[n] += corr;
    }
}

// A-posteriori Reynolds-stress injection source (explicit deferred-correction).
//
// Adds to the momentum source of `component` the body force
//   f_inj V = -[ sum_f (D_f . n_f)_i A_f ],   D = 2 k b_target + 2 nuT dev(S),
// i.e. the Green-Gauss divergence of the difference between the injected
// deviatoric Reynolds stress 2 k b_target and the Boussinesq-modeled deviatoric
// stress -2 nuT dev(S) the implicit nuEff diffusion stands for. k and S are the
// RUNNING fields, so the force vanishes identically when b_target equals the
// solver's own Boussinesq anisotropy b_B = -(nuT/k) dev(S), and the injected
// difference from the baseline solve is exactly -div(2 k (b_target - b_B)).
// Face values of D are distance-weight interpolated; boundary faces take the
// owner value (the wall rows carry k ~ 0, so their stress contribution is
// negligible there). nuT diffusion stays implicit as the stabilizer.
//
// On the x-component pass the target anisotropy is re-checked for realizability
// (barycentric margin of its eigenvalues), fulfilling the every-outer-iteration
// realizability assertion of the pre-registered injection scheme; violations are
// recorded in injDiag_ rather than silently projected, so a study can report
// them (the caller projects before setting the target).
void SIMPLESolver::addInjectionSource(LinearSystem &sys, const FlowFields &f, int component) {
    const int nc = mesh_.nCells();

    // The x-component pass recomputes the full Vec3 source from the current
    // fields, blends it into injSrcBlend_ at injRelax_, and re-asserts
    // realizability; the y and z passes reuse the stored blend. This relies on
    // the momentum components being assembled in order 0, 1, 2, which both
    // solve() and assembleResidual() do.
    if (component == 0) {
        // The vector is borrowed and may have changed since the setter or a
        // previous source-observer call, including in fixed-state diagnostics.
        setTargetAnisotropy(bTarget6_);
        // per-cell deferred-correction tensor D = 2 k b_target + 2 nuT dev(S),
        // stored as xx, yy, zz, xy, xz, yz
        if (!injectionWorkspace_)
            injectionWorkspace_ = std::make_unique<InjectionWorkspace>(mesh_);
        auto &scratch = *injectionWorkspace_;
        computeVelocityGradients(f.U, scratch.gradients, scratch.components);
        const auto &vg = scratch.gradients;
        auto &D = scratch.tensor;
        for (int ci = 0; ci < nc; ++ci) {
            const double S11 = vg.dudx[ci].x;
            const double S22 = vg.dvdx[ci].y;
            const double S33 = vg.dwdx[ci].z;
            const double S12 = 0.5 * (vg.dudx[ci].y + vg.dvdx[ci].x);
            const double S13 = 0.5 * (vg.dudx[ci].z + vg.dwdx[ci].x);
            const double S23 = 0.5 * (vg.dvdx[ci].z + vg.dwdx[ci].y);
            const double trS3 = (S11 + S22 + S33) / 3.0; // discrete div(U)/3 residue

            const double twoK = 2.0 * f.k[ci];
            const double twoNu = 2.0 * f.nuT[ci];
            const double *bt = bTarget6_->data() + 6 * ci;
            D[6 * ci + 0] = twoK * bt[0] + twoNu * (S11 - trS3);
            D[6 * ci + 1] = twoK * bt[1] + twoNu * (S22 - trS3);
            D[6 * ci + 2] = twoK * bt[2] + twoNu * (S33 - trS3);
            D[6 * ci + 3] = twoK * bt[3] + twoNu * S12;
            D[6 * ci + 4] = twoK * bt[4] + twoNu * S13;
            D[6 * ci + 5] = twoK * bt[5] + twoNu * S23;
        }

        // fresh force per cell: (f_inj V)_i = -sum_f (D_f . n_f)_i A_f
        auto &q = scratch.force;
        std::fill(q.begin(), q.end(), Vec3{});
        auto dDotN = [](const double *d, const Vec3 &n) {
            return Vec3(d[0] * n.x + d[3] * n.y + d[4] * n.z, d[3] * n.x + d[1] * n.y + d[5] * n.z,
                        d[4] * n.x + d[5] * n.y + d[2] * n.z);
        };
        const int nIF = mesh_.nInternalFaces();
        for (int fi = 0; fi < nIF; ++fi) {
            const Face &face = mesh_.face(fi);
            const int o = face.owner;
            const int n = face.neighbor;
            double Df[6];
            for (int c = 0; c < 6; ++c)
                Df[c] = face.weight * D[6 * o + c] + (1.0 - face.weight) * D[6 * n + c];
            const Vec3 flux = dDotN(Df, face.normal) * face.area;
            q[o] = q[o] - flux; // f_inj = -div(D): outflux lowers the owner
            q[n] = q[n] + flux; // normal points owner -> neighbor
        }
        // boundary faces: owner-value extrapolation
        for (int fi = nIF; fi < mesh_.nFaces(); ++fi) {
            const Face &face = mesh_.face(fi);
            q[face.owner] =
                q[face.owner] - dDotN(D.data() + 6 * face.owner, face.normal) * face.area;
        }

        // Scale the COMPLETE integrated force (including baseline cancellation)
        // before history blending. Do not change the default arithmetic path.
        if (injectionScale_ != 1.0)
            for (auto &force : q)
                force = force * injectionScale_;

        // blend (the first pass of a solve, or a fresh evaluation, has
        // injSrcBlend_ empty or injRelax_ = 1 and takes q as-is)
        if (injSrcBlend_.size() != static_cast<size_t>(nc) || injRelax_ >= 1.0) {
            injSrcBlend_ = q;
        } else {
            for (int ci = 0; ci < nc; ++ci)
                injSrcBlend_[ci] = injSrcBlend_[ci] * (1.0 - injRelax_) + q[ci] * injRelax_;
        }

        if (sourceObservationActive_)
            freshInjection_ = q;

        // realizability re-assertion, once per outer iteration
        injDiag_.active = true;
        injDiag_.checkedIters += 1;
        for (int ci = 0; ci < nc; ++ci) {
            const double violation = aniso::realizabilityViolation(bTarget6_->data() + 6 * ci);
            // tolerance above the trig eigensolver's conditioning floor at
            // degenerate eigenvalues (see AnisotropyTools::isRealizable): corner
            // targets from the eigenspace family sit exactly on the boundary
            if (violation > 1e-6) {
                injDiag_.allRealizable = false;
                injDiag_.maxViolation = std::max(injDiag_.maxViolation, violation);
            }
        }
    }

    for (int ci = 0; ci < nc; ++ci) {
        if (component == 0)
            sys.source[ci] += injSrcBlend_[ci].x;
        else if (component == 1)
            sys.source[ci] += injSrcBlend_[ci].y;
        else
            sys.source[ci] += injSrcBlend_[ci].z;
    }
}

// The constrained wall value participates in neighboring equations during
// the solve. Solving a wall PDE and overwriting its answer afterward does not
// solve the remaining transport equations.
void SIMPLESolver::omegaWallValues(const FlowFields &f, std::vector<double> &values) const {
    values.assign(mesh_.nCells(), std::numeric_limits<double>::quiet_NaN());
    for (int pi = 0; pi < mesh_.nPatches(); ++pi) {
        if (mesh_.patch(pi).type != "wall")
            continue;
        for (int fi : mesh_.patch(pi).faces) {
            const auto &face = mesh_.face(fi);
            const double y = std::max(face.delta, 1e-20);
            const double resolved = resolvedWallOmega(y);
            double value = resolved;
            if (settings_.useWallFunctions) {
                const double uTau =
                    std::sqrt(std::sqrt(sst_.coeffs.betaStar) * std::max(f.k[face.owner], 1e-30));
                const double log =
                    uTau / (std::sqrt(sst_.coeffs.betaStar) * settings_.vonKarman * y);
                value = std::sqrt(resolved * resolved + log * log);
            }
            values[face.owner] = value; // preserve the existing last-wall-face corner rule
        }
    }
}

void SIMPLESolver::constrainOmega(LinearSystem &sys, const std::vector<double> &values) const {
    // Eliminate each constrained column before replacing its row. Retain the
    // row's natural diagonal scale instead of introducing an arbitrary unit.
    for (int fi = 0; fi < mesh_.nInternalFaces(); ++fi) {
        const auto &face = mesh_.face(fi);
        const int o = face.owner, n = face.neighbor;
        if (std::isfinite(values[o])) {
            sys.source[n] -= sys.lower[fi] * values[o];
            sys.upper[fi] = sys.lower[fi] = 0;
        } else if (std::isfinite(values[n])) {
            sys.source[o] -= sys.upper[fi] * values[n];
            sys.upper[fi] = sys.lower[fi] = 0;
        }
    }
    for (int ci = 0; ci < mesh_.nCells(); ++ci)
        if (std::isfinite(values[ci]))
            sys.source[ci] = sys.diag[ci] * values[ci];
}

// Pressure correction equation assembly
// constructs a finite-volume linear system for the pressure correction equation in SIMPLE algorithm
// constructs Laplacian operator using inverse momentum diagonal and adding source terms from
// divergence of predicted velocity field Laplacian(p') = div(U*)  where coefficients use rAP = 1/aP
// per cell ensures that div(U*) = 0
//
// `predicted` carries the predicted face fluxes (Rhie-Chow included, see
// solve()). Without it this is the legacy assembly: interpolated fluxes,
// with the Rhie-Chow term only on outlet-free meshes and with the relaxed
// coefficient. The Laplacian coefficients are kept in pCoeff_ so the
// corrected face flux can be formed with exactly the operator inverted.
void SIMPLESolver::assemblePressureCorrection(LinearSystem &sys, const FlowFields &f,
                                              const std::vector<double> &aP, ScalarField &pPrime,
                                              const std::vector<double> *predicted) {
    sys.zero();
    pPrime.setUniform(0.0);
    int nIF = mesh_.nInternalFaces();
    massFluxScale_ = 0.0;
    pCoeff_.assign(mesh_.nFaces(), 0.0);

    bool hasOutlet = false;
    for (int pi = 0; pi < mesh_.nPatches(); ++pi)
        if (mesh_.patch(pi).type == "outlet" && !mesh_.patch(pi).faces.empty())
            hasOutlet = true;

    // legacy gate: the term only where the p' system is all-Neumann
    const bool rcActive = !predicted && legacyRhieChowActive();

    // cell pressure gradient for the Rhie-Chow face-flux dissipation below
    VectorField gradP(mesh_, "gradP");
    if (rcActive)
        gradP = greenGaussGrad(f.p);

    // internal faces
    for (int fi = 0; fi < nIF; ++fi) {
        const Face &face = mesh_.face(fi);
        int o = face.owner;
        int n = face.neighbor;
        double Sf = face.area;
        double delta = std::max(face.delta, 1e-20);

        // Pressure correction Laplacian: ∇·((V/aP) ∇p') = div(U*)
        // Face coefficient D_f = (V/aP)_f * A_f / delta
        // Interpolate (V/aP) from owner and neighbor cells
        double Vo = mesh_.cell(o).volume;
        double Vn = mesh_.cell(n).volume;
        double dP_o = (std::abs(aP[o]) > 1e-30) ? Vo / aP[o] : 0.0;
        double dP_n = (std::abs(aP[n]) > 1e-30) ? Vn / aP[n] : 0.0;
        double dP_f = face.weight * dP_o + (1.0 - face.weight) * dP_n;

        // pressure correction Laplacian coefficient
        double coeff = dP_f * Sf / delta;
        pCoeff_[fi] = coeff;

        sys.diag[o] += coeff;
        sys.diag[n] += coeff;
        sys.upper[fi] = -coeff;
        sys.lower[fi] = -coeff;

        double massFlux;
        if (predicted) {
            massFlux = (*predicted)[fi];
        } else {
            Vec3 Uf = f.U[o] * face.weight + f.U[n] * (1.0 - face.weight);
            massFlux = (Uf.x * face.normal.x + Uf.y * face.normal.y + Uf.z * face.normal.z) * Sf;
            if (rcActive) {
                Vec3 ehat = face.d / std::max(face.delta, 1e-30);
                Vec3 gbar = gradP[o] * face.weight + gradP[n] * (1.0 - face.weight);
                massFlux += -dP_f * Sf * ((f.p[n] - f.p[o]) / delta - gbar.dot(ehat));
            }
        }

        // Source = -div(U*)*V: FV Laplacian sum(coeff*(p'_P - p'_N)) = -∇²p'*V,
        // so the Poisson equation rAP*∇²p' = div(U*) maps to source = -div(U*)*V.
        sys.source[o] -= massFlux;
        sys.source[n] += massFlux;
        massFluxScale_ += std::abs(massFlux);
    }

    // boundary faces: mass flux source + Dirichlet diffusion for outlet (p'=0 there)
    // Wall and inlet use zero-gradient p' (Neumann) — no diffusion term needed.
    // Outlet uses Dirichlet p'=0 — the Laplacian coefficient must be added to the diagonal
    // so the pressure correction at the outlet-adjacent cell is actually constrained.
    // Without this, the p' Poisson system has no Dirichlet row and is singular.
    for (int pi = 0; pi < mesh_.nPatches(); ++pi) {
        const Patch &pat = mesh_.patch(pi);
        bool isOutlet = (pat.type == "outlet");
        for (FaceID fi : pat.faces) {
            const Face &face = mesh_.face(fi);
            int o = face.owner;
            double Sf = face.area;
            double delta = std::max(face.delta, 1e-20);

            // boundary mass flux source (same sign convention as internal)
            double massFlux;
            if (predicted) {
                massFlux = (*predicted)[fi];
            } else {
                Vec3 Ub = f.U.bface(fi);
                massFlux =
                    (Ub.x * face.normal.x + Ub.y * face.normal.y + Ub.z * face.normal.z) * Sf;
            }
            sys.source[o] -= massFlux;
            massFluxScale_ += std::abs(massFlux);

            // outlet Dirichlet: p'=0 → add diffusion coeff, source += coeff*0 = 0
            if (isOutlet) {
                double Vo = mesh_.cell(o).volume;
                double dP_o = (std::abs(aP[o]) > 1e-30) ? Vo / aP[o] : 0.0;
                double coeff = dP_o * Sf / delta;
                pCoeff_[fi] = coeff;
                sys.diag[o] += coeff;
                // sys.source[o] += coeff * 0.0 — omitted (p'_outlet = 0)
            }
        }
    }

    // Pressure reference for domains with NO outlet (streamwise-periodic
    // channels bounded by walls and wrap faces): the all-Neumann p' Poisson
    // system is singular up to a constant, so pin cell 0 by doubling its
    // diagonal, a Dirichlet-strength link to p' = 0 at the natural scale of
    // its own row. The physical pressure level is arbitrary in such domains.
    if (!hasOutlet && mesh_.nCells() > 0)
        sys.diag[0] += std::abs(sys.diag[0]) > 1e-30 ? sys.diag[0] : 1.0;
}

bool SIMPLESolver::legacyRhieChowActive() const {
    for (int pi = 0; pi < mesh_.nPatches(); ++pi)
        if (mesh_.patch(pi).type == "outlet" && !mesh_.patch(pi).faces.empty())
            return false;
    return true;
}

// U_f . S_f (owner-outward) from the cell velocities: interpolated on
// internal faces, the boundary value on boundary faces, in the arithmetic
// the solver always used.
double SIMPLESolver::interpolatedFlux(const FlowFields &f, int fi) const {
    const Face &face = mesh_.face(fi);
    if (fi < mesh_.nInternalFaces()) {
        Vec3 Uf = f.U[face.owner] * face.weight + f.U[face.neighbor] * (1.0 - face.weight);
        return (Uf.x * face.normal.x + Uf.y * face.normal.y + Uf.z * face.normal.z) * face.area;
    }
    Vec3 Ub = f.U.bface(fi);
    return (Ub.x * face.normal.x + Ub.y * face.normal.y + Ub.z * face.normal.z) * face.area;
}

// The flux the assemblies convect with: the stored fluxes when convFlux_ is
// set, the interpolated flux otherwise (legacy).
double SIMPLESolver::convectiveFlux(const FlowFields &f, int fi) const {
    return convFlux_ ? (*convFlux_)[fi] : interpolatedFlux(f, fi);
}

void SIMPLESolver::interpolatedFluxes(const FlowFields &f, std::vector<double> &phi) const {
    phi.resize(mesh_.nFaces());
    for (int fi = 0; fi < mesh_.nFaces(); ++fi)
        phi[fi] = interpolatedFlux(f, fi);
}

// Rhie-Chow face-flux dissipation
//   phi_f -= D_f |S| [ (p_N - p_O)/delta - grad(p)bar_f . e ],  D_f = (V/aP)_f
// It vanishes on smooth pressure ((p_N - p_O)/delta equals the interpolated
// gradient along e to truncation order) and penalises the odd-even
// component that the compact Laplacian cannot see through centrally
// interpolated fluxes.
void SIMPLESolver::addRhieChow(const FlowFields &f, const std::vector<double> &aP,
                               std::vector<double> &phi) const {
    VectorField gradP = greenGaussGrad(f.p);
    for (int fi = 0; fi < mesh_.nInternalFaces(); ++fi) {
        const Face &face = mesh_.face(fi);
        int o = face.owner, n = face.neighbor;
        double delta = std::max(face.delta, 1e-20);
        double dP_o = (std::abs(aP[o]) > 1e-30) ? mesh_.cell(o).volume / aP[o] : 0.0;
        double dP_n = (std::abs(aP[n]) > 1e-30) ? mesh_.cell(n).volume / aP[n] : 0.0;
        double dP_f = face.weight * dP_o + (1.0 - face.weight) * dP_n;
        Vec3 ehat = face.d / std::max(face.delta, 1e-30);
        Vec3 gbar = gradP[o] * face.weight + gradP[n] * (1.0 - face.weight);
        phi[fi] += -dP_f * face.area * ((f.p[n] - f.p[o]) / delta - gbar.dot(ehat));
    }
}

// Diagnostic flux reconstruction: exactly two passes on the unrelaxed
// momentum diagonal. This finite algorithm need not satisfy the steady
// flux-consistency equation; it is not a flux elimination for solved tangents.
void SIMPLESolver::steadyFluxes(const FlowFields &f, std::vector<double> &phi) {
    if (!settings_.legacyRhieChowGate && !settings_.useWallFunctions &&
        settings_.momentumConvection != "limitedLinearUpwind" &&
        settings_.turbulenceConvection != "limitedLinearUpwind") {
        phi = diagnostic::reconstructFlux(mesh_, f, nu_, f.nuT.data());
        return;
    }
    interpolatedFluxes(f, phi);
    if (settings_.legacyRhieChowGate)
        return;
    const std::vector<double> *saved = convFlux_;
    LinearSystem sys = makeSystem(mesh_);
    std::vector<double> aPtmp(mesh_.nCells(), 0.0), pass;
    for (int k = 0; k < 2; ++k) {
        pass = phi;
        convFlux_ = (k == 0) ? nullptr : &pass;
        assembleMomentum(sys, f, 0, aPtmp); // sets aPunrelaxed_
        interpolatedFluxes(f, phi);
        addRhieChow(f, aPunrelaxed_, phi);
    }
    convFlux_ = saved;
}

// Correction step of the SIMPLE Algorithm

// Velocity correction:  U = U* - gradP' * V / aP
// adjusts predicted velocity using pressure-correction gradient and momentum diagonal coefficients
void SIMPLESolver::correctVelocity(FlowFields &f, const ScalarField &pPrime,
                                   const std::vector<double> &aP) {
    VectorField gradPp = greenGaussGrad(pPrime);
    for (int ci = 0; ci < mesh_.nCells(); ++ci) {
        double rAP = (std::abs(aP[ci]) > 1e-30) ? 1.0 / aP[ci] : 0.0;
        double vol = mesh_.cell(ci).volume;
        f.U[ci].x -= rAP * gradPp[ci].x * vol;
        f.U[ci].y -= rAP * gradPp[ci].y * vol;
        f.U[ci].z -= rAP * gradPp[ci].z * vol;
    }
    applyVelocityBC(f.U, mesh_,
                    bcs_); // boundary values must be re-enforced after modifying interior cells
}

// Pressure correction:  p = p + alphaP * p'
// updates pressure field with an under-relaxed pressure correction so that corrected U satisfies
// incompressible continuity equation (div U = 0)
void SIMPLESolver::correctPressure(FlowFields &f, const ScalarField &pPrime) {
    for (int ci = 0; ci < mesh_.nCells(); ++ci) {
        f.p[ci] +=
            settings_.alphaP *
            pPrime[ci]; // SIMPLE pressure update with under-relaxation to prevent instability
    }
    applyPressureBC(f.p, mesh_, bcs_);
}

// k-equation assembly
// converts continuous k-transport equation into a finite-volume matrix form (A_k * k = b_k)
//      discretizing convection and diffusion terms across faces, applying boundary conditions,
//      adding production and destruction source terms, and applying under-relaxation for stability
//      before the equation is solved
void SIMPLESolver::assembleKEquation(LinearSystem &sys, const FlowFields &f) {
    sys.zero();
    int nIF = mesh_.nInternalFaces();

    // loop over internal faces
    for (int fi = 0; fi < nIF; ++fi) {
        const Face &face = mesh_.face(fi);
        int o = face.owner;
        int n = face.neighbor;
        double Sf = face.area;
        double delta = std::max(face.delta, 1e-20);

        // diffusion term
        double F1_f = face.weight * f.F1[o] + (1.0 - face.weight) * f.F1[n];
        double sk = sst_.coeffs.sigma_k(F1_f);
        double nuEff_o = nu_ + sk * f.nuT[o]; // effective diffusivity: nu + sigma_k * nuT
        double nuEff_n = nu_ + sk * f.nuT[n];
        double nuEff_f =
            face.weight * nuEff_o + (1.0 - face.weight) * nuEff_n; // interpolate to face
        double Df = nuEff_f * Sf / delta;

        // convection term
        double mFlux = convectiveFlux(f, fi); // mass flux
        double cPos = std::max(mFlux, 0.0);   // upwind discretization
        double cNeg = std::max(-mFlux, 0.0);

        // add matrix coefficients
        sys.diag[o] += Df + cPos;
        sys.diag[n] += Df + cNeg;
        sys.upper[fi] = -(Df + cNeg);
        sys.lower[fi] = -(Df + cPos);
    }

    // boundary faces
    for (int fi = nIF; fi < mesh_.nFaces(); ++fi) {
        const Face &face = mesh_.face(fi);
        int o = face.owner;
        double Sf = face.area;
        double delta = std::max(face.delta, 1e-20);
        double sk = sst_.coeffs.sigma_k(f.F1[o]);
        double nuEff = nu_ + sk * f.nuT[o];
        double Db = nuEff * Sf / delta; // diffusion coefficient

        double kb = f.k.bface(fi);            // boundary k value
        double mFlux = convectiveFlux(f, fi); // mass flux

        // outflow boundary
        if (mFlux >= 0) {
            sys.diag[o] += mFlux + Db;
            sys.source[o] += Db * kb;
        }
        // inflow boundary
        else {
            sys.diag[o] += Db;
            sys.source[o] += (Db - mFlux) * kb;
        }
    }

    // source terms: Pk (explicit), betaStar*omega*k (linearised destruction)
    for (int ci = 0; ci < mesh_.nCells(); ++ci) {
        double vol = mesh_.cell(ci).volume;
        // production term (explicit)
        sys.source[ci] += f.Pk[ci] * vol;
        // destruction term (linearised): betaStar * omega * k → diag += betaStar*omega*V
        double destruction = sst_.coeffs.betaStar * std::max(f.omega[ci], 1e-20);
        sys.diag[ci] += destruction * vol;
    }

    addConvectionCorrection(sys, f, f.k, settings_.turbulenceConvection);

    // under-relaxation
    double alphaK = settings_.alphaK;
    for (int ci = 0; ci < mesh_.nCells(); ++ci) {
        sys.source[ci] += (1.0 - alphaK) / alphaK * sys.diag[ci] * f.k[ci];
        sys.diag[ci] /= alphaK;
    }
}

// omega equation assembly
// converts continous SST omega transport equation into finite-volume matrix form (A_ω * ω = b_ω)
//      discretize convection and diffusion fluxes across faces, applying boundary conditions,
//      adding turbulence production, destruction, and cross-diffusion source terms, and
//      applying under-relaxation before solving for the updated ω field
void SIMPLESolver::assembleOmegaEquation(LinearSystem &sys, const FlowFields &f,
                                         const ScalarField &Smag) {
    if (productionPolicyActive())
        setTargetAnisotropy(bTarget6_);
    sys.zero();
    int nIF = mesh_.nInternalFaces();
    std::vector<double> balance;
    if (!omegaSingularity_.empty())
        balance.assign(mesh_.nCells(), 0);
    // Smag is passed in (frozen at the pre-correction U from computeFields).
    // Do NOT recompute from f.U here: the post-correction velocity may carry
    // pressure-correction oscillations that would inflate S^2 and blow up omega.

    // internal face loop
    for (int fi = 0; fi < nIF; ++fi) {
        const Face &face = mesh_.face(fi);
        int o = face.owner;
        int n = face.neighbor;
        double Sf = face.area;
        double delta = std::max(face.delta, 1e-20);

        // diffusion term div((nu + sw * nuT) * grad(omega))
        double F1_f = face.weight * f.F1[o] + (1.0 - face.weight) * f.F1[n];
        double sw = sst_.coeffs.sigma_w(F1_f);
        double nuEff_o = nu_ + sw * f.nuT[o]; // effective diffusivity: nu + sigma_w * nuT
        double nuEff_n = nu_ + sw * f.nuT[n];
        double nuEff_f = face.weight * nuEff_o + (1.0 - face.weight) * nuEff_n;
        double Df = nuEff_f * Sf / delta; // face diffusion coefficient

        // convection term (U dot grad(omega))
        double mFlux = convectiveFlux(f, fi); // mass flux

        // upwind discretization (determines flow direction)
        double cPos = std::max(mFlux, 0.0);
        double cNeg = std::max(-mFlux, 0.0);

        // matrix coefficients:
        // builds convection-diffusion operator
        sys.diag[o] += Df + cPos;
        sys.diag[n] += Df + cNeg;
        sys.upper[fi] = -(Df + cNeg);
        sys.lower[fi] = -(Df + cPos);

        if (!omegaSingularity_.empty()) {
            // With s=6nu/beta1*(1/y^2+1/(H-y)^2), write
            // div(D grad omega) = div(D grad(omega-s)+(D-nu)grad s)
            //                    + nu*laplacian(s).
            // Differencing the smooth remainder avoids the O(1) relative
            // truncation error in near-wall singular diffusion. Retaining the
            // analytic molecular Laplacian balances midpoint destruction.
            const double amplitude = 6 * nu_ / sst_.coeffs.beta1;
            const double difference = (omegaSingularity_[n] - omegaSingularity_[o]) / delta;
            const double correction =
                amplitude * Sf *
                ((nuEff_f - nu_) * omegaNormalGradient_[fi] - nuEff_f * difference);
            balance[o] += correction;
            balance[n] -= correction;
        }
    }

    // boundary faces
    // computes diffusion coefficient, boundary omega value, and mass flux
    for (int fi = nIF; fi < mesh_.nFaces(); ++fi) {
        const Face &face = mesh_.face(fi);
        int o = face.owner;
        double Sf = face.area;
        double delta = std::max(face.delta, 1e-20);
        double sw = sst_.coeffs.sigma_w(f.F1[o]);
        double nuEff = nu_ + sw * f.nuT[o];
        double Db = nuEff * Sf / delta;

        double wb = f.omega.bface(fi);
        double mFlux = convectiveFlux(f, fi);

        // outflow boundary
        if (mFlux >= 0) {
            sys.diag[o] += mFlux + Db;
            sys.source[o] += Db * wb;
        }
        // inflow boundary
        else {
            sys.diag[o] += Db;
            sys.source[o] += (Db - mFlux) * wb;
        }
    }

    // source terms
    for (int ci = 0; ci < mesh_.nCells(); ++ci) {
        double vol = mesh_.cell(ci).volume;
        double F1 = f.F1[ci];
        double omC = std::max(f.omega[ci], 1e-20);

        // production: alpha * Pk_limited/nuT = alpha * min(S^2, 10*betaStar*k*omega/nuT)
        // (SST-2003 corrected form; the 2003 paper's alpha*S^2 is a documented misprint,
        // see the NASA TMR SST page). min(S^2, lim) <= S^2 pointwise, so this term is
        // bounded above by the S^2 form: reducing nuT cannot amplify it, and destruction
        // (beta*omega^2, implicit below) still grows quadratically, so the equation stays
        // self-limiting. The two forms coincide wherever the k-production limiter is
        // inactive (equilibrium attached flows).
        double alphaB = sst_.coeffs.alpha(F1);
        double S = Smag[ci];
        double production = sst_.productionOmega(f.nuT[ci], S, f.k[ci], f.omega[ci]);
        if (productionPolicyActive()) {
            production = productionPolicy_->omegaProduction(
                {mesh_, f, sst_.coeffs, Smag, *bTarget6_, nu_}, ci);
            if (!std::isfinite(production))
                throw std::runtime_error("production policy returned invalid omega production");
        }
        sys.source[ci] += alphaB * production * vol;

        // destruction term (linearised) beta*omega^2 → diag += beta*omega*V
        double betaB = sst_.coeffs.beta(F1);
        sys.diag[ci] += betaB * omC * vol;

        // cross-diffusion (explicit): (1-F1)*CDkw, UNCLIPPED per SST-2003 (only the CDkw
        // inside the F1 argument is clipped; the source term itself may be negative)
        sys.source[ci] += (1.0 - F1) * f.CDkw[ci] * vol;
        if (!balance.empty()) {
            const double correction =
                balance[ci] + nu_ * (6 * nu_ / sst_.coeffs.beta1) * omegaLaplacian_[ci] * vol;
            // Treat a negative correction as a Picard sink so this term
            // does not add a negative explicit source. No source is clipped:
            // evaluated at the supplied positive state, b-A*omega still
            // contains exactly the signed correction.
            if (correction < 0)
                sys.diag[ci] -= correction / omC;
            else
                sys.source[ci] += correction;
        }
    }

    addConvectionCorrection(sys, f, f.omega, settings_.turbulenceConvection);

    // under-relaxation
    double alphaW = settings_.alphaOmega;
    for (int ci = 0; ci < mesh_.nCells(); ++ci) {
        sys.source[ci] += (1.0 - alphaW) / alphaW * sys.diag[ci] * f.omega[ci];
        sys.diag[ci] /= alphaW;
    }
}

// Main SIMPLE loop
// SIMPLE algorithm runs until flow solution converges or diverges
void SIMPLESolver::reportTimers() const {
    if (!settings_.verbose)
        return;
    double amgSetup = 0.0, amgIter = 0.0;
    if (const auto *amg = dynamic_cast<const AMGSolver *>(pSolver_.get())) {
        amgSetup = amg->setupSeconds;
        amgIter = amg->iterateSeconds;
    }
    std::cout << "  timers: momentum=" << momSeconds_ << "s  p_assemble=" << pAssembleSeconds_
              << "s  p_solve=" << pSolveSeconds_ << "s";
    if (amgSetup > 0.0 || amgIter > 0.0)
        std::cout << " (amg setup=" << amgSetup << "s iterate=" << amgIter << "s)";
    std::cout << "  turbulence=" << turbSeconds_ << "s\n";
}

ConvergenceHistory SIMPLESolver::solve(FlowFields &f) {
    Workspace workspace(mesh_);
    return solve(f, workspace);
}

ConvergenceHistory SIMPLESolver::solve(FlowFields &f, Workspace &workspace) {
    if (workspace.mesh != &mesh_)
        throw std::invalid_argument("solve workspace must match mesh");
    mesh_.requireStamp(workspace.stamp);
    if (workspace.active)
        throw std::logic_error("solve workspace is already active");
    struct WorkspaceReset {
        Workspace &w;
        ~WorkspaceReset() { w.active = false; }
    } reset{workspace};
    workspace.active = true;
    workspace.pressureCorrection.setUniform(0.0);
    workspace.strain.setUniform(0.0);
    workspace.previousObservations.clear();

    mesh_.requireStamp(meshStamp_); // input: FlowField object / flow variables
    simple_admission::fields(mesh_, f);
    // Callbacks may request isolated diagnostics, but cannot recursively solve.
    if (sourceObservationActive_)
        throw std::logic_error("recursive solve from source observer is unsupported");
    setTargetAnisotropy(bTarget6_); // revalidate borrowed storage before use
    sourceObservationActive_ = static_cast<bool>(sourceObserver_);
    struct ObservationReset {
        bool &active;
        ~ObservationReset() { active = false; }
    } observationReset{sourceObservationActive_};
    if (sourceObservationActive_)
        freshInjection_.assign(mesh_.nCells(), Vec3{});
    momSeconds_ = pAssembleSeconds_ = pSolveSeconds_ = turbSeconds_ = 0.0;
    // the AMG split shares the per-solve window (a reused solver object
    // would otherwise print lifetime-accumulated setup/iterate times against
    // per-solve phase times, an incomparable mix)
    if (auto *amg = dynamic_cast<AMGSolver *>(pSolver_.get()))
        amg->setupSeconds = amg->iterateSeconds = 0.0;
    ConvergenceHistory hist;
    // Reynolds-stress injection: fresh blend state and diagnostics per solve;
    // the explicit source is under-relaxed inside the outer loop (see
    // addInjectionSource)
    injSrcBlend_.clear();
    injDiag_ = InjectionDiagnostics{};
    injRelax_ = settings_.alphaInjection;
    // Create working linear systems
    auto &momSys = workspace.momentum;
    auto &pSys = workspace.pressure;
    auto &kSys = workspace.k;
    auto &omSys = workspace.omega;
    auto &pPrime = workspace.pressureCorrection;

    // Smag = Strain-rate magnitude
    // Frozen strain-rate field: computed from the pre-correction U (same U computeFields uses)
    // Passed to assembleOmegaEquation so both k and omega production see the same velocity
    // gradients, preventing omega blow-up.
    auto &SmagFrozen = workspace.strain;

    // Last-computed turbulence field-change norms, CARRIED ACROSS iterations:
    // on non-update iterations (turb_update_interval > 1) the convergence
    // check must see the most recent turbulence state, never a fresh zero
    // (a zero would let the solve declare convergence between turbulence
    // updates while k/omega are still moving). Initialised to 1 so nothing
    // can claim turbulence convergence before the first update.
    double lastKChange = 1.0, lastOmChange = 1.0;

    // Default convergence criterion (SolverSettings::legacyConvergence):
    // contraction estimates for U, k, omega and the monitored QoIs, and the
    // state-scaled residuals, all judged against convergenceTol.
    const int nc = mesh_.nCells();
    const double tol = settings_.convergenceTol;
    const double negligible = 0.01 * tol;
    ContractionEstimate estU(settings_.convergenceWindow), estK(settings_.convergenceWindow),
        estOm(settings_.convergenceWindow), estQ(settings_.convergenceWindow);
    auto &rbuf = workspace.residual;
    rbuf.resize(nc);
    auto &qPrev = workspace.previousObservations;
    auto &Uprev = workspace.oldVelocity;
    Uprev.resize(nc);
    // These solves are sequential. Refill the shared values from each field,
    // except pressure correction, whose initial guess must be zero each time.
    // Keep the x diagonal and old turbulence fields separate until their later
    // correction/change-norm consumers finish. Turbulence storage is lazy.
    auto &solveValues = workspace.solveValues;
    solveValues.resize(nc);
    auto &aPstore = workspace.diagonal;
    aPstore.resize(nc);
    auto &kOld = workspace.oldK;
    auto &omOld = workspace.oldOmega;
    double worst = std::numeric_limits<double>::infinity();
    double pressureFluxReference = 0.0;
    auto l1 = [](const std::vector<double> &v) {
        double s = 0.0;
        for (double x : v)
            s += std::abs(x);
        return s;
    };

    // Stored Rhie-Chow face fluxes (SolverSettings::legacyRhieChowGate).
    // Each iteration predicts phi* = U*_f.S - D_f[Rhie-Chow] + (1 - alphaU)
    // (phi_old - U_f,old.S) with the relaxed D_f of the pressure equation;
    // at a fixed point the last term turns D_f into the UNRELAXED coefficient,
    // so the converged solution does not depend on alphaU (Majumdar 1988).
    // The pressure correction then makes phi divergence-free with exactly
    // the coefficients it inverted, and U, k, omega are convected with it.
    const bool storedFlux = !settings_.legacyRhieChowGate;
    interpolatedFluxes(f, phi_);
    auto &phiBarOld = workspace.interpolatedFlux;
    auto &phiPred = workspace.predictedFlux;
    struct ConvFluxReset { // no assembly outside solve() sees phi_
        SIMPLESolver *s;
        ~ConvFluxReset() { s->convFlux_ = nullptr; }
    } convFluxReset{this};

    // SIMPLE Iteration loop
    for (int iter = 0; iter < settings_.maxIterations; ++iter) {
        for (int ci = 0; ci < nc; ++ci)
            Uprev[ci] = f.U[ci];
        double momResid = 0.0, momScale = 0.0;
        if (storedFlux) {
            interpolatedFluxes(f, phiBarOld); // U_f,old . S for the relaxation term
            convFlux_ = &phi_;
        } else {
            convFlux_ = nullptr;
        }
        // 1. Update SST fields (at turbUpdateInterval cadence after turbStartIter)
        bool turbActive = (iter >= settings_.turbStartIter);
        bool turbUpdate =
            turbActive && ((iter - settings_.turbStartIter) % settings_.turbUpdateInterval == 0);
        if (turbUpdate) {
            // Retain the strain from this same pre-correction U and boundary
            // state. The omega equation uses it after pressure correction,
            // without a second velocity-gradient calculation.
            sst_.computeFields(mesh_, f.k, f.omega, f.U, nu_, f.nuT, f.F1, f.F2, f.Pk, f.CDkw,
                               SmagFrozen, workspace.closure);

            // nuT floor, STARTUP-ONLY: guards against early k-omega collapse
            // (cells where the Bradshaw limiter drives nuT toward zero before
            // the turbulence fields establish). SST requires nuT -> 0 at a
            // resolved no-slip wall, so the floor releases after the startup
            // window and the converged state is floor-free; a permanent floor
            // biases the near-wall diffusion and the wall stress (~+10 percent
            // where it binds at y+ ~ 1). Warm restarts carry
            // f.turbEstablished = true and never re-engage the floor.
            if (!f.turbEstablished && iter >= settings_.turbStartIter + settings_.nuTFloorIters)
                f.turbEstablished = true;
            const double nuTMin = f.turbEstablished ? 0.0 : 0.1 * nu_;
            for (int ci = 0; ci < mesh_.nCells(); ++ci)
                f.nuT[ci] = std::max(f.nuT[ci], nuTMin);
            applyProductionPolicy(f, SmagFrozen);
        }

        // 2. Assemble + solve momentum x, y, z
        SolverResult resUx, resUy, resUz;
        const auto tMom0 = std::chrono::steady_clock::now();
        {
            assembleMomentum(momSys, f, 0, aP_);
            if (sourceObserver_) {
                // No injection means an explicit zero source snapshot. For a
                // legacy solve the current interpolated flux is the convector.
                std::vector<double> legacyFlux;
                if (!convFlux_)
                    interpolatedFluxes(f, legacyFlux);
                sourceObserver_({iter, f, convFlux_ ? *convFlux_ : legacyFlux, freshInjection_,
                                 bTarget6_ ? injSrcBlend_ : freshInjection_});
            }
            for (int ci = 0; ci < mesh_.nCells(); ++ci)
                solveValues[ci] = f.U[ci].x;
            // residual at the current state (relaxation cancels at phi_old),
            // scaled by the momentum magnitude sum_i a_P,i |U_i| of that state
            momSys.residual(solveValues, rbuf);
            momResid = l1(rbuf);
            for (int ci = 0; ci < nc; ++ci) {
                // Preserve the retained diagnostic's separately rounded product
                // and sum. A compound expression can contract to an FMA after
                // unrelated inlining changes.
                const double contribution = aPunrelaxed_[ci] * f.U[ci].norm();
                momScale += contribution;
            }
            resUx = mSolver_->solve(momSys, solveValues, settings_.innerIterations,
                                    settings_.innerTolerance);
            for (int ci = 0; ci < mesh_.nCells(); ++ci)
                f.U[ci].x = solveValues[ci];
        }
        // store aP from Ux for corrections (diagonal dominance is similar for all components)
        aPstore = aP_; // relaxed — for velocity correction
        {
            assembleMomentum(momSys, f, 1, aP_);
            for (int ci = 0; ci < mesh_.nCells(); ++ci)
                solveValues[ci] = f.U[ci].y;
            momSys.residual(solveValues, rbuf);
            momResid = std::max(momResid, l1(rbuf));
            resUy = mSolver_->solve(momSys, solveValues, settings_.innerIterations,
                                    settings_.innerTolerance);
            for (int ci = 0; ci < mesh_.nCells(); ++ci)
                f.U[ci].y = solveValues[ci];
        }
        {
            assembleMomentum(momSys, f, 2, aP_);
            for (int ci = 0; ci < mesh_.nCells(); ++ci)
                solveValues[ci] = f.U[ci].z;
            momSys.residual(solveValues, rbuf);
            momResid = std::max(momResid, l1(rbuf));
            resUz = mSolver_->solve(momSys, solveValues, settings_.innerIterations,
                                    settings_.innerTolerance);
            for (int ci = 0; ci < mesh_.nCells(); ++ci)
                f.U[ci].z = solveValues[ci];
        }
        momSeconds_ +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - tMom0).count();
        applyVelocityBC(f.U, mesh_, bcs_);

        // 3. Assemble + solve pressure correction
        // Use RELAXED aP (= aP_raw / alphaU) for both pressure Laplacian and velocity correction
        // This is necessary for consistency in SIMPLE algorithm: the momentum equation is solved
        // with relaxed diagonal, and the correction step must use the same coefficients (so
        // div(U_corrected) = 0). Note: U_corrected = U* - U' = U* - r_AP * ∇p' (r_AP = V/a_p)
        //  - when you solve the momentum eq, you get provisional velocity U* which does not satisfy
        //  continuity
        //  - SIMPLE introduces a correction U = U* + U' where U' is velocity correction derived
        //  from pressure correction or U_corrected
        const auto tP0 = std::chrono::steady_clock::now();
        if (storedFlux) {
            interpolatedFluxes(f, phiPred); // U*_f . S, boundary values
            addRhieChow(f, aPstore, phiPred);
            const double relaxCorr = 1.0 - settings_.alphaU;
            for (int fi = 0; fi < mesh_.nInternalFaces(); ++fi)
                phiPred[fi] += relaxCorr * (phi_[fi] - phiBarOld[fi]);
            assemblePressureCorrection(pSys, f, aPstore, pPrime, &phiPred);
        } else {
            assemblePressureCorrection(pSys, f, aPstore, pPrime);
        }
        pAssembleSeconds_ +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - tP0).count();
        // net mass imbalance of the predicted fluxes, relative to the flux throughput
        const double contScaled = l1(pSys.source) / std::max(massFluxScale_, 1e-300);
        const auto tPs0 = std::chrono::steady_clock::now();
        std::fill(solveValues.begin(), solveValues.end(), 0.0);
        SolverResult resP =
            pSolver_->solve(pSys, solveValues, settings_.innerIterations, settings_.innerTolerance);
        pSolveSeconds_ +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - tPs0).count();
        for (int ci = 0; ci < mesh_.nCells(); ++ci)
            pPrime[ci] = solveValues[ci];

        // Gradient of pressure correction p' was being computed using incorrect boundary face
        // values Apply p' boundary conditions before computing grad(p') in correctVelocity
        //  - At wall/inlet, grad(p') = 0, but not necessarily p' itself
        //      - to enforce grad(p') = 0, we set boundary face equal to value of adjacent interior
        //      cell
        //  - At outlet, BC is p' = 0 (Dirichlet) - bface stays at 0 (default)
        // Without this, greenGaussGrad sees bface=0 on all faces (initialization default)
        //  - this would give wrong gradients at boundary-adjacent cells and causing velocity to
        //  grow without bound
        for (int pi = 0; pi < mesh_.nPatches(); ++pi) {
            const Patch &pat = mesh_.patch(pi);
            if (pat.type == "outlet")
                continue; // keep Dirichlet p'=0
            for (FaceID fi : pat.faces)
                pPrime.bface(fi) = pPrime[mesh_.face(fi).owner];
        }

        // 4. Correct velocity and pressure (same relaxed aP), and the face
        //    fluxes with exactly the Laplacian coefficients the p' solve
        //    inverted, so phi_ is divergence-free to the linear tolerance
        //    (outlet faces: p' = 0 on the face, so phi_b = phi*_b + c_b p'_o)
        correctVelocity(f, pPrime, aPstore);
        correctPressure(f, pPrime);
        if (storedFlux) {
            for (int fi = 0; fi < mesh_.nFaces(); ++fi) {
                const Face &face = mesh_.face(fi);
                phi_[fi] =
                    face.isBoundary()
                        ? phiPred[fi] + pCoeff_[fi] * pPrime[face.owner]
                        : phiPred[fi] - pCoeff_[fi] * (pPrime[face.neighbor] - pPrime[face.owner]);
            }
        } else {
            interpolatedFluxes(f, phi_);
        }

        // 5. Assembly + solve Turbulence equations (k and omega) (on turbulence update iterations)
        SolverResult resK = {}, resOm = {};
        // Field-change norms for turbulence convergence tracking.
        // Track field convergence independently of the inner relative residual.
        // In legacy reproduction mode the solve-then-pin map can also retain a
        // transport imbalance; it is not eligible for steady certification.
        double omChangeNorm = 0.0, kChangeNorm = 0.0;
        if (turbUpdate) {
            // Save pre-solve fields for convergence metric
            kOld = f.k.data();
            omOld = f.omega.data();

            // k equation
            const auto tT0 = std::chrono::steady_clock::now();
            assembleKEquation(kSys, f);
            solveValues = f.k.data();
            resK = tSolver_->solve(kSys, solveValues, settings_.innerIterations,
                                   settings_.innerTolerance);
            for (int ci = 0; ci < mesh_.nCells(); ++ci)
                f.k[ci] = solveValues[ci];
            f.k.clamp(settings_.kMin, settings_.kMax);
            applyKBC(f.k, mesh_, bcs_);

            // omega equation (uses frozen Smag from pre-correction U)
            assembleOmegaEquation(omSys, f, SmagFrozen);
            omegaWallValues(f, workspace.wallOmega);
            if (!settings_.legacyOmegaWallPinning)
                constrainOmega(omSys, workspace.wallOmega);
            solveValues = f.omega.data();
            if (!settings_.legacyOmegaWallPinning)
                for (int ci = 0; ci < nc; ++ci)
                    if (std::isfinite(workspace.wallOmega[ci]))
                        solveValues[ci] = workspace.wallOmega[ci];
            resOm = tSolver_->solve(omSys, solveValues, settings_.innerIterations,
                                    settings_.innerTolerance);

            for (int ci = 0; ci < mesh_.nCells(); ++ci)
                f.omega[ci] = solveValues[ci];
            f.omega.clamp(settings_.omegaMin, 1e15);
            if (settings_.legacyOmegaWallPinning)
                applyOmegaBC(f.omega, mesh_, bcs_, nu_, sst_.coeffs.beta1);

            // Enforce the solved constraints exactly and refresh extrapolated
            // faces only AFTER pinning (including outlet/wall corner cells).
            for (int ci = 0; ci < nc; ++ci)
                if (std::isfinite(workspace.wallOmega[ci]))
                    f.omega[ci] = workspace.wallOmega[ci];
            if (!settings_.legacyOmegaWallPinning)
                applyOmegaBC(f.omega, mesh_, bcs_, nu_, sst_.coeffs.beta1);

            // Compute field-change norms: ||phi_new - phi_old||_inf / ||phi_new||_inf
            double kMaxVal = 1e-30, omMaxVal = 1e-30;
            double kMaxDiff = 0.0, omMaxDiff = 0.0;
            for (int ci = 0; ci < mesh_.nCells(); ++ci) {
                kMaxVal = std::max(kMaxVal, std::abs(f.k[ci]));
                omMaxVal = std::max(omMaxVal, std::abs(f.omega[ci]));
                kMaxDiff = std::max(kMaxDiff, std::abs(f.k[ci] - kOld[ci]));
                omMaxDiff = std::max(omMaxDiff, std::abs(f.omega[ci] - omOld[ci]));
            }
            kChangeNorm = kMaxDiff / kMaxVal;
            omChangeNorm = omMaxDiff / omMaxVal;
            lastKChange = kChangeNorm;
            lastOmChange = omChangeNorm;
            estK.push(kChangeNorm);
            estOm.push(omChangeNorm);
            // the turbulence phase ends here: solves, clamps, BCs, wall
            // repinning and the change norms are all inside the account
            turbSeconds_ +=
                std::chrono::duration<double>(std::chrono::steady_clock::now() - tT0).count();
        }

        // 6. Track residuals
        //    Momentum and pressure: absolute initial residual normalised by iter-0 values
        //    (convergence = orders-of-magnitude reduction in equation imbalance).
        //    k and omega: field-change norms (||new - old||_inf / ||new||_inf) because
        //    wall re-pinning creates a persistent equation imbalance that never goes to zero.
        ResidualEntry entry;
        entry.iteration = iter;
        entry.Ux = resUx.initialRes;
        entry.Uy = resUy.initialRes;
        entry.Uz = resUz.initialRes;
        entry.p = resP.initialRes;
        // the CARRIED norms: on non-update iterations these hold the most
        // recent turbulence change, so the recorded history and the
        // convergence check below never see a false zero
        entry.k = turbActive ? lastKChange : 0.0;
        entry.omega = turbActive ? lastOmChange : 0.0;

        // default criterion: state-scaled residuals and contraction estimates
        double uMax = 1e-300, duMax = 0.0;
        for (int ci = 0; ci < nc; ++ci) {
            uMax = std::max(uMax, f.U[ci].norm());
            duMax = std::max(duMax, (f.U[ci] - Uprev[ci]).norm());
        }
        entry.momentumScaled = momResid / std::max(momScale, 1e-300);
        entry.continuityScaled = contScaled;
        entry.changeU = duMax / uMax;
        estU.push(entry.changeU);
        entry.remainingU = estU.remaining(negligible);
        if (monitor_ && !settings_.legacyConvergence &&
            iter % std::max(settings_.monitorInterval, 1) == 0) {
            std::vector<double> q = monitor_(f);
            if (!std::all_of(q.begin(), q.end(), [](double value) { return std::isfinite(value); }))
                throw std::runtime_error("convergence monitor returned a nonfinite observation");
            if (qPrev.size() == q.size()) {
                double qMax = 0.0, dq = 0.0;
                for (double v : q)
                    qMax = std::max(qMax, std::abs(v));
                for (size_t i = 0; i < q.size(); ++i) {
                    const double floor =
                        i < monitorFloors_.size() ? monitorFloors_[i] : 1e-6 * qMax;
                    dq = std::max(dq, std::abs(q[i] - qPrev[i]) /
                                          std::max({std::abs(q[i]), floor, 1e-300}));
                }
                estQ.push(dq);
            }
            qPrev = std::move(q);
        }
        worst = std::max(
            {entry.momentumScaled, entry.continuityScaled, entry.changeU, entry.remainingU});
        if (turbActive)
            worst = std::max({worst, lastKChange, estK.remaining(negligible), lastOmChange,
                              estOm.remaining(negligible)});
        if (monitor_)
            worst = std::max({worst, estQ.last(), estQ.remaining(negligible)});
        if (!std::isfinite(worst))
            worst = std::numeric_limits<double>::infinity();

        // store iter-0 norms for normalisation. The pressure norm keeps a
        // running max over a short warmup: on a fully periodic domain a
        // uniform initial field is EXACTLY divergence-free, so the iter-0 mass
        // imbalance is machine round-off and normalising by it turns every
        // later tiny residual into a false divergence alarm; the true residual
        // scale only appears once the flow develops. Bounded inlet/outlet
        // cases peak at iter 0 anyway, so their behaviour is unchanged.
        if (iter == 0) {
            normUx0_ = std::max(entry.Ux, 1e-30);
            normUy0_ = std::max(entry.Uy, 1e-30);
            normP0_ = std::max(entry.p, 1e-30);
            pressureFluxReference = massFluxScale_;
        } else if (iter < 50) {
            normP0_ = std::max(normP0_, entry.p);
            pressureFluxReference = std::max(pressureFluxReference, massFluxScale_);
        }

        // normalised residuals (skip equations with negligible iter-0 residual)
        // in 2D, Uy/Uz have zero forcing so normUy0_ ~ 1e-30.  dividing any tiny
        // Uy residual by 1e-30 produces O(1e10), triggering false divergence.
        // safeNorm returns 0 for these equations so they don't affect convergence.
        auto safeNorm = [](double val, double ref) -> double {
            if (ref < 1e-20)
                return 0.0; // equation has no forcing; always "converged"
            return val / ref;
        };
        // Both momentum components are judged against the common momentum
        // scale max(normUx0_, normUy0_). Normalising Uy by its own iter-0
        // value is meaningless when that value is tiny but nonzero (e.g. the
        // Reynolds-stress injection contributes a small y-force at iter 0):
        // the y-equation would be held to a reference orders of magnitude
        // below the momentum balance of the problem and never "converge".
        const double normMom0 = std::max(normUx0_, normUy0_);
        double nUx = safeNorm(entry.Ux, normMom0);
        double nUy = safeNorm(entry.Uy, normMom0);
        // A divergence-free initial state can cancel its mass imbalance to
        // roundoff throughout warmup. For the default criterion, normalize
        // pressure growth by the physical startup through-flow as well. Both
        // quantities have flux units. Freeze this reference after warmup so
        // later unbounded growth is still visible; retain the legacy metric.
        const double pressureReference =
            settings_.legacyConvergence ? normP0_ : std::max(normP0_, pressureFluxReference);
        double nP = safeNorm(entry.p, pressureReference);
        // k and omega: carried field-change norms (already normalised; the
        // most recent update's change, never a fresh zero between updates)
        double nK = lastKChange;
        double nOm = lastOmChange;
        hist.entries.push_back(entry);

        // computes maximum normalised residual
        double maxRes = std::max({nUx, nUy, nP});
        if (turbActive)
            maxRes = std::max({maxRes, nK, nOm});

        if (settings_.verbose && (iter % settings_.reportInterval == 0 || iter == 0)) {
            std::cout << "  SIMPLE iter " << iter << "  Ux=" << nUx << "  Uy=" << nUy
                      << "  p=" << nP;
            if (turbUpdate)
                std::cout << "  k=" << nK << "  w=" << nOm;
            if (!settings_.legacyConvergence)
                std::cout << "  | mom=" << entry.momentumScaled
                          << " cont=" << entry.continuityScaled << " dU=" << entry.changeU
                          << " remU=" << entry.remainingU << " worst=" << worst;
            std::cout << "\n";
        }

        // If any individual residual or carried norm is non-finite, force
        // maxRes to infinity BEFORE the convergence test so a NaN can neither
        // be masked by chained std::max operand ordering nor slip past the
        // (NaN < tol) == false accident into a later iteration
        if (!std::isfinite(entry.Ux) || !std::isfinite(entry.Uy) || !std::isfinite(entry.Uz) ||
            !std::isfinite(entry.p) || !std::isfinite(entry.k) || !std::isfinite(entry.omega) ||
            !std::isfinite(lastKChange) || !std::isfinite(lastOmChange) ||
            !std::isfinite(entry.momentumScaled) || !std::isfinite(entry.continuityScaled) ||
            !std::isfinite(entry.changeU)) {
            maxRes = std::numeric_limits<double>::infinity();
            worst = std::numeric_limits<double>::infinity();
        }

        // convergence check (all normalised residuals below tolerance).
        // Scheduled turbulence must have RUN and its startup floor released:
        // without the turbScheduled gate a solve could declare convergence
        // before its first turbulence update, on a laminar transient the
        // criterion never sees; with it, no run can freeze a floored or
        // turbulence-free state as its converged solution.
        const bool turbScheduled = settings_.turbStartIter < settings_.maxIterations;
        hist.finalWorstMeasure = settings_.legacyConvergence ? maxRes : worst;
        hist.contractionRateU = estU.rate();
        const bool measureBelowTol =
            settings_.legacyConvergence ? (maxRes < settings_.convergenceTol) : (worst < tol);
        if (measureBelowTol && iter > 0 && (!turbScheduled || (turbActive && f.turbEstablished))) {
            hist.converged = true;
            hist.finalIter = iter;
            if (settings_.verbose)
                std::cout << "  SIMPLE converged at iteration " << iter << "\n";
            reportTimers();
            return hist;
        }

        // divergence check
        if (std::isnan(maxRes) || std::isinf(maxRes) || maxRes > settings_.divergenceLimit) {
            hist.diverged = true;
            hist.finalIter = iter;
            if (settings_.verbose)
                std::cout << "  SIMPLE diverged at iteration " << iter << "\n";
            reportTimers();
            return hist;
        }
    }

    hist.finalIter = settings_.maxIterations;
    // out of iterations: stalled if U had stopped contracting over the last
    // window (rate >= 1), merely unfinished if it was still contracting
    hist.stalled = std::isfinite(hist.contractionRateU) && hist.contractionRateU >= 1.0;
    if (settings_.verbose)
        std::cout << "  SIMPLE reached maxIter (" << settings_.maxIterations << ")"
                  << (hist.stalled ? ", stalled" : "") << "\n";
    reportTimers();
    return hist;
}

// ===================================================================================
// Coefficient partials differentiate the 11 SST parameters while holding the
// primary state (U, p, k, omega) fixed. They do not provide an analytic state
// Jacobian or adjoint.
// ===================================================================================

// Recompute the SST closure of `work` from sst_.coeffs, exactly mirroring the
// turbulence-update branch of solve() at a CONVERGED state: the startup-only
// nuT floor has released by convergence, so the mirror applies only the
// non-negativity clamp (a floored mirror would make residual/derivative
// evaluations disagree with the converged assembly).
void SIMPLESolver::recomputeClosure(FlowFields &work, ScalarField &Smag) {
    sst_.computeFields(mesh_, work.k, work.omega, work.U, nu_, work.nuT, work.F1, work.F2, work.Pk,
                       work.CDkw);
    for (int ci = 0; ci < mesh_.nCells(); ++ci)
        work.nuT[ci] = std::max(work.nuT[ci], 0.0);
    Smag = strainRateMagnitude(computeVelocityGradients(work.U));
    applyProductionPolicy(work, Smag);
}

// R(U,θ): assemble the discrete residual of momentum-x, -y, k, ω at the FIXED state with
// closure recomputed from θ.  Reuses the solver's own assemble* so the discretisation is
// bit-identical to solve().  The unrelaxed residual r = b − A·φ is recovered by evaluating
// LinearSystem::residual at φ = φ_old: the under-relaxation source term (1−α)/α·a_P·φ_old
// cancels the diagonal scaling a_P/α exactly at the current field value.
SIMPLESolver::DiagnosticResidual
SIMPLESolver::diagnosticResidualImpl(const FlowFields &state, const SSTCoefficients &theta,
                                     const std::vector<double> *suppliedFlux) {
    DiagnosticWorkspace workspace(mesh_);
    DiagnosticResidual output;
    diagnosticResidualInto(state, theta, output, workspace, suppliedFlux);
    return output;
}

void SIMPLESolver::diagnosticResidualInto(const FlowFields &state, const SSTCoefficients &theta,
                                          DiagnosticResidual &output,
                                          DiagnosticWorkspace &workspace,
                                          const std::vector<double> *suppliedFlux) {
    mesh_.requireStamp(meshStamp_);
    if (workspace.mesh != &mesh_ || &state == &workspace.fields)
        throw std::invalid_argument(
            "diagnostic workspace must match mesh and be independent of input");
    mesh_.requireStamp(workspace.stamp);
    if (workspace.active)
        throw std::logic_error("diagnostic workspace is already active");
    struct Reset {
        SIMPLESolver &solver;
        DiagnosticWorkspace &workspace;
        SSTCoefficients coefficients;
        const std::vector<double> *flux;
        ~Reset() {
            solver.sst_.coeffs = coefficients;
            solver.convFlux_ = flux;
            workspace.active = false;
        }
    } reset{*this, workspace, sst_.coeffs, convFlux_};
    workspace.active = true;
    const int nc = mesh_.nCells();
    sst_.coeffs = theta;

    // residuals must be state-consistent: the injection source (if any) is
    // evaluated fresh from `state`, not blended across calls
    injRelax_ = 1.0;
    injSrcBlend_.clear();

    auto &work = workspace.fields;
    work = state;
    auto &Smag = workspace.strain;
    sst_.computeFields(mesh_, work.k, work.omega, work.U, nu_, work.nuT, work.F1, work.F2, work.Pk,
                       work.CDkw, Smag, workspace.closure);
    for (int ci = 0; ci < nc; ++ci)
        work.nuT[ci] = std::max(work.nuT[ci], 0.0);
    applyProductionPolicy(work, Smag);

    // Diagnostic convection uses exactly two flux reconstruction passes.
    auto &phiSteady = output.faceFluxes;
    if (suppliedFlux)
        phiSteady = *suppliedFlux;
    else
        steadyFluxes(work, phiSteady);
    if (!settings_.legacyRhieChowGate)
        convFlux_ = &phiSteady;

    auto &R = output.transport;
    R.assign(4 * nc, 0.0);
    auto &aP = workspace.diagonal;
    aP.assign(nc, 0.0);
    auto &sys = workspace.system;
    auto &phi = workspace.values;
    phi.resize(nc);
    auto &r = workspace.residual;
    r.resize(nc);

    auto block = [&](int b) { return b * nc; }; // [Rux|Ruy|Rk|Rω] block offset

    // momentum x
    assembleMomentum(sys, work, 0, aP);
    for (int ci = 0; ci < nc; ++ci)
        phi[ci] = work.U[ci].x;
    sys.residual(phi, r);
    for (int ci = 0; ci < nc; ++ci)
        R[block(0) + ci] = r[ci];
    // momentum y
    assembleMomentum(sys, work, 1, aP);
    for (int ci = 0; ci < nc; ++ci)
        phi[ci] = work.U[ci].y;
    sys.residual(phi, r);
    for (int ci = 0; ci < nc; ++ci)
        R[block(1) + ci] = r[ci];
    // k
    assembleKEquation(sys, work);
    for (int ci = 0; ci < nc; ++ci)
        phi[ci] = work.k[ci];
    sys.residual(phi, r);
    for (int ci = 0; ci < nc; ++ci)
        R[block(2) + ci] = r[ci];
    // omega (frozen Smag, as in solve())
    assembleOmegaEquation(sys, work, Smag);
    for (int ci = 0; ci < nc; ++ci)
        phi[ci] = work.omega[ci];
    sys.residual(phi, r);
    for (int ci = 0; ci < nc; ++ci)
        R[block(3) + ci] = r[ci];

    convFlux_ = nullptr;
    auto &div = output.continuity;
    div.assign(nc, 0.0);
    for (int fi = 0; fi < mesh_.nFaces(); ++fi) {
        const auto &face = mesh_.face(fi);
        div[face.owner] += phiSteady[fi];
        if (!face.isBoundary())
            div[face.neighbor] -= phiSteady[fi];
    }
}

// Discrete continuity residual at a fixed state: net outward steady face flux
// per cell, closure from theta exactly as in assembleResidual (the Rhie-Chow
// coefficient sees the closure through the momentum diagonal).
std::vector<double> SIMPLESolver::continuityResidual(const FlowFields &state,
                                                     const SSTCoefficients &theta) {
    mesh_.requireStamp(meshStamp_);
    const int nc = mesh_.nCells();
    theta.validate();
    CoefficientRestore restore(sst_.coeffs);
    sst_.coeffs = theta;
    injRelax_ = 1.0;
    injSrcBlend_.clear();
    FlowFields work = state;
    ScalarField Smag(mesh_, "Smag");
    recomputeClosure(work, Smag);
    std::vector<double> phi;
    steadyFluxes(work, phi);
    if (settings_.legacyRhieChowGate && legacyRhieChowActive()) {
        // the legacy outlet-free discretisation: relaxed coefficient
        std::vector<double> aPtmp(nc, 0.0);
        LinearSystem sys = makeSystem(mesh_);
        assembleMomentum(sys, work, 0, aPtmp);
        addRhieChow(work, aPtmp, phi);
    }
    std::vector<double> div(nc, 0.0);
    for (int fi = 0; fi < mesh_.nFaces(); ++fi) {
        const Face &face = mesh_.face(fi);
        div[face.owner] += phi[fi];
        if (!face.isBoundary())
            div[face.neighbor] -= phi[fi];
    }
    return div;
}

// Fixed-flux transport closure contribution; the diagnostic API adds flux partials.  Each closure
// field enters the assembly linearly (nuT in the diffusion coefficients, Pk and the cross-diffusion
// in the volumetric sources, F1 in the σ_k/σ_w/α/β blends and the (1−F1) cross-diffusion factor),
// so ∂R/∂θ is the same face/cell walk with the values replaced by their pointwise closure
// sensitivities.  Convection (frozen U), the pressure-gradient source and all boundary VALUES are
// θ-independent and drop out.
std::vector<std::vector<double>>
SIMPLESolver::transportClosurePartials(const FlowFields &state, const SSTCoefficients &theta,
                                       bool includeTransposeTheta) {
    if (productionPolicyActive())
        throw std::logic_error("analytic sensitivities do not support custom production policies");
    if (bTarget6_ && injectionScale_ != 0.0)
        throw std::logic_error(
            "analytic sensitivities do not include target-injection derivatives");
    const int nc = mesh_.nCells();
    const int nIF = mesh_.nInternalFaces();
    theta.validate();
    CoefficientRestore restore(sst_.coeffs);
    sst_.coeffs = theta;

    FlowFields work = state;
    ScalarField Smag(mesh_, "Smag");
    recomputeClosure(work, Smag);

    // pointwise closure sensitivities per cell (Layer 1); the startup-only
    // nuT floor has released at the converged states this is evaluated on,
    // so the derivative carries no floor dead-zone
    const auto &wd = mesh_.wallDistance();
    std::vector<SSTClosureSensitivity> cs(nc);
    for (int ci = 0; ci < nc; ++ci)
        cs[ci] = sst_.closureSensitivity(work.k[ci], work.omega[ci], Smag[ci], wd[ci], nu_,
                                         work.CDkw[ci], 0.0);

    std::vector<std::vector<double>> dR(11, std::vector<double>(4 * nc, 0.0));
    const int BUX = 0, BUY = nc, BK = 2 * nc, BOM = 3 * nc;

    // blend coefficients φ = F1·φ1 + (1−F1)·φ2
    const double sk1 = sst_.coeffs.sigma_k1, sk2 = sst_.coeffs.sigma_k2;
    const double sw1 = sst_.coeffs.sigma_w1, sw2 = sst_.coeffs.sigma_w2;
    const double al1 = sst_.coeffs.alpha1, al2 = sst_.coeffs.alpha2;
    const double be1 = sst_.coeffs.beta1, be2 = sst_.coeffs.beta2;

    // ---- internal faces: convection (θ-indep) + diffusion (θ via nuT, F1) ----------
    for (int fi = 0; fi < nIF; ++fi) {
        const Face &face = mesh_.face(fi);
        const int o = face.owner, n = face.neighbor;
        const double w = face.weight;
        const double SfD = face.area / std::max(face.delta, 1e-20);
        const double nuTw = w * work.nuT[o] + (1.0 - w) * work.nuT[n];
        const double F1f = w * work.F1[o] + (1.0 - w) * work.F1[n];
        const double skf = sk1 * F1f + sk2 * (1.0 - F1f);
        const double swf = sw1 * F1f + sw2 * (1.0 - F1f);
        const double dUx = work.U[n].x - work.U[o].x;
        const double dUy = work.U[n].y - work.U[o].y;
        const double dk = work.k[n] - work.k[o];
        const double dw = work.omega[n] - work.omega[o];

        for (int j = 0; j < 11; ++j) {
            const double dnuTw = w * cs[o].dnuT[j] + (1.0 - w) * cs[n].dnuT[j];
            const double dF1f = w * cs[o].dF1[j] + (1.0 - w) * cs[n].dF1[j];

            // momentum: nuEff_f = ν + nuTw
            const double dDfm = SfD * dnuTw;
            if (dDfm != 0.0) {
                dR[j][BUX + o] += dDfm * dUx;
                dR[j][BUX + n] -= dDfm * dUx;
                dR[j][BUY + o] += dDfm * dUy;
                dR[j][BUY + n] -= dDfm * dUy;
            }
            // k: nuEff_f = ν + σ_k(F1)·nuTw
            double dsk = (sk1 - sk2) * dF1f;
            if (j == 0)
                dsk += F1f;
            else if (j == 4)
                dsk += (1.0 - F1f);
            const double dDfk = SfD * (dsk * nuTw + skf * dnuTw);
            if (dDfk != 0.0) {
                dR[j][BK + o] += dDfk * dk;
                dR[j][BK + n] -= dDfk * dk;
            }
            // ω: nuEff_f = ν + σ_w(F1)·nuTw
            double dsw = (sw1 - sw2) * dF1f;
            if (j == 1)
                dsw += F1f;
            else if (j == 5)
                dsw += (1.0 - F1f);
            const double dDfw = SfD * (dsw * nuTw + swf * dnuTw);
            if (dDfw != 0.0) {
                dR[j][BOM + o] += dDfw * dw;
                dR[j][BOM + n] -= dDfw * dw;
            }
            if (!omegaSingularity_.empty()) {
                const double amplitude = 6 * nu_ / be1;
                const double difference =
                    (omegaSingularity_[n] - omegaSingularity_[o]) / face.delta;
                double correction =
                    amplitude * dDfw * face.delta * (omegaNormalGradient_[fi] - difference);
                if (j == 2)
                    correction -=
                        amplitude * face.area / be1 *
                        (swf * nuTw * omegaNormalGradient_[fi] - (nu_ + swf * nuTw) * difference);
                dR[j][BOM + o] += correction;
                dR[j][BOM + n] -= correction;
            }
        }
    }

    // ---- boundary faces: both in/out branches give dDb·(φ_b − φ_o) (mFlux θ-indep) ---
    // wall faces carry a MOLECULAR momentum diffusion coefficient at resolved
    // walls (assembleMomentum), so their U-block theta-derivative is zero
    // there; the k/omega wall diffusion keeps its nuT dependence unchanged
    std::vector<char> isWallB(mesh_.nFaces(), 0);
    if (!settings_.useWallFunctions) {
        for (int pi = 0; pi < mesh_.nPatches(); ++pi) {
            const Patch &pat = mesh_.patch(pi);
            if (pat.type != "wall")
                continue;
            for (FaceID wfi : pat.faces)
                isWallB[wfi] = 1;
        }
    }
    for (int fi = nIF; fi < mesh_.nFaces(); ++fi) {
        const Face &face = mesh_.face(fi);
        const int o = face.owner;
        const double SfD = face.area / std::max(face.delta, 1e-20);
        const double F1o = work.F1[o], nuTo = work.nuT[o];
        const double sko = sk1 * F1o + sk2 * (1.0 - F1o);
        const double swo = sw1 * F1o + sw2 * (1.0 - F1o);
        const double dUx = work.U.bface(fi).x - work.U[o].x;
        const double dUy = work.U.bface(fi).y - work.U[o].y;
        const double dk = work.k.bface(fi) - work.k[o];
        const double dw = work.omega.bface(fi) - work.omega[o];

        for (int j = 0; j < 11; ++j) {
            const double dnuTo = cs[o].dnuT[j], dF1o = cs[o].dF1[j];
            const double dDbm = isWallB[fi] ? 0.0 : SfD * dnuTo;
            if (dDbm != 0.0) {
                dR[j][BUX + o] += dDbm * dUx;
                dR[j][BUY + o] += dDbm * dUy;
            }
            double dsk = (sk1 - sk2) * dF1o;
            if (j == 0)
                dsk += F1o;
            else if (j == 4)
                dsk += (1.0 - F1o);
            const double dDbk = SfD * (dsk * nuTo + sko * dnuTo);
            if (dDbk != 0.0)
                dR[j][BK + o] += dDbk * dk;
            double dsw = (sw1 - sw2) * dF1o;
            if (j == 1)
                dsw += F1o;
            else if (j == 5)
                dsw += (1.0 - F1o);
            const double dDbw = SfD * (dsw * nuTo + swo * dnuTo);
            if (dDbw != 0.0)
                dR[j][BOM + o] += dDbw * dw;
        }
    }

    // ---- transpose-stress theta-derivative ------------------------------------------
    // the momentum source carries div(nuT (grad U)^T), whose
    // theta-dependence is through nuT alone (U is the frozen state):
    //   d(flux)/dtheta_j = dnuT_f (dU_./dx_i)_f . n A, mirrored face-by-face
    if (includeTransposeTheta) {
        VelocityGradients vgT = computeVelocityGradients(work.U);
        auto gcol = [&](int ci, int comp) -> Vec3 {
            if (comp == 0)
                return Vec3(vgT.dudx[ci].x, vgT.dvdx[ci].x, vgT.dwdx[ci].x);
            return Vec3(vgT.dudx[ci].y, vgT.dvdx[ci].y, vgT.dwdx[ci].y);
        };
        for (int fi = 0; fi < nIF; ++fi) {
            const Face &face = mesh_.face(fi);
            const int o = face.owner, n2 = face.neighbor;
            const double w = face.weight;
            const Vec3 gx = gcol(o, 0) * w + gcol(n2, 0) * (1.0 - w);
            const Vec3 gy = gcol(o, 1) * w + gcol(n2, 1) * (1.0 - w);
            const double fx = gx.dot(face.normal) * face.area;
            const double fy = gy.dot(face.normal) * face.area;
            for (int j = 0; j < 11; ++j) {
                const double dnuTf = w * cs[o].dnuT[j] + (1.0 - w) * cs[n2].dnuT[j];
                if (dnuTf == 0.0)
                    continue;
                dR[j][BUX + o] += dnuTf * fx;
                dR[j][BUX + n2] -= dnuTf * fx;
                dR[j][BUY + o] += dnuTf * fy;
                dR[j][BUY + n2] -= dnuTf * fy;
            }
        }
        // boundary faces, patch-aware: WALL faces carry a zero transpose
        // coefficient in the assembly (the eddy viscosity vanishes at the
        // wall), so their theta-derivative is identically zero and they are
        // skipped here to keep the analytic dR matched to an FD of the
        // assembled residual
        for (int pi = 0; pi < mesh_.nPatches(); ++pi) {
            const Patch &pat = mesh_.patch(pi);
            if (pat.type == "wall")
                continue;
            for (FaceID fi : pat.faces) {
                const Face &face = mesh_.face(fi);
                const int o = face.owner;
                const double fx = gcol(o, 0).dot(face.normal) * face.area;
                const double fy = gcol(o, 1).dot(face.normal) * face.area;
                for (int j = 0; j < 11; ++j) {
                    const double dnuTo = cs[o].dnuT[j];
                    if (dnuTo == 0.0)
                        continue;
                    dR[j][BUX + o] += dnuTo * fx;
                    dR[j][BUY + o] += dnuTo * fy;
                }
            }
        }
    }

    // ---- volumetric sources -------------------------------------------------------
    for (int ci = 0; ci < nc; ++ci) {
        const double V = mesh_.cell(ci).volume;
        if (!omegaSingularity_.empty())
            dR[2][BOM + ci] -= nu_ * (6 * nu_ / be1) * omegaLaplacian_[ci] * V / be1;
        const double F1 = work.F1[ci];
        const double omc = std::max(work.omega[ci], 1e-20);
        const double S = Smag[ci];
        const double ko = work.k[ci], wo = work.omega[ci];
        // corrected omega production q = min(S², 10 β* k ω / νT): the branch decision
        // must mirror productionOmega/the assembly exactly (same guards) so this
        // analytic derivative matches a FD of the assembled residual
        const double bS = sst_.coeffs.betaStar;
        const double alB = sst_.coeffs.alpha(F1);
        const double nuTc = std::max(work.nuT[ci], 1e-30);
        const double lim = 10.0 * bS * std::max(ko, 0.0) * omc / nuTc;
        const bool limActive = lim < S * S; // std::min(S², lim) picks S² on ties
        const double q = limActive ? lim : S * S;
        for (int j = 0; j < 11; ++j) {
            // k production +V·∂Pk ;  k destruction −V·∂β*·ω·k (β* = idx 8)
            dR[j][BK + ci] += V * cs[ci].dPk[j];
            if (j == 8)
                dR[j][BK + ci] -= omc * V * ko;
            // ω production +V·∂[α·q]: ∂α·q always; the limiter branch adds
            // α·∂q with ∂q = lim·(δ_{j,β*}/β* − ∂νT/νT) (k, ω are frozen state;
            // the S² branch has ∂q = 0, reproducing the pre-correction ∂α·S²)
            double dal = (al1 - al2) * cs[ci].dF1[j];
            if (j == 3)
                dal += F1;
            else if (j == 7)
                dal += (1.0 - F1);
            double dq = 0.0;
            if (limActive)
                dq = (j == 8 ? lim / bS : 0.0) - lim * cs[ci].dnuT[j] / nuTc;
            dR[j][BOM + ci] += (dal * q + alB * dq) * V;
            // ω destruction −V·∂β·ω·ω
            double dbe = (be1 - be2) * cs[ci].dF1[j];
            if (j == 2)
                dbe += F1;
            else if (j == 6)
                dbe += (1.0 - F1);
            dR[j][BOM + ci] -= dbe * omc * V * wo;
            // ω cross-diffusion +V·[ −∂F1·CDkw + (1−F1)·∂CDkw ], UNCLIPPED per
            // SST-2003 (matches the corrected assembly; only F1's internal CDkw
            // is clipped, and that path is inside cs.dF1 already)
            double dcross = -cs[ci].dF1[j] * work.CDkw[ci] + (1.0 - F1) * cs[ci].dCDkw[j];
            dR[j][BOM + ci] += V * dcross;
        }
    }

    return dR;
}
// Picard segregated preconditioner blocks at fixed state: each block approximates
// -dR_block/dstate_block, without cross-block coupling or transposition. These
// precondition the matrix-free coupled tangent; they are not its state Jacobian.
void SIMPLESolver::assemblePreconBlocks(const FlowFields &state, const SSTCoefficients &theta,
                                        LinearSystem &Amom, LinearSystem &Ak, LinearSystem &Aom,
                                        std::vector<double> &aP) {
    mesh_.requireStamp(meshStamp_);
    const int nc = mesh_.nCells();
    theta.validate();
    CoefficientRestore restore(sst_.coeffs);
    sst_.coeffs = theta;

    FlowFields work = state;
    ScalarField Smag(mesh_, "Smag");
    recomputeClosure(work, Smag); // closure (nuT/F1/.../floor) + frozen |S| from θ

    aP.assign(nc, 0.0);
    Amom = makeSystem(mesh_);
    assembleMomentum(Amom, work, 0, aP); // x-momentum; aP = relaxed diag
    Ak = makeSystem(mesh_);
    assembleKEquation(Ak, work);
    Aom = makeSystem(mesh_);
    assembleOmegaEquation(Aom, work, Smag);
}

void SIMPLESolver::requireDiagnosticSupport(const FlowFields &state) const {
    mesh_.requireStamp(meshStamp_);
    if (settings_.legacyRhieChowGate || settings_.useWallFunctions || settings_.sstVariant != 0 ||
        sst_.variant != SSTVariant::Full || bTarget6_ || productionPolicy_ ||
        freezeTransposeStress_)
        throw std::logic_error("diagnostic derivatives require baseline Full SST, resolved walls "
                               "and corrected Rhie-Chow");
    for (const auto &scheme : {settings_.momentumConvection, settings_.turbulenceConvection})
        if (scheme != "upwind" && scheme != "linearUpwind")
            throw std::logic_error(
                "diagnostic derivatives support upwind or unlimited linearUpwind");
    auto valid = [&](const auto &field) {
        return &field.mesh() == &mesh_ && field.size() == mesh_.nCells() &&
               field.nBnd() == mesh_.nFaces() - mesh_.nInternalFaces();
    };
    if (!valid(state.U) || !valid(state.p) || !valid(state.k) || !valid(state.omega) ||
        !valid(state.nuT) || !valid(state.F1) || !valid(state.F2) || !valid(state.Pk) ||
        !valid(state.CDkw) || mesh_.wallDistance().size() != static_cast<size_t>(mesh_.nCells()))
        throw std::invalid_argument(
            "diagnostic fields and wall distance must match the prepared mesh");
    for (const auto &u : state.U.data())
        if (u.z != 0.0)
            throw std::logic_error("diagnostic derivatives require zero spanwise velocity");
    for (int fi = mesh_.nInternalFaces(); fi < mesh_.nFaces(); ++fi)
        if (state.U.bface(fi).z != 0.0)
            throw std::logic_error(
                "diagnostic derivatives require zero spanwise boundary velocity");
}

SIMPLESolver::DiagnosticResidual
SIMPLESolver::diagnosticResidual(const FlowFields &state, const SSTCoefficients &theta) const {
    DiagnosticResidual output;
    DiagnosticWorkspace workspace(mesh_);
    diagnosticResidual(state, theta, output, workspace);
    return output;
}

void SIMPLESolver::diagnosticResidual(const FlowFields &state, const SSTCoefficients &theta,
                                      DiagnosticResidual &output,
                                      DiagnosticWorkspace &workspace) const {
    requireDiagnosticSupport(state);
    SIMPLESolver local(mesh_, SSTModel(theta), bcs_, nu_, settings_);
    local.setMomentumSource(momSource_);
    local.diagnosticResidualInto(state, theta, output, workspace);
}

std::vector<double> SIMPLESolver::assembleResidual(const FlowFields &state,
                                                   const SSTCoefficients &theta) {
    // Preserve the historical injection diagnostic side effects. The new
    // diagnostic API owns isolated state and excludes injection entirely.
    return diagnosticResidualImpl(state, theta).transport;
}

std::vector<std::vector<double>>
SIMPLESolver::diagnosticFluxPartials(const FlowFields &work,
                                     const std::vector<SSTClosureSensitivity> &closure,
                                     std::vector<double> &flux) {
    using diagnostic::Partial;
    std::vector<Partial> viscosity(mesh_.nCells());
    for (int i = 0; i < mesh_.nCells(); ++i)
        viscosity[i] = Partial(work.nuT[i], closure[i].dnuT);
    const auto phi = diagnostic::reconstructFlux(mesh_, work, nu_, viscosity);
    flux.resize(phi.size());
    std::vector<std::vector<double>> partials(11, std::vector<double>(phi.size()));
    for (size_t fi = 0; fi < phi.size(); ++fi) {
        flux[fi] = phi[fi].value;
        for (int j = 0; j < 11; ++j)
            partials[j][fi] = phi[fi].derivative[j];
    }
    return partials;
}

std::vector<SIMPLESolver::DiagnosticResidual>
SIMPLESolver::diagnosticParameterPartials(const FlowFields &state,
                                          const SSTCoefficients &theta) const {
    requireDiagnosticSupport(state);
    SIMPLESolver local(mesh_, SSTModel(theta), bcs_, nu_, settings_);
    FlowFields work = state;
    ScalarField strain(mesh_, "diagnostic_strain");
    local.recomputeClosure(work, strain);
    std::vector<SSTClosureSensitivity> closure(mesh_.nCells());
    for (int i = 0; i < mesh_.nCells(); ++i)
        closure[i] = local.sst_.closureSensitivity(work.k[i], work.omega[i], strain[i],
                                                   mesh_.wallDistance()[i], nu_, work.CDkw[i], 0.0);
    auto transport = local.transportClosurePartials(state, theta, true);
    std::vector<double> flux;
    const auto dFlux = local.diagnosticFluxPartials(work, closure, flux);
    std::vector<DiagnosticResidual> result(11);
    for (int j = 0; j < 11; ++j) {
        result[j].transport = std::move(transport[j]);
        result[j].faceFluxes = dFlux[j];
        result[j].continuity.assign(mesh_.nCells(), 0);
        for (int fi = 0; fi < mesh_.nFaces(); ++fi) {
            const auto &face = mesh_.face(fi);
            result[j].continuity[face.owner] += dFlux[j][fi];
            if (!face.isBoundary())
                result[j].continuity[face.neighbor] -= dFlux[j][fi];
        }
    }
    // At fixed fields, every convective residual contribution depends on the
    // flux only. Include both matrix upwinding and deferred correction, using
    // the very same increment kernel as the primal assembly.
    for (int block = 0; block < 4; ++block) {
        ScalarField q(mesh_, "transported");
        auto cellValue = [&](int i) {
            return block == 0   ? work.U[i].x
                   : block == 1 ? work.U[i].y
                   : block == 2 ? work.k[i]
                                : work.omega[i];
        };
        auto boundaryValue = [&](int fi) {
            return block == 0   ? work.U.bface(fi).x
                   : block == 1 ? work.U.bface(fi).y
                   : block == 2 ? work.k.bface(fi)
                                : work.omega.bface(fi);
        };
        for (int i = 0; i < mesh_.nCells(); ++i)
            q[i] = cellValue(i);
        for (int fi = mesh_.nInternalFaces(); fi < mesh_.nFaces(); ++fi)
            q.bface(fi) = boundaryValue(fi);
        const auto gradient = greenGaussGrad(q);
        const bool higher =
            (block < 2 ? settings_.momentumConvection : settings_.turbulenceConvection) != "upwind";
        for (int fi = 0; fi < mesh_.nFaces(); ++fi) {
            const auto &face = mesh_.face(fi);
            const int o = face.owner, n = face.neighbor;
            double upwind = face.isBoundary() ? (flux[fi] >= 0 ? q[o] : q.bface(fi))
                                              : (flux[fi] > 0 ? q[o] : q[n]);
            double increment = 0;
            if (higher && !face.isBoundary() && flux[fi] != 0)
                increment =
                    diagnostic::convectionIncrement(mesh_, face, q, gradient, flux[fi], false);
            for (int j = 0; j < 11; ++j) {
                const double correction = dFlux[j][fi] * upwind + dFlux[j][fi] * increment;
                result[j].transport[block * mesh_.nCells() + o] -= correction;
                if (!face.isBoundary())
                    result[j].transport[block * mesh_.nCells() + n] += correction;
            }
        }
    }
    return result;
}

std::vector<std::vector<double>>
SIMPLESolver::assembleResidualSensitivity(const FlowFields &state, const SSTCoefficients &theta,
                                          bool includeTransposeTheta) {
    mesh_.requireStamp(meshStamp_);
    if (!settings_.legacyRhieChowGate && includeTransposeTheta && !freezeTransposeStress_ &&
        settings_.momentumConvection != "limitedLinearUpwind" &&
        settings_.turbulenceConvection != "limitedLinearUpwind") {
        auto full = diagnosticParameterPartials(state, theta);
        std::vector<std::vector<double>> transport;
        for (auto &column : full)
            transport.push_back(std::move(column.transport));
        return transport;
    }
    SIMPLESolver local(mesh_, sst_, bcs_, nu_, settings_);
    local.setTargetAnisotropy(bTarget6_);
    local.setInjectionScale(injectionScale_);
    local.setProductionPolicy(productionPolicy_);
    return local.transportClosurePartials(state, theta, includeTransposeTheta);
}

std::vector<int> SIMPLESolver::freeFluxFaces() const {
    mesh_.requireStamp(meshStamp_);
    std::vector<int> result;
    for (int fi = 0; fi < mesh_.nInternalFaces(); ++fi)
        result.push_back(fi);
    // Keep face order independent of patch order.
    for (int fi = mesh_.nInternalFaces(); fi < mesh_.nFaces(); ++fi) {
        const int patch = mesh_.face(fi).patchID;
        if (bcs_.pressureBC.at(patch).type == BCType::OutletPressure &&
            bcs_.velocityBC.at(patch).type == BCType::Neumann)
            result.push_back(fi);
    }
    return result;
}

SIMPLESolver::SteadyResidual
SIMPLESolver::steadyResidual(const FlowFields &state, const SSTCoefficients &theta,
                             const std::vector<double> &freeFluxes) const {
    requireDiagnosticSupport(state);
    if (settings_.legacyOmegaWallPinning)
        throw std::logic_error(
            "steady operator requires constrained omega wall cells, not legacy solve-then-pin");
    const auto freeFaces = freeFluxFaces();
    if (freeFluxes.size() != freeFaces.size() ||
        !std::all_of(freeFluxes.begin(), freeFluxes.end(),
                     [](double x) { return std::isfinite(x); }))
        throw std::invalid_argument("steady operator requires one finite flux per free face");
    FlowFields work = state;
    applyVelocityBC(work.U, mesh_, bcs_);
    applyPressureBC(work.p, mesh_, bcs_);
    applyKBC(work.k, mesh_, bcs_);
    applyOmegaBC(work.omega, mesh_, bcs_, nu_, theta.beta1);
    SIMPLESolver local(mesh_, SSTModel(theta), bcs_, nu_, settings_);
    local.setMomentumSource(momSource_);
    std::vector<double> flux;
    local.interpolatedFluxes(work, flux);
    for (size_t i = 0; i < freeFaces.size(); ++i)
        flux[freeFaces[i]] = freeFluxes[i];
    const auto transport = local.diagnosticResidualImpl(work, theta, &flux);
    SteadyResidual result;
    result.equations = transport.equations();
    result.continuity = transport.continuity;
    result.faceFluxes = flux;
    const int nc = mesh_.nCells();
    // Pin order deliberately matches solve(): the last wall face of a corner
    // cell wins. Do not overwrite state omega before evaluating other rows.
    std::vector<double> wallOmega;
    local.omegaWallValues(work, wallOmega);
    for (int cell = 0; cell < nc; ++cell)
        if (std::isfinite(wallOmega[cell]))
            result.equations[3 * nc + cell] = wallOmega[cell] - work.omega[cell];
    result.pressureGauge = true;
    for (const auto &bc : bcs_.pressureBC)
        if (bc.type == BCType::OutletPressure || bc.type == BCType::Dirichlet)
            result.pressureGauge = false;
    if (result.pressureGauge) {
        double integral = 0, volume = 0;
        for (int i = 0; i < nc; ++i) {
            integral += work.p[i] * mesh_.cell(i).volume;
            volume += mesh_.cell(i).volume;
        }
        result.removedContinuity = result.continuity[0];
        result.equations[4 * nc] = integral / volume;
    }
    // aPunrelaxed_ was assembled from this supplied flux and fresh closure,
    // not from the diagnostic two-pass elimination. The primal's signed
    // correction is applied on internal faces only; outlets use U_boundary.S.
    std::vector<double> consistent;
    local.interpolatedFluxes(work, consistent);
    local.addRhieChow(work, local.aPunrelaxed_, consistent);
    for (int fi : freeFaces)
        result.equations.push_back(flux[fi] - consistent[fi]);
    return result;
}
