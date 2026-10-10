#include "DBNSSolver.hpp"
#include "HLLCFlux.hpp"
#include "AnisotropyTools.hpp"
#include <array>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <limits>

namespace dbns {

// primitive-variable indices into grad_ / limiter_ arrays
enum GradIdx { G_RHO = 0, G_U = 1, G_V = 2, G_P = 3, G_K = 4, G_W = 5 };

namespace {
// Validate caller-provided states before writing any solver storage. The flux
// kernels remain unchecked; evolved states use the solver's Diverged outcome.
// Total energy includes rho*k and a laminar solve clears k every step, so a
// laminar k would silently become heat. Laminar states must carry no
// turbulence at all.
void validatePrimitive(const Primitive &state, const IdealGasEOS &eos, bool turbulent) {
    for (double value : {state.rho, state.u, state.v, state.p, state.k, state.omega})
        if (!std::isfinite(value))
            throw std::invalid_argument("DBNS primitive values must be finite");
    if (state.rho <= 0 || state.p <= 0 || state.k < 0 || state.omega < 0)
        throw std::invalid_argument(
            "DBNS primitive density and pressure must be positive; k and omega nonnegative");
    if (!turbulent && (state.k != 0 || state.omega != 0))
        throw std::invalid_argument("laminar DBNS states require k = 0 and omega = 0; set "
                                    "DBNSSettings::turbulent to transport turbulence");
    if (!GasState::admissible(GasState::toConserved(state, eos), eos) ||
        !std::isfinite(GasState::temperature(state, eos)) ||
        GasState::temperature(state, eos) <= 0 || !std::isfinite(GasState::soundSpeed(state, eos)))
        throw std::invalid_argument("DBNS primitive state exceeds the finite thermodynamic range");
}
} // namespace

DBNSSolver::DBNSSolver(const Mesh &mesh, const IdealGasEOS &eos, const SSTCoefficients &sst,
                       const DBNSBoundaryConditions &bcs, const DBNSSettings &settings)
    : mesh_(mesh), eos_(eos), sstCoeffs_(sst), sst_(sst), bcs_(bcs), settings_(settings) {
    eos_.validate();
    settings_.validate();
    for (const auto &entry : bcs_.specs) {
        const auto patch = std::find_if(mesh_.patches().begin(), mesh_.patches().end(),
                                        [&](const Patch &p) { return p.name == entry.first; });
        if (patch == mesh_.patches().end())
            throw std::invalid_argument("DBNS boundary specifies unknown patch " + entry.first);
        const auto &spec = entry.second;
        switch (spec.kind) {
        case BoundaryKind::SupersonicInflow:
        case BoundaryKind::FixedState:
            if (spec.profile.empty())
                validatePrimitive(spec.freestream, eos_, settings_.turbulent);
            else {
                if (spec.profile.size() != patch->faces.size())
                    throw std::invalid_argument(
                        "DBNS prescribed profile requires one state per face on " + patch->name);
                for (const auto &state : spec.profile)
                    validatePrimitive(state, eos_, settings_.turbulent);
            }
            break;
        case BoundaryKind::SubsonicInflow:
            validatePrimitive(spec.freestream, eos_, settings_.turbulent);
            break;
        case BoundaryKind::SubsonicOutflow:
            if (!std::isfinite(spec.backPressure) || spec.backPressure <= 0)
                throw std::invalid_argument("DBNS back pressure must be finite and positive");
            break;
        case BoundaryKind::NoSlipIsothermal:
            if (spec.wallTempProfile.empty()) {
                if (!std::isfinite(spec.wallTemp) || spec.wallTemp <= 0)
                    throw std::invalid_argument(
                        "DBNS wall temperature must be finite and positive");
            } else {
                if (spec.wallTempProfile.size() != patch->faces.size())
                    throw std::invalid_argument(
                        "DBNS wall-temperature profile requires one value per face on " +
                        patch->name);
                for (double value : spec.wallTempProfile)
                    if (!std::isfinite(value) || value <= 0)
                        throw std::invalid_argument(
                            "DBNS wall-temperature profile must be finite and positive");
            }
            [[fallthrough]];
        case BoundaryKind::NoSlipAdiabatic:
            if (!std::isfinite(spec.wallVelocity.x) || !std::isfinite(spec.wallVelocity.y) ||
                !std::isfinite(spec.wallVelocity.z))
                throw std::invalid_argument("DBNS wall velocity must be finite");
            break;
        case BoundaryKind::Extrapolate:
        case BoundaryKind::SlipWall:
            break;
        default:
            throw std::invalid_argument("DBNS boundary kind is not recognized");
        }
    }
    int nc = mesh_.nCells();
    W_.assign(nc, StateVec{});
    // Cell volumes come from the Mesh. It used to store 2/3 of the true
    // volume for z-open 2D cells whenever computeGeometry() ran, so this
    // solver carried a private 2D divergence-theorem copy; the Mesh now has
    // one exact convention on every path (test_mesh_volume_convention).
    muLam_.assign(nc, 0.0);
    muT_.assign(nc, 0.0);
    // patch-local ordinal of each boundary face (for per-face boundary
    // profiles): patchLocalIdx_[globalFace - nInternalFaces] = position of
    // that face within its patch's own face list
    patchLocalIdx_.assign(mesh_.nFaces() - mesh_.nInternalFaces(), 0);
    for (int pi = 0; pi < mesh_.nPatches(); ++pi) {
        const Patch &pat = mesh_.patch(pi);
        for (size_t j = 0; j < pat.faces.size(); ++j) {
            int bidx = pat.faces[j] - mesh_.nInternalFaces();
            if (bidx >= 0 && bidx < (int)patchLocalIdx_.size())
                patchLocalIdx_[bidx] = (int)j;
        }
    }
    grad_.assign(nc, std::array<Vec3, 6>{});
    limiter_.assign(nc, std::array<double, 6>{});
    res_.assign(nc, StateVec{});
    limiterActive_.assign(nc, 0);
    dtCell_.assign(nc, 0.0);

    // wall distance to the no-slip patches of THESE boundary conditions
    // (the mesh's own wallDistance() is lazily populated and was empty for
    // every density-based run, silently breaking the SST blending; and the
    // generator's patch types need not match the imposed conditions, e.g.
    // an imposed-shock top on a channel mesh)
    std::vector<Vec3> wallPts;
    for (int pi = 0; pi < mesh_.nPatches(); ++pi) {
        const Patch &pat = mesh_.patch(pi);
        if (!bcs_.has(pat.name))
            continue;
        BoundaryKind kind = bcs_.get(pat.name).kind;
        if (kind != BoundaryKind::NoSlipAdiabatic && kind != BoundaryKind::NoSlipIsothermal)
            continue;
        for (FaceID fi : pat.faces)
            wallPts.push_back(mesh_.face(fi).center);
    }
    for (int pi = 0; pi < mesh_.nPatches(); ++pi) {
        const Patch &pat = mesh_.patch(pi);
        if (!bcs_.has(pat.name))
            continue;
        BoundaryKind kind = bcs_.get(pat.name).kind;
        if (kind != BoundaryKind::NoSlipAdiabatic && kind != BoundaryKind::NoSlipIsothermal)
            continue;
        for (FaceID fi : pat.faces) {
            wallAdjCell_.push_back(mesh_.face(fi).owner);
            wallAdjDelta_.push_back(std::max(mesh_.face(fi).delta, 1e-12));
        }
    }
    wallDist_.assign(nc, 1e10);
    if (!wallPts.empty()) {
        for (int ci = 0; ci < nc; ++ci) {
            const Vec3 &c = mesh_.cell(ci).center;
            double best = 1e30;
            for (const Vec3 &wpt : wallPts) {
                double dx = c.x - wpt.x, dy = c.y - wpt.y;
                double d2 = dx * dx + dy * dy;
                if (d2 < best)
                    best = d2;
            }
            wallDist_[ci] = std::sqrt(best);
        }
    }
}

void DBNSSolver::setManufacturedSource(const std::vector<StateVec> &src) {
    mesh_.requireStamp(meshStamp_);
    if (!src.empty() && src.size() != static_cast<size_t>(mesh_.nCells()))
        throw std::invalid_argument("MMS source requires one state per cell, or empty to clear");
    if (!settings_.turbulent)
        for (const auto &cell : src)
            if (cell[I_RHOK] != 0 || cell[I_RHOW] != 0)
                throw std::invalid_argument(
                    "laminar DBNS manufactured sources require zero k and omega rows");
    mmsSource_ = src;
}

void DBNSSolver::setBoundaryOverride(const std::vector<Primitive> &ghost) {
    mesh_.requireStamp(meshStamp_);
    if (!ghost.empty() &&
        ghost.size() != static_cast<size_t>(mesh_.nFaces() - mesh_.nInternalFaces()))
        throw std::invalid_argument(
            "ghost override requires one state per boundary face, or empty to clear");
    for (const auto &state : ghost)
        validatePrimitive(state, eos_, settings_.turbulent);
    bndOverride_ = ghost;
}

void DBNSSolver::initUniform(const Primitive &V) {
    mesh_.requireStamp(meshStamp_);
    validatePrimitive(V, eos_, settings_.turbulent);
    initFromField_ = false;
    for (int ci = 0; ci < mesh_.nCells(); ++ci)
        W_[ci] = GasState::toConserved(V, eos_);
    // scale-aware positivity floors (units differ by orders across the test
    // ladder: nondimensional shock tubes against dimensional wall flows)
    rhoFloor_ = 1e-6 * V.rho;
    pFloor_ = 1e-6 * V.p;
}

void DBNSSolver::initField(const std::vector<Primitive> &V) {
    mesh_.requireStamp(meshStamp_);
    if (V.size() != static_cast<size_t>(mesh_.nCells()))
        throw std::invalid_argument("initField requires exactly one primitive state per cell");
    for (const auto &state : V)
        validatePrimitive(state, eos_, settings_.turbulent);
    initFromField_ = true;
    double rhoMax = 0.0, pMax = 0.0;
    for (int ci = 0; ci < mesh_.nCells(); ++ci) {
        W_[ci] = GasState::toConserved(V[ci], eos_);
        rhoMax = std::max(rhoMax, V[ci].rho);
        pMax = std::max(pMax, V[ci].p);
    }
    rhoFloor_ = 1e-6 * rhoMax;
    pFloor_ = 1e-6 * pMax;
}

// --- properties: laminar viscosity (Sutherland) and SST eddy viscosity ------
void DBNSSolver::updateProperties() {
    updateLaminarProperties();
    updateTurbulentProperties();
}

void DBNSSolver::updateLaminarProperties() {
    for (int ci = 0; ci < mesh_.nCells(); ++ci) {
        Primitive V = GasState::toPrimitive(W_[ci], eos_);
        double T = GasState::temperature(V, eos_);
        muLam_[ci] = (settings_.constMu >= 0.0) ? settings_.constMu : eos_.viscosity(T);
    }
}

void DBNSSolver::updateTurbulentProperties() {
    double a1 = sstCoeffs_.a1, bStar = sstCoeffs_.betaStar;
    const auto &wallDist = wallDist_;
    for (int ci = 0; ci < mesh_.nCells(); ++ci) {
        Primitive V = GasState::toPrimitive(W_[ci], eos_);
        if (!settings_.turbulent) {
            muT_[ci] = 0.0;
            continue;
        }

        double nu = muLam_[ci] / V.rho;
        double k = std::max(V.k, settings_.kFloor);
        double w = std::max(V.omega, settings_.omegaFloor);
        double y = (ci < (int)wallDist.size()) ? std::max(wallDist[ci], 1e-9) : 1e-9;

        // strain-rate magnitude |S| = sqrt(2 S_ij S_ij)
        double dudx = grad_[ci][G_U].x, dudy = grad_[ci][G_U].y;
        double dvdx = grad_[ci][G_V].x, dvdy = grad_[ci][G_V].y;
        double S12 = 0.5 * (dudy + dvdx);
        double Smag = std::sqrt(2.0 * (dudx * dudx + dvdy * dvdy) + 4.0 * S12 * S12);

        // F2 blending (Menter 2003): arg2 = max(2 sqrt(k)/(beta* w y), 500 nu/(y^2 w))
        double arg2 = std::max(2.0 * std::sqrt(k) / (bStar * w * y), 500.0 * nu / (y * y * w));
        double F2 = std::tanh(arg2 * arg2);

        // Bradshaw shear-stress limited eddy viscosity: mu_t = rho a1 k/max(a1 w, S F2)
        double denom = std::max(a1 * w, Smag * F2);
        muT_[ci] = V.rho * a1 * k / std::max(denom, 1e-30);
        if (!std::isfinite(muT_[ci]) || muT_[ci] < 0.0)
            muT_[ci] = 0.0;
    }
}

// --- Green-Gauss cell gradients of the six primitive variables --------------
void DBNSSolver::computeGradients() {
    int nc = mesh_.nCells();
    for (int ci = 0; ci < nc; ++ci)
        for (int v = 0; v < 6; ++v)
            grad_[ci][v] = Vec3{};

    auto primVar = [&](const Primitive &V, int v) -> double {
        switch (v) {
        case G_RHO:
            return V.rho;
        case G_U:
            return V.u;
        case G_V:
            return V.v;
        case G_P:
            return V.p;
        case G_K:
            return V.k;
        default:
            return V.omega;
        }
    };

    int nIF = mesh_.nInternalFaces();
    // internal faces: contribute interpolated face value to owner (+) and neighbor (-)
    for (int fi = 0; fi < nIF; ++fi) {
        const Face &f = mesh_.face(fi);
        Primitive VP = GasState::toPrimitive(W_[f.owner], eos_);
        Primitive VN = GasState::toPrimitive(W_[f.neighbor], eos_);
        Vec3 Sf = f.normal * f.area;
        for (int v = 0; v < 6; ++v) {
            double phiF = primVar(VP, v) * f.weight + primVar(VN, v) * (1.0 - f.weight);
            grad_[f.owner][v] = grad_[f.owner][v] + Sf * phiF;
            grad_[f.neighbor][v] = grad_[f.neighbor][v] - Sf * phiF;
        }
    }
    // boundary faces: face value = 0.5 (interior + ghost)
    for (int fi = nIF; fi < mesh_.nFaces(); ++fi) {
        const Face &f = mesh_.face(fi);
        int bidx = fi - nIF;
        Primitive VP = GasState::toPrimitive(W_[f.owner], eos_);
        const Patch &pat = mesh_.patch(f.patchID);
        Primitive Vg = bcs_.has(pat.name) ? ghostState(VP, fi, bcs_.get(pat.name), bidx) : VP;
        Vec3 Sf = f.normal * f.area;
        for (int v = 0; v < 6; ++v) {
            double phiF = 0.5 * (primVar(VP, v) + primVar(Vg, v));
            grad_[f.owner][v] = grad_[f.owner][v] + Sf * phiF;
        }
    }
    for (int ci = 0; ci < nc; ++ci) {
        double V = mesh_.cell(ci).volume;
        if (V > 1e-30)
            for (int v = 0; v < 6; ++v)
                grad_[ci][v] = grad_[ci][v] / V;
    }
}

// --- Venkatakrishnan / Barth-Jespersen multidimensional limiter -------------
void DBNSSolver::computeLimiters() {
    int nc = mesh_.nCells();
    if (settings_.reconstructOrder < 2) {
        for (int ci = 0; ci < nc; ++ci)
            limiter_[ci].fill(0.0); // 1st order
        return;
    }
    if (!settings_.limitReconstruction) {
        for (int ci = 0; ci < nc; ++ci)
            limiter_[ci].fill(1.0); // unlimited (MMS)
        return;
    }
    auto primVar = [&](const Primitive &V, int v) -> double {
        switch (v) {
        case G_RHO:
            return V.rho;
        case G_U:
            return V.u;
        case G_V:
            return V.v;
        case G_P:
            return V.p;
        case G_K:
            return V.k;
        default:
            return V.omega;
        }
    };
    for (int ci = 0; ci < nc; ++ci) {
        Primitive VC = GasState::toPrimitive(W_[ci], eos_);
        double qC[6];
        for (int v = 0; v < 6; ++v)
            qC[v] = primVar(VC, v);
        double qMax[6], qMin[6];
        for (int v = 0; v < 6; ++v) {
            qMax[v] = qC[v];
            qMin[v] = qC[v];
        }
        // neighbour extrema
        for (FaceID fi : mesh_.cell(ci).faces) {
            const Face &f = mesh_.face(fi);
            int other = (f.owner == ci) ? f.neighbor : f.owner;
            if (other < 0)
                continue;
            Primitive VO = GasState::toPrimitive(W_[other], eos_);
            for (int v = 0; v < 6; ++v) {
                double q = primVar(VO, v);
                qMax[v] = std::max(qMax[v], q);
                qMin[v] = std::min(qMin[v], q);
            }
        }
        double phi[6];
        for (int v = 0; v < 6; ++v)
            phi[v] = 1.0;
        for (FaceID fi : mesh_.cell(ci).faces) {
            const Face &f = mesh_.face(fi);
            Vec3 xf = (f.owner == ci) ? f.center : f.center - f.shift;
            Vec3 dr = xf - mesh_.cell(ci).center;
            for (int v = 0; v < 6; ++v) {
                double d = grad_[ci][v].dot(dr); // unlimited increment to face
                double p = Limiters::venkat(d, qC[v], qMax[v], qMin[v], 0.0);
                phi[v] = std::min(phi[v], p);
            }
        }
        for (int v = 0; v < 6; ++v)
            limiter_[ci][v] = phi[v];
    }
}

// reconstruct primitive at face center xf from cell ci, with positivity guard
Primitive DBNSSolver::reconstruct(int ci, const Vec3 &xf) const {
    Primitive V = GasState::toPrimitive(W_[ci], eos_);
    if (settings_.reconstructOrder < 2)
        return V;
    Vec3 dr = xf - mesh_.cell(ci).center;
    Primitive R = V;
    R.rho = V.rho + limiter_[ci][G_RHO] * grad_[ci][G_RHO].dot(dr);
    R.u = V.u + limiter_[ci][G_U] * grad_[ci][G_U].dot(dr);
    R.v = V.v + limiter_[ci][G_V] * grad_[ci][G_V].dot(dr);
    R.p = V.p + limiter_[ci][G_P] * grad_[ci][G_P].dot(dr);
    R.k = V.k + limiter_[ci][G_K] * grad_[ci][G_K].dot(dr);
    R.omega = V.omega + limiter_[ci][G_W] * grad_[ci][G_W].dot(dr);
    // drop to first order if reconstruction left the physical set
    if (!(R.rho > 0.0) || !(R.p > 0.0))
        return V;
    if (R.k < 0.0)
        R.k = V.k;
    if (R.omega < settings_.omegaFloor)
        R.omega = V.omega;
    return R;
}

// the Menter omega wall value seen through the ghost: face value
// omega_w = 60 nu_w / (beta1 dy1^2) with dy1 twice the owner-cell wall
// distance, so the near-wall gradients feel the singular growth the omega
// equation cannot generate on its own
double DBNSSolver::wallOmegaGhost(const Primitive &in, int faceId) const {
    if (!settings_.turbulent)
        return in.omega;
    const Face &f = mesh_.face(faceId);
    double delta = std::max(f.delta, 1e-12);
    double nuW = muLam_[f.owner] / std::max(in.rho, 1e-30);
    double dy1 = 2.0 * delta;
    double omegaWall = 60.0 * nuW / (sstCoeffs_.beta1 * dy1 * dy1);
    return std::max(2.0 * omegaWall - in.omega, omegaWall);
}

// boundary ghost state ------------------------------------------------------
Primitive DBNSSolver::ghostState(const Primitive &in, int faceId, const BoundarySpec &spec,
                                 int boundaryIdx) const {
    if (!bndOverride_.empty())
        return bndOverride_[boundaryIdx];
    const Face &f = mesh_.face(faceId);
    double nx = f.normal.x, ny = f.normal.y;
    Primitive g = in;
    switch (spec.kind) {
    case BoundaryKind::SupersonicInflow:
    case BoundaryKind::FixedState:
        // spatially varying prescribed state (incoming-layer profile or
        // an imposed-shock top boundary) when the patch carries one
        if (!spec.profile.empty()) {
            int j = patchLocalIdx_[boundaryIdx];
            if (j < (int)spec.profile.size())
                return spec.profile[j];
        }
        return spec.freestream;
    case BoundaryKind::Extrapolate:
        return in;
    case BoundaryKind::SubsonicInflow: {
        double Tfs = spec.freestream.p / (spec.freestream.rho * eos_.R);
        g.u = spec.freestream.u;
        g.v = spec.freestream.v;
        g.k = spec.freestream.k;
        g.omega = spec.freestream.omega;
        g.p = in.p; // extrapolate pressure
        g.rho = g.p / (eos_.R * Tfs);
        return g;
    }
    case BoundaryKind::SubsonicOutflow: {
        g = in;
        g.p = spec.backPressure; // impose back pressure
        return g;
    }
    case BoundaryKind::SlipWall: {
        double un = in.u * nx + in.v * ny; // reflect normal velocity
        g.u = in.u - 2.0 * un * nx;
        g.v = in.v - 2.0 * un * ny;
        return g;
    }
    case BoundaryKind::NoSlipAdiabatic: {
        g.u = 2.0 * spec.wallVelocity.x - in.u; // face velocity = wallVelocity
        g.v = 2.0 * spec.wallVelocity.y - in.v;
        g.p = in.p;
        g.rho = in.rho; // dT/dn = 0
        g.k = -in.k;    // face k = 0
        g.omega = wallOmegaGhost(in, faceId);
        return g;
    }
    case BoundaryKind::NoSlipIsothermal: {
        g.u = 2.0 * spec.wallVelocity.x - in.u;
        g.v = 2.0 * spec.wallVelocity.y - in.v;
        double Tw = spec.wallTemp;
        if (!spec.wallTempProfile.empty()) {
            int j = patchLocalIdx_[boundaryIdx];
            if (j < (int)spec.wallTempProfile.size())
                Tw = spec.wallTempProfile[j];
        }
        double Tin = GasState::temperature(in, eos_);
        double Tg = 2.0 * Tw - Tin; // face T = Tw
        g.p = in.p;
        g.rho = g.p / (eos_.R * std::max(Tg, 1.0));
        g.k = -in.k;
        g.omega = wallOmegaGhost(in, faceId);
        return g;
    }
    }
    return in;
}

// --- viscous flux on an internal face (corrected face gradient) -------------
void DBNSSolver::addViscousFace(int faceId) {
    const Face &f = mesh_.face(faceId);
    int P = f.owner, N = f.neighbor;
    Primitive VP = GasState::toPrimitive(W_[P], eos_);
    Primitive VN = GasState::toPrimitive(W_[N], eos_);
    double nx = f.normal.x, ny = f.normal.y;

    Vec3 d = mesh_.cell(N).center + f.shift - mesh_.cell(P).center;
    double dmag = std::max(d.norm(), 1e-30);
    Vec3 dhat = d / dmag;

    // corrected face gradient: averaged cell gradient + normal-difference fix
    auto faceGrad = [&](int v, double phiP, double phiN) -> Vec3 {
        Vec3 gAvg = (grad_[P][v] + grad_[N][v]) * 0.5;
        double corr = (phiN - phiP) / dmag - gAvg.dot(dhat);
        return gAvg + dhat * corr;
    };
    Vec3 gU = faceGrad(G_U, VP.u, VN.u);
    Vec3 gV = faceGrad(G_V, VP.v, VN.v);
    double TP = GasState::temperature(VP, eos_), TN = GasState::temperature(VN, eos_);
    // Cell-centred T gradients are not stored, so use the wall-normal-consistent
    // difference estimate (dominant term for boundary-layer heat conduction).
    Vec3 gT = dhat * ((TN - TP) / dmag);

    double muP = muLam_[P] + muT_[P], muN = muLam_[N] + muT_[N];
    double muEff = 0.5 * (muP + muN);
    double muLamF = 0.5 * (muLam_[P] + muLam_[N]);
    double muTF = 0.5 * (muT_[P] + muT_[N]);

    double dudx = gU.x, dudy = gU.y, dvdx = gV.x, dvdy = gV.y;
    double div = dudx + dvdy;
    double tau_xx = muEff * (2.0 * dudx - 2.0 / 3.0 * div);
    double tau_yy = muEff * (2.0 * dvdy - 2.0 / 3.0 * div);
    double tau_xy = muEff * (dudy + dvdx);
    if (settings_.turbulent) {
        double rhok = 0.5 * (VP.rho * VP.k + VN.rho * VN.k);
        tau_xx -= 2.0 / 3.0 * rhok; // Boussinesq turbulent normal stress
        tau_yy -= 2.0 / 3.0 * rhok;
    }
    double uF = 0.5 * (VP.u + VN.u), vF = 0.5 * (VP.v + VN.v);
    double lamEff = eos_.Cp() * (muLamF / eos_.Pr + muTF / eos_.Pr_T);

    StateVec Fv{};
    Fv[I_RHOU] = tau_xx * nx + tau_xy * ny;
    Fv[I_RHOV] = tau_xy * nx + tau_yy * ny;
    double qn = -lamEff * gT.dot(f.normal); // q.n = -lambda dT/dn
    Fv[I_RHOE] = (tau_xx * uF + tau_xy * vF) * nx + (tau_xy * uF + tau_yy * vF) * ny - qn;
    if (settings_.turbulent) {
        Vec3 gK = faceGrad(G_K, VP.k, VN.k);
        Vec3 gW = faceGrad(G_W, VP.omega, VN.omega);
        double sk = sstCoeffs_.sigma_k1, sw = sstCoeffs_.sigma_w1; // near-wall blend dominant
        Fv[I_RHOK] = (muLamF + sk * muTF) * gK.dot(f.normal);
        Fv[I_RHOW] = (muLamF + sw * muTF) * gW.dot(f.normal);
    }
    double A = f.area;
    for (int i = 0; i < NVAR; ++i) {
        res_[P][i] -= Fv[i] * A; // viscous (diffusive) flux subtracts
        res_[N][i] += Fv[i] * A;
    }
}

// --- boundary face flux (convective + viscous wall treatment) ---------------
void DBNSSolver::addBoundaryFlux(int faceId, int patchIdx) {
    const Face &f = mesh_.face(faceId);
    int P = f.owner;
    int bidx = faceId - mesh_.nInternalFaces();
    double nx = f.normal.x, ny = f.normal.y, A = f.area;
    const Patch &pat = mesh_.patch(patchIdx);
    Primitive VP = GasState::toPrimitive(W_[P], eos_);

    if (!bcs_.has(pat.name)) { // default: extrapolate
        StateVec F = GasState::normalFlux(VP, nx, ny, eos_);
        for (int i = 0; i < NVAR; ++i)
            res_[P][i] += F[i] * A;
        return;
    }
    const BoundarySpec &spec = bcs_.get(pat.name);
    bool isWall =
        (spec.kind == BoundaryKind::SlipWall || spec.kind == BoundaryKind::NoSlipAdiabatic ||
         spec.kind == BoundaryKind::NoSlipIsothermal);

    if (isWall && bndOverride_.empty()) {
        // Inviscid wall flux: pressure only (zero mass flux through the wall).
        double pw = VP.p;
        res_[P][I_RHOU] += pw * nx * A;
        res_[P][I_RHOV] += pw * ny * A;
        if (spec.kind == BoundaryKind::SlipWall || !settings_.viscous)
            return;

        // No-slip viscous wall flux from one-sided near-wall gradients.
        // Transport coefficients at a resolved wall face are MOLECULAR: the
        // eddy viscosity vanishes at the wall, and the owner-cell muT would
        // overestimate the wall shear and heat flux. The difference is
        // immaterial on the reference meshes (muT/mu at the wall owners is of
        // order 1e-15 to 1e-12 at y+ 0.05), but the molecular value is the
        // consistent one and matches the pressure-based solvers'
        // resolved-wall momentum diffusion.
        double delta = std::max(f.delta, 1e-12);
        double muWall = muLam_[P];
        // velocity gradient along the OUTWARD normal (cell value to wall
        // value over delta), matching the energy line's dTdn convention;
        // tau.n_out is then the force per area the wall exerts on the cell:
        // a stationary wall decelerates, a moving wall drags toward its
        // speed. (An inward-gradient form would flip the sign of the wall
        // force.)
        Vec3 Uw = spec.wallVelocity;
        Vec3 dUdn{(Uw.x - VP.u) / delta, (Uw.y - VP.v) / delta, 0.0};
        double dUdn_n = dUdn.dot(f.normal);
        // tau.n = mu_wall [ dU/dn + 1/3 (dU/dn . n) n ]
        Vec3 taun = (dUdn + f.normal * (dUdn_n / 3.0)) * muWall;
        res_[P][I_RHOU] -= taun.x * A;
        res_[P][I_RHOV] -= taun.y * A;
        res_[P][I_RHOE] -= taun.dot(Uw) * A;
        // wall heat flux: q.n = -lambda dT/dn ; adiabatic -> 0
        if (spec.kind == BoundaryKind::NoSlipIsothermal) {
            double lamWall = eos_.Cp() * muWall / eos_.Pr;
            double Tin = GasState::temperature(VP, eos_);
            double Tw = spec.wallTemp;
            if (!spec.wallTempProfile.empty()) {
                int j = patchLocalIdx_[bidx];
                if (j < (int)spec.wallTempProfile.size())
                    Tw = spec.wallTempProfile[j];
            }
            double dTdn = (Tw - Tin) / delta; // (T_wall - T_cell)/delta
            double qn = -lamWall * dTdn;
            res_[P][I_RHOE] -= -qn * A; // energy viscous flux contribution
        }
        // turbulent transport wall fluxes. Without them the buffer layer is
        // starved: there is no k sink to the wall and no omega feed from its
        // wall value, so k and omega drift to an over-mixed state there.
        // One-sided diffusion with k_wall = 0 and the Menter omega
        // wall value omega_w = 60 nu_w / (beta1 dy1^2), dy1 = 2 delta (the
        // first-cell height); together with the sublayer cell floor this is
        // the standard low-Reynolds omega wall treatment. The diffusion
        // coefficients are molecular at the resolved wall face (same
        // convention and same immateriality measurement as the momentum and
        // heat fluxes above).
        if (settings_.turbulent) {
            double nuW = muLam_[P] / VP.rho;
            double dy1 = 2.0 * delta;
            double omegaWall = 60.0 * nuW / (sstCoeffs_.beta1 * dy1 * dy1);
            double fluxK = muLam_[P] * (0.0 - VP.k) / delta;
            double fluxW = muLam_[P] * (omegaWall - VP.omega) / delta;
            res_[P][I_RHOK] -= fluxK * A;
            res_[P][I_RHOW] -= fluxW * A;
        }
        return;
    }

    // open boundary (inflow / outflow / far-field / fixed): reconstruct interior,
    // build ghost, take HLLC; add viscous using ghost as the outer state.
    Primitive Vin = reconstruct(P, f.center);
    Primitive Vg = ghostState(Vin, faceId, spec, bidx);
    StateVec F = HLLCFlux::flux(Vin, Vg, nx, ny, eos_);
    for (int i = 0; i < NVAR; ++i)
        res_[P][i] += F[i] * A;

    if (settings_.viscous && spec.kind != BoundaryKind::Extrapolate) {
        // light viscous boundary contribution from one-sided gradient to ghost
        double delta = std::max(f.delta, 1e-12);
        double muEff = muLam_[P] + muT_[P];
        Vec3 dUdn{(Vg.u - VP.u) / delta, (Vg.v - VP.v) / delta, 0.0};
        double div = dUdn.x * nx + dUdn.y * ny;
        Vec3 taun{muEff * (dUdn.x), muEff * (dUdn.y), 0.0};
        taun = taun + f.normal * (muEff * div / 3.0);
        res_[P][I_RHOU] -= taun.x * A;
        res_[P][I_RHOV] -= taun.y * A;
    }
}

// --- turbulence sources (compressible SST) ----------------------------------
void DBNSSolver::addTurbulenceSources() {
    if (!settings_.turbulent)
        return;
    double bStar = sstCoeffs_.betaStar;
    const auto &wallDist = wallDist_;
    for (int ci = 0; ci < mesh_.nCells(); ++ci) {
        Primitive V = GasState::toPrimitive(W_[ci], eos_);
        double rho = V.rho;
        double k = std::max(V.k, settings_.kFloor);
        double w = std::max(V.omega, settings_.omegaFloor);
        double nu = muLam_[ci] / rho;
        double y = (ci < (int)wallDist.size()) ? std::max(wallDist[ci], 1e-9) : 1e-9;

        double dudx = grad_[ci][G_U].x, dudy = grad_[ci][G_U].y;
        double dvdx = grad_[ci][G_V].x, dvdy = grad_[ci][G_V].y;
        double S12 = 0.5 * (dudy + dvdx);
        double S2 = 2.0 * (dudx * dudx + dvdy * dvdy) + 4.0 * S12 * S12; // |S|^2

        // cross-diffusion CD = 2 rho sigma_w2 / w * gradK.gradOmega
        double gkgw = grad_[ci][G_K].dot(grad_[ci][G_W]);
        double CDkw = 2.0 * rho * sstCoeffs_.sigma_w2 / w * gkgw;
        double CDpos = std::max(CDkw, 1e-10);

        // F1 blending (Menter 2003)
        double a1arg = std::max(std::sqrt(k) / (bStar * w * y), 500.0 * nu / (y * y * w));
        double arg1 = std::min(a1arg, 4.0 * rho * sstCoeffs_.sigma_w2 * k / (CDpos * y * y));
        double F1 = std::tanh(arg1 * arg1 * arg1 * arg1);

        double alpha = sstCoeffs_.blend(sstCoeffs_.alpha1, sstCoeffs_.alpha2, F1);
        double beta = sstCoeffs_.blend(sstCoeffs_.beta1, sstCoeffs_.beta2, F1);

        double muT = muT_[ci];
        double Pk = muT * S2;                          // production
        Pk = std::min(Pk, 10.0 * bStar * rho * k * w); // Menter 2003 limiter

        // dilatational-dissipation (compressibility) correction
        double Mt2 = 2.0 * k / (eos_.gamma * V.p / rho); // M_t^2 = 2k/a^2
        double comprFactor = 1.0;
        if (settings_.compressibility == CompressibilityModel::Sarkar) {
            comprFactor = 1.0 + settings_.comprXiStar * Mt2;
        } else if (settings_.compressibility == CompressibilityModel::Zeman) {
            double Mt0 = settings_.turbMachCutoff;
            double F = std::max(0.0, Mt2 - Mt0 * Mt0);
            comprFactor = 1.0 + settings_.comprXiStar * F;
        }

        double Dk = bStar * rho * k * w * comprFactor; // k dissipation
        double sourceK = Pk - Dk;
        // omega production: alpha*rho*Pk_limited/muT = alpha*rho*min(S2, 10 bStar rho k w / muT)
        // (SST-2003 corrected form; alpha*rho*S2 is the paper's documented misprint, see the
        // NASA TMR SST page). Singular-safe: as muT -> 0 the min selects S2, the correct
        // limit; min(S2, lim) <= S2 so the corrected term is bounded by alpha*rho*S2.
        double lim = 10.0 * bStar * rho * k * w / std::max(muT, 1e-30);
        double PwSpec = std::min(S2, lim);
        if (!limiterActive_.empty())
            limiterActive_[ci] = (lim < S2) ? 1 : 0;
        double Pw = alpha * rho * PwSpec; // omega production
        double Dw = beta * rho * w * w;
        double crossW = 2.0 * (1.0 - F1) * rho * sstCoeffs_.sigma_w2 / w * gkgw;
        double sourceW = Pw - Dw + crossW;

        double Vol = mesh_.cell(ci).volume;
        res_[ci][I_RHOK] -= sourceK * Vol; // sources move to RHS of dW/dt=-res/V
        res_[ci][I_RHOW] -= sourceW * Vol;
    }
}

void DBNSSolver::computeResidual() {
    int nc = mesh_.nCells();
    for (int ci = 0; ci < nc; ++ci)
        res_[ci].fill(0.0);

    int nIF = mesh_.nInternalFaces();
    for (int fi = 0; fi < nIF; ++fi) {
        const Face &f = mesh_.face(fi);
        Primitive VL = reconstruct(f.owner, f.center);
        Primitive VR = reconstruct(f.neighbor, f.center - f.shift);
        StateVec F = HLLCFlux::flux(VL, VR, f.normal.x, f.normal.y, eos_);
        double A = f.area;
        for (int i = 0; i < NVAR; ++i) {
            res_[f.owner][i] += F[i] * A;
            res_[f.neighbor][i] -= F[i] * A;
        }
        if (settings_.viscous)
            addViscousFace(fi);
    }
    for (int fi = nIF; fi < mesh_.nFaces(); ++fi)
        addBoundaryFlux(fi, mesh_.face(fi).patchID);

    if (!dbTarget6_.empty())
        addInjectionFluxes();

    addTurbulenceSources();

    // manufactured-solution source: dW/dt = -(res - S V)/V
    if (!mmsSource_.empty())
        for (int ci = 0; ci < nc; ++ci) {
            double Vol = mesh_.cell(ci).volume;
            for (int i = 0; i < NVAR; ++i)
                res_[ci][i] -= mmsSource_[ci][i] * Vol;
        }
}

// --- model-form injection (the deferred-correction coupling) -----------------
void DBNSSolver::setTargetCorrection(const std::vector<double> &db6,
                                     const std::vector<double> &bDiag6,
                                     const std::vector<double> &dq2, bool energyReach,
                                     const std::vector<std::uint8_t> &mask) {
    mesh_.requireStamp(meshStamp_);
    int nc = mesh_.nCells();
    // the STORED discrepancy is the operative input: the injected flux is
    // -div(2 rho k db) with db fixed at conditioning time, so the
    // zero-correction case (db = 0) is exact BY CONSTRUCTION regardless of
    // any operator difference between the conditioning-side gradients and
    // this solver's Green-Gauss reconstruction (a target-minus-Boussinesq
    // form would break that identity at any nonzero mu_t). The absolute target enters
    // only the realizability diagnostics.
    if ((int)db6.size() != 6 * nc)
        throw std::invalid_argument("setTargetCorrection: db6 must be nCells*6");
    if ((int)bDiag6.size() != 6 * nc)
        throw std::invalid_argument("setTargetCorrection: bDiag6 must be "
                                    "nCells*6");
    if (!dq2.empty() && (int)dq2.size() != 2 * nc)
        throw std::invalid_argument("setTargetCorrection: dq2 must be nCells*2 "
                                    "or empty");
    if (!mask.empty() && mask.size() != static_cast<size_t>(nc))
        throw std::invalid_argument("setTargetCorrection: mask must be nCells or empty");
    for (const auto *values : {&db6, &bDiag6, &dq2})
        for (double value : *values)
            if (!std::isfinite(value))
                throw std::invalid_argument(
                    "setTargetCorrection: correction fields must be finite");
    injMask_ = mask;
    dbTarget6_ = db6;
    bTarget6_ = bDiag6;
    dqTarget2_ = dq2;
    injectEnergyReach_ = energyReach;
    injDiag_ = InjectionDiagnostics{};
    injDiag_.active = true;
    injRhoK0_.assign(nc, 0.0);
    for (int ci = 0; ci < nc; ++ci) {
        Primitive V = GasState::toPrimitive(W_[ci], eos_);
        injRhoK0_[ci] = V.rho * std::max(V.k, 0.0);
    }
    for (double v : dbTarget6_)
        injDiag_.maxDb = std::max(injDiag_.maxDb, std::abs(v));
    for (double v : dqTarget2_)
        injDiag_.maxDq = std::max(injDiag_.maxDq, std::abs(v));
}

std::vector<double> DBNSSolver::reconstructionLimiter() const {
    mesh_.requireStamp(meshStamp_);
    int nc = mesh_.nCells();
    std::vector<double> out(6 * nc);
    for (int ci = 0; ci < nc; ++ci)
        for (int v = 0; v < 6; ++v)
            out[6 * ci + v] = limiter_[ci][v];
    return out;
}

void DBNSSolver::setFrozenReconstructionLimiter(const std::vector<double> &lim6) {
    mesh_.requireStamp(meshStamp_);
    int nc = mesh_.nCells();
    if ((int)lim6.size() != 6 * nc)
        throw std::invalid_argument("setFrozenReconstructionLimiter: lim6 must "
                                    "be nCells*6");
    for (double value : lim6)
        if (!std::isfinite(value) || value < 0 || value > 1)
            throw std::invalid_argument(
                "setFrozenReconstructionLimiter: factors must be finite in [0, 1]");
    for (int ci = 0; ci < nc; ++ci)
        for (int v = 0; v < 6; ++v)
            limiter_[ci][v] = lim6[6 * ci + v];
    startLimiterFrozen_ = true;
}

void DBNSSolver::clearTargetCorrection() {
    dbTarget6_.clear();
    bTarget6_.clear();
    dqTarget2_.clear();
    injMask_.clear();
    injRhoK0_.clear();
    injDiag_ = InjectionDiagnostics{};
}

// Deferred-correction fluxes of the sampled closure, mirroring the
// incompressible injection and extending it to the energy equation (the
// heat-flux reach the pre-registered scheme names as the compressible
// novelty). Per cell the injected turbulent stress difference is
//   tau_inj = -(2 <rho> k b_target + 2 mu_t dev(S)),
// the difference between the target deviatoric Favre stress and the
// Boussinesq deviatoric stress the mu_t diffusion already carries (running
// rho, k, mu_t and strain, so a target equal to the solver's own Boussinesq
// anisotropy reproduces the baseline solve identically). It is assembled as
// a conservative internal-face flux (distance-weighted face values); wall
// and open boundary faces carry no injected flux (the near-wall rows have
// k ~ 0, the same negligible-contribution note as the incompressible
// scheme). With the energy reach on, the consistent stress work
// (tau_inj . u) . n and the injected turbulent heat flux
//   q_inj = <rho> cp dq_target
// (the deferred correction against the implicit gradient-diffusion term)
// enter the energy flux. The barycentric margin of the target is re-checked
// once per residual evaluation and violations are recorded, never silently
// projected (the caller projects before setting).
void DBNSSolver::addInjectionFluxes() {
    int nc = mesh_.nCells();
    injDiag_.checkedIters += 1;
    const bool masked = (int)injMask_.size() == nc;
    for (int ci = 0; ci < nc; ++ci) {
        // realizability is a property of ACTIVE cells; outside the
        // injection mask no correction flux exists (skipped below), so
        // they are not checked
        if (masked && !injMask_[ci])
            continue;
        // the EFFECTIVE RUNNING anisotropy the momentum equation carries:
        // b_eff = b_B(W) + db_stored with b_B = -mu_t/(rho k) dev(S) from
        // the solver's own current gradients, eddy viscosity and state
        // (the same floored k the mu_t formula used), rather than the frozen
        // absolute target. Diagnostic only: the
        // margin is recorded with its residual-evaluation ordinal and cell,
        // never projected or clipped.
        Primitive Vd = GasState::toPrimitive(W_[ci], eos_);
        double kd = std::max(Vd.k, settings_.kFloor);
        double coef = -muT_[ci] / (Vd.rho * kd);
        double dudx = grad_[ci][G_U].x, dudy = grad_[ci][G_U].y;
        double dvdx = grad_[ci][G_V].x, dvdy = grad_[ci][G_V].y;
        double div3 = (dudx + dvdy) / 3.0;
        const double *dbc = dbTarget6_.data() + 6 * ci;
        double beff[6];
        beff[0] = coef * (dudx - div3) + dbc[0];
        beff[1] = coef * (dvdy - div3) + dbc[1];
        beff[2] = coef * (-div3) + dbc[2];
        beff[3] = coef * 0.5 * (dudy + dvdx) + dbc[3];
        beff[4] = dbc[4];
        beff[5] = dbc[5];
        double margin = aniso::barycentricMargin(beff);
        if (margin < injDiag_.minMargin) {
            injDiag_.minMargin = margin;
            injDiag_.minMarginIter = injDiag_.checkedIters;
            injDiag_.minMarginCell = ci;
        }
        const double violation = aniso::realizabilityViolation(beff);
        if (violation > 1e-9) {
            injDiag_.allRealizable = false;
            if (violation > injDiag_.maxViolation) {
                injDiag_.maxViolation = violation;
                injDiag_.maxViolationIter = injDiag_.checkedIters;
                injDiag_.maxViolationCell = ci;
            }
        }
    }

    // per-cell injected stress difference tau_inj (xx, yy, xy needed for the
    // 2-D fluxes; zz never crosses a face) and heat-flux vector rho cp dq
    auto &txx = correctionXX_;
    auto &tyy = correctionYY_;
    auto &txy = correctionXY_;
    auto &qx = correctionQx_;
    auto &qy = correctionQy_;
    txx.resize(nc);
    tyy.resize(nc);
    txy.resize(nc);
    qx.assign(nc, 0.0);
    qy.assign(nc, 0.0);
    bool withHeat = injectEnergyReach_ && !dqTarget2_.empty();
    double cp = eos_.Cp();
    for (int ci = 0; ci < nc; ++ci) {
        Primitive V = GasState::toPrimitive(W_[ci], eos_);
        double rho = V.rho;
        double k = std::max(V.k, 0.0);
        // the injected stress difference is 2 rho k db with the STORED
        // discrepancy: db = 0 gives exactly zero flux at every state (the
        // discrete zero-correction contract), the registered form rides the
        // running rho k, and the frozen-k variant is the pure difference
        // 2 (rho k)_0 db with the conditioning-time scale
        double rhok = settings_.injectionFrozenK ? injRhoK0_[ci] : rho * k;
        const double *db = dbTarget6_.data() + 6 * ci;
        // db ordered xx, yy, zz, xy, xz, yz; only xx, yy, xy enter the fluxes
        txx[ci] = -2.0 * rhok * db[0];
        tyy[ci] = -2.0 * rhok * db[1];
        txy[ci] = -2.0 * rhok * db[3];
        if (withHeat) {
            qx[ci] = rho * cp * dqTarget2_[2 * ci];
            qy[ci] = rho * cp * dqTarget2_[2 * ci + 1];
        }
    }

    int nIF = mesh_.nInternalFaces();
    for (int fi = 0; fi < nIF; ++fi) {
        const Face &f = mesh_.face(fi);
        int P = f.owner, N = f.neighbor;
        // the solver-side injection mask: correction fluxes exist only on
        // faces INTERIOR to the active region (both cells active). A flux
        // either crosses a face fully or not at all, so the correction ends
        // conservatively at the mask boundary and cells outside the training
        // support receive no injection as the member state evolves (the
        // python-side baseline placeholder only cancelled at the initial
        // state; this gate makes the inactivity exact for all iterations).
        if (masked && (!injMask_[P] || !injMask_[N]))
            continue;
        double w = f.weight;
        double fxx = w * txx[P] + (1.0 - w) * txx[N];
        double fyy = w * tyy[P] + (1.0 - w) * tyy[N];
        double fxy = w * txy[P] + (1.0 - w) * txy[N];
        double nx = f.normal.x, ny = f.normal.y, A = f.area;
        StateVec Fv{};
        Fv[I_RHOU] = fxx * nx + fxy * ny;
        Fv[I_RHOV] = fxy * nx + fyy * ny;
        if (injectEnergyReach_) {
            Primitive VP = GasState::toPrimitive(W_[P], eos_);
            Primitive VN = GasState::toPrimitive(W_[N], eos_);
            double uF = 0.5 * (VP.u + VN.u), vF = 0.5 * (VP.v + VN.v);
            Fv[I_RHOE] = (fxx * uF + fxy * vF) * nx + (fxy * uF + fyy * vF) * ny;
            if (withHeat) {
                double qnx = w * qx[P] + (1.0 - w) * qx[N];
                double qny = w * qy[P] + (1.0 - w) * qy[N];
                Fv[I_RHOE] -= qnx * nx + qny * ny;
            }
        }
        for (int i = 0; i < NVAR; ++i) {
            // same sign convention as the viscous flux; the ramp factor
            // scales the whole correction during the configured window
            res_[P][i] -= injRampScale_ * Fv[i] * A;
            res_[N][i] += injRampScale_ * Fv[i] * A;
        }
    }
}

void DBNSSolver::computeTimeStep(double cfl, bool includeViscous) {
    // includeViscous: the explicit march must respect the viscous stability
    // limit (spectral radius ~ 1/dy^2, the documented wall-clustered-mesh
    // blocker); the implicit march drops it from the STEP SIZE and absorbs
    // the stiffness in its diagonal instead, which is the whole point of the
    // implicit driver (a viscous-limited pseudo-time step would freeze the
    // wall cells at any Courant number).
    for (int ci = 0; ci < mesh_.nCells(); ++ci) {
        Primitive V = GasState::toPrimitive(W_[ci], eos_);
        double a = GasState::soundSpeed(V, eos_);
        double lamC = 0.0, lamV = 0.0;
        double Vol = mesh_.cell(ci).volume;
        for (FaceID fi : mesh_.cell(ci).faces) {
            const Face &f = mesh_.face(fi);
            double sgn = (f.owner == ci) ? 1.0 : -1.0;
            double un = (V.u * f.normal.x + V.v * f.normal.y) * sgn;
            lamC += (std::abs(un) + a) * f.area;
            if (settings_.viscous && includeViscous) {
                double muEff = muLam_[ci] + muT_[ci];
                double diff = std::max(4.0 / 3.0, eos_.gamma / eos_.Pr) * muEff / V.rho;
                lamV += diff * f.area * f.area / std::max(Vol, 1e-30);
            }
        }
        dtCell_[ci] = cfl * Vol / std::max(lamC + 2.0 * lamV, 1e-30);
    }
}

// face spectral radii for the LU-SGS diagonal and off-diagonal relaxation:
// convective |u.n| + a from the larger side, viscous mirroring the explicit
// stability estimate (diff A / V), both entering as lam with velocity units
// so that lam * A matches the V/dt aggregate.
void DBNSSolver::computeFaceSpectralRadii() {
    int nF = mesh_.nFaces();
    if ((int)lamFace_.size() != nF)
        lamFace_.assign(nF, 0.0);
    double diffFac = std::max(4.0 / 3.0, eos_.gamma / eos_.Pr);
    for (int fi = 0; fi < nF; ++fi) {
        const Face &f = mesh_.face(fi);
        Primitive VP = GasState::toPrimitive(W_[f.owner], eos_);
        double unP = VP.u * f.normal.x + VP.v * f.normal.y;
        double lam = std::abs(unP) + GasState::soundSpeed(VP, eos_);
        double nuP = (muLam_[f.owner] + muT_[f.owner]) / VP.rho;
        double volP = mesh_.cell(f.owner).volume;
        double nu = nuP, vol = volP;
        if (f.neighbor >= 0) {
            Primitive VN = GasState::toPrimitive(W_[f.neighbor], eos_);
            double unN = VN.u * f.normal.x + VN.v * f.normal.y;
            lam = std::max(lam, std::abs(unN) + GasState::soundSpeed(VN, eos_));
            double nuN = (muLam_[f.neighbor] + muT_[f.neighbor]) / VN.rho;
            nu = 0.5 * (nuP + nuN);
            vol = 0.5 * (volP + mesh_.cell(f.neighbor).volume);
        }
        if (settings_.viscous)
            lam += 2.0 * diffFac * nu * f.area / std::max(vol, 1e-30);
        lamFace_[fi] = lam;
    }
}

std::array<double, NVAR> DBNSSolver::equationResidualNorms() const {
    std::array<double, NVAR> s{};
    for (int ci = 0; ci < mesh_.nCells(); ++ci)
        for (int v = 0; v < NVAR; ++v) {
            double r = res_[ci][v] / std::max(mesh_.cell(ci).volume, 1e-30);
            s[v] += r * r;
        }
    for (int v = 0; v < NVAR; ++v)
        s[v] = std::sqrt(s[v] / std::max(1, mesh_.nCells()));
    return s;
}

bool DBNSSolver::stateIsValid() const {
    for (int ci = 0; ci < mesh_.nCells(); ++ci) {
        for (int v = 0; v < NVAR; ++v)
            if (!std::isfinite(W_[ci][v]))
                return false;
        if (!GasState::admissible(W_[ci], eos_))
            return false;
    }
    return true;
}

double DBNSSolver::rhoResidualNorm() const {
    double s = 0.0;
    for (int ci = 0; ci < mesh_.nCells(); ++ci) {
        double r = res_[ci][I_RHO] / std::max(mesh_.cell(ci).volume, 1e-30);
        s += r * r;
    }
    return std::sqrt(s / std::max(1, mesh_.nCells()));
}

SolveReport DBNSSolver::solve() {
    mesh_.requireStamp(meshStamp_);
    if (settings_.timeMode == TimeMode::Steady && settings_.implicitSteady)
        return solveImplicitSteady();
    if (settings_.frozenMeanFlow)
        throw std::runtime_error("frozenMeanFlow requires the implicit "
                                 "steady driver");

    SolveReport rep;
    const int nc = mesh_.nCells();
    if (!stateIsValid()) {
        rep.status = EvaluationStatus::Diverged;
        return rep;
    }
    if (settings_.timeMode == TimeMode::Unsteady && settings_.tEnd == 0.0) {
        prepareProperties();
        rep.status = EvaluationStatus::Converged;
        return rep;
    }

    // Fixed characteristic scales from the initial state, independent of CFL
    // and of the initial residual. Each equation has its own physical units.
    double rhoRef = 0.0, speedRef = 0.0, volume = 0.0;
    double kRef = settings_.kFloor, wRef = settings_.omegaFloor;
    for (int ci = 0; ci < nc; ++ci) {
        const auto v = GasState::toPrimitive(W_[ci], eos_);
        rhoRef = std::max(rhoRef, v.rho);
        speedRef = std::max(speedRef, GasState::soundSpeed(v, eos_) + std::hypot(v.u, v.v));
        kRef = std::max(kRef, v.k);
        wRef = std::max(wRef, v.omega);
        volume += mesh_.cell(ci).volume;
    }
    const double timeRef = std::sqrt(volume) / speedRef;
    const StateVec scales{
        rhoRef,        rhoRef * speedRef, rhoRef * speedRef, rhoRef * speedRef * speedRef,
        rhoRef * kRef, rhoRef * wRef};
    std::vector<double> wallDelta(nc, 0.0);
    for (size_t j = 0; j < wallAdjCell_.size(); ++j)
        wallDelta[wallAdjCell_[j]] = wallAdjDelta_[j];
    auto steadyMeasure = [&]() {
        double measure = 0.0;
        for (int ci = 0; ci < nc; ++ci) {
            for (int v = 0; v < (settings_.turbulent ? NVAR : 4); ++v) {
                double r = res_[ci][v] * timeRef / (mesh_.cell(ci).volume * scales[v]);
                if (v == I_RHOW && wallDelta[ci] > 0.0) {
                    const auto state = GasState::toPrimitive(W_[ci], eos_);
                    const double g = 60.0 * muLam_[ci] /
                                     (state.rho * sstCoeffs_.beta1 * wallDelta[ci] * wallDelta[ci]);
                    r = (state.omega - g) / std::max(wRef, g);
                }
                if (!std::isfinite(r))
                    return std::numeric_limits<double>::infinity();
                measure = std::max(measure, std::abs(r));
            }
        }
        return measure;
    };
    // Boundary omega needs current molecular viscosity before gradients;
    // eddy viscosity then needs those gradients. Repeat at every RK state.
    auto evalSpatial = [&]() {
        prepareProperties();
        computeLimiters();
        computeResidual();
    };
    auto finishStage = [&]() {
        // A pressure floor must never turn NaN energy into an accepted state.
        for (const auto &state : W_)
            for (double x : state)
                if (!std::isfinite(x))
                    return false;
        updateLaminarProperties();
        clampPositivity();
        return stateIsValid();
    };
    std::vector<StateVec> W0(nc);
    double t = 0.0, densityRef = -1.0;
    for (int iter = 0; iter < settings_.maxIterations; ++iter) {
        evalSpatial();
        computeTimeStep(settings_.cfl, true);
        double dtGlobal = std::numeric_limits<double>::infinity();
        if (settings_.timeMode == TimeMode::Unsteady) {
            for (double dt : dtCell_)
                dtGlobal = std::min(dtGlobal, dt);
            dtGlobal = std::min(dtGlobal, settings_.tEnd - t);
            if (!(dtGlobal > 0.0) || !std::isfinite(dtGlobal)) {
                rep.status = EvaluationStatus::Diverged;
                break;
            }
        }
        auto dtOf = [&](int ci) {
            return settings_.timeMode == TimeMode::Unsteady ? dtGlobal : dtCell_[ci];
        };
        W0 = W_;
        const int stages = settings_.rkStages >= 3 ? 3 : 1;
        bool valid = true;
        for (int stage = 0; stage < stages; ++stage) {
            if (stage > 0)
                evalSpatial();
            const double oldWeight = stage == 0 ? 0.0 : (stage == 1 ? 0.75 : 1.0 / 3.0);
            const double stepWeight = stage == 0 ? 1.0 : (stage == 1 ? 0.25 : 2.0 / 3.0);
            for (int ci = 0; ci < nc; ++ci) {
                const double f = dtOf(ci) / mesh_.cell(ci).volume;
                for (int v = 0; v < NVAR; ++v)
                    W_[ci][v] = oldWeight * W0[ci][v] + stepWeight * (W_[ci][v] - f * res_[ci][v]);
            }
            if (!finishStage()) {
                valid = false;
                break;
            }
        }
        if (!valid) {
            rep.status = EvaluationStatus::Diverged;
            break;
        }
        rep.iterations = iter + 1;
        if (settings_.timeMode == TimeMode::Steady)
            evalSpatial();
        const double rn = rhoResidualNorm();
        if (densityRef < 0.0)
            densityRef = std::max(rn, 1e-30);
        rep.finalResidual =
            settings_.timeMode == TimeMode::Steady ? steadyMeasure() : rn / densityRef;
        if (settings_.verbose && iter % settings_.reportInterval == 0)
            std::printf("  iter %6d  res_rho %.3e  (measure %.3e)\n", iter, rn, rep.finalResidual);
        if (iter % settings_.reportInterval == 0)
            rep.residualHistory.push_back(rn);
        if (!std::isfinite(rep.finalResidual) || (settings_.timeMode == TimeMode::Steady &&
                                                  rep.finalResidual > settings_.divergenceLimit)) {
            rep.status = EvaluationStatus::Diverged;
            break;
        }
        if (settings_.timeMode == TimeMode::Unsteady) {
            t += dtGlobal;
            if (t >= settings_.tEnd) {
                rep.status = EvaluationStatus::Converged;
                break;
            }
        } else if (rep.finalResidual < settings_.convergenceTol) {
            rep.status = EvaluationStatus::Converged;
            break;
        }
    }
    if (rep.status == EvaluationStatus::Unknown)
        rep.status = EvaluationStatus::Unconverged;
    rep.tFinal = t;
    if (rep.status != EvaluationStatus::Diverged)
        prepareProperties();
    return rep;
}

// --- implicit (LU-SGS) steady march ------------------------------------------
// Backward-Euler pseudo-time on the second-order residual:
//   (V/dt) dW + dRes(dW) = -Res(W^n),
// with the face-flux Jacobian replaced by its spectral-radius-split
// first-order (Rusanov) form (Yoon and Jameson 1988). Per face between i and
// its neighbor j (outward normal n from i):
//   dRes_i ~= 0.5 A [ dFn(W_i) + lam I ] dW_i + 0.5 A [ dFn(W_j) - lam I ] dW_j,
// the diagonal bounded by its spectral radius (drop dFn(W_i), keep lam), the
// off-diagonal applied matrix-free as Fn(W_j + dW_j) - Fn(W_j). Two sweeps:
//   forward   dW*_i = D^-1 ( -Res_i - sum_{j<i} c_ij(dW*_j) )
//   backward  dW_i  = dW*_i - D^-1 sum_{j>i} c_ij(dW_j)
// with c_ij(dW_j) = 0.5 A ( Fn(W_j+dW_j) - Fn(W_j) - lam dW_j ) and
//   D_i = V/dt + 0.5 sum_f lam_f A_f  (+ the point-implicit SST destruction
// on the k and omega rows: d(beta* rho k w)/d(rho k) = beta* w and
// d(beta rho w^2)/d(rho w) = 2 beta w, with beta bounded by max(beta1, beta2)
// so the diagonal over-estimates the blended value, which only stabilizes).
// Boundary faces couple to no unknown neighbor; their lam enters the diagonal.
SolveReport DBNSSolver::solveImplicitSteady() {
    SolveReport rep;
    int nc = mesh_.nCells();
    double res0 = -1.0;
    std::array<double, NVAR> eqRes0{};
    // carried turbulence field-change norms (never fresh zeros): a rejected
    // step keeps the previous accepted step's values
    double lastKChange = 1.0, lastOmChange = 1.0;
    if ((int)dW_.size() != nc)
        dW_.assign(nc, StateVec{});

    // frozen-mean transport mode: capture the primitive mean state once and
    // re-pin it after every update, so only k and omega march (the
    // registered constitutive-error-at-matched-mean baseline). Pinning the
    // PRIMITIVE mean (not the conserved rows) keeps the pressure exactly at
    // the captured value as rho k evolves inside the total energy.
    std::vector<std::array<double, 4>> frozenMean;
    if (settings_.frozenMeanFlow) {
        frozenMean.resize(nc);
        for (int ci = 0; ci < nc; ++ci) {
            Primitive V = GasState::toPrimitive(W_[ci], eos_);
            frozenMean[ci] = {V.rho, V.u, V.v, V.p};
        }
    }

    double betaMax = std::max(sstCoeffs_.beta1, sstCoeffs_.beta2);
    double bStar = sstCoeffs_.betaStar;

    // first-evaluation gradient initialization for prescribed-field
    // starts (see the explicit driver's note and initFromField_)
    if (initFromField_)
        computeGradients();

    // matrix-free off-diagonal action of neighbor `other` on cell ci
    auto neighborTerm = [&](int ci, int other, FaceID fi) -> StateVec {
        const Face &f = mesh_.face(fi);
        double sgn = (f.owner == ci) ? 1.0 : -1.0;
        double nx = f.normal.x * sgn, ny = f.normal.y * sgn;
        double A = f.area, lam = lamFace_[fi];
        StateVec Wp;
        for (int i = 0; i < NVAR; ++i)
            Wp[i] = W_[other][i] + dW_[other][i];
        StateVec out{};
        if (GasState::admissible(Wp, eos_)) {
            Primitive V0 = GasState::toPrimitive(W_[other], eos_);
            Primitive V1 = GasState::toPrimitive(Wp, eos_);
            StateVec F0 = GasState::normalFlux(V0, nx, ny, eos_);
            StateVec F1 = GasState::normalFlux(V1, nx, ny, eos_);
            for (int i = 0; i < NVAR; ++i)
                out[i] = 0.5 * A * (F1[i] - F0[i] - lam * dW_[other][i]);
        } else {
            // inadmissible perturbed state: keep only the relaxation part
            for (int i = 0; i < NVAR; ++i)
                out[i] = -0.5 * A * lam * dW_[other][i];
        }
        return out;
    };

    std::vector<StateVec> D(nc);
    std::vector<StateVec> Wsave(nc);
    // step-rejection backoff: an inadmissible update is rejected (state
    // restored) and the CFL scaled down; it recovers multiplicatively on
    // success. Persistent rejection classifies as Diverged.
    double cflScale = 1.0;
    int rejectCount = 0;
    double lastCriterion = 1.0;
    int consecutiveRejects = 0;
    bool quiescentConverged = false;
    // the uniform initial state has a machine-zero density residual, so the
    // relative-convergence normalizer is the residual maximum over the ramp
    // (frozen afterwards), not the first iterate
    // the reference window covers BOTH ramps so the frozen injection
    // response is the full-force response, not a partially ramped one
    int freezeIter = std::max({settings_.cflRampIters, settings_.injectionRampIters, 50});
    // A restored checkpoint explicitly requests the frozen operator.
    // Automatic freezing is an opt-in kept for reproducing older results: it
    // can change the converged answer substantially, so ordinary solves
    // refresh limiters.
    bool limiterFrozen = startLimiterFrozen_;
    // carried mean-flow field-change norms of the last accepted step (the
    // quiescence route below)
    double lastMeanChange = 1.0;
    int iter = 0;
    for (; iter < settings_.maxIterations; ++iter) {
        // linear CFL ramp: low while the transient is strong, then the target
        double frac = (settings_.cflRampIters > 0)
                          ? std::min(1.0, double(iter) / settings_.cflRampIters)
                          : 1.0;
        double cfl = cflScale * (settings_.cflRampStart +
                                 frac * (settings_.cflImplicit - settings_.cflRampStart));

        updateProperties();
        computeTimeStep(cfl, false); // convective step; viscosity in the diagonal
        computeGradients();
        if (!limiterFrozen)
            computeLimiters();
        injRampScale_ = (settings_.injectionRampIters > 0)
                            ? std::min(1.0, double(iter + 1) / settings_.injectionRampIters)
                            : 1.0;
        computeResidual();
        computeFaceSpectralRadii();

        for (int ci = 0; ci < nc; ++ci) {
            double sumLam = 0.0;
            for (FaceID fi : mesh_.cell(ci).faces) {
                const Face &f = mesh_.face(fi);
                // boundary ghosts move with the interior state, so their
                // Jacobian falls fully on this cell: full weight there,
                // half on internal faces (the split Rusanov diagonal)
                double weight = (f.neighbor < 0) ? 1.0 : 0.5;
                sumLam += weight * lamFace_[fi] * f.area;
            }
            double base = mesh_.cell(ci).volume / std::max(dtCell_[ci], 1e-30) + sumLam;
            for (int i = 0; i < NVAR; ++i)
                D[ci][i] = base;
            if (settings_.turbulent) {
                Primitive V = GasState::toPrimitive(W_[ci], eos_);
                double w = std::max(V.omega, settings_.omegaFloor);
                D[ci][I_RHOK] += mesh_.cell(ci).volume * bStar * w;
                D[ci][I_RHOW] += mesh_.cell(ci).volume * 2.0 * betaMax * w;
            }
        }

        // forward sweep (lower neighbors already updated this sweep).
        // Frozen-mean mode zeroes the mean-flow increment rows THE MOMENT
        // each cell's increment exists: the neighbor terms consume the full
        // six-component dW of already-swept cells, so a nonzero phantom
        // mean increment (the pinned mean's residual never vanishes) would
        // otherwise contaminate the k and omega coupling even though the
        // final mean is re-pinned after the update; with the rows zeroed the turbulence
        // increments solve the genuine frozen-mean transport equations.
        for (int ci = 0; ci < nc; ++ci) {
            StateVec rhs;
            for (int i = 0; i < NVAR; ++i)
                rhs[i] = -res_[ci][i];
            for (FaceID fi : mesh_.cell(ci).faces) {
                const Face &f = mesh_.face(fi);
                if (f.neighbor < 0)
                    continue;
                int other = (f.owner == ci) ? f.neighbor : f.owner;
                if (other >= ci)
                    continue;
                StateVec c = neighborTerm(ci, other, fi);
                for (int i = 0; i < NVAR; ++i)
                    rhs[i] -= c[i];
            }
            for (int i = 0; i < NVAR; ++i)
                dW_[ci][i] = rhs[i] / D[ci][i];
            if (settings_.frozenMeanFlow)
                for (int i = 0; i < 4; ++i)
                    dW_[ci][i] = 0.0;
        }
        // backward sweep (upper neighbors carry their final increments)
        for (int ci = nc - 1; ci >= 0; --ci) {
            StateVec corr{};
            for (FaceID fi : mesh_.cell(ci).faces) {
                const Face &f = mesh_.face(fi);
                if (f.neighbor < 0)
                    continue;
                int other = (f.owner == ci) ? f.neighbor : f.owner;
                if (other <= ci)
                    continue;
                StateVec c = neighborTerm(ci, other, fi);
                for (int i = 0; i < NVAR; ++i)
                    corr[i] += c[i];
            }
            for (int i = 0; i < NVAR; ++i)
                dW_[ci][i] -= corr[i] / D[ci][i];
            if (settings_.frozenMeanFlow)
                for (int i = 0; i < 4; ++i)
                    dW_[ci][i] = 0.0;
        }

        Wsave = W_;
        for (int ci = 0; ci < nc; ++ci)
            for (int i = 0; i < NVAR; ++i)
                W_[ci][i] += dW_[ci][i];
        if (settings_.frozenMeanFlow)
            for (int ci = 0; ci < nc; ++ci) {
                Primitive V = GasState::toPrimitive(W_[ci], eos_);
                V.rho = frozenMean[ci][0];
                V.u = frozenMean[ci][1];
                V.v = frozenMean[ci][2];
                V.p = frozenMean[ci][3];
                if (V.k < 0.0)
                    V.k = settings_.kFloor;
                W_[ci] = GasState::toConserved(V, eos_);
            }

        bool admissible = true;
        int badCell = -1;
        for (int ci = 0; ci < nc; ++ci)
            if (!GasState::admissible(W_[ci], eos_)) {
                admissible = false;
                badCell = ci;
                break;
            }
        if (!admissible) {
            if (settings_.verbose && (rejectCount % 500 == 0)) {
                // the failing cell's turbulence and energy budget: the
                // limiter branch, the specific production against its cap,
                // destruction, and the internal-energy margin that a
                // runaway k growth would consume
                const Primitive Vb = GasState::toPrimitive(Wsave[badCell], eos_);
                const auto &c = mesh_.cell(badCell).center;
                double dudx = grad_[badCell][G_U].x;
                double dudy = grad_[badCell][G_U].y;
                double dvdx = grad_[badCell][G_V].x;
                double dvdy = grad_[badCell][G_V].y;
                double S12 = 0.5 * (dudy + dvdx);
                double S2 = 2.0 * (dudx * dudx + dvdy * dvdy) + 4.0 * S12 * S12;
                double kk = std::max(Vb.k, 0.0);
                double ww = std::max(Vb.omega, 1e-30);
                double Pk = std::min(muT_[badCell] * S2, 10.0 * 0.09 * Vb.rho * kk * ww);
                double Dk = 0.09 * Vb.rho * kk * ww;
                double eInt = Wsave[badCell][I_RHOE] - 0.5 * Vb.rho * (Vb.u * Vb.u + Vb.v * Vb.v) -
                              Vb.rho * kk;
                double tauInj = 0.0;
                if (!dbTarget6_.empty()) {
                    const double *dbc = dbTarget6_.data() + 6 * badCell;
                    double rk = settings_.injectionFrozenK ? injRhoK0_[badCell] : Vb.rho * kk;
                    tauInj =
                        2.0 * rk * std::max({std::abs(dbc[0]), std::abs(dbc[1]), std::abs(dbc[3])});
                }
                int lim = (badCell < (int)limiterActive_.size()) ? limiterActive_[badCell] : -1;
                std::printf("  [reject %d] iter %d cell %d (x %.4g, y %.4g)"
                            "  rho %.3e p %.3e k %.3e om %.3e lim %d"
                            "  Pk %.3e Dk %.3e eInt %.3e tauInj %.3e\n",
                            rejectCount, iter, badCell, c.x, c.y, Vb.rho, Vb.p, Vb.k, Vb.omega, lim,
                            Pk, Dk, eInt, tauInj);
            }
            ++rejectCount;
            W_ = Wsave; // reject the step, back the CFL off
            cflScale *= 0.5;
            if (++consecutiveRejects > 20) {
                rep.status = EvaluationStatus::Diverged;
                rep.iterations = iter;
                // the last accepted step's criterion value: a divergence
                // record must not carry a falsely excellent default zero
                rep.finalResidual = lastCriterion;
                return rep;
            }
            continue;
        }
        consecutiveRejects = 0;
        cflScale = std::min(1.0, cflScale * 1.5);
        clampPositivity();

        const std::array<double, NVAR> rns = equationResidualNorms();
        double rn = rns[I_RHO];
        // Convergence judges EVERY solved equation, split by semantics as in
        // the compressible SIMPLE criterion: the mean-flow equations
        // (commensurate conservative scales) gate on relative residual decay,
        // while k and omega gate on dimensionless field-change norms. The
        // omega equation's volume-scaled residual RMS is dominated by
        // wall-adjacent cells whose omega ~ nu / y^2 is re-pinned every
        // iteration, a scale many orders above the mean flow; pooling it into
        // the reduction references floors every other equation into
        // quiescence and reduces the criterion to an omega decay it can never
        // deliver from a wall-slaved scale. The field-change norm measures
        // what a steady solve actually owes: the turbulence state stopped
        // moving (the update through the destruction-stiff diagonal, and the
        // re-pinned wall rows read exactly stationary). Frozen-mean mode
        // never solves the mean-flow equations, so only the turbulence
        // field-change norms gate there.
        const int nEq = settings_.frozenMeanFlow ? 0 : 4;
        if (iter <= freezeIter || res0 < 0.0) {
            res0 = std::max(res0, std::max(rn, 1e-30));
            for (int v = 0; v < nEq; ++v)
                eqRes0[v] = std::max(eqRes0[v], std::max(rns[v], 1e-30));
        }
        // an equation whose reference residual is negligible against the
        // system scale (a quiescent direction) must not gate convergence on
        // round-off; its reference is floored at 1e-6 of the largest
        double eqRefMax = 1e-30;
        for (int v = 0; v < nEq; ++v)
            eqRefMax = std::max(eqRefMax, eqRes0[v]);
        double relMax = 0.0;
        for (int v = 0; v < nEq; ++v)
            relMax = std::max(relMax, rns[v] / std::max(eqRes0[v], 1e-6 * eqRefMax));
        if (settings_.turbulent) {
            // ||W - Wsave||_inf / ||W||_inf on the conservative turbulence
            // variables, measured on the ACCEPTED post-clamp state so the
            // wall omega re-pin cannot masquerade as motion; carried between
            // iterations by assignment (rejected steps skip this block)
            double kVal = 1e-30, omVal = 1e-30, kDiff = 0.0, omDiff = 0.0;
            for (int ci = 0; ci < nc; ++ci) {
                kVal = std::max(kVal, std::abs(W_[ci][4]));
                omVal = std::max(omVal, std::abs(W_[ci][5]));
                kDiff = std::max(kDiff, std::abs(W_[ci][4] - Wsave[ci][4]));
                omDiff = std::max(omDiff, std::abs(W_[ci][5] - Wsave[ci][5]));
            }
            lastKChange = kDiff / kVal;
            lastOmChange = omDiff / omVal;
        }
        {
            // mean-flow field-change norms of the accepted step, per
            // conservative component against its own field scale (the
            // quiescence route's measure)
            double change = 0.0;
            for (int v = 0; v < 4; ++v) {
                double val = 1e-30, diff = 0.0;
                for (int ci = 0; ci < nc; ++ci) {
                    val = std::max(val, std::abs(W_[ci][v]));
                    diff = std::max(diff, std::abs(W_[ci][v] - Wsave[ci][v]));
                }
                change = std::max(change, diff / val);
            }
            lastMeanChange = change;
        }
        const double turbMax = settings_.turbulent ? std::max(lastKChange, lastOmChange) : 0.0;
        lastCriterion = std::max(relMax, turbMax);
        if (settings_.verbose && (iter % settings_.reportInterval == 0)) {
            std::printf("  [lusgs] iter %6d  cfl %.1f  res_rho %.3e (rel %.3e)"
                        "  rel_max %.3e  rel_eq",
                        iter, cfl, rn, rn / res0, relMax);
            for (int v = 0; v < nEq; ++v)
                std::printf(" %.2e", rns[v] / std::max(eqRes0[v], 1e-6 * eqRefMax));
            std::printf("  dk %.2e dom %.2e\n", lastKChange, lastOmChange);
        }
        if (iter % settings_.reportInterval == 0)
            rep.residualHistory.push_back(rn);
        // frozen-mean mode: the density residual never decays (the mean is
        // pinned, its discrete residual is a constant mismatch), so the
        // limiter-chatter freeze triggers on the live criterion instead
        if (settings_.legacyAutoFreezeLimiter && iter > freezeIter && !limiterFrozen &&
            (settings_.frozenMeanFlow ? lastCriterion < 1e-2 : rn / res0 < 1e-2))
            limiterFrozen = true;
        // convergence requires EVERY live equation below the tolerance under
        // its own semantics AND a directly validated state (the per-cell
        // sweep; the acceptance logic alone cannot prove finiteness)
        // fail genuinely non-settling solves fast: past earlyAbortIter a
        // response still above earlyAbortRelMax is classified Unconverged
        // without burning the remaining budget (the measured separation:
        // settling members pass through order 1e-2 there, stuck ones sit
        // at order one and above)
        if (settings_.earlyAbortIter > 0 && iter >= settings_.earlyAbortIter && iter > freezeIter &&
            std::max(relMax, turbMax) > settings_.earlyAbortRelMax)
            break;
        // quiescence route, RESTORED-CHECKPOINT RESTARTS ONLY: a reloaded
        // converged state whose conservative components all stop moving at
        // the full Courant number over an accepted step is holding its
        // fixed point, and the relative criterion can never classify it
        // (its reference residual IS the tiny restart residual, so the
        // ratio idles near one). The route is gated on the restored
        // limiter checkpoint because on a COLD solve a per-step field
        // change below tolerance does NOT bound the accumulated remaining
        // drift (slow creep), and the registered relative-decay criterion
        // must govern. The
        // full-CFL guard keeps a rejection-throttled solve from
        // masquerading as quiescent.
        // and the injected correction must be exactly zero: a nonzero
        // stored discrepancy is a fresh force, so the restart is NOT at a
        // fixed point of the injected operator and its slowly developing
        // response could sit under the member tolerance per step and
        // falsely look quiescent; forced members converge only by the
        // registered relative decay of the injection response
        const bool zeroInjection =
            dbTarget6_.empty() || (injDiag_.maxDb == 0.0 && injDiag_.maxDq == 0.0);
        const bool quiescent =
            (startLimiterFrozen_ && zeroInjection && cflScale >= 1.0 && iter > freezeIter &&
             lastMeanChange < settings_.convergenceTol && turbMax < settings_.convergenceTol);
        if ((iter > freezeIter && std::max(relMax, turbMax) < settings_.convergenceTol) ||
            quiescent) {
            if (stateIsValid()) {
                rep.status = EvaluationStatus::Converged;
                quiescentConverged =
                    quiescent && std::max(relMax, turbMax) >= settings_.convergenceTol;
            } else {
                rep.status = EvaluationStatus::Diverged;
                rep.iterations = iter;
                rep.finalResidual = lastCriterion;
                return rep;
            }
            break;
        }
    }
    if (rep.status == EvaluationStatus::Unknown)
        rep.status = EvaluationStatus::Unconverged;
    rep.iterations = iter;
    {
        // the recorded final residual mirrors the live criterion: mean-flow
        // relative decay joined with the carried turbulence change norms
        const std::array<double, NVAR> rnsF = equationResidualNorms();
        const int nEq = settings_.frozenMeanFlow ? 0 : 4;
        double eqRefMax = 1e-30, relMax = 0.0;
        for (int v = 0; v < nEq; ++v)
            eqRefMax = std::max(eqRefMax, eqRes0[v]);
        for (int v = 0; v < nEq; ++v)
            relMax = std::max(relMax, rnsF[v] / std::max(eqRes0[v], 1e-6 * eqRefMax));
        if (settings_.turbulent)
            relMax = std::max(relMax, std::max(lastKChange, lastOmChange));
        // a quiescence-classified solve records the measure that actually
        // converged it (the field-change norms), not the idling relative
        // ratio of a warm restart's tiny reference
        rep.finalResidual =
            quiescentConverged
                ? std::max(lastMeanChange,
                           settings_.turbulent ? std::max(lastKChange, lastOmChange) : 0.0)
                : relMax;
        rep.quiescent = quiescentConverged;
    }
    return rep;
}

// floor density / pressure / turbulence to keep states physical; then re-pin
// the wall-adjacent omega. The validated incompressible treatment OVERRIDES
// the first cell's omega every iteration with the Menter value
// 60 nu / (beta1 y1^2) (ten times the analytic sublayer solution at the
// first-cell scale): that deliberately strong anchor feeds the diffusive
// omega tail that selects the log-law branch. A floor at the analytic value
// alone leaves a second, spurious low-shear/high-mixing near-wall
// equilibrium available, which gave a mesh-converged 27 percent
// skin-friction excess on a supersonic turbulent boundary layer.
void DBNSSolver::clampPositivity() {
    if (settings_.turbulent) {
        for (size_t idx = 0; idx < wallAdjCell_.size(); ++idx) {
            int ci = wallAdjCell_[idx];
            double y1 = wallAdjDelta_[idx];
            Primitive V = GasState::toPrimitive(W_[ci], eos_);
            double nu = muLam_[ci] / std::max(V.rho, rhoFloor_);
            const double omega = 60.0 * nu / (sstCoeffs_.beta1 * y1 * y1);
            // Omega is not part of total energy. Re-encoding unrelated rows
            // adds roundoff to conservative momentum/energy balances.
            W_[ci][I_RHOW] = V.rho * omega;
        }
    }
    clampPositivityCore();
}

void DBNSSolver::clampPositivityCore() {
    for (int ci = 0; ci < mesh_.nCells(); ++ci) {
        StateVec &W = W_[ci];
        if (W[I_RHO] < rhoFloor_)
            W[I_RHO] = rhoFloor_;
        Primitive V = GasState::toPrimitive(W, eos_);
        bool fix = false;
        if (!(V.p > pFloor_)) {
            V.p = pFloor_;
            fix = true;
        }
        if (settings_.turbulent) {
            if (V.k < settings_.kFloor) {
                V.k = settings_.kFloor;
                fix = true;
            }
            if (V.omega < settings_.omegaFloor) {
                V.omega = settings_.omegaFloor;
                fix = true;
            }
            // omega wall anchoring: the SST omega equation needs its singular
            // near-wall value, and the wall ghost is zero-gradient, so the
            // viscous-sublayer analytic solution omega = 6 nu / (beta1 y^2)
            // is imposed as a floor. It decays as 1/y^2 and binds only inside
            // the sublayer (the automatic-wall-treatment limit); without it
            // an imposed turbulent layer LAMINARIZES, with a laminar-like Cf
            // decay (test_dbns_sst_wall guards this).
            const auto &wd = wallDist_;
            if (ci < (int)wd.size()) {
                double y = std::max(wd[ci], 1e-12);
                double nu = muLam_[ci] / std::max(V.rho, rhoFloor_);
                double omegaVis = 6.0 * nu / (sstCoeffs_.beta1 * y * y);
                if (V.omega < omegaVis) {
                    V.omega = omegaVis;
                    fix = true;
                }
            }
        } else {
            V.k = 0.0;
            V.omega = 0.0;
        }
        if (fix)
            W_[ci] = GasState::toConserved(V, eos_);
        else {
            W_[ci][I_RHOK] = V.rho * V.k;
            W_[ci][I_RHOW] = V.rho * V.omega;
        }
    }
}

} // namespace dbns
