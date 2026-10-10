#pragma once

#include "Mesh.hpp"
#include "Field.hpp"
#include "FlowFields.hpp"
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <stdexcept>

// Measurement object for a single observed quantity with uncertainty
struct Measurement {
    double observed = 0.0;    // y_obs
    double sigma_obs = 0.01;  // observation uncertainty
    double sigma_model = 0.0; // model discrepancy (additive)
    double totalVariance() const {
        return sigma_obs * sigma_obs + sigma_model * sigma_model;
    } // total variance
    void validate() const {
        if (!std::isfinite(observed) || !std::isfinite(sigma_obs) || sigma_obs < 0 ||
            !std::isfinite(sigma_model) || sigma_model < 0 || !std::isfinite(totalVariance()) ||
            totalVariance() <= 0)
            throw std::invalid_argument(
                "Gaussian measurements require finite values, nonnegative uncertainties "
                "and finite positive total variance");
    }
};

// Observable type / kind of measurement
enum class ObsType {
    Drag,
    PressureTap,
    VelocityProfile,
    SkinFriction,
    SeparationPoint,
    ReattachmentLength
};

// Observable object - full description of one measurement
struct Observable {
    ObsType type;
    Measurement meas;

    // additional data depending on type
    std::string patchName;      // wall patch for drag / Cf / separation
    Vec3 location = {};         // spatial point for pressure tap / profile
    int component = 0;          // velocity component for profiles (0=x,1=y,2=z)
    double referenceArea = 1.0; // for drag normalisation
    double refVelocity = 1.0;   // for Cf / Cd normalisation
    // Pressure-coefficient references. The incompressible solver stores the
    // KINEMATIC pressure p/rho, so its legacy taps (refPressure 0,
    // refDensity 1) are already dimensionally consistent and unchanged. A
    // COMPRESSIBLE field stores absolute static pressure in Pa, so a tap
    // must carry the freestream reference pressure and density for
    // Cp = (p - p_ref) / (0.5 rho_ref U_ref^2) to be a coefficient at all.
    double refPressure = 0.0;
    double refDensity = 1.0;
};

// maps flow fields to predicted measurements
// converts the full CFD state into the predicted measurements that correspond to experimental data
class PreparedObservation;

class ObservationOperator {
  public:
    ObservationOperator() = default;

    // add observable types
    void addDrag(const std::string &wallPatch, double Cd_obs, double sigma, double refArea,
                 double refVel, double sigma_model = 0.0);
    void addPressureTap(const Vec3 &loc, double Cp_obs, double sigma, double refVel,
                        double sigma_model = 0.0, double refPressure = 0.0,
                        double refDensity = 1.0);
    void addVelocityProfile(const Vec3 &loc, int comp, double Uobs, double sigma,
                            double sigma_model = 0.0);
    void addSkinFriction(const std::string &wallPatch, const Vec3 &loc, double Cf_obs, double sigma,
                         double refVel, double sigma_model = 0.0);
    void addSeparationPoint(const std::string &wallPatch, double xSep_obs, double sigma,
                            double sigma_model = 0.0);
    // x-location where wall shear recovers from negative (recirculation) to positive
    // (reattachment).
    void addReattachmentLength(const std::string &wallPatch, double xr_obs, double sigma,
                               double sigma_model = 0.0);

    // evaluate all observables
    // returns predicted values H(fields) in same order as observables_
    std::vector<double> evaluate(const Mesh &mesh, const FlowFields &fields, double nu) const;

    PreparedObservation prepare(const Mesh &mesh) const;

    // log-likelihood
    // sum of Gaussian log-likelihoods: sum_i -0.5*(H_i - y_i)^2 / sigma_i^2
    double logLikelihood(const Mesh &mesh, const FlowFields &fields, double nu) const;
    double logLikelihood(const std::vector<double> &predictions) const;

    int nObs() const { return static_cast<int>(observables_.size()); }
    const std::vector<Observable> &observables() const { return observables_; }

    // Skin-friction sampling stencil: the observable is
    // (1 - t) Cf(faceA) + t Cf(faceB), linear between the two wall-face
    // centres that bracket the location (faceB = -1 and t = 0 beyond the last
    // face). Public so derivative code can use the same weights.
    struct WallStencil {
        int faceA = -1;
        int faceB = -1;
        double t = 0.0;
    };
    WallStencil skinFrictionStencil(const Mesh &mesh, const Observable &obs) const;

  private:
    friend class PreparedObservation;
    std::vector<Observable> observables_;

    static void validateLocation(const Vec3 &location) {
        if (!std::isfinite(location.x) || !std::isfinite(location.y) || !std::isfinite(location.z))
            throw std::invalid_argument("observation location must be finite");
    }
    static void validateReference(double value, const char *name) {
        if (!std::isfinite(value) || value <= 0.)
            throw std::invalid_argument(std::string(name) + " must be finite and positive");
    }

    // helpers
    double computeDrag(const Mesh &mesh, const FlowFields &fields, double nu,
                       const Observable &obs) const;
    double computeSkinFrictionAt(const Mesh &mesh, const FlowFields &fields, double nu,
                                 const Observable &obs) const;
    double computeVelocityAt(const Mesh &mesh, const FlowFields &fields,
                             const Observable &obs) const;
    double computePressureAt(const Mesh &mesh, const FlowFields &fields,
                             const Observable &obs) const;
    double computeSeparationPoint(const Mesh &mesh, const FlowFields &fields, double nu,
                                  const Observable &obs) const;
    double computeReattachmentLength(const Mesh &mesh, const FlowFields &fields, double nu,
                                     const Observable &obs) const;
    int findNearestCell(const Mesh &mesh, const Vec3 &loc) const;
    int findNearestWallFace(const Mesh &mesh, const std::string &patchName, const Vec3 &loc) const;
    int findContainingCell(const Mesh &mesh, const Vec3 &loc, Vec3 &locInCell) const;
    double wallFaceSkinFriction(const Mesh &mesh, const FlowFields &fields, double nu,
                                const Observable &obs, int fi) const;
};

// ---- interpolating probe helpers ----------------------------------------------

// Least-squares gradient (x and y components) of a cell field at cell ci,
// fitted to the face neighbours' centres (periodic images included) and to the
// boundary-face values, with 1/|d|^2 weights. It is exact for linear fields on
// any mesh. The Green-Gauss gradient with distance-weighted face values is not
// exact on the skewed cells of terrain-following meshes, so the probes do not
// use it. Faces with a z normal (front/back of OpenFOAM meshes) are skipped.
template <class CellValue, class FaceValue>
inline Vec3 leastSquaresGradient2D(const Mesh &mesh, int ci, CellValue cellValue,
                                   FaceValue boundaryValue) {
    const Vec3 &c = mesh.cell(ci).center;
    const double phiC = cellValue(ci);
    double axx = 0.0, axy = 0.0, ayy = 0.0, bx = 0.0, by = 0.0;
    for (FaceID fi : mesh.cell(ci).faces) {
        const Face &f = mesh.face(fi);
        if (std::abs(f.normal.z) > 0.5)
            continue;
        Vec3 d;
        double dphi;
        if (f.isBoundary()) {
            d = f.center - c;
            dphi = boundaryValue(fi) - phiC;
        } else {
            const bool own = (f.owner == ci);
            const int nb = own ? f.neighbor : f.owner;
            d = own ? mesh.cell(nb).center + f.shift - c : mesh.cell(nb).center - f.shift - c;
            dphi = cellValue(nb) - phiC;
        }
        const double w = 1.0 / std::max(d.x * d.x + d.y * d.y, 1e-300);
        axx += w * d.x * d.x;
        axy += w * d.x * d.y;
        ayy += w * d.y * d.y;
        bx += w * d.x * dphi;
        by += w * d.y * dphi;
    }
    const double det = axx * ayy - axy * axy;
    if (std::abs(det) <= 1e-14 * (axx * ayy + 1e-300))
        return Vec3(0.0, 0.0, 0.0);
    return Vec3((ayy * bx - axy * by) / det, (axx * by - axy * bx) / det, 0.0);
}

//  Inline implementations functions
// create empty observable, define measurement, store observed value + uncertainties

inline void ObservationOperator::addDrag(const std::string &wallPatch, double Cd_obs, double sigma,
                                         double refArea, double refVel, double sigma_model) {
    validateReference(refArea, "drag reference area");
    validateReference(refVel, "drag reference speed");
    Observable obs;
    obs.type = ObsType::Drag;
    obs.meas = {Cd_obs, sigma, sigma_model};
    obs.patchName = wallPatch;
    obs.referenceArea = refArea;
    obs.refVelocity = refVel;
    obs.meas.validate();
    observables_.push_back(obs);
}

inline void ObservationOperator::addPressureTap(const Vec3 &loc, double Cp_obs, double sigma,
                                                double refVel, double sigma_model,
                                                double refPressure, double refDensity) {
    validateLocation(loc);
    validateReference(refVel, "pressure reference speed");
    validateReference(refDensity, "pressure reference density");
    if (!std::isfinite(refPressure))
        throw std::invalid_argument("reference pressure must be finite");
    Observable obs;
    obs.type = ObsType::PressureTap;
    obs.meas = {Cp_obs, sigma, sigma_model};
    obs.location = loc;
    obs.refVelocity = refVel;
    obs.refPressure = refPressure;
    obs.refDensity = refDensity;
    obs.meas.validate();
    observables_.push_back(obs);
}

inline void ObservationOperator::addVelocityProfile(const Vec3 &loc, int comp, double Uobs,
                                                    double sigma, double sigma_model) {
    validateLocation(loc);
    if (comp < 0 || comp > 2)
        throw std::invalid_argument("velocity observation component must be in [0,2]");
    Observable obs;
    obs.type = ObsType::VelocityProfile;
    obs.meas = {Uobs, sigma, sigma_model};
    obs.location = loc;
    obs.component = comp;
    obs.meas.validate();
    observables_.push_back(obs);
}

inline void ObservationOperator::addSkinFriction(const std::string &wallPatch, const Vec3 &loc,
                                                 double Cf_obs, double sigma, double refVel,
                                                 double sigma_model) {
    validateLocation(loc);
    validateReference(refVel, "friction reference speed");
    Observable obs;
    obs.type = ObsType::SkinFriction;
    obs.meas = {Cf_obs, sigma, sigma_model};
    obs.patchName = wallPatch;
    obs.location = loc;
    obs.refVelocity = refVel;
    obs.meas.validate();
    observables_.push_back(obs);
}

inline void ObservationOperator::addSeparationPoint(const std::string &wallPatch, double xSep_obs,
                                                    double sigma, double sigma_model) {
    Observable obs;
    obs.type = ObsType::SeparationPoint;
    obs.meas = {xSep_obs, sigma, sigma_model};
    obs.patchName = wallPatch;
    obs.meas.validate();
    observables_.push_back(obs);
}

inline void ObservationOperator::addReattachmentLength(const std::string &wallPatch, double xr_obs,
                                                       double sigma, double sigma_model) {
    Observable obs;
    obs.type = ObsType::ReattachmentLength;
    obs.meas = {xr_obs, sigma, sigma_model};
    obs.patchName = wallPatch;
    obs.meas.validate();
    observables_.push_back(obs);
}

// nearest cell search (brute force)
inline int ObservationOperator::findNearestCell(const Mesh &mesh, const Vec3 &loc) const {
    int best = 0;
    double bestDist = 1e30;
    for (int ci = 0; ci < mesh.nCells(); ++ci) {
        double d = (mesh.cell(ci).center - loc).norm2();
        if (d < bestDist) {
            bestDist = d;
            best = ci;
        }
    }
    return best;
}

// Cell containing a point, in the x-y plane: walk from the nearest cell
// centre, each step crossing the face the point lies furthest outside of,
// until it lies inside every face. A point beyond a boundary face (e.g. a tap
// exactly on a wall) stays in the cell owning that face. Crossing a periodic
// wrap face moves the point to the neighbour's frame, returned in locInCell.
inline int ObservationOperator::findContainingCell(const Mesh &mesh, const Vec3 &loc,
                                                   Vec3 &locInCell) const {
    int ci = findNearestCell(mesh, loc);
    locInCell = loc;
    const int maxSteps = 2 * mesh.nCells() + 10;
    for (int step = 0; step < maxSteps; ++step) {
        double worst = 0.0;
        int crossFace = -1;
        for (FaceID fi : mesh.cell(ci).faces) {
            const Face &f = mesh.face(fi);
            if (std::abs(f.normal.z) > 0.5)
                continue;
            const bool own = (f.owner == ci);
            Vec3 n = own ? f.normal : f.normal * -1.0;
            Vec3 cf = own ? f.center : f.center - f.shift;
            Vec3 d = locInCell - cf;
            double out = d.x * n.x + d.y * n.y;
            if (out > worst + 1e-12 * std::sqrt(f.area)) {
                worst = out;
                crossFace = fi;
            }
        }
        if (crossFace < 0)
            return ci; // inside
        const Face &f = mesh.face(crossFace);
        if (f.isBoundary())
            return ci; // outside the domain
        const bool own = (f.owner == ci);
        locInCell = own ? locInCell - f.shift : locInCell + f.shift;
        ci = own ? f.neighbor : f.owner;
    }
    return findNearestCell(mesh, loc); // no convergence (non-convex cells): fall back
}

// nearest wall face on a named patch
inline int ObservationOperator::findNearestWallFace(const Mesh &mesh, const std::string &patchName,
                                                    const Vec3 &loc) const {
    PatchID pid = mesh.patchByName(patchName);
    const Patch &pat = mesh.patch(pid);
    int best = pat.faces.empty() ? -1 : pat.faces[0];
    double bestDist = 1e30;
    for (FaceID fi : pat.faces) {
        double d = (mesh.face(fi).center - loc).norm2();
        if (d < bestDist) {
            bestDist = d;
            best = fi;
        }
    }
    return best;
}

// drag: ∫(p*n + tau*n) dA normalised
inline double ObservationOperator::computeDrag(const Mesh &mesh, const FlowFields &fields,
                                               double nu, const Observable &obs) const {
    PatchID pid = mesh.patchByName(obs.patchName);
    const Patch &pat = mesh.patch(pid);

    double force = 0.0;
    for (FaceID fi : pat.faces) {
        const Face &face = mesh.face(fi);
        int ow = face.owner;
        double Sf = face.area;

        // UNITS CONTRACT: this generic adapter is valid on KINEMATIC-pressure
        // fields only (the incompressible convention, p stored as p/rho), where
        // the p*n*A term and the nu*dU/dn viscous term are commensurate
        // (both m^2/s^2 per unit density). CompressibleForwardModel rejects
        // Drag observables at construction because its dimensional pressure
        // cannot be combined with this kinematic viscous term; DBNS owns the
        // compressible wall-observation path.
        double pf = fields.p[ow];
        force += pf * face.normal.x * Sf;

        // viscous contribution: MOLECULAR wall shear tau_w = nu * dU/dn.
        // At a no-slip wall the eddy viscosity vanishes (SST asymptotics),
        // so the molecular stress IS the wall stress and matches the DNS
        // definition; including the wall-adjacent cell's nuT would import
        // any numerical nuT bound into the observable (a permanent 0.1*nu
        // floor biased it by ~+10 percent). Valid for wall-resolved first
        // cells; wall-function runs must use the log-law stress instead.
        double delta = std::max(face.delta, 1e-20);
        double Uw_x = fields.U.bface(fi).x;
        double tau_x = nu * (fields.U[ow].x - Uw_x) / delta;
        force += tau_x * Sf;
    }

    double dynPressure = 0.5 * obs.refVelocity * obs.refVelocity;
    return force / (dynPressure * obs.referenceArea);
}

// Skin friction is sampled linearly between the two wall faces that bracket
// the location: the nearest face, and the nearest face on the far side of the
// location from it. Snapping to the nearest face alone left a first-order,
// sign-alternating sampling error along the wall.
inline ObservationOperator::WallStencil
ObservationOperator::skinFrictionStencil(const Mesh &mesh, const Observable &obs) const {
    WallStencil st;
    st.faceA = findNearestWallFace(mesh, obs.patchName, obs.location);
    if (st.faceA < 0)
        return st;
    const Patch &pat = mesh.patch(mesh.patchByName(obs.patchName));
    const Vec3 ca = mesh.face(st.faceA).center;
    const Vec3 dloc = obs.location - ca;
    double best = 1e300;
    for (FaceID fi : pat.faces) {
        if (fi == st.faceA)
            continue;
        Vec3 e = mesh.face(fi).center - ca;
        if (e.dot(dloc) <= 0.0)
            continue;
        if (e.norm2() < best) {
            best = e.norm2();
            st.faceB = fi;
        }
    }
    if (st.faceB >= 0) {
        Vec3 e = mesh.face(st.faceB).center - ca;
        st.t = std::min(std::max(dloc.dot(e) / e.norm2(), 0.0), 1.0);
    }
    return st;
}

inline double ObservationOperator::computeSkinFrictionAt(const Mesh &mesh, const FlowFields &fields,
                                                         double nu, const Observable &obs) const {
    WallStencil st = skinFrictionStencil(mesh, obs);
    if (st.faceA < 0)
        return 0.0;
    double cf = wallFaceSkinFriction(mesh, fields, nu, obs, st.faceA);
    if (st.faceB >= 0)
        cf = (1.0 - st.t) * cf + st.t * wallFaceSkinFriction(mesh, fields, nu, obs, st.faceB);
    return cf;
}

// skin friction on one wall face
inline double ObservationOperator::wallFaceSkinFriction(const Mesh &mesh, const FlowFields &fields,
                                                        double nu, const Observable &obs,
                                                        int fi) const {
    const Face &face = mesh.face(fi);
    int ow = face.owner;
    double delta = std::max(face.delta, 1e-20);

    // MOLECULAR wall shear (see computeDrag): the DNS-comparable Cf, free of
    // any numerical nuT bound at the wall-adjacent cell. On a compressible
    // field this normalization is a LOW-MACH NEARLY-CONSTANT-DENSITY
    // convention: the kinematic stress over 0.5 U^2 equals the dimensional
    // Cf only up to the wall-to-reference density ratio, which is ~1 for the
    // committed Ma 0.1 validation; cooled walls or strong density ratios are
    // NOT supported here (that regime belongs to the DBNS observation path).
    Vec3 Uc = fields.U[ow];
    Vec3 Un = face.normal * Uc.dot(face.normal); // normal component
    Vec3 Ut = Uc - Un;                           // tangential
    double tau = nu * Ut.norm() / delta;
    double dynP = 0.5 * obs.refVelocity * obs.refVelocity;
    return tau / std::max(dynP, 1e-20);
}

// Velocity and pressure probes reconstruct linearly inside the containing
// cell, phi_c + grad(phi)_c . (x - x_c), with the least-squares gradient: exact
// for linear fields, second order otherwise, and the cell value at the cell
// centre. They used to return the nearest cell's value, a first-order,
// sign-alternating sampling error that read as structured closure error.
inline double ObservationOperator::computeVelocityAt(const Mesh &mesh, const FlowFields &fields,
                                                     const Observable &obs) const {
    Vec3 x;
    int ci = findContainingCell(mesh, obs.location, x);
    const int comp = obs.component;
    auto comp3 = [comp](const Vec3 &v) { return comp == 0 ? v.x : (comp == 1 ? v.y : v.z); };
    Vec3 g = leastSquaresGradient2D(
        mesh, ci, [&](int c) { return comp3(fields.U[c]); },
        [&](int fi) { return comp3(fields.U.bface(fi)); });
    Vec3 d = x - mesh.cell(ci).center;
    return comp3(fields.U[ci]) + g.x * d.x + g.y * d.y;
}

// pressure (Cp) at a point
inline double ObservationOperator::computePressureAt(const Mesh &mesh, const FlowFields &fields,
                                                     const Observable &obs) const {
    Vec3 x;
    int ci = findContainingCell(mesh, obs.location, x);
    Vec3 g = leastSquaresGradient2D(
        mesh, ci, [&](int c) { return fields.p[c]; }, [&](int fi) { return fields.p.bface(fi); });
    Vec3 d = x - mesh.cell(ci).center;
    double p = fields.p[ci] + g.x * d.x + g.y * d.y;
    // Cp = (p - p_ref) / (0.5 rho_ref U_ref^2). Legacy incompressible taps
    // (kinematic p, refPressure 0, refDensity 1) are dimensionally consistent;
    // compressible absolute-pressure fields need both references or the
    // "coefficient" is dominated by the ~1e5 Pa offset.
    double dynP = 0.5 * obs.refDensity * obs.refVelocity * obs.refVelocity;
    return (p - obs.refPressure) / std::max(dynP, 1e-20);
}

// separation point: x-location where wall shear = 0.
// Sub-cell interpolation of the tau_x zero-crossing (rather than snapping to a face
// centre) makes x_sep a *differentiable* function of the flow field — a prerequisite
// for the FD/adjoint gradient path (the discrete snap is piecewise-constant in theta).
inline double ObservationOperator::computeSeparationPoint(const Mesh &mesh,
                                                          const FlowFields &fields, double nu,
                                                          const Observable &obs) const {
    PatchID pid = mesh.patchByName(obs.patchName);
    const Patch &pat = mesh.patch(pid);

    bool havePrev = false;
    double prevTau = 1.0; // assume positive (attached) at start
    double prevX = 0.0;
    for (FaceID fi : pat.faces) {
        const Face &face = mesh.face(fi);
        int ow = face.owner;
        double delta = std::max(face.delta, 1e-20);
        Vec3 Uc = fields.U[ow];
        Vec3 Ut = Uc - face.normal * Uc.dot(face.normal);
        // signed MOLECULAR tangential shear (x-component as streamwise
        // indicator); the zero-crossing is invariant to the positive
        // viscosity factor, so this is a consistency change only
        double tau = nu * Ut.x / delta;
        double x = face.center.x;

        // positive -> non-positive crossing: interpolate x where tau = 0
        if (havePrev && prevTau > 0 && tau <= 0) {
            double denom = prevTau - tau; // > 0 here
            double frac = (std::abs(denom) > 1e-30) ? prevTau / denom : 0.0;
            return prevX + frac * (x - prevX);
        }
        prevTau = tau;
        prevX = x;
        havePrev = true;
    }
    return 0.0; // no separation found
}

// reattachment length: x-location where wall shear recovers from negative to positive.
// Sub-cell interpolated zero-crossing (differentiable; see computeSeparationPoint).
inline double ObservationOperator::computeReattachmentLength(const Mesh &mesh,
                                                             const FlowFields &fields, double nu,
                                                             const Observable &obs) const {
    PatchID pid = mesh.patchByName(obs.patchName);
    const Patch &pat = mesh.patch(pid);

    double xr_default = pat.faces.empty() ? 0.0 : mesh.face(pat.faces.back()).center.x;
    bool havePrev = false;
    double prevTau = -1.0; // assume negative (recirculating) at start
    double prevX = 0.0;
    for (FaceID fi : pat.faces) {
        const Face &face = mesh.face(fi);
        int ow = face.owner;
        double delta = std::max(face.delta, 1e-20);
        Vec3 Uc = fields.U[ow];
        Vec3 Ut = Uc - face.normal * Uc.dot(face.normal);
        // molecular shear; zero-crossing invariant (see computeSeparationPoint)
        double tau = nu * Ut.x / delta;
        double x = face.center.x;

        // negative -> non-negative crossing: interpolate x where tau = 0
        if (havePrev && prevTau < 0 && tau >= 0) {
            double denom = tau - prevTau; // > 0 here
            double frac = (std::abs(denom) > 1e-30) ? (-prevTau) / denom : 0.0;
            return prevX + frac * (x - prevX);
        }
        prevTau = tau;
        prevX = x;
        havePrev = true;
    }
    return xr_default; // no reattachment detected within domain — last face x
}

// evaluate all observables
inline std::vector<double> ObservationOperator::evaluate(const Mesh &mesh, const FlowFields &fields,
                                                         double nu) const {
    if (&fields.U.mesh() != &mesh || &fields.p.mesh() != &mesh)
        throw std::invalid_argument("observation fields must belong to this mesh");
    std::vector<double> pred(observables_.size());
    for (size_t i = 0; i < observables_.size(); ++i) {
        const Observable &obs = observables_[i];
        switch (obs.type) {
        case ObsType::Drag:
            pred[i] = computeDrag(mesh, fields, nu, obs);
            break;
        case ObsType::PressureTap:
            pred[i] = computePressureAt(mesh, fields, obs);
            break;
        case ObsType::VelocityProfile:
            pred[i] = computeVelocityAt(mesh, fields, obs);
            break;
        case ObsType::SkinFriction:
            pred[i] = computeSkinFrictionAt(mesh, fields, nu, obs);
            break;
        case ObsType::SeparationPoint:
            pred[i] = computeSeparationPoint(mesh, fields, nu, obs);
            break;
        case ObsType::ReattachmentLength:
            pred[i] = computeReattachmentLength(mesh, fields, nu, obs);
            break;
        }
    }
    return pred;
}

// Gaussian log-likelihood
inline double ObservationOperator::logLikelihood(const Mesh &mesh, const FlowFields &fields,
                                                 double nu) const {
    return logLikelihood(evaluate(mesh, fields, nu));
}

inline double ObservationOperator::logLikelihood(const std::vector<double> &pred) const {
    if (pred.size() != observables_.size())
        throw std::invalid_argument("Likelihood prediction count does not match observations");
    double ll = 0.0;
    for (size_t i = 0; i < observables_.size(); ++i) {
        observables_[i].meas.validate();
        if (!std::isfinite(pred[i]))
            throw std::invalid_argument("Likelihood predictions must be finite");
        const double var = observables_[i].meas.totalVariance();
        double diff = pred[i] - observables_[i].meas.observed;
        ll += -0.5 * diff * diff / var;
        if (!std::isfinite(ll))
            throw std::overflow_error("Gaussian log-likelihood exceeds finite range");
    }
    return ll;
}

// Immutable definitions and geometric stencils, borrowing only the mesh.
// Cell/boundary values and separation crossings are read on every evaluation.
class PreparedObservation {
    struct Neighbor {
        int cell, face;
        Vec3 d;
        double weight;
    };
    struct Stencil {
        int cell = -1;
        Vec3 offset;
        double axx = 0, axy = 0, ayy = 0, determinant = 0;
        std::vector<Neighbor> neighbors;
        ObservationOperator::WallStencil friction;
        std::vector<int> wallFaces;
    };
    const Mesh *mesh_;
    Mesh::Stamp stamp_;
    ObservationOperator definitions_;
    std::vector<Stencil> stencils_;
    friend class ObservationOperator;
    PreparedObservation(const Mesh &mesh, const ObservationOperator &definitions)
        : mesh_(&mesh), stamp_(mesh.stamp()), definitions_(definitions) {
        for (const auto &obs : definitions_.observables()) {
            Stencil st;
            if (obs.type == ObsType::VelocityProfile || obs.type == ObsType::PressureTap) {
                if (mesh.nCells() == 0)
                    throw std::invalid_argument("cannot prepare a probe on an empty mesh");
                Vec3 point;
                st.cell = definitions_.findContainingCell(mesh, obs.location, point);
                const auto &center = mesh.cell(st.cell).center;
                st.offset = point - center;
                for (int fi : mesh.cell(st.cell).faces) {
                    const auto &face = mesh.face(fi);
                    if (std::abs(face.normal.z) > .5)
                        continue;
                    Vec3 d;
                    int neighbor = -1;
                    if (face.isBoundary())
                        d = face.center - center;
                    else {
                        const bool own = face.owner == st.cell;
                        neighbor = own ? face.neighbor : face.owner;
                        d = own ? mesh.cell(neighbor).center + face.shift - center
                                : mesh.cell(neighbor).center - face.shift - center;
                    }
                    const double w = 1.0 / std::max(d.x * d.x + d.y * d.y, 1e-300);
                    st.axx += w * d.x * d.x;
                    st.axy += w * d.x * d.y;
                    st.ayy += w * d.y * d.y;
                    st.neighbors.push_back({neighbor, fi, d, w});
                }
                st.determinant = st.axx * st.ayy - st.axy * st.axy;
            } else if (obs.type == ObsType::SkinFriction) {
                st.friction = definitions_.skinFrictionStencil(mesh, obs);
            } else {
                st.wallFaces = mesh.patch(mesh.patchByName(obs.patchName)).faces;
            }
            stencils_.push_back(std::move(st));
        }
    }

  public:
    int nObs() const { return definitions_.nObs(); }
    std::vector<double> evaluate(const FlowFields &fields, double nu) const {
        std::vector<double> output;
        evaluate(fields, nu, output);
        return output;
    }
    void evaluate(const FlowFields &fields, double nu, std::vector<double> &output) const {
        mesh_->requireStamp(stamp_);
        if (&fields.U.mesh() != mesh_ || &fields.p.mesh() != mesh_)
            throw std::invalid_argument("prepared observation fields must match the mesh");
        for (const auto *scalar : {&fields.p, &fields.k, &fields.omega, &fields.nuT, &fields.F1,
                                   &fields.F2, &fields.Pk, &fields.CDkw})
            if (&output == &scalar->data())
                throw std::invalid_argument("prepared observation output cannot alias fields");
        const Mesh &mesh = *mesh_;
        output.resize(stencils_.size());
        for (size_t index = 0; index < stencils_.size(); ++index) {
            const auto &st = stencils_[index];
            const auto &obs = definitions_.observables()[index];
            if (st.cell >= 0) {
                const bool pressure = obs.type == ObsType::PressureTap;
                auto component = [&](const Vec3 &u) {
                    return obs.component == 0 ? u.x : obs.component == 1 ? u.y : u.z;
                };
                auto cell = [&](int i) { return pressure ? fields.p[i] : component(fields.U[i]); };
                auto boundary = [&](int fi) {
                    return pressure ? fields.p.bface(fi) : component(fields.U.bface(fi));
                };
                const double center = cell(st.cell);
                double bx = 0, by = 0;
                for (const auto &neighbor : st.neighbors) {
                    const double difference =
                        (neighbor.cell < 0 ? boundary(neighbor.face) : cell(neighbor.cell)) -
                        center;
                    bx += neighbor.weight * neighbor.d.x * difference;
                    by += neighbor.weight * neighbor.d.y * difference;
                }
                Vec3 gradient;
                if (std::abs(st.determinant) > 1e-14 * (st.axx * st.ayy + 1e-300))
                    gradient = {(st.ayy * bx - st.axy * by) / st.determinant,
                                (st.axx * by - st.axy * bx) / st.determinant, 0};
                const double value = center + gradient.x * st.offset.x + gradient.y * st.offset.y;
                const double dynamicPressure =
                    .5 * obs.refDensity * obs.refVelocity * obs.refVelocity;
                output[index] =
                    pressure ? (value - obs.refPressure) / std::max(dynamicPressure, 1e-20) : value;
            } else if (obs.type == ObsType::SkinFriction) {
                double value = 0;
                if (st.friction.faceA >= 0) {
                    value =
                        definitions_.wallFaceSkinFriction(mesh, fields, nu, obs, st.friction.faceA);
                    if (st.friction.faceB >= 0)
                        value = (1.0 - st.friction.t) * value +
                                st.friction.t * definitions_.wallFaceSkinFriction(
                                                    mesh, fields, nu, obs, st.friction.faceB);
                }
                output[index] = value;
            } else if (obs.type == ObsType::Drag) {
                double force = 0;
                for (int fi : st.wallFaces) {
                    const auto &face = mesh.face(fi);
                    const int owner = face.owner;
                    force += fields.p[owner] * face.normal.x * face.area;
                    const double delta = std::max(face.delta, 1e-20);
                    const double tau = nu * (fields.U[owner].x - fields.U.bface(fi).x) / delta;
                    force += tau * face.area;
                }
                const double dynamicPressure = .5 * obs.refVelocity * obs.refVelocity;
                output[index] = force / (dynamicPressure * obs.referenceArea);
            } else {
                const bool separation = obs.type == ObsType::SeparationPoint;
                double value = separation || st.wallFaces.empty()
                                   ? 0
                                   : mesh.face(st.wallFaces.back()).center.x;
                bool havePrevious = false;
                double previousTau = separation ? 1 : -1, previousX = 0;
                for (int fi : st.wallFaces) {
                    const auto &face = mesh.face(fi);
                    const auto u = fields.U[face.owner];
                    const auto tangent = u - face.normal * u.dot(face.normal);
                    const double tau = nu * tangent.x / std::max(face.delta, 1e-20),
                                 x = face.center.x;
                    if (havePrevious &&
                        (separation ? previousTau > 0 && tau <= 0 : previousTau < 0 && tau >= 0)) {
                        const double denominator =
                            separation ? previousTau - tau : tau - previousTau;
                        const double fraction =
                            std::abs(denominator) > 1e-30
                                ? (separation ? previousTau : -previousTau) / denominator
                                : 0;
                        value = previousX + fraction * (x - previousX);
                        break;
                    }
                    previousTau = tau;
                    previousX = x;
                    havePrevious = true;
                }
                output[index] = value;
            }
        }
    }
};

inline PreparedObservation ObservationOperator::prepare(const Mesh &mesh) const {
    return PreparedObservation(mesh, *this);
}
