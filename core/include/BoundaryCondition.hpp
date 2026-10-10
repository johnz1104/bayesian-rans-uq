#pragma once

#include "Mesh.hpp"
#include "Field.hpp"
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <utility>

// Boundary condition types
enum class BCType {
    Dirichlet,
    Neumann,
    InletVelocity,
    OutletPressure,
    WallNoSlip,
    WallKOmega,
    WallMoving, // no-slip wall translating tangentially at vecValue (Couette)
    Symmetry,
    Cyclic
};
// Boundary conditions for a single mesh patch
struct PatchBC {
    BCType type = BCType::Neumann;
    double value = 0.0;
    Vec3 vecValue = {};
    std::string patchName;
    // Optional per-face boundary values (a profile), indexed by the face's
    // position in the patch face list. When non-empty they override the uniform
    // vecValue/value for the Dirichlet-like types (InletVelocity / Dirichlet),
    // so an inlet can prescribe a boundary-layer profile U(y), k(y), omega(y)
    // instead of a uniform state (the Le-Moin BFS inflow is such a profile).
    std::vector<Vec3> vecProfile = {};
    std::vector<double> scalarProfile = {};
};

// Field applicators distinguish values they actually use. Both is the
// conservative default for callers that omit the scalar/vector context.
// ScalarUniform covers pressure, whose applicator does not consume profiles.
enum class BoundaryValueKind { Scalar, ScalarUniform, Vector, Both };

// Check all structural inputs before applying any face value. Inactive values
// are ignored; an active profile replaces its matching uniform value.
inline void validatePatchBCs(const Mesh &mesh, const std::vector<PatchBC> &bcs,
                             BoundaryValueKind kind = BoundaryValueKind::Both) {
    if (kind != BoundaryValueKind::Scalar && kind != BoundaryValueKind::ScalarUniform &&
        kind != BoundaryValueKind::Vector && kind != BoundaryValueKind::Both)
        throw std::invalid_argument("unknown boundary value context");
    if (bcs.size() != static_cast<size_t>(mesh.nPatches()))
        throw std::invalid_argument("boundary conditions require one entry per mesh patch");
    const auto scalarFinite = [](double value) {
        if (!std::isfinite(value))
            throw std::invalid_argument("active scalar boundary values must be finite");
    };
    const auto vectorFinite = [](const Vec3 &value) {
        if (!std::isfinite(value.x) || !std::isfinite(value.y) || !std::isfinite(value.z))
            throw std::invalid_argument("active vector boundary values must be finite");
    };
    for (int p = 0; p < mesh.nPatches(); ++p) {
        const auto &bc = bcs[p];
        switch (bc.type) {
        case BCType::Dirichlet:
        case BCType::Neumann:
        case BCType::InletVelocity:
        case BCType::OutletPressure:
        case BCType::WallNoSlip:
        case BCType::WallKOmega:
        case BCType::WallMoving:
        case BCType::Symmetry:
        case BCType::Cyclic:
            break;
        default:
            throw std::invalid_argument("unknown boundary condition type");
        }
        const size_t count = mesh.patch(p).faces.size();
        if ((!bc.vecProfile.empty() && bc.vecProfile.size() != count) ||
            (!bc.scalarProfile.empty() && bc.scalarProfile.size() != count))
            throw std::invalid_argument("boundary profile requires one value per patch face");
        if (kind != BoundaryValueKind::Vector &&
            (bc.type == BCType::Dirichlet ||
             (bc.type == BCType::OutletPressure && kind != BoundaryValueKind::Scalar))) {
            if (bc.type == BCType::OutletPressure || kind == BoundaryValueKind::ScalarUniform ||
                bc.scalarProfile.empty())
                scalarFinite(bc.value);
            else
                for (double value : bc.scalarProfile)
                    scalarFinite(value);
        }
        if ((kind == BoundaryValueKind::Vector || kind == BoundaryValueKind::Both) &&
            (bc.type == BCType::Dirichlet || bc.type == BCType::InletVelocity ||
             bc.type == BCType::WallMoving)) {
            if (bc.type == BCType::WallMoving || bc.vecProfile.empty())
                vectorFinite(bc.vecValue);
            else
                for (const auto &value : bc.vecProfile)
                    vectorFinite(value);
        }
    }
}

// Container for boundary conditions of solved fields
// stores conditions for every boundary patch in mesh
struct FlowBoundaryConditions {
    std::vector<PatchBC> velocityBC; // velocity u
    std::vector<PatchBC> pressureBC; // pressure p
    std::vector<PatchBC> kBC;        // tubulent kinetic energy k
    std::vector<PatchBC> omegaBC;    // specific dissipation rate (omega/w)

    // defines defaults for channle flow
    // input args: reference to mesh, inlet u, inlet k, inlet omega
    static FlowBoundaryConditions channelDefaults(const Mesh &mesh, double Uin, double kIn,
                                                  double omIn) {
        FlowBoundaryConditions bc;
        int np = mesh.nPatches(); // np = number of patches

        // resizes boundary conditions to np
        bc.velocityBC.resize(np);
        bc.pressureBC.resize(np);
        bc.kBC.resize(np);
        bc.omegaBC.resize(np);

        // assign default BC types and values for each field on each mesh patch
        // channel flow case
        for (int p = 0; p < np; ++p) {
            const Patch &pat = mesh.patch(p);
            bc.velocityBC[p].patchName = pat.name;
            bc.pressureBC[p].patchName = pat.name;
            bc.kBC[p].patchName = pat.name;
            bc.omegaBC[p].patchName = pat.name;

            if (pat.type == "wall") {
                bc.velocityBC[p].type = BCType::WallNoSlip;
                bc.pressureBC[p].type = BCType::Neumann;
                bc.kBC[p].type = BCType::WallKOmega;
                bc.omegaBC[p].type = BCType::WallKOmega;
            } else if (pat.type == "inlet") {
                bc.velocityBC[p] = {BCType::InletVelocity, 0.0, Vec3(Uin, 0, 0), pat.name};
                bc.pressureBC[p] = {BCType::Neumann, 0.0, {}, pat.name};
                bc.kBC[p] = {BCType::Dirichlet, kIn, {}, pat.name};
                bc.omegaBC[p] = {BCType::Dirichlet, omIn, {}, pat.name};
            } else if (pat.type == "outlet") {
                bc.velocityBC[p] = {BCType::Neumann, 0.0, {}, pat.name};
                bc.pressureBC[p] = {BCType::OutletPressure, 0.0, {}, pat.name};
                bc.kBC[p] = {BCType::Neumann, 0.0, {}, pat.name};
                bc.omegaBC[p] = {BCType::Neumann, 0.0, {}, pat.name};
            } else if (pat.type == "symmetry") {
                // free-slip / zero-stress boundary: no penetration (velocity
                // Symmetry removes the wall-normal component), zero gradient of
                // everything else. Not a turbulence wall: the patch type also
                // excludes it from computeWallDistance, so the SST blending and
                // wall omega treat only true walls.
                bc.velocityBC[p] = {BCType::Symmetry, 0.0, {}, pat.name};
                bc.pressureBC[p] = {BCType::Neumann, 0.0, {}, pat.name};
                bc.kBC[p] = {BCType::Neumann, 0.0, {}, pat.name};
                bc.omegaBC[p] = {BCType::Neumann, 0.0, {}, pat.name};
            } else {
                // default zero gradient
                bc.velocityBC[p].patchName = pat.name;
                bc.pressureBC[p].patchName = pat.name;
                bc.kBC[p].patchName = pat.name;
                bc.omegaBC[p].patchName = pat.name;
            }
        }
        return bc;
    }
    // backward-facing step: uses channelDefaults since all patches carry type attributes
    // (top_wall, bottom_wall_up, step_face, bottom_wall_down → wall; inlet, outlet as usual)
    static FlowBoundaryConditions bfsDefaults(const Mesh &mesh, double Uin, double kIn,
                                              double omIn) {
        return channelDefaults(mesh, Uin, kIn, omIn);
    }

    // flat plate: bottom = wall, top = freestream (Dirichlet at inlet values), inlet/outlet same as
    // channel the top boundary must be freestream, not wall — otherwise we solve a channel problem,
    // not a flat plate
    static FlowBoundaryConditions flatPlateDefaults(const Mesh &mesh, double Uinf, double kIn,
                                                    double omIn) {
        FlowBoundaryConditions bc = channelDefaults(mesh, Uinf, kIn, omIn);
        // override "top" patch from wall to freestream
        for (int p = 0; p < mesh.nPatches(); ++p) {
            const Patch &pat = mesh.patch(p);
            if (pat.name == "top") {
                bc.velocityBC[p] = {BCType::Dirichlet, 0.0, Vec3(Uinf, 0, 0), pat.name};
                bc.pressureBC[p] = {BCType::Neumann, 0.0, {}, pat.name};
                bc.kBC[p] = {BCType::Dirichlet, kIn, {}, pat.name};
                bc.omegaBC[p] = {BCType::Dirichlet, omIn, {}, pat.name};
            }
        }
        return bc;
    }

    // plane Couette: bottom wall stationary, top wall translating at Uwall in x
    // (both remain no-slip k-omega walls). Fully-developed Couette is streamwise
    // invariant, so rather than force a developing inlet/outlet flow the streamwise
    // ends are made zero-gradient (Neumann) on every field and the flow is driven
    // purely by the moving-wall shear with no mean pressure gradient. With the wall
    // velocities pinning the profile (0 at the bottom, Uwall at the top) and the
    // outlet pressure pinning the level, the solution is the x-invariant turbulent
    // Couette profile (constant total stress dU/dy(nu+nuT) = u_tau^2), reached on a
    // short domain rather than after a long, slow development length. The top wall
    // keeps its WallKOmega k/omega BC (k = 0, Menter omega); only its velocity moves.
    static FlowBoundaryConditions couetteDefaults(const Mesh &mesh, double Uwall, double kIn,
                                                  double omIn) {
        FlowBoundaryConditions bc = channelDefaults(mesh, 0.5 * Uwall, kIn, omIn);
        for (int p = 0; p < mesh.nPatches(); ++p) {
            const Patch &pat = mesh.patch(p);
            if (pat.name == "top") {
                // moving wall: tangential velocity Uwall, k/omega stay WallKOmega
                bc.velocityBC[p] = {BCType::WallMoving, 0.0, Vec3(Uwall, 0, 0), pat.name};
            } else if (pat.name == "inlet") {
                // streamwise-invariant: zero-gradient inflow (purely wall-driven)
                bc.velocityBC[p] = {BCType::Neumann, 0.0, {}, pat.name};
                bc.pressureBC[p] = {BCType::Neumann, 0.0, {}, pat.name};
                bc.kBC[p] = {BCType::Neumann, 0.0, {}, pat.name};
                bc.omegaBC[p] = {BCType::Neumann, 0.0, {}, pat.name};
            }
            // outlet keeps channelDefaults: zero-gradient velocity, fixed pressure
        }
        return bc;
    }

    // ---- per-face boundary profiles -------------------------------------
    // Attach a profile (one value per patch face, in patch-face order) to a
    // Dirichlet-like BC, so an inlet can carry U(y), k(y), omega(y). The apply
    // functions use the profile when present and fall back to the uniform value
    // otherwise. Face order and count come from mesh.patch(...).faces, exposed
    // to Python through Mesh::wall_patch_data(name).

    static int patchIndex(const Mesh &mesh, const std::string &name) {
        for (int p = 0; p < mesh.nPatches(); ++p)
            if (mesh.patch(p).name == name)
                return p;
        throw std::runtime_error("FlowBoundaryConditions: unknown patch '" + name + "'");
    }

    void setVelocityProfile(const Mesh &mesh, const std::string &name,
                            const std::vector<Vec3> &vals) {
        if (velocityBC.size() != static_cast<size_t>(mesh.nPatches()))
            throw std::invalid_argument("boundary conditions require one entry per mesh patch");
        const int p = patchIndex(mesh, name);
        if (vals.size() != mesh.patch(p).faces.size())
            throw std::runtime_error("velocity profile size != patch face count");
        auto candidate = velocityBC;
        candidate[p].vecProfile = vals;
        validatePatchBCs(mesh, candidate, BoundaryValueKind::Vector);
        velocityBC = std::move(candidate);
    }
    void setKProfile(const Mesh &mesh, const std::string &name, const std::vector<double> &vals) {
        if (kBC.size() != static_cast<size_t>(mesh.nPatches()))
            throw std::invalid_argument("boundary conditions require one entry per mesh patch");
        const int p = patchIndex(mesh, name);
        if (vals.size() != mesh.patch(p).faces.size())
            throw std::runtime_error("k profile size != patch face count");
        auto candidate = kBC;
        candidate[p].scalarProfile = vals;
        validatePatchBCs(mesh, candidate, BoundaryValueKind::Scalar);
        kBC = std::move(candidate);
    }
    void setOmegaProfile(const Mesh &mesh, const std::string &name,
                         const std::vector<double> &vals) {
        if (omegaBC.size() != static_cast<size_t>(mesh.nPatches()))
            throw std::invalid_argument("boundary conditions require one entry per mesh patch");
        const int p = patchIndex(mesh, name);
        if (vals.size() != mesh.patch(p).faces.size())
            throw std::runtime_error("omega profile size != patch face count");
        auto candidate = omegaBC;
        candidate[p].scalarProfile = vals;
        validatePatchBCs(mesh, candidate, BoundaryValueKind::Scalar);
        omegaBC = std::move(candidate);
    }
};

// Apply functions

// Apply velocity boundary conditions to boundary faces
// inputs: U (velocity vector field), mesh, and BCs
inline void applyVelocityBC(VectorField &U, const Mesh &mesh, const FlowBoundaryConditions &bcs) {
    validatePatchBCs(mesh, bcs.velocityBC, BoundaryValueKind::Vector);
    for (int p = 0; p < mesh.nPatches(); ++p) {         // loops over every boundary patch
        const Patch &pat = mesh.patch(p);               // faces that this patch belongs to
        const PatchBC &bc = bcs.velocityBC[p];          // velocity BC kind
        for (size_t k = 0; k < pat.faces.size(); ++k) { // loops over faces in patch
            FaceID fi = pat.faces[k];
            switch (bc.type) {
            case BCType::WallNoSlip:
                U.bface(fi) = Vec3(0, 0, 0);
                break;               // enforces u = 0 for no-slip walls
            case BCType::WallMoving: // moving wall: u = wall velocity (no-slip,
                                     // but the wall translates tangentially)
                U.bface(fi) = bc.vecValue;
                break;
            case BCType::InletVelocity: // enforces u = U_in (profile-aware)
            case BCType::Dirichlet:     // enforces u = u_boundary
                U.bface(fi) = bc.vecProfile.empty() ? bc.vecValue : bc.vecProfile[k];
                break;
            case BCType::Symmetry: { // enforces no normal velocity and unchanged tangential
                                     // velocity
                Vec3 Uo = U[mesh.face(fi).owner];
                U.bface(fi) = Uo - mesh.face(fi).normal * Uo.dot(mesh.face(fi).normal);
                break;
            }
            default:
                U.bface(fi) = U[mesh.face(fi).owner];
                break; // default case
            }
        }
    }
}

// Apply pressure boundary conditions to boundary faces
// inputs: pf (scalar pressure field), mesh, and BCs
inline void applyPressureBC(ScalarField &pf, const Mesh &mesh, const FlowBoundaryConditions &bcs) {
    validatePatchBCs(mesh, bcs.pressureBC, BoundaryValueKind::ScalarUniform);
    for (int p = 0; p < mesh.nPatches(); ++p) {
        const Patch &pat = mesh.patch(p);
        const PatchBC &bc = bcs.pressureBC[p];
        for (FaceID fi : pat.faces) {
            switch (bc.type) {
            case BCType::OutletPressure:
            case BCType::Dirichlet:
                pf.bface(fi) = bc.value;
                break;
            default:
                pf.bface(fi) = pf[mesh.face(fi).owner];
                break; // default case
            }
        }
    }
}

// Apply turbulence kinetic energy boundary conditions to boundary faces
// inputs: k (turbulence kinetic energy field), mesh, and BCs
inline void applyKBC(ScalarField &k, const Mesh &mesh, const FlowBoundaryConditions &bcs) {
    validatePatchBCs(mesh, bcs.kBC, BoundaryValueKind::Scalar);
    for (int p = 0; p < mesh.nPatches(); ++p) {
        const Patch &pat = mesh.patch(p);
        const PatchBC &bc = bcs.kBC[p];
        for (size_t j = 0; j < pat.faces.size(); ++j) {
            FaceID fi = pat.faces[j];
            switch (bc.type) {
            case BCType::WallKOmega:
                k.bface(fi) = 0.0;
                break;              // enforces k = 0 at wall (u = 0 at no slip wall)
            case BCType::Dirichlet: // enforces k = k_in (profile-aware)
                k.bface(fi) = bc.scalarProfile.empty() ? bc.value : bc.scalarProfile[j];
                break;
            default:
                k.bface(fi) = k[mesh.face(fi).owner];
                break;
            }
        }
    }
}

// Menter (1994) k-omega SST model: omega_wall = 60*nu / (beta1 * y1^2)
// Apply boundary conditions to specific dissipation rate omega
// inputs: omega (dissipation scalar field), mesh, BCs, nu (kinematic viscosity), beta1 (default
// 0.075) enforces the k-omega boundary condition system
inline void applyOmegaBC(ScalarField &omega, const Mesh &mesh, const FlowBoundaryConditions &bcs,
                         double nu, double beta1 = 0.075) {
    validatePatchBCs(mesh, bcs.omegaBC, BoundaryValueKind::Scalar);
    for (int p = 0; p < mesh.nPatches(); ++p) {
        const Patch &pat = mesh.patch(p);
        const PatchBC &bc = bcs.omegaBC[p];
        for (size_t j = 0; j < pat.faces.size(); ++j) {
            FaceID fi = pat.faces[j];
            switch (bc.type) {
            case BCType::WallKOmega: { // near-wall omega formula Menter (1994)
                double y1 = std::max(mesh.face(fi).delta, 1e-20);
                omega.bface(fi) = 60.0 * nu / (beta1 * y1 * y1);
                break;
            }
            case BCType::Dirichlet: // enforces omega = omega_in (profile-aware)
                omega.bface(fi) = bc.scalarProfile.empty() ? bc.value : bc.scalarProfile[j];
                break;
            default:
                omega.bface(fi) = omega[mesh.face(fi).owner];
                break;
            }
        }
    }
}

// Per-cell-viscosity overload of applyOmegaBC: wall faces use the OWNER-cell
// kinematic viscosity (compressible mu(T)/rho), matching the local wall
// anchor treatment; the other BC types are viscosity-independent.
inline void applyOmegaBC(ScalarField &omega, const Mesh &mesh, const FlowBoundaryConditions &bcs,
                         const ScalarField &nuLocal, double beta1 = 0.075) {
    validatePatchBCs(mesh, bcs.omegaBC, BoundaryValueKind::Scalar);
    for (int p = 0; p < mesh.nPatches(); ++p) {
        const Patch &pat = mesh.patch(p);
        const PatchBC &bc = bcs.omegaBC[p];
        for (size_t j = 0; j < pat.faces.size(); ++j) {
            FaceID fi = pat.faces[j];
            switch (bc.type) {
            case BCType::WallKOmega: {
                double y1 = std::max(mesh.face(fi).delta, 1e-20);
                double nuO = nuLocal[mesh.face(fi).owner];
                omega.bface(fi) = 60.0 * nuO / (beta1 * y1 * y1);
                break;
            }
            case BCType::Dirichlet:
                omega.bface(fi) = bc.scalarProfile.empty() ? bc.value : bc.scalarProfile[j];
                break;
            default:
                omega.bface(fi) = omega[mesh.face(fi).owner];
                break;
            }
        }
    }
}

// Applies all boundary conditions
inline void applyAllBCs(VectorField &U, ScalarField &p, ScalarField &k, ScalarField &omega,
                        const Mesh &mesh, const FlowBoundaryConditions &bcs, double nu) {
    applyVelocityBC(U, mesh, bcs);
    applyPressureBC(p, mesh, bcs);
    applyKBC(k, mesh, bcs);
    applyOmegaBC(omega, mesh, bcs, nu);
}
