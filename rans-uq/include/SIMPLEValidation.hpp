#pragma once

#include "BoundaryCondition.hpp"
#include "FlowFields.hpp"
#include <cmath>
#include <stdexcept>

// Shared admission for direct solves, forward models, and batch-owned models.
// These are dimensional physical domains, not calibration bounds or floors.
namespace simple_admission {
inline bool finite(const Vec3 &value) {
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}
inline void turbulenceBoundary(const std::vector<PatchBC> &boundaries) {
    // A zero prescribed value is admitted for a laminar or floored boundary;
    // negative k and omega are not physical boundary data. Only Dirichlet
    // values are consumed here; nonempty profiles override uniform storage.
    for (const auto &boundary : boundaries) {
        if (boundary.type != BCType::Dirichlet)
            continue;
        if (boundary.scalarProfile.empty()) {
            if (boundary.value < 0)
                throw std::invalid_argument("SIMPLE prescribed turbulence must be nonnegative");
        } else {
            for (double value : boundary.scalarProfile)
                if (value < 0)
                    throw std::invalid_argument(
                        "SIMPLE prescribed turbulence profile must be nonnegative");
        }
    }
}
inline void problem(const Mesh &mesh, const FlowBoundaryConditions &boundaries, double nu,
                    const SolverSettings &settings) {
    if (!std::isfinite(nu) || nu <= 0)
        throw std::invalid_argument("SIMPLE kinematic viscosity must be finite and positive");
    settings.validate();
    validatePatchBCs(mesh, boundaries.velocityBC, BoundaryValueKind::Vector);
    validatePatchBCs(mesh, boundaries.pressureBC, BoundaryValueKind::ScalarUniform);
    validatePatchBCs(mesh, boundaries.kBC, BoundaryValueKind::Scalar);
    validatePatchBCs(mesh, boundaries.omegaBC, BoundaryValueKind::Scalar);
    turbulenceBoundary(boundaries.kBC);
    turbulenceBoundary(boundaries.omegaBC);
}
inline void fields(const Mesh &mesh, const FlowFields &state) {
    requireFieldStorage(mesh, state.U);
    for (const ScalarField *field : {&state.p, &state.k, &state.omega, &state.nuT, &state.F1,
                                     &state.F2, &state.Pk, &state.CDkw})
        requireFieldStorage(mesh, *field);
    // Check this prerequisite before an initializer writes any primary state.
    (void)mesh.wallDistance();
}
inline void initial(const Vec3 &velocity, double pressure, double k, double omega) {
    if (!finite(velocity) || !std::isfinite(pressure))
        throw std::invalid_argument("SIMPLE initial velocity and pressure must be finite");
    if (!std::isfinite(k) || k < 0)
        throw std::invalid_argument("SIMPLE initial k must be finite and nonnegative");
    if (!std::isfinite(omega) || omega <= 0)
        throw std::invalid_argument("SIMPLE initial omega must be finite and positive");
}
} // namespace simple_admission
