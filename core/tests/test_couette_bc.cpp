// Moving-wall (Couette) boundary-condition test.
//
// Asserts that FlowBoundaryConditions::couetteDefaults builds the streamwise-
// invariant Couette setup and that applyVelocityBC realises the moving wall:
//   1. The "top" patch velocity BC is WallMoving with the prescribed wall speed,
//      while its k and omega BCs stay WallKOmega (it is still a no-slip k-omega
//      wall, only translating).
//   2. The "bottom" patch stays a stationary WallNoSlip wall.
//   3. The streamwise ends ("inlet") are zero-gradient (Neumann) so the flow is
//      driven purely by the moving wall (fully-developed Couette is x-invariant).
//   4. applyVelocityBC sets the top-wall face velocity to (Uwall,0,0) and the
//      bottom-wall face velocity to (0,0,0).

#include "Mesh.hpp"
#include "Field.hpp"
#include "BoundaryCondition.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>

namespace {

#define REQUIRE(cond, msg)                                                                         \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            std::fprintf(stderr, "FAIL [%s:%d] %s\n  required: %s\n", __FILE__, __LINE__, (msg),   \
                         #cond);                                                                   \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

const PatchBC &velBC(const FlowBoundaryConditions &bc, const Mesh &mesh, const std::string &name) {
    for (int p = 0; p < mesh.nPatches(); ++p)
        if (mesh.patch(p).name == name)
            return bc.velocityBC[p];
    std::fprintf(stderr, "patch %s not found\n", name.c_str());
    std::exit(1);
}

} // namespace

int main() {
    const double Uwall = 2.0, kIn = 1e-3, omIn = 100.0;
    Mesh mesh = Mesh::makeChannel2D(8, 16, 2.0, 2.0);
    mesh.computeWallDistance();

    FlowBoundaryConditions bc = FlowBoundaryConditions::couetteDefaults(mesh, Uwall, kIn, omIn);

    // 1-3. BC structure
    const PatchBC &top = velBC(bc, mesh, "top");
    REQUIRE(top.type == BCType::WallMoving, "top wall is not WallMoving");
    REQUIRE(std::fabs(top.vecValue.x - Uwall) < 1e-12, "top wall speed mismatch");
    REQUIRE(std::fabs(top.vecValue.y) < 1e-12 && std::fabs(top.vecValue.z) < 1e-12,
            "top wall velocity must be purely streamwise");
    REQUIRE(velBC(bc, mesh, "bottom").type == BCType::WallNoSlip,
            "bottom wall is not stationary WallNoSlip");
    REQUIRE(velBC(bc, mesh, "inlet").type == BCType::Neumann,
            "inlet is not zero-gradient (streamwise-invariant)");
    for (int p = 0; p < mesh.nPatches(); ++p) {
        if (mesh.patch(p).name == "top") {
            REQUIRE(bc.kBC[p].type == BCType::WallKOmega, "moving wall must keep WallKOmega k BC");
            REQUIRE(bc.omegaBC[p].type == BCType::WallKOmega,
                    "moving wall must keep WallKOmega omega BC");
        }
    }

    // 4. applyVelocityBC realises the wall velocities on the boundary faces
    VectorField U(mesh, "U", Vec3(0, 0, 0));
    applyVelocityBC(U, mesh, bc);
    int n_top = 0, n_bottom = 0;
    for (int p = 0; p < mesh.nPatches(); ++p) {
        const Patch &pat = mesh.patch(p);
        for (FaceID fi : pat.faces) {
            if (pat.name == "top") {
                REQUIRE(std::fabs(U.bface(fi).x - Uwall) < 1e-12,
                        "top-wall face velocity != Uwall");
                ++n_top;
            } else if (pat.name == "bottom") {
                REQUIRE(std::fabs(U.bface(fi).x) < 1e-12, "bottom-wall face velocity != 0");
                ++n_bottom;
            }
        }
    }
    REQUIRE(n_top > 0 && n_bottom > 0, "wall patches have no faces");

    std::printf("test_couette_bc: moving-wall BC OK (top faces=%d, bottom faces=%d)\n", n_top,
                n_bottom);
    return 0;
}
