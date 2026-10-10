// Le-Moin BFS boundary-condition test: free-slip top and profile inflow.
//
// The Le-Moin data shows zero mean shear at the top edge of every station (a
// free-slip boundary, not a wall) and an inflow boundary layer, so the BFS
// baseline retypes top_wall to "symmetry" and prescribes per-face inlet
// profiles. Asserts:
//   1. setPatchType retypes top_wall and computeWallDistance then measures to
//      the true walls only (a cell under the top boundary is far from a wall).
//   2. bfsDefaults on the retyped mesh assigns the symmetry BCs to the top
//      (velocity Symmetry, k/omega/pressure Neumann) and walls stay WallNoSlip.
//   3. applyVelocityBC at a symmetry face removes the wall-normal component and
//      keeps the tangential one.
//   4. A per-face inlet profile overrides the uniform inlet value in
//      applyVelocityBC / applyKBC / applyOmegaBC, in patch-face order.

#include "Mesh.hpp"
#include "Field.hpp"
#include "BoundaryCondition.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>

namespace {

#define REQUIRE(cond, msg)                                                                         \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            std::fprintf(stderr, "FAIL [%s:%d] %s\n  required: %s\n", __FILE__, __LINE__, (msg),   \
                         #cond);                                                                   \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

int patchIdx(const Mesh &mesh, const std::string &name) {
    for (int p = 0; p < mesh.nPatches(); ++p)
        if (mesh.patch(p).name == name)
            return p;
    std::fprintf(stderr, "patch %s not found\n", name.c_str());
    std::exit(1);
}

} // namespace

int main() {
    // small Le-Moin-shaped BFS: step h=1, total height 6, Lu=4, Ld=10
    Mesh mesh = Mesh::makeBackwardFacingStep2D(8, 12, 10, 8, 4.0, 10.0, 1.0, 6.0);

    // 1. retype the top and recompute wall distance
    mesh.setPatchType("top_wall", "symmetry");
    mesh.computeWallDistance();
    REQUIRE(mesh.patch(patchIdx(mesh, "top_wall")).type == "symmetry", "top_wall not retyped");
    // owner cell of a top face: its wall distance must be measured to the
    // bottom walls (order of the domain height), not to the adjacent top face
    const Patch &top = mesh.patch(patchIdx(mesh, "top_wall"));
    int topOwner = mesh.face(top.faces[top.faces.size() / 2]).owner;
    REQUIRE(mesh.wallDistance()[topOwner] > 1.0, "top-adjacent cell still sees the top as a wall");

    // 2. BC factory honours the symmetry type
    const double Uin = 1.0, kIn = 1e-3, omIn = 10.0;
    FlowBoundaryConditions bc = FlowBoundaryConditions::bfsDefaults(mesh, Uin, kIn, omIn);
    int pTop = patchIdx(mesh, "top_wall");
    REQUIRE(bc.velocityBC[pTop].type == BCType::Symmetry, "top velocity BC is not Symmetry");
    REQUIRE(bc.kBC[pTop].type == BCType::Neumann, "top k BC is not Neumann");
    REQUIRE(bc.omegaBC[pTop].type == BCType::Neumann, "top omega BC is not Neumann");
    REQUIRE(bc.velocityBC[patchIdx(mesh, "bottom_wall_down")].type == BCType::WallNoSlip,
            "bottom wall is not WallNoSlip");

    // 3. symmetry face: no penetration, tangential preserved
    VectorField U(mesh, "U");
    U.setUniform(Vec3(0.7, 0.3, 0.0)); // owner cells carry a diagonal velocity
    applyVelocityBC(U, mesh, bc);
    FaceID fTop = top.faces[0];
    REQUIRE(std::fabs(U.bface(fTop).y) < 1e-14, "symmetry face has wall-normal velocity");
    REQUIRE(std::fabs(U.bface(fTop).x - 0.7) < 1e-14, "symmetry face lost the tangential velocity");

    // 4. per-face inlet profiles override the uniform values
    const Patch &inlet = mesh.patch(patchIdx(mesh, "inlet"));
    int nf = static_cast<int>(inlet.faces.size());
    std::vector<Vec3> uProf(nf);
    std::vector<double> kProf(nf), omProf(nf);
    for (int j = 0; j < nf; ++j) {
        uProf[j] = Vec3(0.1 * j, 0.0, 0.0);
        kProf[j] = 1e-4 * (j + 1);
        omProf[j] = 2.0 * (j + 1);
    }
    bc.setVelocityProfile(mesh, "inlet", uProf);
    bc.setKProfile(mesh, "inlet", kProf);
    bc.setOmegaProfile(mesh, "inlet", omProf);

    ScalarField k(mesh, "k"), om(mesh, "omega");
    k.setUniform(9.0);
    om.setUniform(9.0);
    applyVelocityBC(U, mesh, bc);
    applyKBC(k, mesh, bc);
    applyOmegaBC(om, mesh, bc, 1e-4);
    for (int j = 0; j < nf; ++j) {
        FaceID fi = inlet.faces[j];
        REQUIRE(std::fabs(U.bface(fi).x - 0.1 * j) < 1e-14,
                "inlet velocity profile not applied in patch-face order");
        REQUIRE(std::fabs(k.bface(fi) - 1e-4 * (j + 1)) < 1e-18, "inlet k profile not applied");
        REQUIRE(std::fabs(om.bface(fi) - 2.0 * (j + 1)) < 1e-12, "inlet omega profile not applied");
    }

    std::printf("test_bfs_lemoin_bc: all assertions passed\n");
    return 0;
}
