// Regression test: Mesh::makeBackwardFacingStep2D
//
// Asserts that the BFS factory produces:
//   1. Exactly the expected cell, internal-face, and boundary-face counts.
//   2. All six named patches with non-empty face lists.
//   3. No degenerate cells (volume > 0) or zero-area faces.
//   4. Owner indices in [0, nCells), neighbor indices in [0, nCells) for
//      internal faces and -1 for boundary faces.
//   5. Wall distance is finite and non-negative for every cell.
//
// This catches regressions in the mesh factory before any solver/observation
// operator code runs against it.

#include "Mesh.hpp"
#include <cstdio>
#include <cstdlib>
#include <set>
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

void check_bfs_geometry(int nx_up, int nx_down, int ny_up, int ny_down, double Lu, double Ld,
                        double h_s, double H, double Re, double yPlusTarget) {
    Mesh mesh = Mesh::makeBackwardFacingStep2D(nx_up, nx_down, ny_up, ny_down, Lu, Ld, h_s, H, Re,
                                               yPlusTarget);

    const int nx_tot = nx_up + nx_down;
    const int expected_cells = nx_tot * ny_up + nx_down * ny_down;
    const int expected_internal = nx_tot * (ny_up - 1)       // horizontal upper
                                  + nx_down * (ny_down - 1)  // horizontal lower
                                  + nx_down                  // y=h_s coupling row
                                  + (nx_tot - 1) * ny_up     // vertical upper
                                  + (nx_down - 1) * ny_down; // vertical lower
    const int expected_boundary = nx_tot                     // top_wall
                                  + nx_up                    // bottom_wall_up
                                  + ny_down                  // step_face
                                  + nx_down                  // bottom_wall_down
                                  + ny_up                    // inlet
                                  + (ny_up + ny_down);       // outlet (upper + lower)

    REQUIRE(mesh.nCells() == expected_cells, "cell count mismatch");
    REQUIRE(mesh.nInternalFaces() == expected_internal, "internal face count mismatch");
    REQUIRE(mesh.nFaces() == expected_internal + expected_boundary, "total face count mismatch");

    // Six named boundary patches, in the documented order.
    const std::vector<std::string> expected_patches = {
        "top_wall", "bottom_wall_up", "step_face", "bottom_wall_down", "inlet", "outlet"};
    REQUIRE((int)mesh.patches().size() == (int)expected_patches.size(), "wrong number of patches");
    std::set<std::string> seen;
    for (const auto &p : mesh.patches()) {
        REQUIRE(!p.faces.empty(), "patch has no faces");
        seen.insert(p.name);
    }
    for (const auto &name : expected_patches) {
        REQUIRE(seen.count(name) == 1, ("missing patch " + name).c_str());
        const int pid = mesh.patchByName(name);
        REQUIRE(pid >= 0, ("patchByName failed for " + name).c_str());
    }

    // All cells positive volume; first-cell reuse of the cell vector yields
    // wallDist_.empty() until computeWallDistance() is called, so do that.
    mesh.computeWallDistance();
    const auto &wd = mesh.wallDistance();
    REQUIRE((int)wd.size() == mesh.nCells(), "wall distance vector size mismatch");

    int n_zero_vol = 0, n_zero_area = 0, n_neg_wd = 0, n_bad_owner = 0;
    for (int ci = 0; ci < mesh.nCells(); ++ci) {
        if (!(mesh.cell(ci).volume > 0.0))
            ++n_zero_vol;
        if (!(wd[ci] >= 0.0) || !std::isfinite(wd[ci]))
            ++n_neg_wd;
    }
    for (int fi = 0; fi < mesh.nFaces(); ++fi) {
        const Face &f = mesh.face(fi);
        if (!(f.area > 0.0))
            ++n_zero_area;
        if (f.owner < 0 || f.owner >= mesh.nCells())
            ++n_bad_owner;
        if (fi < mesh.nInternalFaces()) {
            REQUIRE(f.neighbor >= 0 && f.neighbor < mesh.nCells(),
                    "internal face neighbor out of range");
        } else {
            REQUIRE(f.neighbor < 0, "boundary face has neighbor");
            REQUIRE(f.patchID >= 0, "boundary face missing patchID");
        }
    }
    REQUIRE(n_zero_vol == 0, "found cells with non-positive volume");
    REQUIRE(n_zero_area == 0, "found faces with non-positive area");
    REQUIRE(n_neg_wd == 0, "found cells with non-finite or negative wall distance");
    REQUIRE(n_bad_owner == 0, "found faces with out-of-range owner");

    std::printf("  bfs(%d+%d, %d+%d, Re=%.1f, y+=%.1f): cells=%d  faces=%d  "
                "internal=%d  patches=%d  OK\n",
                nx_up, nx_down, ny_up, ny_down, Re, yPlusTarget, mesh.nCells(), mesh.nFaces(),
                mesh.nInternalFaces(), (int)mesh.patches().size());
}

} // namespace

int main() {
    // Driver-Seegmiller geometry, fast 18x10/12x8 mesh suitable for testing.
    check_bfs_geometry(/*nx_up=*/6, /*nx_down=*/12,
                       /*ny_up=*/8, /*ny_down=*/6,
                       /*Lu=*/6.0, /*Ld=*/20.0,
                       /*h_s=*/1.0, /*H=*/3.0,
                       /*Re=*/37600.0, /*yPlus=*/1.0);

    // Larger mesh used by the BFS examples.
    check_bfs_geometry(20, 40, 20, 15, 6.0, 20.0, 1.0, 3.0, 37600.0, 1.0);

    // No-Re-clustering overload: stretch_up = stretch_low = 2 in factory.
    Mesh m = Mesh::makeBackwardFacingStep2D(8, 16, 8, 6, 6.0, 20.0, 1.0, 3.0);
    REQUIRE(m.nCells() == (8 + 16) * 8 + 16 * 6, "default-overload cell count");

    std::printf("test_mesh_bfs: all checks passed\n");
    return 0;
}
