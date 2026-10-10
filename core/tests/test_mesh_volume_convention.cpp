// One cell-geometry convention for every mesh path.
//
// The generators write exact cell volumes (shoelace area times depth) and
// centroids, while the general path Mesh::computeGeometry() used to compute
// V = |sum_f S_f . (c_f - c_C)| / 3, which is 2/3 of the true volume for the
// z-open 2D cells the generators build, and to place cell centres at the
// average of the face centres. It also rebuilt periodic wrap faces from the
// raw neighbour centre, giving them a delta of the whole domain length.
// computeGeometry() is public and documented as "call after modifying
// topology", so any caller silently got the second convention (the DBNS
// tests call it, which is why DBNS kept a private volume array).
//
// This test pins:
//   1. computeGeometry() leaves every generator's geometry unchanged
//      (volumes, centroids, face distance vectors, deltas and weights),
//      including the periodic wrap faces;
//   2. total volume equals the analytic domain volume;
//   3. an OpenFOAM polyMesh with empty front/back faces and non-rectangular
//      cells loads with exact prism volumes and polygon centroids.

#include "Mesh.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
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

struct Geometry {
    std::vector<double> volume, delta, weight;
    std::vector<Vec3> center, d;
};

Geometry snapshot(const Mesh &m) {
    Geometry g;
    for (int ci = 0; ci < m.nCells(); ++ci) {
        g.volume.push_back(m.cell(ci).volume);
        g.center.push_back(m.cell(ci).center);
    }
    for (int fi = 0; fi < m.nFaces(); ++fi) {
        g.delta.push_back(m.face(fi).delta);
        g.d.push_back(m.face(fi).d);
        g.weight.push_back(m.face(fi).weight);
    }
    return g;
}

// computeGeometry() must reproduce the generator's own geometry to round-off
void requireIdempotent(Mesh m, const char *name, double expectTotal) {
    const Geometry before = snapshot(m);
    double total = 0.0, scale = 0.0;
    for (int ci = 0; ci < m.nCells(); ++ci) {
        total += before.volume[ci];
        scale = std::max(scale, before.center[ci].norm());
    }
    std::printf("  %s: %d cells, total volume %.15g (expected %.15g)\n", name, m.nCells(), total,
                expectTotal);
    REQUIRE(std::fabs(total - expectTotal) < 1e-12 * expectTotal,
            "generator total volume must equal the domain volume");

    m.computeGeometry();
    const Geometry after = snapshot(m);
    double worstV = 0.0, worstC = 0.0, worstD = 0.0, worstW = 0.0;
    for (int ci = 0; ci < m.nCells(); ++ci) {
        worstV = std::max(worstV, std::fabs(after.volume[ci] / before.volume[ci] - 1.0));
        worstC = std::max(worstC, (after.center[ci] - before.center[ci]).norm() / scale);
    }
    for (int fi = 0; fi < m.nFaces(); ++fi) {
        worstD = std::max(worstD, std::fabs(after.delta[fi] / before.delta[fi] - 1.0));
        worstD = std::max(worstD, (after.d[fi] - before.d[fi]).norm() / before.delta[fi]);
        worstW = std::max(worstW, std::fabs(after.weight[fi] - before.weight[fi]));
    }
    std::printf("    after computeGeometry(): max rel dV %.2e, dC %.2e, dDelta %.2e, dW %.2e\n",
                worstV, worstC, worstD, worstW);
    REQUIRE(worstV < 1e-12, "computeGeometry() must not change cell volumes");
    REQUIRE(worstC < 1e-12, "computeGeometry() must not move cell centroids");
    REQUIRE(worstD < 1e-12, "computeGeometry() must not change face deltas");
    REQUIRE(worstW < 1e-12, "computeGeometry() must not change interpolation weights");
}

// ---- a two-cell OpenFOAM polyMesh, depth 0.5 --------------------------------
// Cell 0 has corners (0,0) (1,0) (1,1.5) (0,1), cell 1 is its mirror image
// about x = 1. Front (z=0) and back (z=0.5) faces form an "empty" patch, so
// the cells are closed prisms.
void writeFile(const std::filesystem::path &p, const std::string &s) {
    std::ofstream os(p);
    os << s;
}

std::filesystem::path writeTwoCellPolyMesh() {
    namespace fs = std::filesystem;
    // create_directory is atomic: a concurrent process skips an existing
    // candidate rather than sharing its fixture. No POSIX-only API is needed.
    fs::path dir;
    for (int candidate = 0; candidate < 1024; ++candidate) {
        auto path = fs::temp_directory_path() / ("cfdr_two_cell_" + std::to_string(candidate));
        if (fs::create_directory(path)) {
            dir = path;
            break;
        }
    }
    if (dir.empty())
        throw std::runtime_error("could not create mesh fixture directory");
    writeFile(dir / "points",
              "12\n(\n"
              "(0 0 0)\n(1 0 0)\n(2 0 0)\n(0 1 0)\n(1 1.5 0)\n(2 1 0)\n"
              "(0 0 0.5)\n(1 0 0.5)\n(2 0 0.5)\n(0 1 0.5)\n(1 1.5 0.5)\n(2 1 0.5)\n)\n");
    // node orders give right-hand normals pointing out of the owner
    writeFile(dir / "faces",
              "11\n(\n"
              "4(1 4 10 7)\n"                    // 0 internal, +x
              "4(0 1 7 6)\n4(1 2 8 7)\n"         // 1-2 bottom, -y
              "4(3 9 10 4)\n4(4 10 11 5)\n"      // 3-4 top (slanted)
              "4(0 6 9 3)\n"                     // 5 left, -x
              "4(2 5 11 8)\n"                    // 6 right, +x
              "4(0 3 4 1)\n4(1 4 5 2)\n"         // 7-8 front, -z
              "4(6 7 10 9)\n4(7 8 11 10)\n)\n"); // 9-10 back, +z
    writeFile(dir / "owner", "11\n(\n0\n0\n1\n0\n1\n0\n1\n0\n1\n0\n1\n)\n");
    writeFile(dir / "neighbour", "1\n(\n1\n)\n");
    writeFile(dir / "boundary", "5\n(\n"
                                "bottom\n{\n type wall;\n nFaces 2;\n startFace 1;\n}\n"
                                "top\n{\n type wall;\n nFaces 2;\n startFace 3;\n}\n"
                                "left\n{\n type inlet;\n nFaces 1;\n startFace 5;\n}\n"
                                "right\n{\n type outlet;\n nFaces 1;\n startFace 6;\n}\n"
                                "frontAndBack\n{\n type empty;\n nFaces 4;\n startFace 7;\n}\n)\n");
    return dir;
}

// polygon area and centroid by the shoelace formulas
void polygon(const std::vector<double> &x, const std::vector<double> &y, double &A, double &cx,
             double &cy) {
    double a2 = 0.0, sx = 0.0, sy = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        size_t j = (i + 1) % x.size();
        double cr = x[i] * y[j] - x[j] * y[i];
        a2 += cr;
        sx += (x[i] + x[j]) * cr;
        sy += (y[i] + y[j]) * cr;
    }
    A = 0.5 * a2;
    cx = sx / (3.0 * a2);
    cy = sy / (3.0 * a2);
}

} // namespace

int main() {
    // 1-2. generators: uniform, wall-clustered, L-shaped, curved periodic
    requireIdempotent(Mesh::makeChannel2D(6, 5, 2.0, 1.0), "channel uniform", 2.0);
    requireIdempotent(Mesh::makeChannel2D(6, 8, 3.0, 2.0, 2000.0, 1.0), "channel clustered", 6.0);
    requireIdempotent(Mesh::makePlate2D(6, 8, 3.0, 2.0, 1.0e5, 1.0), "plate clustered", 6.0);
    requireIdempotent(Mesh::makeBackwardFacingStep2D(4, 6, 3, 2, 2.0, 5.0, 1.0, 3.0, 5000.0, 1.0),
                      "backward-facing step", 2.0 * 2.0 + 5.0 * 3.0);
    {
        const int nx = 12, ny = 6;
        const double Lx = 4.5, yTop = 3.0, bump = 1.0;
        std::vector<double> xN(nx + 1), yB(nx + 1);
        for (int i = 0; i <= nx; ++i) {
            xN[i] = Lx * i / nx;
            yB[i] = 0.5 * bump * (1.0 + std::cos(2.0 * M_PI * xN[i] / Lx));
        }
        double exact = 0.0; // trapezoidal integral of the gap (straight-sided cells)
        for (int i = 0; i < nx; ++i)
            exact += 0.5 * ((yTop - yB[i]) + (yTop - yB[i + 1])) * (xN[i + 1] - xN[i]);
        requireIdempotent(Mesh::makeCurvedChannelPeriodic2D(xN, yB, yTop, ny, 0.0, 1.0),
                          "cosine-bump periodic", exact);
    }

    // 3. OpenFOAM polyMesh: closed prisms with trapezoidal cross-sections
    {
        auto dir = writeTwoCellPolyMesh();
        Mesh m = Mesh::loadFromFile(dir.string());
        REQUIRE(m.nCells() == 2, "two cells");
        double A, cx, cy;
        polygon({0, 1, 1, 0}, {0, 0, 1.5, 1}, A, cx, cy);
        const double dz = 0.5;
        const Cell &c0 = m.cell(0);
        const Cell &c1 = m.cell(1);
        std::printf("  polyMesh: V0 %.15g V1 %.15g (exact %.15g); c0 (%.15g, %.15g, %.15g) "
                    "(exact %.15g, %.15g, 0.25)\n",
                    c0.volume, c1.volume, A * dz, c0.center.x, c0.center.y, c0.center.z, cx, cy);
        REQUIRE(std::fabs(c0.volume - A * dz) < 1e-14, "cell 0 volume = area * depth");
        REQUIRE(std::fabs(c1.volume - A * dz) < 1e-14, "cell 1 volume = area * depth");
        REQUIRE(std::fabs(c0.center.x - cx) < 1e-14 && std::fabs(c0.center.y - cy) < 1e-14 &&
                    std::fabs(c0.center.z - 0.25) < 1e-14,
                "cell 0 centroid");
        REQUIRE(std::fabs(c1.center.x - (2.0 - cx)) < 1e-14 && std::fabs(c1.center.y - cy) < 1e-14,
                "cell 1 centroid (mirror image)");
        std::filesystem::remove_all(dir);
    }

    std::printf("test_mesh_volume_convention: all checks passed\n");
    return 0;
}
