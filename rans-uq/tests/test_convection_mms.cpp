// Order of accuracy of SIMPLE convection by manufactured solution.
//
// SIMPLE convected U, k and omega with first-order upwind only, so on
// separated flows its truncation error, not the closure, could carry part of
// any model-data discrepancy. This test measures the observed order of the
// laminar momentum discretisation on a convection-dominated manufactured
// solution, on an x-periodic channel with no-slip walls (tanh-clustered
// mesh, 16x8, 32x16, 64x32):
//   u = 4 Um y (H - y) / H^2 + A (pi/H) sin(2 pi y / H) cos(k x)
//   v = A k sin^2(pi y / H) sin(k x)            (divergence-free, zero on walls)
//   p = P cos(k x) cos(2 pi y / H)              (dp/dn = 0 on walls)
// with the momentum source S = (u.grad)u + grad p - nu lap(u) added per cell,
// here with Um = 0 and nu = 0.005 so convection dominates (cell Peclet up to
// ~8 on the coarsest mesh). Requires, for linear upwind and for the limited
// (van Albada) variant, observed order >= 1.9 on the finest pair and on
// average, and an error at least 10x below upwind's; upwind is reported for
// contrast.
//
// With a Poiseuille base flow added (Um = 1, nu = 0.01) the observed order
// over 16x8 ... 128x64 is 2.02, 1.66, 1.72 while the base flow alone (1.96,
// 1.99), the perturbation alone (2.10, 2.05) and the diffusion-dominated
// case (1.89, 1.99) are each second order: the error ratios grow toward 4
// (3.16, 3.29) as their components, which partly cancel, separate. That case
// is a poor order test, not evidence of a first-order term.

#include "BoundaryCondition.hpp"
#include "Field.hpp"
#include "FlowFields.hpp"
#include "Mesh.hpp"
#include "SIMPLESolver.hpp"
#include "SSTModel.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
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

const double Lx = 2.0, H = 1.0, Um = 0.0, A = 0.1, P = 0.05, nu = 0.005;
const double kx = 2.0 * M_PI / Lx, a = M_PI / H, b = 2.0 * M_PI / H;

struct Exact {
    double u, v, p;
    double ux, uy, uxx, uyy, vx, vy, vxx, vyy, px, py;
};

Exact exact(double x, double y) {
    const double s1 = std::sin(a * y), sb = std::sin(b * y), cb = std::cos(b * y);
    const double cx = std::cos(kx * x), sx = std::sin(kx * x);
    Exact e;
    e.u = 4.0 * Um * y * (H - y) / (H * H) + A * a * sb * cx;
    e.v = A * kx * s1 * s1 * sx;
    e.p = P * cx * cb;
    e.ux = -A * a * kx * sb * sx;
    e.uy = 4.0 * Um * (H - 2.0 * y) / (H * H) + A * a * b * cb * cx;
    e.uxx = -A * a * kx * kx * sb * cx;
    e.uyy = -8.0 * Um / (H * H) - A * a * b * b * sb * cx;
    e.vx = A * kx * kx * s1 * s1 * cx;
    e.vy = A * kx * a * sb * sx;
    e.vxx = -A * kx * kx * kx * s1 * s1 * sx;
    e.vyy = A * kx * a * b * cb * sx;
    e.px = -P * kx * sx * cb;
    e.py = -P * b * cx * sb;
    return e;
}

// volume-weighted RMS velocity error of the converged discrete solution
double solveError(int nx, int ny, const std::string &scheme, int &iters) {
    std::vector<double> xN(nx + 1), yB(nx + 1, 0.0);
    for (int i = 0; i <= nx; ++i)
        xN[i] = Lx * i / nx;
    Mesh mesh = Mesh::makeCurvedChannelPeriodic2D(xN, yB, H, ny, 0.0, 1.0);
    mesh.computeWallDistance();
    FlowBoundaryConditions bcs = FlowBoundaryConditions::channelDefaults(mesh, 1.0, 0.0, 1.0);

    std::vector<Vec3> src(mesh.nCells());
    for (int ci = 0; ci < mesh.nCells(); ++ci) {
        const Vec3 &c = mesh.cell(ci).center;
        Exact e = exact(c.x, c.y);
        src[ci] = Vec3(e.u * e.ux + e.v * e.uy + e.px - nu * (e.uxx + e.uyy),
                       e.u * e.vx + e.v * e.vy + e.py - nu * (e.vxx + e.vyy), 0.0);
    }

    SolverSettings s;
    s.maxIterations = 40000;
    s.convergenceTol = 1e-9;
    s.alphaU = 0.7;
    s.alphaP = 0.3;
    s.verbose = false;
    s.turbStartIter = s.maxIterations; // laminar: no turbulence model
    s.momentumConvection = scheme;
    SSTModel sst{SSTCoefficients{}};
    SIMPLESolver solver(mesh, sst, bcs, nu, s);
    solver.setMomentumSource(&src);
    FlowFields f(mesh);
    solver.initUniform(f, Vec3(0.0, 0.0, 0.0), 0.0, 0.0, 1.0); // k = 0, so nuT = 0
    for (int ci = 0; ci < mesh.nCells(); ++ci) {
        const Vec3 &c = mesh.cell(ci).center;
        Exact e = exact(c.x, c.y);
        f.U[ci] = Vec3(e.u, e.v, 0.0);
        f.p[ci] = e.p;
    }
    applyAllBCs(f.U, f.p, f.k, f.omega, mesh, bcs, nu);
    ConvergenceHistory h = solver.solve(f);
    REQUIRE(h.converged, "MMS solve must converge");
    iters = h.finalIter;

    double err = 0.0, vol = 0.0;
    for (int ci = 0; ci < mesh.nCells(); ++ci) {
        const Vec3 &c = mesh.cell(ci).center;
        Exact e = exact(c.x, c.y);
        double du = f.U[ci].x - e.u, dv = f.U[ci].y - e.v;
        err += (du * du + dv * dv) * mesh.cell(ci).volume;
        vol += mesh.cell(ci).volume;
    }
    return std::sqrt(err / vol);
}

} // namespace

int main() {
    const int sizes[3][2] = {{16, 8}, {32, 16}, {64, 32}};
    const char *schemes[3] = {"upwind", "linearUpwind", "limitedLinearUpwind"};
    double e[3][3];
    for (int k = 0; k < 3; ++k) {
        int it[3];
        for (int m = 0; m < 3; ++m)
            e[k][m] = solveError(sizes[m][0], sizes[m][1], schemes[k], it[m]);
        std::printf(
            "  %-19s RMS errors %.3e %.3e %.3e (iterations %d %d %d), observed order %.2f %.2f\n",
            schemes[k], e[k][0], e[k][1], e[k][2], it[0], it[1], it[2],
            std::log(e[k][0] / e[k][1]) / std::log(2.0),
            std::log(e[k][1] / e[k][2]) / std::log(2.0));
    }
    for (int k = 1; k < 3; ++k) {
        const double finest = std::log(e[k][1] / e[k][2]) / std::log(2.0);
        const double average = std::log(e[k][0] / e[k][2]) / std::log(4.0);
        std::printf("  %s: order %.2f on the finest pair, %.2f on average; %.0fx below upwind\n",
                    schemes[k], finest, average, e[0][2] / e[k][2]);
        REQUIRE(finest >= 1.9 && average >= 1.9, "second-order convection must reach second order");
        REQUIRE(e[k][2] < 0.1 * e[0][2],
                "second-order convection must be far more accurate than upwind");
    }
    std::printf("test_convection_mms: all checks passed\n");
    return 0;
}
