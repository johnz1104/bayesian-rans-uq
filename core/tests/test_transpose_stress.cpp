// Variable-viscosity transpose stress div(coeff (grad U)^T): manufactured-field
// verification of the explicit deferred correction that completes the Boussinesq
// deviatoric stress divergence (the implicit momentum operator carries only the
// componentwise Laplacian div(coeff grad U_i)).
//
//   1. coeff = c0, U = (a x^2, 0, 0): the i = x source is d/dx(c0 * 2 a x)
//      = 2 a c0 exactly (linear flux, Green-Gauss exact on interior cells).
//   2. coeff = b y, U = (g y, d x, 0): the i = x source is
//      d/dy(coeff * dU_y/dx) = d/dy(b y d) = b d, while the componentwise
//      Laplacian would give d/dy(coeff dU_x/dy) = b g. With g != d the value
//      DISCRIMINATES the transpose operator from the Laplacian; the i = y
//      source is d/dx(b y g) = 0.
//   3. Parallel shear flow U = (U(y), 0, 0) with coeff(y): the transpose term
//      vanishes identically (this is why fully-developed channel and Couette
//      results are invariant under this correction), and the discrete operator
//      reproduces the zero exactly on this mesh.

#include "Mesh.hpp"
#include "Field.hpp"
#include "StressOperators.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>
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

} // namespace

int main() {
    const int nx = 12, ny = 14;
    Mesh mesh = Mesh::makeChannel2D(nx, ny, 2.0, 1.0);
    mesh.computeWallDistance();
    const int nc = mesh.nCells();
    const int nIF = mesh.nInternalFaces();

    // fill a manufactured state: cell values from the analytic fields, and
    // boundary-face values from the same expressions so Green-Gauss cell
    // gradients are exact for the (at most quadratic) fields below
    auto fill = [&](VectorField &U, ScalarField &coeff, auto uFn, auto cFn) {
        for (int ci = 0; ci < nc; ++ci) {
            const Vec3 &c = mesh.cell(ci).center;
            U[ci] = uFn(c.x, c.y);
            coeff[ci] = cFn(c.x, c.y);
        }
        for (int fi = nIF; fi < mesh.nFaces(); ++fi) {
            const Vec3 &c = mesh.face(fi).center;
            U.bface(fi) = uFn(c.x, c.y);
            coeff.bface(fi) = cFn(c.x, c.y);
        }
    };

    // interior = two rings in from every boundary (cell gradients feeding the
    // interior faces are themselves built from interior data only)
    auto isInterior = [&](int ci) {
        int i = ci % nx, j = ci / nx;
        return i >= 2 && i < nx - 2 && j >= 2 && j < ny - 2;
    };

    // 1. exactness: coeff = c0, U = (a x^2, 0, 0) -> src_x / V = 2 a c0
    {
        const double a = 0.8, c0 = 0.35;
        VectorField U(mesh, "U");
        ScalarField coeff(mesh, "coeff");
        fill(
            U, coeff, [&](double x, double) { return Vec3(a * x * x, 0.0, 0.0); },
            [&](double, double) { return c0; });
        std::vector<double> sx = transposeStressSource(mesh, coeff, U, 0);
        for (int ci = 0; ci < nc; ++ci) {
            if (!isInterior(ci))
                continue;
            double v = sx[ci] / mesh.cell(ci).volume;
            REQUIRE(std::fabs(v - 2.0 * a * c0) < 1e-11,
                    "quadratic-U transpose source must equal 2 a c0 exactly");
        }
    }

    // 2. discrimination: coeff = b y, U = (g y, d x, 0)
    //    src_x / V = b d (transpose), NOT b g (componentwise Laplacian)
    {
        const double b = 0.6, g = 1.3, d = 0.4;
        VectorField U(mesh, "U");
        ScalarField coeff(mesh, "coeff");
        fill(
            U, coeff, [&](double x, double y) { return Vec3(g * y, d * x, 0.0); },
            [&](double, double y) { return b * y; });
        std::vector<double> zeroWall(nc, 0.0);
        std::vector<double> sx = transposeStressSource(mesh, coeff, U, 0, &zeroWall);
        std::vector<double> sy = transposeStressSource(mesh, coeff, U, 1, &zeroWall);
        // wall-adjacent rows (j = 0): the analytic coefficient b*y vanishes
        // AT the wall, so the wall-zero boundary treatment makes these rows
        // EXACT too, which owner extrapolation (coefficient b*y1 != 0) would
        // violate; this pins the wall-zero boundary treatment
        for (int i = 2; i < nx - 2; ++i) {
            int ci = 0 * nx + i;
            double vx = sx[ci] / mesh.cell(ci).volume;
            REQUIRE(std::fabs(vx - b * d) < 1e-11,
                    "wall-adjacent transpose source must be exact under wall-zero");
        }
        for (int ci = 0; ci < nc; ++ci) {
            if (!isInterior(ci))
                continue;
            double vx = sx[ci] / mesh.cell(ci).volume;
            double vy = sy[ci] / mesh.cell(ci).volume;
            REQUIRE(std::fabs(vx - b * d) < 1e-11,
                    "transpose x-source must be b*d (the Laplacian would give b*g)");
            REQUIRE(std::fabs(vx - b * g) > 0.1 * std::fabs(b * (g - d)),
                    "test fields must discriminate transpose from Laplacian");
            REQUIRE(std::fabs(vy) < 1e-11, "transpose y-source must vanish here");
        }
    }

    // 3. parallel-shear invariance: U = (tanh profile(y), 0, 0), coeff(y)
    {
        VectorField U(mesh, "U");
        ScalarField coeff(mesh, "coeff");
        fill(
            U, coeff, [&](double, double y) { return Vec3(std::tanh(4.0 * (y - 0.5)), 0.0, 0.0); },
            [&](double, double y) { return 0.02 + 0.05 * y * (1.0 - y); });
        for (int comp = 0; comp < 2; ++comp) {
            std::vector<double> s = transposeStressSource(mesh, coeff, U, comp);
            for (int ci = 0; ci < nc; ++ci) {
                REQUIRE(std::fabs(s[ci]) < 1e-14,
                        "transpose source must vanish identically in parallel shear");
            }
        }
    }

    std::printf("test_transpose_stress: all checks passed\n");
    return 0;
}
