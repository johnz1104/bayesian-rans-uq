// Observation probes must interpolate, not snap to the nearest cell.
//
// Nearest-cell velocity and pressure probes, or nearest-face skin-friction
// probes, introduce first-order sampling error that changes sign between
// stations. Interpolation must recover the field at the requested location.
//
// This test pins:
//   1. velocity probes reproduce a linear field exactly, at random points
//      including the half-cells next to walls, on uniform, wall-clustered,
//      L-shaped and curved periodic meshes;
//   2. pressure taps reproduce a linear field exactly, in the interior and on
//      the wall;
//   3. on a quadratic field the probe error converges at second order (the
//      nearest-cell operator gives first order);
//   4. a probe at a cell centre returns that cell's value;
//   5. skin friction interpolates linearly between wall faces: a wall shear
//      that varies linearly along a flat wall is reproduced exactly between
//      face centres.

#include "Field.hpp"
#include "FlowFields.hpp"
#include "Mesh.hpp"
#include "ObservationOperator.hpp"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <vector>
#include <limits>
#include <stdexcept>

namespace {

#define REQUIRE(cond, msg)                                                                         \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            std::fprintf(stderr, "FAIL [%s:%d] %s\n  required: %s\n", __FILE__, __LINE__, (msg),   \
                         #cond);                                                                   \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

// This is the same maximum-error reduction used by the numerical controls.
// Independent nonfinite controls below must fail before values can disappear.
static double checkedMaximum(double current, double error) {
    if (!std::isfinite(current) || !std::isfinite(error))
        throw std::runtime_error("nonfinite value in numerical error reduction");
    return std::max(current, error);
}

static bool maximumOracleRejectsNonfinite() {
    for (double invalid :
         {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(),
          -std::numeric_limits<double>::infinity()}) {
        bool rejected = false;
        try {
            (void)checkedMaximum(0.0, invalid);
        } catch (const std::runtime_error &) {
            rejected = true;
        }
        if (!rejected)
            return false;
    }
    return checkedMaximum(0.25, 0.5) == 0.5 && checkedMaximum(0.5, 0.25) == 0.5;
}

// deterministic uniform numbers in [0, 1) (no global RNG)
struct Lcg {
    uint64_t s;
    explicit Lcg(uint64_t seed) : s(seed) {}
    double next() {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<double>(s >> 11) / 9007199254740992.0;
    }
};

using Fn = std::function<double(double, double)>;

// cell values and boundary-face values of U.x and p from analytic functions
void fill(const Mesh &mesh, FlowFields &f, const Fn &u, const Fn &p) {
    for (int ci = 0; ci < mesh.nCells(); ++ci) {
        const Vec3 &c = mesh.cell(ci).center;
        f.U[ci] = Vec3(u(c.x, c.y), 0.0, 0.0);
        f.p[ci] = p(c.x, c.y);
    }
    for (int fi = mesh.nInternalFaces(); fi < mesh.nFaces(); ++fi) {
        const Vec3 &c = mesh.face(fi).center;
        f.U.bface(fi) = Vec3(u(c.x, c.y), 0.0, 0.0);
        f.p.bface(fi) = p(c.x, c.y);
    }
}

// probe U.x and Cp (dynamic pressure 1, reference 0, so Cp = p) at points
std::vector<double> probe(const Mesh &mesh, const FlowFields &f, const std::vector<Vec3> &uPts,
                          const std::vector<Vec3> &pPts) {
    ObservationOperator obs;
    for (const Vec3 &x : uPts)
        obs.addVelocityProfile(x, 0, 0.0, 1.0);
    for (const Vec3 &x : pPts)
        obs.addPressureTap(x, 0.0, 1.0, std::sqrt(2.0));
    return obs.evaluate(mesh, f, 1.0);
}

// check exactness for a linear field at random interior and near-wall
// points and at wall taps; yLo/yHi give the local wall heights at x
void requireLinearExact(const Mesh &mesh, const char *name, double x0, double x1, const Fn &yLo,
                        const Fn &yHi, const Fn &u, const Fn &p, double nearWall) {
    FlowFields f(mesh);
    fill(mesh, f, u, p);
    Lcg rng(12345);
    std::vector<Vec3> uPts, pPts;
    for (int k = 0; k < 40; ++k) {
        double x = x0 + (x1 - x0) * (0.02 + 0.96 * rng.next());
        double lo = yLo(x, 0.0), hi = yHi(x, 0.0);
        double y = (k % 4 == 0)   ? lo + nearWall * rng.next() // wall half-cell
                   : (k % 4 == 1) ? hi - nearWall * rng.next()
                                  : lo + (hi - lo) * rng.next();
        uPts.emplace_back(x, y, 0.5);
        pPts.emplace_back(x, y, 0.5);
    }
    for (int k = 0; k < 10; ++k) { // taps on the walls
        double x = x0 + (x1 - x0) * (0.02 + 0.96 * rng.next());
        pPts.emplace_back(x, (k % 2 == 0) ? yLo(x, 0.0) : yHi(x, 0.0), 0.5);
    }
    std::vector<double> got = probe(mesh, f, uPts, pPts);
    double worst = 0.0, scale = 0.0;
    for (size_t i = 0; i < uPts.size(); ++i) {
        double ex = u(uPts[i].x, uPts[i].y);
        worst = checkedMaximum(worst, std::fabs(got[i] - ex));
        scale = std::max(scale, std::fabs(ex));
    }
    for (size_t i = 0; i < pPts.size(); ++i) {
        double ex = p(pPts[i].x, pPts[i].y);
        worst = checkedMaximum(worst, std::fabs(got[uPts.size() + i] - ex));
        scale = std::max(scale, std::fabs(ex));
    }
    std::printf("  %s: max probe error on a linear field %.3e (scale %.3g)\n", name, worst, scale);
    REQUIRE(worst < 1e-11 * scale, "probes must reproduce a linear field exactly");
}

} // namespace

int main() {
    REQUIRE(maximumOracleRejectsNonfinite(), "probe maximum-error oracle accepts nonfinite values");
    const Fn lin = [](double x, double y) { return 1.0 + 0.7 * x - 0.4 * y; };
    const Fn linP = [](double x, double y) { return -0.3 + 0.2 * x + 0.9 * y; };
    const Fn flat0 = [](double, double) { return 0.0; };

    // 1-2. linear fields on several meshes
    requireLinearExact(
        Mesh::makeChannel2D(10, 8, 2.0, 1.0), "channel uniform", 0.0, 2.0, flat0,
        [](double, double) { return 1.0; }, lin, linP, 0.06);
    requireLinearExact(
        Mesh::makeChannel2D(12, 24, 4.0, 2.0, 5000.0, 1.0), "channel clustered", 0.0, 4.0, flat0,
        [](double, double) { return 2.0; }, lin, linP, 1e-3);
    requireLinearExact(
        Mesh::makeBackwardFacingStep2D(8, 12, 6, 6, 2.0, 6.0, 1.0, 3.0, 5000.0, 1.0),
        "step downstream", 0.0, 6.0, flat0, [](double, double) { return 3.0; }, lin, linP, 1e-3);
    {
        // curved periodic: a linear field in x is not periodic, so use y only
        const int nx = 24, ny = 16;
        const double Lx = 4.5, yTop = 3.0;
        std::vector<double> xN(nx + 1), yB(nx + 1);
        for (int i = 0; i <= nx; ++i) {
            xN[i] = Lx * i / nx;
            yB[i] = 0.5 * (1.0 + std::cos(2.0 * M_PI * xN[i] / Lx));
        }
        // the bottom wall is straight between nodes
        Fn bottom = [xN, yB, Lx](double x, double) {
            const int n = static_cast<int>(xN.size()) - 1;
            int i = std::min(static_cast<int>(x / Lx * n), n - 1);
            double t = (x - xN[i]) / (xN[i + 1] - xN[i]);
            return yB[i] + t * (yB[i + 1] - yB[i]);
        };
        requireLinearExact(
            Mesh::makeCurvedChannelPeriodic2D(xN, yB, yTop, ny, 5000.0, 1.0),
            "cosine-bump periodic", 0.0, Lx, bottom, [yTop](double, double) { return yTop; },
            [](double, double y) { return 0.5 - 0.8 * y; },
            [](double, double y) { return 2.0 + 0.3 * y; }, 2e-3);
    }

    // 3. second-order convergence on a quadratic field (uniform meshes)
    {
        const Fn quad = [](double x, double y) { return x * x + 0.5 * x * y + 2.0 * y * y; };
        Lcg rng(777);
        std::vector<Vec3> pts;
        for (int k = 0; k < 30; ++k)
            pts.emplace_back(0.05 + 0.9 * rng.next(), 0.05 + 0.9 * rng.next(), 0.5);
        std::vector<double> err;
        for (int n : {8, 16, 32, 64}) {
            Mesh m = Mesh::makeChannel2D(n, n, 1.0, 1.0);
            FlowFields f(m);
            fill(m, f, quad, quad);
            std::vector<double> got = probe(m, f, pts, {});
            double e = 0.0;
            for (size_t i = 0; i < pts.size(); ++i)
                e = checkedMaximum(e, std::fabs(got[i] - quad(pts[i].x, pts[i].y)));
            err.push_back(e);
        }
        double order = std::log(err[2] / err[3]) / std::log(2.0);
        std::printf("  quadratic field: max errors %.3e %.3e %.3e %.3e, observed order %.2f\n",
                    err[0], err[1], err[2], err[3], order);
        REQUIRE(order > 1.9, "probe error must converge at second order");
    }

    // 4. a probe at a cell centre returns the cell value
    {
        Mesh m = Mesh::makeChannel2D(6, 6, 1.0, 1.0, 3000.0, 1.0);
        FlowFields f(m);
        const Fn wavy = [](double x, double y) { return std::sin(3.0 * x) * std::cos(2.0 * y); };
        fill(m, f, wavy, wavy);
        std::vector<Vec3> pts;
        for (int ci : {0, 7, 20, 35})
            pts.push_back(m.cell(ci).center);
        std::vector<double> got = probe(m, f, pts, {});
        int k = 0;
        for (int ci : {0, 7, 20, 35}) {
            REQUIRE(std::fabs(got[k] - f.U[ci].x) < 1e-13,
                    "probe at a cell centre must return the cell value");
            ++k;
        }
    }

    // 5. skin friction: linear wall shear along a flat wall
    {
        const double nu = 1e-3, Uref = 1.0;
        Mesh m = Mesh::makeChannel2D(10, 6, 2.0, 1.0, 2000.0, 1.0);
        FlowFields f(m);
        const Fn tau = [](double x, double) { return 2e-3 + 1e-3 * x; };
        const Patch &bot = m.patch(m.patchByName("bottom"));
        for (FaceID fi : bot.faces) {
            const Face &fc = m.face(fi);
            f.U[fc.owner] = Vec3(tau(fc.center.x, 0.0) * fc.delta / nu, 0.0, 0.0);
            f.U.bface(fi) = Vec3(0.0, 0.0, 0.0);
        }
        ObservationOperator obs;
        Lcg rng(4242);
        std::vector<double> xs;
        for (int k = 0; k < 15; ++k) {
            double x = 0.2 + 1.6 * rng.next(); // between the first and last face centres
            xs.push_back(x);
            obs.addSkinFriction("bottom", Vec3(x, 0.0, 0.0), 0.0, 1.0, Uref);
        }
        std::vector<double> got = obs.evaluate(m, f, nu);
        double worst = 0.0;
        for (size_t i = 0; i < xs.size(); ++i)
            worst =
                checkedMaximum(worst, std::fabs(got[i] - tau(xs[i], 0.0) / (0.5 * Uref * Uref)));
        std::printf("  skin friction: max error on a linear wall shear %.3e\n", worst);
        REQUIRE(worst < 1e-12, "Cf must interpolate linearly between wall faces");
    }

    std::printf("test_probe_interpolation: all checks passed\n");
    return 0;
}
