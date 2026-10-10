// AMG hierarchy reuse and V-cycle symmetry.
//
// The AMG-preconditioned CG used to rebuild its whole hierarchy on every
// pressure solve (CSR conversion with per-row sorting, aggregation, hashed
// Galerkin products, level allocations), although between SIMPLE iterations
// only the coefficients change. On the BFS and hills goldens that setup was
// 26% and 11% of total solve time. Its V-cycle smoothed with forward
// Gauss-Seidel before AND after the coarse correction, which is not a
// symmetric operator, so CG ran with a non-symmetric preconditioner.
//
// This test pins:
//   1. a solve on a refreshed (frozen) hierarchy with unchanged coefficients
//      is bit-identical to a solve after a fresh full setup;
//   2. with changed coefficients the refreshed hierarchy still converges to
//      the requested tolerance;
//   3. the V-cycle preconditioner is symmetric: <y, M x> = <M y, x>;
//   4. a sequence of slowly changing systems is served by refreshes, not
//      full setups.

#include "LinearSolver.hpp"
#include "Mesh.hpp"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

struct Lcg {
    uint64_t s;
    explicit Lcg(uint64_t seed) : s(seed) {}
    double next() {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<double>(s >> 11) / 9007199254740992.0 - 0.5;
    }
};

// pressure-correction-like Laplacian: face coefficient kappa(x) S / delta,
// Dirichlet rows on the outlet faces (right end)
LinearSystem laplacian(const Mesh &m, double phase) {
    LinearSystem A = makeSystem(m);
    for (int fi = 0; fi < m.nInternalFaces(); ++fi) {
        const Face &f = m.face(fi);
        double kappa = 1.0 + 0.5 * std::sin(2.0 * f.center.x + phase) * std::cos(3.0 * f.center.y);
        double c = kappa * f.area / f.delta;
        A.diag[f.owner] += c;
        A.diag[f.neighbor] += c;
        A.upper[fi] = -c;
        A.lower[fi] = -c;
    }
    const Patch &out = m.patch(m.patchByName("outlet"));
    for (FaceID fi : out.faces)
        A.diag[m.face(fi).owner] += m.face(fi).area / m.face(fi).delta;
    return A;
}

double relResidual(const LinearSystem &A, const std::vector<double> &x) {
    std::vector<double> r(A.nCells);
    A.residual(x, r);
    return linalg::norm(r) / linalg::norm(A.source);
}

} // namespace

int main() {
    Mesh mesh = Mesh::makeChannel2D(60, 40, 6.0, 2.0, 5000.0, 1.0);
    const int n = mesh.nCells();
    Lcg rng(2026);
    std::vector<double> rhs(n);
    for (double &v : rhs)
        v = rng.next();

    // 1. refreshed hierarchy, unchanged coefficients: bit-identical solve
    {
        LinearSystem A = laplacian(mesh, 0.0);
        A.source = rhs;
        AMGSolver fresh, reused;
        std::vector<double> x1(n, 0.0), x2(n, 0.0), tmp(n, 0.0);
        fresh.solve(A, x1, 500, 1e-10);
        reused.solve(A, tmp, 500, 1e-10); // full setup
        reused.solve(A, x2, 500, 1e-10);  // refresh
        REQUIRE(reused.fullSetups == 1 && reused.refreshes == 1, "second solve must refresh");
        bool identical = true;
        for (int i = 0; i < n; ++i)
            identical = identical && (x1[i] == x2[i]);
        std::printf("  refresh vs fresh setup: %s\n", identical ? "bit-identical" : "DIFFERENT");
        REQUIRE(identical,
                "refreshed hierarchy with unchanged values must reproduce a fresh setup exactly");
    }

    // 2 and 4. slowly changing coefficients: refreshes converge
    {
        AMGSolver amg;
        int worstIters = 0;
        double worstRes = 0.0;
        for (int k = 0; k < 20; ++k) {
            LinearSystem A = laplacian(mesh, 0.02 * k);
            A.source = rhs;
            std::vector<double> x(n, 0.0);
            SolverResult r = amg.solve(A, x, 500, 1e-8);
            REQUIRE(r.converged, "every solve must converge");
            worstIters = std::max(worstIters, r.iterations);
            worstRes = std::max(worstRes, relResidual(A, x));
        }
        std::printf("  20 changing systems: %d full setups, %d refreshes, worst %d PCG iterations, "
                    "worst relative residual %.2e\n",
                    amg.fullSetups, amg.refreshes, worstIters, worstRes);
        REQUIRE(worstRes < 1e-7, "refreshed solves must meet the tolerance");
        REQUIRE(amg.fullSetups <= 3 && amg.refreshes >= 17,
                "slowly changing systems must reuse the hierarchy");
    }

    // 3. the V-cycle is a symmetric operator
    {
        LinearSystem A = laplacian(mesh, 0.3);
        A.source = rhs;
        AMGSolver amg;
        std::vector<double> x0(n, 0.0);
        amg.solve(A, x0, 1, 1e-30); // build the hierarchy
        std::vector<double> x(n), y(n), Mx(n), My(n);
        for (int i = 0; i < n; ++i) {
            x[i] = rng.next();
            y[i] = rng.next();
        }
        amg.applyPreconditioner(x, Mx);
        amg.applyPreconditioner(y, My);
        double a = linalg::dot(y, Mx), b = linalg::dot(My, x);
        std::printf("  <y, Mx> = %.15e, <My, x> = %.15e, rel diff %.2e\n", a, b,
                    std::fabs(a - b) / std::fabs(a));
        REQUIRE(std::fabs(a - b) < 1e-10 * std::fabs(a),
                "V-cycle preconditioner must be symmetric");
    }

    std::printf("test_amg_reuse: all checks passed\n");
    return 0;
}
