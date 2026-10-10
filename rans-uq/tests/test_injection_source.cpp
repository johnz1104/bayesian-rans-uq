// Reynolds-stress injection: manufactured-field verification of the explicit
// deferred-correction source and the realizability re-assertion.
//
// The injection adds f_inj = -div(2 k b_target + 2 nuT dev(S)) to the momentum
// source. assembleResidual(state, theta) reuses assembleMomentum bit-for-bit,
// so the difference between residuals with and without a target isolates the
// injected flux exactly (diagonal and under-relaxation terms cancel).
//
//   1. sym3Eigenvalues: analytic eigenvalues match hand-computed spectra.
//   2. Uniform U (S = 0), uniform k, constant b_target: D is a constant tensor,
//      its Green-Gauss divergence telescopes to zero in every cell (boundary
//      owner-extrapolation included), so the residual is unchanged.
//   3. Uniform U, k linear in y, constant b_target: the x-momentum force is
//      -d/dy(2 k b_xy) = -2 c b_xy per unit volume, exact for Green-Gauss on
//      linear data; verified on interior cells (boundary rows use owner
//      extrapolation and are excluded).
//   4. Realizability diagnostics: a realizable target passes; planting one
//      unrealizable cell flips allRealizable and records the violation.

#include "Mesh.hpp"
#include "Field.hpp"
#include "BoundaryCondition.hpp"
#include "SIMPLESolver.hpp"
#include "SSTModel.hpp"
#include "AnisotropyTools.hpp"
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
    // 1. analytic symmetric-3x3 eigenvalues
    {
        double b[6] = {0.5, -0.2, -0.3, 0.0, 0.0, 0.0}; // diagonal
        double lam[3];
        aniso::sym3Eigenvalues(b, lam);
        REQUIRE(std::fabs(lam[0] - 0.5) < 1e-12, "diag eig l1");
        REQUIRE(std::fabs(lam[1] + 0.2) < 1e-12, "diag eig l2");
        REQUIRE(std::fabs(lam[2] + 0.3) < 1e-12, "diag eig l3");

        double c[6] = {0.0, 0.0, 0.0, 0.5, 0.0, 0.0}; // pure xy shear
        aniso::sym3Eigenvalues(c, lam);
        REQUIRE(std::fabs(lam[0] - 0.5) < 1e-12, "shear eig l1");
        REQUIRE(std::fabs(lam[1] - 0.0) < 1e-12, "shear eig l2");
        REQUIRE(std::fabs(lam[2] + 0.5) < 1e-12, "shear eig l3");

        // isotropic third: b = 0 sits at the 3C corner, margin = 1/3... the
        // margin is min(c1, c2, c3) = min(0, 0, 1) = 0 (boundary, realizable)
        double z[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        REQUIRE(aniso::isRealizable(z), "isotropic state must be realizable");
        // one-component limit diag(2/3, -1/3, -1/3): realizable corner
        double one[6] = {2.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0, 0.0, 0.0, 0.0};
        REQUIRE(aniso::isRealizable(one), "1C corner must be realizable");
        // beyond the 1C corner: unrealizable
        double bad[6] = {0.9, -0.45, -0.45, 0.0, 0.0, 0.0};
        REQUIRE(!aniso::isRealizable(bad), "l3 < -1/3 must be unrealizable");
    }

    // shared small channel: uniform 8x10 cells, Lx=2, Ly=1
    const int nx = 8, ny = 10;
    Mesh mesh = Mesh::makeChannel2D(nx, ny, 2.0, 1.0);
    mesh.computeWallDistance();
    const int nc = mesh.nCells();
    const double nu = 1e-3, kIn = 1e-3, omIn = 50.0;
    FlowBoundaryConditions bcs = FlowBoundaryConditions::channelDefaults(mesh, 1.0, kIn, omIn);
    SolverSettings settings;
    settings.verbose = false;
    SSTModel sst{SSTCoefficients{}};

    // fixed state: uniform U (S = 0 exactly), uniform omega; k set per test
    auto makeState = [&](double kA, double kC) {
        FlowFields f(mesh);
        f.U.setUniform(Vec3(0.7, 0.0, 0.0));
        for (int fi = mesh.nInternalFaces(); fi < mesh.nFaces(); ++fi)
            f.U.bface(fi) = Vec3(0.7, 0.0, 0.0); // uniform boundary values too
        f.p.setUniform(0.0);
        f.omega.setUniform(omIn);
        for (int ci = 0; ci < nc; ++ci)
            f.k[ci] = kA + kC * mesh.cell(ci).center.y; // k = kA + kC y
        for (int fi = mesh.nInternalFaces(); fi < mesh.nFaces(); ++fi)
            f.k.bface(fi) = kA + kC * mesh.face(fi).center.y;
        f.nuT.setUniform(0.0);
        f.F1.setUniform(1.0);
        f.F2.setUniform(1.0);
        f.Pk.setUniform(0.0);
        f.CDkw.setUniform(0.0);
        return f;
    };

    const SSTCoefficients theta{};
    const double bxx = 0.10, byy = -0.04, bxy = 0.05;

    // 2. constant D field: injected flux telescopes to zero everywhere
    {
        SIMPLESolver solver(mesh, sst, bcs, nu, settings);
        FlowFields f = makeState(2e-3, 0.0); // uniform k
        std::vector<double> R0 = solver.assembleResidual(f, theta);

        std::vector<double> b6(6 * nc, 0.0);
        for (int ci = 0; ci < nc; ++ci) {
            b6[6 * ci + 0] = bxx;
            b6[6 * ci + 1] = byy;
            b6[6 * ci + 2] = -(bxx + byy);
            b6[6 * ci + 3] = bxy;
        }
        solver.setTargetAnisotropy(&b6);
        std::vector<double> R1 = solver.assembleResidual(f, theta);
        double maxDiff = 0.0;
        for (int i = 0; i < 2 * nc; ++i) // Rux and Ruy blocks
            maxDiff = std::max(maxDiff, std::fabs(R1[i] - R0[i]));
        REQUIRE(maxDiff < 1e-13, "constant-tensor injection must telescope to zero");
        REQUIRE(solver.injectionDiagnostics().active, "diagnostics not active");
        REQUIRE(solver.injectionDiagnostics().allRealizable,
                "realizable target flagged unrealizable");
    }

    // 3. k linear in y: x-momentum force density is exactly -2 c b_xy
    {
        SIMPLESolver solver(mesh, sst, bcs, nu, settings);
        const double kA = 2e-3, kC = 1e-3;
        FlowFields f = makeState(kA, kC);
        std::vector<double> R0 = solver.assembleResidual(f, theta);

        std::vector<double> b6(6 * nc, 0.0);
        for (int ci = 0; ci < nc; ++ci) {
            b6[6 * ci + 0] = bxx;
            b6[6 * ci + 1] = byy;
            b6[6 * ci + 2] = -(bxx + byy);
            b6[6 * ci + 3] = bxy;
        }
        solver.setTargetAnisotropy(&b6);
        std::vector<double> R1 = solver.assembleResidual(f, theta);

        // interior cells only (one full ring away from every boundary)
        const double dxCell = 2.0 / nx, dyCell = 1.0 / ny;
        int checked = 0;
        for (int ci = 0; ci < nc; ++ci) {
            const Vec3 &cc = mesh.cell(ci).center;
            if (cc.x < 1.5 * dxCell || cc.x > 2.0 - 1.5 * dxCell)
                continue;
            if (cc.y < 1.5 * dyCell || cc.y > 1.0 - 1.5 * dyCell)
                continue;
            const double vol = mesh.cell(ci).volume;
            const double expectX = -2.0 * kC * bxy * vol; // -d/dy(2 k b_xy) V
            const double expectY = -2.0 * kC * byy * vol; // -d/dy(2 k b_yy) V
            REQUIRE(std::fabs((R1[ci] - R0[ci]) - expectX) < 1e-12,
                    "x-momentum injected force wrong on interior cell");
            REQUIRE(std::fabs((R1[nc + ci] - R0[nc + ci]) - expectY) < 1e-12,
                    "y-momentum injected force wrong on interior cell");
            ++checked;
        }
        REQUIRE(checked > 10, "too few interior cells checked");
    }

    // 4. one unrealizable cell is detected and recorded, never masked
    {
        SIMPLESolver solver(mesh, sst, bcs, nu, settings);
        FlowFields f = makeState(2e-3, 0.0);
        std::vector<double> b6(6 * nc, 0.0);
        b6[6 * (nc / 2) + 0] = 0.9; // l1 = 0.9 > 2/3
        b6[6 * (nc / 2) + 1] = -0.45;
        b6[6 * (nc / 2) + 2] = -0.45;
        solver.setTargetAnisotropy(&b6);
        (void)solver.assembleResidual(f, theta);
        REQUIRE(!solver.injectionDiagnostics().allRealizable, "unrealizable target not detected");
        REQUIRE(solver.injectionDiagnostics().maxViolation > 0.05,
                "violation magnitude not recorded");
    }

    std::printf("test_injection_source: all assertions passed\n");
    return 0;
}
