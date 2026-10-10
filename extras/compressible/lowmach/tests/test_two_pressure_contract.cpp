// Public-contract tests for the two-pressure convention of the compressible
// SIMPLE solver: the field carries the MECHANICAL working pressure
// p_mech = p_thermo + (2/3) rho k, while every external input (initialization
// pressure, prescribed outlet pressure) and every exported quantity is
// THERMODYNAMIC. These tests pin the conversion at each crossing point, plus
// the closed-form EOS inversion consistency at iteration zero.
#include "Mesh.hpp"
#include "SSTModel.hpp"
#include "CompressibleFlowFields.hpp"
#include "CompressibleBCs.hpp"
#include "CompressibleSIMPLESolver.hpp"
#include "IdealGasEOS.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>

static void REQUIRE(bool ok, const char *msg) {
    if (!ok) {
        std::printf("FAIL: %s\n", msg);
        std::exit(1);
    }
}

int main() {
    IdealGasEOS eos;
    const double T_in = 300.0;
    const double p_ref = 101325.0; // thermodynamic initialization pressure
    const double Uin = 30.0;
    const double kIn = 2.0; // large enough that 2/3 rho k is visible
    const double omIn = 50.0;

    Mesh mesh = Mesh::makeChannel2D(8, 6, 4.0, 1.0, 1e5, 1.0);
    mesh.computeWallDistance();
    auto bcs = CompressibleBoundaryConditions::channelDefaults(mesh, Uin, T_in, p_ref, kIn, omIn);
    SSTModel sst;
    SolverSettings settings;
    CompressibleSIMPLESolver solver(mesh, sst, bcs, eos, settings);

    CompressibleFlowFields f(mesh);
    solver.initUniform(f, Vec3(Uin, 0, 0), p_ref, T_in, kIn, omIn);

    // 1. initialization: the stored field is the mechanical pressure formed
    //    from the plain-EOS density at the thermodynamic input state, so
    //    p_mech - (2/3) rho k returns p_init exactly
    const double rho0 = eos.density(p_ref, T_in);
    const double pMech0 = p_ref + (2.0 / 3.0) * rho0 * kIn;
    for (int ci = 0; ci < mesh.nCells(); ++ci) {
        REQUIRE(std::fabs(f.p[ci] - pMech0) < 1e-9 * pMech0,
                "initialized field must be the mechanical pressure");
        REQUIRE(std::fabs(f.rho[ci] - rho0) < 1e-12 * rho0,
                "initialized density must solve the plain EOS at (p_init, T)");
        const double pThermo = f.p[ci] - (2.0 / 3.0) * f.rho[ci] * f.k[ci];
        REQUIRE(std::fabs(pThermo - p_ref) < 1e-9 * p_ref,
                "thermodynamic recovery must return p_init exactly");
    }

    // 2. the closed-form EOS inversion is consistent at iteration zero:
    //    rho = p_mech / (R T + (2/3) k) reproduces the plain-EOS rho0
    const double R = p_ref / (rho0 * T_in);
    const double rhoInv = pMech0 / (R * T_in + (2.0 / 3.0) * kIn);
    REQUIRE(std::fabs(rhoInv - rho0) < 1e-12 * rho0,
            "two-pressure EOS inversion must reproduce rho0 at t=0");

    // 3. outlet boundary faces: the prescribed value is thermodynamic; the
    //    stored boundary value is mechanical with the owner-cell state
    for (int pi = 0; pi < mesh.nPatches(); ++pi) {
        const Patch &pat = mesh.patch(pi);
        if (pat.type != "outlet")
            continue;
        for (FaceID fi : pat.faces) {
            int o = mesh.face(fi).owner;
            const double expect = p_ref + (2.0 / 3.0) * f.rho[o] * std::max(f.k[o], 0.0);
            REQUIRE(std::fabs(f.p.bface(fi) - expect) < 1e-9 * expect,
                    "outlet boundary face must carry the mechanical value");
            REQUIRE(std::fabs(f.p.bface(fi) - p_ref) > 1e-12 * p_ref,
                    "conversion must be nonzero at finite k (test premise)");
        }
    }

    std::printf("test_two_pressure_contract: all checks passed\n");
    return 0;
}
