// Dimensional consistency of the pressure-coefficient observable.
//
// Applying p/(0.5 U^2) to absolute static pressure without reference pressure
// and density makes Cp depend on the pressure offset. The operator computes
// Cp = (p - p_ref)/(0.5 rho_ref U_ref^2). This test pins:
//   1. legacy incompressible taps (kinematic pressure, default references)
//      evaluate bit-identically to the old p/(0.5 U^2);
//   2. a compressible-style tap with SI references recovers the physically
//      correct coefficient from an absolute-pressure field;
//   3. omitting the references on an absolute-pressure field visibly produces
//      the pathological offset-dominated number the fix eliminates.

#include "Mesh.hpp"
#include "Field.hpp"
#include "FlowFields.hpp"
#include "ObservationOperator.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>

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
    Mesh mesh = Mesh::makeChannel2D(8, 6, 2.0, 1.0);
    mesh.computeWallDistance();
    FlowFields f(mesh);

    const Vec3 loc(1.0, 0.5, 0.0);

    // 1. legacy incompressible tap: kinematic pressure field, defaults
    {
        f.p.setUniform(0.32); // kinematic p (p/rho)
        ObservationOperator obs;
        obs.addPressureTap(loc, 0.0, 1.0, 2.0); // refVel = 2
        double cp = obs.evaluate(mesh, f, 1e-3)[0];
        REQUIRE(std::fabs(cp - 0.32 / (0.5 * 2.0 * 2.0)) < 1e-15,
                "legacy kinematic tap must be unchanged");
    }

    // 2. compressible-style tap: absolute pressure with SI references
    {
        const double p_ref = 101325.0, rho_ref = 1.177, Uref = 34.7;
        const double dynP = 0.5 * rho_ref * Uref * Uref;
        f.p.setUniform(p_ref + 0.25 * dynP); // true Cp = 0.25
        ObservationOperator obs;
        obs.addPressureTap(loc, 0.0, 1.0, Uref, 0.0, p_ref, rho_ref);
        double cp = obs.evaluate(mesh, f, 1e-3)[0];
        REQUIRE(std::fabs(cp - 0.25) < 1e-12, "referenced tap must recover the physical Cp");

        // 3. the pathological reading: no references on the same field
        ObservationOperator bad;
        bad.addPressureTap(loc, 0.0, 1.0, Uref);
        double cp_bad = bad.evaluate(mesh, f, 1e-3)[0];
        REQUIRE(cp_bad > 100.0, "unreferenced absolute-pressure Cp is offset-dominated");
    }

    std::printf("test_cp_normalization: all checks passed\n");
    return 0;
}
