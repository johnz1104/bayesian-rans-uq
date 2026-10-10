#include "SIMPLESolver.hpp"
#include <cmath>
#include <iostream>
#include <limits>

// Qualify the comparison itself before checking actual numerical results.
static bool within(double actual, double expected, double tolerance) {
    return std::isfinite(actual) && std::isfinite(expected) &&
           std::abs(actual - expected) <= tolerance;
}

static bool comparisonRejectsNonfinite() {
    for (double invalid :
         {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(),
          -std::numeric_limits<double>::infinity()}) {
        if (within(invalid, 1.0, 1e-12) || within(1.0, invalid, 1e-12) ||
            within(invalid, invalid, 1e-12))
            return false;
    }
    return within(1.0, 1.0, 1e-12) && !within(1.0, 2.0, 1e-12);
}

int main() {
    if (!comparisonRejectsNonfinite()) {
        std::cerr << "steady residual oracle accepts nonfinite values\n";
        return 9;
    }
    for (bool periodic : {false, true}) {
        Mesh mesh = periodic ? Mesh::makePeriodicChannel2D(3, 4, 1.0, 0.5, 20)
                             : Mesh::makeChannel2D(3, 4, 1.0, 1.0);
        mesh.computeWallDistance();
        auto bc = FlowBoundaryConditions::channelDefaults(mesh, 1.0, 0.01, 2.0);
        SolverSettings settings;
        settings.verbose = false;
        settings.momentumConvection = settings.turbulenceConvection = "upwind";
        const double nu = 0.001;
        SSTCoefficients theta;
        SIMPLESolver solver(mesh, SSTModel(theta), bc, nu, settings);
        FlowFields f(mesh);
        solver.initUniform(f, {0.8, 0.13, 0}, 0, 0.01, 2);
        for (int i = 0; i < mesh.nCells(); ++i)
            f.p[i] = 0.03 * i;
        for (int pi = 0; pi < mesh.nPatches(); ++pi)
            if (mesh.patch(pi).type == "wall")
                for (int fi : mesh.patch(pi).faces) {
                    const auto &face = mesh.face(fi);
                    f.omega[face.owner] = solver.resolvedWallOmega(face.delta);
                }
        const auto faces = solver.freeFluxFaces();
        if (faces.size() != static_cast<size_t>(mesh.nInternalFaces() + (periodic ? 0 : 4)))
            return 1;
        std::vector<double> flux(faces.size());
        for (size_t i = 0; i < flux.size(); ++i)
            flux[i] = 0.08 + 0.003 * i;
        const auto r = solver.steadyResidual(f, theta, flux);
        for (double value : r.equations)
            if (!within(value, value, 0.0))
                return 10;
        const int nc = mesh.nCells();
        if (r.equations.size() != 5 * static_cast<size_t>(nc) + flux.size())
            return 2;
        for (int pi = 0; pi < mesh.nPatches(); ++pi)
            if (mesh.patch(pi).type == "wall")
                for (int fi : mesh.patch(pi).faces)
                    if (!within(r.equations[3 * nc + mesh.face(fi).owner], 0.0, 1e-12))
                        return 3;
        auto shifted = f;
        for (int i = 0; i < nc; ++i)
            shifted.p[i] += 3.0;
        const auto rs = solver.steadyResidual(shifted, theta, flux);
        for (double value : rs.equations)
            if (!within(value, value, 0.0))
                return 10;
        if (periodic) {
            if (!within(rs.equations[4 * nc] - r.equations[4 * nc], 3.0, 1e-12))
                return 4;
            for (size_t i = 0; i < r.equations.size(); ++i)
                if (i != static_cast<size_t>(4 * nc) &&
                    !within(rs.equations[i], r.equations[i], 1e-10))
                    return 5;
            if (r.removedContinuity != r.continuity[0])
                return 6;
        } else {
            for (int i = 0; i < nc; ++i)
                if (r.equations[4 * nc + i] != r.continuity[i])
                    return 7;
        }
        auto changed = theta;
        changed.beta1 *= 1.01;
        const auto rb = solver.steadyResidual(f, changed, flux);
        for (double value : rb.equations)
            if (!within(value, value, 0.0))
                return 10;
        bool boundaryResponded = false;
        for (int i = 0; i < nc; ++i)
            boundaryResponded |=
                std::abs(rb.equations[3 * nc + i] - r.equations[3 * nc + i]) > 1e-7;
        if (!boundaryResponded)
            return 8;
    }
    std::cout << "steady layout, boundary response, pinning and pressure gauge passed\n";
}
