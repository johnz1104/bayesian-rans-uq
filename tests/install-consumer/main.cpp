// A small deterministic laminar geometry/installation check, not a research result.
#include <BuildInfo.hpp>
#include <DBNSSolver.hpp>
#include <SIMPLESolver.hpp>
#include <cmath>
#include <iostream>

int main() {
    constexpr int nx = 8, ny = 8;
    constexpr double nu = 0.1, length = 2.0, height = 1.0;
    std::vector<double> x(nx + 1), bottom(nx + 1);
    for (int i = 0; i <= nx; ++i) {
        x[i] = length * i / nx;
        bottom[i] = 0.05 * (1.0 + std::cos(2.0 * std::acos(-1.0) * x[i] / length));
    }
    Mesh mesh = Mesh::makeCurvedChannelPeriodic2D(x, bottom, height, ny, 0.0, 1.0);
    mesh.computeWallDistance();
    const auto bcs = FlowBoundaryConditions::channelDefaults(mesh, 0.0, 0.0, 1.0);
    SolverSettings settings;
    settings.verbose = false;
    settings.maxIterations = 2000;
    settings.turbStartIter = settings.maxIterations;
    settings.convergenceTol = 1e-7;
    settings.bodyForce = Vec3(0.001, 0.0, 0.0);
    const SSTModel sst{SSTCoefficients{}};
    SIMPLESolver solver(mesh, sst, bcs, nu, settings);
    FlowFields fields(mesh);
    solver.initUniform(fields, Vec3(0.0, 0.0, 0.0), 0.0, 0.0, 1.0);
    const auto history = solver.solve(fields);
    if (!history.converged || history.diverged || !std::isfinite(fields.U.l2Norm()))
        return 1;
    // Exercise the separately exported DBNS library as well as its headers.
    dbns::DBNSSolver compressible(mesh, IdealGasEOS{}, SSTCoefficients{},
                                 dbns::DBNSBoundaryConditions{}, dbns::DBNSSettings{});
    if (compressible.nCells() != mesh.nCells()) return 2;
    std::cout << "cfdr " << cfdr::version << " source=" << cfdr::sourceDigest
              << " mesh=" << nx << 'x' << ny << " nu=" << nu
              << " iterations=" << history.finalIter
              << " data=none seed=none provisional=installation-check\n";
}
