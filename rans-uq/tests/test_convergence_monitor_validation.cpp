#include "SIMPLESolver.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

// Independent oracle: a nonfinite quantity cannot certify convergence.
// The finite constant control also proves this fixture can reach convergence.
static void checkMonitor(double value, bool mustReject) {
    Mesh mesh = Mesh::makeChannel2D(4, 8, 2.0, 1.0, 100.0, 1.0);
    mesh.computeWallDistance();
    const auto bc = FlowBoundaryConditions::couetteDefaults(mesh, 1.0, 1e-4, 1.0);
    SolverSettings settings;
    settings.verbose = false;
    settings.maxIterations = 3000;
    settings.turbStartIter = 4000;
    settings.convergenceTol = 1e-7;
    settings.convergenceWindow = 2;
    settings.monitorInterval = 1;
    SIMPLESolver solver(mesh, SSTModel{}, bc, 0.01, settings);
    FlowFields fields(mesh);
    solver.initUniform(fields, {0.5, 0.0, 0.0}, 0.0, 1e-4, 1.0);
    int calls = 0;
    solver.setConvergenceMonitor([&](const FlowFields &) {
        ++calls;
        return std::vector<double>{value};
    });
    bool rejected = false;
    ConvergenceHistory history;
    try {
        history = solver.solve(fields);
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    if (calls == 0)
        throw std::runtime_error("fixture never called convergence monitor");
    if (mustReject && !rejected)
        throw std::runtime_error("nonfinite convergence monitor was not rejected");
    if (!mustReject && (rejected || !history.converged))
        throw std::runtime_error("finite constant monitor failed the convergence control");
}

int main() {
    try {
        checkMonitor(1.0, false);
        checkMonitor(std::numeric_limits<double>::quiet_NaN(), true);
        checkMonitor(std::numeric_limits<double>::infinity(), true);
        checkMonitor(-std::numeric_limits<double>::infinity(), true);
        std::cout << "finite monitor converges; NaN and both infinities rejected\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
