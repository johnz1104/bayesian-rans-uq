#include "CompressibleSIMPLESolver.hpp"
#include "SIMPLESolver.hpp"
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>

int main() {
    const Mesh mesh = Mesh::makeChannel2D(2, 2, 1.0, 1.0);
    const auto bc = FlowBoundaryConditions::channelDefaults(mesh, 1.0, 1e-4, 1.0);
    const auto compressibleBC =
        CompressibleBoundaryConditions::channelDefaults(mesh, 1.0, 300.0, 101325.0, 1e-4, 1.0);
    int failures = 0;
    const auto check = [&](const char *label, const std::function<void(SolverSettings &)> &change) {
        SolverSettings settings;
        change(settings);
        bool simpleRejected = false, compressibleRejected = false;
        try {
            SIMPLESolver solver(mesh, SSTModel{}, bc, 0.01, settings);
        } catch (const std::invalid_argument &) {
            simpleRejected = true;
        }
        try {
            CompressibleSIMPLESolver solver(mesh, SSTModel{}, compressibleBC, IdealGasEOS{},
                                            settings);
        } catch (const std::invalid_argument &) {
            compressibleRejected = true;
        }
        if (!simpleRejected || !compressibleRejected) {
            std::cerr << label << " rejected: SIMPLE=" << simpleRejected
                      << " compressible=" << compressibleRejected << '\n';
            ++failures;
        }
    };
    check("zero turbulence cadence", [](auto &s) { s.turbUpdateInterval = 0; });
    check("negative turbulence cadence", [](auto &s) { s.turbUpdateInterval = -1; });
    check("zero report cadence", [](auto &s) { s.reportInterval = 0; });
    check("zero monitor cadence", [](auto &s) { s.monitorInterval = 0; });
    check("negative iteration limit", [](auto &s) { s.maxIterations = -1; });
    check("negative startup", [](auto &s) { s.turbStartIter = -1; });
    check("startup counter overflow",
          [](auto &s) { s.turbStartIter = std::numeric_limits<int>::max(); });
    check("zero inner iterations", [](auto &s) { s.innerIterations = 0; });
    check("infinite convergence tolerance",
          [](auto &s) { s.convergenceTol = std::numeric_limits<double>::infinity(); });
    check("NaN inner tolerance",
          [](auto &s) { s.innerTolerance = std::numeric_limits<double>::quiet_NaN(); });
    check("zero velocity relaxation", [](auto &s) { s.alphaU = 0.0; });
    check("zero turbulence relaxation", [](auto &s) { s.alphaK = 0.0; });
    check("zero omega relaxation", [](auto &s) { s.alphaOmega = 0.0; });
    check("zero temperature relaxation", [](auto &s) { s.alphaT = 0.0; });
    // A zero outer cap remains a valid request that cannot certify a solve.
    SolverSettings capped;
    capped.maxIterations = 0;
    capped.divergenceLimit = 0.0; // Existing strict divergence-detection control.
    SIMPLESolver simple(mesh, SSTModel{}, bc, 0.01, capped);
    CompressibleSIMPLESolver compressible(mesh, SSTModel{}, compressibleBC, IdealGasEOS{}, capped);
    if (failures)
        return 1;
    std::cout << "14 invalid settings rejected by both SIMPLE constructors; zero cap accepted\n";
}
