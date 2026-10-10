#include "CompressibleSIMPLESolver.hpp"
#include "SIMPLESolver.hpp"
#include <iostream>
#include <limits>
#include <stdexcept>

int main() {
    try {
        Mesh mesh = Mesh::makeChannel2D(2, 2, 1.0, 1.0);
        mesh.computeWallDistance();
        const auto bc = FlowBoundaryConditions::channelDefaults(mesh, 1.0, 1e-4, 1.0);
        const auto compressibleBC =
            CompressibleBoundaryConditions::channelDefaults(mesh, 1.0, 300.0, 101325.0, 1e-4, 1.0);
        SolverSettings settings;
        settings.verbose = false;
        settings.maxIterations = 8;
        settings.turbStartIter = settings.maxIterations;
        settings.convergenceTol = 0.0;
        const auto requireCapped = [&](const auto &history) {
            if (history.converged || history.diverged ||
                history.finalIter != settings.maxIterations ||
                history.entries.size() != static_cast<size_t>(settings.maxIterations))
                throw std::runtime_error(
                    "zero convergence threshold must run to the cap without certification");
        };
        for (bool legacy : {false, true}) {
            settings.legacyConvergence = legacy;
            SIMPLESolver solver(mesh, SSTModel{}, bc, 0.01, settings);
            FlowFields fields(mesh);
            solver.initUniform(fields, {1.0, 0.0, 0.0}, 0.0, 1e-4, 1.0);
            requireCapped(solver.solve(fields));
        }
        CompressibleSIMPLESolver solver(mesh, SSTModel{}, compressibleBC, IdealGasEOS{}, settings);
        CompressibleFlowFields fields(mesh);
        solver.initUniform(fields, {1.0, 0.0, 0.0}, 101325.0, 300.0, 1e-4, 1.0);
        requireCapped(solver.solve(fields));
        for (double invalid :
             {-1.0, std::numeric_limits<double>::quiet_NaN(),
              std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()}) {
            settings.convergenceTol = invalid;
            bool rejected = false;
            try {
                settings.validate();
            } catch (const std::invalid_argument &) {
                rejected = true;
            }
            if (!rejected)
                throw std::runtime_error("invalid convergence threshold was accepted");
        }
        std::cout << "zero convergence threshold caps both SIMPLE drivers without certification; "
                     "invalid thresholds rejected\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
