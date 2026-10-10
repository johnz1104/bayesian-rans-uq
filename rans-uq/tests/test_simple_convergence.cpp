#include "SIMPLESolver.hpp"
#include "ObservationOperator.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool condition, const char *message) {
    if (!condition)
        throw std::runtime_error(message);
}
void cliReference() {
    // Retain the demonstration's 60x40 case inputs. The independent oracle
    // queries the library observation path, not the CLI's wall-stress loop.
    constexpr double nu = 1.0 / 6800, k = 1.5 * 0.05 * 0.05;
    Mesh mesh = Mesh::makeChannel2D(60, 40, 10, 2, 6800);
    mesh.computeWallDistance();
    const auto bc = FlowBoundaryConditions::channelDefaults(mesh, 1, k, k / (100 * nu));
    SolverSettings settings;
    settings.verbose = false;
    SIMPLESolver solver(mesh, SSTModel{}, bc, nu, settings);
    FlowFields fields(mesh);
    solver.initUniform(fields, {1, 0, 0}, 0, k, k / (100 * nu));
    ObservationOperator observations;
    for (const auto &patch : mesh.patches())
        if (patch.type == "wall")
            for (int fi : patch.faces)
                if (mesh.face(fi).center.x >= 5)
                    observations.addSkinFriction(patch.name, mesh.face(fi).center, 0, 1, 1);
    const auto values = observations.evaluate(mesh, fields, nu);
    double sum = 0;
    for (double value : values)
        sum += value;
    std::cout << "CORE_WALL_CF=" << sum / values.size() << '\n';
    std::cout << "DEAN_CF=" << 0.073 * std::pow(2 * 6800, -0.25) << '\n';
}
void checkRestoration() {
    Mesh mesh = Mesh::makePeriodicChannel2D(3, 8, 1, 1, 182.088, 1);
    mesh.computeWallDistance();
    const auto bc = FlowBoundaryConditions::channelDefaults(mesh, 1, 1, 10);
    SolverSettings settings;
    settings.verbose = false;
    SIMPLESolver solver(mesh, SSTModel{}, bc, 1 / 182.088, settings);
    FlowFields state(mesh);
    solver.initUniform(state, {15, 0, 0}, 0, 1, 10);
    const double expected = state.omega[0];
    Mesh other = Mesh::makeChannel2D(2, 2, 1, 1);
    FlowFields invalid(other);
    SSTCoefficients theta;
    theta.beta1 *= 2;
    for (int method : {0, 1}) {
        bool rejected = false;
        try {
            if (method == 0) {
                (void)solver.continuityResidual(invalid, theta);
            } else {
                LinearSystem momentum, k, omega;
                std::vector<double> diagonal;
                solver.assemblePreconBlocks(invalid, theta, momentum, k, omega, diagonal);
            }
        } catch (const std::exception &) {
            rejected = true;
        }
        require(rejected, "diagnostic must reject a state on a different mesh");
        solver.initUniform(state, {15, 0, 0}, 0, 1, 10);
        require(state.omega[0] == expected,
                "diagnostic exception must restore the solver's closure coefficients");
    }
    // The preconditioner is a real finite operator, not just an exception path.
    LinearSystem momentum, k, omega;
    std::vector<double> diagonal;
    solver.assemblePreconBlocks(state, theta, momentum, k, omega, diagonal);
    require(diagonal.size() == static_cast<size_t>(mesh.nCells()),
            "preconditioner must return one momentum diagonal per cell");
    for (double value : diagonal)
        require(std::isfinite(value) && value > 0, "momentum preconditioner must be positive");
    solver.initUniform(state, {15, 0, 0}, 0, 1, 10);
    require(state.omega[0] == expected, "successful diagnostic must restore coefficients");
}
void checkPressureAlarm() {
    constexpr double re = 182.088;
    const double firstHeight = 1 + std::tanh(2 * (1.0 / 16 - 1)) / std::tanh(2.0);
    Mesh mesh = Mesh::makePeriodicChannel2D(3, 32, 1, 1, re, firstHeight * re);
    mesh.computeWallDistance();
    const auto bc = FlowBoundaryConditions::channelDefaults(mesh, 1, 1, 10);
    for (bool tight : {false, true}) {
        SolverSettings settings;
        settings.maxIterations = 15000;
        settings.convergenceTol = 1e-8;
        settings.alphaU = 0.5;
        settings.bodyForce = {1, 0, 0};
        settings.verbose = false;
        if (tight)
            settings.divergenceLimit = 1e-12;
        SIMPLESolver solver(mesh, SSTModel{}, bc, 1 / re, settings);
        FlowFields state(mesh);
        solver.initUniform(state, {15, 0, 0}, 0, 1, 10);
        const auto history = solver.solve(state);
        std::cout << "tight=" << tight << " iterations=" << history.finalIter
                  << " converged=" << history.converged << " diverged=" << history.diverged
                  << " worst=" << history.finalWorstMeasure << '\n';
        if (tight) {
            require(history.diverged && !history.converged,
                    "finite residual growth must still honor the divergence limit");
        } else {
            require(history.converged && !history.diverged,
                    "bounded channel must not diverge from a roundoff pressure normalizer");
            require(history.finalWorstMeasure < settings.convergenceTol,
                    "the original convergence tolerance must still be satisfied");
        }
    }
}
} // namespace
int main(int argc, char **argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--cli-reference") {
            cliReference();
            return 0;
        }
        if (argc == 1)
            checkRestoration();
        checkPressureAlarm();
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
