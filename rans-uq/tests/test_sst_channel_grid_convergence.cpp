#include "ForwardModel.hpp"
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>

// Same-model verification, not a DNS acceptance band. Unit half-height and
// friction velocity, Re_tau=182.088, default SST-2003 constants, no data or RNG.
// Reference: independent NumPy half-channel with analytic total shear, the
// near-wall omega singularity (coefficient 6) subtracted, a 1e-10 cross-diffusion
// floor, tanh stretch 2 and nonlinear tolerance 1e-10, solved on 1024, 2048 and
// 4096 cells. The three reference grids give a provisional profile GCI below 2e-6.
int main() {
    constexpr std::array<double, 7> stations{0.01, 0.03, 0.1, 0.2, 0.4, 0.7, 0.95};
    constexpr std::array<double, 7> reference{
        1.8115554680911752, 5.240990685584829, 10.940331282319287, 13.625406245058477,
        15.908282983994223, 17.29794978109906, 17.665856864022285};
    std::array<double, 3> error{};
    for (int level = 0; level < 3; ++level) {
        const int halfCells = 16 << level;
        const double reTau = 182.088, nu = 1 / reTau;
        const double firstHeight = 1 + std::tanh(2 * (1.0 / halfCells - 1)) / std::tanh(2.0);
        Mesh mesh = Mesh::makePeriodicChannel2D(3, 2 * halfCells, 1, 1, reTau, firstHeight * reTau);
        mesh.computeWallDistance();
        auto bc = FlowBoundaryConditions::channelDefaults(mesh, 1, 1, 10);
        ObservationOperator obs;
        obs.addSkinFriction("bottom", {0.5, 0, 0}, 2, 1, 1);
        for (double y : stations)
            obs.addVelocityProfile({0.5, y, 0}, 0, 0, 1);
        SolverSettings settings;
        settings.maxIterations = 150000;
        settings.convergenceTol = 1e-8;
        settings.alphaU = 0.5;
        settings.bodyForce = {1, 0, 0};
        settings.verbose = false;
        auto parameters = InferenceParameterSet::a1_betaStar();
        ForwardModel model(mesh, parameters, obs, bc, nu, settings, {15, 0, 0}, 0, 1, 10);
        const auto evaluation = model.evaluateWithDiagnostics(parameters.pack(SSTCoefficients{}));
        const auto &result = evaluation.result;
        if (result.status != EvaluationStatus::Converged)
            std::cerr << "status=" << statusString(result.status)
                      << " iterations=" << result.simpleIters << " error=" << evaluation.error
                      << std::endl;
        if (result.status != EvaluationStatus::Converged)
            throw std::runtime_error(
                "channel refinement requires an iteratively converged solution");
        if (!std::isfinite(result.predictions[0]) || std::abs(result.predictions[0] / 2 - 1) > 2e-6)
            throw std::runtime_error("channel wall force does not balance the applied body force");
        for (size_t i = 0; i < reference.size(); ++i) {
            const double difference = result.predictions[i + 1] - reference[i];
            if (!std::isfinite(difference))
                throw std::runtime_error("nonfinite channel profile comparison");
            error[level] += difference * difference / reference.size();
        }
        error[level] = std::sqrt(error[level]);
        std::cout << "ny=" << 2 * halfCells << " iterations=" << result.simpleIters
                  << " same-model profile RMS error=" << error[level] << std::endl;
    }
    for (size_t level = 1; level < error.size(); ++level) {
        const double order = std::log2(error[level - 1] / error[level]);
        std::cout << "reference-error order=" << order << '\n';
        if (!std::isfinite(order) || order < 1.7)
            throw std::runtime_error(
                "resolved planar SST channel must converge beyond first order");
    }
    if (error.back() > 0.025)
        throw std::runtime_error(
            "fine planar SST channel does not match the independent reference");
}
