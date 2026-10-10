#include "CompressibleForwardModel.hpp"
#include <cmath>
#include <iostream>

int main() {
    Mesh mesh = Mesh::makeChannel2D(3, 4, 1, 1);
    mesh.computeWallDistance();
    const auto parameters = InferenceParameterSet::a1_betaStar();
    const auto boundaries =
        CompressibleBoundaryConditions::channelDefaults(mesh, 1, 300, 101325, 1e-4, 1);
    SolverSettings settings;
    settings.verbose = false;
    settings.maxIterations = 0;
    ObservationOperator observations;
    observations.addVelocityProfile({.5, .5, 0}, 0, 1, 1);
    CompressibleForwardModel model(mesh, parameters, observations, boundaries, IdealGasEOS{},
                                   settings, {1, 0, 0}, 101325, 300, 1e-4, 1);
    const auto theta = parameters.pack(SSTCoefficients{});
    const auto result = model.evaluate(theta);
    if (result.status != EvaluationStatus::Unconverged || !std::isfinite(result.loglik)) {
        std::cerr << "FAIL: fixture must have a capped state and finite diagnostic likelihood\n";
        return 1;
    }
    const auto penalty = model.penalizedLogLikelihood(theta);
    std::cout << "status=" << statusString(result.status) << " diagnostic=" << result.loglik
              << " penalized=" << penalty << '\n';
    if (penalty != -1e30) {
        std::cerr << "FAIL: capped compressible solve was admitted as a likelihood\n";
        return 1;
    }
    if (model.penalizedLogLikelihood({-1, theta[1]}) != -1e30) {
        std::cerr << "FAIL: invalid compressible parameters were admitted\n";
        return 1;
    }
    return 0;
}
