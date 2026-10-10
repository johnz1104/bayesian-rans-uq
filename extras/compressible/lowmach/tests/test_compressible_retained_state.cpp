#include "CompressibleForwardModel.hpp"
#include <iostream>
#include <stdexcept>

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
    const auto first = model.evaluate(theta);
    if (first.status != EvaluationStatus::Unconverged || !model.hasLastFields()) {
        std::cerr << "FAIL: capped solve must retain its own diagnostic state\n";
        return 1;
    }
    const auto invalid = model.evaluate({-1, theta[1]});
    if (invalid.status != EvaluationStatus::InvalidParameters || model.hasLastFields()) {
        std::cerr << "FAIL: invalid evaluation exposes its predecessor's fields\n";
        return 1;
    }
    bool rejected = false;
    try {
        (void)model.lastCompressibleFields();
    } catch (const std::logic_error &) {
        rejected = true;
    }
    if (!rejected) {
        std::cerr << "FAIL: native field getter must reject unavailable state\n";
        return 1;
    }
    const auto recovered = model.evaluate(theta);
    if (recovered.status != EvaluationStatus::Unconverged || !model.hasLastFields() ||
        model.lastCompressibleFields().U.size() != mesh.nCells()) {
        std::cerr << "FAIL: next valid evaluation must expose its own diagnostic state\n";
        return 1;
    }
    return 0;
}
