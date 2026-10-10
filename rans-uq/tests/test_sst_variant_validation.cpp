#include "CompressibleSIMPLESolver.hpp"
#include "ForwardModel.hpp"
#include "SIMPLESolver.hpp"
#include <iostream>
#include <stdexcept>
#include <string>

int main() {
    Mesh mesh = Mesh::makeChannel2D(3, 4, 1.0, 1.0);
    mesh.computeWallDistance();
    const auto boundary = FlowBoundaryConditions::channelDefaults(mesh, 1.0, 0.02, 3.0);
    const auto compressibleBoundary =
        CompressibleBoundaryConditions::channelDefaults(mesh, 1.0, 300.0, 101325.0, 0.02, 3.0);
    const auto parameters = InferenceParameterSet::a1_betaStar();
    ObservationOperator observations;
    observations.addVelocityProfile({0.5, 0.4, 0.0}, 0, 0.0, 1.0);
    const auto theta = parameters.pack(SSTCoefficients{});
    for (int variant : {-1, 3, 99, 0, 1, 2}) {
        SolverSettings settings;
        settings.verbose = false;
        settings.maxIterations = 0;
        settings.sstVariant = variant;
        const bool invalid = variant < 0 || variant > 2;
        bool simpleRejected = false, compressibleRejected = false;
        try {
            SIMPLESolver solver(mesh, SSTModel{}, boundary, 0.01, settings);
        } catch (const std::invalid_argument &) {
            simpleRejected = true;
        }
        try {
            CompressibleSIMPLESolver solver(mesh, SSTModel{}, compressibleBoundary, IdealGasEOS{},
                                            settings);
        } catch (const std::invalid_argument &) {
            compressibleRejected = true;
        }
        if (simpleRejected != invalid || compressibleRejected != invalid) {
            std::cerr << "SST variant " << variant << " has incorrect constructor admission\n";
            return 1;
        }
        if (invalid) {
            try {
                ForwardModel model(mesh, parameters, observations, boundary, 0.01, settings);
                std::cerr << "unknown SST variant admitted by the forward constructor\n";
                return 2;
            } catch (const std::invalid_argument &error) {
                if (std::string(error.what()).find("SST variant") == std::string::npos) {
                    std::cerr << "forward constructor did not identify the invalid SST variant\n";
                    return 2;
                }
            }
        } else {
            ForwardModel model(mesh, parameters, observations, boundary, 0.01, settings);
            const auto evaluation = model.evaluateWithDiagnostics(theta);
            if (!evaluation.error.empty() ||
                evaluation.result.status != EvaluationStatus::Unconverged) {
                std::cerr << "valid SST variant was rejected\n";
                return 3;
            }
        }
    }
    std::cout << "three invalid SST selections rejected; Full/NoLimiter/KOmega accepted\n";
}
