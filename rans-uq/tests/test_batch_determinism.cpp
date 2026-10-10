#include "ForwardBatch.hpp"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace {
bool same(double a, double b) {
    return std::memcmp(&a, &b, sizeof(double)) == 0;
}
void compare(const ForwardBatchRow &a, const ForwardBatchRow &b) {
    if (a.result.status != EvaluationStatus::Converged || b.result.status != a.result.status ||
        a.result.simpleIters != b.result.simpleIters ||
        a.result.predictions != b.result.predictions || !same(a.result.loglik, b.result.loglik) ||
        !same(a.finalWorstMeasure, b.finalWorstMeasure) ||
        !same(a.contractionRateU, b.contractionRateU) || a.error != b.error ||
        a.stalled != b.stalled)
        throw std::runtime_error("converged batch output depends on worker count or row order");
}
} // namespace

int main() {
    try {
        Mesh mesh = Mesh::makeChannel2D(3, 4, 1, 1);
        mesh.computeWallDistance();
        const auto parameters = InferenceParameterSet::a1_betaStar();
        auto bc = FlowBoundaryConditions::channelDefaults(mesh, 1, 0.01, 3);
        for (int patch = 0; patch < mesh.nPatches(); ++patch)
            if (mesh.patch(patch).type == "inlet")
                for (int fi : mesh.patch(patch).faces) {
                    const double y = mesh.face(fi).center.y;
                    bc.velocityBC[patch].vecProfile.push_back({0.7 + 0.4 * y, 0.06 * (0.2 + y), 0});
                }
        SolverSettings settings;
        settings.verbose = false;
        settings.maxIterations = 20000;
        settings.convergenceTol = 1e-8;
        settings.nuTFloorIters = 0;
        settings.innerTolerance = 1e-10;
        settings.innerIterations = 400;
        settings.momentumConvection = settings.turbulenceConvection = "upwind";
        ObservationOperator observations;
        observations.addVelocityProfile({0.63, 0.41, 0}, 0, 0, 1);
        observations.addPressureTap({0.63, 0.41, 0}, 0, 1, 1);
        ForwardBatch batch(mesh, parameters, observations, bc, 0.003, settings, {1, 0, 0}, 0, 0.01,
                           3);
        if (batch.nParameters() != 2 || batch.nObservations() != 2)
            throw std::runtime_error("batch dimensions do not match the supplied problem");
        const std::vector<std::vector<double>> theta{{0.31, 0.09}, {0.34, 0.10}, {0.29, 0.08}};
        const auto serial = batch.evaluate(theta, 1);
        auto reversed = theta;
        std::reverse(reversed.begin(), reversed.end());
        const auto parallel = batch.evaluate(theta, 2);
        const auto permuted = batch.evaluate(reversed, 2);
        const auto repeated = batch.evaluate(theta, 1);
        for (size_t row = 0; row < theta.size(); ++row) {
            compare(serial.rows[row], parallel.rows[row]);
            compare(serial.rows[row], permuted.rows[theta.size() - row - 1]);
            compare(serial.rows[row], repeated.rows[row]);
            if (!serial.converged[row] || !parallel.converged[row] ||
                !permuted.converged[theta.size() - row - 1])
                throw std::runtime_error("batch convergence mask disagrees with the result");
        }
        if (serial.rows.front().result.predictions == serial.rows.back().result.predictions)
            throw std::runtime_error(
                "determinism fixture must resolve the different coefficient rows");
        std::cout
            << "converged coefficient rows are bit-identical across repeats, order and workers\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
