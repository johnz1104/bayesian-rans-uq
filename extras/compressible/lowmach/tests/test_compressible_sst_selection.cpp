#include "CompressibleForwardModel.hpp"
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>

static void sameScalar(const ScalarField &actual, const ScalarField &expected) {
    if (actual.size() != expected.size() || actual.nBnd() != expected.nBnd())
        throw std::runtime_error("field shapes differ");
    for (int i = 0; i < actual.size(); ++i)
        if (!std::isfinite(actual[i]) || actual[i] != expected[i])
            throw std::runtime_error("configured closure differs from explicitly selected closure");
    for (int i = 0; i < actual.nBnd(); ++i)
        if (!std::isfinite(actual.bnd(i)) || actual.bnd(i) != expected.bnd(i))
            throw std::runtime_error("configured closure boundary differs from explicit closure");
}

int main() {
    try {
        Mesh mesh = Mesh::makeChannel2D(3, 8, 2.0, 1.0);
        mesh.computeWallDistance();
        const auto parameters = InferenceParameterSet::a1_betaStar();
        const auto theta = parameters.pack(SSTCoefficients{});
        IdealGasEOS eos;
        const auto boundary =
            CompressibleBoundaryConditions::channelDefaults(mesh, 10.0, 300.0, 101325.0, 0.2, 3.0);
        ObservationOperator observations;
        observations.addVelocityProfile({1.0, 0.4, 0.0}, 0, 0.0, 1.0);
        SolverSettings settings;
        settings.verbose = false;
        settings.maxIterations = 20;
        settings.turbStartIter = 0;
        settings.turbUpdateInterval = 1;
        settings.nuTFloorIters = 0;
        const std::array<SSTVariant, 3> variants{SSTVariant::Full, SSTVariant::NoLimiter,
                                                 SSTVariant::KOmega};
        std::vector<double> fullViscosity;
        for (int selector = 0; selector < 3; ++selector) {
            settings.sstVariant = selector;
            CompressibleForwardModel model(mesh, parameters, observations, boundary, eos, settings,
                                           {10, 0, 0}, 101325.0, 300.0, 0.2, 3.0);
            const auto result = model.evaluate(theta);
            SSTModel selected(parameters.unpack(theta), variants[selector]);
            CompressibleSIMPLESolver direct(mesh, selected, boundary, eos, settings);
            CompressibleFlowFields expected(mesh);
            direct.initUniform(expected, {10, 0, 0}, 101325.0, 300.0, 0.2, 3.0);
            const auto history = direct.solve(expected);
            // Deliberately capped: this checks configuration routing, not steady physics.
            if (result.status != EvaluationStatus::Unconverged || history.diverged ||
                history.converged || !model.hasLastFields())
                throw std::runtime_error("closure selection fixture is not a finite capped run");
            const auto &actual = model.lastCompressibleFields();
            for (auto field : {&CompressibleFlowFields::p, &CompressibleFlowFields::T,
                               &CompressibleFlowFields::rho, &CompressibleFlowFields::k,
                               &CompressibleFlowFields::omega, &CompressibleFlowFields::nuT,
                               &CompressibleFlowFields::F1, &CompressibleFlowFields::F2})
                sameScalar(actual.*field, expected.*field);
            for (int i = 0; i < mesh.nCells(); ++i)
                if ((actual.U[i] - expected.U[i]).norm2() != 0.0)
                    throw std::runtime_error("configured velocity differs from explicit closure");
            if (selector == 0)
                fullViscosity = expected.nuT.data();
            else if (expected.nuT.data() == fullViscosity)
                throw std::runtime_error("fixture does not distinguish closure variants");
            std::cout << "SST selector " << selector << " exactly matches explicit model\n";
        }
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
