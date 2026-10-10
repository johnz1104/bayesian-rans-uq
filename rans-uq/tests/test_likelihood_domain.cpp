#include "ForwardModel.hpp"
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>

int main() {
    int failures = 0;
    auto check = [&](bool valid, const char *message) {
        if (!valid) {
            std::cerr << "FAIL: " << message << '\n';
            ++failures;
        }
    };
    auto rejects = [&](const std::function<void()> &operation, const char *message) {
        try {
            operation();
            check(false, message);
        } catch (const std::invalid_argument &) {
        } catch (const std::overflow_error &) {
        }
    };
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    for (int component : {-1, 3, 99})
        rejects(
            [&] {
                ObservationOperator obs;
                obs.addVelocityProfile({0., 0., 0.}, component, 0., 1.);
            },
            "invalid velocity component was accepted");
    for (double bad : {0., -1., nan, inf}) {
        rejects([&] { ObservationOperator{}.addDrag("wall", 0., 1., bad, 1.); },
                "invalid drag reference area was accepted");
        rejects([&] { ObservationOperator{}.addDrag("wall", 0., 1., 1., bad); },
                "invalid drag reference speed was accepted");
        rejects([&] { ObservationOperator{}.addPressureTap({}, 0., 1., bad); },
                "invalid pressure reference speed was accepted");
        rejects([&] { ObservationOperator{}.addPressureTap({}, 0., 1., 1., 0., 0., bad); },
                "invalid pressure reference density was accepted");
        rejects([&] { ObservationOperator{}.addSkinFriction("wall", {}, 0., 1., bad); },
                "invalid friction reference speed was accepted");
    }
    for (double bad : {nan, inf}) {
        rejects([&] { ObservationOperator{}.addPressureTap({}, 0., 1., 1., 0., bad); },
                "nonfinite pressure reference was accepted");
        rejects([&] { ObservationOperator{}.addVelocityProfile({bad, 0., 0.}, 0, 0., 1.); },
                "nonfinite velocity-probe location was accepted");
        rejects([&] { ObservationOperator{}.addPressureTap({0., bad, 0.}, 0., 1., 1.); },
                "nonfinite pressure-probe location was accepted");
        rejects([&] { ObservationOperator{}.addSkinFriction("wall", {0., 0., bad}, 0., 1., 1.); },
                "nonfinite friction-probe location was accepted");
    }
    for (const Measurement measurement : {Measurement{nan, 1, 0},
                                          {inf, 1, 0},
                                          {0, nan, 0},
                                          {0, inf, 0},
                                          {0, -1, 0},
                                          {0, 1, -1},
                                          {0, 1, nan},
                                          {0, 1, inf},
                                          {0, 0, 0},
                                          {0, 1e200, 0}}) {
        for (int kind = 0; kind < 6; ++kind) {
            rejects(
                [&] {
                    ObservationOperator obs;
                    const auto [value, sigma, discrepancy] = measurement;
                    switch (kind) {
                    case 0:
                        obs.addVelocityProfile({1, .5, 0}, 0, value, sigma, discrepancy);
                        break;
                    case 1:
                        obs.addPressureTap({1, .5, 0}, value, sigma, 1, discrepancy);
                        break;
                    case 2:
                        obs.addDrag("top", value, sigma, 1, 1, discrepancy);
                        break;
                    case 3:
                        obs.addSkinFriction("top", {1, 1, 0}, value, sigma, 1, discrepancy);
                        break;
                    case 4:
                        obs.addSeparationPoint("top", value, sigma, discrepancy);
                        break;
                    case 5:
                        obs.addReattachmentLength("top", value, sigma, discrepancy);
                        break;
                    }
                    (void)obs.logLikelihood({1.0});
                },
                "invalid Gaussian measurement was accepted");
        }
    }
    ObservationOperator valid;
    valid.addVelocityProfile({1, .5, 0}, 0, 1, 3, 4);
    // Independent arithmetic: variance=3^2+4^2=25, residual=5, -residual^2/(2*variance)=-1/2.
    check(valid.logLikelihood({6}) == -.5, "independent Gaussian likelihood control");
    rejects([&] { (void)valid.logLikelihood({nan}); }, "NaN prediction was accepted");
    rejects([&] { (void)valid.logLikelihood({inf}); }, "infinite prediction was accepted");
    rejects([&] { (void)valid.logLikelihood({1e300}); }, "overflowed likelihood was accepted");
    rejects([&] { (void)valid.logLikelihood({}); }, "wrong prediction count was accepted");
    ObservationOperator discrepancyOnly;
    discrepancyOnly.addVelocityProfile({1, .5, 0}, 0, 0, 0, 2);
    check(discrepancyOnly.logLikelihood({2}) == -.5,
          "zero observation noise with positive discrepancy remains valid");
    Mesh mesh = Mesh::makeChannel2D(4, 8, 2, 1, 10, 1);
    mesh.computeWallDistance();
    auto boundaries = FlowBoundaryConditions::couetteDefaults(mesh, 1, 1e-4, 1);
    SolverSettings settings;
    settings.maxIterations = 12000;
    settings.convergenceTol = 1e-7;
    settings.nuTFloorIters = 0;
    settings.verbose = false;
    const auto parameters = InferenceParameterSet::a1_betaStar();
    const auto theta = parameters.pack(SSTCoefficients{});
    ObservationOperator overflow;
    overflow.addVelocityProfile({1, .5, 0}, 0, 1e300, 1);
    ForwardModel model(mesh, parameters, overflow, boundaries, .1, settings);
    const auto result = model.evaluateWithDiagnostics(theta, true);
    check(result.convergence.converged, "fixture must converge before observation failure");
    check(result.result.status == EvaluationStatus::Unknown && !result.error.empty(),
          "unrepresentable likelihood is an explicit failed evaluation");
    check(result.result.predictions.empty() && !result.snapshot && !model.hasLastFields(),
          "observation failure must clear predictions and retained state");
    check(model.cache().size() == 0, "observation failure must not cache the failed evaluation");
    std::cout << "likelihood domain failures=" << failures << '\n';
    return failures ? 1 : 0;
}
