#include "ParameterSensitivity.hpp"
#include <iostream>
#include <cmath>
#include <limits>

// Kept as a helper so independent nonfinite controls exercise the same oracle
// used for actual derivative columns below. Tolerances are unchanged.
static bool derivativesAgree(double a, double b) {
    return std::isfinite(a) && std::isfinite(b) &&
           std::abs(a - b) <= 1e-7 + 1e-3 * std::max(std::abs(a), std::abs(b));
}

// Full-FD regressions, not ambient-Jacobian certification: these symmetric
// flows can have zero transverse flux. Fixed unit nondimensional scales.
int main() {
    for (double value :
         {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(),
          -std::numeric_limits<double>::infinity()}) {
        if (derivativesAgree(value, 1.) || derivativesAgree(1., value) ||
            derivativesAgree(value, value)) {
            std::cerr << "nonfinite derivative passed refinement oracle\n";
            return 4;
        }
    }
    if (!derivativesAgree(1., 1.) || derivativesAgree(1., 2.))
        return 5;
    for (bool periodic : {false, true}) {
        Mesh mesh = periodic ? Mesh::makePeriodicChannel2D(3, 4, 1, .5, 20)
                             : Mesh::makeChannel2D(3, 4, 1, 1);
        mesh.computeWallDistance();
        auto bc = periodic ? FlowBoundaryConditions::channelDefaults(mesh, 1, 1e-4, 1)
                           : FlowBoundaryConditions::couetteDefaults(mesh, 1, 1e-4, 1);
        SolverSettings settings;
        settings.verbose = false;
        settings.maxIterations = 15000;
        settings.convergenceTol = 1e-9;
        settings.nuTFloorIters = 0;
        settings.momentumConvection = settings.turbulenceConvection = "upwind";
        if (periodic)
            settings.bodyForce.x = .2;
        ObservationOperator obs;
        obs.addVelocityProfile({.43, .38, 0}, 0, 0, 1);
        if (!periodic)
            obs.addPressureTap({.6, .4, 0}, 0, 1, 1);
        ParameterSensitivity sensitivity(mesh, obs, bc, .1, settings);
        const auto theta = InferenceParameterSet::toVector(SSTCoefficients{});
        if (sensitivity.solveState(theta) != EvaluationStatus::Converged)
            return 1;
        std::vector<double> steps(11);
        for (int j = 0; j < 11; ++j)
            steps[j] = theta[j] * 2e-4;
        const auto coarse = sensitivity.etaJacobianWarmFD(theta, 2e-4, 1e-9, 15000, 1e-10, steps);
        for (double &h : steps)
            h *= .5;
        const auto fine = sensitivity.etaJacobianWarmFD(theta, 1e-4, 1e-9, 15000, 1e-11, steps);
        for (int j = 0; j < 11; ++j) {
            if (!coarse.krylovConverged[j] || !fine.krylovConverged[j])
                return 2;
            for (int i = 0; i < obs.nObs(); ++i) {
                const double a = coarse.dObsDTheta[i][j], b = fine.dObsDTheta[i][j];
                if (!derivativesAgree(a, b)) {
                    std::cerr << "FD refinement failed periodic=" << periodic << " col=" << j
                              << " obs=" << i << " coarse=" << a << " fine=" << b << '\n';
                    return 3;
                }
            }
        }
        std::cout << "full-FD refinement passed all eleven columns, periodic=" << periodic << '\n';
    }
}
