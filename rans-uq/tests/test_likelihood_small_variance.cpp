#include "ObservationOperator.hpp"
#include <cmath>
#include <iostream>
#include <limits>

int main() {
    // For residual == standard deviation, the documented unnormalized
    // Gaussian log-likelihood is -1/2 at every physical unit scale.
    for (double sigma : {1., 1e-16, 1e-20}) {
        ObservationOperator observations;
        observations.addVelocityProfile({0, 0, 0}, 0, 0, sigma);
        const double actual = observations.logLikelihood({sigma});
        std::cout << "sigma=" << sigma << " likelihood=" << actual << '\n';
        // A few floating point operations allow roundoff, not a changed noise model.
        if (!std::isfinite(actual) ||
            std::abs(actual + .5) > 4 * std::numeric_limits<double>::epsilon()) {
            std::cerr << "FAIL: declared Gaussian variance was replaced by a floor\n";
            return 1;
        }
    }
    return 0;
}
