#include "MatrixFreeSolver.hpp"
#include <cmath>
#include <iostream>
#include <limits>

// Qualify the comparison itself before checking actual numerical results.
static bool within(double actual, double expected, double tolerance) {
    return std::isfinite(actual) && std::isfinite(expected) &&
           std::abs(actual - expected) <= tolerance;
}

static bool comparisonRejectsNonfinite() {
    for (double invalid :
         {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(),
          -std::numeric_limits<double>::infinity()}) {
        if (within(invalid, 1.0, 1e-12) || within(1.0, invalid, 1e-12) ||
            within(invalid, invalid, 1e-12))
            return false;
    }
    return within(1.0, 1.0, 1e-12) && !within(1.0, 2.0, 1e-12);
}

int main() {
    if (!comparisonRejectsNonfinite()) {
        std::cerr << "Krylov residual oracle accepts nonfinite values\n";
        return 2;
    }
    const auto identity = [](const std::vector<double> &x, std::vector<double> &y) { y = x; };
    const auto zero = [](const std::vector<double> &x, std::vector<double> &y) {
        y.assign(x.size(), 0.0);
    };
    std::vector<double> b{1.0, -2.0}, x(2, 0.0);
    auto failed = fgmresMatrixFree(zero, identity, b, x, 2, 3, 1e-12);
    if (failed.converged || !within(failed.finalRes, 1.0, 1e-14)) {
        std::cerr << "FAIL: Arnoldi breakdown on A=0 must not certify b-Ax=0\n";
        return 1;
    }
    x.assign(2, 0.0);
    auto solved = fgmresMatrixFree(identity, identity, b, x, 2, 3, 1e-12);
    if (!solved.converged || solved.iterations != 1 || solved.matvecs < 2 ||
        !within(x[0], b[0], 1e-14) || !within(x[1], b[1], 1e-14)) {
        std::cerr << "FAIL: exact one-step solve needs a checked residual and one iteration\n";
        return 1;
    }
    // Restarted progress must count every step, including a truncated final cycle.
    const auto diagonal = [](const std::vector<double> &input, std::vector<double> &output) {
        output[0] = input[0];
        output[1] = 7.0 * input[1];
    };
    FGMRESWorkspace workspace;
    x.assign(2, 0.0);
    auto capped = fgmresMatrixFree(diagonal, identity, b, x, 1, 3, 1e-15, &workspace);
    const double residual = std::hypot(b[0] - x[0], b[1] - 7.0 * x[1]) / std::hypot(b[0], b[1]);
    if (capped.converged || capped.iterations != 3 || !within(capped.finalRes, residual, 1e-14)) {
        std::cerr << "FAIL: capped restart count and explicit residual must agree\n";
        return 1;
    }
    const auto *basis = workspace.V[0].data();
    x.assign(2, 0.0);
    auto reused = fgmresMatrixFree(identity, identity, b, x, 1, 3, 1e-12, &workspace);
    if (!reused.converged || workspace.V[0].data() != basis ||
        FGMRESWorkspace::basisBytes(500000, 150) != size_t{1204000000}) {
        std::cerr << "FAIL: sequential RHS solves reuse basis storage and expose its byte cost\n";
        return 1;
    }
    // A previously solved initial guess needs a checked residual but no iteration.
    auto initial = fgmresMatrixFree(identity, identity, b, x, 1, 0, 1e-12, &workspace);
    if (!initial.converged || initial.iterations != 0 || initial.matvecs != 1)
        return 1;
    std::cout << "Krylov residual checks passed\n";
}
