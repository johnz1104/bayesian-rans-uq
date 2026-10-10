#include "ParameterSensitivity.hpp"
#include <iostream>
#include <cmath>

// Independent partial-pivoting dense elimination; no production Krylov code.
static std::vector<double> denseSolve(std::vector<std::vector<double>> a, std::vector<double> b) {
    const size_t n = b.size();
    for (size_t k = 0; k < n; ++k) {
        size_t pivot = k;
        for (size_t r = k + 1; r < n; ++r)
            if (std::abs(a[r][k]) > std::abs(a[pivot][k]))
                pivot = r;
        if (std::abs(a[pivot][k]) < 1e-14)
            throw std::runtime_error("singular dense reference");
        std::swap(a[k], a[pivot]);
        std::swap(b[k], b[pivot]);
        for (size_t r = k + 1; r < n; ++r) {
            const double factor = a[r][k] / a[k][k];
            for (size_t c = k + 1; c < n; ++c)
                a[r][c] -= factor * a[k][c];
            b[r] -= factor * b[k];
        }
    }
    for (size_t r = n; r-- > 0;) {
        for (size_t c = r + 1; c < n; ++c)
            b[r] -= a[r][c] * b[c];
        b[r] /= a[r][r];
    }
    return b;
}
static bool close(double a, double b, double atol, double rtol) {
    return std::isfinite(a) && std::isfinite(b) &&
           std::abs(a - b) <= atol + rtol * std::max(std::abs(a), std::abs(b));
}

// Deterministic asymmetric developing outlet candidate. No external data/RNG.
// Unit length/velocity/density; fixed state scales U=1,k=.01,omega=100,p=1,flux=1.
int main() {
    Mesh mesh = Mesh::makeChannel2D(3, 4, 1.0, 1.0);
    mesh.computeWallDistance();
    auto bc = FlowBoundaryConditions::channelDefaults(mesh, 1.0, .01, 3.0);
    for (int pi = 0; pi < mesh.nPatches(); ++pi) {
        if (mesh.patch(pi).type == "inlet")
            for (int fi : mesh.patch(pi).faces) {
                const double y = mesh.face(fi).center.y;
                bc.velocityBC[pi].vecProfile.push_back({.7 + .4 * y, .06 * (.2 + y), 0});
            }
    }
    SolverSettings settings;
    settings.verbose = false;
    settings.maxIterations = 20000;
    settings.convergenceTol = 1e-10;
    settings.nuTFloorIters = 0;
    settings.innerTolerance = 1e-10;
    settings.innerIterations = 400;
    settings.momentumConvection = settings.turbulenceConvection = "upwind";
    ObservationOperator obs;
    obs.addVelocityProfile({.63, .41, 0}, 0, 0, 1);
    obs.addPressureTap({.63, .41, 0}, 0, 1, 1);
    ParameterSensitivity sensitivity(mesh, obs, bc, .003, settings);
    const auto theta = InferenceParameterSet::toVector(SSTCoefficients{});
    const auto status = sensitivity.solveState(theta);
    if (status != EvaluationStatus::Converged) {
        std::cout << "candidate not qualified: primal status=" << statusString(status) << '\n';
        return 1;
    }
    const auto base = sensitivity.steadyBase();
    if (!base.history.converged || base.coefficients != theta || base.sourceDigest.size() != 64 ||
        base.correctedFluxes.size() != static_cast<size_t>(mesh.nFaces()))
        return 2;
    const auto q = sensitivity.steadyStateVector();
    const size_t n = q.size();
    if (sensitivity.steadyKrylovBasisBytes(7) != 15 * n * sizeof(double))
        return 3;
    SteadySensitivityOptions options;
    options.stateScales = {{1, 1, .01, 100, 1, 1}};
    options.equationScales = {{1, 1, .01, 100, 1, 1}};
    options.restart = static_cast<int>(n);
    options.actionStep = 1e-5;
    // The accepted 1e-8 ceiling alone does not guarantee accurate weakly
    // coupled observation derivatives: this fixture needs a tighter solve.
    options.krylovTolerance = 1e-10;
    const auto tangent = sensitivity.etaJacobianSteady(options);
    if (!tangent.experimental || tangent.stateTangents.size() != 11)
        return 4;
    std::cout << "candidate normalized steady residual=" << tangent.normalizedBaseResidual << "; "
              << tangent.qualification << '\n';
    const auto fixedPoint = sensitivity.steadyResidualAt(q, theta);
    for (size_t block = 0; block < 6; ++block) {
        double maximum = 0;
        for (size_t i = block * mesh.nCells();
             i < fixedPoint.equations.size() && (block == 5 || i < (block + 1) * mesh.nCells());
             ++i)
            maximum = std::max(maximum,
                               std::abs(fixedPoint.equations[i]) / options.equationScales[block]);
        std::cout << "fixed-point block " << block << " residual " << maximum << '\n';
    }
    // Convergence of the iteration must also solve its declared constrained
    // steady equations. This is a primal consistency gate, not certification
    // of a tangent. A capped or branch-unstable candidate cannot pass it.
    if (tangent.normalizedBaseResidual > 1e-8)
        return 9;
    for (int j = 0; j < 11; ++j) {
        std::cout << "tangent column " << j << " converged "
                  << tangent.derivatives.krylovConverged[j] << " residual "
                  << tangent.derivatives.krylovRelRes[j] << '\n';
        if (!tangent.derivatives.krylovConverged[j] || !tangent.branchQualified[j])
            return 10;
    }
    auto tighterSettings = settings;
    tighterSettings.convergenceTol *= .1;
    tighterSettings.innerTolerance *= .1;
    ParameterSensitivity tighter(mesh, obs, bc, .003, tighterSettings);
    if (tighter.solveState(theta) != EvaluationStatus::Converged)
        return 7;
    const auto tight = tighter.etaJacobianSteady(options);
    std::cout << "tenfold tighter base residual=" << tight.normalizedBaseResidual << '\n';
    if (tight.normalizedBaseResidual > 1e-8)
        return 8;
    // Independently assemble the dense central-difference matrix columnwise.
    // This check is meaningful for actions even if the base steady residual
    // disqualifies total derivatives. No tangent is certified by it alone.
    const int nc = mesh.nCells();
    const double step = 1e-7;
    std::vector<std::vector<double>> dense(n, std::vector<double>(n));
    std::vector<double> direction(n), denseAction(n, 0);
    for (size_t c = 0; c < n; ++c) {
        const double scale = options.stateScales[std::min(c / static_cast<size_t>(nc), size_t{5})];
        auto plus = q, minus = q;
        plus[c] += step * scale;
        minus[c] -= step * scale;
        const auto rp = sensitivity.steadyResidualAt(plus, theta).equations;
        const auto rm = sensitivity.steadyResidualAt(minus, theta).equations;
        for (size_t r = 0; r < n; ++r)
            dense[r][c] = (rp[r] - rm[r]) / (2 * step * scale);
        direction[c] = scale * std::sin(.17 * (c + 1));
    }
    try {
        const auto action =
            sensitivity.steadyStateAction(direction, step * linalg::norm(direction));
        const auto coarseAction =
            sensitivity.steadyStateAction(direction, 2 * step * linalg::norm(direction));
        for (size_t r = 0; r < n; ++r) {
            for (size_t c = 0; c < n; ++c)
                denseAction[r] += dense[r][c] * direction[c];
            const double scale =
                options.equationScales[std::min(r / static_cast<size_t>(nc), size_t{5})];
            const double a = action[r] / scale, b = denseAction[r] / scale;
            if (!close(a, b, 1e-8, 1e-5) || !close(a, coarseAction[r] / scale, 1e-8, 1e-5)) {
                std::cerr << "action mismatch row=" << r << " matrix-free=" << a << " dense=" << b
                          << '\n';
                return 6;
            }
        }
        std::cout << "dense/state-action comparison passed\n";
    } catch (const std::domain_error &e) {
        std::cerr << "candidate state-action branch gate: " << e.what() << '\n';
        return 11;
    }
    auto refinedOptions = options;
    refinedOptions.actionStep *= .5;
    refinedOptions.parameterRelativeStep *= .5;
    const auto refined = sensitivity.etaJacobianSteady(refinedOptions);
    for (int j = 0; j < 11; ++j) {
        if (!refined.derivatives.krylovConverged[j] || !tight.derivatives.krylovConverged[j]) {
            std::cerr << "refined/tight Krylov column failed " << j << " residuals "
                      << refined.derivatives.krylovRelRes[j] << " "
                      << tight.derivatives.krylovRelRes[j] << '\n';
            return 12;
        }
        const double h = 2e-5 * theta[j];
        auto tp = theta, tm = theta;
        tp[j] += h;
        tm[j] -= h;
        const auto rp = sensitivity.steadyResidualAt(q, tp).equations;
        const auto rm = sensitivity.steadyResidualAt(q, tm).equations;
        auto scaled = dense;
        std::vector<double> rhs(n);
        for (size_t r = 0; r < n; ++r) {
            const double rowScale =
                options.equationScales[std::min(r / static_cast<size_t>(nc), size_t{5})];
            rhs[r] = -(rp[r] - rm[r]) / (2 * h * rowScale);
            for (size_t c = 0; c < n; ++c)
                scaled[r][c] *=
                    options.stateScales[std::min(c / static_cast<size_t>(nc), size_t{5})] /
                    rowScale;
        }
        // These are total state derivatives, compared at the declared total
        // derivative tolerance; operator actions above use 1e-8 / 1e-5.
        const auto reference = denseSolve(scaled, rhs);
        for (size_t r = 0; r < n; ++r) {
            const double colScale =
                options.stateScales[std::min(r / static_cast<size_t>(nc), size_t{5})];
            if (!close(reference[r], tangent.stateTangents[j][r] / colScale, 1e-7, 1e-3)) {
                std::cerr << "dense tangent mismatch column " << j << " row " << r << " values "
                          << reference[r] << " " << tangent.stateTangents[j][r] / colScale << '\n';
                return 13;
            }
        }
    }
    std::cout << "all eleven dense/matrix-free tangent comparisons passed\n";
    std::vector<std::vector<double>> previous(11, std::vector<double>(obs.nObs()));
    for (int refinement = 0; refinement < 3; ++refinement) {
        const double relativeStep = refinement == 0 ? 2e-4 : 1e-4;
        const auto solveSettings = refinement == 2 ? tighterSettings : settings;
        auto &reference = refinement == 2 ? tighter : sensitivity;
        std::vector<double> steps(11);
        for (int j = 0; j < 11; ++j)
            steps[j] = relativeStep * theta[j];
        const auto warm =
            reference.etaJacobianWarmFD(theta, relativeStep, 1e-12, solveSettings.maxIterations,
                                        solveSettings.convergenceTol, steps);
        for (int j = 0; j < 11; ++j) {
            auto tp = theta, tm = theta;
            tp[j] += steps[j];
            tm[j] -= steps[j];
            ParameterSensitivity plus(mesh, obs, bc, .003, solveSettings),
                minus(mesh, obs, bc, .003, solveSettings);
            if (plus.solveState(tp) != EvaluationStatus::Converged ||
                minus.solveState(tm) != EvaluationStatus::Converged || !warm.krylovConverged[j])
                return 14;
            // Qualify the actual cold re-solves at both steps and tolerances.
            reference.requireSteadyPair(plus.steadyStateVector(), minus.steadyStateVector(), tp,
                                        tm);
            const auto ep = obs.evaluate(mesh, plus.steadyBase().fields, .003);
            const auto em = obs.evaluate(mesh, minus.steadyBase().fields, .003);
            for (int i = 0; i < obs.nObs(); ++i) {
                const double fd = (ep[i] - em[i]) / (tp[j] - tm[j]);
                for (double value :
                     {tangent.derivatives.dObsDTheta[i][j], refined.derivatives.dObsDTheta[i][j],
                      tight.derivatives.dObsDTheta[i][j], warm.dObsDTheta[i][j]}) {
                    if (!close(fd, value, 1e-7, 1e-3)) {
                        std::cerr << "total derivative mismatch refinement " << refinement
                                  << " column " << j << " observation " << i << " FD " << fd
                                  << " candidate " << value << '\n';
                        return 15;
                    }
                }
                if (refinement && !close(fd, previous[j][i], 1e-7, 1e-3))
                    return 16;
                previous[j][i] = fd;
            }
        }
        std::cout << "all eleven full-FD/branch/observation columns passed refinement "
                  << refinement << '\n';
    }
    auto legacySettings = settings;
    legacySettings.legacyOmegaWallPinning = true;
    ParameterSensitivity legacy(mesh, obs, bc, .003, legacySettings);
    if (legacy.solveState(theta) != EvaluationStatus::Converged)
        return 17;
    try {
        legacy.etaJacobianSteady(options);
        return 18;
    } catch (const std::logic_error &) {
    }
    return 0;
}
