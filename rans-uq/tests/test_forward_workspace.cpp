#include "ForwardModel.hpp"
#include <array>
#include <iostream>
#include <limits>
#include <cstring>
static void requireAt(bool b, int line) {
    if (!b)
        throw std::runtime_error("fresh/reused forward workspace mismatch at line " +
                                 std::to_string(line));
}
#define require(b) requireAt((b), __LINE__)
static void same(const FlowFields &a, const FlowFields &b) {
    for (int i = 0; i < a.U.size(); ++i)
        require((a.U[i] - b.U[i]).norm2() == 0);
    for (int i = 0; i < a.U.nBnd(); ++i)
        require((a.U.bnd(i) - b.U.bnd(i)).norm2() == 0);
    for (auto member : {&FlowFields::p, &FlowFields::k, &FlowFields::omega, &FlowFields::nuT,
                        &FlowFields::F1, &FlowFields::F2, &FlowFields::Pk, &FlowFields::CDkw}) {
        const auto &x = a.*member;
        const auto &y = b.*member;
        require(x.data() == y.data());
        for (int i = 0; i < x.nBnd(); ++i)
            require(x.bnd(i) == y.bnd(i));
    }
    require(a.turbEstablished == b.turbEstablished);
}
static bool sameDouble(double a, double b) {
    return std::memcmp(&a, &b, sizeof(double)) == 0;
}
static void same(const ConvergenceHistory &a, const ConvergenceHistory &b) {
    require(a.converged == b.converged && a.diverged == b.diverged && a.stalled == b.stalled &&
            a.finalIter == b.finalIter && a.finalWorstMeasure == b.finalWorstMeasure &&
            sameDouble(a.contractionRateU, b.contractionRateU) &&
            a.entries.size() == b.entries.size());
    for (size_t i = 0; i < a.entries.size(); ++i) {
        const auto &x = a.entries[i];
        const auto &y = b.entries[i];
        require(x.iteration == y.iteration);
        for (auto member : {&ResidualEntry::Ux, &ResidualEntry::Uy, &ResidualEntry::Uz,
                            &ResidualEntry::p, &ResidualEntry::k, &ResidualEntry::omega,
                            &ResidualEntry::momentumScaled, &ResidualEntry::continuityScaled,
                            &ResidualEntry::changeU, &ResidualEntry::remainingU})
            require(sameDouble(x.*member, y.*member));
    }
}
int main() {
    auto mesh = Mesh::makeChannel2D(4, 5, 2, 1);
    mesh.computeWallDistance();
    auto bc = FlowBoundaryConditions::channelDefaults(mesh, 1, .02, 3);
    SolverSettings settings;
    settings.verbose = false;
    settings.maxIterations = 8;
    settings.convergenceTol = 0;
    SIMPLESolver::Workspace workspace(mesh);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (int pass = 0; pass < 3; ++pass) {
        auto coeff = SSTCoefficients{};
        coeff.a1 += .01 * pass;
        settings.turbStartIter = pass;
        settings.turbUpdateInterval = pass + 1;
        FlowFields a(mesh), b(mesh);
        SIMPLESolver fresh(mesh, SSTModel(coeff), bc, .01, settings),
            reused(mesh, SSTModel(coeff), bc, .01, settings);
        fresh.initUniform(a, {1, .01 * pass, 0}, 0, .02, 3);
        reused.initUniform(b, {1, .01 * pass, 0}, 0, .02, 3);
        workspace.pressureCorrection.setUniform(nan);
        workspace.strain.setUniform(nan);
        for (auto *values : {&workspace.residual, &workspace.solveValues, &workspace.diagonal,
                             &workspace.oldK, &workspace.oldOmega, &workspace.interpolatedFlux,
                             &workspace.predictedFlux, &workspace.previousObservations})
            values->assign(mesh.nCells(), nan);
        for (auto *system :
             {&workspace.momentum, &workspace.pressure, &workspace.k, &workspace.omega}) {
            for (auto *values : {&system->diag, &system->source, &system->lower, &system->upper})
                std::fill(values->begin(), values->end(), nan);
        }
        workspace.oldVelocity.assign(mesh.nCells(), {nan, nan, nan});
        auto ah = fresh.solve(a), bh = reused.solve(b, workspace);
        same(a, b);
        same(ah, bh);
        require(fresh.faceFluxes() == reused.faceFluxes());
    }
    SIMPLESolver interrupted(mesh, SSTModel{}, bc, .01, settings);
    FlowFields a(mesh);
    interrupted.initUniform(a, {1, 0, 0}, 0, .02, 3);
    interrupted.setSourceObserver(
        [](const auto &) { throw std::runtime_error("observer interruption"); });
    bool threw = false;
    try {
        interrupted.solve(a, workspace);
    } catch (const std::runtime_error &) {
        threw = true;
    }
    require(threw && !workspace.active);
    interrupted.setSourceObserver({});
    interrupted.initUniform(a, {1, 0, 0}, 0, .02, 3);
    auto recovered = interrupted.solve(a, workspace);
    SIMPLESolver fresh(mesh, SSTModel{}, bc, .01, settings);
    FlowFields b(mesh);
    fresh.initUniform(b, {1, 0, 0}, 0, .02, 3);
    auto reference = fresh.solve(b);
    same(a, b);
    same(recovered, reference);
    ObservationOperator obs;
    obs.addVelocityProfile({1, .4, 0}, 0, 0, 1);
    const auto parameters = InferenceParameterSet::a1_betaStar();
    ForwardModel shared(mesh, parameters, obs, bc, .01, settings);
    shared.cache().setByteBudget(0);
    for (int pass = 0; pass < 3; ++pass) {
        ForwardModel isolated(mesh, parameters, obs, bc, .01, settings);
        isolated.cache().setByteBudget(0);
        if (pass == 1) {
            std::vector<double> target(6 * mesh.nCells());
            shared.setTargetAnisotropy(target);
            isolated.setTargetAnisotropy(target);
        } else
            shared.clearTargetAnisotropy();
        const std::vector<double> theta{.31 + .01 * pass, .09};
        const auto x = shared.evaluateWithDiagnostics(theta, true),
                   y = isolated.evaluateWithDiagnostics(theta, true);
        require(x.result.predictions == y.result.predictions &&
                x.result.status == y.result.status && x.result.loglik == y.result.loglik);
        require(!x.usedWarmStart && x.snapshot && y.snapshot &&
                x.snapshot->faceFluxes == y.snapshot->faceFluxes);
        same(x.convergence, y.convergence);
        same(shared.lastFields(), isolated.lastFields());
        require(shared.evaluate({-1, .09}).status == EvaluationStatus::InvalidParameters);
    }
    mesh.computeGeometry();
    threw = false;
    try {
        fresh.solve(b, workspace);
    } catch (const std::logic_error &) {
        threw = true;
    }
    require(threw);
    std::cout
        << "fresh/reused full states, boundaries, fluxes, histories and observer recovery passed\n";
}
