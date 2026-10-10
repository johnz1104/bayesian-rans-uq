#include "HillsProductionPolicy.hpp"
#include <BuildInfo.hpp>
#include <array>
#include <iostream>
#include <limits>
static void require(bool condition, const char *message) {
    if (!condition)
        throw std::runtime_error(message);
}
static void close(double a, double b) {
    require(std::abs(a - b) <= 1e-12 * std::max(1., std::abs(b)), "production mismatch");
}
static Vec3 velocity(const Vec3 &p) {
    return {-2 * p.x + 3 * p.y, 5 * p.x - 4 * p.y, 7 * p.x + 11 * p.y};
}
struct StagePolicy final : SIMPLESolver::ProductionPolicy {
    HillsTargetProduction policy;
    mutable int kCalls = 0, omegaCalls = 0;
    mutable std::vector<double> lastProduction;
    void kProduction(const SIMPLESolver::ProductionContext &c,
                     std::vector<double> &out) const override {
        ++kCalls;
        policy.kProduction(c, out);
        lastProduction = out;
    }
    double omegaProduction(const SIMPLESolver::ProductionContext &c, int cell) const override {
        ++omegaCalls;
        require(c.fields.Pk[cell] == lastProduction[cell], "omega lost pre-momentum production");
        const double value = policy.omegaProduction(c, cell);
        require(value == c.fields.Pk[cell] / std::max(c.fields.nuT[cell], 1e-30),
                "omega units/formula changed");
        return value;
    }
};
int main() {
    auto mesh = Mesh::makeChannel2D(3, 4, 1, 1);
    mesh.computeWallDistance();
    FlowFields f(mesh);
    ScalarField strain(mesh, "|S|");
    SSTCoefficients coefficients;
    for (int i = 0; i < mesh.nCells(); ++i) {
        f.U[i] = velocity(mesh.cell(i).center);
        f.k[i] = .2;
        f.omega[i] = 100;
        f.nuT[i] = .03;
    }
    for (int fi = mesh.nInternalFaces(); fi < mesh.nFaces(); ++fi)
        f.U.bface(fi) = velocity(mesh.face(fi).center);
    std::vector<double> target(6 * mesh.nCells());
    const std::array<double, 6> b{.1, -.04, -.06, .02, .03, -.01};
    for (int i = 0; i < mesh.nCells(); ++i)
        for (int j = 0; j < 6; ++j)
            target[6 * i + j] = b[j];
    HillsTargetProduction policy;
    std::vector<double> output;
    policy.kProduction({mesh, f, coefficients, strain, target, .01}, output);
    const double gradient[3][3] = {{-2, 3, 0}, {5, -4, 0}, {7, 11, 0}};
    const double tensor[3][3] = {
        {b[0] + 1. / 3, b[3], b[4]}, {b[3], b[1] + 1. / 3, b[5]}, {b[4], b[5], b[2] + 1. / 3}};
    double contraction = 0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            contraction += tensor[i][j] * gradient[i][j];
    const double raw = -2 * .2 * contraction;
    require(raw > 0, "synthetic positive branch missing");
    for (double value : output)
        close(value, raw);
    f.Pk.data() = output;
    for (int i = 0; i < mesh.nCells(); ++i)
        close(policy.omegaProduction({mesh, f, coefficients, strain, target, .01}, i), raw / .03);
    f.omega.setUniform(.1);
    policy.kProduction({mesh, f, coefficients, strain, target, .01}, output);
    for (double value : output)
        close(value, 10 * coefficients.betaStar * .2 * .1);
    for (int i = 0; i < mesh.nCells(); ++i)
        f.U[i] = f.U[i] * -1;
    for (int fi = mesh.nInternalFaces(); fi < mesh.nFaces(); ++fi)
        f.U.bface(fi) = f.U.bface(fi) * -1;
    policy.kProduction({mesh, f, coefficients, strain, target, .01}, output);
    for (double value : output)
        require(value == 0, "negative production was not clipped");
    f.k.setUniform(-1);
    policy.kProduction({mesh, f, coefficients, strain, target, .01}, output);
    for (double value : output)
        require(value == 0, "negative k safeguard changed");

    auto bc = FlowBoundaryConditions::channelDefaults(mesh, 1, .02, 3);
    SolverSettings settings;
    settings.verbose = false;
    settings.maxIterations = 3;
    settings.turbStartIter = 0;
    settings.turbUpdateInterval = 1;
    SIMPLESolver solver(mesh, SSTModel{}, bc, .01, settings);
    solver.initUniform(f, {1, .02, 0}, 0, .02, 3);
    auto stage = std::make_shared<StagePolicy>();
    solver.setTargetAnisotropy(&target);
    solver.setProductionPolicy(stage);
    int observations = 0;
    solver.setSourceObserver([&](const SIMPLESolver::SourceSnapshotView &view) {
        ++observations;
        require(view.stage == SIMPLESolver::SourceStage::PreMomentumX, "wrong source stage");
        require(view.fields.Pk.data() == stage->lastProduction,
                "observer lost pre-momentum production");
        const auto before = view.fields.Pk.data(), flux = view.faceFluxes;
        auto diagnostic = solver.diagnoseMomentum(view.fields, coefficients, flux,
                                                  SIMPLESolver::ClosureMode::Stored);
        require(diagnostic.fields.Pk.data() == before && view.fields.Pk.data() == before,
                "saved-flux diagnostic changed live closure");
        require(solver.faceFluxes() == flux, "diagnostic changed live flux");
        for (size_t i = 0; i < view.freshIntegratedForce.size(); ++i)
            require((view.freshIntegratedForce[i] - diagnostic.freshIntegratedForce[i]).norm2() ==
                        0,
                    "saved-flux force lost integrated-source units");
    });
    (void)solver.solve(f);
    require(stage->kCalls == 3 && stage->omegaCalls > 0 && observations == 3,
            "production callbacks missing");
    solver.setSourceObserver({});
    solver.setInjectionScale(0);
    const int previousK = stage->kCalls, previousOmega = stage->omegaCalls;
    solver.initUniform(f, {1, .02, 0}, 0, .02, 3);
    const auto zeroHistory = solver.solve(f);
    require(stage->kCalls == previousK && stage->omegaCalls == previousOmega,
            "zero scale did not bypass policy");
    SIMPLESolver baseline(mesh, SSTModel{}, bc, .01, settings);
    FlowFields base(mesh);
    baseline.initUniform(base, {1, .02, 0}, 0, .02, 3);
    const auto baseHistory = baseline.solve(base);
    require(zeroHistory.finalIter == baseHistory.finalIter &&
                solver.faceFluxes() == baseline.faceFluxes(),
            "zero control flux/history changed");
    require(f.p.data() == base.p.data() && f.k.data() == base.k.data() &&
                f.omega.data() == base.omega.data(),
            "zero control primary fields changed");
    for (int i = 0; i < mesh.nCells(); ++i)
        require((f.U[i] - base.U[i]).norm2() == 0, "zero control velocity changed");
    std::cout << "library=" << cfdr::sourceDigest << " policy=" << HillsTargetProduction::identity
              << " source_patch=" << HillsTargetProduction::sourcePatchSha256
              << " mesh=3x4 data=synthetic seed=none provisional=adapter-contract-check\n";
}
