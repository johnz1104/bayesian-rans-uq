#include "SIMPLESolver.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>

namespace {
void require(bool ok, const char *message) {
    if (!ok)
        throw std::runtime_error(message);
}
template <class Fn> void rejects(Fn fn) {
    bool caught = false;
    try {
        fn();
    } catch (const std::exception &) {
        caught = true;
    }
    require(caught, "invalid research interface input was accepted");
}
double distance(const Vec3 &a, const Vec3 &b) {
    return (a - b).norm();
}
bool sameNumber(double a, double b) {
    return a == b || (std::isnan(a) && std::isnan(b));
}
void sameHistory(const ConvergenceHistory &a, const ConvergenceHistory &b) {
    require(a.converged == b.converged && a.diverged == b.diverged && a.stalled == b.stalled &&
                a.finalIter == b.finalIter && a.entries.size() == b.entries.size() &&
                sameNumber(a.finalWorstMeasure, b.finalWorstMeasure) &&
                sameNumber(a.contractionRateU, b.contractionRateU),
            "solve status changed");
    for (size_t i = 0; i < a.entries.size(); ++i) {
        const auto &x = a.entries[i];
        const auto &y = b.entries[i];
        require(x.iteration == y.iteration, "iteration index changed");
        const double xx[] = {
            x.Ux,      x.Uy,        x.Uz, x.p, x.k, x.omega, x.momentumScaled, x.continuityScaled,
            x.changeU, x.remainingU};
        const double yy[] = {
            y.Ux,      y.Uy,        y.Uz, y.p, y.k, y.omega, y.momentumScaled, y.continuityScaled,
            y.changeU, y.remainingU};
        for (int j = 0; j < 10; ++j)
            require(sameNumber(xx[j], yy[j]), "convergence history changed");
    }
}
void sameFields(const FlowFields &a, const FlowFields &b) {
    const ScalarField *aa[] = {&a.p, &a.k, &a.omega, &a.nuT, &a.F1, &a.F2, &a.Pk, &a.CDkw};
    const ScalarField *bb[] = {&b.p, &b.k, &b.omega, &b.nuT, &b.F1, &b.F2, &b.Pk, &b.CDkw};
    for (int i = 0; i < a.U.size(); ++i)
        require(distance(a.U[i], b.U[i]) == 0., "cell U changed");
    for (int i = 0; i < a.U.nBnd(); ++i)
        require(distance(a.U.bnd(i), b.U.bnd(i)) == 0., "boundary U changed");
    for (int f = 0; f < 8; ++f) {
        require(aa[f]->data() == bb[f]->data(), "cell scalar changed");
        for (int i = 0; i < aa[f]->nBnd(); ++i)
            require(aa[f]->bnd(i) == bb[f]->bnd(i), "boundary scalar changed");
    }
    require(a.turbEstablished == b.turbEstablished, "startup state changed");
}
struct ProbePolicy final : SIMPLESolver::ProductionPolicy {
    mutable int kCalls = 0, omegaCalls = 0;
    mutable std::vector<double> preK, preStrain;
    mutable std::vector<Vec3> preU;
    mutable bool sawUpdatedK = false, sawUpdatedU = false;
    void kProduction(const SIMPLESolver::ProductionContext &c,
                     std::vector<double> &out) const override {
        require(c.strainMagnitude.name == "|S|", "production strain field identity changed");
        ++kCalls;
        preK = c.fields.k.data();
        preStrain = c.strainMagnitude.data();
        preU = c.fields.U.data();
        for (int i = 0; i < c.mesh.nCells(); ++i)
            out[i] = 2. * preK[i];
    }
    double omegaProduction(const SIMPLESolver::ProductionContext &c, int i) const override {
        ++omegaCalls;
        require(c.fields.Pk[i] == 2. * preK[i], "production was recomputed after momentum/k");
        require(c.strainMagnitude[i] == preStrain[i], "omega production lost pre-momentum strain");
        sawUpdatedK = sawUpdatedK || c.fields.k[i] != preK[i];
        sawUpdatedU = sawUpdatedU || distance(c.fields.U[i], preU[i]) != 0.;
        return c.fields.Pk[i] / std::max(c.fields.nuT[i], 1e-30);
    }
};
struct InvalidPolicy final : SIMPLESolver::ProductionPolicy {
    void kProduction(const SIMPLESolver::ProductionContext &,
                     std::vector<double> &out) const override {
        out.clear();
    }
    double omegaProduction(const SIMPLESolver::ProductionContext &, int) const override {
        return 0.;
    }
};
} // namespace

int main() {
    try {
        Mesh mesh = Mesh::makeChannel2D(8, 10, 2., 1.);
        mesh.computeWallDistance();
        SSTCoefficients theta;
        SSTModel sst(theta);
        auto bc = FlowBoundaryConditions::channelDefaults(mesh, 1., .001, 50.);
        SolverSettings st;
        st.verbose = false;
        st.maxIterations = 6;
        st.turbStartIter = 0;
        st.turbUpdateInterval = 2;
        st.convergenceTol = 1e-12;
        SIMPLESolver solver(mesh, sst, bc, .001, st);
        FlowFields initial(mesh);
        solver.initUniform(initial, Vec3(.7, 0., 0.), 0., .001, 50.);
        for (int i = 0; i < mesh.nCells(); ++i)
            initial.k[i] = .001 + .002 * mesh.cell(i).center.y;
        const auto saved = initial;
        std::vector<double> flux(mesh.nFaces(), 0.);
        std::vector<double> target(6 * mesh.nCells(), 0.);
        for (int i = 0; i < mesh.nCells(); ++i) {
            target[6 * i] = .1;
            target[6 * i + 1] = -.05;
            target[6 * i + 2] = -.05;
            target[6 * i + 3] = .03;
            target[6 * i + 4] = .02;
            target[6 * i + 5] = .02;
        }
        using Mode = SIMPLESolver::ClosureMode;
        auto base = solver.diagnoseMomentum(initial, theta, flux, Mode::Stored);
        solver.setTargetAnisotropy(&target);
        solver.setInjectionScale(0.);
        auto zero = solver.diagnoseMomentum(initial, theta, flux, Mode::Stored);
        solver.setInjectionScale(1.);
        auto full = solver.diagnoseMomentum(initial, theta, flux, Mode::Stored);
        solver.setInjectionScale(.37);
        auto part = solver.diagnoseMomentum(initial, theta, flux, Mode::Stored);
        // The analytic coefficient assembly omits the derivative of injected
        // Boussinesq cancellation, even with default production.
        rejects([&] { solver.assembleResidualSensitivity(initial, theta); });
        double signal = 0., zSignal = 0.;
        for (int i = 0; i < mesh.nCells(); ++i) {
            require(distance(base.residual[i], zero.residual[i]) == 0., "zero changed residual");
            const auto delta = full.residual[i] - base.residual[i];
            require(distance(delta, full.freshIntegratedForce[i]) < 1e-13,
                    "force/residual units or sign");
            require(distance(part.residual[i] - base.residual[i], delta * .37) < 1e-13,
                    "partial force scale");
            require(distance(part.freshIntegratedForce[i], full.freshIntegratedForce[i] * .37) <
                        1e-14,
                    "whole source was not scaled");
            signal = std::max(signal, delta.norm());
            zSignal = std::max(zSignal, std::abs(delta.z));
        }
        require(signal > 1e-7 && zSignal > 1e-7,
                "test injection did not exercise three components");
        sameFields(saved, initial);
        auto fresh = solver.diagnoseMomentum(initial, theta, flux, Mode::Recomputed);
        require(fresh.fields.Pk.data() != initial.Pk.data(), "fresh closure not recomputed");
        sameFields(saved, initial);
        rejects([&] { solver.setInjectionScale(-1.); });
        rejects([&] { solver.setInjectionScale(std::numeric_limits<double>::quiet_NaN()); });
        rejects([&] { solver.diagnoseMomentum(initial, theta, {}, Mode::Stored); });
        auto badFlux = flux;
        badFlux[0] = std::numeric_limits<double>::infinity();
        rejects([&] { solver.diagnoseMomentum(initial, theta, badFlux, Mode::Stored); });
        auto badFields = initial;
        badFields.p.data().pop_back();
        rejects([&] { solver.diagnoseMomentum(badFields, theta, flux, Mode::Stored); });
        auto badTarget = target;
        badTarget.pop_back();
        rejects([&] { solver.setTargetAnisotropy(&badTarget); });
        badTarget = target;
        badTarget[0] = std::numeric_limits<double>::quiet_NaN();
        rejects([&] { solver.setTargetAnisotropy(&badTarget); });
        rejects([&] { solver.setProductionPolicy(std::make_shared<ProbePolicy>()); });

        solver.setInjectionScale(1.);
        solver.setProductionPolicy(std::make_shared<InvalidPolicy>());
        rejects([&] { solver.diagnoseMomentum(initial, theta, flux, Mode::Recomputed); });
        sameFields(saved, initial);
        auto policy = std::make_shared<ProbePolicy>();
        solver.setProductionPolicy(policy);
        rejects([&] { solver.setInjectionScale(.37); });
        rejects([&] { solver.assembleResidualSensitivity(initial, theta); });
        solver.setInjectionScale(0.);
        auto plain = initial, controlled = initial;
        SIMPLESolver reference(mesh, sst, bc, .001, st);
        const auto plainHist = reference.solve(plain);
        const auto zeroHist = solver.solve(controlled);
        sameFields(plain, controlled);
        require(reference.faceFluxes() == solver.faceFluxes(), "zero changed corrected flux");
        sameHistory(plainHist, zeroHist);
        require(policy->kCalls == 0 && policy->omegaCalls == 0, "zero failed to bypass policy");

        solver.setInjectionScale(1.);
        solver.setProductionPolicy(nullptr);
        auto unobserved = initial, observed = initial;
        const auto unobservedHist = solver.solve(unobserved);
        const auto unobservedFlux = solver.faceFluxes();
        int observations = 0;
        std::vector<Vec3> previousBlend;
        solver.setSourceObserver([&](const SIMPLESolver::SourceSnapshotView &v) {
            require(v.iteration == observations++, "source observer stage/order");
            require(v.faceFluxes.size() == flux.size(), "missing convecting flux");
            for (int i = 0; i < mesh.nCells(); ++i) {
                const auto expected = previousBlend.empty()
                                          ? v.freshIntegratedForce[i]
                                          : previousBlend[i] * (1. - st.alphaInjection) +
                                                v.freshIntegratedForce[i] * st.alphaInjection;
                require(distance(expected, v.appliedIntegratedForce[i]) < 1e-15,
                        "source-memory staging");
            }
            previousBlend = v.appliedIntegratedForce;
            const auto before = solver.faceFluxes();
            const auto diag = solver.diagnoseMomentum(v.fields, theta, v.faceFluxes, Mode::Stored);
            require(before == solver.faceFluxes(), "diagnostics mutated active flux");
            for (int i = 0; i < mesh.nCells(); ++i)
                require(distance(diag.freshIntegratedForce[i], v.freshIntegratedForce[i]) < 1e-14,
                        "observer is not pre-x source state");
        });
        const auto observedHist = solver.solve(observed);
        require(observations == static_cast<int>(observedHist.entries.size()),
                "observer missing outer iterations");
        sameFields(unobserved, observed);
        require(unobservedFlux == solver.faceFluxes(), "observer/diagnostics changed active solve");
        sameHistory(unobservedHist, observedHist);
        solver.diagnoseMomentum(observed, theta, solver.faceFluxes(), Mode::Recomputed);
        require(observations == static_cast<int>(observedHist.entries.size()),
                "diagnostics emitted solve observations");

        solver.setSourceObserver({});
        solver.setProductionPolicy(policy);
        auto custom = initial;
        solver.solve(custom);
        require(policy->kCalls == 3 && policy->omegaCalls == 3 * mesh.nCells(),
                "production cadence changed");
        require(policy->sawUpdatedK, "test missed sequential k/omega production staging");
        require(policy->sawUpdatedU, "test missed velocity update after frozen strain");
        std::weak_ptr<const SIMPLESolver::ProductionPolicy> weak = policy;
        policy.reset();
        require(!weak.expired(), "production policy not owned");
        solver.setProductionPolicy(nullptr);
        require(weak.expired(), "policy release failed");

        // Synthetic periodic hills, not a DNS or converged research result.
        // The full nonzero target is present while scale zero bypasses BOTH
        // source and custom production. Compare every saved numerical field.
        const int nx = 12, ny = 8;
        std::vector<double> x(nx + 1), wall(nx + 1);
        for (int i = 0; i <= nx; ++i) {
            x[i] = 4.5 * i / nx;
            wall[i] = .5 * (1. + std::cos(2. * std::acos(-1.) * i / nx));
        }
        auto hills = Mesh::makeCurvedChannelPeriodic2D(x, wall, 3., ny, 0., 1.);
        hills.computeWallDistance();
        auto hbc = FlowBoundaryConditions::channelDefaults(hills, 1., 1e-4, 10.);
        auto hs = st;
        hs.alphaU = .3;
        hs.alphaP = .2;
        hs.alphaK = .4;
        hs.alphaOmega = .4;
        hs.bodyForce = Vec3(.00906, 0., 0.);
        SIMPLESolver hbase(hills, sst, hbc, 1. / 5600., hs), hzero(hills, sst, hbc, 1. / 5600., hs);
        FlowFields hf(hills);
        hbase.initUniform(hf, Vec3{}, 0., 1e-4, 10.);
        auto hz = hf;
        std::vector<double> hb(6 * hills.nCells(), 0.);
        for (int i = 0; i < hills.nCells(); ++i) {
            hb[6 * i] = .1;
            hb[6 * i + 1] = -.05;
            hb[6 * i + 2] = -.05;
            hb[6 * i + 3] = .02;
        }
        auto hp = std::make_shared<ProbePolicy>();
        hzero.setTargetAnisotropy(&hb);
        hzero.setProductionPolicy(hp);
        hzero.setInjectionScale(0.);
        const auto hh = hbase.solve(hf), zh = hzero.solve(hz);
        sameFields(hf, hz);
        sameHistory(hh, zh);
        require(hbase.faceFluxes() == hzero.faceFluxes(), "hills zero changed flux");
        require(hp->kCalls == 0 && hp->omegaCalls == 0, "hills zero called production policy");

        // An interrupted solve must not seed the next invocation's workspace,
        // injection history or observer guard. Use PCG so this control isolates
        // solve-state reset from AMG's separate hierarchy-reuse contract.
        auto resetSettings = st;
        resetSettings.maxIterations = 9;
        resetSettings.turbStartIter = 2;
        resetSettings.turbUpdateInterval = 3;
        resetSettings.nuTFloorIters = 3;
        resetSettings.pressureSolver = "PCG";
        SIMPLESolver reused(mesh, sst, bc, .001, resetSettings);
        SIMPLESolver clean(mesh, sst, bc, .001, resetSettings);
        auto restart = initial;
        for (int i = 0; i < mesh.nCells(); ++i) {
            const auto &c = mesh.cell(i).center;
            restart.U[i] = Vec3(.7 + .03 * c.y, .01 * c.x * c.y, .02 * c.y);
            restart.p[i] = .02 * c.x + .01 * c.y * c.y;
        }
        reused.setTargetAnisotropy(&target);
        auto injected = restart;
        const auto injectedHistory = reused.solve(injected);
        require(!injectedHistory.diverged && injectedHistory.entries.size() == 9,
                "reset fixture did not complete its injected iterations");
        int interruptedCalls = 0;
        reused.setSourceObserver([&](const SIMPLESolver::SourceSnapshotView &v) {
            ++interruptedCalls;
            if (v.iteration == 4)
                throw std::runtime_error("intentional observer interruption");
        });
        auto interrupted = restart;
        rejects([&] { reused.solve(interrupted); });
        require(interruptedCalls == 5, "reset fixture missed the intended interruption");

        reused.setInjectionScale(0.);
        int recoveryCalls = 0;
        reused.setSourceObserver([&](const SIMPLESolver::SourceSnapshotView &v) {
            require(v.iteration == recoveryCalls++, "recovery observer did not restart at zero");
            for (int i = 0; i < mesh.nCells(); ++i)
                require(v.freshIntegratedForce[i].norm2() == 0. &&
                            v.appliedIntegratedForce[i].norm2() == 0.,
                        "recovery inherited a previous injection source");
        });
        auto recovered = restart, expected = restart;
        const auto recoveredHistory = reused.solve(recovered);
        const auto expectedHistory = clean.solve(expected);
        require(recoveryCalls == 9 && recovered.turbEstablished,
                "recovery fixture missed its cadence or startup-floor release");
        sameFields(recovered, expected);
        sameHistory(recoveredHistory, expectedHistory);
        require(reused.faceFluxes() == clean.faceFluxes(), "recovery changed corrected flux");

        reused.setSourceObserver({});
        reused.setTargetAnisotropy(nullptr);
        auto repeated = restart;
        const auto repeatedHistory = reused.solve(repeated);
        sameFields(repeated, expected);
        sameHistory(repeatedHistory, expectedHistory);
        require(reused.faceFluxes() == clean.faceFluxes(), "repeated solve changed corrected flux");
        std::cout << "research interface contracts passed\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
