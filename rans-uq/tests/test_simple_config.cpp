#include "ForwardBatch.hpp"
#include "ParameterSensitivity.hpp"
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
void rejects(const std::function<void()> &call, const char *contract) {
    try {
        call();
    } catch (const std::invalid_argument &error) {
        if (std::string(error.what()).empty())
            throw std::runtime_error("admission error must explain the invalid input");
        return;
    }
    throw std::runtime_error(contract);
}
} // namespace

int main() {
    try {
        Mesh mesh = Mesh::makeChannel2D(3, 3, 1, 1);
        mesh.computeWallDistance();
        const auto bc = FlowBoundaryConditions::channelDefaults(mesh, 1, 0.01, 10);
        SolverSettings settings;
        settings.verbose = false;
        settings.maxIterations = 0;
        const double nan = std::numeric_limits<double>::quiet_NaN();
        const double inf = std::numeric_limits<double>::infinity();
        for (bool omega : {false, true}) {
            auto prescribed = bc;
            auto &patch = omega ? prescribed.omegaBC[0] : prescribed.kBC[0];
            patch.type = BCType::Dirichlet;
            patch.value = -1;
            rejects([&] { SIMPLESolver invalid(mesh, SSTModel{}, prescribed, 0.01, settings); },
                    "prescribed turbulence values must be nonnegative");
            patch.value = 0;
            SIMPLESolver zeroBoundary(mesh, SSTModel{}, prescribed, 0.01, settings);
            patch.value = -1; // overridden storage is not consumed
            patch.scalarProfile.assign(mesh.patch(0).faces.size(), 1);
            SIMPLESolver profileControl(mesh, SSTModel{}, prescribed, 0.01, settings);
            patch.scalarProfile.back() = -1;
            rejects([&] { SIMPLESolver invalid(mesh, SSTModel{}, prescribed, 0.01, settings); },
                    "each prescribed turbulence profile value must be nonnegative");
        }
        for (double nu : {0.0, -1.0, nan, inf})
            rejects([&] { SIMPLESolver solver(mesh, SSTModel{}, bc, nu, settings); },
                    "SIMPLE must reject nonpositive or nonfinite viscosity at construction");
        SIMPLESolver solver(mesh, SSTModel{}, bc, 0.01, settings);
        Mesh other = Mesh::makeChannel2D(3, 3, 2, 1);
        FlowFields foreign(other);
        foreign.U.setUniform({7, 0, 0});
        rejects([&] { solver.initUniform(foreign, {1, 0, 0}, 0, 0.01, 10); },
                "initialization must reject foreign same-size fields before mutation");
        if (foreign.U[0].x != 7)
            throw std::runtime_error("rejected initialization partially changed foreign fields");
        rejects([&] { (void)solver.solve(foreign); },
                "solve must reject foreign field ownership even with a zero iteration cap");
        for (int field = 0; field < 9; ++field) {
            FlowFields shortState(mesh);
            if (field == 0)
                shortState.U.data().resize(1);
            else {
                ScalarField *scalar[] = {&shortState.p,   &shortState.k,   &shortState.omega,
                                         &shortState.nuT, &shortState.F1,  &shortState.F2,
                                         &shortState.Pk,  &shortState.CDkw};
                scalar[field - 1]->data().resize(1);
            }
            rejects([&] { solver.initUniform(shortState, {1, 0, 0}, 0, 0.01, 10); },
                    "initialization must reject any incomplete primary or closure field");
            rejects([&] { (void)solver.solve(shortState); },
                    "solve must reject any incomplete primary or closure field");
        }
        Mesh unprepared = Mesh::makeChannel2D(3, 3, 1, 1);
        const auto unpreparedBC = FlowBoundaryConditions::channelDefaults(unprepared, 1, 0.01, 10);
        SIMPLESolver needsDistance(unprepared, SSTModel{}, unpreparedBC, 0.01, settings);
        FlowFields unchanged(unprepared);
        unchanged.U.setUniform({7, 0, 0});
        bool missingDistance = false;
        try {
            needsDistance.initUniform(unchanged, {1, 0, 0}, 0, 0.01, 10);
        } catch (const std::logic_error &) {
            missingDistance = true;
        }
        if (!missingDistance || unchanged.U[0].x != 7)
            throw std::runtime_error("wall distance must be checked before initialization writes");
        FlowFields fields(mesh);
        solver.initUniform(fields, {1, 0, 0}, 0, 0, 10); // zero k is a valid laminar state
        for (const auto &u : {Vec3{nan, 0, 0}, Vec3{0, inf, 0}, Vec3{0, 0, nan}})
            rejects([&] { solver.initUniform(fields, u, 0, 0.01, 10); },
                    "SIMPLE initial velocity must be finite");
        for (double value : {nan, inf})
            rejects([&] { solver.initUniform(fields, {1, 0, 0}, value, 0.01, 10); },
                    "SIMPLE initial pressure must be finite");
        for (double value : {-1.0, nan, inf})
            rejects([&] { solver.initUniform(fields, {1, 0, 0}, 0, value, 10); },
                    "SIMPLE initial k must be finite and nonnegative");
        for (double value : {0.0, -1.0, nan, inf})
            rejects([&] { solver.initUniform(fields, {1, 0, 0}, 0, 0.01, value); },
                    "SIMPLE initial omega must be finite and positive");
        std::vector<Vec3> source(1, {1, 0, 0});
        rejects([&] { solver.setMomentumSource(&source); },
                "momentum source must contain exactly one vector per cell");
        source.assign(mesh.nCells(), {1, 0, 0});
        solver.setMomentumSource(&source);
        source[0].y = nan;
        rejects([&] { solver.setMomentumSource(&source); },
                "momentum source must contain only finite vector components");
        solver.setMomentumSource(nullptr);
        SolverSettings running = settings;
        running.maxIterations = 2;
        for (bool duringSolve : {false, true}) {
            SIMPLESolver borrowing(mesh, SSTModel{}, bc, 0.01, running);
            FlowFields state(mesh);
            borrowing.initUniform(state, {1, 0, 0}, 0, 0.01, 10);
            std::vector<double> target(static_cast<size_t>(6 * mesh.nCells()), 0);
            borrowing.setTargetAnisotropy(&target);
            if (duringSolve) {
                borrowing.setSourceObserver([&](const SIMPLESolver::SourceSnapshotView &) {
                    target.resize(1);
                    target.shrink_to_fit();
                });
                rejects([&] { (void)borrowing.solve(state); },
                        "each injection use must revalidate borrowed storage after callbacks");
            } else {
                target.resize(1);
                rejects([&] { (void)borrowing.assembleResidual(state, SSTCoefficients{}); },
                        "fixed-state residual must reject shortened borrowed target storage");
                target.assign(static_cast<size_t>(6 * mesh.nCells()), 0);
                borrowing.setTargetAnisotropy(&target);
                target[0] = nan;
                rejects([&] { (void)borrowing.assembleResidual(state, SSTCoefficients{}); },
                        "fixed-state residual must reject mutated nonfinite target values");
            }
        }

        const auto parameters = InferenceParameterSet::a1_betaStar();
        ObservationOperator observations;
        observations.addVelocityProfile({0.5, 0.4, 0}, 0, 0, 1);
        ForwardModel targetModel(mesh, parameters, observations, bc, 0.01, settings);
        std::vector<double> ownedTarget(static_cast<size_t>(6 * mesh.nCells()), 0);
        targetModel.setTargetAnisotropy(ownedTarget);
        for (double value : {nan, inf}) {
            ownedTarget.back() = value;
            rejects([&] { targetModel.setTargetAnisotropy(ownedTarget); },
                    "forward target setter must reject nonfinite values immediately");
        }
        for (int kind : {0, 1, 2}) {
            auto construct = [&](double nu, const Vec3 &u, double p, double k, double omega) {
                if (kind == 2) {
                    ParameterSensitivity model(mesh, observations, bc, nu, settings, u, p, k,
                                               omega);
                } else if (kind == 1) {
                    ForwardBatch model(mesh, parameters, observations, bc, nu, settings, u, p, k,
                                       omega);
                } else {
                    ForwardModel model(mesh, parameters, observations, bc, nu, settings, u, p, k,
                                       omega);
                }
            };
            construct(0.01, {1, 0, 0}, 0, 0, 10);
            for (double nu : {0.0, -1.0, nan, inf})
                rejects([&] { construct(nu, {1, 0, 0}, 0, 0.01, 10); },
                        "forward constructors must reject invalid viscosity before evaluation");
            rejects([&] { construct(0.01, {nan, 0, 0}, 0, 0.01, 10); },
                    "forward constructor must reject nonfinite initial velocity");
            rejects([&] { construct(0.01, {1, 0, 0}, nan, 0.01, 10); },
                    "forward constructor must reject nonfinite initial pressure");
            rejects([&] { construct(0.01, {1, 0, 0}, 0, -1, 10); },
                    "forward constructor must reject negative initial k");
            rejects([&] { construct(0.01, {1, 0, 0}, 0, 0.01, -1); },
                    "forward constructor must reject nonpositive initial omega");
        }
        std::cout << "SIMPLE physical input and momentum source admission passed\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
