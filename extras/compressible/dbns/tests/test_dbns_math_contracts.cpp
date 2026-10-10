#include "DBNSSolver.hpp"
#include "HLLCFlux.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace dbns;
static void require(bool condition, const char *message) {
    if (!condition)
        throw std::runtime_error(message);
}
static bool close(double a, double b, double tol = 1e-11) {
    return std::isfinite(a) && std::isfinite(b) &&
           std::abs(a - b) <= tol * (1 + std::max(std::abs(a), std::abs(b)));
}
static DBNSBoundaryConditions walls(const Mesh &mesh, BoundaryKind kind) {
    DBNSBoundaryConditions bc;
    for (const auto &patch : mesh.patches()) {
        BoundarySpec b;
        b.kind = kind;
        bc.set(patch.name, b);
    }
    return bc;
}
static void capped() {
    const auto mesh = Mesh::makeChannel2D(2, 2, 1, 1);
    const auto bc = walls(mesh, BoundaryKind::Extrapolate);
    DBNSSettings s;
    s.viscous = false;
    s.timeMode = TimeMode::Unsteady;
    s.tEnd = 1;
    s.cfl = .1;
    for (int cap : {0, 1}) {
        s.maxIterations = cap;
        DBNSSolver solver(mesh, IdealGasEOS{}, SSTCoefficients{}, bc, s);
        solver.initUniform({1, 1, 0, 1, 0, 0});
        const auto report = solver.solve();
        require(report.tFinal < s.tEnd, "cap fixture unexpectedly reached endpoint");
        require(report.status == EvaluationStatus::Unconverged,
                "capped time march reported convergence");
    }
}
static void steady() {
    const auto mesh = Mesh::makePeriodicChannel2D(3, 4, 1, .5, 20);
    auto bc = walls(mesh, BoundaryKind::NoSlipAdiabatic);
    bc.specs.at("top").wallVelocity.x = 1;
    DBNSSettings s;
    s.constMu = .1;
    s.rkStages = 1;
    s.maxIterations = 1;
    s.cfl = .01;
    DBNSSolver solver(mesh, IdealGasEOS{}, SSTCoefficients{}, bc, s);
    solver.initUniform({1, 0, 0, 1, 0, 0});
    const auto report = solver.solve();
    require(report.status == EvaluationStatus::Unconverged,
            "zero density residual hid moving momentum");
    require(report.finalResidual > s.convergenceTol,
            "joint residual did not expose moving momentum");
    bc.specs.at("top").wallVelocity.x = 0;
    DBNSSolver equilibrium(mesh, IdealGasEOS{}, SSTCoefficients{}, bc, s);
    equilibrium.initUniform({1, 0, 0, 1, 0, 0});
    const auto balanced = equilibrium.solve();
    require(balanced.status == EvaluationStatus::Converged &&
                balanced.finalResidual < s.convergenceTol,
            "exact steady state was not accepted");

    // A one-cell SST box exercises the explicit wall-constraint stopping
    // equation independently of the long implicit boundary-layer regression.
    // Its wall omega is prescribed; the transport residual for that row is
    // not a residual of the constrained equation and must not gate stopping.
    const auto box = Mesh::makeChannel2D(1, 1, 1, 1);
    const auto wall = walls(box, BoundaryKind::NoSlipAdiabatic);
    DBNSSettings t;
    t.turbulent = true;
    t.constMu = 1e-6;
    t.omegaFloor = 1e-9;
    t.maxIterations = 1;
    t.convergenceTol = 5e-4;
    t.cfl = .1;
    const Primitive initial{1, 0, 0, 1, t.kFloor, t.omegaFloor};
    const SSTCoefficients coefficients;
    const double wallOmega = 60.0 * t.constMu / (coefficients.beta1 * .5 * .5);
    const double expectedMeasure =
        (8.0 * t.constMu + coefficients.betaStar * wallOmega) / std::sqrt(1.4);
    DBNSSolver pinned(box, IdealGasEOS{}, coefficients, wall, t);
    pinned.initUniform(initial);
    const auto stopped = pinned.solve();
    require(stopped.status == EvaluationStatus::Converged && stopped.iterations == 1,
            "explicit SST wall constraint did not admit the declared stopping tolerance");
    require(close(pinned.primitive(0).omega, wallOmega), "explicit SST omega pin is incorrect");
    require(close(stopped.finalResidual, expectedMeasure, 1e-10),
            "explicit SST wall stopping measure differs from the independent k balance");
    require(pinned.nCells() == 1 && pinned.limiterActive().size() == 1 &&
                pinned.limiterActive()[0] == 0,
            "zero-strain SST box reported production-limiter activation");

    DBNSSolver repeated(box, IdealGasEOS{}, coefficients, wall, t);
    repeated.initUniform(initial);
    const auto repeat = repeated.solve();
    require(repeat.status == stopped.status && repeat.iterations == stopped.iterations &&
                repeat.finalResidual == stopped.finalResidual &&
                repeated.conserved(0) == pinned.conserved(0),
            "identical explicit SST wall runs differ");

    t.convergenceTol = 0.0;
    DBNSSolver cappedWall(box, IdealGasEOS{}, coefficients, wall, t);
    cappedWall.initUniform(initial);
    const auto cappedReport = cappedWall.solve();
    require(cappedReport.status == EvaluationStatus::Unconverged && cappedReport.iterations == 1,
            "capped explicit SST wall solve reported convergence");
    t.divergenceLimit = 0.0;
    DBNSSolver divergent(box, IdealGasEOS{}, coefficients, wall, t);
    divergent.initUniform(initial);
    const auto diverged = divergent.solve();
    require(diverged.status == EvaluationStatus::Diverged && diverged.iterations == 1 &&
                std::isfinite(diverged.finalResidual) && diverged.finalResidual > 0.0,
            "explicit SST wall divergence threshold did not report a completed divergent step");
}
static void endpoint() {
    const auto mesh = Mesh::makeChannel2D(1, 1, 1, 1);
    const auto bc = walls(mesh, BoundaryKind::Extrapolate);
    for (double end : {0., 1e-6}) {
        DBNSSettings s;
        s.viscous = false;
        s.timeMode = TimeMode::Unsteady;
        s.tEnd = end;
        s.maxIterations = 1;
        DBNSSolver solver(mesh, IdealGasEOS{}, SSTCoefficients{}, bc, s);
        solver.initUniform({1, 0, 0, 1, 0, 0});
        const auto report = solver.solve();
        require(report.status == EvaluationStatus::Converged && close(report.tFinal, end),
                "endpoint status mismatch");
        require(report.iterations == (end == 0 ? 0 : 1), "iterations must count completed steps");
    }
}
static void nonfinite() {
    const auto mesh = Mesh::makeChannel2D(1, 1, 1, 1);
    const auto bc = walls(mesh, BoundaryKind::Extrapolate);
    DBNSSettings s;
    s.viscous = false;
    s.timeMode = TimeMode::Unsteady;
    s.tEnd = 1e-6;
    s.maxIterations = 1;
    DBNSSolver solver(mesh, IdealGasEOS{}, SSTCoefficients{}, bc, s);
    solver.initUniform({1, 0, 0, 1, 0, 0});
    std::vector<StateVec> source(1);
    source[0][I_RHOE] = std::numeric_limits<double>::quiet_NaN();
    solver.setManufacturedSource(source);
    require(solver.solve().status == EvaluationStatus::Diverged,
            "positivity repair concealed nonfinite energy");
}
static void wallWork(bool viscous) {
    const auto mesh = Mesh::makeChannel2D(1, 1, 1, 1);
    auto bc = walls(mesh, BoundaryKind::NoSlipAdiabatic);
    bc.specs.at("top").wallVelocity.x = 1;
    DBNSSettings s;
    s.viscous = viscous;
    s.constMu = .1;
    s.rkStages = 1;
    s.timeMode = TimeMode::Unsteady;
    s.tEnd = 1e-6;
    s.maxIterations = 5;
    s.cfl = .1;
    DBNSSolver solver(mesh, IdealGasEOS{}, SSTCoefficients{}, bc, s);
    solver.initUniform({1, 0, 0, 1, 0, 0});
    const auto before = solver.conserved(0);
    const auto report = solver.solve();
    require(report.status == EvaluationStatus::Converged, "wall test did not reach endpoint");
    if (viscous) {
        const double work = s.constMu / .5 * s.tEnd; // integral tau.n dot Uwall, area=volume=1
        std::cout << "wall energy change " << solver.conserved(0)[I_RHOE] - before[I_RHOE]
                  << " expected " << work << '\n';
        require(close(solver.conserved(0)[I_RHOE] - before[I_RHOE], work, 1e-13),
                "moving-wall stress work missing");
    } else
        for (int v = 0; v < NVAR; ++v)
            require(close(solver.conserved(0)[v], before[v], 1e-13),
                    "viscous wall flux survived viscous=false");
}
static void passiveEnergy() {
    IdealGasEOS eos;
    Primitive l{1, .2, .1, 1, 0, 2}, r{.125, -.1, .03, .1, 0, 2};
    const auto baseline = HLLCFlux::flux(l, r, .8, .6, eos);
    const double k = 3.7;
    l.k = r.k = k;
    const auto shifted = HLLCFlux::flux(l, r, .8, .6, eos);
    for (int v : {I_RHO, I_RHOU, I_RHOV}) {
        std::cout << "passive-k flux row " << v << " " << baseline[v] << " " << shifted[v] << '\n';
        require(close(baseline[v], shifted[v]), "passive TKE changed acoustic flux");
    }
    require(close(shifted[I_RHOE], baseline[I_RHOE] + k * baseline[I_RHO]),
            "passive energy transformation failed");
    require(close(shifted[I_RHOK], k * baseline[I_RHO]), "passive contact transport failed");
}
template <class Action> static void rejects(Action action, const char *message) {
    try {
        action();
    } catch (const std::invalid_argument &error) {
        require(std::string(error.what()).size() > 0, "input rejection has no explanation");
        return;
    }
    throw std::runtime_error(message);
}
// Total energy includes rho*k, and a laminar solve carries no turbulence
// state. A laminar k admitted at any entry point would turn into heat when
// the solver clears it, so every caller-provided laminar state and source
// must have zero k and omega. A turbulent solve still accepts both.
static void laminarEnergy() {
    const auto mesh = Mesh::makeChannel2D(3, 3, 1, 1);
    const auto bc = walls(mesh, BoundaryKind::Extrapolate);
    DBNSSettings s;
    s.viscous = false;
    s.timeMode = TimeMode::Unsteady;
    s.rkStages = 1;
    s.tEnd = 1e-3;
    s.maxIterations = 1;
    const Primitive rest{1, 0, 0, 1, 0, 0};
    Primitive withK = rest, withOmega = rest;
    withK.k = .5;
    withOmega.omega = 2;

    DBNSSolver control(mesh, IdealGasEOS{}, SSTCoefficients{}, bc, s);
    control.initUniform(rest);
    control.solve();
    for (int ci = 0; ci < mesh.nCells(); ++ci)
        require(close(control.primitive(ci).p, 1), "quiescent laminar step changed pressure");

    DBNSSolver solver(mesh, IdealGasEOS{}, SSTCoefficients{}, bc, s);
    solver.initUniform(rest);
    const auto before = solver.conserved(4);
    bool admitted = true;
    try {
        solver.initUniform(withK);
    } catch (const std::invalid_argument &) {
        admitted = false;
    }
    if (admitted) {
        solver.solve();
        std::cout << "laminar k=0.5 admitted; pressure after one step " << solver.primitive(4).p
                  << '\n';
        throw std::runtime_error("laminar DBNS admitted turbulent kinetic energy");
    }
    require(solver.conserved(4) == before, "rejected laminar state changed the old solution");
    rejects([&] { solver.initUniform(withOmega); }, "laminar DBNS admitted omega");
    std::vector<Primitive> field(mesh.nCells(), rest);
    field.back() = withK;
    rejects([&] { solver.initField(field); }, "laminar DBNS field admitted k");
    field.back() = withOmega;
    rejects([&] { solver.initField(field); }, "laminar DBNS field admitted omega");

    const int nBoundary = mesh.nFaces() - mesh.nInternalFaces();
    std::vector<Primitive> ghost(nBoundary, rest);
    solver.setBoundaryOverride(ghost);
    ghost.back() = withK;
    rejects([&] { solver.setBoundaryOverride(ghost); }, "laminar ghost override admitted k");
    ghost.back() = Primitive{-1, 0, 0, 1, 0, 0};
    rejects([&] { solver.setBoundaryOverride(ghost); }, "ghost override admitted negative density");
    solver.setBoundaryOverride({});

    std::vector<StateVec> source(mesh.nCells());
    solver.setManufacturedSource(source);
    for (int row : {I_RHOK, I_RHOW}) {
        source.back()[row] = 1;
        rejects([&] { solver.setManufacturedSource(source); },
                "laminar manufactured source admitted a turbulence row");
        source.back()[row] = 0;
    }
    solver.setManufacturedSource({});

    const auto &patch = mesh.patch(0);
    for (auto kind : {BoundaryKind::FixedState, BoundaryKind::SubsonicInflow}) {
        auto inflow = bc;
        inflow.specs.at(patch.name).kind = kind;
        inflow.specs.at(patch.name).freestream = withK;
        rejects([&] { DBNSSolver rejected(mesh, IdealGasEOS{}, SSTCoefficients{}, inflow, s); },
                "laminar boundary freestream admitted k");
    }
    auto profiled = bc;
    profiled.specs.at(patch.name).kind = BoundaryKind::FixedState;
    profiled.specs.at(patch.name).profile.assign(patch.faces.size(), rest);
    profiled.specs.at(patch.name).profile.back() = withOmega;
    rejects([&] { DBNSSolver rejected(mesh, IdealGasEOS{}, SSTCoefficients{}, profiled, s); },
            "laminar boundary profile admitted omega");

    auto turbulent = s;
    turbulent.turbulent = true;
    auto turbulentInflow = bc;
    turbulentInflow.specs.at(patch.name).kind = BoundaryKind::FixedState;
    turbulentInflow.specs.at(patch.name).freestream = withK;
    DBNSSolver accepted(mesh, IdealGasEOS{}, SSTCoefficients{}, turbulentInflow, turbulent);
    accepted.initUniform(withK);
    require(accepted.primitive(4).k == .5, "turbulent DBNS changed an admitted k");
}
static void rkProperties() {
    const auto mesh = Mesh::makeChannel2D(1, 1, 1, 1);
    auto bc = walls(mesh, BoundaryKind::NoSlipIsothermal);
    for (auto &entry : bc.specs)
        entry.second.wallTemp = 2;
    IdealGasEOS eos;
    eos.R = 1;
    eos.mu_ref = 1;
    eos.T_ref = 1;
    eos.S_suth = .1;
    const double end = .1;
    // Independent method-of-lines ODE for a uniform one-cell heated box.
    auto rhs = [&](double t) { return 8 * eos.gamma / eos.Pr * eos.viscosity(t) * (2 - t); };
    double reference = 1;
    const int n = 20000;
    const double dt = end / n;
    for (int i = 0; i < n; ++i) {
        const double a = rhs(reference), b = rhs(reference + .5 * dt * a),
                     c = rhs(reference + .5 * dt * b), d = rhs(reference + dt * c);
        reference += dt * (a + 2 * b + 2 * c + d) / 6;
    }
    double errors[3];
    int index = 0;
    for (double cfl : {.2, .1, .05}) {
        DBNSSettings s;
        s.timeMode = TimeMode::Unsteady;
        s.tEnd = end;
        s.cfl = cfl;
        s.maxIterations = 10000;
        DBNSSolver solver(mesh, eos, SSTCoefficients{}, bc, s);
        solver.initUniform({1, 0, 0, 1, 0, 0});
        const auto report = solver.solve();
        require(report.status == EvaluationStatus::Converged && close(report.tFinal, end),
                "RK test endpoint failed");
        errors[index++] = std::abs(GasState::temperature(solver.primitive(0), eos) - reference);
    }
    const double order = std::log(errors[0] / errors[2]) / std::log(4.);
    std::cout << "RK property errors " << errors[0] << " " << errors[1] << " " << errors[2]
              << " order " << order << '\n';
    require(errors[0] > errors[1] && errors[1] > errors[2] && order > 2.7,
            "stage-frozen viscosity destroys third-order time accuracy");
}
static void periodic() {
    const int nx = 6;
    const auto mesh = Mesh::makePeriodicChannel2D(nx, 4, 1, .5, 20);
    const auto bc = walls(mesh, BoundaryKind::SlipWall);
    for (bool viscous : {false, true})
        for (bool limited : {false, true}) {
            DBNSSettings s;
            s.viscous = viscous;
            s.constMu = .01;
            s.limitReconstruction = false;
            s.limitReconstruction = limited;
            s.timeMode = TimeMode::Unsteady;
            s.tEnd = .001;
            s.cfl = .1;
            s.maxIterations = 100;
            DBNSSolver first(mesh, IdealGasEOS{}, SSTCoefficients{}, bc, s),
                shifted(mesh, IdealGasEOS{}, SSTCoefficients{}, bc, s);
            std::vector<Primitive> a(mesh.nCells()), b(mesh.nCells());
            for (int ci = 0; ci < mesh.nCells(); ++ci) {
                const double x = mesh.cell(ci).center.x;
                a[ci] = {1 + .01 * std::sin(2 * M_PI * x), .1, 0, 1, 0, 0};
                b[ci] = {1 + .01 * std::sin(2 * M_PI * (x - 1. / nx)), .1, 0, 1, 0, 0};
            }
            first.initField(a);
            shifted.initField(b);
            const auto firstReport = first.solve(), shiftedReport = shifted.solve();
            for (const auto &r : {firstReport, shiftedReport})
                require(r.status == EvaluationStatus::Converged && close(r.tFinal, s.tEnd),
                        "periodic comparison did not reach its physical endpoint");
            for (int ci = 0; ci < mesh.nCells(); ++ci) {
                const auto c = mesh.cell(ci).center;
                double x = c.x + 1. / nx;
                if (x > 1)
                    x -= 1;
                int match = -1;
                for (int j = 0; j < mesh.nCells(); ++j)
                    if (close(mesh.cell(j).center.x, x) && close(mesh.cell(j).center.y, c.y))
                        match = j;
                require(match >= 0, "missing periodic comparison cell");
                for (int v = 0; v < NVAR; ++v)
                    require(close(first.conserved(ci)[v], shifted.conserved(match)[v]),
                            "periodic flux depends on seam position");
            }
        }
}
int main(int argc, char **argv) {
    try {
        require(argc == 2, "expected one case name");
        const std::string name = argv[1];
        if (name == "capped")
            capped();
        else if (name == "steady")
            steady();
        else if (name == "wall_work")
            wallWork(true);
        else if (name == "inviscid_wall")
            wallWork(false);
        else if (name == "passive_energy")
            passiveEnergy();
        else if (name == "rk_properties")
            rkProperties();
        else if (name == "periodic")
            periodic();
        else if (name == "endpoint")
            endpoint();
        else if (name == "nonfinite")
            nonfinite();
        else if (name == "laminar_energy")
            laminarEnergy();
        else
            throw std::runtime_error("unknown case");
        std::cout << name << " passed\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
