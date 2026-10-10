#include "AnisotropyTools.hpp"
#include "RealizabilityProjection.hpp"
#include "SIMPLESolver.hpp"
#include "DBNSSolver.hpp"
#include "InferenceParameters.hpp"
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

static void require(bool ok, const char *message) {
    if (!ok)
        throw std::runtime_error(message);
}

static void configurationContracts() {
    const auto rejects = [](auto action) {
        bool rejected = false;
        try {
            action();
        } catch (const std::invalid_argument &) {
            rejected = true;
        }
        require(rejected, "invalid public configuration was accepted");
    };
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    const std::function<void(const SSTModel &)> consumers[] = {
        [](const auto &m) { (void)m.computeF1(.1, 2., .1, .001, 0.); },
        [](const auto &m) { (void)m.computeF2(.1, 2., .1, .001); },
        [](const auto &m) { (void)m.crossDiffusion({1, 0, 0}, {1, 0, 0}, 2.); },
        [](const auto &m) { (void)m.eddyViscosity(.1, 2., 1., .5); },
        [](const auto &m) { (void)m.production(.1, 1., .1, 2.); },
        [](const auto &m) { (void)m.productionOmega(.1, 1., .1, 2.); },
        [](const auto &m) { (void)m.sourceK(.1, .1, 2.); },
        [](const auto &m) { (void)m.sourceOmega(1., .1, .1, 2., .5, 0.); },
        [](const auto &m) { (void)m.closureSensitivity(.1, 2., 1., .1, .001, 0., 0.); }};
    for (auto member :
         {&SSTCoefficients::sigma_k1, &SSTCoefficients::sigma_w1, &SSTCoefficients::beta1,
          &SSTCoefficients::alpha1, &SSTCoefficients::sigma_k2, &SSTCoefficients::sigma_w2,
          &SSTCoefficients::beta2, &SSTCoefficients::alpha2, &SSTCoefficients::betaStar,
          &SSTCoefficients::a1, &SSTCoefficients::kappa}) {
        for (double bad : {nan, inf, -inf, -1.}) {
            SSTCoefficients c;
            c.*member = bad;
            rejects([&] { (void)SSTModel(c); });
            rejects([&] { (void)SSTModel(c, SSTVariant::NoLimiter); });
            SSTModel mutated;
            mutated.coeffs = c;
            for (const auto &consume : consumers)
                rejects([&] { consume(mutated); });
        }
    }
    for (auto member : {&SSTCoefficients::beta1, &SSTCoefficients::betaStar}) {
        SSTCoefficients c;
        c.*member = 0.;
        rejects([&] { (void)SSTModel(c); });
    }
    for (int invalid : {-1, 99}) {
        rejects([&] { (void)SSTModel(SSTCoefficients{}, static_cast<SSTVariant>(invalid)); });
        SSTModel mutated;
        mutated.variant = static_cast<SSTVariant>(invalid);
        for (const auto &consume : consumers)
            rejects([&] { consume(mutated); });
    }
    SSTCoefficients disabled;
    disabled.sigma_k1 = disabled.sigma_w1 = disabled.alpha1 = disabled.sigma_k2 =
        disabled.sigma_w2 = disabled.beta2 = disabled.alpha2 = disabled.a1 = disabled.kappa = 0.;
    SSTModel disabledModel(disabled);
    require(disabledModel.eddyViscosity(.1, 2., 1., .5) == 0. &&
                disabledModel.crossDiffusion({1, 0, 0}, {1, 0, 0}, 2.) == 0.,
            "valid zero term coefficients must remain supported");

    for (auto member : {&SolverSettings::alphaP, &SolverSettings::alphaInjection,
                        &SolverSettings::kMin, &SolverSettings::kMax, &SolverSettings::omegaMin,
                        &SolverSettings::vonKarman, &SolverSettings::wallFnE})
        for (double bad : {nan, inf, -inf, -1.}) {
            SolverSettings settings;
            settings.*member = bad;
            rejects([&] { settings.validate(); });
        }
    for (double bad : {nan, inf, -inf})
        for (auto member : {&Vec3::x, &Vec3::y, &Vec3::z}) {
            SolverSettings settings;
            settings.bodyForce.*member = bad;
            rejects([&] { settings.validate(); });
        }
    for (auto member :
         {&SolverSettings::momentumConvection, &SolverSettings::turbulenceConvection}) {
        SolverSettings settings;
        settings.*member = "unsupported";
        rejects([&] { settings.validate(); });
        for (const std::string valid : {"upwind", "linearUpwind", "limitedLinearUpwind"}) {
            settings.*member = valid;
            settings.validate();
        }
    }
    SolverSettings settings;
    settings.kMin = 2.;
    settings.kMax = 1.;
    rejects([&] { settings.validate(); });
    settings = SolverSettings{};
    settings.alphaP = settings.alphaInjection = settings.kMin = settings.kMax = settings.omegaMin =
        settings.convergenceTol = settings.divergenceLimit = 0.;
    settings.validate(); // Zero limits/updates are permitted where no denominator uses them.
    settings.alphaP = settings.alphaInjection = 1.2;
    settings.validate(); // Do not impose an arbitrary blanket upper relaxation bound.

    auto mesh = Mesh::makeChannel2D(2, 2, 1., 1.);
    const auto original = FlowBoundaryConditions::channelDefaults(mesh, 1., .01, 10.);
    const auto inlet = mesh.patchByName("inlet");
    for (int invalid : {-1, 99}) {
        auto bc = original.velocityBC;
        bc[inlet].type = static_cast<BCType>(invalid);
        rejects([&] { validatePatchBCs(mesh, bc); });
    }
    VectorField velocity(mesh, "admission U");
    ScalarField k(mesh, "admission k");
    for (double bad : {nan, inf, -inf}) {
        auto bc = original;
        bc.velocityBC[inlet].vecValue.x = bad;
        rejects([&] { applyVelocityBC(velocity, mesh, bc); });
        bc = original;
        bc.velocityBC[inlet].vecProfile.assign(mesh.patch(inlet).faces.size(), {bad, 0., 0.});
        rejects([&] { applyVelocityBC(velocity, mesh, bc); });
        bc = original;
        bc.kBC[inlet].value = bad;
        rejects([&] { applyKBC(k, mesh, bc); });
        bc = original;
        bc.kBC[inlet].scalarProfile.assign(mesh.patch(inlet).faces.size(), bad);
        rejects([&] { applyKBC(k, mesh, bc); });
        bc = original;
        rejects([&] { bc.setVelocityProfile(mesh, "inlet", std::vector<Vec3>(2, {bad, 0., 0.})); });
        require(bc.velocityBC[inlet].vecProfile.empty(), "failed profile update changed storage");
        rejects([&] { bc.setKProfile(mesh, "inlet", std::vector<double>(2, bad)); });
        require(bc.kBC[inlet].scalarProfile.empty(),
                "failed scalar profile update changed storage");
        rejects([&] { bc.setOmegaProfile(mesh, "inlet", std::vector<double>(2, bad)); });
        require(bc.omegaBC[inlet].scalarProfile.empty(),
                "failed omega profile update changed storage");
    }
    // Pressure uses its uniform scalar even if a profile is stored. Check
    // the consumed value and preserve that existing field-specific behavior.
    ScalarField pressure(mesh, "admission p");
    for (double bad : {nan, inf, -inf}) {
        auto pressureBC = original;
        pressureBC.pressureBC[inlet].type = BCType::Dirichlet;
        pressureBC.pressureBC[inlet].value = bad;
        pressureBC.pressureBC[inlet].scalarProfile.assign(2, 4.);
        rejects([&] { applyPressureBC(pressure, mesh, pressureBC); });
        pressureBC.pressureBC[inlet].value = 3.;
        pressureBC.pressureBC[inlet].scalarProfile.assign(2, bad);
        applyPressureBC(pressure, mesh, pressureBC);
        for (int face : mesh.patch(inlet).faces)
            require(pressure.bface(face) == 3., "pressure no longer uses its uniform value");
    }
    auto ignored = original;
    ignored.kBC[inlet].type = BCType::OutletPressure;
    ignored.kBC[inlet].value = nan;
    ignored.kBC[inlet].scalarProfile.assign(2, nan);
    applyKBC(k, mesh, ignored);
    for (int face : mesh.patch(inlet).faces)
        require(k.bface(face) == k[mesh.face(face).owner],
                "inactive scalar boundary payload changed owner extrapolation");
    auto bc = original;
    bc.velocityBC[inlet].type = BCType::Dirichlet;
    bc.velocityBC[inlet].value = nan; // Unused scalar storage must not reject a vector BC.
    bc.velocityBC[inlet].vecValue = {nan, nan, nan}; // Profile overrides this uniform value.
    bc.setVelocityProfile(mesh, "inlet", std::vector<Vec3>(2, {3., 0., 0.}));
    bc.kBC[inlet].vecValue = {nan, nan, nan}; // Unused vector storage of scalar BC.
    bc.kBC[inlet].value = nan;
    bc.setKProfile(mesh, "inlet", std::vector<double>(2, .2));
    applyVelocityBC(velocity, mesh, bc);
    applyKBC(k, mesh, bc);
    for (int face : mesh.patch(inlet).faces)
        require(velocity.bface(face).x == 3. && k.bface(face) == .2,
                "finite profiles must override unused nonfinite uniform values");
}

static void publicCoreInputs() {
    const Vec3 scaled = 2. * Vec3{.25, -.5, .75};
    require(scaled.x == .5 && scaled.y == -1. && scaled.z == 1.5,
            "left scalar multiplication changed vector components");
    const auto rejects = [](auto action) {
        bool rejected = false;
        try {
            action();
        } catch (const std::invalid_argument &) {
            rejected = true;
        }
        require(rejected, "malformed public inference definition was accepted");
    };
    for (const auto &indices :
         {std::vector<int>{}, std::vector<int>{-1}, std::vector<int>{11}, std::vector<int>{1, 1}}) {
        auto parameters = InferenceParameterSet::a1_betaStar();
        parameters.activeIndices = indices;
        rejects([&] { (void)parameters.pack(SSTCoefficients{}); });
        rejects([&] { (void)parameters.unpack(std::vector<double>(indices.size(), .1)); });
        rejects([&] { (void)parameters.activeNames(); });
        rejects([&] { (void)parameters.lowerBounds(); });
        rejects([&] { (void)parameters.upperBounds(); });
        rejects([&] { (void)parameters.inBounds(std::vector<double>(indices.size(), .1)); });
    }
    const auto parameters = InferenceParameterSet::a1_betaStar();
    require(parameters.pack(SSTCoefficients{}) == std::vector<double>({.31, .09}),
            "valid active parameter order changed");
    require(parameters.activeNames() == std::vector<std::string>({"a1", "betaStar"}),
            "valid active parameter names changed");
    require(parameters.inBounds({.31, .09}) && !parameters.inBounds({.31}),
            "valid parameter bounds or wrong-size rejection changed");
    const auto inlet = InferenceParameterSet::inletTurb();
    require(inlet.activeNames() == std::vector<std::string>({"beta1", "sigma_w1"}) &&
                inlet.pack(SSTCoefficients{}) == std::vector<double>({.075, .5}),
            "inlet turbulence preset names or coefficient order changed");
    const auto wall = InferenceParameterSet::nearWall4();
    require(wall.activeNames() ==
                    std::vector<std::string>({"a1", "betaStar", "beta1", "sigma_k1"}) &&
                wall.pack(SSTCoefficients{}) == std::vector<double>({.31, .09, .075, .85}),
            "near-wall preset names or coefficient order changed");
    const auto custom = InferenceParameterSet::fromIndices("ordered", {8, 9});
    const auto customCoefficients = custom.unpack({.08, .30});
    require(customCoefficients.betaStar == .08 && customCoefficients.a1 == .30 &&
                customCoefficients.sigma_k1 == .85 &&
                custom.pack(customCoefficients) == std::vector<double>({.08, .30}),
            "custom inference subset failed round trip or changed an inactive coefficient");
    for (const auto &indices :
         {std::vector<int>{}, std::vector<int>{-1}, std::vector<int>{11}, std::vector<int>{2, 2}}) {
        bool rejected = false;
        try {
            (void)InferenceParameterSet::fromIndices("invalid", indices);
        } catch (const std::runtime_error &) {
            rejected = true;
        }
        require(rejected, "invalid inference factory definition was accepted");
    }

    auto mesh = Mesh::makeChannel2D(2, 2, 1., 1.);
    mesh.computeWallDistance();
    const auto plate = FlowBoundaryConditions::flatPlateDefaults(mesh, 3., .2, 7.);
    VectorField velocity(mesh, "plate velocity", {1., 1., 1.});
    ScalarField k(mesh, "plate k", .1), omega(mesh, "plate omega", 2.);
    applyVelocityBC(velocity, mesh, plate);
    applyKBC(k, mesh, plate);
    applyOmegaBC(omega, mesh, plate, .01);
    for (int face : mesh.patch(mesh.patchByName("top")).faces)
        require(velocity.bface(face).x == 3. && velocity.bface(face).y == 0. &&
                    velocity.bface(face).z == 0. && k.bface(face) == .2 && omega.bface(face) == 7.,
                "flat-plate freestream boundary does not impose the supplied state");
    for (int face : mesh.patch(mesh.patchByName("bottom")).faces)
        require(velocity.bface(face).norm2() == 0. && k.bface(face) == 0. &&
                    omega.bface(face) > 0. && std::isfinite(omega.bface(face)),
                "flat-plate wall boundary is not stationary with finite wall omega");
    ScalarField scalar(mesh, "norm", -3.);
    VectorField vector(mesh, "norm", {3., 4., 0.});
    require(scalar.linfNorm() == 3. && vector.linfNorm() == 5., "finite field norm changed");
    for (double bad :
         {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(),
          -std::numeric_limits<double>::infinity()}) {
        scalar[1] = bad;
        vector[1].x = bad;
        require(!std::isfinite(scalar.linfNorm()) && !std::isfinite(vector.linfNorm()),
                "nonfinite field value must not produce a finite infinity norm");
    }
}

static void membership(bool jacobi) {
    auto accepted = [jacobi](const std::array<double, 6> &b, double tol = 1e-6) {
        if (jacobi)
            return dbns::RealizabilityProjection::isRealizable({b[0], b[1], b[2], b[3], b[4], b[5]},
                                                               tol);
        return aniso::isRealizable(b.data(), tol);
    };
    require(accepted({0, 0, 0, 0, 0, 0}), "isotropic anisotropy rejected");
    require(accepted({2. / 3, -1. / 3, -1. / 3, 0, 0, 0}), "one-component corner rejected");
    require(!accepted({.1, .1, .1, 0, 0, 0}), "nonzero-trace tensor accepted as anisotropy");
    require(!accepted({-.01, -.01, -.01, 0, 0, 0}), "negative trace accepted as anisotropy");
    // Rotate the one-component state by 45 degrees about z: eigenvalues
    // remain (2/3,-1/3,-1/3), independently of either eigensolver.
    require(accepted({1. / 6, 1. / 6, -1. / 3, .5, 0, 0}), "rotated corner rejected");
    for (double bad :
         {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(),
          -std::numeric_limits<double>::infinity()}) {
        for (int component = 0; component < 6; ++component) {
            std::array<double, 6> b{};
            b[component] = bad;
            require(!accepted(b), "nonfinite component accepted as anisotropy");
        }
        require(!accepted({0, 0, 0, 0, 0, 0}, bad), "invalid tolerance accepted");
    }
    require(!accepted({0, 0, 0, 0, 0, 0}, -1), "negative tolerance accepted");
    require(!accepted({1e300, -1e300, 0, 1e300, 0, 0}), "overflowing spectrum accepted");
}

static void simpleDiagnostic() {
    auto mesh = Mesh::makeChannel2D(3, 4, 1, 1);
    mesh.computeWallDistance();
    SSTModel sst{SSTCoefficients{}};
    const auto bc = FlowBoundaryConditions::channelDefaults(mesh, 1, .01, 10);
    SolverSettings settings;
    settings.verbose = false;
    SIMPLESolver solver(mesh, sst, bc, .001, settings);
    FlowFields state(mesh);
    state.U.setUniform({1, 0, 0});
    state.k.setUniform(.01);
    state.omega.setUniform(10);
    std::vector<double> b(6 * mesh.nCells(), 0);
    b[0] = b[1] = b[2] = .1;
    solver.setTargetAnisotropy(&b);
    solver.assembleResidual(state, SSTCoefficients{});
    const auto &d = solver.injectionDiagnostics();
    require(d.active && d.checkedIters > 0, "SIMPLE diagnostic was not exercised");
    require(!d.allRealizable && d.maxViolation > 0, "SIMPLE diagnostic hid trace violation");
}

static void dbnsDiagnostic() {
    const auto mesh = Mesh::makeChannel2D(2, 2, 1, 1);
    dbns::DBNSBoundaryConditions bc;
    for (const auto &p : mesh.patches()) {
        dbns::BoundarySpec b;
        b.kind = dbns::BoundaryKind::Extrapolate;
        bc.set(p.name, b);
    }
    dbns::DBNSSettings settings;
    settings.viscous = false;
    settings.timeMode = dbns::TimeMode::Unsteady;
    settings.tEnd = 1e-6;
    settings.maxIterations = 1;
    settings.rkStages = 1;
    dbns::DBNSSolver solver(mesh, IdealGasEOS{}, SSTCoefficients{}, bc, settings);
    solver.initUniform({1, 0, 0, 1, 0, 0});
    std::vector<double> b(6 * mesh.nCells(), 0), diagnostic(b.size(), 0), heat;
    b[0] = b[1] = b[2] = .1;
    solver.setTargetCorrection(b, diagnostic, heat, true);
    require(solver.solve().status == EvaluationStatus::Converged, "DBNS endpoint not reached");
    const auto &d = solver.injectionDiagnostics();
    require(d.active && d.checkedIters > 0, "DBNS diagnostic was not exercised");
    require(!d.allRealizable && d.maxViolation > 0, "DBNS diagnostic hid trace violation");
}

static void duplicateEdges() {
    // Two connections between the same cells arise naturally with a two-cell
    // periodic direction. A=[[4,-3],[-3,5]] is SPD, det=11. Small AMG levels
    // promise an exact dense solve, so test M*b against the independent inverse.
    AMGSolver solver;
    for (double factor : {1., 2.}) {
        LinearSystem a(2, 2);
        a.diag = {4 * factor, 5 * factor};
        a.own = {0, 0};
        a.nbr = {1, 1};
        a.upper = {-factor, -2 * factor};
        a.lower = a.upper;
        a.source = {1, 2};
        std::vector<double> x(2, 0), z(2), r(2);
        solver.solve(a, x, 1, 1e-13);
        solver.applyPreconditioner(a.source, z);
        const double expected[2] = {1 / factor, 1 / factor};
        for (int i = 0; i < 2; ++i)
            require(std::isfinite(z[i]) && std::abs(z[i] - expected[i]) < 1e-13,
                    "AMG dense solve overwrote a repeated CSR column");
        a.residual(z, r);
        require(linalg::norm(r) < 1e-13, "independent LDU residual is nonzero");
    }
}

int main(int argc, char **argv) {
    try {
        require(argc == 2, "one contract name required");
        const std::string which = argv[1];
        if (which == "trig") {
            configurationContracts();
            publicCoreInputs();
            membership(false);
        } else if (which == "jacobi")
            membership(true);
        else if (which == "simple")
            simpleDiagnostic();
        else if (which == "dbns")
            dbnsDiagnostic();
        else if (which == "amg")
            duplicateEdges();
        else
            require(false, "unknown contract");
        std::cout << which << " contract passed\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
