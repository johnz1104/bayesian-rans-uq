#include "DBNSSolver.hpp"
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

template <class Action> void invalidArgument(Action action, const char *message) {
    try {
        action();
    } catch (const std::invalid_argument &error) {
        if (std::string(error.what()).empty())
            throw std::runtime_error("input rejection has no explanation");
        return;
    }
    throw std::runtime_error(message);
}

template <class Action> void invalidIndex(Action action) {
    try {
        action();
    } catch (const std::out_of_range &error) {
        if (std::string(error.what()).empty())
            throw std::runtime_error("index rejection has no explanation");
        return;
    }
    throw std::runtime_error("DBNS accessor accepted an invalid cell index");
}

// Initial-state admission applies in both modes; the turbulent run keeps the
// finite and nonnegative k/omega checks independent of the laminar zero rule.
static void initialAdmission(const Mesh &mesh, const dbns::DBNSSettings &settings,
                             const dbns::Primitive &initial) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    dbns::DBNSSolver solver(mesh, {}, {}, {}, settings);
    solver.initUniform(initial);
    const auto before = solver.conserved(0);
    auto unchanged = [&] {
        for (int cell = 0; cell < mesh.nCells(); ++cell)
            if (solver.conserved(cell) != before)
                throw std::runtime_error("invalid initial state changed the old solution");
    };
    using P = dbns::Primitive;
    for (auto member : {&P::rho, &P::u, &P::v, &P::p, &P::k, &P::omega})
        for (double value : {nan, inf}) {
            auto bad = initial;
            bad.*member = value;
            invalidArgument([&] { solver.initUniform(bad); }, "nonfinite uniform state admitted");
            unchanged();
            std::vector<P> field(mesh.nCells(), initial);
            field.front().rho *= 2;
            field.back() = bad;
            invalidArgument([&] { solver.initField(field); }, "nonfinite field state admitted");
            unchanged();
        }
    for (auto member : {&P::rho, &P::p, &P::k, &P::omega}) {
        auto bad = initial;
        bad.*member = -1;
        invalidArgument([&] { solver.initUniform(bad); }, "negative primitive admitted");
        unchanged();
    }
    for (auto member : {&P::rho, &P::p}) {
        auto bad = initial;
        bad.*member = 0;
        invalidArgument([&] { solver.initUniform(bad); }, "zero density or pressure admitted");
        unchanged();
    }
    auto overflow = initial;
    overflow.u = 1e300;
    invalidArgument([&] { solver.initUniform(overflow); }, "overflowing conserved state admitted");
    unchanged();
    auto zeroTurbulence = initial;
    zeroTurbulence.k = zeroTurbulence.omega = 0;
    solver.initUniform(zeroTurbulence);
    if (solver.primitive(0).rho != initial.rho || solver.primitive(0).k != 0)
        throw std::runtime_error("valid zero-turbulence initial state changed");
}

static void secondDbnsAdmission(const Mesh &mesh, const dbns::DBNSSettings &settings,
                                const dbns::Primitive &initial, const std::string &contract) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    if (contract == "all" || contract == "sst2") {
        for (double value : {0.0, nan}) {
            SSTCoefficients bad;
            bad.betaStar = value;
            invalidArgument([&] { dbns::DBNSSolver rejected(mesh, {}, bad, {}, settings); },
                            "DBNS admitted invalid SST coefficients");
        }
    }
    if (contract == "all" || contract == "settings2") {
        using D = dbns::DBNSSettings;
        for (auto member : {&D::cfl, &D::kFloor, &D::omegaFloor, &D::cflImplicit, &D::cflRampStart})
            for (double value : {-1.0, 0.0, nan, inf}) {
                auto bad = settings;
                bad.*member = value;
                invalidArgument([&] { dbns::DBNSSolver rejected(mesh, {}, {}, {}, bad); },
                                "DBNS accepted a nonpositive or nonfinite positive setting");
            }
        for (auto member : {&D::tEnd, &D::convergenceTol, &D::divergenceLimit, &D::turbMachCutoff,
                            &D::comprXiStar, &D::earlyAbortRelMax})
            for (double value : {-1.0, nan, inf}) {
                auto bad = settings;
                bad.*member = value;
                invalidArgument([&] { dbns::DBNSSolver rejected(mesh, {}, {}, {}, bad); },
                                "DBNS accepted a negative or nonfinite nonnegative setting");
            }
        for (double value : {nan, inf}) {
            auto bad = settings;
            bad.constMu = value;
            invalidArgument([&] { dbns::DBNSSolver rejected(mesh, {}, {}, {}, bad); },
                            "DBNS accepted nonfinite viscosity selection");
        }
        for (auto member :
             {&D::maxIterations, &D::earlyAbortIter, &D::injectionRampIters, &D::cflRampIters}) {
            auto bad = settings;
            bad.*member = -1;
            invalidArgument([&] { dbns::DBNSSolver rejected(mesh, {}, {}, {}, bad); },
                            "DBNS accepted a negative iteration count");
        }
        auto bad = settings;
        bad.timeMode = static_cast<dbns::TimeMode>(99);
        invalidArgument([&] { dbns::DBNSSolver rejected(mesh, {}, {}, {}, bad); },
                        "DBNS accepted an invalid time mode");
        bad = settings;
        bad.compressibility = static_cast<dbns::CompressibilityModel>(99);
        invalidArgument([&] { dbns::DBNSSolver rejected(mesh, {}, {}, {}, bad); },
                        "DBNS accepted an invalid compressibility model");
        auto zero = settings;
        zero.maxIterations = zero.earlyAbortIter = zero.injectionRampIters = zero.cflRampIters = 0;
        zero.tEnd = zero.convergenceTol = zero.divergenceLimit = 0;
        zero.turbMachCutoff = zero.comprXiStar = zero.earlyAbortRelMax = 0;
        zero.constMu = 0;
        dbns::DBNSSolver supportedZero(mesh, {}, {}, {}, zero);
        zero.constMu = -2; // every finite negative value selects Sutherland
        dbns::DBNSSolver supportedLaw(mesh, {}, {}, {}, zero);
    }
    if (contract == "all" || contract == "mask2" || contract == "atomic2") {
        auto shortRun = settings;
        shortRun.timeMode = dbns::TimeMode::Unsteady;
        shortRun.tEnd = 1e-6;
        shortRun.maxIterations = 1;
        shortRun.viscous = false;
        shortRun.turbulent = true;
        dbns::DBNSSolver solver(mesh, {}, {}, {}, shortRun);
        solver.initUniform({1, .2, 0, 1, 1, 10});
        std::vector<double> db(6 * mesh.nCells(), 0), target(db.size(), 0);
        db[0] = 2; // visibly unrealizable if the all-zero activity mask is lost
        const std::vector<std::uint8_t> zeroMask(mesh.nCells(), 0);
        solver.setTargetCorrection(db, target, {}, true, zeroMask);
        if (contract != "atomic2") {
            for (int offset : {-1, 1})
                invalidArgument(
                    [&] {
                        solver.setTargetCorrection(
                            db, target, {}, true,
                            std::vector<std::uint8_t>(mesh.nCells() + offset, 0));
                    },
                    "DBNS accepted a nonempty correction mask with the wrong size");
        }
        bool rejected = false;
        try {
            solver.setTargetCorrection({}, target, {}, true,
                                       std::vector<std::uint8_t>(mesh.nCells(), 1));
        } catch (const std::exception &) {
            rejected = true;
        }
        if (!rejected)
            throw std::runtime_error("malformed correction update was accepted");
        if (solver.solve().status != EvaluationStatus::Converged ||
            !solver.injectionDiagnostics().allRealizable)
            throw std::runtime_error("rejected correction update changed the previous mask");
    }
    if (contract == "all" || contract == "correction2") {
        dbns::DBNSSolver solver(mesh, {}, {}, {}, settings);
        solver.initUniform(initial);
        for (int component : {0, 1, 2}) {
            std::vector<double> db(6 * mesh.nCells(), 0), target(db.size(), 0),
                heat(2 * mesh.nCells(), 0);
            if (component == 0)
                db.back() = nan;
            if (component == 1)
                target.back() = nan;
            if (component == 2)
                heat.back() = nan;
            invalidArgument([&] { solver.setTargetCorrection(db, target, heat); },
                            "DBNS accepted a nonfinite correction field");
        }
    }
    if (contract == "all" || contract == "limiter2") {
        dbns::DBNSSolver solver(mesh, {}, {}, {}, settings);
        const std::vector<double> half(6 * mesh.nCells(), .5);
        solver.setFrozenReconstructionLimiter(half);
        for (double value : {-.1, 1.1, nan, inf}) {
            auto bad = half;
            bad.front() = 1;
            bad.back() = value;
            invalidArgument([&] { solver.setFrozenReconstructionLimiter(bad); },
                            "DBNS admitted an invalid frozen reconstruction factor");
            if (solver.reconstructionLimiter() != half)
                throw std::runtime_error("invalid frozen limiter modified prior factors");
        }
        auto valid = half;
        valid.front() = 0;
        valid.back() = 1;
        solver.setFrozenReconstructionLimiter(valid);
        if (solver.reconstructionLimiter() != valid)
            throw std::runtime_error("valid frozen reconstruction factors were changed");
    }
    if (contract == "all" || contract == "initial2") {
        initialAdmission(mesh, settings, initial);
        auto turbulent = settings;
        turbulent.turbulent = true;
        auto turbulentInitial = initial;
        turbulentInitial.k = 0.1;
        turbulentInitial.omega = 4.0;
        initialAdmission(mesh, turbulent, turbulentInitial);
    }
    if (contract == "all" || contract == "name2") {
        dbns::DBNSBoundaryConditions bad;
        bad.set("misspelled-patch", {});
        invalidArgument([&] { dbns::DBNSSolver rejected(mesh, {}, {}, bad, settings); },
                        "DBNS silently ignored an unknown patch name");
    }
    if (contract == "all" || contract == "boundary2") {
        auto reject = [&](const dbns::BoundarySpec &spec) {
            dbns::DBNSBoundaryConditions bc;
            bc.set(mesh.patch(0).name, spec);
            invalidArgument([&] { dbns::DBNSSolver rejected(mesh, {}, {}, bc, settings); },
                            "DBNS admitted an invalid used boundary value");
        };
        dbns::BoundarySpec spec;
        spec.kind = static_cast<dbns::BoundaryKind>(99);
        reject(spec);
        using P = dbns::Primitive;
        for (auto kind : {dbns::BoundaryKind::SupersonicInflow, dbns::BoundaryKind::FixedState,
                          dbns::BoundaryKind::SubsonicInflow}) {
            spec = {};
            spec.kind = kind;
            for (auto member : {&P::rho, &P::p, &P::k, &P::omega}) {
                spec.freestream = initial;
                spec.freestream.*member = -1;
                reject(spec);
            }
            for (auto member : {&P::rho, &P::u, &P::v, &P::p, &P::k, &P::omega}) {
                spec.freestream = initial;
                spec.freestream.*member = nan;
                reject(spec);
            }
        }
        spec = {};
        spec.kind = dbns::BoundaryKind::FixedState;
        spec.profile.assign(mesh.patch(0).faces.size(), initial);
        spec.profile.back().p = -1;
        reject(spec);
        spec = {};
        spec.kind = dbns::BoundaryKind::SubsonicOutflow;
        for (double value : {-1.0, 0.0, nan, inf}) {
            spec.backPressure = value;
            reject(spec);
        }
        spec = {};
        spec.kind = dbns::BoundaryKind::NoSlipIsothermal;
        for (double value : {-1.0, 0.0, nan, inf}) {
            spec.wallTemp = value;
            reject(spec);
        }
        spec.wallTemp = 300;
        spec.wallTempProfile.assign(mesh.patch(0).faces.size(), 300);
        spec.wallTempProfile.back() = nan;
        reject(spec);
        for (auto kind :
             {dbns::BoundaryKind::NoSlipIsothermal, dbns::BoundaryKind::NoSlipAdiabatic}) {
            spec = {};
            spec.kind = kind;
            spec.wallVelocity.x = nan;
            reject(spec);
        }
    }
}

int main(int argc, char **argv) {
    const auto mesh = Mesh::makeChannel2D(2, 2, 1.0, 1.0);
    dbns::DBNSSettings settings;
    settings.verbose = false;
    dbns::DBNSSolver solver(mesh, IdealGasEOS{}, SSTCoefficients{}, dbns::DBNSBoundaryConditions{},
                            settings);
    const dbns::Primitive initial{1.0, 2.0, 3.0, 100000.0, 0.0, 0.0};
    solver.initUniform(initial);
    const auto before = solver.conserved(0);
    const int offset = argc > 1 && std::string(argv[1]) == "short" ? -1 : 1;
    std::vector<dbns::Primitive> malformed(mesh.nCells() + offset, initial);
    malformed.front().rho = 2.0;
    bool rejected = false;
    try {
        solver.initField(malformed);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    if (!rejected) {
        std::cerr << "initField accepted a field with the wrong cell count\n";
        return 1;
    }
    for (int i = 0; i < mesh.nCells(); ++i) {
        if (solver.conserved(i) != before) {
            std::cerr << "initField modified state before rejecting field size\n";
            return 1;
        }
    }
    std::vector<dbns::Primitive> valid(mesh.nCells(), initial);
    valid.back().rho = 2.0;
    solver.initField(valid);
    if (solver.primitive(mesh.nCells() - 1).rho != 2.0) {
        std::cerr << "initField rejected or changed a valid field\n";
        return 1;
    }
    const std::string contract = argc > 2 ? argv[2] : "all";
    if (contract == "all" || contract == "source") {
        invalidArgument(
            [&] {
                solver.setManufacturedSource(std::vector<dbns::StateVec>(mesh.nCells() + offset));
            },
            "MMS source accepted the wrong cell count");
        solver.setManufacturedSource(std::vector<dbns::StateVec>(mesh.nCells()));
        solver.setManufacturedSource({}); // empty explicitly clears a source
    }
    if (contract == "all" || contract == "ghost") {
        const int count = mesh.nFaces() - mesh.nInternalFaces();
        invalidArgument(
            [&] {
                solver.setBoundaryOverride(std::vector<dbns::Primitive>(count + offset, initial));
            },
            "ghost override accepted the wrong boundary-face count");
        solver.setBoundaryOverride(std::vector<dbns::Primitive>(count, initial));
        solver.setBoundaryOverride({});
    }
    if (contract == "all" || contract == "index") {
        for (int index : {-1, mesh.nCells()}) {
            invalidIndex([&] { (void)solver.primitive(index); });
            invalidIndex([&] { (void)solver.conserved(index); });
            invalidIndex([&] { (void)solver.eddyViscosity(index); });
            invalidIndex([&] { (void)solver.laminarViscosity(index); });
        }
        (void)solver.primitive(0);
        (void)solver.conserved(mesh.nCells() - 1);
        (void)solver.eddyViscosity(0);
        (void)solver.laminarViscosity(mesh.nCells() - 1);
    }
    if (contract == "all" || contract == "interval") {
        for (int interval : {0, -1}) {
            auto bad = settings;
            bad.reportInterval = interval;
            invalidArgument([&] { dbns::DBNSSolver rejected(mesh, {}, {}, {}, bad); },
                            "DBNS accepted a nonpositive report interval");
        }
    }
    if (contract == "all" || contract == "modes") {
        auto bad = settings;
        bad.reconstructOrder = 3;
        invalidArgument([&] { dbns::DBNSSolver rejected(mesh, {}, {}, {}, bad); },
                        "DBNS silently substituted an unsupported reconstruction order");
        bad = settings;
        bad.rkStages = 2;
        invalidArgument([&] { dbns::DBNSSolver rejected(mesh, {}, {}, {}, bad); },
                        "DBNS silently substituted an unsupported RK scheme");
    }
    if (contract == "all" || contract == "eos") {
        for (int parameter = 0; parameter < 7; ++parameter) {
            for (double invalid : {0.0, -1.0, std::numeric_limits<double>::quiet_NaN(),
                                   std::numeric_limits<double>::infinity()}) {
                IdealGasEOS eos;
                switch (parameter) {
                case 0:
                    eos.gamma = invalid == 0.0 ? 1.0 : invalid;
                    break;
                case 1:
                    eos.R = invalid;
                    break;
                case 2:
                    eos.Pr = invalid;
                    break;
                case 3:
                    eos.Pr_T = invalid;
                    break;
                case 4:
                    eos.mu_ref = invalid;
                    break;
                case 5:
                    eos.T_ref = invalid;
                    break;
                case 6:
                    eos.S_suth = invalid;
                    break;
                }
                if (invalid == 0.0 && (parameter == 4 || parameter == 6))
                    continue; // zero viscosity or Sutherland offset is a supported limit
                invalidArgument([&] { dbns::DBNSSolver rejected(mesh, eos, {}, {}, settings); },
                                "DBNS accepted inadmissible ideal-gas parameters");
            }
        }
    }
    if (contract == "all" || contract == "profile") {
        dbns::BoundarySpec spec;
        spec.kind = dbns::BoundaryKind::FixedState;
        spec.freestream = initial;
        const auto &patch = mesh.patch(0);
        spec.profile.assign(patch.faces.size() + offset, initial);
        dbns::DBNSBoundaryConditions bc;
        bc.set(patch.name, spec);
        invalidArgument([&] { dbns::DBNSSolver rejected(mesh, {}, {}, bc, settings); },
                        "DBNS accepted a partial or oversized prescribed-state profile");
        bc.specs.at(patch.name).profile.assign(patch.faces.size(), initial);
        dbns::DBNSSolver accepted(mesh, {}, {}, bc, settings);

        spec.profile.clear();
        spec.kind = dbns::BoundaryKind::NoSlipIsothermal;
        spec.wallTempProfile.assign(patch.faces.size() + offset, 300.0);
        bc.set(patch.name, spec);
        invalidArgument([&] { dbns::DBNSSolver rejected(mesh, {}, {}, bc, settings); },
                        "DBNS accepted a partial or oversized wall-temperature profile");
        bc.specs.at(patch.name).wallTempProfile.assign(patch.faces.size(), 300.0);
        dbns::DBNSSolver acceptedWall(mesh, {}, {}, bc, settings);
    }
    secondDbnsAdmission(mesh, settings, initial, contract);
    return 0;
}
