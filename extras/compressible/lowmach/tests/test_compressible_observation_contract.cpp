// The generic drag adapter uses kinematic pressure and kinematic viscosity.
// CompressibleForwardModel carries dimensional pressure but passes a
// density-free FlowFields view to that adapter, so the combination must fail
// at construction rather than return a plausible mixed-units coefficient.

#include "Mesh.hpp"
#include "ObservationOperator.hpp"
#include "CompressibleForwardModel.hpp"
#include "CompressibleSIMPLESolver.hpp"
#include "CompressibleBCs.hpp"
#include "IdealGasEOS.hpp"
#include "InferenceParameters.hpp"
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <limits>

namespace {

#define REQUIRE(cond, msg)                                                                         \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            std::fprintf(stderr, "FAIL [%s:%d] %s\n  required: %s\n", __FILE__, __LINE__, (msg),   \
                         #cond);                                                                   \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

} // namespace

static void secondCompressibleAdmission(const std::string &contract) {
    Mesh mesh = Mesh::makeChannel2D(3, 3, 1, 1);
    mesh.computeWallDistance();
    const IdealGasEOS eos;
    const SSTModel sst;
    auto bc = CompressibleBoundaryConditions::channelDefaults(mesh, 2, 300, 100000, .1, 4);
    const auto params = InferenceParameterSet::a1_betaStar();
    const ObservationOperator observations;
    SolverSettings settings;
    settings.maxIterations = 0;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    auto expect = [&](auto action, const char *message) {
        bool rejected = false;
        try {
            action();
        } catch (const std::invalid_argument &error) {
            rejected = !std::string(error.what()).empty();
        }
        REQUIRE(rejected, message);
    };
    if (contract == "all" || contract == "sst2") {
        for (double value : {0.0, nan}) {
            SSTModel bad = sst;
            bad.coeffs.betaStar = value;
            expect([&] { CompressibleSIMPLESolver solver(mesh, bad, bc, eos, settings); },
                   "low-Mach constructor admitted invalid mutated SST coefficients");
        }
        SSTModel bad = sst;
        bad.variant = static_cast<SSTVariant>(99);
        expect([&] { CompressibleSIMPLESolver solver(mesh, bad, bc, eos, settings); },
               "low-Mach constructor admitted invalid mutated SST variant");
    }
    if (contract == "all" || contract == "initial2") {
        CompressibleSIMPLESolver solver(mesh, sst, bc, eos, settings);
        CompressibleFlowFields fields(mesh);
        fields.U.setUniform({7, 8, 9});
        fields.p.setUniform(17);
        fields.T.setUniform(317);
        fields.k.setUniform(.2);
        fields.omega.setUniform(13);
        for (int component = 0; component < 5; ++component)
            for (double value : {-1.0, 0.0, nan, inf}) {
                if ((component == 0 && std::isfinite(value)) ||
                    ((component == 3 || component == 4) && value == 0))
                    continue;
                Vec3 velocity{2, 0, 0};
                double pressure = 100000, temperature = 300, k = .1, omega = 4;
                switch (component) {
                case 0:
                    velocity.x = value;
                    break;
                case 1:
                    pressure = value;
                    break;
                case 2:
                    temperature = value;
                    break;
                case 3:
                    k = value;
                    break;
                case 4:
                    omega = value;
                    break;
                }
                expect(
                    [&] { solver.initUniform(fields, velocity, pressure, temperature, k, omega); },
                    "low-Mach initializer admitted an invalid physical value");
                REQUIRE(fields.U[0].x == 7 && fields.p[0] == 17 && fields.T[0] == 317 &&
                            fields.k[0] == .2 && fields.omega[0] == 13,
                        "invalid low-Mach initialization modified the old state");
                expect(
                    [&] {
                        CompressibleForwardModel bad(mesh, params, observations, bc, eos, settings,
                                                     velocity, pressure, temperature, k, omega);
                    },
                    "forward model admitted an invalid initial state");
            }
        solver.initUniform(fields, {2, 0, 0}, 100000, 300, 0, 0);
        REQUIRE(solver.stateIsValid(fields), "valid zero-turbulence initial state rejected");
    }
    if (contract == "all" || contract == "layout2") {
        CompressibleSIMPLESolver solver(mesh, sst, bc, eos, settings);
        for (int side : {3, 1}) {
            Mesh other = Mesh::makeChannel2D(side, side, 1, 1);
            other.computeWallDistance();
            CompressibleFlowFields fields(other);
            fields.U.setUniform({2, 0, 0});
            fields.p.setUniform(100000);
            fields.T.setUniform(300);
            fields.rho.setUniform(1);
            fields.k.setUniform(.1);
            fields.omega.setUniform(4);
            expect([&] { solver.initUniform(fields, {2, 0, 0}, 100000, 300, .1, 4); },
                   "low-Mach initialization accepted fields from another mesh");
            expect([&] { (void)solver.stateIsValid(fields); },
                   "low-Mach state validation accepted another mesh's fields");
            expect([&] { (void)solver.solve(fields); },
                   "low-Mach solve accepted fields from another mesh");
        }
        CompressibleFlowFields resized(mesh);
        resized.T.data().resize(1);
        expect([&] { solver.initUniform(resized, {2, 0, 0}, 100000, 300, .1, 4); },
               "low-Mach initialization accepted resized field storage");
        Mesh missing = Mesh::makeChannel2D(3, 3, 1, 1);
        const auto missingBC =
            CompressibleBoundaryConditions::channelDefaults(missing, 2, 300, 100000, .1, 4);
        CompressibleSIMPLESolver missingSolver(missing, sst, missingBC, eos, settings);
        CompressibleFlowFields missingFields(missing);
        missingFields.U.setUniform({7, 0, 0});
        bool rejected = false;
        try {
            missingSolver.initUniform(missingFields, {2, 0, 0}, 100000, 300, .1, 4);
        } catch (const std::logic_error &) {
            rejected = true;
        }
        REQUIRE(rejected && missingFields.U[0].x == 7,
                "missing wall-distance preflight modified fields before throwing");
    }
    if (contract == "all" || contract == "boundary2") {
        auto reject = [&](const auto &bad) {
            expect([&] { CompressibleSIMPLESolver solver(mesh, sst, bad, eos, settings); },
                   "low-Mach constructor admitted an invalid boundary value or mode");
        };
        for (int component = 0; component < 5; ++component) {
            auto bad = bc;
            switch (component) {
            case 0:
                bad.velocityBC[0].type = static_cast<BCType>(99);
                break;
            case 1:
                bad.pressureBC[0].type = static_cast<BCType>(99);
                break;
            case 2:
                bad.temperatureBC[0].type = static_cast<TBCType>(99);
                break;
            case 3:
                bad.kBC[0].type = static_cast<BCType>(99);
                break;
            case 4:
                bad.omegaBC[0].type = static_cast<BCType>(99);
                break;
            }
            reject(bad);
        }
        for (int component = 0; component < 5; ++component)
            for (double value : {-1.0, 0.0, nan, inf}) {
                if ((component == 0 && std::isfinite(value)) ||
                    ((component == 3 || component == 4) && value == 0))
                    continue;
                auto bad = bc;
                switch (component) {
                case 0:
                    bad.velocityBC[0].type = BCType::Dirichlet;
                    bad.velocityBC[0].vecValue.x = value;
                    break;
                case 1:
                    bad.pressureBC[0].type = BCType::Dirichlet;
                    bad.pressureBC[0].value = value;
                    break;
                case 2:
                    bad.temperatureBC[0].type = TBCType::Dirichlet;
                    bad.temperatureBC[0].value = value;
                    break;
                case 3:
                    bad.kBC[0].type = BCType::Dirichlet;
                    bad.kBC[0].value = value;
                    break;
                case 4:
                    bad.omegaBC[0].type = BCType::Dirichlet;
                    bad.omegaBC[0].value = value;
                    break;
                }
                reject(bad);
            }
        auto bad = bc;
        bad.velocityBC[0].type = BCType::Dirichlet;
        bad.velocityBC[0].vecProfile.assign(mesh.patch(0).faces.size(), {7, 0, 0});
        reject(bad);
        bad = bc;
        bad.pressureBC[0].type = BCType::Dirichlet;
        bad.pressureBC[0].value = 100000; // isolate unsupported profile from pressure positivity
        bad.pressureBC[0].scalarProfile.assign(mesh.patch(0).faces.size(), 200000);
        reject(bad);
        bad = bc;
        bad.velocityBC[0].type = BCType::WallMoving;
        bad.velocityBC[0].vecValue = {9, 0, 0};
        reject(bad);
    }
}

int main(int argc, char **argv) {
    Mesh mesh = Mesh::makeChannel2D(8, 6, 2.0, 1.0, 5000.0, 1.0);
    mesh.computeWallDistance();
    IdealGasEOS eos;
    const double T = 300.0, p = 101325.0, U = 10.0;
    auto bcs = CompressibleBoundaryConditions::channelDefaults(mesh, U, T, p, 1e-3, 10.0);
    InferenceParameterSet params = InferenceParameterSet::a1_betaStar();

    ObservationOperator drag;
    drag.addDrag("bottom", 0.0, 1.0, 1.0, U);
    bool rejected = false;
    try {
        CompressibleForwardModel unsupported(mesh, params, drag, bcs, eos, SolverSettings{},
                                             Vec3(U, 0.0, 0.0), p, T, 1e-3, 10.0);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    REQUIRE(rejected, "compressible generic drag must be rejected at construction");

    // A referenced pressure tap remains supported through the thermodynamic
    // pressure shim and the explicit Cp normalization contract.
    ObservationOperator pressure;
    pressure.addPressureTap(Vec3(1.0, 0.5, 0.0), 0.0, 1.0, U, 0.0, p, eos.density(p, T));
    bool accepted = true;
    try {
        CompressibleForwardModel supported(mesh, params, pressure, bcs, eos, SolverSettings{},
                                           Vec3(U, 0.0, 0.0), p, T, 1e-3, 10.0);
        REQUIRE(supported.paramSet().activeIndices == params.activeIndices &&
                    supported.paramSet().pack(SSTCoefficients{}) == params.pack(SSTCoefficients{}),
                "compressible parameter metadata differs from its constructor input");
    } catch (...) {
        accepted = false;
    }
    REQUIRE(accepted, "referenced compressible pressure tap must stay supported");
    const std::string contract = argc > 1 ? argv[1] : "all";
    if (contract == "all" || contract == "eos")
        for (int parameter : {0, 1}) {
            auto badEOS = eos;
            if (parameter == 0)
                badEOS.gamma = 1.0;
            else
                badEOS.R = 0.0;
            bool gasRejected = false;
            try {
                CompressibleSIMPLESolver rejectedGas(mesh, SSTModel{}, bcs, badEOS);
            } catch (const std::invalid_argument &) {
                gasRejected = true;
            }
            bool forwardGasRejected = false;
            try {
                CompressibleForwardModel rejectedGas(mesh, params, pressure, bcs, badEOS,
                                                     SolverSettings{}, Vec3(U, 0, 0), p, T, 1e-3,
                                                     10.0);
            } catch (const std::invalid_argument &) {
                forwardGasRejected = true;
            }
            REQUIRE(forwardGasRejected,
                    "low-Mach forward model accepted singular ideal-gas parameters");
            REQUIRE(gasRejected, "low-Mach solver accepted singular ideal-gas parameters");
        }

    // Every patch array must be checked before any field is modified. An
    // oversized layout is safe to exercise even on an unguarded implementation;
    // the undersized layouts additionally expose unchecked indexing to ASan.
    if (contract == "all" || contract == "boundary")
        for (int offset : {1, -1}) {
            for (int component = 0; component < 5; ++component) {
                auto malformed = bcs;
                const auto count = static_cast<size_t>(mesh.nPatches() + offset);
                switch (component) {
                case 0:
                    malformed.velocityBC.resize(count);
                    break;
                case 1:
                    malformed.pressureBC.resize(count);
                    break;
                case 2:
                    malformed.temperatureBC.resize(count);
                    break;
                case 3:
                    malformed.kBC.resize(count);
                    break;
                case 4:
                    malformed.omegaBC.resize(count);
                    break;
                }
                CompressibleFlowFields fields(mesh);
                fields.U.setUniform({7, 8, 9});
                fields.p.setUniform(17);
                fields.T.setUniform(317);
                fields.k.setUniform(.2);
                fields.omega.setUniform(13);
                bool layoutRejected = false;
                try {
                    applyAllCompressibleBCs(fields.U, fields.p, fields.T, fields.k, fields.omega,
                                            mesh, malformed, 1e-5);
                } catch (const std::invalid_argument &error) {
                    layoutRejected = !std::string(error.what()).empty();
                }
                REQUIRE(layoutRejected, "compressible BC application accepted a wrong patch count");
                for (int face = mesh.nInternalFaces(); face < mesh.nFaces(); ++face) {
                    REQUIRE(fields.U.bface(face).x == 7 && fields.p.bface(face) == 17 &&
                                fields.T.bface(face) == 317 && fields.k.bface(face) == .2 &&
                                fields.omega.bface(face) == 13,
                            "compressible BC rejection modified a boundary field");
                }
                bool constructorRejected = false;
                try {
                    CompressibleSIMPLESolver rejected(mesh, SSTModel{}, malformed, eos);
                } catch (const std::invalid_argument &) {
                    constructorRejected = true;
                }
                REQUIRE(constructorRejected, "compressible solver accepted a wrong BC patch count");
                if (component == 2) {
                    bool temperatureRejected = false;
                    try {
                        applyTemperatureBC(fields.T, mesh, malformed);
                    } catch (const std::invalid_argument &) {
                        temperatureRejected = true;
                    }
                    REQUIRE(temperatureRejected, "temperature helper accepted a wrong patch count");
                }
            }
        }

    secondCompressibleAdmission(contract);
    std::printf("test_compressible_observation_contract: all checks passed\n");
    return 0;
}
