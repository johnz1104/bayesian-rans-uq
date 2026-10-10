#include "SSTModel.hpp"
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
void require(bool condition, const char *message) {
    if (!condition)
        throw std::runtime_error(message);
}

template <class Fn> void rejects(Fn fn) {
    bool rejected = false;
    try {
        fn();
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    require(rejected, "invalid strain output was accepted");
}

void sameField(const ScalarField &a, const ScalarField &b) {
    require(a.data() == b.data() && a.nBnd() == b.nBnd(), "cell values changed");
    for (int i = 0; i < a.nBnd(); ++i)
        require(a.bnd(i) == b.bnd(i), "boundary values changed");
}

using Closure = std::array<ScalarField, 5>;
Closure closure(const Mesh &mesh) {
    return {ScalarField(mesh, "nuT", -7.), ScalarField(mesh, "F1", -7.),
            ScalarField(mesh, "F2", -7.), ScalarField(mesh, "Pk", -7.),
            ScalarField(mesh, "CDkw", -7.)};
}
void sameClosure(const Closure &a, const Closure &b) {
    for (std::size_t i = 0; i < a.size(); ++i)
        sameField(a[i], b[i]);
}

void fillInputs(const Mesh &mesh, VectorField &U, ScalarField &k, ScalarField &omega) {
    const auto velocity = [](const Vec3 &p) {
        return Vec3(2. * p.x + 13. * p.y * p.y, -p.x * p.y + 4. * p.y,
                    5. * p.x - 2. * p.y + .3 * p.x * p.y);
    };
    for (int i = 0; i < mesh.nCells(); ++i) {
        const auto &p = mesh.cell(i).center;
        U[i] = velocity(p);
        k[i] = .02 + .03 * p.x + .01 * p.y;
        omega[i] = 4. + 2. * p.x + p.y;
    }
    for (int fi = mesh.nInternalFaces(); fi < mesh.nFaces(); ++fi) {
        const auto &p = mesh.face(fi).center;
        U.bface(fi) = velocity(p);
        k.bface(fi) = .02 + .03 * p.x + .01 * p.y;
        omega.bface(fi) = 4. + 2. * p.x + p.y;
    }
}

void checkStrainHelper(const Mesh &mesh) {
    VectorField U(mesh, "linear U");
    const auto velocity = [](const Vec3 &p) {
        return Vec3(2. * p.x + 3. * p.y, -p.x + 4. * p.y, 5. * p.x - 2. * p.y);
    };
    for (int i = 0; i < mesh.nCells(); ++i)
        U[i] = velocity(mesh.cell(i).center);
    for (int fi = mesh.nInternalFaces(); fi < mesh.nFaces(); ++fi)
        U.bface(fi) = velocity(mesh.face(fi).center);
    const auto gradients = computeVelocityGradients(U);
    // curl(U)=(-2,-5,-4), independent of the symmetric strain tensor.
    const auto vorticity = vorticityMagnitude(gradients);
    for (double value : vorticity.data())
        require(std::abs(value - std::sqrt(45.)) < 1e-12,
                "linear three-component vorticity is incorrect");
    ScalarField out(mesh, "retained name", std::numeric_limits<double>::quiet_NaN());
    const double *cells = out.data().data();
    const double *boundary = &out.bnd(0);
    strainRateMagnitude(gradients, out);
    require(out.name == "retained name" && cells == out.data().data() && boundary == &out.bnd(0),
            "strain helper replaced caller storage");
    for (double value : out.data())
        require(std::abs(value - std::sqrt(73.)) < 1e-12,
                "linear three-component strain is incorrect");
    sameField(out, strainRateMagnitude(gradients));
    for (int i = 0; i < out.nBnd(); ++i)
        require(out.bnd(i) == 0., "strain boundary was not reset");

    ScalarField unbound;
    rejects([&] { strainRateMagnitude(gradients, unbound); });
    auto otherMesh = mesh;
    ScalarField other(otherMesh, "other", -9.);
    const auto savedOther = other;
    rejects([&] { strainRateMagnitude(gradients, other); });
    sameField(other, savedOther);
    auto shortOut = out;
    shortOut.data().pop_back();
    const auto savedShort = shortOut;
    rejects([&] { strainRateMagnitude(gradients, shortOut); });
    sameField(shortOut, savedShort);
    auto badGradients = gradients;
    badGradients.dwdx.data().pop_back();
    const auto savedOut = out;
    rejects([&] { strainRateMagnitude(badGradients, out); });
    sameField(out, savedOut);
    VelocityGradients unboundGradients;
    rejects([&] { strainRateMagnitude(unboundGradients, out); });
    sameField(out, savedOut);
}

void checkClosure(const Mesh &mesh, SSTVariant variant) {
    SSTModel model(SSTCoefficients{}, variant);
    VectorField U(mesh, "U");
    ScalarField k(mesh, "k"), omega(mesh, "omega"), nuLocal(mesh, "nu", .002);
    fillInputs(mesh, U, k, omega);
    auto output = closure(mesh), reference = closure(mesh), convenience = closure(mesh);
    ScalarField strain(mesh, "reused strain", -7.);
    const double *cells = strain.data().data();
    const double *boundary = &strain.bnd(0);
    std::vector<double> previous;
    for (int pass = 0; pass < 2; ++pass) {
        if (pass == 1) {
            // Only boundary U changes: stale cell-only gradient caches must not pass.
            for (int i = 0; i < U.nBnd(); ++i) {
                U.bnd(i).x += .2;
                U.bnd(i).z -= .7;
            }
        }
        const auto expectedStrain = strainRateMagnitude(computeVelocityGradients(U));
        strain.setUniform(std::numeric_limits<double>::quiet_NaN());
        model.computeFields(mesh, k, omega, U, .002, output[0], output[1], output[2], output[3],
                            output[4], strain);
        // The untouched per-cell-viscosity implementation is an independent closure path.
        model.computeFields(mesh, k, omega, U, nuLocal, reference[0], reference[1], reference[2],
                            reference[3], reference[4]);
        model.computeFields(mesh, k, omega, U, .002, convenience[0], convenience[1], convenience[2],
                            convenience[3], convenience[4]);
        sameClosure(output, reference);
        sameClosure(output, convenience);
        sameField(strain, expectedStrain);
        require(cells == strain.data().data() && boundary == &strain.bnd(0),
                "SST replaced strain storage");
        require(strain.name == "reused strain", "SST replaced strain metadata");
        if (pass == 1)
            require(previous != strain.data(), "boundary change did not exercise strain");
        previous = strain.data();
    }

    const auto savedK = k, savedOmega = omega;
    const auto savedOutput = output;
    const auto invalidCall = [&](ScalarField &candidate) {
        const auto savedCandidate = candidate;
        rejects([&] {
            model.computeFields(mesh, k, omega, U, .002, output[0], output[1], output[2], output[3],
                                output[4], candidate);
        });
        sameField(candidate, savedCandidate);
        sameField(k, savedK);
        sameField(omega, savedOmega);
        sameClosure(output, savedOutput);
    };
    ScalarField unbound;
    invalidCall(unbound);
    auto otherMesh = mesh;
    ScalarField other(otherMesh);
    invalidCall(other);
    auto shortOut = strain;
    shortOut.data().pop_back();
    invalidCall(shortOut);
    auto longOut = strain;
    longOut.data().push_back(1.);
    invalidCall(longOut);
    invalidCall(k);
    invalidCall(omega);
    for (auto &field : output)
        invalidCall(field);

    auto shortK = k;
    shortK.data().pop_back();
    const auto savedStrain = strain;
    rejects([&] {
        model.computeFields(mesh, shortK, omega, U, .002, output[0], output[1], output[2],
                            output[3], output[4], strain);
    });
    sameField(strain, savedStrain);
    sameClosure(output, savedOutput);

    // The local-viscosity public overload has the same atomic admission
    // contract: malformed or aliased storage must not modify any field.
    const auto invalidLocal = [&](const ScalarField &candidateNu, ScalarField &candidateNuT,
                                  ScalarField &candidateF1) {
        const auto savedNu = candidateNu, savedNuT = candidateNuT, savedF1 = candidateF1;
        rejects([&] {
            model.computeFields(mesh, k, omega, U, candidateNu, candidateNuT, candidateF1,
                                output[2], output[3], output[4]);
        });
        sameField(candidateNu, savedNu);
        sameField(candidateNuT, savedNuT);
        sameField(candidateF1, savedF1);
        sameField(k, savedK);
        sameField(omega, savedOmega);
        sameClosure(output, savedOutput);
    };
    invalidLocal(nuLocal, output[0], output[0]);
    invalidLocal(nuLocal, k, output[1]);
    invalidLocal(nuLocal, omega, output[1]);
    invalidLocal(nuLocal, nuLocal, output[1]);
    invalidLocal(nuLocal, unbound, output[1]);
    invalidLocal(nuLocal, other, output[1]);
    invalidLocal(nuLocal, shortOut, output[1]);
    auto shortNu = nuLocal;
    shortNu.data().pop_back();
    invalidLocal(shortNu, output[0], output[1]);
    invalidLocal(other, output[0], output[1]);
}

void checkBoundaryStorage() {
    auto mesh = Mesh::makeChannel2D(5, 6, 2., 1.);
    ScalarField oldStrain(mesh, "stale boundary storage", -3.);
    const auto &oldCells = oldStrain.data();
    const auto savedCells = oldCells;
    // Same cell count and mesh address, but more boundary faces.
    mesh = Mesh::makeChannel2D(1, 30, 2., 1.);
    mesh.computeWallDistance();
    VectorField U(mesh, "U");
    ScalarField k(mesh, "k"), omega(mesh, "omega");
    fillInputs(mesh, U, k, omega);
    auto output = closure(mesh);
    const auto savedOutput = output;
    SSTModel model;
    rejects([&] {
        model.computeFields(mesh, k, omega, U, .002, output[0], output[1], output[2], output[3],
                            output[4], oldStrain);
    });
    require(oldCells == savedCells, "rejected stale output was mutated");
    sameClosure(output, savedOutput);
    const auto gradients = computeVelocityGradients(U);
    rejects([&] { strainRateMagnitude(gradients, oldStrain); });
    require(oldCells == savedCells, "rejected stale output was mutated");
    bool stale = false;
    try {
        (void)oldStrain.data();
    } catch (const std::logic_error &) {
        stale = true;
    }
    require(stale, "stale data access was accepted");
}
} // namespace

int main() {
    try {
        auto mesh = Mesh::makeChannel2D(5, 6, 2., 1.);
        mesh.computeWallDistance();
        checkStrainHelper(mesh);
        for (const auto variant : {SSTVariant::Full, SSTVariant::NoLimiter, SSTVariant::KOmega})
            checkClosure(mesh, variant);
        checkBoundaryStorage();
        std::cout << "SST strain output contracts passed\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
