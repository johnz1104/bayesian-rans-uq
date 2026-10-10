#include "SSTModel.hpp"
#include "FlowFields.hpp"
#include <iostream>
#include <limits>

static void require(bool value) {
    if (!value)
        throw std::runtime_error("gradient/SST workspace mismatch");
}
template <class T> static void same(const Field<T> &a, const Field<T> &b) {
    for (int i = 0; i < a.size(); ++i) {
        if constexpr (std::is_same_v<T, double>)
            require(a[i] == b[i]);
        else
            require((a[i] - b[i]).norm2() == 0);
    }
    for (int i = 0; i < a.nBnd(); ++i) {
        if constexpr (std::is_same_v<T, double>)
            require(a.bnd(i) == b.bnd(i));
        else
            require((a.bnd(i) - b.bnd(i)).norm2() == 0);
    }
}
int main() {
    auto mesh = Mesh::makeChannel2D(5, 6, 2, 1);
    mesh.computeWallDistance();
    FlowFields fields(mesh), result(mesh), reference(mesh);
    ScalarField strain(mesh, "|S|");
    SSTWorkspace scratch(mesh);
    SSTModel model;
    const double poison = std::numeric_limits<double>::quiet_NaN();
    for (int i = 0; i < mesh.nCells(); ++i) {
        const auto p = mesh.cell(i).center;
        fields.U[i] = {1 + p.x * p.y, p.x - .2 * p.y, 2 * p.y};
        fields.k[i] = .1 + .02 * p.y;
        fields.omega[i] = 3 + p.x;
    }
    for (int fi = mesh.nInternalFaces(); fi < mesh.nFaces(); ++fi) {
        const auto p = mesh.face(fi).center;
        fields.U.bface(fi) = {1 + p.x * p.y, p.x - .2 * p.y, 2 * p.y};
        fields.k.bface(fi) = .1 + .02 * p.y;
        fields.omega.bface(fi) = 3 + p.x;
    }
    const auto *retained = scratch.gradK.data().data();
    for (int pass = 0; pass < 3; ++pass) {
        for (auto *f : {&scratch.velocity.dudx, &scratch.velocity.dvdx, &scratch.velocity.dwdx,
                        &scratch.gradK, &scratch.gradOmega})
            f->setUniform({poison, poison, poison});
        for (auto *f : {&scratch.components.Ux, &scratch.components.Uy, &scratch.components.Uz})
            f->setUniform(poison);
        model.computeFields(mesh, fields.k, fields.omega, fields.U, .01, result.nuT, result.F1,
                            result.F2, result.Pk, result.CDkw, strain, scratch);
        ScalarField nu(mesh, "nu", .01);
        model.computeFields(mesh, fields.k, fields.omega, fields.U, nu, reference.nuT, reference.F1,
                            reference.F2, reference.Pk, reference.CDkw);
        same(result.nuT, reference.nuT);
        same(result.F1, reference.F1);
        same(result.F2, reference.F2);
        same(result.Pk, reference.Pk);
        same(result.CDkw, reference.CDkw);
        same(strain, strainRateMagnitude(computeVelocityGradients(fields.U)));
        require(retained == scratch.gradK.data().data());
        for (int fi = mesh.nInternalFaces(); fi < mesh.nFaces(); ++fi) {
            fields.U.bface(fi).x += .1;
            fields.k.bface(fi) += .01;
        }
    }
    const auto before = result.nuT;
    bool rejected = false;
    try {
        model.computeFields(mesh, fields.k, fields.omega, fields.U, .01, result.nuT, result.nuT,
                            result.F2, result.Pk, result.CDkw, strain, scratch);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    require(rejected);
    same(before, result.nuT);
    rejected = false;
    try {
        computeVelocityGradients(scratch.velocity.dudx, scratch.velocity, scratch.components);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    require(rejected);
    model.computeFields(mesh, fields.k, fields.omega, fields.U, .01, result.nuT, result.F1,
                        result.F2, result.Pk, result.CDkw, strain, scratch);
    mesh.computeGeometry();
    rejected = false;
    try {
        greenGaussGrad(fields.k, scratch.gradK);
    } catch (const std::logic_error &) {
        rejected = true;
    }
    require(rejected);
    std::cout << "poisoned gradient/SST workspace, boundary changes, alias and recovery passed\n";
}
