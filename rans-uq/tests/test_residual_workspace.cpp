#include "SIMPLESolver.hpp"
#include <iostream>
#include <limits>
static void require(bool b) {
    if (!b)
        throw std::runtime_error("residual workspace assertion failed");
}
int main() {
    auto mesh = Mesh::makeChannel2D(4, 5, 2, 1);
    mesh.computeWallDistance();
    auto bc = FlowBoundaryConditions::channelDefaults(mesh, 1, .02, 3);
    SolverSettings settings;
    settings.verbose = false;
    settings.momentumConvection = "linearUpwind";
    settings.turbulenceConvection = "linearUpwind";
    SIMPLESolver solver(mesh, SSTModel{}, bc, .01, settings);
    FlowFields fields(mesh);
    solver.initUniform(fields, {1, .02, 0}, 0, .02, 3);
    SIMPLESolver::DiagnosticWorkspace workspace(mesh), independent(mesh);
    SIMPLESolver::DiagnosticResidual output;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (int pass = 0; pass < 3; ++pass) {
        for (auto *values :
             {&workspace.diagonal, &workspace.values, &workspace.residual, &workspace.system.diag,
              &workspace.system.upper, &workspace.system.lower, &workspace.system.source})
            std::fill(values->begin(), values->end(), nan);
        workspace.strain.setUniform(nan);
        auto theta = SSTCoefficients{};
        theta.a1 += .001 * pass;
        const auto reference = solver.diagnosticResidual(fields, theta);
        solver.diagnosticResidual(fields, theta, output, workspace);
        require(reference.transport == output.transport &&
                reference.continuity == output.continuity &&
                reference.faceFluxes == output.faceFluxes);
        const auto first = output;
        solver.diagnosticResidual(fields, theta, output, independent);
        require(first.equations() == output.equations());
        for (int fi = mesh.nInternalFaces(); fi < mesh.nFaces(); ++fi) {
            fields.U.bface(fi).x += .01;
            fields.k.bface(fi) += .002;
        }
    }
    bool rejected = false;
    workspace.active = true;
    try {
        solver.diagnosticResidual(fields, SSTCoefficients{}, output, workspace);
    } catch (const std::logic_error &) {
        rejected = true;
    }
    require(rejected);
    workspace.active = false;
    rejected = false;
    try {
        solver.diagnosticResidual(workspace.fields, SSTCoefficients{}, output, workspace);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    require(rejected);
    fields.U[0].z = 1;
    rejected = false;
    try {
        solver.diagnosticResidual(fields, SSTCoefficients{}, output, workspace);
    } catch (const std::logic_error &) {
        rejected = true;
    }
    require(rejected);
    fields.U[0].z = 0;
    solver.diagnosticResidual(fields, SSTCoefficients{}, output, workspace);
    const auto ref = solver.diagnosticResidual(fields, SSTCoefficients{});
    require(ref.equations() == output.equations());
    std::cout
        << "diagnostic caller output, independent workspaces, poisoning and recovery passed\n";
}
