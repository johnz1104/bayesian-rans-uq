#include "SIMPLESolver.hpp"
#include "../apps/CLIMesh.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool condition, const char *message) {
    if (!condition)
        throw std::runtime_error(message);
}

// The cell-centred resolved-wall asymptote follows directly from
// nu*omega'' - beta1*omega^2 = 0. The value at a mathematical wall and
// the value at the first interior cell are different boundary approximations.
void wallRows(int ny) {
    Mesh mesh = Mesh::makePeriodicChannel2D(3, ny, 1, 1, 182.088);
    mesh.computeWallDistance();
    const double nu = 1 / 182.088;
    double skew = 0;
    for (const auto &face : mesh.faces())
        skew = std::max(skew, (face.d - face.normal * face.delta).norm() / face.delta);
    std::cout << "mesh normalized skew=" << skew << std::endl;
    auto bc = FlowBoundaryConditions::channelDefaults(mesh, 1, 1, 10);
    SolverSettings settings;
    settings.verbose = false;
    settings.momentumConvection = settings.turbulenceConvection = "upwind";
    SIMPLESolver solver(mesh, SSTModel{}, bc, nu, settings);
    FlowFields fields(mesh);
    solver.initUniform(fields, {1, 0, 0}, 0, 1, 10);
    for (int fi = mesh.nInternalFaces(); fi < mesh.nFaces(); ++fi) {
        const auto &face = mesh.face(fi);
        fields.omega[face.owner] = 6 * nu / (0.075 * face.delta * face.delta);
    }
    const auto residual = solver.steadyResidual(
        fields, SSTCoefficients{}, std::vector<double>(solver.freeFluxFaces().size(), 0));
    for (int fi = mesh.nInternalFaces(); fi < mesh.nFaces(); ++fi) {
        const int cell = mesh.face(fi).owner;
        const double value = residual.equations[3 * mesh.nCells() + cell];
        if (std::abs(value) >= 1e-10 * fields.omega[cell])
            std::cerr << "wall row=" << cell << " residual=" << value
                      << " omega=" << fields.omega[cell]
                      << " resolved=" << solver.resolvedWallOmega(mesh.face(fi).delta) << std::endl;
        require(std::isfinite(value) && std::abs(value) < 1e-10 * fields.omega[cell],
                "planar interior omega must use the differential-equation asymptote");
    }
}

// An analytic singular state makes the balancing correction observable without
// an expensive nonlinear solve. Compare to the continuous molecular Laplacian,
// retaining the actual face turbulent diffusion and pointwise SST sources.
void singularBalance(int ny) {
    Mesh mesh = Mesh::makePeriodicChannel2D(3, ny, 1, 1, 182.088);
    mesh.computeWallDistance();
    const double nu = 1 / 182.088, amplitude = 6 * nu / 0.075;
    auto bc = FlowBoundaryConditions::channelDefaults(mesh, 0, 0.001, 10);
    SolverSettings settings;
    settings.verbose = false;
    settings.alphaOmega = 1;
    settings.momentumConvection = settings.turbulenceConvection = "upwind";
    SSTModel model;
    SIMPLESolver solver(mesh, model, bc, nu, settings);
    FlowFields fields(mesh);
    solver.initUniform(fields, {}, 0, 0.001, 10);
    for (int cell = 0; cell < mesh.nCells(); ++cell) {
        const double y = mesh.cell(cell).center.y;
        fields.omega[cell] = amplitude * (1 / (y * y) + 1 / ((2 - y) * (2 - y)));
    }
    model.computeFields(mesh, fields.k, fields.omega, fields.U, nu, fields.nuT, fields.F1,
                        fields.F2, fields.Pk, fields.CDkw);
    const auto residual = solver.assembleResidual(fields, model.coeffs);
    std::vector<double> expected(mesh.nCells());
    for (int cell = 0; cell < mesh.nCells(); ++cell) {
        const double y = mesh.cell(cell).center.y, omega = fields.omega[cell];
        expected[cell] =
            mesh.cell(cell).volume * (6 * nu * amplitude * (std::pow(y, -4) + std::pow(2 - y, -4)) -
                                      model.coeffs.beta(fields.F1[cell]) * omega * omega +
                                      (1 - fields.F1[cell]) * fields.CDkw[cell]);
    }
    for (int fi = 0; fi < mesh.nInternalFaces(); ++fi) {
        const auto &face = mesh.face(fi);
        const double y = face.center.y;
        const double derivative = 2 * amplitude * (std::pow(2 - y, -3) - std::pow(y, -3));
        const double f1 =
            face.weight * fields.F1[face.owner] + (1 - face.weight) * fields.F1[face.neighbor];
        const double nut =
            face.weight * fields.nuT[face.owner] + (1 - face.weight) * fields.nuT[face.neighbor];
        const double flux = model.coeffs.sigma_w(f1) * nut * derivative * face.normal.y * face.area;
        expected[face.owner] += flux;
        expected[face.neighbor] -= flux;
    }
    std::vector<bool> wall(mesh.nCells(), false);
    for (int fi = mesh.nInternalFaces(); fi < mesh.nFaces(); ++fi)
        wall[mesh.face(fi).owner] = true;
    double worst = 0;
    for (int cell = 0; cell < mesh.nCells(); ++cell) {
        if (wall[cell])
            continue;
        const double scale =
            mesh.cell(cell).volume * 0.075 * fields.omega[cell] * fields.omega[cell];
        const double error = std::abs(residual[3 * mesh.nCells() + cell] - expected[cell]) / scale;
        require(std::isfinite(error), "nonfinite singular-balance error");
        worst = std::max(worst, error);
    }
    std::cout << "ny=" << ny << " normalized singular-balance defect=" << worst << '\n';
    require(worst < 2e-12, "singular molecular diffusion must balance its pointwise destruction");
}

void geometryScope() {
    const double nu = 0.001, distance = 0.01;
    auto check = [&](const Mesh &mesh, const SolverSettings &settings, double coefficient) {
        const auto bc = FlowBoundaryConditions::channelDefaults(mesh, 1, 0.01, 10);
        const SIMPLESolver solver(mesh, SSTModel{}, bc, nu, settings);
        const double expected = coefficient * nu / (0.075 * distance * distance);
        require(std::abs(solver.resolvedWallOmega(distance) / expected - 1) < 1e-14,
                "singular wall balance geometry/policy qualification failed");
    };
    SolverSettings settings;
    settings.verbose = false;
    Mesh planar = Mesh::makePeriodicChannel2D(3, 8, 1, 1, 180);
    check(planar, settings, 6);
    settings.useWallFunctions = true;
    check(planar, settings, 60);
    settings.useWallFunctions = false;
    settings.legacyOmegaWallPinning = true;
    check(planar, settings, 60);
    settings.legacyOmegaWallPinning = false;
    check(Mesh::makeChannel2D(3, 8, 1, 2), settings, 60);
    check(Mesh::makeCurvedChannelPeriodic2D({0, 0.25, 0.5, 0.75, 1}, {0, 0.1, 0.2, 0.1, 0}, 2, 8,
                                            180),
          settings, 60);
    // Isolate the orthogonality gate from the wall-plane gate. This synthetic
    // face fixture has unchanged parallel planes but a skew centre-to-centre
    // direction. No solve is performed on the altered geometric fixture.
    Mesh skew = planar;
    auto &face = const_cast<Face &>(skew.face(0));
    face.d = face.d + Vec3(-face.normal.y, face.normal.x, 0) * (0.1 * face.delta);
    check(skew, settings, 60);
    for (double invalid : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                           std::numeric_limits<double>::quiet_NaN()}) {
        const auto bc = FlowBoundaryConditions::channelDefaults(planar, 1, 0.01, 10);
        SIMPLESolver solver(planar, SSTModel{}, bc, nu, settings);
        bool rejected = false;
        try {
            (void)solver.resolvedWallOmega(invalid);
        } catch (const std::invalid_argument &) {
            rejected = true;
        }
        require(rejected, "resolved wall omega must reject invalid distances");
    }
}

// With zero velocity and k, the first segregated sweep leaves momentum and
// k unchanged. Its omega system is therefore the public fixed-state block.
// Independently enforce wall ROWS in a dense matrix (keeping the columns),
// then solve by pivoted elimination and compare every interior omega value.
// This tests the neighboring-equation effect, not merely post-solve pinning.
void constrainedOmegaOutcome() {
    Mesh mesh = Mesh::makePeriodicChannel2D(3, 8, 1, 1, 180, 20);
    mesh.computeWallDistance();
    const auto bc = FlowBoundaryConditions::channelDefaults(mesh, 0, 0, 10);
    SolverSettings settings;
    settings.verbose = false;
    settings.maxIterations = 1;
    settings.turbStartIter = settings.nuTFloorIters = 0;
    settings.kMin = 0;
    settings.useWallFunctions = true;
    settings.innerTolerance = 1e-13;
    settings.innerIterations = 400;
    SIMPLESolver solver(mesh, SSTModel{}, bc, 0.001, settings);
    FlowFields fields(mesh);
    solver.initUniform(fields, {}, 0, 0, 10);
    LinearSystem momentum, k, omega;
    std::vector<double> diagonal;
    solver.assemblePreconBlocks(fields, SSTCoefficients{}, momentum, k, omega, diagonal);
    const int count = mesh.nCells();
    std::vector<std::vector<double>> a(count, std::vector<double>(count + 1, 0));
    for (int row = 0; row < count; ++row) {
        a[row][row] = omega.diag[row];
        a[row][count] = omega.source[row];
    }
    for (int fi = 0; fi < mesh.nInternalFaces(); ++fi) {
        const auto &face = mesh.face(fi);
        a[face.owner][face.neighbor] += omega.upper[fi];
        a[face.neighbor][face.owner] += omega.lower[fi];
    }
    for (int fi = mesh.nInternalFaces(); fi < mesh.nFaces(); ++fi) {
        const auto &face = mesh.face(fi);
        auto &row = a[face.owner];
        std::fill(row.begin(), row.end(), 0);
        row[face.owner] = 1;
        row[count] = 60 * 0.001 / (0.075 * face.delta * face.delta);
    }
    for (int col = 0; col < count; ++col) {
        int pivot = col;
        for (int row = col + 1; row < count; ++row)
            if (std::abs(a[row][col]) > std::abs(a[pivot][col]))
                pivot = row;
        std::swap(a[pivot], a[col]);
        require(std::abs(a[col][col]) > 1e-14, "independent constrained matrix is singular");
        const double scale = a[col][col];
        for (int entry = col; entry <= count; ++entry)
            a[col][entry] /= scale;
        for (int row = 0; row < count; ++row) {
            if (row == col)
                continue;
            const double factor = a[row][col];
            for (int entry = col; entry <= count; ++entry)
                a[row][entry] -= factor * a[col][entry];
        }
    }
    const auto history = solver.solve(fields);
    require(!history.diverged, "zero-shear constrained omega solve diverged");
    for (int cell = 0; cell < count; ++cell)
        require(std::abs(fields.omega[cell] - a[cell][count]) <
                    1e-9 * (1 + std::abs(a[cell][count])),
                "constrained omega neighbors disagree with independent dense row enforcement");
}

void flatPlateBoundaryGeometry() {
    Mesh mesh = rans_cli::flatPlateMesh(3, 4, 1, 1, 100);
    mesh.computeWallDistance();
    const auto bc = FlowBoundaryConditions::flatPlateDefaults(mesh, 1, 0.001, 1);
    int top = -1;
    for (int patch = 0; patch < mesh.nPatches(); ++patch)
        if (mesh.patch(patch).name == "top")
            top = patch;
    require(top >= 0 && mesh.patch(top).type == "freestream",
            "flat-plate upper boundary must be freestream geometry before wall distance");
    SolverSettings settings;
    settings.verbose = false;
    settings.maxIterations = 1;
    settings.turbStartIter = 0;
    SIMPLESolver solver(mesh, SSTModel{}, bc, 0.01, settings);
    FlowFields fields(mesh);
    solver.initUniform(fields, {1, 0, 0}, 0, 0.001, 1);
    for (int fi : mesh.patch(top).faces) {
        const auto &face = mesh.face(fi);
        require(std::abs(mesh.wallDistance()[face.owner] - mesh.cell(face.owner).center.y) < 1e-12,
                "flat-plate wall distance must refer to the bottom plate only");
    }
    (void)solver.solve(fields);
    for (int fi : mesh.patch(top).faces) {
        const auto &face = mesh.face(fi);
        const double wallPin = 60 * 0.01 / (0.075 * face.delta * face.delta);
        require(std::abs(fields.omega[face.owner] / wallPin - 1) > 0.5,
                "flat-plate freestream owner must not receive a wall omega constraint");
    }
}

void constrainedWallFunctions() {
    Mesh mesh = Mesh::makePeriodicChannel2D(3, 8, 1, 1, 180, 20);
    mesh.computeWallDistance();
    const double nu = 0.001;
    const auto bc = FlowBoundaryConditions::channelDefaults(mesh, 1, 0.01, 10);
    std::vector<double> constrained;
    for (bool legacy : {false, true}) {
        SolverSettings settings;
        settings.verbose = false;
        settings.maxIterations = 1;
        settings.turbStartIter = settings.nuTFloorIters = 0;
        settings.useWallFunctions = true;
        settings.legacyOmegaWallPinning = legacy;
        settings.innerTolerance = 1e-12;
        SIMPLESolver solver(mesh, SSTModel{}, bc, nu, settings);
        FlowFields fields(mesh);
        solver.initUniform(fields, {1, 0, 0}, 0, 0.01, 10);
        const auto history = solver.solve(fields);
        require(!history.diverged, "wall-function branch unexpectedly diverged");
        std::vector<bool> wall(mesh.nCells(), false);
        for (int fi = mesh.nInternalFaces(); fi < mesh.nFaces(); ++fi) {
            const auto &face = mesh.face(fi);
            wall[face.owner] = true;
            const double resolved = 60 * nu / (0.075 * face.delta * face.delta);
            const double friction = std::sqrt(0.3 * fields.k[face.owner]);
            const double logarithmic = friction / (0.3 * settings.vonKarman * face.delta);
            const double expected = std::hypot(resolved, logarithmic);
            require(std::isfinite(fields.omega[face.owner]) &&
                        std::abs(fields.omega[face.owner] / expected - 1) < 1e-13,
                    "constrained wall-function omega must match the blended condition");
        }
        if (!legacy) {
            constrained = fields.omega.data();
        } else {
            double difference = 0;
            for (int cell = 0; cell < mesh.nCells(); ++cell)
                if (!wall[cell])
                    difference =
                        std::max(difference, std::abs(fields.omega[cell] - constrained[cell]));
            require(difference > 1e-8,
                    "wall-value column elimination must affect neighboring omega equations");
        }
    }
}
} // namespace

int main() {
    try {
        for (int ny : {16, 17, 32}) {
            wallRows(ny);
            singularBalance(ny);
        }
        geometryScope();
        constrainedWallFunctions();
        constrainedOmegaOutcome();
        flatPlateBoundaryGeometry();
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
