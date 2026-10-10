#include "Mesh.hpp"
#include "CLIMesh.hpp"
#include "Field.hpp"
#include "BoundaryCondition.hpp"
#include "LinearSolver.hpp"
#include "SSTModel.hpp"
#include "SIMPLESolver.hpp"
#include "InferenceParameters.hpp"
#include "ObservationOperator.hpp"
#include "ForwardModel.hpp"
#include <iostream>
#include <cmath>
#include <string>
#include <cstring>
#include <charconv>

// Developing-channel demonstration with a provisional empirical comparison.

static bool validateChannel(int maxIterations) {
    std::cout << "=== Channel Flow Validation (Re_b = 6800) ===\n\n";

    // Channel geometry = [0, Lx] x [0, 2h]
    // Re_b = U_b * h / nu  (bulk Reynolds number based on half-height)
    // Re_b = 6800, nu chosen so U_b = 1
    // h = half height, Lx = length of top/bottom wall
    // Dean correlation below uses the separate full-height Reynolds number, 2 U_b h / nu.

    double h = 1.0;       // half-height
    double Lx = 10.0 * h; // channel length (10h for developed flow)
    int nx = 60, ny = 40;
    double Ub = 1.0;
    double Re_b = 6800.0;
    double nu = Ub * h / Re_b; // nu = U_b * h / Re_b = 1.0 * 1.0 / 6800

    std::cout << "  nu = " << nu << "  (Re_b = " << Re_b << ")\n";
    std::cout << "  Mesh: " << nx << " x " << ny << "  domain: " << Lx << " x " << 2 * h << "\n";

    // build mesh (Re-adaptive wall clustering)
    Mesh mesh = Mesh::makeChannel2D(nx, ny, Lx, 2.0 * h, Re_b);
    mesh.computeWallDistance(); // SST model used wall distance in F_1 (blending function) and nu_t
                                // (turbulence viscosity)

    std::cout << "  Cells: " << mesh.nCells() << "  Faces: " << mesh.nFaces()
              << "  Internal: " << mesh.nInternalFaces() << "\n";

    // boundary conditions
    // inlet turbulence intensity estimate: Tu ~ 5% (ratio of RMS velocity fluctiation to mean
    // velocity - u'/U)
    double Tu = 0.05;
    double kIn = 1.5 * (Ub * Tu) * (Ub * Tu); // turbulence kinetic energy
    // Initialize omega from the prescribed inlet ratio k / (omega * nu) = 100.
    // This inlet choice does not establish a fully developed equilibrium.
    double omIn = kIn / (nu * 100.0);

    FlowBoundaryConditions bcs = FlowBoundaryConditions::channelDefaults(mesh, Ub, kIn, omIn);

    // SST model with default coefficients
    SSTModel sst;
    SolverSettings settings;
    settings.maxIterations = maxIterations;
    settings.convergenceTol = 1e-4;
    settings.reportInterval = 200;
    settings.innerIterations = 500;
    settings.innerTolerance = 1e-6;
    settings.verbose = true;
    // Standard relaxation
    settings.alphaU = 0.7;
    settings.alphaP = 0.5;
    settings.alphaK = 0.5;
    settings.alphaOmega = 0.5;
    // turbStartIter=0: couple turbulence from the first iteration so the velocity
    // field never builds unphysical laminar gradients.  A delayed start (e.g. 20)
    // lets pure-laminar SIMPLE create high-shear profiles that explode Pk the
    // moment SST activates, driving k to blow up within ~18 turbulence iterations.
    settings.turbStartIter = 0;
    settings.turbUpdateInterval = 1; // update every iteration (p' BC fix provides stability)

    // With `kMax = 10 * kIn = 0.0375`, k was being clamped in cells where it should have been
    // higher. Relaxed to `kMax = 0.1` Physical bounds: generous limits that prevent NaN but don't
    // interfere with the natural turbulence equilibrium. An overly tight kMax prevents k from
    // reaching its natural level, forcing all production energy into omega growth, which then
    // collapses nuT via the Bradshaw limiter.
    settings.kMax = 0.1;      // generous physical bound
    settings.omegaMin = 1e-6; // default

    // running SIMPLE solver
    SIMPLESolver solver(mesh, sst, bcs, nu, settings); // pressure-velocity coupling algorithm
    FlowFields fields(mesh);
    solver.initUniform(fields, Vec3(Ub, 0, 0), 0.0, kIn, omIn);

    std::cout << "\n  Running SIMPLE...\n";
    ConvergenceHistory hist = solver.solve(fields);

    // Reuse the resolved-wall observation: cell-centre nuT is not wall stress.
    ObservationOperator wallObservations;
    for (const auto &patch : mesh.patches())
        if (patch.type == "wall")
            for (int fi : patch.faces)
                if (mesh.face(fi).center.x >= 0.5 * Lx)
                    wallObservations.addSkinFriction(patch.name, mesh.face(fi).center, 0, 1, Ub);
    const auto wallCf = wallObservations.evaluate(mesh, fields, nu);
    if (wallCf.empty())
        throw std::runtime_error("channel demonstration has no downstream wall observations");
    double Cf_sim = 0;
    for (double value : wallCf)
        Cf_sim += value / static_cast<double>(wallCf.size());
    // Dean's Reynolds number uses the full channel height, 2h. Preserve the
    // existing simulation viscosity and the mesh's Reynolds input.
    const double Re_full_height = 2 * Ub * h / nu;
    const double Cf_dean = 0.073 * std::pow(Re_full_height, -0.25);

    std::cout << "\n  === Results ===\n";
    std::cout << "  SIMPLE iterations: " << hist.finalIter << " ("
              << (hist.converged  ? "converged"
                  : hist.diverged ? "DIVERGED"
                                  : "maxIter")
              << ")\n";
    std::cout << "  Cf (simulation):   " << Cf_sim << "\n";
    std::cout << "  Cf (Dean 1978):    " << Cf_dean << "\n";
    double error = std::abs(Cf_sim - Cf_dean) / Cf_dean * 100.0;
    std::cout << "  Relative error:    " << error << "%\n";

    std::cout << "  Dean full-height Reynolds number: " << Re_full_height << "\n";
    const bool agreement = std::isfinite(error) && error < 30.0;
    if (!hist.converged)
        std::cout << "  STATUS: NOT CONVERGED (comparison is diagnostic only)\n";
    else if (agreement)
        std::cout << "  STATUS: CONVERGED; within provisional 30% correlation band\n";
    else
        std::cout << "  STATUS: FAIL (outside provisional correlation band)\n";
    return hist.converged && agreement;
}

//  Flat plate validation — Schoenherr Cf(x)
//  Re_L = U_inf * L / nu = 1e5
//  Cf = 0.0592 * Re_x^(-0.2)   (turbulent Blasius / Schoenherr)
static bool validatePlate(int maxIterations) {
    std::cout << "\n=== Flat Plate Validation (Re_L = 2e6) ===\n\n";

    double L = 1.0;
    double Uinf = 1.0;
    // Re-adaptive mesh clustering now computes stretch automatically for y+ ~ 1
    double Re_L = 2e6;
    double nu = Uinf * L / Re_L;
    int nx = 60, ny = 40;

    std::cout << "  nu = " << nu << "  (Re_L = " << Re_L << ")\n";
    std::cout << "  Mesh: " << nx << " x " << ny << "\n";

    Mesh mesh = rans_cli::flatPlateMesh(nx, ny, L, 0.1 * L, Re_L);
    mesh.computeWallDistance();

    double Tu = 0.01;
    double kIn = 1.5 * (Uinf * Tu) * (Uinf * Tu);
    // nuT/nu ~ 100 at inlet (same reasoning as channel — with nuT/nu ~ 5 turbulence never develops)
    double omIn = kIn / (nu * 100.0);

    FlowBoundaryConditions bcs = FlowBoundaryConditions::flatPlateDefaults(mesh, Uinf, kIn, omIn);

    SSTModel sst;
    SolverSettings settings;
    settings.maxIterations = maxIterations;
    settings.convergenceTol = 1e-4;
    settings.reportInterval = 200;
    settings.innerIterations = 500;
    settings.innerTolerance = 1e-6;
    settings.verbose = true;
    // same relaxation and turbulence settings as channel validation
    settings.alphaU = 0.7;
    settings.alphaP = 0.5;
    settings.alphaK = 0.5;
    settings.alphaOmega = 0.5;
    settings.turbStartIter = 0; // couple turbulence from iter 0 (prevents laminar gradient buildup)
    settings.turbUpdateInterval = 1;
    settings.kMax = 0.1;
    settings.omegaMin = 1e-6;

    SIMPLESolver solver(mesh, sst, bcs, nu, settings);
    FlowFields fields(mesh);
    solver.initUniform(fields, Vec3(Uinf, 0, 0), 0.0, kIn, omIn);

    std::cout << "\n  Running SIMPLE...\n";
    ConvergenceHistory hist = solver.solve(fields);

    // skin friction at x = 0.8L (well past transition)
    double xProbe = 0.8 * L;
    double Re_x = Uinf * xProbe / nu;
    double Cf_schoenherr = 0.0592 * std::pow(Re_x, -0.2);

    // Use the same interpolated molecular-wall observation as library consumers.
    ObservationOperator wallObservation;
    wallObservation.addSkinFriction("bottom", {xProbe, 0, 0}, 0, 1, Uinf);
    const double Cf_sim = wallObservation.evaluate(mesh, fields, nu).front();

    std::cout << "\n  === Results at x/L = 0.8 ===\n";
    std::cout << "  SIMPLE iterations: " << hist.finalIter << " ("
              << (hist.converged  ? "converged"
                  : hist.diverged ? "DIVERGED"
                                  : "maxIter")
              << ")\n";
    std::cout << "  Re_x:              " << Re_x << "\n";
    std::cout << "  Cf (simulation):   " << Cf_sim << "\n";
    std::cout << "  Cf (Schoenherr):   " << Cf_schoenherr << "\n";
    double error = std::abs(Cf_sim - Cf_schoenherr) / Cf_schoenherr * 100.0;
    std::cout << "  Relative error:    " << error << "%\n";

    const bool agreement = std::isfinite(error) && error < 30.0;
    if (!hist.converged)
        std::cout << "  STATUS: NOT CONVERGED (comparison is diagnostic only)\n";
    else if (agreement)
        std::cout << "  STATUS: CONVERGED; within provisional 30% correlation band\n";
    else
        std::cout << "  STATUS: FAIL (outside provisional correlation band)\n";
    return hist.converged && agreement;
}

// Forward model Bayesian inference interface test
static bool demoForwardModel(int maxIterations) {
    std::cout << "\n=== Forward Model Demo (a1_betaStar preset) ===\n\n";

    double h = 1.0, Lx = 10.0 * h;
    int nx = 30, ny = 20;
    double Ub = 1.0, Re_b = 6800.0;
    double nu = Ub * h / Re_b;

    // Demo uses relaxed y+ target — coarse 30x20 mesh can't benefit from y+=1
    // and the extreme stretching (3.4) creates terrible aspect ratios that stall
    // pressure convergence.  y+=5 gives stretch~2.0, well-conditioned cells.
    Mesh mesh = Mesh::makeChannel2D(nx, ny, Lx, 2.0 * h, Re_b, 5.0);
    mesh.computeWallDistance();

    double Tu = 0.05;
    double kIn = 1.5 * (Ub * Tu) * (Ub * Tu);
    double omIn = kIn / (nu * 100.0); // nuT/nu ~ 100 at inlet

    FlowBoundaryConditions bcs = FlowBoundaryConditions::channelDefaults(mesh, Ub, kIn, omIn);

    // observation: skin friction on bottom wall
    ObservationOperator obsOp;
    obsOp.addSkinFriction("bottom", Vec3(7.0, 0, 0), 0.008, 0.001, Ub, 0.001);

    InferenceParameterSet params = InferenceParameterSet::a1_betaStar();
    SolverSettings settings;
    settings.maxIterations = maxIterations;
    settings.convergenceTol = 1e-4;
    settings.reportInterval = 500;
    settings.alphaP = 0.8; // aggressive pressure relaxation (stable on coarse demo mesh)
    settings.innerTolerance = 1e-6;
    settings.innerIterations = 500;
    settings.verbose = true;

    ForwardModel fm(mesh, params, obsOp, bcs, nu, settings, Vec3(Ub, 0, 0), 0.0, kIn, omIn);

    // evaluate at default parameters
    std::vector<double> theta = params.pack(SSTCoefficients{});
    std::cout << "  Active params: ";
    auto names = params.activeNames();
    for (int i = 0; i < params.nActive(); ++i)
        std::cout << names[i] << "=" << theta[i] << "  ";
    std::cout << "\n\n";

    EvaluationResult res = fm.evaluate(theta);
    std::cout << "\n  Status: " << statusString(res.status) << "\n";
    std::cout << "  Log-likelihood: " << res.loglik << "\n";
    std::cout << "  SIMPLE iters:   " << res.simpleIters << "\n";
    if (!res.predictions.empty())
        std::cout << "  Cf prediction:  " << res.predictions[0] << "\n";
    std::cout << "  Cache size:     " << fm.cache().size() << "\n";
    return res.status == EvaluationStatus::Converged;
}
static void usage(std::ostream &stream) {
    stream << "Usage: rans_sst MODE [--max-iterations N]\n"
           << "  --validate-channel   Developing-channel correlation demonstration\n"
           << "  --validate-plate     Flat-plate correlation demonstration\n"
           << "  --demo               Forward-model demonstration\n"
           << "  --all                Run all demonstrations\n"
           << "  --help               Show this help\n"
           << "N must be a nonnegative integer; zero requests a capped diagnostic.\n"
           << "Correlation comparisons are provisional, not grid-independent validation.\n";
}

int main(int argc, char *argv[]) {
    if (argc == 1 || (argc == 2 && std::string(argv[1]) == "--help")) {
        usage(std::cout);
        return 0;
    }
    const std::string mode = argv[1];
    int limit = -1;
    const bool knownMode = mode == "--validate-channel" || mode == "--validate-plate" ||
                           mode == "--demo" || mode == "--all";
    bool valid = knownMode && (argc == 2 || argc == 4);
    if (valid && argc == 4) {
        const char *text = argv[3];
        const char *end = text + std::strlen(text);
        const auto parsed = std::from_chars(text, end, limit);
        valid = std::string(argv[2]) == "--max-iterations" && text != end &&
                parsed.ec == std::errc{} && parsed.ptr == end && limit >= 0;
    }
    if (!valid) {
        std::cerr << "Invalid command arguments.\n";
        usage(std::cerr);
        return 2;
    }
    std::cout << "cfdr RANS demonstrations\n";
    try {
        bool success = true;
        if (mode == "--validate-channel" || mode == "--all")
            success = validateChannel(limit < 0 ? 5000 : limit) && success;
        if (mode == "--validate-plate" || mode == "--all")
            success = validatePlate(limit < 0 ? 5000 : limit) && success;
        if (mode == "--demo" || mode == "--all")
            success = demoForwardModel(limit < 0 ? 8000 : limit) && success;
        return success ? 0 : 1;
    } catch (const std::exception &error) {
        std::cerr << "CFD demonstration failed: " << error.what() << '\n';
        return 1;
    }
}
