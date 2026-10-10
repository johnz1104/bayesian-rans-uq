// Installed-consumer fixture exporting fields and policy diagnostics.
#include "HillsProductionPolicy.hpp"
#include <BuildInfo.hpp>
#include "SIMPLESolver.hpp"
#include "ObservationOperator.hpp"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

static std::ofstream output(const std::string &path) {
    std::ofstream f(path);
    f << std::setprecision(17);
    if (!f)
        throw std::runtime_error("Cannot open " + path);
    return f;
}
static void vector(std::ostream &o, const Vec3 &v) {
    o << v.x << ',' << v.y << ',' << v.z;
}
static void saveFields(const Mesh &m, const FlowFields &f, const std::string &dir,
                       const std::string &prefix) {
    auto out = output(dir + "/" + prefix + "cells.csv");
    out << "x,y,z,volume,wall_distance,Ux,Uy,Uz,p,k,omega,nuT,F1,F2,Pk,CDkw\n";
    for (int i = 0; i < m.nCells(); ++i) {
        vector(out, m.cell(i).center);
        out << ',' << m.cell(i).volume << ',' << m.wallDistance()[i] << ',';
        vector(out, f.U[i]);
        out << ',' << f.p[i] << ',' << f.k[i] << ',' << f.omega[i] << ',' << f.nuT[i] << ','
            << f.F1[i] << ',' << f.F2[i] << ',' << f.Pk[i] << ',' << f.CDkw[i] << '\n';
    }
    auto bd = output(dir + "/" + prefix + "boundary.csv");
    bd << "face,Ux,Uy,Uz,p,k,omega,nuT,F1,F2,Pk,CDkw\n";
    for (int i = m.nInternalFaces(); i < m.nFaces(); ++i) {
        bd << i << ',';
        vector(bd, f.U.bface(i));
        bd << ',' << f.p.bface(i) << ',' << f.k.bface(i) << ',' << f.omega.bface(i) << ','
           << f.nuT.bface(i) << ',' << f.F1.bface(i) << ',' << f.F2.bface(i) << ',' << f.Pk.bface(i)
           << ',' << f.CDkw.bface(i) << '\n';
    }
}
int main(int argc, char **argv) {
    if (argc != 8) {
        std::cerr << "geometry output tolerance max_iter mode probes target\n";
        return 2;
    }
    const std::string dir = argv[2], mode = argv[5];
    std::ifstream in(argv[1]);
    int nx, ny;
    double top, re;
    if (!(in >> nx >> ny >> top >> re) || nx < 1 || ny < 1 || !std::isfinite(top) ||
        !std::isfinite(re) || re <= 0)
        throw std::runtime_error("invalid geometry header");
    std::vector<double> x(nx + 1), y(nx + 1);
    for (int i = 0; i <= nx; ++i)
        in >> x[i] >> y[i];
    if (!in)
        return 3;
    Mesh mesh = Mesh::makeCurvedChannelPeriodic2D(x, y, top, ny, re, 1.);
    mesh.computeWallDistance();
    auto bc = FlowBoundaryConditions::channelDefaults(mesh, 1., 1e-4, 10.);
    SSTCoefficients theta;
    SSTModel sst(theta);
    SolverSettings st;
    st.maxIterations = std::stoi(argv[4]);
    st.convergenceTol = std::stod(argv[3]);
    st.alphaU = .3;
    st.alphaP = .2;
    st.alphaK = .4;
    st.alphaOmega = .4;
    st.innerIterations = 300;
    st.innerTolerance = 1e-4;
    st.turbStartIter = 30;
    st.turbUpdateInterval = 2;
    st.nuTFloorIters = 500;
    st.divergenceLimit = 1e8;
    st.verbose = true;
    st.reportInterval = 1000;
    st.legacyConvergence = false;
    st.convergenceWindow = 25;
    st.legacyRhieChowGate = false;
    st.useWallFunctions = false;
    // This installed fixture reproduces the source-recorded historical map.
    st.legacyOmegaWallPinning = true;
    st.momentumConvection = "limitedLinearUpwind";
    st.turbulenceConvection = "limitedLinearUpwind";
    st.alphaInjection = .3;
    st.bodyForce = Vec3(.00906, 0., 0.);
    SIMPLESolver solver(mesh, sst, bc, 1. / re, st);
    FlowFields f(mesh);
    solver.initUniform(f, Vec3(0., 0., 0.), 0., 1e-4, 10.);
    std::vector<double> b(6 * mesh.nCells(), 0.);
    auto policy = std::make_shared<HillsTargetProduction>();
    if (mode == "zero" || mode == "target") {
        std::ifstream target(argv[7]);
        for (double &value : b)
            if (!(target >> value) || !std::isfinite(value))
                throw std::runtime_error("target needs exactly six finite entries per cell");
        std::string extra;
        if (target >> extra)
            throw std::runtime_error("extra target entries");
        solver.setTargetAnisotropy(&b);
        solver.setInjectionScale(mode == "zero" ? 0.0 : 1.0);
        solver.setProductionPolicy(policy);
        if (mode == "zero")
            solver.setSourceObserver([](const SIMPLESolver::SourceSnapshotView &view) {
                for (const auto *force : {&view.freshIntegratedForce, &view.appliedIntegratedForce})
                    for (const auto &value : *force)
                        if (value.x != 0 || value.y != 0 || value.z != 0)
                            throw std::runtime_error("zero control retained source force");
            });
    } else if (mode != "baseline")
        return 5;
    auto metadata = output(dir + "/adapter_identity.txt");
    metadata << "library_revision " << cfdr::sourceRevision << "\nlibrary_digest "
             << cfdr::sourceDigest << "\npolicy " << HillsTargetProduction::identity
             << "\npolicy_source_patch " << HillsTargetProduction::sourcePatchSha256 << "\nmode "
             << mode << "\nmesh " << nx << 'x' << ny << "\nnu " << 1. / re << "\ntolerance "
             << st.convergenceTol << "\nmax_iterations " << st.maxIterations << "\ndata_geometry "
             << argv[1] << "\ndata_probes " << argv[6] << "\ndata_target " << argv[7]
             << "\nomega_wall_treatment legacy_solve_then_pin"
             << "\nseed none\nprovisional installed-adapter-control\n";
    metadata.close();
    auto faces = output(dir + "/faces.csv");
    faces << "owner,neighbor,weight,area,delta,nx,ny,nz,x,y,z,sx,sy,sz,patch\n";
    for (int i = 0; i < mesh.nFaces(); ++i) {
        const auto &a = mesh.face(i);
        faces << a.owner << ',' << a.neighbor << ',' << a.weight << ',' << a.area << ',' << a.delta
              << ',';
        vector(faces, a.normal);
        faces << ',';
        vector(faces, a.center);
        faces << ',';
        vector(faces, a.shift);
        faces << ',' << a.patchID << '\n';
    }
    faces.close();
    auto nodes = output(dir + "/nodes.csv");
    nodes << "x,y,z\n";
    for (int i = 0; i < mesh.nNodes(); ++i) {
        vector(nodes, mesh.node(i));
        nodes << '\n';
    }
    nodes.close();
    auto patches = output(dir + "/patches.csv");
    patches << "id,name,type\n";
    for (int i = 0; i < mesh.nPatches(); ++i)
        patches << i << ',' << mesh.patch(i).name << ',' << mesh.patch(i).type << '\n';
    patches.close();
    saveFields(mesh, f, dir, "initial_");
    auto start = std::chrono::steady_clock::now();
    std::cout << std::unitbuf;
    const auto hist = solver.solve(f);
    const auto elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    // Capture actual flux before calling anything that could recompute closure.
    auto flux = output(dir + "/flux.csv");
    flux << "flux\n";
    for (double v : solver.faceFluxes())
        flux << v << '\n';
    flux.close();
    saveFields(mesh, f, dir, "");
    auto history = output(dir + "/history.csv");
    history << "iteration,Ux,Uy,Uz,p,k,omega,momentumScaled,continuityScaled,changeU,remainingU\n";
    for (const auto &r : hist.entries)
        history << r.iteration << ',' << r.Ux << ',' << r.Uy << ',' << r.Uz << ',' << r.p << ','
                << r.k << ',' << r.omega << ',' << r.momentumScaled << ',' << r.continuityScaled
                << ',' << r.changeU << ',' << r.remainingU << '\n';
    auto result = output(dir + "/native_result.txt");
    result << "converged " << hist.converged << "\ndiverged " << hist.diverged << "\nstalled "
           << hist.stalled << "\nfinalIter " << hist.finalIter << "\nfinalWorstMeasure "
           << hist.finalWorstMeasure << "\ncontractionRateU " << hist.contractionRateU
           << "\nseconds " << elapsed << "\nturbEstablished " << f.turbEstablished << "\n";
    const auto &d = solver.injectionDiagnostics();
    result << "target_active " << d.active << "\nall_target_realizable " << d.allRealizable
           << "\ntarget_max_violation " << d.maxViolation << "\n";
    history.close();
    result.close();
    // Velocity probes for comparison with independently sampled fields.
    std::ifstream pin(argv[6]);
    double px, py;
    ObservationOperator obs;
    while (pin >> px >> py)
        for (int c = 0; c < 3; ++c)
            obs.addVelocityProfile(Vec3(px, py, 0.), c, 0., 1.);
    auto ref = output(dir + "/probe_reference.csv");
    ref << "U\n";
    for (double v : obs.evaluate(mesh, f, 1. / re))
        ref << v << '\n';
    // Additive diagnostics consume the corrected saved flux; no reconstruction
    // replaces the flux used by the solved state or its compatible exports.
    const auto savedFlux = solver.faceFluxes();
    for (const auto modeClosure :
         {SIMPLESolver::ClosureMode::Stored, SIMPLESolver::ClosureMode::Recomputed}) {
        const auto diagnostic = solver.diagnoseMomentum(f, theta, savedFlux, modeClosure);
        auto out = output(dir + (modeClosure == SIMPLESolver::ClosureMode::Stored
                                     ? "/momentum_stored.csv"
                                     : "/momentum_recomputed.csv"));
        out << "Rux,Ruy,Ruz,fresh_force_x,fresh_force_y,fresh_force_z\n";
        for (int i = 0; i < mesh.nCells(); ++i) {
            vector(out, diagnostic.residual[i]);
            out << ',';
            vector(out, diagnostic.freshIntegratedForce.empty()
                            ? Vec3{}
                            : diagnostic.freshIntegratedForce[i]);
            out << '\n';
        }
    }
    if (solver.faceFluxes() != savedFlux)
        throw std::runtime_error("diagnostics mutated saved flux");
    return 0;
}
