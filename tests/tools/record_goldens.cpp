// Golden reference numbers: run a fixed set of cases with the current code and
// write one JSON file per case.
//
//   record_goldens <out_dir> <git_sha> <dirty 0|1> [case ...]
//
// With no case names every case runs. tests/tools/record_goldens.sh is the normal
// entry point: it builds the repro preset and fills in the code identity.
//
// The cases mirror the research configurations as closely as the C++ alone
// allows. Where a study took an input from DNS data (matched viscosity, probe
// stations), the derived scalar is written here with its derivation, so no DNS
// data is needed or copied. Two cases deliberately differ from their studies,
// and each says so in its "notes": the BFS uses the legacy uniform inlet
// rather than the DNS inlet profile, and the hills use the ERCOFTAC hill
// polynomial rather than the DNS blanking-mask curve.

#include "BoundaryCondition.hpp"
#include "DBNSSolver.hpp"
#include "ExactRiemann.hpp"
#include "Field.hpp"
#include "FlowFields.hpp"
#include "IdealGasEOS.hpp"
#include "InferenceParameters.hpp"
#include "Mesh.hpp"
#include "ObservationOperator.hpp"
#include "ParameterSensitivity.hpp"
#include "SIMPLESolver.hpp"
#include "SSTModel.hpp"
#include "GoldenWriter.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace cfdr::golden;

double seconds(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// ---- shared SIMPLE helpers ---------------------------------------------------

JObj settingsJson(const SolverSettings &s) {
    JObj o;
    o.integer("max_iterations", s.maxIterations)
        .num("convergence_tol", s.convergenceTol)
        .num("alpha_u", s.alphaU)
        .num("alpha_p", s.alphaP)
        .num("alpha_k", s.alphaK)
        .num("alpha_omega", s.alphaOmega)
        .integer("inner_iterations", s.innerIterations)
        .num("inner_tolerance", s.innerTolerance)
        .integer("turb_start_iter", s.turbStartIter)
        .integer("turb_update_interval", s.turbUpdateInterval)
        .boolean("legacy_rhie_chow_gate", s.legacyRhieChowGate)
        .num("body_force_x", s.bodyForce.x)
        .num("divergence_limit", s.divergenceLimit);
    return o;
}

std::string statusOf(const ConvergenceHistory &h) {
    if (h.diverged)
        return "diverged";
    if (h.converged)
        return "converged";
    return h.stalled ? "stalled" : "max_iter";
}

// The monitor plays the role the observation vector plays in ForwardModel:
// the solve must certify those quantities, not only the fields.
struct Monitor {
    SIMPLESolver::ConvergenceMonitor fn;
    std::vector<double> floors;
};

// a monitor that certifies an observation operator, as ForwardModel does
// (floors are the observation standard deviations)
Monitor observing(const Mesh &mesh, const ObservationOperator &obs, double nu) {
    Monitor m;
    m.fn = [&mesh, obs, nu](const FlowFields &f) { return obs.evaluate(mesh, f, nu); };
    for (const Observable &ob : obs.observables())
        m.floors.push_back(std::sqrt(ob.meas.totalVariance()));
    return m;
}

// Sums of squares of each solved field: a change anywhere in the solution moves
// these, so they back up the case-specific QoIs in the "no numerical change"
// comparison.
JObj fieldNorms(const Mesh &mesh, const FlowFields &f) {
    double ux = 0, uy = 0, p = 0, k = 0, w = 0, nut = 0;
    for (int ci = 0; ci < mesh.nCells(); ++ci) {
        ux += f.U[ci].x * f.U[ci].x;
        uy += f.U[ci].y * f.U[ci].y;
        p += f.p[ci] * f.p[ci];
        k += f.k[ci] * f.k[ci];
        w += f.omega[ci] * f.omega[ci];
        nut += f.nuT[ci] * f.nuT[ci];
    }
    JObj o;
    o.num("sumsq_ux", ux)
        .num("sumsq_uy", uy)
        .num("sumsq_p", p)
        .num("sumsq_k", k)
        .num("sumsq_omega", w)
        .num("sumsq_nut", nut);
    return o;
}

// Area-averaged molecular wall shear nu*|U_t|/delta over the faces of a patch
// whose centre satisfies the x filter.
double meanWallShear(const Mesh &mesh, const FlowFields &f, double nu, const std::string &patch,
                     const std::function<bool(double)> &keepX = nullptr) {
    const Patch &pat = mesh.patch(mesh.patchByName(patch));
    double tauA = 0.0, area = 0.0;
    for (FaceID fi : pat.faces) {
        const Face &fc = mesh.face(fi);
        if (keepX && !keepX(fc.center.x))
            continue;
        Vec3 du = f.U[fc.owner] - f.U.bface(fi);
        Vec3 dut = du - fc.normal * du.dot(fc.normal);
        tauA += nu * dut.norm() / std::max(fc.delta, 1e-20) * fc.area;
        area += fc.area;
    }
    return area > 0 ? tauA / area : 0.0;
}

double bulkVelocity(const Mesh &mesh, const FlowFields &f) {
    double num = 0.0, den = 0.0;
    for (int ci = 0; ci < mesh.nCells(); ++ci) {
        num += f.U[ci].x * mesh.cell(ci).volume;
        den += mesh.cell(ci).volume;
    }
    return num / den;
}

struct Solved {
    ConvergenceHistory hist;
    FlowFields fields;
    double wall = 0.0;
};

Solved runSimple(const Mesh &mesh, const FlowBoundaryConditions &bcs, double nu,
                 const SolverSettings &s, const Vec3 &U0, double k0, double w0,
                 Monitor monitor = {}) {
    SSTModel sst{SSTCoefficients{}};
    SIMPLESolver solver(mesh, sst, bcs, nu, s);
    if (monitor.fn)
        solver.setConvergenceMonitor(std::move(monitor.fn), std::move(monitor.floors));
    Solved r{ConvergenceHistory{}, FlowFields(mesh), 0.0};
    solver.initUniform(r.fields, U0, 0.0, k0, w0);
    auto t0 = std::chrono::steady_clock::now();
    r.hist = solver.solve(r.fields);
    r.wall = seconds(t0);
    return r;
}

JObj resultJson(const Mesh &mesh, const Solved &r) {
    JObj o;
    o.str("status", statusOf(r.hist))
        .integer("final_iter", r.hist.finalIter)
        .boolean("stalled", r.hist.stalled)
        .num("final_worst_measure", r.hist.finalWorstMeasure)
        .num("contraction_rate_u", r.hist.contractionRateU)
        .num("wall_seconds", r.wall)
        .num("odd_even_ratio_p", oddEvenEnergyRatio(mesh, r.fields.p))
        .obj("field_norms", fieldNorms(mesh, r.fields));
    return o;
}

// ---- probe stations derived from DNS (see the notes in each case) -----------

// Lee & Moser Re_tau 180 channel: 20 stations from
// geomspace(1, 0.99*182.088, 20) in y+, snapped to DNS points, as y/delta.
const std::vector<double> kChannelStations = {
    0.0054122475544012838, 0.007226453384460596, 0.0093212456328103377, 0.01169603225078431,
    0.01728282491824162,   0.0204932519790928,   0.02774363101539001,   0.036093083078929437,
    0.050657039215853072,  0.061710189121104042, 0.086984823115415555,  0.10865626277918521,
    0.14978293699234291,   0.1969136123310693,   0.2497154648141782,    0.33244849854053271,
    0.43823518366344799,   0.56927062692230235,  0.75874318696007337,   0.99159793016503039};

// Pirozzoli et al. Couette Re_tau 171: 18 stations from
// geomspace(1, 0.95*171, 18) in y+, snapped to DNS points, as y/h.
const std::vector<double> kCouetteStations = {0.0060595270000000003,
                                              0.0082881450000000002,
                                              0.010915940000000001,
                                              0.014000540000000001,
                                              0.018934969999999999,
                                              0.026660920000000001,
                                              0.03431588,
                                              0.048740029999999997,
                                              0.063940170000000005,
                                              0.086693759999999995,
                                              0.11515359999999999,
                                              0.15664819999999999,
                                              0.2164153,
                                              0.29060780000000003,
                                              0.39210030000000001,
                                              0.52665810000000002,
                                              0.71408780000000005,
                                              0.9557407};

// ---- cases -------------------------------------------------------------------

// The flat periodic channel of test_periodic_channel.cpp (block 2).
void casePeriodicChannel(const RunInfo &ri) {
    const int nx = 24, ny = 28;
    const double Lx = 3.0, H = 1.0, nu = 2.0e-4, fb = 5.0e-3;
    std::vector<double> xN(nx + 1), yB(nx + 1, 0.0);
    for (int i = 0; i <= nx; ++i)
        xN[i] = Lx * i / nx;
    Mesh mesh = Mesh::makeCurvedChannelPeriodic2D(xN, yB, H, ny, 2500.0, 1.0);
    mesh.computeWallDistance();
    const double kIn = 1e-4, omIn = 10.0;
    FlowBoundaryConditions bcs = FlowBoundaryConditions::channelDefaults(mesh, 1.0, kIn, omIn);
    SolverSettings s;
    s.maxIterations = 12000;
    s.convergenceTol = 1e-4;
    s.alphaU = 0.5;
    s.alphaP = 0.3;
    s.verbose = false;
    s.bodyForce = Vec3(fb, 0.0, 0.0);
    Solved r =
        runSimple(mesh, bcs, nu, s, Vec3(0.3, 0.0, 0.0), kIn, omIn,
                  Monitor{[&mesh, nu](const FlowFields &g) {
                              return std::vector<double>{meanWallShear(mesh, g, nu, "bottom_wall"),
                                                         meanWallShear(mesh, g, nu, "top_wall"),
                                                         bulkVelocity(mesh, g)};
                          },
                          {}});

    double ub = bulkVelocity(mesh, r.fields);
    double tauB = meanWallShear(mesh, r.fields, nu, "bottom_wall");
    double tauT = meanWallShear(mesh, r.fields, nu, "top_wall");
    double drive = fb * Lx * H;        // f V (dz = 1)
    double shear = (tauB + tauT) * Lx; // integrated wall shear
    double uTau = std::sqrt(0.5 * (tauB + tauT));
    std::vector<double> col0(ny);
    for (int j = 0; j < ny; ++j)
        col0[j] = r.fields.U[j * nx].x;

    JObj mj;
    mj.str("generator", "makeCurvedChannelPeriodic2D (flat)")
        .integer("nx", nx)
        .integer("ny", ny)
        .num("Lx", Lx)
        .num("H", H)
        .num("Re_mesh", 2500.0)
        .num("yplus_target", 1.0);
    JObj q;
    q.num("bulk_velocity", ub)
        .num("tau_w_bottom", tauB)
        .num("tau_w_top", tauT)
        .num("u_tau", uTau)
        .num("re_tau_half_height", uTau * 0.5 * H / nu)
        .num("momentum_balance_shear_over_drive", shear / drive)
        .arr("u_column0_cells", col0);
    JObj body;
    body.num("nu", nu)
        .obj("mesh", mj)
        .obj("settings", settingsJson(s))
        .obj("result", resultJson(mesh, r))
        .obj("qoi", q)
        .str("notes", "Flat body-force periodic channel; init U=(0.3,0,0), k=1e-4, omega=10");
    writeCase(ri, "periodic_channel_flat", body);
}

// Moving-wall Couette with nu=0.0003060182410114865, Uwall=2, Ub=1 and Lx=1.5.
// The ny=384 case checks strict stopping on a refined mesh.
Solved couetteSolve(int ny, int maxIter, double tol, Mesh &meshOut, ObservationOperator &obsOut,
                    SolverSettings &sOut, double &nu) {
    nu = 0.0003060182410114865;
    const double Uwall = 2.0, Ub = 1.0, Lx = 1.5;
    meshOut = Mesh::makeChannel2D(4, ny, Lx, 2.0, Uwall / nu, 0.3);
    meshOut.computeWallDistance();
    const double kIn = 1.5 * (Ub * 0.05) * (Ub * 0.05);
    const double omIn = kIn / (nu * 100.0);
    FlowBoundaryConditions bcs = FlowBoundaryConditions::couetteDefaults(meshOut, Uwall, kIn, omIn);
    obsOut = ObservationOperator();
    // Cf observed value 2/(Ub+)^2 is irrelevant to the prediction; any value works
    obsOut.addSkinFriction("bottom", Vec3(0.5 * Lx, 0.0, 0.0), 6.0e-3, 1.0e-3, Ub);
    for (double yh : kCouetteStations)
        obsOut.addVelocityProfile(Vec3(0.5 * Lx, yh, 0.0), 0, 0.0, 0.01);
    sOut = SolverSettings();
    sOut.maxIterations = maxIter;
    sOut.convergenceTol = tol;
    sOut.alphaU = 0.7;
    sOut.alphaP = 0.5;
    sOut.verbose = false;
    return runSimple(meshOut, bcs, nu, sOut, Vec3(Ub, 0.0, 0.0), kIn, omIn,
                     observing(meshOut, obsOut, nu));
}

void caseCouette(const RunInfo &ri, int ny, int maxIter) {
    Mesh mesh;
    ObservationOperator obs;
    SolverSettings s;
    double nu = 0.0;
    Solved r = couetteSolve(ny, maxIter, 1e-4, mesh, obs, s, nu);
    double tauB = meanWallShear(mesh, r.fields, nu, "bottom");
    double tauT = meanWallShear(mesh, r.fields, nu, "top");
    std::vector<double> pred = obs.evaluate(mesh, r.fields, nu);

    JObj mj;
    mj.str("generator", "makeChannel2D")
        .integer("nx", 4)
        .integer("ny", ny)
        .num("Lx", 1.5)
        .num("Ly", 2.0)
        .num("Re_mesh", 2.0 / nu)
        .num("yplus_target", 0.3);
    JObj q;
    q.num("tau_w_bottom", tauB)
        .num("tau_w_top", tauT)
        .num("re_tau_bottom", std::sqrt(tauB) / nu)
        .num("re_tau_top", std::sqrt(tauT) / nu)
        .num("cf_probe", pred[0])
        .arr("probe_y_over_h", kCouetteStations)
        .arr("u_probes", std::vector<double>(pred.begin() + 1, pred.end()));
    JObj body;
    body.num("nu", nu)
        .obj("mesh", mj)
        .obj("settings", settingsJson(s))
        .obj("result", resultJson(mesh, r))
        .obj("qoi", q)
        .str("notes", "Moving-wall Couette with varied ny; Uwall=2, kIn=1.5*(0.05)^2, "
                      "omIn=kIn/(100 nu), nu=0.0003060182410114865");
    writeCase(ri, "couette_re171_ny" + std::to_string(ny), body);
}

// Developing channel with nu = 1/(Ub+ Re_tau). Ub+ = 15.6683865799 comes
// from trapezoidal integration of the Lee & Moser mean profile at Re_tau=182.088.
void caseDevelopingChannel(const RunInfo &ri) {
    const double ubPlus = 15.6683865799, reTau = 182.088;
    const double reB = ubPlus * reTau, nu = 1.0 / reB;
    const int nx = 40, ny = 56;
    const double Lx = 18.0, Ub = 1.0, xProbe = 0.9 * Lx;
    Mesh mesh = Mesh::makeChannel2D(nx, ny, Lx, 2.0, reB, 0.5);
    mesh.computeWallDistance();
    const double kIn = 1.5 * (Ub * 0.05) * (Ub * 0.05), omIn = kIn / (nu * 100.0);
    FlowBoundaryConditions bcs = FlowBoundaryConditions::channelDefaults(mesh, Ub, kIn, omIn);
    ObservationOperator obs;
    obs.addSkinFriction("bottom", Vec3(xProbe, 0.0, 0.0), 8.0e-3, 4.0e-4, 1.0);
    for (double yd : kChannelStations)
        obs.addVelocityProfile(Vec3(xProbe, yd, 0.0), 0, 0.0, 0.005);
    SolverSettings s;
    s.maxIterations = 45000;
    s.convergenceTol = 1e-3;
    s.alphaU = 0.7;
    s.alphaP = 0.5;
    s.verbose = false;
    Solved r = runSimple(mesh, bcs, nu, s, Vec3(Ub, 0.0, 0.0), kIn, omIn, observing(mesh, obs, nu));
    std::vector<double> pred = obs.evaluate(mesh, r.fields, nu);

    // developed-state diagnostics along the bottom wall
    double tauOut =
        meanWallShear(mesh, r.fields, nu, "bottom", [&](double x) { return x > 0.85 * Lx; });
    std::vector<double> xs, cfs;
    const Patch &pat = mesh.patch(mesh.patchByName("bottom"));
    for (FaceID fi : pat.faces) {
        const Face &fc = mesh.face(fi);
        double tau = nu * std::fabs(r.fields.U[fc.owner].x) / std::max(fc.delta, 1e-20);
        xs.push_back(fc.center.x);
        cfs.push_back(tau / (0.5 * Ub * Ub));
    }
    double dlnCf = 0.0;
    for (size_t i = 1; i < xs.size(); ++i)
        if (xs[i - 1] <= xProbe && xs[i] > xProbe)
            dlnCf = (std::log(cfs[i]) - std::log(cfs[i - 1])) / (xs[i] - xs[i - 1]);

    JObj mj;
    mj.str("generator", "makeChannel2D")
        .integer("nx", nx)
        .integer("ny", ny)
        .num("Lx", Lx)
        .num("Ly", 2.0)
        .num("Re_mesh", reB)
        .num("yplus_target", 0.5);
    JObj q;
    q.num("cf_probe", pred[0])
        .num("re_tau_outlet_region", std::sqrt(tauOut) / nu)
        .num("dlnCf_dx_at_probe", dlnCf)
        .arr("probe_y_over_delta", kChannelStations)
        .arr("u_probes", std::vector<double>(pred.begin() + 1, pred.end()))
        .arr("cf_bottom_x", xs)
        .arr("cf_bottom", cfs);
    JObj body;
    body.num("nu", nu)
        .obj("mesh", mj)
        .obj("settings", settingsJson(s))
        .obj("result", resultJson(mesh, r))
        .obj("qoi", q)
        .str("notes", "Developing channel: 40x56, Lx=18, tolerance=1e-3, iteration cap=45000; "
                      "probes at x=0.9 Lx; Re_tau from bottom shear for x > 0.85 Lx");
    writeCase(ri, "developing_channel_re180", body);
}

// The canonical channel (makePeriodicChannel2D) at the Lee & Moser Re_tau
// 182.088, at the resolution and probe stations of the developing-channel
// case above (ny = 56 over 2h, y+ 0.5), so the two compare directly. u_tau = 1
// units: nu = h / Re_tau, body force 1/h; probes report U+ (= U / u_tau).
void casePeriodicChannelRe180(const RunInfo &ri) {
    const double reTau = 182.088, h = 1.0, nu = h / reTau;
    const int nx = 4, ny = 56;
    Mesh mesh = Mesh::makePeriodicChannel2D(nx, ny, 1.0, h, reTau, 0.5);
    mesh.computeWallDistance();
    const double kIn = 1.0, omIn = 10.0;
    FlowBoundaryConditions bcs = FlowBoundaryConditions::channelDefaults(mesh, 1.0, kIn, omIn);
    ObservationOperator obs;
    obs.addSkinFriction("bottom", Vec3(0.5, 0.0, 0.0), 2.0, 0.1, 1.0);
    for (double yd : kChannelStations)
        obs.addVelocityProfile(Vec3(0.5, yd * h, 0.0), 0, 0.0, 0.1);
    SolverSettings s;
    s.maxIterations = 45000;
    s.convergenceTol = 1e-4;
    s.alphaU = 0.5;
    s.alphaP = 0.3;
    s.verbose = false;
    s.bodyForce = Vec3(1.0 / h, 0.0, 0.0);
    Solved r =
        runSimple(mesh, bcs, nu, s, Vec3(15.0, 0.0, 0.0), kIn, omIn, observing(mesh, obs, nu));
    std::vector<double> pred = obs.evaluate(mesh, r.fields, nu);
    double tb = meanWallShear(mesh, r.fields, nu, "bottom"),
           tt = meanWallShear(mesh, r.fields, nu, "top");
    double ub = bulkVelocity(mesh, r.fields);

    JObj mj;
    mj.str("generator", "makePeriodicChannel2D")
        .integer("nx", nx)
        .integer("ny", ny)
        .num("Lx", 1.0)
        .num("h", h)
        .num("Re_tau", reTau)
        .num("yplus_target", 0.5);
    JObj q;
    q.num("tau_w_bottom", tb)
        .num("tau_w_top", tt)
        .num("re_tau_bottom", std::sqrt(tb) * h / nu)
        .num("bulk_velocity_plus", ub)
        .num("cf_bulk", 2.0 * tb / (ub * ub))
        .arr("probe_y_over_delta", kChannelStations)
        .arr("u_plus_probes", std::vector<double>(pred.begin() + 1, pred.end()));
    JObj body;
    body.num("nu", nu)
        .obj("mesh", mj)
        .obj("settings", settingsJson(s))
        .obj("result", resultJson(mesh, r))
        .obj("qoi", q)
        .str("notes", "canonical periodic channel, u_tau = 1 units; compare re_tau_bottom and "
                      "cf_bulk with developing_channel_re180 (same ny, y+, stations)");
    writeCase(ri, "periodic_channel_re180", body);
}

// Le-Moin-Kim BFS at Re_h=5100 with a slip top and uniform inlet.
// The uniform inlet is not matched to a measured-profile validation case.
void caseBFS(const RunInfo &ri) {
    const double h = 1.0, H = 6.0, Re = 5100.0, Ub = 1.0, nu = Ub * h / Re;
    Mesh mesh = Mesh::makeBackwardFacingStep2D(40, 48, 24, 18, 10.0, 22.0, h, H, Re, 1.0);
    mesh.setPatchType("top_wall", "symmetry");
    mesh.computeWallDistance();
    const double kIn = 1.5 * (Ub * 0.05) * (Ub * 0.05), omIn = kIn / (nu * 100.0);
    FlowBoundaryConditions bcs = FlowBoundaryConditions::bfsDefaults(mesh, Ub, kIn, omIn);
    ObservationOperator obs;
    obs.addReattachmentLength("bottom_wall_down", 6.28, 0.5);
    const std::vector<double> tapX = {1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0};
    for (double x : tapX)
        obs.addPressureTap(Vec3(x, 0.0, 0.0), 0.0, 0.01, Ub);
    for (double x : tapX)
        obs.addSkinFriction("bottom_wall_down", Vec3(x, 0.0, 0.0), 0.0, 1e-3, Ub);
    const std::vector<double> profY = {0.05, 0.25, 0.5, 1.0, 2.0, 4.0};
    for (double y : profY)
        obs.addVelocityProfile(Vec3(4.0, y, 0.0), 0, 0.0, 0.01);
    SolverSettings s;
    s.maxIterations = 12000;
    s.convergenceTol = 1e-4;
    s.divergenceLimit = 1e8;
    s.alphaU = 0.3;
    s.alphaP = 0.2;
    s.alphaK = 0.4;
    s.alphaOmega = 0.4;
    s.innerIterations = 300;
    s.innerTolerance = 1e-4;
    s.turbStartIter = 30;
    s.turbUpdateInterval = 2;
    s.verbose = false;
    Solved r = runSimple(mesh, bcs, nu, s, Vec3(Ub, 0.0, 0.0), kIn, omIn, observing(mesh, obs, nu));
    std::vector<double> pred = obs.evaluate(mesh, r.fields, nu);

    JObj mj;
    mj.str("generator", "makeBackwardFacingStep2D")
        .integer("nx_up", 40)
        .integer("nx_down", 48)
        .integer("ny_up", 24)
        .integer("ny_down", 18)
        .num("Lu", 10.0)
        .num("Ld", 22.0)
        .num("h", h)
        .num("H", H)
        .num("Re_mesh", Re)
        .num("yplus_target", 1.0);
    JObj q;
    q.num("x_reattachment", pred[0])
        .arr("tap_x", tapX)
        .arr("cp_taps", std::vector<double>(pred.begin() + 1, pred.begin() + 1 + tapX.size()))
        .arr("cf_taps", std::vector<double>(pred.begin() + 1 + tapX.size(),
                                            pred.begin() + 1 + 2 * tapX.size()))
        .arr("profile_y_at_x4", profY)
        .arr("u_profile_x4", std::vector<double>(pred.begin() + 1 + 2 * tapX.size(), pred.end()));
    JObj body;
    body.num("nu", nu)
        .obj("mesh", mj)
        .obj("settings", settingsJson(s))
        .obj("result", resultJson(mesh, r))
        .obj("qoi", q)
        .str("notes", "Uniform-inlet BFS at Re_h=5100, Tu=5%, with a slip top; "
                      "not a measured-inlet-profile validation case");
    writeCase(ri, "bfs_lemoin_uniform_inlet", body);
}

// ERCOFTAC periodic-hill polynomial (UFR 3-30), x in mm on [0, 54], H = 28 mm.
double hillMM(double x) {
    if (x <= 9.0)
        return std::min(28.0, 28.0 + 6.775070969851e-3 * x * x - 2.124527775800e-3 * x * x * x);
    if (x <= 14.0)
        return 2.507355893131e+01 + 9.754803562315e-01 * x - 1.016116352781e-01 * x * x +
               1.889794677828e-03 * x * x * x;
    if (x <= 20.0)
        return 2.579601052357e+01 + 8.206693007457e-01 * x - 9.055370274339e-02 * x * x +
               1.626510569859e-03 * x * x * x;
    if (x <= 30.0)
        return 4.046435022819e+01 - 1.379581654948e+00 * x + 1.945884504128e-02 * x * x -
               2.070318932190e-04 * x * x * x;
    if (x <= 40.0)
        return 1.792461334664e+01 + 8.743920332081e-01 * x - 5.567361123058e-02 * x * x +
               6.277731764683e-04 * x * x * x;
    if (x <= 54.0)
        return std::max(0.0, 5.639011190988e+01 - 2.010520359035e+00 * x +
                                 1.644919857549e-02 * x * x + 2.674976141766e-05 * x * x * x);
    return 0.0;
}

// Hill surface in units of H on the periodic cell [0, 9]: one half hill at
// each end, crest at x = 0 and x = 9.
double hillBottom(double x) {
    double xmm = std::min(x, 9.0 - x) * 28.0;
    return hillMM(std::max(xmm, 0.0)) / 28.0;
}

void caseHills(const RunInfo &ri) {
    const int nx = 60, ny = 40;
    const double Lx = 9.0, yTop = 3.035, reB = 5600.0, nu = 1.0 / reB, fb = 0.00906;
    std::vector<double> xN(nx + 1), yB(nx + 1);
    for (int i = 0; i <= nx; ++i) {
        xN[i] = Lx * i / nx;
        yB[i] = hillBottom(xN[i]);
    }
    yB[nx] = yB[0];
    Mesh mesh = Mesh::makeCurvedChannelPeriodic2D(xN, yB, yTop, ny, reB, 1.0);
    mesh.computeWallDistance();
    const double kIn = 1e-4, omIn = 10.0;
    FlowBoundaryConditions bcs = FlowBoundaryConditions::channelDefaults(mesh, 1.0, kIn, omIn);
    ObservationOperator obs;
    obs.addSeparationPoint("bottom_wall", 0.2, 0.1);
    obs.addReattachmentLength("bottom_wall", 4.7, 1.0);
    const std::vector<double> profX = {0.5, 2.0, 4.0, 6.0, 8.0};
    for (double x : profX)
        for (double dy : {0.1, 0.5, 1.0, 2.0})
            obs.addVelocityProfile(Vec3(x, hillBottom(x) + dy, 0.0), 0, 0.0, 0.01);
    SolverSettings s;
    s.maxIterations = 25000;
    s.convergenceTol = 1e-4;
    s.divergenceLimit = 1e8;
    s.alphaU = 0.3;
    s.alphaP = 0.2;
    s.alphaK = 0.4;
    s.alphaOmega = 0.4;
    s.innerIterations = 300;
    s.innerTolerance = 1e-4;
    s.turbStartIter = 30;
    s.turbUpdateInterval = 2;
    s.verbose = false;
    s.bodyForce = Vec3(fb, 0.0, 0.0);
    Solved r =
        runSimple(mesh, bcs, nu, s, Vec3(0.0, 0.0, 0.0), kIn, omIn, observing(mesh, obs, nu));
    std::vector<double> pred = obs.evaluate(mesh, r.fields, nu);

    // bulk velocity through the crest column (x ~ 0): flow rate / gap
    double q0 = 0.0;
    for (int j = 0; j < ny; ++j) {
        int ci = j * nx;
        const Cell &c = mesh.cell(ci);
        q0 += r.fields.U[ci].x * c.volume / (Lx / nx);
    }
    JObj mj;
    mj.str("generator", "makeCurvedChannelPeriodic2D (ERCOFTAC hill polynomial)")
        .integer("nx", nx)
        .integer("ny", ny)
        .num("Lx", Lx)
        .num("yTop", yTop)
        .num("Re_mesh", reB)
        .num("yplus_target", 1.0);
    JObj q;
    q.num("x_separation", pred[0])
        .num("x_reattachment", pred[1])
        .num("bulk_velocity_crest_column", q0 / (yTop - yB[0]))
        .num("bulk_velocity_volume", bulkVelocity(mesh, r.fields))
        .arr("profile_x", profX)
        .arr("u_profiles_dy_0p1_0p5_1_2", std::vector<double>(pred.begin() + 2, pred.end()));
    JObj body;
    body.num("nu", nu)
        .obj("mesh", mj)
        .obj("settings", settingsJson(s))
        .obj("result", resultJson(mesh, r))
        .obj("qoi", q)
        .str("notes", "Periodic hills from rest using the ERCOFTAC polynomial; this geometry "
                      "differs from a DNS blanking-mask curve, so outputs are numerical "
                      "regressions rather than matched physical validation");
    writeCase(ri, "hills_ercoftac_60x40", body);
}

// Cosine-bump periodic channel of test_periodic_channel.cpp (block 3).
void caseCosineBump(const RunInfo &ri) {
    const int nx = 36, ny = 24;
    const double Lx = 4.5, yTop = 3.0, bump = 1.0, nu = 3.0e-4;
    std::vector<double> xN(nx + 1), yB(nx + 1);
    for (int i = 0; i <= nx; ++i) {
        xN[i] = Lx * i / nx;
        yB[i] = 0.5 * bump * (1.0 + std::cos(2.0 * M_PI * xN[i] / Lx));
    }
    Mesh mesh = Mesh::makeCurvedChannelPeriodic2D(xN, yB, yTop, ny, 0.0, 1.0);
    mesh.computeWallDistance();
    const double kIn = 1e-4, omIn = 10.0;
    FlowBoundaryConditions bcs = FlowBoundaryConditions::channelDefaults(mesh, 1.0, kIn, omIn);
    ObservationOperator obs;
    obs.addSeparationPoint("bottom_wall", 0.0, 0.1);
    obs.addReattachmentLength("bottom_wall", 0.0, 0.1);
    SolverSettings s;
    s.maxIterations = 15000;
    s.convergenceTol = 1e-4;
    s.alphaU = 0.4;
    s.alphaP = 0.25;
    s.verbose = false;
    s.bodyForce = Vec3(4.0e-3, 0.0, 0.0);
    Solved r =
        runSimple(mesh, bcs, nu, s, Vec3(0.3, 0.0, 0.0), kIn, omIn, observing(mesh, obs, nu));
    std::vector<double> pred = obs.evaluate(mesh, r.fields, nu);

    JObj mj;
    mj.str("generator", "makeCurvedChannelPeriodic2D (cosine bump)")
        .integer("nx", nx)
        .integer("ny", ny)
        .num("Lx", Lx)
        .num("yTop", yTop)
        .num("bump", bump);
    JObj q;
    q.num("bulk_velocity", bulkVelocity(mesh, r.fields))
        .num("x_separation", pred[0])
        .num("x_reattachment", pred[1]);
    JObj body;
    body.num("nu", nu)
        .obj("mesh", mj)
        .obj("settings", settingsJson(s))
        .obj("result", resultJson(mesh, r))
        .obj("qoi", q)
        .str("notes", "Periodic cosine bump; init U=(0.3,0,0); a capped solve is diagnostic only");
    writeCase(ri, "cosine_bump_periodic", body);
}

// ---- DBNS reference cases -------------------------------------------------

void caseDbnsShockTubes(const RunInfo &ri) {
    using namespace dbns;
    auto run = [](RiemannState1D L, RiemannState1D R, double tEnd, JObj &out) {
        IdealGasEOS eos;
        eos.gamma = 1.4;
        Mesh mesh = Mesh::makeChannel2D(200, 1, 1.0, 0.02);
        mesh.computeGeometry();
        SSTCoefficients sst;
        DBNSBoundaryConditions bcs;
        BoundarySpec ext;
        ext.kind = BoundaryKind::Extrapolate;
        BoundarySpec slip;
        slip.kind = BoundaryKind::SlipWall;
        bcs.set("inlet", ext);
        bcs.set("outlet", ext);
        bcs.set("top", slip);
        bcs.set("bottom", slip);
        DBNSSettings st;
        st.timeMode = TimeMode::Unsteady;
        st.tEnd = tEnd;
        st.cfl = 0.4;
        st.viscous = false;
        st.turbulent = false;
        st.reconstructOrder = 2;
        st.limitReconstruction = true;
        st.maxIterations = 100000;
        DBNSSolver solver(mesh, eos, sst, bcs, st);
        std::vector<Primitive> init(mesh.nCells());
        for (int ci = 0; ci < mesh.nCells(); ++ci) {
            const RiemannState1D &sv = (mesh.cell(ci).center.x < 0.5) ? L : R;
            init[ci] = {sv.rho, sv.u, 0.0, sv.p, 0.0, 0.0};
        }
        solver.initField(init);
        SolveReport rep = solver.solve();
        ExactRiemann ex(1.4);
        double er = 0, eu = 0, ep = 0;
        std::vector<double> rho(mesh.nCells());
        for (int ci = 0; ci < mesh.nCells(); ++ci) {
            double x = mesh.cell(ci).center.x;
            RiemannState1D e = ex.sample(L, R, (x - 0.5) / tEnd);
            Primitive V = solver.primitive(ci);
            er += std::fabs(V.rho - e.rho);
            eu += std::fabs(V.u - e.u);
            ep += std::fabs(V.p - e.p);
            rho[ci] = V.rho;
        }
        int n = mesh.nCells();
        out.str("status", statusString(rep.status))
            .integer("iterations", rep.iterations)
            .num("l1_rho", er / n)
            .num("l1_u", eu / n)
            .num("l1_p", ep / n)
            .arr("rho", rho);
    };
    JObj sod, lax;
    run({1.0, 0.0, 1.0}, {0.125, 0.0, 0.1}, 0.2, sod);
    run({0.445, 0.698, 3.528}, {0.5, 0.0, 0.571}, 0.13, lax);
    JObj body;
    body.obj("sod", sod)
        .obj("lax", lax)
        .str("notes", "Sod and Lax shock tubes: 200x1 strip, x0=0.5, CFL=0.4, "
                      "limited second-order MUSCL reconstruction");
    writeCase(ri, "dbns_shock_tubes", body);
}

void caseDbnsMms(const RunInfo &ri) {
    using namespace dbns;
    struct MMS {
        double rho0 = 1.2, p0 = 1.0e5, u0 = 60.0, v0 = 40.0, A = 0.2, kx, ky, gamma = 1.4;
        MMS() : kx(2.0 * M_PI), ky(2.0 * M_PI) {}
        Primitive exact(double x, double y) const {
            double sn = std::sin(kx * x + ky * y);
            return {rho0 * (1 + A * sn), u0, v0, p0 * (1 + A * sn), 0.0, 0.0};
        }
        StateVec source(double x, double y) const {
            double c = std::cos(kx * x + ky * y);
            double drdx = rho0 * A * kx * c, drdy = rho0 * A * ky * c;
            double dpdx = p0 * A * kx * c, dpdy = p0 * A * ky * c;
            double u = u0, v = v0, q2 = u * u + v * v, g = gamma / (gamma - 1.0);
            StateVec S{};
            S[I_RHO] = u * drdx + v * drdy;
            S[I_RHOU] = u * u * drdx + dpdx + u * v * drdy;
            S[I_RHOV] = u * v * drdx + v * v * drdy + dpdy;
            S[I_RHOE] = u * (g * dpdx + 0.5 * q2 * drdx) + v * (g * dpdy + 0.5 * q2 * drdy);
            S[I_RHOK] = 0.0;
            S[I_RHOW] = 0.0;
            return S;
        }
    };
    auto run = [](int n, int order) {
        IdealGasEOS eos;
        eos.gamma = 1.4;
        Mesh mesh = Mesh::makeChannel2D(n, n, 1.0, 1.0);
        mesh.computeGeometry();
        MMS mms;
        SSTCoefficients sst;
        DBNSBoundaryConditions bcs;
        BoundarySpec fixed;
        fixed.kind = BoundaryKind::FixedState;
        bcs.set("inlet", fixed);
        bcs.set("outlet", fixed);
        bcs.set("top", fixed);
        bcs.set("bottom", fixed);
        DBNSSettings st;
        st.timeMode = TimeMode::Steady;
        st.cfl = 0.6;
        st.viscous = false;
        st.turbulent = false;
        st.reconstructOrder = order;
        st.limitReconstruction = false;
        st.convergenceTol = 1e-9;
        st.maxIterations = 30000;
        DBNSSolver solver(mesh, eos, sst, bcs, st);
        std::vector<Primitive> init(mesh.nCells());
        std::vector<StateVec> src(mesh.nCells());
        for (int ci = 0; ci < mesh.nCells(); ++ci) {
            const Vec3 &c = mesh.cell(ci).center;
            init[ci] = mms.exact(c.x, c.y);
            src[ci] = mms.source(c.x, c.y);
        }
        solver.initField(init);
        solver.setManufacturedSource(src);
        int nIF = mesh.nInternalFaces();
        std::vector<Primitive> ghost(mesh.nFaces() - nIF);
        for (int fi = nIF; fi < mesh.nFaces(); ++fi)
            ghost[fi - nIF] = mms.exact(mesh.face(fi).center.x, mesh.face(fi).center.y);
        solver.setBoundaryOverride(ghost);
        std::vector<Primitive> pert = init;
        for (auto &pv : pert) {
            pv.rho *= 1.05;
            pv.p *= 0.97;
        }
        solver.initField(pert);
        solver.solve();
        // unweighted RMS, so the number does not depend on the stored cell volume
        double err = 0.0;
        for (int ci = 0; ci < mesh.nCells(); ++ci) {
            const Vec3 &c = mesh.cell(ci).center;
            double d = solver.primitive(ci).rho - mms.exact(c.x, c.y).rho;
            err += d * d;
        }
        return std::sqrt(err / mesh.nCells());
    };
    std::vector<double> e2 = {run(16, 2), run(32, 2), run(64, 2)};
    std::vector<double> e1 = {run(16, 1), run(32, 1)};
    JObj body;
    body.arr("rms_rho_error_order2_n16_32_64", e2)
        .num("observed_order2_32_64", std::log(e2[1] / e2[2]) / std::log(2.0))
        .arr("rms_rho_error_order1_n16_32", e1)
        .num("observed_order1_16_32", std::log(e1[0] / e1[1]) / std::log(2.0))
        .str("notes", "Steady inviscid Euler manufactured solution; density error is the "
                      "unweighted cellwise RMS");
    writeCase(ri, "dbns_mms", body);
}

// Fixed-state and tangent sensitivities on the production Couette case.
void caseSensitivity(const RunInfo &ri) {
    const double nu = 0.0003060182410114865, Uwall = 2.0, Ub = 1.0, Lx = 1.5;
    Mesh mesh = Mesh::makeChannel2D(4, 96, Lx, 2.0, Uwall / nu, 0.3);
    mesh.computeWallDistance();
    const double kIn = 1.5 * (Ub * 0.05) * (Ub * 0.05), omIn = kIn / (nu * 100.0);
    FlowBoundaryConditions bcs = FlowBoundaryConditions::couetteDefaults(mesh, Uwall, kIn, omIn);
    ObservationOperator obs;
    obs.addSkinFriction("bottom", Vec3(0.5 * Lx, 0.0, 0.0), 6.0e-3, 1.0e-3, Ub);
    for (double yh : kCouetteStations)
        obs.addVelocityProfile(Vec3(0.5 * Lx, yh, 0.0), 0, 0.0, 0.01);
    SolverSettings s;
    s.maxIterations = 12000;
    s.convergenceTol = 1e-4;
    s.alphaU = 0.7;
    s.alphaP = 0.5;
    s.verbose = false;
    ParameterSensitivity ps(mesh, obs, bcs, nu, s, Vec3(Ub, 0.0, 0.0), 0.0, kIn, omIn);
    // Keep this historical diagnostic reproducible without advertising the
    // incomplete frozen-pressure tangent as a supported research gradient.
    ps.setAllowExperimentalTangents(true);
    std::vector<double> theta = InferenceParameterSet::toVector(SSTCoefficients{});
    auto t0 = std::chrono::steady_clock::now();
    EvaluationStatus st = ps.solveState(theta);
    std::vector<double> eta = ps.observe(theta);
    std::vector<std::vector<double>> dG = ps.dObsDTheta(theta);
    TangentGradientResult tg = ps.etaJacobianTangent(theta);
    double wall = seconds(t0);
    std::vector<double> kIters(tg.krylovIters.begin(), tg.krylovIters.end());
    JObj body;
    body.num("nu", nu)
        .str("state_status", statusString(st))
        .num("wall_seconds", wall)
        .arr("theta11", theta)
        .arr("observations", eta)
        .mat("dobs_dtheta_fixed_state", dG)
        .mat("deta_dtheta_tangent", tg.dObsDTheta)
        .arr("tangent_krylov_iters", kIters)
        .arr("tangent_krylov_relres", tg.krylovRelRes)
        .str("notes",
             "Couette Re171 ny96 (production); observation operator: Cf + 18 velocity "
             "probes; tangent = explicitly experimental frozen-pressure etaJacobianTangent; "
             "not a verified full-response gradient");
    writeCase(ri, "sensitivity_couette_re171_ny96", body);
}

} // namespace

int main(int argc, char **argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s <out_dir> <git_sha> <dirty 0|1> [case ...]\n", argv[0]);
        return 2;
    }
    RunInfo ri{argv[1], argv[2], std::string(argv[3]) == "1"};

    const std::vector<std::pair<std::string, std::function<void()>>> cases = {
        {"periodic_channel_flat", [&] { casePeriodicChannel(ri); }},
        {"couette_re171_ny96", [&] { caseCouette(ri, 96, 12000); }},
        {"couette_re171_ny384", [&] { caseCouette(ri, 384, 40000); }},
        {"developing_channel_re180", [&] { caseDevelopingChannel(ri); }},
        {"periodic_channel_re180", [&] { casePeriodicChannelRe180(ri); }},
        {"bfs_lemoin_uniform_inlet", [&] { caseBFS(ri); }},
        {"hills_ercoftac_60x40", [&] { caseHills(ri); }},
        {"cosine_bump_periodic", [&] { caseCosineBump(ri); }},
        {"dbns_shock_tubes", [&] { caseDbnsShockTubes(ri); }},
        {"dbns_mms", [&] { caseDbnsMms(ri); }},
        {"sensitivity_couette_re171_ny96", [&] { caseSensitivity(ri); }},
    };

    std::vector<std::string> wanted(argv + 4, argv + argc);
    for (const auto &name : wanted) {
        const auto found = std::find_if(cases.begin(), cases.end(),
                                        [&](const auto &entry) { return entry.first == name; });
        if (found == cases.end()) {
            std::fprintf(stderr, "unknown case: %s\n", name.c_str());
            return 2;
        }
    }
    int ran = 0;
    for (const auto &[name, fn] : cases) {
        if (!wanted.empty() && std::find(wanted.begin(), wanted.end(), name) == wanted.end())
            continue;
        std::printf("case %s\n", name.c_str());
        std::fflush(stdout);
        try {
            fn();
        } catch (const std::exception &error) {
            std::fprintf(stderr, "failed to record %s: %s\n", name.c_str(), error.what());
            return 1;
        }
        ++ran;
    }
    if (ran == 0) {
        std::fprintf(stderr, "no case matched\n");
        return 2;
    }
    return 0;
}
