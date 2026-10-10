#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <stdexcept>
#include <array>
#include "cfdr/BuildInfo.hpp"
#include "ForwardModel.hpp"
#include "ForwardBatch.hpp"
#include "ParameterSensitivity.hpp"
#include "CompressibleForwardModel.hpp"
#include "IdealGasEOS.hpp"
#include "CompressibleBCs.hpp"
#include "DBNSSolver.hpp"
#include "DBNSObservation.hpp"
#include "RealizabilityProjection.hpp"

namespace py = pybind11;

static void requirePreparedMesh(const Mesh &mesh) {
    if (mesh.nCells() <= 0 || !mesh.wallDistanceValid())
        throw py::value_error("compute_wall_distance() is required before constructing a solver");
}

// Extract the density-based solver state as a dict of numpy arrays.
static py::dict dbnsFields(const dbns::DBNSSolver &s) {
    int nc = s.nCells();
    py::array_t<double> rho(nc), u(nc), v(nc), p(nc), T(nc), mach(nc), muT(nc);
    py::array_t<double> k(nc), omega(nc);
    auto r = rho.mutable_unchecked<1>();
    auto uu = u.mutable_unchecked<1>();
    auto vv = v.mutable_unchecked<1>();
    auto pp = p.mutable_unchecked<1>();
    auto tt = T.mutable_unchecked<1>();
    auto mm = mach.mutable_unchecked<1>();
    auto mt = muT.mutable_unchecked<1>();
    auto kk = k.mutable_unchecked<1>();
    auto om = omega.mutable_unchecked<1>();
    const IdealGasEOS &eos = s.eos();
    for (int i = 0; i < nc; ++i) {
        dbns::Primitive V = s.primitive(i);
        double Tc = V.p / (V.rho * eos.R);
        double a = std::sqrt(eos.gamma * V.p / V.rho);
        r(i) = V.rho;
        uu(i) = V.u;
        vv(i) = V.v;
        pp(i) = V.p;
        tt(i) = Tc;
        mm(i) = std::sqrt(V.u * V.u + V.v * V.v) / a;
        mt(i) = s.eddyViscosity(i);
        kk(i) = V.k;
        om(i) = V.omega;
    }
    py::dict d;
    d["rho"] = rho;
    d["u"] = u;
    d["v"] = v;
    d["p"] = p;
    d["T"] = T;
    d["mach"] = mach;
    d["muT"] = muT;
    d["k"] = k;
    d["omega"] = omega;
    return d;
}

// Set the solver state from a numpy array of primitives, shape (n_cells, >=4)
// columns [rho, u, v, p] (k, omega optional as columns 4,5).
static void dbnsInitField(dbns::DBNSSolver &s, py::array_t<double> arr) {
    auto a = arr.unchecked<2>();
    int nc = s.nCells();
    if (a.shape(0) != nc || a.shape(1) < 4 || a.shape(1) > 6)
        throw py::value_error("init_field requires (n_cells, 4..6) primitives");
    std::vector<dbns::Primitive> f(nc);
    for (int i = 0; i < nc; ++i) {
        f[i].rho = a(i, 0);
        f[i].u = a(i, 1);
        f[i].v = a(i, 2);
        f[i].p = a(i, 3);
        f[i].k = a.shape(1) > 4 ? a(i, 4) : 0.0;
        f[i].omega = a.shape(1) > 5 ? a(i, 5) : 0.0;
    }
    s.initField(f);
}

// Wall observation record -> dict of numpy arrays.
static py::dict dbnsWallToDict(const dbns::WallRecord &w) {
    auto toArr = [](const std::vector<double> &v) {
        py::array_t<double> a((py::ssize_t)v.size());
        auto r = a.mutable_unchecked<1>();
        for (size_t i = 0; i < v.size(); ++i)
            r(i) = v[i];
        return a;
    };
    py::dict d;
    d["x"] = toArr(w.x);
    d["Cf"] = toArr(w.Cf);
    d["Cp"] = toArr(w.Cp);
    d["qw"] = toArr(w.qw);
    d["St"] = toArr(w.St);
    return d;
}

static py::dict dbnsWall(const dbns::DBNSObservation &obs, const std::string &patch,
                         double wallTemp) {
    dbns::WallRecord w = obs.wall(patch, wallTemp);
    return dbnsWallToDict(w);
}

static py::dict dbnsWallProfile(const dbns::DBNSObservation &obs, const std::string &patch,
                                py::array_t<double> wallTemps, double fallback) {
    auto a = wallTemps.unchecked<1>();
    std::vector<double> tw(static_cast<size_t>(a.shape(0)));
    for (py::ssize_t i = 0; i < a.shape(0); ++i)
        tw[i] = a(i);
    dbns::WallRecord w = obs.wallProfile(patch, tw, fallback);
    return dbnsWallToDict(w);
}

// Realizability projection of a Reynolds stress given its six components.
static std::vector<double> dbnsProjectStress(double xx, double yy, double zz, double xy, double xz,
                                             double yz) {
    dbns::Sym3 R{xx, yy, zz, xy, xz, yz};
    dbns::Sym3 P = dbns::RealizabilityProjection::projectReynoldsStress(R);
    return {P.xx, P.yy, P.zz, P.xy, P.xz, P.yz};
}

// Helper: extract cell centers from Mesh as numpy array (n_cells, 3)
static py::array_t<double> meshCellCenters(const Mesh &mesh) {
    int nc = mesh.nCells();
    py::array_t<double> arr({nc, 3});
    auto r = arr.mutable_unchecked<2>();
    for (int i = 0; i < nc; ++i) {
        const Vec3 &c = mesh.cell(i).center;
        r(i, 0) = c.x;
        r(i, 1) = c.y;
        r(i, 2) = c.z;
    }
    return arr;
}

// Helper: extract cell volumes from Mesh as numpy array (n_cells,)
static py::array_t<double> meshCellVolumes(const Mesh &mesh) {
    int nc = mesh.nCells();
    py::array_t<double> arr(nc);
    auto r = arr.mutable_unchecked<1>();
    for (int i = 0; i < nc; ++i)
        r(i) = mesh.cell(i).volume;
    return arr;
}

// Helper: extract node coordinates from Mesh as numpy array (n_nodes, 3)
static py::array_t<double> meshNodeCoords(const Mesh &mesh) {
    int nn = mesh.nNodes();
    py::array_t<double> arr({nn, 3});
    auto r = arr.mutable_unchecked<2>();
    for (int i = 0; i < nn; ++i) {
        const Vec3 &n = mesh.node(i);
        r(i, 0) = n.x;
        r(i, 1) = n.y;
        r(i, 2) = n.z;
    }
    return arr;
}

// Helper: extract face centers from Mesh as numpy array (n_faces, 3)
// Patch introspection helpers for wall-function diagnostics.
// We don't bind the full Patch / Face structs (too much surface area); we
// just expose enough per-wall-face data to compute y+ and Cf in Python.

static std::vector<std::string> meshPatchNames(const Mesh &mesh) {
    std::vector<std::string> out;
    out.reserve(mesh.nPatches());
    for (int pi = 0; pi < mesh.nPatches(); ++pi)
        out.push_back(mesh.patch(pi).name);
    return out;
}

static std::vector<std::string> meshPatchTypes(const Mesh &mesh) {
    std::vector<std::string> out;
    out.reserve(mesh.nPatches());
    for (int pi = 0; pi < mesh.nPatches(); ++pi)
        out.push_back(mesh.patch(pi).type);
    return out;
}

static py::dict meshWallPatchData(const Mesh &mesh, const std::string &name) {
    int patchIdx = -1;
    for (int pi = 0; pi < mesh.nPatches(); ++pi) {
        if (mesh.patch(pi).name == name) {
            patchIdx = pi;
            break;
        }
    }
    if (patchIdx < 0)
        throw std::runtime_error("Mesh::wall_patch_data: unknown patch '" + name + "'");
    const Patch &pat = mesh.patch(patchIdx);
    int nf = static_cast<int>(pat.faces.size());

    py::array_t<int> owner_arr(nf);
    py::array_t<double> delta_arr(nf);
    py::array_t<double> center_arr({nf, 3});
    py::array_t<double> normal_arr({nf, 3});
    py::array_t<double> area_arr(nf);
    auto own = owner_arr.mutable_unchecked<1>();
    auto del = delta_arr.mutable_unchecked<1>();
    auto cen = center_arr.mutable_unchecked<2>();
    auto nor = normal_arr.mutable_unchecked<2>();
    auto are = area_arr.mutable_unchecked<1>();

    for (int k = 0; k < nf; ++k) {
        const Face &face = mesh.face(pat.faces[k]);
        own(k) = face.owner;
        del(k) = face.delta;
        cen(k, 0) = face.center.x;
        cen(k, 1) = face.center.y;
        cen(k, 2) = face.center.z;
        nor(k, 0) = face.normal.x;
        nor(k, 1) = face.normal.y;
        nor(k, 2) = face.normal.z;
        are(k) = face.area;
    }

    py::dict d;
    d["name"] = pat.name;
    d["type"] = pat.type;
    d["n_faces"] = nf;
    d["owner"] = owner_arr;
    d["delta"] = delta_arr;
    d["center"] = center_arr;
    d["normal"] = normal_arr;
    d["area"] = area_arr;
    return d;
}

static py::array_t<double> meshFaceCenters(const Mesh &mesh) {
    int nf = mesh.nFaces();
    py::array_t<double> arr({nf, 3});
    auto r = arr.mutable_unchecked<2>();
    for (int i = 0; i < nf; ++i) {
        const Vec3 &c = mesh.face(i).center;
        r(i, 0) = c.x;
        r(i, 1) = c.y;
        r(i, 2) = c.z;
    }
    return arr;
}

// Helper: extract scalar field as numpy array (n_cells,)
static py::array_t<double> scalarToNumpy(const ScalarField &f) {
    int n = f.size();
    py::array_t<double> arr(n);
    auto r = arr.mutable_unchecked<1>();
    for (int i = 0; i < n; ++i)
        r(i) = f[i];
    return arr;
}

// Helper: extract vector field as numpy array (n_cells, 3)
static py::array_t<double> vectorToNumpy(const VectorField &f) {
    int n = f.size();
    py::array_t<double> arr({n, 3});
    auto r = arr.mutable_unchecked<2>();
    for (int i = 0; i < n; ++i) {
        r(i, 0) = f[i].x;
        r(i, 1) = f[i].y;
        r(i, 2) = f[i].z;
    }
    return arr;
}

// Helper: extract all fields from ForwardModel's last solution as a dict of numpy arrays
static py::dict extractFields(const ForwardModel &fm) {
    if (!fm.hasLastFields())
        throw std::runtime_error("No fields available — call evaluate() first");
    const FlowFields &ff = fm.lastFields();
    py::dict d;
    d["U"] = vectorToNumpy(ff.U);
    d["p"] = scalarToNumpy(ff.p);
    d["k"] = scalarToNumpy(ff.k);
    d["omega"] = scalarToNumpy(ff.omega);
    d["nuT"] = scalarToNumpy(ff.nuT);
    d["F1"] = scalarToNumpy(ff.F1);
    d["F2"] = scalarToNumpy(ff.F2);
    d["Pk"] = scalarToNumpy(ff.Pk);
    d["CDkw"] = scalarToNumpy(ff.CDkw);
    return d;
}

// Helper: extract fields from CompressibleForwardModel as a dict.
// Includes density and temperature on top of the incompressible field set so
// the existing visualization/observation tooling can reuse the keys.
static py::dict extractCompressibleFields(const CompressibleForwardModel &fm) {
    if (!fm.hasLastFields())
        throw std::runtime_error("No fields available — call evaluate() first");
    const CompressibleFlowFields &ff = fm.lastCompressibleFields();
    py::dict d;
    d["U"] = vectorToNumpy(ff.U);
    // the field stores the MECHANICAL working pressure; consumers get the
    // thermodynamic static pressure under "p" (the physical quantity every
    // diagnostic compares against) and the raw working variable separately
    ScalarField pThermo = ff.p;
    for (int ci = 0; ci < pThermo.mesh().nCells(); ++ci)
        pThermo[ci] = ff.p[ci] - (2.0 / 3.0) * ff.rho[ci] * std::max(ff.k[ci], 0.0);
    d["p"] = scalarToNumpy(pThermo);
    d["p_mechanical"] = scalarToNumpy(ff.p);
    d["T"] = scalarToNumpy(ff.T);
    d["rho"] = scalarToNumpy(ff.rho);
    d["k"] = scalarToNumpy(ff.k);
    d["omega"] = scalarToNumpy(ff.omega);
    d["nuT"] = scalarToNumpy(ff.nuT);
    d["F1"] = scalarToNumpy(ff.F1);
    d["F2"] = scalarToNumpy(ff.F2);
    d["Pk"] = scalarToNumpy(ff.Pk);
    d["CDkw"] = scalarToNumpy(ff.CDkw);
    return d;
}

// Every exported array owns its storage; callbacks cannot retain solver views.
static py::array_t<double> vectorValues(const std::vector<Vec3> &values) {
    py::array_t<double> result({static_cast<py::ssize_t>(values.size()), py::ssize_t(3)});
    auto a = result.mutable_unchecked<2>();
    for (size_t i = 0; i < values.size(); ++i) {
        a(i, 0) = values[i].x;
        a(i, 1) = values[i].y;
        a(i, 2) = values[i].z;
    }
    return result;
}
static py::array_t<double> scalarValues(const std::vector<double> &values) {
    py::array_t<double> result(values.size());
    std::copy(values.begin(), values.end(), result.mutable_data());
    return result;
}
template <class T> static std::vector<T> boundaryValues(const Field<T> &field) {
    std::vector<T> values(field.nBnd());
    for (int i = 0; i < field.nBnd(); ++i)
        values[i] = field.bnd(i);
    return values;
}
using ScalarMember = ScalarField FlowFields::*;
static const std::array<std::pair<const char *, ScalarMember>, 8> flowScalars{
    {{"p", &FlowFields::p},
     {"k", &FlowFields::k},
     {"omega", &FlowFields::omega},
     {"nuT", &FlowFields::nuT},
     {"F1", &FlowFields::F1},
     {"F2", &FlowFields::F2},
     {"Pk", &FlowFields::Pk},
     {"CDkw", &FlowFields::CDkw}}};
static py::dict completeFields(const FlowFields &fields) {
    py::dict result, boundary;
    result["U"] = vectorToNumpy(fields.U);
    boundary["U"] = vectorValues(boundaryValues(fields.U));
    for (const auto &[name, member] : flowScalars) {
        result[name] = scalarToNumpy(fields.*member);
        boundary[name] = scalarValues(boundaryValues(fields.*member));
    }
    result["boundary"] = boundary;
    result["turb_established"] = fields.turbEstablished;
    return result;
}
static void restoreFields(FlowFields &fields, const py::dict &data) {
    FlowFields next(fields.U.mesh());
    const auto boundary = data["boundary"].cast<py::dict>();
    auto copyVector = [](VectorField &f, py::handle value, bool bnd) {
        auto arr = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(value);
        if (!arr || arr.ndim() != 2 || arr.shape(0) != (bnd ? f.nBnd() : f.size()) ||
            arr.shape(1) != 3)
            throw py::value_error("U snapshot has an invalid shape");
        auto a = arr.unchecked<2>();
        for (py::ssize_t i = 0; i < a.shape(0); ++i) {
            for (int j = 0; j < 3; ++j)
                if (!std::isfinite(a(i, j)))
                    throw py::value_error("snapshot values must be finite");
            (bnd ? f.bnd(i) : f[i]) = Vec3(a(i, 0), a(i, 1), a(i, 2));
        }
    };
    copyVector(next.U, data["U"], false);
    copyVector(next.U, boundary["U"], true);
    for (const auto &[name, member] : flowScalars) {
        auto &f = next.*member;
        for (bool bnd : {false, true}) {
            auto arr = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(
                bnd ? boundary[name] : data[name]);
            if (!arr || arr.ndim() != 1 || arr.shape(0) != (bnd ? f.nBnd() : f.size()))
                throw py::value_error(std::string(name) + " snapshot has an invalid shape");
            auto a = arr.unchecked<1>();
            for (py::ssize_t i = 0; i < a.shape(0); ++i) {
                if (!std::isfinite(a(i)))
                    throw py::value_error("snapshot values must be finite");
                (bnd ? f.bnd(i) : f[i]) = a(i);
            }
        }
    }
    next.turbEstablished = data["turb_established"].cast<bool>();
    fields = std::move(next);
}

// Own the target vector that the legacy C++ SIMPLE interface borrows.
struct PythonSIMPLESolver {
    const Mesh &mesh;
    std::vector<double> target;
    bool solving = false;
    SIMPLESolver solver;
    PythonSIMPLESolver(const Mesh &m, const SSTCoefficients &coefficients,
                       const FlowBoundaryConditions &bcs, double nu, const SolverSettings &settings)
        : mesh(m), solver(m, SSTModel(coefficients, static_cast<SSTVariant>(settings.sstVariant)),
                          bcs, nu, settings) {
        requirePreparedMesh(m);
    }
    void validate(const FlowFields &fields) const {
        if (&fields.U.mesh() != &mesh)
            throw py::value_error("FlowFields must belong to this solver's mesh");
    }
    void setTarget(py::array_t<double, py::array::c_style | py::array::forcecast> values) {
        if (values.ndim() != 2 || values.shape(0) != mesh.nCells() || values.shape(1) != 6)
            throw py::value_error("target requires shape (n_cells, 6): xx yy zz xy xz yz");
        std::vector<double> next(values.data(), values.data() + values.size());
        for (double value : next)
            if (!std::isfinite(value))
                throw py::value_error("target must contain finite values");
        target = std::move(next);
        solver.setTargetAnisotropy(&target);
    }
};

static py::dict ownedSnapshot(const OwnedFlowSnapshot &f) {
    py::dict data, boundary;
    data["U"] = vectorValues(f.U);
    boundary["U"] = vectorValues(f.UBoundary);
#define CFDR_SNAPSHOT_SCALAR(name)                                                                 \
    data[#name] = scalarValues(f.name);                                                            \
    boundary[#name] = scalarValues(f.name##Boundary)
    CFDR_SNAPSHOT_SCALAR(p);
    CFDR_SNAPSHOT_SCALAR(k);
    CFDR_SNAPSHOT_SCALAR(omega);
    CFDR_SNAPSHOT_SCALAR(nuT);
    CFDR_SNAPSHOT_SCALAR(F1);
    CFDR_SNAPSHOT_SCALAR(F2);
    CFDR_SNAPSHOT_SCALAR(Pk);
    CFDR_SNAPSHOT_SCALAR(CDkw);
#undef CFDR_SNAPSHOT_SCALAR
    data["boundary"] = boundary;
    data["face_fluxes"] = scalarValues(f.faceFluxes);
    data["turb_established"] = f.turbEstablished;
    data["source_digest"] = cfdr::sourceDigest;
    return data;
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "RANS-SST Bayesian Calibration – C++ Forward Model";
    m.def("build_info", []() {
        py::dict d;
        d["version"] = cfdr::version;
        d["revision"] = cfdr::sourceRevision;
        d["source_digest"] = cfdr::sourceDigest;
        d["compiler"] = cfdr::compiler;
        d["compiler_flags"] = cfdr::compilerFlags;
        d["python_version"] = PY_VERSION;
        d["build_type"] = cfdr::buildType;
        return d;
    });
    // EvaluationStatus
    py::enum_<EvaluationStatus>(m, "EvaluationStatus")
        .value("Converged", EvaluationStatus::Converged)
        .value("Unconverged", EvaluationStatus::Unconverged)
        .value("DivergenceDetected", EvaluationStatus::Diverged)
        .value("InvalidParameters", EvaluationStatus::InvalidParameters)
        .value("Unknown", EvaluationStatus::Unknown);

    // EvaluationResult
    py::class_<EvaluationResult>(m, "EvaluationResult")
        .def_readonly("status", &EvaluationResult::status)
        .def_readonly("log_lik", &EvaluationResult::loglik)
        .def_readonly("predictions", &EvaluationResult::predictions)
        .def_readonly("simple_iters", &EvaluationResult::simpleIters);

    // SSTCoefficients
    py::class_<SSTCoefficients>(m, "SSTCoefficients")
        .def(py::init<>())
        .def_readwrite("sigma_k1", &SSTCoefficients::sigma_k1)
        .def_readwrite("sigma_w1", &SSTCoefficients::sigma_w1)
        .def_readwrite("beta1", &SSTCoefficients::beta1)
        .def_readwrite("alpha1", &SSTCoefficients::alpha1)
        .def_readwrite("sigma_k2", &SSTCoefficients::sigma_k2)
        .def_readwrite("sigma_w2", &SSTCoefficients::sigma_w2)
        .def_readwrite("beta2", &SSTCoefficients::beta2)
        .def_readwrite("alpha2", &SSTCoefficients::alpha2)
        .def_readwrite("betaStar", &SSTCoefficients::betaStar)
        .def_readwrite("a1", &SSTCoefficients::a1)
        .def_readwrite("kappa", &SSTCoefficients::kappa);

    // InferenceParameterSet
    py::class_<InferenceParameterSet>(m, "InferenceParameterSet")
        .def(py::init<>())
        .def_readonly("name", &InferenceParameterSet::name)
        .def("n_active", &InferenceParameterSet::nActive)
        .def("active_names", &InferenceParameterSet::activeNames)
        .def("pack", &InferenceParameterSet::pack)
        .def("unpack", &InferenceParameterSet::unpack)
        .def("in_bounds", &InferenceParameterSet::inBounds)
        .def("lower_bounds", &InferenceParameterSet::lowerBounds)
        .def("upper_bounds", &InferenceParameterSet::upperBounds)
        .def_static("a1_betaStar", &InferenceParameterSet::a1_betaStar)
        .def_static("inlet_turb", &InferenceParameterSet::inletTurb)
        .def_static("near_wall4", &InferenceParameterSet::nearWall4)
        .def_static("all11", &InferenceParameterSet::all11)
        .def_static("live10", &InferenceParameterSet::live10)
        .def_static("from_indices", &InferenceParameterSet::fromIndices, py::arg("name"),
                    py::arg("indices"));

    // Mesh
    py::class_<Mesh>(m, "Mesh")
        .def_static("make_periodic_channel_2d", &Mesh::makePeriodicChannel2D, py::arg("nx"),
                    py::arg("ny"), py::arg("Lx"), py::arg("half_height"), py::arg("re_tau"),
                    py::arg("y_plus_target") = 1.0)
        .def_static("make_channel_2d",
                    py::overload_cast<int, int, double, double>(&Mesh::makeChannel2D),
                    py::arg("nx"), py::arg("ny"), py::arg("Lx"), py::arg("Ly"))
        .def_static("make_plate_2d", &Mesh::makePlate2D, py::arg("nx"), py::arg("ny"),
                    py::arg("Lx"), py::arg("Ly"), py::arg("Re"), py::arg("y_plus_target") = 1.0)
        .def_static(
            "make_channel_2d",
            py::overload_cast<int, int, double, double, double, double>(&Mesh::makeChannel2D),
            py::arg("nx"), py::arg("ny"), py::arg("Lx"), py::arg("Ly"), py::arg("Re"),
            py::arg("yPlusTarget") = 1.0)
        .def_static("make_backward_facing_step_2d",
                    py::overload_cast<int, int, int, int, double, double, double, double>(
                        &Mesh::makeBackwardFacingStep2D),
                    py::arg("nx_up"), py::arg("nx_down"), py::arg("ny_up"), py::arg("ny_down"),
                    py::arg("Lu"), py::arg("Ld"), py::arg("h_s"), py::arg("H"))
        .def_static(
            "make_backward_facing_step_2d",
            py::overload_cast<int, int, int, int, double, double, double, double, double, double>(
                &Mesh::makeBackwardFacingStep2D),
            py::arg("nx_up"), py::arg("nx_down"), py::arg("ny_up"), py::arg("ny_down"),
            py::arg("Lu"), py::arg("Ld"), py::arg("h_s"), py::arg("H"), py::arg("Re"),
            py::arg("yPlusTarget") = 1.0)
        .def_static(
            "make_curved_channel_periodic_2d",
            [](py::array_t<double, py::array::c_style | py::array::forcecast> xN,
               py::array_t<double, py::array::c_style | py::array::forcecast> yB, double yTop,
               int ny, double Re, double yPlusTarget) {
                auto x = xN.unchecked<1>();
                auto y = yB.unchecked<1>();
                std::vector<double> xv(x.shape(0)), yv(y.shape(0));
                for (py::ssize_t i = 0; i < x.shape(0); ++i)
                    xv[i] = x(i);
                for (py::ssize_t i = 0; i < y.shape(0); ++i)
                    yv[i] = y(i);
                return Mesh::makeCurvedChannelPeriodic2D(xv, yv, yTop, ny, Re, yPlusTarget);
            },
            py::arg("x_nodes"), py::arg("y_bottom"), py::arg("y_top"), py::arg("ny"), py::arg("Re"),
            py::arg("yPlusTarget") = 1.0)
        .def_static("load_from_file", &Mesh::loadFromFile)
        .def("compute_wall_distance", &Mesh::computeWallDistance)
        .def("compute_geometry", &Mesh::computeGeometry)
        .def("wall_distance_valid", &Mesh::wallDistanceValid)
        .def("set_patch_type", &Mesh::setPatchType, py::arg("name"), py::arg("type"))
        .def("n_cells", &Mesh::nCells)
        .def("n_faces", &Mesh::nFaces)
        .def("n_nodes", &Mesh::nNodes)
        .def("n_patches", &Mesh::nPatches)
        .def("n_internal_faces", &Mesh::nInternalFaces)
        // Patch introspection
        .def("patch_names", &meshPatchNames)
        .def("patch_types", &meshPatchTypes)
        .def("wall_patch_data", &meshWallPatchData, py::arg("name"))
        .def("cell_centers", &meshCellCenters)
        .def("cell_volumes", &meshCellVolumes)
        .def("node_coords", &meshNodeCoords)
        .def("face_centers", &meshFaceCenters);

    // FlowBoundaryConditions
    py::class_<FlowBoundaryConditions>(m, "FlowBoundaryConditions")
        .def(py::init<>())
        .def_static("channel_defaults", &FlowBoundaryConditions::channelDefaults)
        .def_static("flat_plate_defaults", &FlowBoundaryConditions::flatPlateDefaults)
        .def_static("bfs_defaults", &FlowBoundaryConditions::bfsDefaults)
        .def_static("couette_defaults", &FlowBoundaryConditions::couetteDefaults, py::arg("mesh"),
                    py::arg("Uwall"), py::arg("kIn"), py::arg("omIn"))
        // per-face boundary profiles (patch-face order = wall_patch_data(name))
        .def(
            "set_velocity_profile",
            [](FlowBoundaryConditions &bc, const Mesh &mesh, const std::string &name,
               py::array_t<double, py::array::c_style | py::array::forcecast> vals) {
                auto v = vals.unchecked<2>();
                if (v.shape(1) != 3)
                    throw std::runtime_error("velocity profile must be (n_faces, 3)");
                std::vector<Vec3> out(v.shape(0));
                for (py::ssize_t i = 0; i < v.shape(0); ++i)
                    out[i] = Vec3(v(i, 0), v(i, 1), v(i, 2));
                bc.setVelocityProfile(mesh, name, out);
            },
            py::arg("mesh"), py::arg("name"), py::arg("values"))
        .def(
            "set_k_profile",
            [](FlowBoundaryConditions &bc, const Mesh &mesh, const std::string &name,
               py::array_t<double, py::array::c_style | py::array::forcecast> vals) {
                auto v = vals.unchecked<1>();
                std::vector<double> out(v.shape(0));
                for (py::ssize_t i = 0; i < v.shape(0); ++i)
                    out[i] = v(i);
                bc.setKProfile(mesh, name, out);
            },
            py::arg("mesh"), py::arg("name"), py::arg("values"))
        .def(
            "set_omega_profile",
            [](FlowBoundaryConditions &bc, const Mesh &mesh, const std::string &name,
               py::array_t<double, py::array::c_style | py::array::forcecast> vals) {
                auto v = vals.unchecked<1>();
                std::vector<double> out(v.shape(0));
                for (py::ssize_t i = 0; i < v.shape(0); ++i)
                    out[i] = v(i);
                bc.setOmegaProfile(mesh, name, out);
            },
            py::arg("mesh"), py::arg("name"), py::arg("values"));

    // SolverSettings
    py::class_<SolverSettings>(m, "SolverSettings")
        .def(py::init<>())
        .def_readwrite("max_iterations", &SolverSettings::maxIterations)
        .def_readwrite("convergence_tol", &SolverSettings::convergenceTol)
        .def_readwrite("convergence_window", &SolverSettings::convergenceWindow)
        .def_readwrite("monitor_interval", &SolverSettings::monitorInterval)
        .def_readwrite("legacy_convergence", &SolverSettings::legacyConvergence)
        .def_readwrite("momentum_convection", &SolverSettings::momentumConvection)
        .def_readwrite("turbulence_convection", &SolverSettings::turbulenceConvection)
        .def_readwrite("divergence_limit", &SolverSettings::divergenceLimit)
        .def_readwrite("alpha_u", &SolverSettings::alphaU)
        .def_readwrite("alpha_p", &SolverSettings::alphaP)
        .def_readwrite("alpha_k", &SolverSettings::alphaK)
        .def_readwrite("alpha_omega", &SolverSettings::alphaOmega)
        .def_readwrite("inner_iterations", &SolverSettings::innerIterations)
        .def_readwrite("inner_tolerance", &SolverSettings::innerTolerance)
        .def_readwrite("turb_start_iter", &SolverSettings::turbStartIter)
        .def_readwrite("turb_update_interval", &SolverSettings::turbUpdateInterval)
        .def_readwrite("nut_floor_iters", &SolverSettings::nuTFloorIters)
        .def_readwrite("legacy_rhie_chow_gate", &SolverSettings::legacyRhieChowGate)
        .def_readwrite("legacy_omega_wall_pinning", &SolverSettings::legacyOmegaWallPinning)
        .def_readwrite("alpha_injection", &SolverSettings::alphaInjection)
        .def_readwrite("body_force", &SolverSettings::bodyForce)
        .def_readwrite("k_min", &SolverSettings::kMin)
        .def_readwrite("omega_min", &SolverSettings::omegaMin)
        .def_readwrite("verbose", &SolverSettings::verbose)
        .def_readwrite("report_interval", &SolverSettings::reportInterval)
        .def_readwrite("alpha_t", &SolverSettings::alphaT)
        // Wall-function settings
        .def_readwrite("use_wall_functions", &SolverSettings::useWallFunctions)
        .def_readwrite("von_karman", &SolverSettings::vonKarman)
        .def_readwrite("wall_fn_E", &SolverSettings::wallFnE)
        // Closure selection: 0=Full, 1=NoLimiter, 2=KOmega
        .def_readwrite("sst_variant", &SolverSettings::sstVariant);

    py::class_<PreparedObservation>(m, "PreparedObservation")
        .def("n_obs", &PreparedObservation::nObs)
        .def("evaluate",
             py::overload_cast<const FlowFields &, double>(&PreparedObservation::evaluate,
                                                           py::const_),
             py::arg("fields"), py::arg("nu"));
    // ObservationOperator
    py::class_<ObservationOperator>(m, "ObservationOperator")
        .def(py::init<>())
        .def("add_drag", &ObservationOperator::addDrag, py::arg("wall_patch"), py::arg("cd_obs"),
             py::arg("sigma"), py::arg("ref_area"), py::arg("ref_vel"),
             py::arg("sigma_model") = 0.0)
        .def("add_skin_friction", &ObservationOperator::addSkinFriction, py::arg("wall_patch"),
             py::arg("location"), py::arg("cf_obs"), py::arg("sigma"), py::arg("ref_vel"),
             py::arg("sigma_model") = 0.0)
        .def("add_velocity_profile", &ObservationOperator::addVelocityProfile, py::arg("location"),
             py::arg("component"), py::arg("u_obs"), py::arg("sigma"), py::arg("sigma_model") = 0.0)
        .def("add_reattachment_length", &ObservationOperator::addReattachmentLength,
             py::arg("wall_patch"), py::arg("xr_obs"), py::arg("sigma"),
             py::arg("sigma_model") = 0.0)
        .def("prepare", &ObservationOperator::prepare, py::arg("mesh"), py::keep_alive<0, 2>())
        .def("evaluate", &ObservationOperator::evaluate, py::arg("mesh"), py::arg("fields"),
             py::arg("nu"))
        .def("n_obs", &ObservationOperator::nObs)
        .def("log_likelihood_from_predictions",
             py::overload_cast<const std::vector<double> &>(&ObservationOperator::logLikelihood,
                                                            py::const_));

    // Vec3 (for location arguments)
    py::class_<Vec3>(m, "Vec3")
        .def(py::init<double, double, double>())
        .def_readwrite("x", &Vec3::x)
        .def_readwrite("y", &Vec3::y)
        .def_readwrite("z", &Vec3::z);

    py::class_<ResidualEntry>(m, "ResidualEntry")
        .def_readonly("iteration", &ResidualEntry::iteration)
        .def_readonly("momentum_scaled", &ResidualEntry::momentumScaled)
        .def_readonly("continuity_scaled", &ResidualEntry::continuityScaled)
        .def_readonly("change_u", &ResidualEntry::changeU)
        .def_readonly("remaining_u", &ResidualEntry::remainingU);
    py::class_<ConvergenceHistory>(m, "ConvergenceHistory")
        .def_readonly("entries", &ConvergenceHistory::entries)
        .def_readonly("converged", &ConvergenceHistory::converged)
        .def_readonly("diverged", &ConvergenceHistory::diverged)
        .def_readonly("stalled", &ConvergenceHistory::stalled)
        .def_readonly("final_iteration", &ConvergenceHistory::finalIter)
        .def_readonly("final_worst_measure", &ConvergenceHistory::finalWorstMeasure)
        .def_readonly("contraction_rate_u", &ConvergenceHistory::contractionRateU);
    py::class_<FlowFields>(m, "FlowFields")
        .def(py::init<const Mesh &>(), py::arg("mesh"), py::keep_alive<1, 2>())
        .def("snapshot", &completeFields,
             "Owned interior and boundary arrays; mesh/config identity is external.")
        .def("restore", &restoreFields, py::arg("snapshot"),
             "Restore shape-checked numeric fields; caller must ensure mesh/config identity. "
             "This does not restore solver workspaces or face fluxes.");
    py::class_<SIMPLESolver::DiagnosticResidual>(m, "DiagnosticResidual")
        .def_readonly("transport", &SIMPLESolver::DiagnosticResidual::transport)
        .def_readonly("continuity", &SIMPLESolver::DiagnosticResidual::continuity)
        .def_readonly("face_fluxes", &SIMPLESolver::DiagnosticResidual::faceFluxes);
    py::class_<PythonSIMPLESolver>(m, "SIMPLESolver")
        .def(py::init<const Mesh &, const SSTCoefficients &, const FlowBoundaryConditions &, double,
                      const SolverSettings &>(),
             py::arg("mesh"), py::arg("coefficients"), py::arg("bcs"), py::arg("nu"),
             py::arg("settings") = SolverSettings{}, py::keep_alive<1, 2>())
        .def("diagnostic_residual",
             [](PythonSIMPLESolver &s, const FlowFields &f, const SSTCoefficients &theta) {
                 s.validate(f);
                 return s.solver.diagnosticResidual(f, theta);
             })
        .def("diagnostic_parameter_partials",
             [](PythonSIMPLESolver &s, const FlowFields &f, const SSTCoefficients &theta) {
                 s.validate(f);
                 return s.solver.diagnosticParameterPartials(f, theta);
             })
        .def(
            "init_uniform",
            [](PythonSIMPLESolver &s, FlowFields &f, const Vec3 &u, double p, double k,
               double omega) {
                s.validate(f);
                s.solver.initUniform(f, u, p, k, omega);
            },
            py::arg("fields"), py::arg("u"), py::arg("p"), py::arg("k"), py::arg("omega"))
        .def(
            "solve",
            [](PythonSIMPLESolver &s, FlowFields &f, py::object observer) {
                // Reject before any callback installation/cleanup can touch an
                // outer solve's observer, including a caught recursive error.
                if (s.solving)
                    throw std::logic_error("recursive solve from source observer is unsupported");
                s.validate(f);
                if (!observer.is_none() && !PyCallable_Check(observer.ptr()))
                    throw py::type_error("source_observer must be callable or None");
                // Clear even when Python raises. No native callback ownership remains
                // after this call, so a callback may safely capture its solver.
                struct ClearObserver {
                    PythonSIMPLESolver &owner;
                    ~ClearObserver() {
                        owner.solver.setSourceObserver({});
                        owner.solving = false;
                    }
                } clear{s};
                s.solving = true;
                if (!observer.is_none()) {
                    s.solver.setSourceObserver([observer](
                                                   const SIMPLESolver::SourceSnapshotView &v) {
                        py::dict data;
                        data["iteration"] = v.iteration;
                        data["stage"] = "pre_momentum_x";
                        data["fields"] = completeFields(v.fields);
                        data["face_fluxes"] = scalarValues(v.faceFluxes);
                        data["fresh_integrated_force"] = vectorValues(v.freshIntegratedForce);
                        data["applied_integrated_force"] = vectorValues(v.appliedIntegratedForce);
                        observer(data);
                    });
                }
                return s.solver.solve(f);
            },
            py::arg("fields"), py::arg("source_observer") = py::none(),
            "Optional observer receives owned snapshots under the GIL and is retained only for "
            "this solve.")
        .def("set_target_anisotropy", &PythonSIMPLESolver::setTarget)
        .def("clear_target_anisotropy",
             [](PythonSIMPLESolver &s) {
                 s.solver.setTargetAnisotropy(nullptr);
                 s.target.clear();
             })
        .def("set_injection_scale",
             [](PythonSIMPLESolver &s, double scale) { s.solver.setInjectionScale(scale); })
        .def("face_fluxes",
             [](const PythonSIMPLESolver &s) { return scalarValues(s.solver.faceFluxes()); })
        .def(
            "diagnose_momentum",
            [](PythonSIMPLESolver &s, const FlowFields &fields, const SSTCoefficients &theta,
               const std::vector<double> &flux, bool recompute) {
                s.validate(fields);
                auto d = s.solver.diagnoseMomentum(fields, theta, flux,
                                                   recompute ? SIMPLESolver::ClosureMode::Recomputed
                                                             : SIMPLESolver::ClosureMode::Stored);
                py::dict result;
                result["fields"] = completeFields(d.fields);
                result["residual"] = vectorValues(d.residual);
                result["fresh_integrated_force"] = vectorValues(d.freshIntegratedForce);
                return result;
            },
            py::arg("fields"), py::arg("coefficients"), py::arg("face_fluxes"),
            py::arg("recompute_closure") = false);

    py::class_<SIMPLESolver::InjectionDiagnostics>(m, "InjectionDiagnostics")
        .def_readonly("active", &SIMPLESolver::InjectionDiagnostics::active)
        .def_readonly("checked_iterations", &SIMPLESolver::InjectionDiagnostics::checkedIters)
        .def_readonly("all_realizable", &SIMPLESolver::InjectionDiagnostics::allRealizable)
        .def_readonly("max_violation", &SIMPLESolver::InjectionDiagnostics::maxViolation);
    py::class_<ForwardEvaluationResult>(m, "ForwardEvaluationResult")
        .def_readonly("result", &ForwardEvaluationResult::result)
        .def_readonly("convergence", &ForwardEvaluationResult::convergence)
        .def_readonly("injection", &ForwardEvaluationResult::injection)
        .def_readonly("error", &ForwardEvaluationResult::error)
        .def_readonly("used_warm_start", &ForwardEvaluationResult::usedWarmStart)
        .def_property_readonly(
            "snapshot",
            [](const ForwardEvaluationResult &result) -> py::object {
                if (!result.snapshot)
                    return py::none();
                return ownedSnapshot(*result.snapshot);
            },
            "Owned numeric interiors, boundaries, corrected face fluxes and turbulence marker. "
            "A restart additionally requires matching mesh and configuration identity.");
    py::class_<WarmStartCache>(m, "WarmStartCache")
        .def("size", &WarmStartCache::size)
        .def("clear", &WarmStartCache::clear)
        .def("set_byte_budget", &WarmStartCache::setByteBudget)
        .def("byte_budget", &WarmStartCache::byteBudget)
        .def("retained_bytes", &WarmStartCache::retainedBytes);

    py::class_<ForwardBatchRow>(m, "ForwardBatchRow")
        .def_readonly("result", &ForwardBatchRow::result)
        .def_readonly("final_worst_measure", &ForwardBatchRow::finalWorstMeasure)
        .def_readonly("contraction_rate_u", &ForwardBatchRow::contractionRateU)
        .def_readonly("stalled", &ForwardBatchRow::stalled)
        .def_readonly("error", &ForwardBatchRow::error);
    py::class_<ForwardBatchResult>(m, "ForwardBatchResult")
        .def_readonly("rows", &ForwardBatchResult::rows)
        .def_property_readonly("predictions",
                               [](const ForwardBatchResult &r) {
                                   py::array_t<double> a(
                                       {static_cast<py::ssize_t>(r.rows.size()),
                                        static_cast<py::ssize_t>(r.nObservations)});
                                   auto v = a.mutable_unchecked<2>();
                                   for (size_t i = 0; i < r.rows.size(); ++i)
                                       for (int j = 0; j < r.nObservations; ++j)
                                           v(i, j) = r.rows[i].result.predictions[j];
                                   return a;
                               })
        .def_property_readonly("log_likelihoods",
                               [](const ForwardBatchResult &r) {
                                   std::vector<double> values;
                                   values.reserve(r.rows.size());
                                   for (const auto &row : r.rows)
                                       values.push_back(row.result.loglik);
                                   return scalarValues(values);
                               })
        .def_property_readonly("statuses",
                               [](const ForwardBatchResult &r) {
                                   std::vector<EvaluationStatus> values;
                                   values.reserve(r.rows.size());
                                   for (const auto &row : r.rows)
                                       values.push_back(row.result.status);
                                   return values;
                               })
        .def_property_readonly("iteration_counts",
                               [](const ForwardBatchResult &r) {
                                   py::array_t<int> a(r.rows.size());
                                   auto v = a.mutable_unchecked<1>();
                                   for (size_t i = 0; i < r.rows.size(); ++i)
                                       v(i) = r.rows[i].result.simpleIters;
                                   return a;
                               })
        .def_property_readonly("errors",
                               [](const ForwardBatchResult &r) {
                                   std::vector<std::string> values;
                                   values.reserve(r.rows.size());
                                   for (const auto &row : r.rows)
                                       values.push_back(row.error);
                                   return values;
                               })
        .def_property_readonly("convergence_mask", [](const ForwardBatchResult &r) {
            py::array_t<bool> a(r.rows.size());
            auto v = a.mutable_unchecked<1>();
            for (size_t i = 0; i < r.rows.size(); ++i)
                v(i) = r.converged[i] != 0;
            return a;
        });
    py::class_<ForwardBatch>(m, "ForwardBatch")
        .def(py::init<const Mesh &, const InferenceParameterSet &, const ObservationOperator &,
                      const FlowBoundaryConditions &, double, const SolverSettings &, const Vec3 &,
                      double, double, double>(),
             py::arg("mesh"), py::arg("param_set"), py::arg("obs_op"), py::arg("bcs"),
             py::arg("nu"), py::arg("settings") = SolverSettings{},
             py::arg("u_init") = Vec3{1, 0, 0}, py::arg("p_init") = 0.0, py::arg("k_init") = 1e-4,
             py::arg("omega_init") = 1.0)
        .def(
            "evaluate",
            [](const ForwardBatch &batch,
               py::array_t<double, py::array::c_style | py::array::forcecast> theta, int workers) {
                if (theta.ndim() != 2 || theta.shape(1) != batch.nParameters())
                    throw py::value_error("theta must have shape (n_rows, n_parameters)");
                const auto values = theta.unchecked<2>();
                std::vector<std::vector<double>> rows(values.shape(0),
                                                      std::vector<double>(values.shape(1)));
                for (py::ssize_t i = 0; i < values.shape(0); ++i)
                    for (py::ssize_t j = 0; j < values.shape(1); ++j)
                        rows[i][j] = values(i, j);
                ForwardBatchResult result;
                {
                    py::gil_scoped_release release;
                    result = batch.evaluate(rows, workers);
                }
                return result;
            },
            py::arg("theta"), py::arg("workers") = 1);

    // ForwardModel
    py::class_<ForwardModel>(m, "ForwardModel")
        .def(py::init([](const Mesh &mesh, const InferenceParameterSet &params,
                         const ObservationOperator &obs, const FlowBoundaryConditions &bcs,
                         double nu, const SolverSettings &settings, const Vec3 &u, double p,
                         double k, double omega) {
                 requirePreparedMesh(mesh);
                 return std::make_unique<ForwardModel>(mesh, params, obs, bcs, nu, settings, u, p,
                                                       k, omega);
             }),
             py::arg("mesh"), py::arg("param_set"), py::arg("obs_op"), py::arg("bcs"),
             py::arg("nu"), py::arg("settings") = SolverSettings{},
             py::arg("u_init") = Vec3{1, 0, 0}, py::arg("p_init") = 0.0, py::arg("k_init") = 1e-4,
             py::arg("omega_init") = 1.0, py::keep_alive<1, 2>())
        .def("evaluate", &ForwardModel::evaluate)
        .def("evaluate_with_diagnostics", &ForwardModel::evaluateWithDiagnostics, py::arg("theta"),
             py::arg("snapshot") = false)
        .def("set_retain_last_fields", &ForwardModel::setRetainLastFields)
        .def("cache", &ForwardModel::cache, py::return_value_policy::reference_internal)
        .def("penalized_log_likelihood", &ForwardModel::penalizedLogLikelihood)
        .def("param_set", &ForwardModel::paramSet, py::return_value_policy::reference_internal)
        .def("has_last_fields", &ForwardModel::hasLastFields)
        .def("last_fields", &extractFields)
        // a-posteriori Reynolds-stress injection: (nCells, 3, 3) symmetric
        // anisotropy batch (project it into the realizable set first); stored
        // by value, applied by every subsequent evaluate() until cleared
        .def(
            "set_target_anisotropy",
            [](ForwardModel &fm, py::array_t<double, py::array::c_style | py::array::forcecast> b) {
                auto v = b.unchecked<3>();
                if (v.shape(1) != 3 || v.shape(2) != 3)
                    throw std::runtime_error("set_target_anisotropy: expected (n_cells, 3, 3)");
                const py::ssize_t n = v.shape(0);
                std::vector<double> b6(6 * n);
                for (py::ssize_t i = 0; i < n; ++i) {
                    // xx, yy, zz, xy, xz, yz (symmetrised off-diagonals)
                    b6[6 * i + 0] = v(i, 0, 0);
                    b6[6 * i + 1] = v(i, 1, 1);
                    b6[6 * i + 2] = v(i, 2, 2);
                    b6[6 * i + 3] = 0.5 * (v(i, 0, 1) + v(i, 1, 0));
                    b6[6 * i + 4] = 0.5 * (v(i, 0, 2) + v(i, 2, 0));
                    b6[6 * i + 5] = 0.5 * (v(i, 1, 2) + v(i, 2, 1));
                }
                fm.setTargetAnisotropy(std::move(b6));
            },
            py::arg("b_target"))
        .def("clear_target_anisotropy", &ForwardModel::clearTargetAnisotropy)
        .def("has_target_anisotropy", &ForwardModel::hasTargetAnisotropy)
        .def("injection_diagnostics", [](const ForwardModel &fm) {
            const auto &d = fm.injectionDiagnostics();
            py::dict out;
            out["active"] = d.active;
            out["checked_iters"] = d.checkedIters;
            out["all_realizable"] = d.allRealizable;
            out["max_violation"] = d.maxViolation;
            return out;
        });

    // Results of experimental tangent and finite-difference sensitivity interfaces,
    // including per-coefficient Krylov diagnostics.
    py::class_<TangentGradientResult>(m, "TangentGradientResult")
        .def_readonly("d_obs_d_theta", &TangentGradientResult::dObsDTheta)        // nObs × 11
        .def_readonly("log_lik_gradient", &TangentGradientResult::logLikGradient) // 11 (warm-FD)
        .def_readonly("krylov_iters", &TangentGradientResult::krylovIters)
        .def_readonly("krylov_rel_res", &TangentGradientResult::krylovRelRes)
        .def_readonly("krylov_converged", &TangentGradientResult::krylovConverged)
        .def_readonly("n_residual_evals", &TangentGradientResult::nResidualEvals)
        .def_readonly("finite_difference_steps", &TangentGradientResult::finiteDifferenceSteps)
        .def_readonly("perturbed_solve_tolerances",
                      &TangentGradientResult::perturbedSolveTolerances)
        .def_readonly("positive_status", &TangentGradientResult::positiveStatus)
        .def_readonly("negative_status", &TangentGradientResult::negativeStatus)
        .def_readonly("n_colors", &TangentGradientResult::nColors);

    py::class_<SIMPLESolver::SteadyResidual>(m, "SteadyResidual")
        .def_readonly("equations", &SIMPLESolver::SteadyResidual::equations)
        .def_readonly("continuity", &SIMPLESolver::SteadyResidual::continuity)
        .def_readonly("face_fluxes", &SIMPLESolver::SteadyResidual::faceFluxes)
        .def_readonly("removed_continuity", &SIMPLESolver::SteadyResidual::removedContinuity)
        .def_readonly("pressure_gauge", &SIMPLESolver::SteadyResidual::pressureGauge);
    py::class_<SteadySensitivityOptions>(m, "SteadySensitivityOptions")
        .def(py::init<>())
        .def_readwrite("state_scales", &SteadySensitivityOptions::stateScales)
        .def_readwrite("equation_scales", &SteadySensitivityOptions::equationScales)
        .def_readwrite("action_step", &SteadySensitivityOptions::actionStep)
        .def_readwrite("parameter_relative_step", &SteadySensitivityOptions::parameterRelativeStep)
        .def_readwrite("krylov_tolerance", &SteadySensitivityOptions::krylovTolerance)
        .def_readwrite("restart", &SteadySensitivityOptions::restart)
        .def_readwrite("max_iterations", &SteadySensitivityOptions::maxIterations);
    py::class_<SteadyTangentResult>(m, "SteadyTangentResult")
        .def_readonly("derivatives", &SteadyTangentResult::derivatives)
        .def_readonly("state_tangents", &SteadyTangentResult::stateTangents)
        .def_readonly("branch_qualified", &SteadyTangentResult::branchQualified)
        .def_readonly("normalized_base_residual", &SteadyTangentResult::normalizedBaseResidual)
        .def_readonly("removed_continuity", &SteadyTangentResult::removedContinuity)
        .def_readonly("krylov_basis_bytes", &SteadyTangentResult::krylovBasisBytes)
        .def_readonly("experimental", &SteadyTangentResult::experimental)
        .def_readonly("qualification", &SteadyTangentResult::qualification);

    // Fixed-state coefficient partials, experimental tangents and total finite
    // differences are separate interfaces. theta is the full 11-coefficient vector.
    // A converged matching state is required by the solve-based sensitivity paths;
    // each derivative retains its own qualification limits.
    py::class_<ParameterSensitivity>(m, "ParameterSensitivity")
        // keep_alive<1,2>: ParameterSensitivity stores `const Mesh&`, so the Python Mesh
        // (arg 2) must outlive the instance (arg 1).  Without this, `ParameterSensitivity(
        // *build_case())` would free the mesh temporary right after construction and the
        // stored reference would dangle.
        .def(py::init([](const Mesh &mesh, const ObservationOperator &obs,
                         const FlowBoundaryConditions &bcs, double nu,
                         const SolverSettings &settings, const Vec3 &u, double p, double k,
                         double omega) {
                 requirePreparedMesh(mesh);
                 return std::make_unique<ParameterSensitivity>(mesh, obs, bcs, nu, settings, u, p,
                                                               k, omega);
             }),
             py::arg("mesh"), py::arg("obs_op"), py::arg("bcs"), py::arg("nu"),
             py::arg("settings") = SolverSettings{}, py::arg("u_init") = Vec3{1, 0, 0},
             py::arg("p_init") = 0.0, py::arg("k_init") = 1e-4, py::arg("omega_init") = 1.0,
             py::keep_alive<1, 2>())
        .def("solve_state", &ParameterSensitivity::solveState, py::arg("theta"))
        .def("has_state", &ParameterSensitivity::hasState)
        .def("n_state", &ParameterSensitivity::nState)
        .def("n_cells", &ParameterSensitivity::nCells)
        .def("n_obs", &ParameterSensitivity::nObs)
        .def("set_coupled_krylov_restart", &ParameterSensitivity::setCoupledKrylovRestart)
        .def("coupled_krylov_restart", &ParameterSensitivity::coupledKrylovRestart)
        .def("coupled_krylov_basis_bytes", &ParameterSensitivity::coupledKrylovBasisBytes)
        .def("set_allow_experimental_tangents", &ParameterSensitivity::setAllowExperimentalTangents,
             "Explicitly opt into derivative paths with known operator inconsistencies; "
             "this does not certify the returned sensitivity.")
        .def("residual", &ParameterSensitivity::residual, py::arg("theta"))
        .def("d_residual_d_theta", &ParameterSensitivity::dResidualDTheta, py::arg("theta"))
        .def("observe", &ParameterSensitivity::observe, py::arg("theta"))
        .def("log_lik", &ParameterSensitivity::logLik, py::arg("theta"))
        .def("d_obs_d_theta", &ParameterSensitivity::dObsDTheta, py::arg("theta"))
        // Experimental frozen-pressure tangent. Requires a converged solve_state(theta)
        // and the same theta; explicit experimental opt-in is also required.
        .def("eta_jacobian_tangent", &ParameterSensitivity::etaJacobianTangent, py::arg("theta"),
             py::arg("krylov_tol") = 1e-8, py::arg("max_iter") = 3000, py::arg("fd_step") = 1e-6)
        // Experimental pressure-coupled sensitivity. Explicit opt-in and a
        // converged matching base state are required. Krylov convergence does
        // not certify derivative correctness while operator inconsistencies remain.
        .def("eta_jacobian_tangent_coupled", &ParameterSensitivity::etaJacobianTangentCoupled,
             py::arg("theta"), py::arg("krylov_tol") = 1e-7, py::arg("max_iter") = 2000,
             py::arg("fd_step") = 1e-6)
        .def("steady_state_vector", &ParameterSensitivity::steadyStateVector)
        .def("require_steady_pair", &ParameterSensitivity::requireSteadyPair)
        .def("steady_residual_at", &ParameterSensitivity::steadyResidualAt)
        .def("steady_state_action", &ParameterSensitivity::steadyStateAction, py::arg("direction"),
             py::arg("step") = 1e-5)
        .def("steady_krylov_basis_bytes", &ParameterSensitivity::steadyKrylovBasisBytes)
        .def("eta_jacobian_steady", &ParameterSensitivity::etaJacobianSteady,
             py::arg("options") = SteadySensitivityOptions{})
        .def("steady_base",
             [](const ParameterSensitivity &sensitivity) {
                 const auto base = sensitivity.steadyBase();
                 py::dict result;
                 result["fields"] = completeFields(base.fields);
                 result["corrected_fluxes"] = base.correctedFluxes;
                 result["history"] = base.history;
                 result["coefficients"] = base.coefficients;
                 result["settings"] = base.settings;
                 result["boundaries"] = base.boundaries;
                 result["molecular_viscosity"] = base.molecularViscosity;
                 result["source_digest"] = base.sourceDigest;
                 return result;
             })
        // Total sensitivities from warm-started central finite differences.
        // Base and perturbed solves must converge; step refinement remains necessary.
        .def("eta_jacobian_warm_fd", &ParameterSensitivity::etaJacobianWarmFD, py::arg("theta"),
             py::arg("h_rel") = 5e-4, py::arg("h_floor") = 1e-7, py::arg("warm_max_iter") = 0,
             py::arg("warm_tol") = 0.0, py::arg("absolute_steps") = std::vector<double>{});

    // ---- Compressible bindings -------------------------------------------
    //
    // Mirrors the incompressible API: same EvaluationStatus / EvaluationResult
    // (already bound above), same SSTCoefficients / InferenceParameterSet, but
    // with a temperature/density-aware solver and EOS.

    py::class_<IdealGasEOS>(m, "IdealGasEOS")
        .def(py::init<>())
        .def_readwrite("gamma", &IdealGasEOS::gamma)
        .def_readwrite("R", &IdealGasEOS::R)
        .def_readwrite("Pr", &IdealGasEOS::Pr)
        .def_readwrite("Pr_T", &IdealGasEOS::Pr_T)
        .def_readwrite("mu_ref", &IdealGasEOS::mu_ref)
        .def_readwrite("T_ref", &IdealGasEOS::T_ref)
        .def_readwrite("S_suth", &IdealGasEOS::S_suth)
        .def("Cp", &IdealGasEOS::Cp)
        .def("Cv", &IdealGasEOS::Cv)
        .def("viscosity", &IdealGasEOS::viscosity, py::arg("T"))
        .def("conductivity", &IdealGasEOS::conductivity, py::arg("mu"), py::arg("muT"))
        .def("density", &IdealGasEOS::density, py::arg("p"), py::arg("T"))
        .def("temperature", &IdealGasEOS::temperature, py::arg("p"), py::arg("rho"))
        .def("pressure", &IdealGasEOS::pressure, py::arg("rho"), py::arg("T"))
        .def("sound_speed", &IdealGasEOS::soundSpeed, py::arg("T"))
        .def("mach_number", &IdealGasEOS::machNumber, py::arg("Umag"), py::arg("T"))
        .def("total_enthalpy", &IdealGasEOS::totalEnthalpy, py::arg("T"), py::arg("Umag2"));

    py::class_<CompressibleBoundaryConditions>(m, "CompressibleBoundaryConditions")
        .def(py::init<>())
        .def_static("channel_defaults", &CompressibleBoundaryConditions::channelDefaults,
                    py::arg("mesh"), py::arg("Uin"), py::arg("T_in"), py::arg("p_out"),
                    py::arg("kIn"), py::arg("omIn"));

    py::class_<CompressibleForwardModel>(m, "CompressibleForwardModel")
        .def(py::init([](const Mesh &mesh, const InferenceParameterSet &params,
                         const ObservationOperator &obs, const CompressibleBoundaryConditions &bcs,
                         const IdealGasEOS &eos, const SolverSettings &settings, const Vec3 &u,
                         double p, double temperature, double k, double omega) {
                 requirePreparedMesh(mesh);
                 return std::make_unique<CompressibleForwardModel>(
                     mesh, params, obs, bcs, eos, settings, u, p, temperature, k, omega);
             }),
             py::arg("mesh"), py::arg("param_set"), py::arg("obs_op"), py::arg("bcs"),
             py::arg("eos"), py::arg("settings") = SolverSettings{},
             py::arg("u_init") = Vec3{1.0, 0.0, 0.0}, py::arg("p_init") = 101325.0,
             py::arg("T_init") = 300.0, py::arg("k_init") = 1e-4, py::arg("omega_init") = 100.0,
             py::keep_alive<1, 2>())
        .def("evaluate", &CompressibleForwardModel::evaluate)
        .def("penalized_log_likelihood", &CompressibleForwardModel::penalizedLogLikelihood)
        .def("param_set", &CompressibleForwardModel::paramSet,
             py::return_value_policy::reference_internal)
        .def("has_last_fields", &CompressibleForwardModel::hasLastFields)
        .def("last_fields", &extractCompressibleFields);

    // ---- Density-based shock-capturing solver (dbns) ----------------------
    using namespace dbns;

    py::class_<Primitive>(m, "Primitive")
        .def(py::init<>())
        .def(py::init([](double rho, double u, double v, double p, double k, double w) {
                 Primitive V;
                 V.rho = rho;
                 V.u = u;
                 V.v = v;
                 V.p = p;
                 V.k = k;
                 V.omega = w;
                 return V;
             }),
             py::arg("rho"), py::arg("u"), py::arg("v"), py::arg("p"), py::arg("k") = 0.0,
             py::arg("omega") = 0.0)
        .def_readwrite("rho", &Primitive::rho)
        .def_readwrite("u", &Primitive::u)
        .def_readwrite("v", &Primitive::v)
        .def_readwrite("p", &Primitive::p)
        .def_readwrite("k", &Primitive::k)
        .def_readwrite("omega", &Primitive::omega);

    py::enum_<TimeMode>(m, "TimeMode")
        .value("Steady", TimeMode::Steady)
        .value("Unsteady", TimeMode::Unsteady);
    py::enum_<CompressibilityModel>(m, "CompressibilityModel")
        .value("None_", CompressibilityModel::None)
        .value("Sarkar", CompressibilityModel::Sarkar)
        .value("Zeman", CompressibilityModel::Zeman);
    py::enum_<BoundaryKind>(m, "DBNSBoundaryKind")
        .value("SupersonicInflow", BoundaryKind::SupersonicInflow)
        .value("Extrapolate", BoundaryKind::Extrapolate)
        .value("SubsonicInflow", BoundaryKind::SubsonicInflow)
        .value("SubsonicOutflow", BoundaryKind::SubsonicOutflow)
        .value("SlipWall", BoundaryKind::SlipWall)
        .value("NoSlipAdiabatic", BoundaryKind::NoSlipAdiabatic)
        .value("NoSlipIsothermal", BoundaryKind::NoSlipIsothermal)
        .value("FixedState", BoundaryKind::FixedState);

    py::class_<BoundarySpec>(m, "DBNSBoundarySpec")
        .def(py::init<>())
        .def_readwrite("kind", &BoundarySpec::kind)
        .def_readwrite("freestream", &BoundarySpec::freestream)
        .def_readwrite("wall_temp", &BoundarySpec::wallTemp)
        .def_readwrite("back_pressure", &BoundarySpec::backPressure)
        .def_readwrite("wall_velocity", &BoundarySpec::wallVelocity)
        .def(
            "set_wall_temp_profile",
            [](BoundarySpec &spec, py::array_t<double> arr) {
                // per-face wall temperatures for NoSlipIsothermal patches in
                // the patch's own face order (the measured wall-temperature
                // row: recovery upstream of the thermal switch, the imposed
                // s-condition downstream)
                auto a = arr.unchecked<1>();
                spec.wallTempProfile.resize(static_cast<size_t>(a.shape(0)));
                for (py::ssize_t i = 0; i < a.shape(0); ++i)
                    spec.wallTempProfile[i] = a(i);
            },
            py::arg("array"))
        .def(
            "set_profile",
            [](BoundarySpec &spec, py::array_t<double> arr) {
                // per-face prescribed states for SupersonicInflow/FixedState
                // patches, (n_faces, 6) rows of rho, u, v, p, k, omega in the
                // patch's own face order (e.g. a measured incoming-layer
                // profile, or an imposed-shock top boundary)
                auto a = arr.unchecked<2>();
                if (a.shape(1) != 6)
                    throw std::runtime_error("set_profile: expected (n_faces, 6)");
                spec.profile.resize(a.shape(0));
                for (py::ssize_t i = 0; i < a.shape(0); ++i) {
                    Primitive &V = spec.profile[i];
                    V.rho = a(i, 0);
                    V.u = a(i, 1);
                    V.v = a(i, 2);
                    V.p = a(i, 3);
                    V.k = a(i, 4);
                    V.omega = a(i, 5);
                }
            },
            py::arg("array"))
        .def("clear_profile", [](BoundarySpec &spec) { spec.profile.clear(); });

    py::class_<DBNSBoundaryConditions>(m, "DBNSBoundaryConditions")
        .def(py::init<>())
        .def("set", &DBNSBoundaryConditions::set, py::arg("patch"), py::arg("spec"));

    py::class_<DBNSSettings>(m, "DBNSSettings")
        .def(py::init<>())
        .def_readwrite("time_mode", &DBNSSettings::timeMode)
        .def_readwrite("cfl", &DBNSSettings::cfl)
        .def_readwrite("max_iterations", &DBNSSettings::maxIterations)
        .def_readwrite("t_end", &DBNSSettings::tEnd)
        .def_readwrite("convergence_tol", &DBNSSettings::convergenceTol)
        .def_readwrite("reconstruct_order", &DBNSSettings::reconstructOrder)
        .def_readwrite("limit_reconstruction", &DBNSSettings::limitReconstruction)
        .def_readwrite("viscous", &DBNSSettings::viscous)
        .def_readwrite("turbulent", &DBNSSettings::turbulent)
        .def_readwrite("const_mu", &DBNSSettings::constMu)
        .def_readwrite("compressibility", &DBNSSettings::compressibility)
        .def_readwrite("rk_stages", &DBNSSettings::rkStages)
        .def_readwrite("verbose", &DBNSSettings::verbose)
        .def_readwrite("report_interval", &DBNSSettings::reportInterval)
        .def_readwrite("early_abort_iter", &DBNSSettings::earlyAbortIter)
        .def_readwrite("early_abort_rel_max", &DBNSSettings::earlyAbortRelMax)
        .def_readwrite("injection_ramp_iters", &DBNSSettings::injectionRampIters)
        .def_readwrite("injection_frozen_k", &DBNSSettings::injectionFrozenK)
        .def_readwrite("frozen_mean_flow", &DBNSSettings::frozenMeanFlow)
        .def_readwrite("implicit_steady", &DBNSSettings::implicitSteady)
        .def_readwrite("legacy_auto_freeze_limiter", &DBNSSettings::legacyAutoFreezeLimiter)
        .def_readwrite("cfl_implicit", &DBNSSettings::cflImplicit)
        .def_readwrite("cfl_ramp_start", &DBNSSettings::cflRampStart)
        .def_readwrite("cfl_ramp_iters", &DBNSSettings::cflRampIters);

    py::class_<ReferenceState>(m, "ReferenceState")
        .def(py::init<>())
        .def_readwrite("rho", &ReferenceState::rho)
        .def_readwrite("U", &ReferenceState::U)
        .def_readwrite("T", &ReferenceState::T)
        .def_readwrite("p", &ReferenceState::p)
        .def_readwrite("recovery_factor", &ReferenceState::recoveryFactor);

    py::class_<SolveReport>(m, "DBNSSolveReport")
        .def_readonly("status", &SolveReport::status)
        .def_readonly("iterations", &SolveReport::iterations)
        .def_readonly("final_residual", &SolveReport::finalResidual)
        .def_readonly("quiescent", &SolveReport::quiescent)
        .def_readonly("t_final", &SolveReport::tFinal);

    py::class_<DBNSSolver>(m, "DBNSSolver")
        .def(py::init<const Mesh &, const IdealGasEOS &, const SSTCoefficients &,
                      const DBNSBoundaryConditions &, const DBNSSettings &>(),
             py::arg("mesh"), py::arg("eos"), py::arg("sst"), py::arg("bcs"),
             py::arg("settings") = DBNSSettings{}, py::keep_alive<1, 2>())
        .def("init_uniform", &DBNSSolver::initUniform, py::arg("state"))
        .def("init_field", &dbnsInitField, py::arg("array"))
        .def("solve", &DBNSSolver::solve)
        .def("prepare_properties", &DBNSSolver::prepareProperties)
        .def("n_cells", &DBNSSolver::nCells)
        .def(
            "primitive",
            [](const DBNSSolver &s, int cell) {
                if (cell < 0 || cell >= s.nCells())
                    throw py::index_error("cell index must lie in [0, n_cells)");
                return s.primitive(cell);
            },
            py::arg("cell"))
        .def("fields", &dbnsFields)
        .def(
            "set_target_correction",
            [](DBNSSolver &s, py::array_t<double> db, py::array_t<double> b, py::array_t<double> dq,
               bool energy_reach, py::array_t<bool> mask) {
                // db: (n_cells, 3, 3) STORED anisotropy discrepancy, the
                // operative injection input (db = 0 is exactly zero flux,
                // the discrete zero-correction contract); b: (n_cells, 3, 3)
                // absolute target anisotropy, used only for the running
                // realizability diagnostics; dq: (n_cells, 2) or (0,)
                // turbulent heat-flux correction in solver units
                // (<rho u_i''T''>/<rho>, m/s K). Values are copied into the
                // solver (no lifetime coupling, unlike the incompressible
                // forward model's reference semantics).
                const int nc = s.nCells();
                if (db.ndim() != 3 || db.shape(0) != nc || db.shape(1) != 3 || db.shape(2) != 3 ||
                    b.ndim() != 3 || b.shape(0) != nc || b.shape(1) != 3 || b.shape(2) != 3)
                    throw py::value_error(
                        "set_target_correction: db and b must have shape (n_cells, 3, 3)");
                if (dq.size() > 0 && (dq.ndim() != 2 || dq.shape(0) != nc || dq.shape(1) != 2))
                    throw py::value_error("set_target_correction: dq must have shape (n_cells, 2)");
                if (mask.size() > 0 && (mask.ndim() != 1 || mask.shape(0) != nc))
                    throw py::value_error("set_target_correction: mask must have shape (n_cells,)");

                // Compare full-width array dimensions before any conversion or copy.
                // Buffer sizes come from the admitted native mesh, not user row counts.
                auto adb = db.unchecked<3>();
                auto ab = b.unchecked<3>();
                std::vector<double> db6(6 * static_cast<size_t>(nc)), b6(db6.size());
                for (int ci = 0; ci < nc; ++ci) {
                    db6[6 * ci + 0] = adb(ci, 0, 0);
                    db6[6 * ci + 1] = adb(ci, 1, 1);
                    db6[6 * ci + 2] = adb(ci, 2, 2);
                    db6[6 * ci + 3] = adb(ci, 0, 1);
                    db6[6 * ci + 4] = adb(ci, 0, 2);
                    db6[6 * ci + 5] = adb(ci, 1, 2);
                    b6[6 * ci + 0] = ab(ci, 0, 0);
                    b6[6 * ci + 1] = ab(ci, 1, 1);
                    b6[6 * ci + 2] = ab(ci, 2, 2);
                    b6[6 * ci + 3] = ab(ci, 0, 1);
                    b6[6 * ci + 4] = ab(ci, 0, 2);
                    b6[6 * ci + 5] = ab(ci, 1, 2);
                }
                std::vector<double> dq2;
                if (dq.size() > 0) {
                    auto aq = dq.unchecked<2>();
                    dq2.resize(2 * static_cast<size_t>(nc));
                    for (int ci = 0; ci < nc; ++ci) {
                        dq2[2 * ci] = aq(ci, 0);
                        dq2[2 * ci + 1] = aq(ci, 1);
                    }
                }
                std::vector<std::uint8_t> m8;
                if (mask.size() > 0) {
                    auto am = mask.unchecked<1>();
                    m8.resize(static_cast<size_t>(nc));
                    for (int ci = 0; ci < nc; ++ci)
                        m8[ci] = am(ci) ? 1 : 0;
                }
                s.setTargetCorrection(db6, b6, dq2, energy_reach, m8);
            },
            py::arg("db"), py::arg("b"), py::arg("dq"), py::arg("energy_reach") = true,
            py::arg("mask") = py::array_t<bool>())
        .def("clear_target_correction", &DBNSSolver::clearTargetCorrection)
        .def(
            "limiter_active",
            [](const DBNSSolver &s) {
                const auto &v = s.limiterActive();
                py::array_t<bool> out((py::ssize_t)v.size());
                auto a = out.mutable_unchecked<1>();
                for (py::ssize_t i = 0; i < (py::ssize_t)v.size(); ++i)
                    a(i) = v[i] != 0;
                return out;
            },
            "Per-cell omega-production limiter activation of the last "
            "residual evaluation (the solver's own branch record).")
        .def(
            "reconstruction_limiter",
            [](const DBNSSolver &s) {
                auto v = s.reconstructionLimiter();
                py::ssize_t nc = (py::ssize_t)(v.size() / 6);
                py::array_t<double> out({nc, (py::ssize_t)6});
                auto a = out.mutable_unchecked<2>();
                for (py::ssize_t ci = 0; ci < nc; ++ci)
                    for (py::ssize_t k = 0; k < 6; ++k)
                        a(ci, k) = v[6 * ci + k];
                return out;
            },
            "Venkatakrishnan reconstruction-limiter values of the last "
            "evaluation (n_cells, 6), the checkpointable frozen-limiter "
            "state of a converged solve.")
        .def(
            "set_frozen_reconstruction_limiter",
            [](DBNSSolver &s, py::array_t<double> lim) {
                auto a = lim.unchecked<2>();
                if (a.shape(1) != 6)
                    throw std::runtime_error("set_frozen_reconstruction_limiter: lim must be "
                                             "(n_cells, 6)");
                std::vector<double> v(6 * a.shape(0));
                for (py::ssize_t ci = 0; ci < a.shape(0); ++ci)
                    for (py::ssize_t k = 0; k < 6; ++k)
                        v[6 * ci + k] = a(ci, k);
                s.setFrozenReconstructionLimiter(v);
            },
            py::arg("lim"),
            "Restore a checkpointed reconstruction-limiter state and start "
            "the implicit march limiter-frozen, so a reloaded converged "
            "state resumes the exact discrete operator it converged under.")
        .def("injection_diagnostics", [](const DBNSSolver &s) {
            // the realizability record is on the EFFECTIVE RUNNING
            // anisotropy b_eff(W) = b_B(W) + db_stored, diagnostic
            // only (never projected or clipped); the extremes carry
            // the residual-evaluation ordinal and the cell index
            const auto &d = s.injectionDiagnostics();
            py::dict out;
            out["active"] = d.active;
            out["checked_iters"] = d.checkedIters;
            out["all_realizable"] = d.allRealizable;
            out["min_margin"] = d.minMargin;
            out["min_margin_iter"] = d.minMarginIter;
            out["min_margin_cell"] = d.minMarginCell;
            out["max_violation"] = d.maxViolation;
            out["max_violation_iter"] = d.maxViolationIter;
            out["max_violation_cell"] = d.maxViolationCell;
            out["max_db"] = d.maxDb;
            out["max_dq"] = d.maxDq;
            return out;
        });

    py::class_<DBNSObservation>(m, "DBNSObservation")
        .def(py::init<const DBNSSolver &, const ReferenceState &>(), py::arg("solver"),
             py::arg("ref"), py::keep_alive<1, 2>())
        .def("wall", &dbnsWall, py::arg("patch"), py::arg("wall_temp"))
        .def("wall_profile", &dbnsWallProfile, py::arg("patch"), py::arg("wall_temps"),
             py::arg("fallback"));

    // Realizability projection (also reused by the Track B Python layer).
    m.def("project_reynolds_stress", &dbnsProjectStress, py::arg("xx"), py::arg("yy"),
          py::arg("zz"), py::arg("xy"), py::arg("xz") = 0.0, py::arg("yz") = 0.0,
          "Project a Reynolds-stress tensor into the realizable (barycentric) set.");

    m.def(
        "odd_even_energy_ratio",
        [](const Mesh &mesh, const std::vector<double> &values) {
            if ((int)values.size() != mesh.nCells()) {
                throw std::invalid_argument(
                    "odd_even_energy_ratio requires exactly one value per cell");
            }
            ScalarField phi(mesh, "probe");
            for (int ci = 0; ci < mesh.nCells(); ++ci)
                phi[ci] = values[ci];
            return oddEvenEnergyRatio(mesh, phi);
        },
        py::arg("mesh"), py::arg("values"),
        "Checkerboard energy ratio of a cell field (companion diagnostic to "
        "SolverSettings.legacy_rhie_chow_gate).");
}
