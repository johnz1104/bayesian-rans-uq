#include "ForwardModel.hpp"
#include "DBNSSolver.hpp"
#include <iostream>
#include <limits>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <cstring>
#include <csignal>
#include <sys/resource.h>

static void require(bool value) {
    if (!value)
        throw std::runtime_error("mesh validity assertion failed");
}
template <class F> static void stale(F action) {
    bool rejected = false;
    try {
        action();
    } catch (const std::logic_error &) {
        rejected = true;
    }
    require(rejected);
}
// Encode the historical native-endian MSH1 layout independently of the writer.
static std::vector<char> legacyMeshBytes(const Mesh &mesh) {
    std::vector<char> bytes;
    const auto scalar = [&](auto value) {
        const auto *raw = reinterpret_cast<const char *>(&value);
        bytes.insert(bytes.end(), raw, raw + sizeof(value));
    };
    const auto vec = [&](const Vec3 &v) {
        scalar(v.x);
        scalar(v.y);
        scalar(v.z);
    };
    const auto str = [&](const std::string &v) {
        scalar(static_cast<int>(v.size()));
        bytes.insert(bytes.end(), v.begin(), v.end());
    };
    scalar(std::uint32_t{0x4D534831});
    for (int count :
         {mesh.nCells(), mesh.nFaces(), mesh.nNodes(), mesh.nPatches(), mesh.nInternalFaces()})
        scalar(count);
    for (int n = 0; n < mesh.nNodes(); ++n)
        vec(mesh.node(n));
    for (const auto &face : mesh.faces()) {
        scalar(face.owner);
        scalar(face.neighbor);
        scalar(face.patchID);
        vec(face.center);
        vec(face.normal);
        scalar(face.area);
        scalar(face.delta);
        vec(face.d);
        scalar(face.weight);
    }
    for (const auto &cell : mesh.cells()) {
        vec(cell.center);
        scalar(cell.volume);
        scalar(static_cast<int>(cell.faces.size()));
        for (int face : cell.faces)
            scalar(face);
    }
    for (const auto &patch : mesh.patches()) {
        str(patch.name);
        str(patch.type);
        scalar(static_cast<int>(patch.faces.size()));
        for (int face : patch.faces)
            scalar(face);
    }
    return bytes;
}
static void persistence() {
    const auto path =
        std::filesystem::temp_directory_path() /
        ("cfdr-mesh-contract-" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".msh");
    struct Cleanup {
        std::filesystem::path path;
        ~Cleanup() {
            std::error_code error;
            std::filesystem::remove(path, error);
        }
    } cleanup{path};
    const auto write = [&](const std::vector<char> &bytes) {
        std::ofstream out(path, std::ios::binary);
        out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        out.close();
        require(static_cast<bool>(out));
    };
    const auto reject = [&](const std::vector<char> &bytes) {
        write(bytes);
        bool rejected = false;
        try {
            (void)Mesh::loadFromFile(path.string());
        } catch (const std::runtime_error &error) {
            rejected = std::string(error.what()).find("mesh") != std::string::npos;
        }
        require(rejected);
    };
    const auto close = [](Vec3 a, Vec3 b) { return (a - b).norm() < 1e-13; };
    // Trigger both buffered-close and mid-write failures without depending on
    // a platform-specific full device. Limits and the signal handler are scoped
    // to this test process and restored before its assertions or diagnostics.
    for (const auto &mesh :
         {Mesh::makeChannel2D(2, 2, 1., 1.), Mesh::makeChannel2D(20, 4, 1., 1.)}) {
        bool rejected = false;
        {
            struct RestoreLimit {
                rlimit limit{};
                using Handler = void (*)(int);
                Handler handler;
                RestoreLimit() {
                    require(getrlimit(RLIMIT_FSIZE, &limit) == 0);
                    handler = std::signal(SIGXFSZ, SIG_IGN);
                    require(handler != SIG_ERR);
                    rlimit restricted = limit;
                    restricted.rlim_cur = std::min<rlim_t>(limit.rlim_cur, 64);
                    if (setrlimit(RLIMIT_FSIZE, &restricted) != 0) {
                        std::signal(SIGXFSZ, handler);
                        throw std::runtime_error("cannot install mesh writer failure control");
                    }
                }
                ~RestoreLimit() {
                    if (setrlimit(RLIMIT_FSIZE, &limit) != 0)
                        std::terminate();
                    std::signal(SIGXFSZ, handler);
                }
            } restore;
            try {
                mesh.saveBinary(path.string());
            } catch (const std::runtime_error &error) {
                rejected = std::string(error.what()).find("mesh") != std::string::npos;
            }
        }
        require(rejected && std::filesystem::file_size(path) <= 64);
    }

    for (const Mesh &mesh :
         {Mesh::makeChannel2D(3, 4, 1., 1.), Mesh::makePeriodicChannel2D(3, 4, 1., 1., 180.),
          Mesh::makeCurvedChannelPeriodic2D({0., .2, .7, 1.}, {.1, .15, .1, .1}, 1., 4, 0.)}) {
        mesh.saveBinary(path.string());
        auto loaded = Mesh::loadFromFile(path.string());
        require(loaded.nCells() == mesh.nCells() && loaded.nFaces() == mesh.nFaces() &&
                loaded.nNodes() == mesh.nNodes() && loaded.nPatches() == mesh.nPatches());
        require(loaded.ownerList() == mesh.ownerList() &&
                loaded.neighborList() == mesh.neighborList());
        require(!loaded.wallDistanceValid());
        for (int n = 0; n < mesh.nNodes(); ++n)
            require(close(loaded.node(n), mesh.node(n)));
        for (int f = 0; f < mesh.nFaces(); ++f) {
            const auto &a = mesh.face(f);
            const auto &b = loaded.face(f);
            require(a.owner == b.owner && a.neighbor == b.neighbor && a.patchID == b.patchID);
            require(close(a.center, b.center) && close(a.normal, b.normal) && close(a.d, b.d) &&
                    close(a.shift, b.shift));
            require(a.area == b.area && a.delta == b.delta && a.weight == b.weight);
            if (!b.isBoundary())
                require(close(
                    loaded.cell(b.neighbor).center + b.shift - loaded.cell(b.owner).center, b.d));
        }
        for (int c = 0; c < mesh.nCells(); ++c)
            require(close(loaded.cell(c).center, mesh.cell(c).center) &&
                    loaded.cell(c).volume == mesh.cell(c).volume &&
                    loaded.cell(c).faces == mesh.cell(c).faces);
        for (int p = 0; p < mesh.nPatches(); ++p)
            require(loaded.patch(p).name == mesh.patch(p).name &&
                    loaded.patch(p).type == mesh.patch(p).type &&
                    loaded.patch(p).faces == mesh.patch(p).faces &&
                    loaded.patchByName(mesh.patch(p).name) == p);
        loaded.computeGeometry();
        for (int c = 0; c < mesh.nCells(); ++c)
            require(close(loaded.cell(c).center, mesh.cell(c).center) &&
                    std::abs(loaded.cell(c).volume - mesh.cell(c).volume) < 1e-13);
    }
    const auto legacy = legacyMeshBytes(Mesh::makeChannel2D(2, 2, 1., 1.));
    write(legacy);
    auto imported = Mesh::loadFromFile(path.string());
    require(imported.nCells() == 4 && imported.cell(0).volume > 0.);
    imported.computeGeometry();
    // Every possible truncation of this tiny historical fixture must reject.
    for (size_t end = 0; end < legacy.size(); ++end)
        reject(
            std::vector<char>(legacy.begin(), legacy.begin() + static_cast<std::ptrdiff_t>(end)));
    // Counts are checked before allocating or indexing. Corrupt each header count.
    for (size_t field = 1; field <= 5; ++field) {
        auto bytes = legacy;
        const int bad = -1;
        std::memcpy(bytes.data() + field * sizeof(int), &bad, sizeof bad);
        reject(bytes);
        bytes = legacy;
        const int huge = std::numeric_limits<int>::max();
        std::memcpy(bytes.data() + field * sizeof(int), &huge, sizeof huge);
        reject(bytes);
    }
    const auto fixture = Mesh::makeChannel2D(2, 2, 1., 1.);
    const size_t faceStart = 24 + static_cast<size_t>(fixture.nNodes()) * 24;
    const size_t cellStart = faceStart + static_cast<size_t>(fixture.nFaces()) * 108;
    const auto corrupt = [&](size_t offset, auto bad) {
        auto bytes = legacy;
        require(offset + sizeof bad <= bytes.size());
        std::memcpy(bytes.data() + offset, &bad, sizeof bad);
        reject(bytes);
    };
    corrupt(0, std::uint32_t{0});                          // unsupported format
    corrupt(24, std::numeric_limits<double>::quiet_NaN()); // first node
    corrupt(faceStart, -1);                                // owner
    corrupt(faceStart + 4, fixture.nCells());              // neighbour
    corrupt(faceStart + 8, 0);                 // an internal face cannot belong to a patch
    corrupt(faceStart + 60, 0.);               // area
    corrupt(faceStart + 68, 0.);               // distance
    corrupt(faceStart + 76, 123.);             // displacement disagrees with cells
    corrupt(faceStart + 100, 2.);              // interpolation weight
    corrupt(cellStart + 24, -1.);              // cell volume
    corrupt(cellStart + 32, -1);               // cell face count
    corrupt(cellStart + 36, fixture.nFaces()); // cell face index
    size_t patchStart = cellStart;
    for (const auto &cell : fixture.cells())
        patchStart += 36 + cell.faces.size() * sizeof(int);
    corrupt(patchStart, -1); // patch-name length
    const auto &patch = fixture.patch(0);
    const size_t patchFaces = patchStart + 12 + patch.name.size() + patch.type.size();
    corrupt(patchFaces, 0); // an internal face cannot be a boundary patch member
    // Swapping two boundary members preserves the total incidence counts but
    // must still reject a face assigned to the wrong named patch.
    const auto &otherPatch = fixture.patch(1);
    const size_t otherPatchStart = patchFaces + patch.faces.size() * sizeof(int);
    const size_t otherPatchFaces =
        otherPatchStart + 12 + otherPatch.name.size() + otherPatch.type.size();
    auto swapped = legacy;
    const int first = patch.faces.front(), second = otherPatch.faces.front();
    std::memcpy(swapped.data() + patchFaces, &second, sizeof second);
    std::memcpy(swapped.data() + otherPatchFaces, &first, sizeof first);
    reject(swapped);
    auto trailing = legacy;
    trailing.push_back('x');
    reject(trailing);
    // Old periodic files cannot recover the discarded translation unambiguously.
    reject(legacyMeshBytes(Mesh::makePeriodicChannel2D(3, 4, 1., 1., 180.)));
}
int main() {
    persistence();
    const auto rejects = [](auto factory) {
        bool rejected = false;
        try {
            (void)factory();
        } catch (const std::invalid_argument &) {
            rejected = true;
        }
        require(rejected);
    };
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    for (double bad : {nan, inf, 0., -1.}) {
        rejects([&] { return Mesh::makeChannel2D(3, 4, bad, 1.); });
        rejects([&] { return Mesh::makeChannel2D(3, 4, 1., bad); });
        rejects([&] { return Mesh::makeChannel2D(3, 4, 1., 1., bad); });
        rejects([&] { return Mesh::makeChannel2D(3, 4, 1., 1., 100., bad); });
        rejects([&] { return Mesh::makePlate2D(3, 4, 1., 1., bad); });
        rejects([&] { return Mesh::makePeriodicChannel2D(3, 4, 1., 1., bad); });
        rejects([&] { return Mesh::makePeriodicChannel2D(3, 4, 1., bad, 180.); });
        rejects([&] { return Mesh::makePeriodicChannel2D(3, 4, 1., 1., 180., bad); });
        rejects([&] { return Mesh::makeBackwardFacingStep2D(2, 2, 2, 2, bad, 1., .5, 1.); });
    }
    for (int bad : {-1, 0, std::numeric_limits<int>::max()}) {
        rejects([&] { return Mesh::makeChannel2D(bad, 4, 1., 1.); });
        rejects([&] { return Mesh::makeChannel2D(3, bad, 1., 1.); });
        rejects([&] { return Mesh::makePlate2D(bad, 4, 1., 1., 100.); });
        rejects([&] { return Mesh::makePeriodicChannel2D(3, bad, 1., 1., 180.); });
        rejects([&] { return Mesh::makeBackwardFacingStep2D(2, bad, 2, 2, 1., 1., .5, 1.); });
    }
    rejects([&] { return Mesh::makeBackwardFacingStep2D(2, 2, 2, 2, 1., 1., 1., 1.); });
    rejects([&] { return Mesh::makeBackwardFacingStep2D(2, 2, 2, 2, 1., 1., .5, 1., -1.); });
    const std::vector<double> xs{0., 1., 2., 3.}, bottom(4, 0.);
    for (auto badXs : {std::vector<double>{0., 1., 1., 3.}, std::vector<double>{0., 2., 1., 3.},
                       std::vector<double>{0., nan, 2., 3.}})
        rejects([&] { return Mesh::makeCurvedChannelPeriodic2D(badXs, bottom, 1., 4, 0.); });
    rejects([&] { return Mesh::makeCurvedChannelPeriodic2D(xs, {0., 2., 0., 0.}, 1., 4, 0.); });
    rejects([&] { return Mesh::makeCurvedChannelPeriodic2D(xs, {0., nan, 0., 0.}, 1., 4, 0.); });
    rejects([&] { return Mesh::makeCurvedChannelPeriodic2D(xs, bottom, 1., 0, 0.); });
    rejects([&] { return Mesh::makeCurvedChannelPeriodic2D(xs, bottom, 1., 4, nan); });
    require(Mesh::makeCurvedChannelPeriodic2D(xs, bottom, 1., 4, 0.).nCells() == 12);
    require(Mesh::makeBackwardFacingStep2D(2, 2, 2, 2, 1., 1., .5, 1.).nCells() == 12);

    Mesh mesh = Mesh::makeChannel2D(3, 4, 1, 1);
    require(!mesh.wallDistanceValid());
    mesh.computeWallDistance();
    const auto distances = mesh.wallDistance();
    const auto original = mesh.stamp();
    Mesh copy(mesh);
    require(copy.stamp() != original && copy.wallDistance() == distances);
    auto copyStamp = copy.stamp();
    Mesh &sameCopy = copy;
    copy = sameCopy;
    copy = std::move(sameCopy);
    require(copy.stamp() == copyStamp);
    FlowFields fields(mesh);
    auto bc = FlowBoundaryConditions::channelDefaults(mesh, 1, 1e-4, 1);
    SolverSettings settings;
    settings.verbose = false;
    settings.maxIterations = 1;
    SIMPLESolver solver(mesh, SSTModel{}, bc, .01, settings);
    solver.initUniform(fields, {1, 0, 0}, 0, 1e-4, 1);
    dbns::DBNSBoundaryConditions dbcs;
    for (const auto &patch : mesh.patches()) {
        dbns::BoundarySpec spec;
        spec.kind = dbns::BoundaryKind::Extrapolate;
        dbcs.set(patch.name, spec);
    }
    dbns::DBNSSolver dbnsSolver(mesh, IdealGasEOS{}, SSTCoefficients{}, dbcs);
    dbnsSolver.initUniform({1, 1, 0, 1, 0, 0});
    WarmStartCache cache;
    cache.store({1}, fields);
    mesh.setPatchType("top", "symmetry");
    require(!mesh.wallDistanceValid());
    stale([&] { (void)mesh.wallDistance(); });
    stale([&] { (void)fields.U[0]; });
    stale([&] { (void)dbnsSolver.primitive(0); });
    stale([&] { dbnsSolver.prepareProperties(); });
    stale([&] { dbnsSolver.initUniform({1, 1, 0, 1, 0, 0}); });
    mesh.computeWallDistance();
    FlowFields reconstructed(mesh);
    stale([&] { solver.initUniform(reconstructed, {1, 0, 0}, 0, 1e-4, 1); });
    stale([&] { cache.findNearest({1}, reconstructed); });
    SIMPLESolver rebuilt(mesh, SSTModel{},
                         FlowBoundaryConditions::channelDefaults(mesh, 1, 1e-4, 1), .01, settings);
    rebuilt.initUniform(reconstructed, {1, 0, 0}, 0, 1e-4, 1);
    const auto beforeGeometry = mesh.stamp();
    mesh.computeGeometry();
    require(mesh.stamp() != beforeGeometry && !mesh.wallDistanceValid());
    stale([&] { (void)reconstructed.k.data(); });
    mesh = copy; // same-size replacement must still invalidate destination state
    require(mesh.stamp() != original && mesh.wallDistanceValid());
    FlowFields movedSource(copy);
    Mesh moved(std::move(copy));
    require(moved.wallDistanceValid());
    stale([&] { (void)movedSource.U.mesh(); });
    const auto destination = mesh.stamp();
    mesh = std::move(moved);
    require(mesh.stamp() != destination && !moved.wallDistanceValid());
    std::cout << "mesh copy/move/revision, wall validity and stale state checks passed\n";
}
