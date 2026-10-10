#include "Mesh.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <iostream>
#include <filesystem>
#include <queue>
#include <numeric>
#include <cstring>

namespace {
void requirePositive(double value, const char *name) {
    if (!std::isfinite(value) || value <= 0.)
        throw std::invalid_argument(std::string(name) + " must be finite and positive");
}

void requireMeshSize(int nx, int ny) {
    if (nx <= 0 || ny <= 0)
        throw std::invalid_argument("mesh cell counts must be positive");
    const auto x = static_cast<std::uint64_t>(nx), y = static_cast<std::uint64_t>(ny);
    const auto limit = static_cast<std::uint64_t>(std::numeric_limits<int>::max());
    if ((x + 1) * (y + 1) > limit / 2 || 2 * x * y + x + y > limit)
        throw std::invalid_argument("mesh counts exceed the supported integer indexing range");
}

void requireRectangle(int nx, int ny, double length, double height) {
    requireMeshSize(nx, ny);
    requirePositive(length, "mesh length");
    requirePositive(height, "mesh height");
    requirePositive(length * height, "mesh area");
}

void requireOptionalReynolds(double reynolds) {
    if (!std::isfinite(reynolds) || reynolds < 0.)
        throw std::invalid_argument("optional mesh Reynolds number must be finite and nonnegative");
}
} // namespace

// loadFromFile supported mesh formats:
//      - .foam / directory path    => OpenFoam polyMesh
//      - .msh                      => custom bindary format
// For OpenFOAM, pass case directory or the path to the the polyMesh directory

Mesh Mesh::loadFromFile(const std::string &path) {
    namespace fs = std::filesystem;

    // Directory-based dispatch
    // checks if path is a directory (treat as OpenFOAM case) or a direct polyMesh directory
    if (fs::is_directory(path)) {
        fs::path p(path);
        if (fs::exists(p / "points") && fs::exists(p / "faces")) {
            return loadOpenFOAM(path);
        }
        fs::path polyMesh = p / "constant" / "polyMesh";
        if (fs::exists(polyMesh / "points")) {
            return loadOpenFOAM(polyMesh.string());
        }
        throw std::runtime_error("Cannot find polyMesh in: " + path);
    }

    // File-based dispatch
    auto ext = fs::path(path).extension().string();
    if (ext == ".foam" || ext == "") {
        return loadOpenFOAM(path);
    }
    if (ext == ".msh") {
        return loadBinary(path);
    }
    throw std::runtime_error("Unknown mesh format: " + ext);
}

// OpenFOAM polyMesh reader (OpenFOAM mesh usually defined by 5 files)
// points   -- node coordinates
// faces    -- face node connectives
// owner    -- owner cell for each face
// neighbor -- neighbor cell for internal faces
// boundary -- patch definitions

// Parsing Helper: skips OpenfOAM file header until we find the start of data
static void skipFoamHeader(std::istream &is) {
    std::string line;
    while (std::getline(is, line)) {
        auto pos = line.find("//");
        if (pos != std::string::npos)
            line.erase(pos);
        line.erase(0, line.find_first_not_of(" \t\r"));
        if (line == "(")
            return;
    }
    throw std::runtime_error("Could not find data block");
}

// Parsing helper: reads the element count from an OpenFOAM data file
static int readFoamCount(std::istream &is) {
    std::string line;
    while (std::getline(is, line)) {
        auto pos = line.find("//");
        if (pos != std::string::npos)
            line.erase(pos);
        line.erase(0, line.find_first_not_of(" \t\r"));
        line.erase(line.find_last_not_of(" \t\r") + 1);
        if (line.empty() || line[0] == '/' || line[0] == 'F' || line[0] == 'O' || line[0] == 'o' ||
            line[0] == 'v' || line[0] == 'C' || line[0] == 'a' || line[0] == '{' || line[0] == '}')
            continue;
        try {
            return std::stoi(line);
        } catch (...) {
            continue;
        }
    }
    throw std::runtime_error("Could not read count");
}

Mesh Mesh::loadOpenFOAM(const std::string &polyMeshDir) {
    namespace fs = std::filesystem;
    Mesh m;

    // extracts OpenFOAM mesh point data (defines geometric node coordinates)
    {
        std::ifstream ifs(fs::path(polyMeshDir) / "points");
        if (!ifs)
            throw std::runtime_error("Cannot open points file");
        int nPts = readFoamCount(ifs);
        skipFoamHeader(ifs);
        m.nodes_.resize(nPts);
        for (int i = 0; i < nPts; ++i) {
            char c;
            ifs >> c; // '('
            ifs >> m.nodes_[i].x >> m.nodes_[i].y >> m.nodes_[i].z;
            ifs >> c; // ')'
        }
    }

    // extracts OpenFOAM mesh face data (defines face-node connectivity topology)
    std::vector<std::vector<int>> faceNodes;
    {
        std::ifstream ifs(fs::path(polyMeshDir) / "faces");
        if (!ifs)
            throw std::runtime_error("Cannot open faces file");
        int nF = readFoamCount(ifs);
        skipFoamHeader(ifs);
        faceNodes.resize(nF);
        for (int i = 0; i < nF; ++i) {
            int nv;
            ifs >> nv;
            char c;
            ifs >> c; // '('
            faceNodes[i].resize(nv);
            for (int j = 0; j < nv; ++j)
                ifs >> faceNodes[i][j];
            ifs >> c; // ')'
        }
        m.faces_.resize(nF);
    }

    // extracts OpenFOAM mesh owner data (defines face-cell connectivitytopology)
    {
        std::ifstream ifs(fs::path(polyMeshDir) / "owner");
        if (!ifs)
            throw std::runtime_error("Cannot open owner file in " + polyMeshDir);
        int nOwn = readFoamCount(ifs);
        skipFoamHeader(ifs);
        for (int i = 0; i < nOwn; ++i) {
            int o;
            ifs >> o;
            m.faces_[i].owner = o;
        }
    }

    // extracts OpenFOAM mesh neighbor data (defines adjacent cell assignments internal faces)
    {
        fs::path nbrPath = fs::path(polyMeshDir) / "neighbour";
        std::ifstream ifs(nbrPath);
        if (!ifs)
            throw std::runtime_error("Cannot open neighbour file in " + polyMeshDir);
        int nNbr = readFoamCount(ifs);
        m.nInternal_ = nNbr;
        skipFoamHeader(ifs);
        for (int i = 0; i < nNbr; ++i) {
            int n;
            ifs >> n;
            m.faces_[i].neighbor = n;
        }
    }

    // determine total cell count from face owner and neighbor indices
    int nCells = 0;
    for (auto &f : m.faces_) {
        nCells = std::max(nCells, f.owner + 1);
        if (f.neighbor >= 0)
            nCells = std::max(nCells, f.neighbor + 1);
    }
    m.cells_.resize(nCells);

    // build cell-face adjacency (assigns each face to its owner and neighbor cells)
    for (int fi = 0; fi < static_cast<int>(m.faces_.size()); ++fi) {
        m.cells_[m.faces_[fi].owner].faces.push_back(fi);
        if (m.faces_[fi].neighbor >= 0)
            m.cells_[m.faces_[fi].neighbor].faces.push_back(fi);
    }

    // reads OpenFOAM boundary patch definitions and construct patch objects
    // assigns boundary faces to patches and label faces with their patch ID
    {
        std::ifstream ifs(fs::path(polyMeshDir) / "boundary");
        if (ifs) {
            int nPatches = readFoamCount(ifs);
            skipFoamHeader(ifs);
            m.patches_.resize(nPatches);
            for (int p = 0; p < nPatches; ++p) {
                std::string line;
                // read patch name
                while (std::getline(ifs, line)) {
                    auto pos = line.find("//");
                    if (pos != std::string::npos)
                        line.erase(pos);
                    line.erase(0, line.find_first_not_of(" \t\r"));
                    line.erase(line.find_last_not_of(" \t\r") + 1);
                    if (!line.empty() && line != "{" && line != "}") {
                        m.patches_[p].name = line;
                        break;
                    }
                }
                // parse patch body { type ...; nFaces ...; startFace ...; }
                int pnFaces = 0, startFace = 0;
                while (std::getline(ifs, line)) {
                    auto pos = line.find("//");
                    if (pos != std::string::npos)
                        line.erase(pos);
                    line.erase(0, line.find_first_not_of(" \t\r"));
                    if (line.find('}') != std::string::npos)
                        break;
                    std::istringstream ss(line);
                    std::string key;
                    ss >> key;
                    if (key == "type") {
                        std::string val;
                        ss >> val;
                        if (!val.empty() && val.back() == ';')
                            val.pop_back();
                        m.patches_[p].type = val;
                    } else if (key == "nFaces") {
                        ss >> pnFaces;
                    } else if (key == "startFace") {
                        ss >> startFace;
                    }
                }
                m.patches_[p].faces.resize(pnFaces);
                for (int i = 0; i < pnFaces; ++i) {
                    int fIdx = startFace + i;
                    m.patches_[p].faces[i] = fIdx;
                    m.faces_[fIdx].patchID = p;
                }
            }
        }
    }
    // Mesh structure now: (-> means maps to)
    // Topology: face -> owner, face-> neighbor, cell -> faces
    // Boundary structure: patch -> faces, face -> patche, patch -> type

    // build geometry from topology
    // computes face centroid, area, and normal vector
    for (int fi = 0; fi < static_cast<int>(m.faces_.size()); ++fi) {
        const auto &fn = faceNodes[fi];
        int nv = static_cast<int>(fn.size());

        Vec3 ctr{}; // node average: the fan apex
        for (int j = 0; j < nv; ++j) {
            ctr = ctr + m.nodes_[fn[j]];
        }
        ctr = ctr / static_cast<double>(nv);

        Vec3 areaNormal{}; // compute face area normal vector
        for (int j = 0; j < nv; ++j) {
            const Vec3 &a = m.nodes_[fn[j]];
            const Vec3 &b = m.nodes_[fn[(j + 1) % nv]];
            areaNormal = areaNormal + (a - ctr).cross(b - ctr) * 0.5;
        }
        m.faces_[fi].area = areaNormal.norm();   // face area (scalar)
        m.faces_[fi].normal = areaNormal.unit(); // face unit normal

        // Area centroid from the triangle fan about the node average. The
        // node average itself is the centroid only for parallelograms; on a
        // trapezoidal front/back face it would move every cell centre built
        // from it.
        Vec3 csum{};
        double asum = 0.0;
        for (int j = 0; j < nv; ++j) {
            const Vec3 &a = m.nodes_[fn[j]];
            const Vec3 &b = m.nodes_[fn[(j + 1) % nv]];
            double at = 0.5 * (a - ctr).cross(b - ctr).dot(m.faces_[fi].normal);
            csum = csum + (ctr + a + b) * (at / 3.0);
            asum += at;
        }
        m.faces_[fi].center = (std::abs(asum) > 1e-300) ? csum / asum : ctr;
    }

    // build owner/neighbor connectivity arrays
    m.ownerList_.resize(m.nInternal_); // m.nInternal_ = number of internal faces.
    m.neighborList_.resize(m.nInternal_);
    for (int fi = 0; fi < m.nInternal_; ++fi) {
        m.ownerList_[fi] = m.faces_[fi].owner;
        m.neighborList_[fi] = m.faces_[fi].neighbor;
    }

    // compute cell centers, volumes, interpolation weights, delta vectors
    m.computeCellCentersAndVolumes();
    m.computeFaceGeometry();
    m.ComputeInterpolationWeights();

    // prints summary
    std::cout << "OpenFOAM mesh loaded: " << m.nCells() << " cells, " << m.nFaces() << " faces, "
              << m.nInternal_ << " internal faces, " << m.nPatches() << " patches\n";
    return m;
}

// Geometry Computations

// Exact cell volumes and centroids, the ONE convention every mesh path uses.
//
// The cell is split into pieces that share an apex c0 (the average of the
// face centres), one piece per face, with signed size
// h_f = (c_f - c0) . n_out S_f. Two kinds of cell occur:
//   - closed polyhedra (every OpenFOAM cell, front/back faces included):
//     pyramids, V = sum h_f / 3, centroid = sum h_f (3/4 c_f + 1/4 c0) / sum h_f;
//   - z-open prisms (the 2D generator cells, which carry no front/back
//     faces): triangles extruded in z, V = sum h_f / 2,
//     centroid = sum h_f (c0 + 2 c_f) / 3 / sum h_f.
// A cell is z-open when none of its faces has a z normal component. Both are
// exact for planar faces with exact face centroids; for the generators'
// straight-edged quads they reproduce shoelace area times depth and the
// polygon centroid. The old single formula |sum h_f| / 3 gave 2/3 of the
// volume of a z-open cell and the face-centre average gave the centroid.
// On a periodic wrap face the neighbour sees the face at c_f - shift.
void Mesh::computeCellCentersAndVolumes() {
    for (int ci = 0; ci < nCells(); ++ci) {
        Cell &cell = cells_[ci];
        auto localCenter = [&](const Face &f) {
            return (f.owner == ci) ? f.center : f.center - f.shift;
        };

        Vec3 c0{};
        bool zOpen = true;
        for (FaceID fi : cell.faces) {
            c0 = c0 + localCenter(faces_[fi]);
            if (std::abs(faces_[fi].normal.z) > 1e-9)
                zOpen = false;
        }
        c0 = c0 / static_cast<double>(cell.faces.size());

        double hsum = 0.0;
        Vec3 csum{};
        for (FaceID fi : cell.faces) {
            const Face &f = faces_[fi];
            double sign = (f.owner == ci) ? 1.0 : -1.0;
            Vec3 cf = localCenter(f);
            double h = sign * (cf - c0).dot(f.normal) * f.area;
            Vec3 cPiece = zOpen ? (c0 + cf * 2.0) / 3.0 : cf * 0.75 + c0 * 0.25;
            hsum += h;
            csum = csum + cPiece * h;
        }
        cell.volume = hsum / (zOpen ? 2.0 : 3.0);
        cell.center = (std::abs(hsum) > 1e-300) ? csum / hsum : c0;
        if (cell.volume < 1e-30) {
            cell.volume = 1e-20; // fallback for degenerate or nearly flat cells
        }
    }
}

// computes face-cell distance vectors and magnitudes
void Mesh::computeFaceGeometry() {
    // delta = distance between owner and neighbor centers
    // d     = vector from owner to neighbor centers
    // (the neighbour of a periodic wrap face is taken at its image, centre + shift)
    for (int fi = 0; fi < nFaces(); ++fi) {
        Face &f = faces_[fi];
        if (!f.isBoundary()) {
            f.d = cells_[f.neighbor].center + f.shift - cells_[f.owner].center;
            f.delta = f.d.norm();
            // owner -> neighbour must cross the face along its normal; a wrap
            // face whose periodic shift was lost (e.g. a periodic mesh read
            // back from the historical MSH1 format) fails
            // here instead of silently spanning the whole domain
            if (f.d.dot(f.normal) <= 0.0)
                throw std::runtime_error(
                    "Mesh::computeFaceGeometry: internal face " + std::to_string(fi) +
                    " has its neighbour behind it; periodic face without its shift?");
        } else {
            f.d = f.center - cells_[f.owner].center;
            f.delta = f.d.norm();
        }
        if (f.delta < 1e-30)
            f.delta = 1e-20;
    }
}

// computes linear interpolation weights for internal faces from cell-face center distances
void Mesh::ComputeInterpolationWeights() {
    // weight = |fC - N| / (|fC - P| + |fC - N|)
    // phi_f = w * phi_P + (1-w) * phi_N
    for (int fi = 0; fi < nInternal_; ++fi) {
        Face &f = faces_[fi];
        double dP = (f.center - cells_[f.owner].center).norm();
        double dN = (f.center - (cells_[f.neighbor].center + f.shift)).norm();
        double sum = dP + dN;
        f.weight = (sum > 1e-30) ? dN / sum : 0.5;
    }
}

// converts topological mesh into a discretization-ready computational grid
void Mesh::computeGeometry() {
    invalidate();
    computeCellCentersAndVolumes();
    computeFaceGeometry();
    ComputeInterpolationWeights();

    ownerList_.resize(nInternal_);
    neighborList_.resize(nInternal_);
    for (int fi = 0; fi < nInternal_; ++fi) {
        ownerList_[fi] = faces_[fi].owner;
        neighborList_[fi] = faces_[fi].neighbor;
    }
}

// computeWallDistance — brute-force from wall patches
void Mesh::computeWallDistance() {
    wallDistanceValid_ = false;
    wallDist_.assign(nCells(), std::numeric_limits<double>::max());

    // collects wall face centers
    std::vector<Vec3> wallPts;
    for (const auto &p : patches_) {
        if (p.type == "wall") {
            for (FaceID fi : p.faces) {
                wallPts.push_back(faces_[fi].center);
            }
        }
    }
    // handles no wall case
    if (wallPts.empty()) {
        std::fill(wallDist_.begin(), wallDist_.end(), 1e10);
        wallDistanceValid_ = true;
        return;
    }
    // brute force distance search
    // computes minimum Euclidean distance from each cell center to any wall face center
    for (int ci = 0; ci < nCells(); ++ci) {
        double minD = std::numeric_limits<double>::max();
        const Vec3 &cc = cells_[ci].center;
        for (const Vec3 &wp : wallPts) {
            double d = (cc - wp).norm();
            if (d < minD)
                minD = d;
        }
        wallDist_[ci] = minD;
    }
    wallDistanceValid_ = true;
}

// lookup mechanism to retrieve a boundary patch index by its name
PatchID Mesh::patchByName(const std::string &name) const {
    for (int i = 0; i < static_cast<int>(patches_.size()); ++i) {
        if (patches_[i].name == name)
            return i;
    }
    throw std::runtime_error("Patch not found: " + name);
}

// retype a boundary patch; the caller re-runs computeWallDistance so a patch
// retyped away from "wall" stops contributing wall distances (SST blending)
void Mesh::setPatchType(const std::string &name, const std::string &type) {
    auto &patch = patches_[patchByName(name)];
    if (patch.type == type)
        return;
    invalidate();
    patch.type = type;
}

// Streamwise-periodic curved-bottom channel (see the header note). All cell and
// face geometry is computed HERE, exactly, from the quad corners: the generic
// helpers cannot be used because (a) the wrap face's geometric center sits at
// x = Lx while its neighbor cell sits at x ~ 0, so face-center-averaged cell
// centers and center-to-center distances would span the whole domain, and (b)
// the wrap face's d/delta/weight must use the PERIODIC IMAGE of the neighbor.
Mesh Mesh::makeCurvedChannelPeriodic2D(const std::vector<double> &xNodes,
                                       const std::vector<double> &yBottom, double yTop, int ny,
                                       double Re, double yPlusTarget) {
    requireOptionalReynolds(Re); // Zero selects the documented default stretching.
    requirePositive(yPlusTarget, "mesh y-plus target");
    // first-cell height from a skin-friction correlation at this Re (the
    // historical sizing; makePeriodicChannel2D sizes it from Re_tau instead)
    double y1t = 0.0;
    if (Re > 0 && static_cast<int>(xNodes.size()) == static_cast<int>(yBottom.size())) {
        double meanH = 0.0;
        for (size_t i = 0; i < yBottom.size(); ++i)
            meanH += (yTop - yBottom[i]);
        meanH /= static_cast<double>(yBottom.size());
        double Cf = 0.058 * std::pow(Re, -0.2);
        double uTau = std::sqrt(Cf / 2.0);
        double nu = meanH / Re;
        y1t = yPlusTarget * nu / uTau;
    }
    return buildPeriodic2D(xNodes, yBottom, yTop, ny, y1t, "bottom_wall", "top_wall");
}

Mesh Mesh::makePeriodicChannel2D(int nx, int ny, double Lx, double h, double ReTau,
                                 double yPlusTarget) {
    requireRectangle(nx, ny, Lx, 2.0 * h);
    requirePositive(ReTau, "friction Reynolds number");
    requirePositive(yPlusTarget, "mesh y-plus target");
    std::vector<double> xN(nx + 1), yB(nx + 1, 0.0);
    for (int i = 0; i <= nx; ++i)
        xN[i] = Lx * i / nx;
    // u_tau = 1 units: nu = h / Re_tau, so y1 = y+ nu / u_tau = y+ h / Re_tau
    return buildPeriodic2D(xN, yB, 2.0 * h, ny, yPlusTarget * h / ReTau, "bottom", "top");
}

Mesh Mesh::buildPeriodic2D(const std::vector<double> &xNodes, const std::vector<double> &yBottom,
                           double yTop, int ny, double y1t, const char *bottomName,
                           const char *topName) {
    if (xNodes.size() > static_cast<size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("periodic node count exceeds the supported indexing range");
    const int nx = static_cast<int>(xNodes.size()) - 1;
    if (nx < 3 || static_cast<int>(yBottom.size()) != nx + 1)
        throw std::invalid_argument("makeCurvedChannelPeriodic2D: need nx+1 x nodes "
                                    "and matching yBottom samples");
    requireMeshSize(nx, ny);
    if (!std::isfinite(yTop) || !std::isfinite(y1t) || y1t < 0.)
        throw std::invalid_argument("periodic mesh top and wall spacing must be finite");
    for (size_t i = 0; i < xNodes.size(); ++i) {
        if (!std::isfinite(xNodes[i]) || (i > 0 && xNodes[i] <= xNodes[i - 1]))
            throw std::invalid_argument("periodic x nodes must be finite and strictly increasing");
        requirePositive(yTop - yBottom[i], "periodic mesh local height");
    }
    if (std::abs(yBottom.front() - yBottom.back()) > 1e-12)
        throw std::invalid_argument("makeCurvedChannelPeriodic2D: yBottom must be "
                                    "periodic (first == last sample)");
    const double Lx = xNodes.back() - xNodes.front();
    requirePositive(Lx, "periodic mesh length");
    const double dz = 1.0;

    // wall-normal distribution eta_j in [0,1], symmetric tanh clustering toward
    // both walls; the stretch is solved on the MEAN channel height so every
    // column shares the same eta (smooth terrain-following grid lines)
    double meanH = 0.0;
    for (int i = 0; i <= nx; ++i)
        meanH += (yTop - yBottom[i]);
    meanH /= (nx + 1);
    double stretch = 2.0;
    if (y1t > 0) {
        auto fc = [&](double s) {
            return 0.5 * meanH * (1.0 + std::tanh(s * (2.0 / ny - 1.0)) / std::tanh(s));
        };
        if (y1t < meanH / ny) {
            double sLo = 0.1, sHi = 20.0;
            if (y1t < fc(sHi)) {
                stretch = sHi;
            } else {
                for (int it = 0; it < 100; ++it) {
                    double sm = 0.5 * (sLo + sHi);
                    if (fc(sm) > y1t)
                        sLo = sm;
                    else
                        sHi = sm;
                    if (sHi - sLo < 1e-10)
                        break;
                }
                stretch = 0.5 * (sLo + sHi);
            }
        }
    }
    std::vector<double> eta(ny + 1);
    for (int j = 0; j <= ny; ++j) {
        double e = static_cast<double>(j) / ny;
        eta[j] = 0.5 * (1.0 + std::tanh(stretch * (2.0 * e - 1.0)) / std::tanh(stretch));
    }

    // node coordinates (front plane; back plane duplicated at z = dz)
    auto ynode = [&](int i, int j) { return yBottom[i] + (yTop - yBottom[i]) * eta[j]; };
    Mesh m;
    const int ptsPerPlane = (nx + 1) * (ny + 1);
    m.nodes_.resize(2 * ptsPerPlane);
    auto nid = [&](int i, int j, int k) { return k * ptsPerPlane + j * (nx + 1) + i; };
    for (int j = 0; j <= ny; ++j)
        for (int i = 0; i <= nx; ++i) {
            m.nodes_[nid(i, j, 0)] = Vec3(xNodes[i], ynode(i, j), 0.0);
            m.nodes_[nid(i, j, 1)] = Vec3(xNodes[i], ynode(i, j), dz);
        }

    // cells: exact quad centroid and area (shoelace), volume = area * dz
    const int nC = nx * ny;
    m.cells_.resize(nC);
    auto cid = [&](int i, int j) { return j * nx + i; };
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            const double px[4] = {xNodes[i], xNodes[i + 1], xNodes[i + 1], xNodes[i]};
            const double py[4] = {ynode(i, j), ynode(i + 1, j), ynode(i + 1, j + 1),
                                  ynode(i, j + 1)};
            double A2 = 0.0, cxA = 0.0, cyA = 0.0;
            for (int kv = 0; kv < 4; ++kv) {
                int kn = (kv + 1) % 4;
                double cross = px[kv] * py[kn] - px[kn] * py[kv];
                A2 += cross;
                cxA += (px[kv] + px[kn]) * cross;
                cyA += (py[kv] + py[kn]) * cross;
            }
            double A = 0.5 * A2; // positive for CCW corners
            Cell &c = m.cells_[cid(i, j)];
            c.center = Vec3(cxA / (6.0 * A), cyA / (6.0 * A), 0.5 * dz);
            c.volume = std::abs(A) * dz;
        }

    // face counts: vertical internal (nx-1 columns + nx wrap? no: nx-1 interior
    // columns plus ONE wrap column) + horizontal internal + boundaries
    const int nVint = (nx - 1) * ny; // between columns i and i+1
    const int nWrap = ny;            // between column nx-1 and column 0
    const int nHint = nx * (ny - 1); // between rows j and j+1
    m.nInternal_ = nVint + nWrap + nHint;
    const int nBnd = 2 * nx; // bottom + top
    m.faces_.resize(m.nInternal_ + nBnd);

    int fi = 0;
    auto edgeFace = [&](Face &f, double x0, double y0, double x1, double y1, bool normalPlusX) {
        // 2D edge (x0,y0)-(x1,y1) extruded by dz; unit normal in-plane
        double ex = x1 - x0, ey = y1 - y0;
        double len = std::sqrt(ex * ex + ey * ey);
        f.center = Vec3(0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.5 * dz);
        f.area = len * dz;
        // the two in-plane normals are (ey,-ex)/len and (-ey,ex)/len
        if (normalPlusX)
            f.normal = Vec3(ey / len, -ex / len, 0.0);
        else
            f.normal = Vec3(-ey / len, ex / len, 0.0);
    };
    auto setDW = [&](Face &f, const Vec3 &cO, const Vec3 &cN) {
        f.d = cN - cO;
        f.delta = std::max(f.d.norm(), 1e-20);
        double dP = (f.center - cO).norm();
        double dN = (f.center - cN).norm();
        double sum = dP + dN;
        f.weight = (sum > 1e-30) ? dN / sum : 0.5;
    };

    // internal vertical faces (columns are straight: edge along +y at x_{i+1},
    // normal exactly +x, orthogonal to the column direction)
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx - 1; ++i) {
            Face &f = m.faces_[fi];
            f.owner = cid(i, j);
            f.neighbor = cid(i + 1, j);
            edgeFace(f, xNodes[i + 1], ynode(i + 1, j), xNodes[i + 1], ynode(i + 1, j + 1), true);
            setDW(f, m.cells_[f.owner].center, m.cells_[f.neighbor].center);
            m.cells_[f.owner].faces.push_back(fi);
            m.cells_[f.neighbor].faces.push_back(fi);
            ++fi;
        }
    // wrap faces: owner = last column, neighbor = first column; the geometric
    // face sits at x = Lx and every distance uses the neighbor's PERIODIC IMAGE
    for (int j = 0; j < ny; ++j) {
        Face &f = m.faces_[fi];
        f.owner = cid(nx - 1, j);
        f.neighbor = cid(0, j);
        edgeFace(f, xNodes[nx], ynode(nx, j), xNodes[nx], ynode(nx, j + 1), true);
        f.shift = Vec3(Lx, 0.0, 0.0);
        Vec3 cNimage = m.cells_[f.neighbor].center + f.shift;
        setDW(f, m.cells_[f.owner].center, cNimage);
        m.cells_[f.owner].faces.push_back(fi);
        m.cells_[f.neighbor].faces.push_back(fi);
        ++fi;
    }
    // internal horizontal faces (terrain-following: tilted near the slope)
    for (int j = 0; j < ny - 1; ++j)
        for (int i = 0; i < nx; ++i) {
            Face &f = m.faces_[fi];
            f.owner = cid(i, j);
            f.neighbor = cid(i, j + 1);
            edgeFace(f, xNodes[i], ynode(i, j + 1), xNodes[i + 1], ynode(i + 1, j + 1), false);
            setDW(f, m.cells_[f.owner].center, m.cells_[f.neighbor].center);
            m.cells_[f.owner].faces.push_back(fi);
            m.cells_[f.neighbor].faces.push_back(fi);
            ++fi;
        }

    // bottom wall (curved), outward normal points down-slope-outward
    {
        Patch p;
        p.name = bottomName;
        p.type = "wall";
        for (int i = 0; i < nx; ++i) {
            Face &f = m.faces_[fi];
            f.owner = cid(i, 0);
            f.neighbor = -1;
            f.patchID = static_cast<int>(m.patches_.size());
            edgeFace(f, xNodes[i], ynode(i, 0), xNodes[i + 1], ynode(i + 1, 0), true);
            // normalPlusX=true gives (ey,-ex): ey ~ slope, -ex < 0 so it points
            // DOWN out of the domain as required for the bottom boundary
            f.d = f.center - m.cells_[f.owner].center;
            f.delta = std::max(f.d.norm(), 1e-20);
            m.cells_[f.owner].faces.push_back(fi);
            p.faces.push_back(fi);
            ++fi;
        }
        m.patches_.push_back(std::move(p));
    }
    // top wall (flat), outward normal +y
    {
        Patch p;
        p.name = topName;
        p.type = "wall";
        for (int i = 0; i < nx; ++i) {
            Face &f = m.faces_[fi];
            f.owner = cid(i, ny - 1);
            f.neighbor = -1;
            f.patchID = static_cast<int>(m.patches_.size());
            edgeFace(f, xNodes[i], ynode(i, ny), xNodes[i + 1], ynode(i + 1, ny), false);
            f.d = f.center - m.cells_[f.owner].center;
            f.delta = std::max(f.d.norm(), 1e-20);
            m.cells_[f.owner].faces.push_back(fi);
            p.faces.push_back(fi);
            ++fi;
        }
        m.patches_.push_back(std::move(p));
    }

    // owner/neighbor lists for the linear solver
    m.ownerList_.resize(m.nInternal_);
    m.neighborList_.resize(m.nInternal_);
    for (int f = 0; f < m.nInternal_; ++f) {
        m.ownerList_[f] = m.faces_[f].owner;
        m.neighborList_[f] = m.faces_[f].neighbor;
    }

    std::cout << "CurvedChannelPeriodic2D: " << nx << "x" << ny << " (" << nC << " cells, "
              << m.faces_.size() << " faces, " << nWrap << " wrap faces)\n";
    return m;
}

// MSH2 retains the historical native-endian int32/IEEE-double layout and adds
// the periodic translation after each face's interpolation weight. MSH1 can
// still represent nonperiodic meshes; a lost periodic shift is not inferred.
static_assert(sizeof(int) == 4 && sizeof(double) == 8 && std::numeric_limits<double>::is_iec559,
              "binary mesh storage requires 32-bit int and IEEE 64-bit double");
static constexpr uint32_t MESH_MAGIC_V1 = 0x4D534831;
static constexpr uint32_t MESH_MAGIC_V2 = 0x4D534832;

void Mesh::saveBinary(const std::string &path) const {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs)
        throw std::runtime_error("Cannot open binary mesh for writing: " + path);
    const auto write = [&](const char *bytes, size_t count) {
        ofs.write(bytes, static_cast<std::streamsize>(count));
        if (!ofs)
            throw std::runtime_error("Cannot write binary mesh: " + path);
    };
    const auto writeInt = [&](int v) { write(reinterpret_cast<const char *>(&v), sizeof v); };
    const auto writeDbl = [&](double v) { write(reinterpret_cast<const char *>(&v), sizeof v); };
    const auto writeVec = [&](const Vec3 &v) {
        writeDbl(v.x);
        writeDbl(v.y);
        writeDbl(v.z);
    };
    const auto writeCount = [&](size_t count) {
        if (count > static_cast<size_t>(std::numeric_limits<int>::max()))
            throw std::runtime_error("Binary mesh count exceeds the supported index range");
        writeInt(static_cast<int>(count));
    };
    const auto writeStr = [&](const std::string &value) {
        writeCount(value.size());
        write(value.data(), value.size());
    };
    const uint32_t magic = MESH_MAGIC_V2;
    write(reinterpret_cast<const char *>(&magic), sizeof magic);
    writeCount(cells_.size());
    writeCount(faces_.size());
    writeCount(nodes_.size());
    writeCount(patches_.size());
    writeInt(nInternal_);
    for (const auto &node : nodes_)
        writeVec(node);
    for (const auto &face : faces_) {
        writeInt(face.owner);
        writeInt(face.neighbor);
        writeInt(face.patchID);
        writeVec(face.center);
        writeVec(face.normal);
        writeDbl(face.area);
        writeDbl(face.delta);
        writeVec(face.d);
        writeDbl(face.weight);
        writeVec(face.shift);
    }
    for (const auto &cell : cells_) {
        writeVec(cell.center);
        writeDbl(cell.volume);
        writeCount(cell.faces.size());
        for (int face : cell.faces)
            writeInt(face);
    }
    for (const auto &patch : patches_) {
        writeStr(patch.name);
        writeStr(patch.type);
        writeCount(patch.faces.size());
        for (int face : patch.faces)
            writeInt(face);
    }
    // Buffered output can fail only on flush/close, including a full filesystem.
    ofs.close();
    if (!ofs)
        throw std::runtime_error("Cannot finish writing binary mesh: " + path);
}

Mesh Mesh::loadBinary(const std::string &path) {
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs)
        throw std::runtime_error("Cannot open binary mesh: " + path);
    const auto length = ifs.tellg();
    if (length < 0)
        throw std::runtime_error("Cannot determine binary mesh length: " + path);
    auto remaining = static_cast<std::uint64_t>(length);
    ifs.seekg(0);
    const auto read = [&](char *bytes, size_t count) {
        if (count > remaining)
            throw std::runtime_error("Truncated binary mesh: " + path);
        ifs.read(bytes, static_cast<std::streamsize>(count));
        if (!ifs)
            throw std::runtime_error("Cannot read binary mesh: " + path);
        remaining -= count;
    };
    const auto readInt = [&]() {
        int v = 0;
        read(reinterpret_cast<char *>(&v), sizeof v);
        return v;
    };
    const auto readDbl = [&]() {
        double v = 0.;
        read(reinterpret_cast<char *>(&v), sizeof v);
        if (!std::isfinite(v))
            throw std::runtime_error("Nonfinite binary mesh geometry");
        return v;
    };
    const auto readVec = [&]() -> Vec3 { return {readDbl(), readDbl(), readDbl()}; };
    const auto readCount = [&](size_t bytesPerItem) {
        const int count = readInt();
        if (count < 0 || static_cast<std::uint64_t>(count) > remaining / bytesPerItem)
            throw std::runtime_error("Invalid binary mesh count");
        return count;
    };
    const auto readStr = [&]() {
        const int count = readCount(1);
        std::string value(static_cast<size_t>(count), '\0');
        read(value.data(), value.size());
        return value;
    };
    uint32_t magic = 0;
    read(reinterpret_cast<char *>(&magic), sizeof magic);
    if (magic != MESH_MAGIC_V1 && magic != MESH_MAGIC_V2)
        throw std::runtime_error("Invalid binary mesh magic or byte order");
    const bool hasShift = magic == MESH_MAGIC_V2;
    const int nc = readCount(36), nf = readCount(hasShift ? 132 : 108), nn = readCount(24),
              np = readCount(12);
    Mesh m;
    m.nInternal_ = readInt();
    if (m.nInternal_ < 0 || m.nInternal_ > nf)
        throw std::runtime_error("Invalid binary mesh internal-face count");
    // Bound allocations by bytes actually present, including the combined
    // minimum storage rather than checking each independent array alone.
    const auto minimum = static_cast<std::uint64_t>(nc) * 36 +
                         static_cast<std::uint64_t>(nf) * (hasShift ? 132 : 108) +
                         static_cast<std::uint64_t>(nn) * 24 + static_cast<std::uint64_t>(np) * 12;
    if (minimum > remaining)
        throw std::runtime_error("Truncated binary mesh arrays");
    m.nodes_.resize(static_cast<size_t>(nn));
    for (auto &node : m.nodes_)
        node = readVec();
    m.faces_.resize(static_cast<size_t>(nf));
    for (auto &face : m.faces_) {
        face.owner = readInt();
        face.neighbor = readInt();
        face.patchID = readInt();
        face.center = readVec();
        face.normal = readVec();
        face.area = readDbl();
        face.delta = readDbl();
        face.d = readVec();
        face.weight = readDbl();
        if (hasShift)
            face.shift = readVec();
    }
    m.cells_.resize(static_cast<size_t>(nc));
    for (auto &cell : m.cells_) {
        cell.center = readVec();
        cell.volume = readDbl();
        const int count = readCount(sizeof(int));
        cell.faces.resize(static_cast<size_t>(count));
        for (auto &face : cell.faces)
            face = readInt();
    }
    m.patches_.resize(static_cast<size_t>(np));
    for (auto &patch : m.patches_) {
        patch.name = readStr();
        patch.type = readStr();
        const int count = readCount(sizeof(int));
        patch.faces.resize(static_cast<size_t>(count));
        for (auto &face : patch.faces)
            face = readInt();
    }
    if (remaining != 0)
        throw std::runtime_error("Unexpected trailing bytes in binary mesh");

    // Admit connectivity before any indexed access. Internal faces precede
    // boundary faces, and every face must occur in its incident cells/patch.
    std::vector<int> ownerHits(static_cast<size_t>(nf)), neighborHits(static_cast<size_t>(nf)),
        patchHits(static_cast<size_t>(nf));
    for (int fi = 0; fi < nf; ++fi) {
        const auto &face = m.faces_[fi];
        const bool internal = fi < m.nInternal_;
        if (face.owner < 0 || face.owner >= nc ||
            (internal ? (face.neighbor < 0 || face.neighbor >= nc || face.patchID != -1)
                      : (face.neighbor != -1 || face.patchID < 0 || face.patchID >= np)))
            throw std::runtime_error("Invalid binary mesh face connectivity");
        if (face.area <= 0. || face.delta <= 0. || face.weight < 0. || face.weight > 1. ||
            std::abs(face.normal.norm() - 1.) > 1e-10 || (!internal && face.shift.norm2() != 0.))
            throw std::runtime_error("Invalid binary mesh face geometry");
        const Vec3 expected =
            internal ? m.cells_[face.neighbor].center + face.shift - m.cells_[face.owner].center
                     : face.center - m.cells_[face.owner].center;
        const double expectedSize = std::hypot(expected.x, expected.y, expected.z);
        const double storedSize = std::hypot(face.d.x, face.d.y, face.d.z);
        const double scale = std::max({expectedSize, storedSize, 1e-300});
        if (!std::isfinite(expectedSize) || !std::isfinite(storedSize) ||
            (expected - face.d).norm() > 1e-10 * scale ||
            (internal && expected.dot(face.normal) <= 0.))
            throw std::runtime_error(hasShift ? "Inconsistent binary mesh face displacement"
                                              : "Legacy binary mesh has inconsistent geometry; "
                                                "periodic shifts were not stored");
    }
    for (int ci = 0; ci < nc; ++ci) {
        const auto &cell = m.cells_[ci];
        if (cell.volume <= 0. || cell.faces.empty())
            throw std::runtime_error("Invalid binary mesh cell geometry");
        for (int fi : cell.faces) {
            if (fi < 0 || fi >= nf)
                throw std::runtime_error("Invalid binary mesh cell-face index");
            const auto &face = m.faces_[fi];
            if (face.owner == ci)
                ++ownerHits[fi];
            else if (face.neighbor == ci)
                ++neighborHits[fi];
            else
                throw std::runtime_error("Inconsistent binary mesh cell-face incidence");
        }
    }
    std::set<std::string> names;
    for (int pi = 0; pi < np; ++pi) {
        const auto &patch = m.patches_[pi];
        if (patch.name.empty() || !names.insert(patch.name).second)
            throw std::runtime_error("Invalid binary mesh patch name");
        for (int fi : patch.faces) {
            if (fi < m.nInternal_ || fi >= nf || m.faces_[fi].patchID != pi)
                throw std::runtime_error("Invalid binary mesh patch-face incidence");
            ++patchHits[fi];
        }
    }
    for (int fi = 0; fi < nf; ++fi) {
        const auto &face = m.faces_[fi];
        const bool internal = fi < m.nInternal_, self = internal && face.owner == face.neighbor;
        if (ownerHits[fi] != (self ? 2 : 1) || neighborHits[fi] != (internal && !self ? 1 : 0) ||
            patchHits[fi] != (internal ? 0 : 1))
            throw std::runtime_error("Incomplete or duplicate binary mesh face incidence");
    }
    m.ownerList_.resize(static_cast<size_t>(m.nInternal_));
    m.neighborList_.resize(static_cast<size_t>(m.nInternal_));
    for (int fi = 0; fi < m.nInternal_; ++fi) {
        m.ownerList_[fi] = m.faces_[fi].owner;
        m.neighborList_[fi] = m.faces_[fi].neighbor;
    }
    return m;
}

// add validation mesh
// patches: "inlet" (x=0), "outlet" (x=Lx), "top" (y=Ly, wall), "bottom" (y=0, wall)
// y-direction - tanh stretching toward both walls for BL resolution.
Mesh Mesh::makeChannel2D(int nx, int ny, double Lx, double Ly) {
    requireRectangle(nx, ny, Lx, Ly);
    Mesh m;
    double dz = 1.0; // unit depth for per-unit-depth quantities

    // y-coordinates with tanh stretching toward both walls
    double stretch = 2.0;
    std::vector<double> yc(ny + 1);
    for (int j = 0; j <= ny; ++j) {
        double eta = static_cast<double>(j) / ny;
        yc[j] = 0.5 * Ly * (1.0 + std::tanh(stretch * (2.0 * eta - 1.0)) / std::tanh(stretch));
    }
    // uniform x-coordinates
    std::vector<double> xc(nx + 1);
    for (int i = 0; i <= nx; ++i) {
        xc[i] = Lx * static_cast<double>(i) / nx;
    }
    // nodes: (nx+1)*(ny+1) * 2 planes (front z=0, back z=dz)
    int ptsPerPlane = (nx + 1) * (ny + 1);
    m.nodes_.resize(2 * ptsPerPlane);
    auto nid = [&](int i, int j, int k) { return k * ptsPerPlane + j * (nx + 1) + i; };
    for (int j = 0; j <= ny; ++j)
        for (int i = 0; i <= nx; ++i) {
            m.nodes_[nid(i, j, 0)] = Vec3(xc[i], yc[j], 0.0);
            m.nodes_[nid(i, j, 1)] = Vec3(xc[i], yc[j], dz);
        }

    // cells
    int nC = nx * ny;
    m.cells_.resize(nC);
    auto cid = [&](int i, int j) { return j * nx + i; };

    // faces
    int nInternalH = nx * (ny - 1); // horizontal internal (between j, j+1)
    int nInternalV = (nx - 1) * ny; // vertical internal   (between i, i+1)
    m.nInternal_ = nInternalH + nInternalV;
    int nBnd = 2 * nx + 2 * ny; // bottom + top + inlet + outlet
    m.faces_.resize(m.nInternal_ + nBnd);

    int fi = 0;

    // internal horizontal faces
    for (int j = 0; j < ny - 1; ++j) {
        for (int i = 0; i < nx; ++i) {
            Face &f = m.faces_[fi];
            f.owner = cid(i, j);
            f.neighbor = cid(i, j + 1);
            double x0 = xc[i], x1 = xc[i + 1];
            f.center = Vec3(0.5 * (x0 + x1), yc[j + 1], 0.5 * dz);
            f.area = (x1 - x0) * dz;
            f.normal = Vec3(0, 1, 0);
            m.cells_[f.owner].faces.push_back(fi);
            m.cells_[f.neighbor].faces.push_back(fi);
            ++fi;
        }
    }

    // internal vertical faces
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx - 1; ++i) {
            Face &f = m.faces_[fi];
            f.owner = cid(i, j);
            f.neighbor = cid(i + 1, j);
            double y0 = yc[j], y1 = yc[j + 1];
            f.center = Vec3(xc[i + 1], 0.5 * (y0 + y1), 0.5 * dz);
            f.area = (y1 - y0) * dz;
            f.normal = Vec3(1, 0, 0);
            m.cells_[f.owner].faces.push_back(fi);
            m.cells_[f.neighbor].faces.push_back(fi);
            ++fi;
        }
    }

    // boundary faces
    // bottom wall (y = 0)
    Patch bottom;
    bottom.name = "bottom";
    bottom.type = "wall";
    for (int i = 0; i < nx; ++i) {
        Face &f = m.faces_[fi];
        f.owner = cid(i, 0);
        f.neighbor = -1;
        f.patchID = static_cast<int>(m.patches_.size());
        f.center = Vec3(0.5 * (xc[i] + xc[i + 1]), 0.0, 0.5 * dz);
        f.area = (xc[i + 1] - xc[i]) * dz;
        f.normal = Vec3(0, -1, 0);
        m.cells_[f.owner].faces.push_back(fi);
        bottom.faces.push_back(fi);
        ++fi;
    }
    m.patches_.push_back(std::move(bottom));

    // top wall (y = Ly)
    Patch top;
    top.name = "top";
    top.type = "wall";
    for (int i = 0; i < nx; ++i) {
        Face &f = m.faces_[fi];
        f.owner = cid(i, ny - 1);
        f.neighbor = -1;
        f.patchID = static_cast<int>(m.patches_.size());
        f.center = Vec3(0.5 * (xc[i] + xc[i + 1]), Ly, 0.5 * dz);
        f.area = (xc[i + 1] - xc[i]) * dz;
        f.normal = Vec3(0, 1, 0);
        m.cells_[f.owner].faces.push_back(fi);
        top.faces.push_back(fi);
        ++fi;
    }
    m.patches_.push_back(std::move(top));

    // inlet (x = 0)
    Patch inlet;
    inlet.name = "inlet";
    inlet.type = "inlet";
    for (int j = 0; j < ny; ++j) {
        Face &f = m.faces_[fi];
        f.owner = cid(0, j);
        f.neighbor = -1;
        f.patchID = static_cast<int>(m.patches_.size());
        f.center = Vec3(0.0, 0.5 * (yc[j] + yc[j + 1]), 0.5 * dz);
        f.area = (yc[j + 1] - yc[j]) * dz;
        f.normal = Vec3(-1, 0, 0);
        m.cells_[f.owner].faces.push_back(fi);
        inlet.faces.push_back(fi);
        ++fi;
    }
    m.patches_.push_back(std::move(inlet));

    // outlet (x = Lx)
    Patch outlet;
    outlet.name = "outlet";
    outlet.type = "outlet";
    for (int j = 0; j < ny; ++j) {
        Face &f = m.faces_[fi];
        f.owner = cid(nx - 1, j);
        f.neighbor = -1;
        f.patchID = static_cast<int>(m.patches_.size());
        f.center = Vec3(Lx, 0.5 * (yc[j] + yc[j + 1]), 0.5 * dz);
        f.area = (yc[j + 1] - yc[j]) * dz;
        f.normal = Vec3(1, 0, 0);
        m.cells_[f.owner].faces.push_back(fi);
        outlet.faces.push_back(fi);
        ++fi;
    }
    m.patches_.push_back(std::move(outlet));

    // owner/neighbor lists
    m.ownerList_.resize(m.nInternal_);
    m.neighborList_.resize(m.nInternal_);
    for (int f = 0; f < m.nInternal_; ++f) {
        m.ownerList_[f] = m.faces_[f].owner;
        m.neighborList_[f] = m.faces_[f].neighbor;
    }

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            Cell &c = m.cells_[cid(i, j)];
            c.center = Vec3(0.5 * (xc[i] + xc[i + 1]), 0.5 * (yc[j] + yc[j + 1]), 0.5 * dz);
            c.volume = (xc[i + 1] - xc[i]) * (yc[j + 1] - yc[j]) * dz;
        }
    }
    m.computeFaceGeometry();
    m.ComputeInterpolationWeights();

    std::cout << "Channel2D mesh: " << nx << "x" << ny << " (" << m.nCells() << " cells, "
              << m.nFaces() << " faces)\n";
    return m;
}

// One-sided wall-clustered plate mesh: every refinement cell goes to the
// BOTTOM wall (the top is a far-field boundary, not a viscous wall), so the
// near-wall growth ratio at matched ny is roughly half the two-sided
// channel's. The turbulent omega field's 1/y^2 sublayer tail needs that
// resolution: on the two-sided channel clustering the transported omega
// collapses through the buffer layer and the SST settles into an over-mixed
// state (measured at the shock-interaction baseline bring-up: skin friction
// 35 to 60 percent high, converging as the near-wall growth drops).
Mesh Mesh::makePlate2D(int nx, int ny, double Lx, double Ly, double Re, double yPlusTarget) {
    requireRectangle(nx, ny, Lx, Ly);
    requirePositive(Re, "plate Reynolds number");
    requirePositive(yPlusTarget, "mesh y-plus target");
    // friction estimate as makeChannel2D: Cf ~ 0.058 Re^-0.2, U_ref = 1
    double Cf = 0.058 * std::pow(Re, -0.2);
    double u_tau = std::sqrt(Cf / 2.0);
    double nu = (Ly / 2.0) / Re;
    double y1_target = yPlusTarget * nu / u_tau;
    double y1_uniform = Ly / ny;

    // GEOMETRIC growth from the first cell: the tanh mapping concentrates
    // its refinement below the first few y+ and then roughly doubles each
    // cell, leaving the buffer layer (y+ 5 to 30) with a handful of cells at
    // any stretch; a geometric ratio spends the same ny at a near-constant
    // growth (about 1.03 to 1.10 for these targets), which is what the
    // transported near-wall omega needs. Solve r from
    //   y1 (r^ny - 1) / (r - 1) = Ly
    double ratio = 1.0;
    if (y1_target < y1_uniform) {
        auto totalHeight = [&](double r) {
            return y1_target * (std::pow(r, ny) - 1.0) / (r - 1.0);
        };
        double rLo = 1.0 + 1e-9, rHi = 1.5;
        if (totalHeight(rHi) < Ly) {
            std::cout << "  WARNING: Ly unreachable at growth 1.5 with ny=" << ny
                      << "; increase ny\n";
            ratio = rHi;
        } else {
            for (int iter = 0; iter < 200; ++iter) {
                double rMid = 0.5 * (rLo + rHi);
                if (totalHeight(rMid) < Ly)
                    rLo = rMid;
                else
                    rHi = rMid;
                if (rHi - rLo < 1e-12)
                    break;
            }
            ratio = 0.5 * (rLo + rHi);
        }
    }
    std::cout << "  Plate mesh: Re=" << Re << " y+_target=" << yPlusTarget
              << " y1_target=" << y1_target << " growth=" << ratio << "\n";

    Mesh m;
    double dz = 1.0;

    std::vector<double> yc(ny + 1);
    if (ratio > 1.0 + 1e-9) {
        yc[0] = 0.0;
        double h = y1_target;
        for (int j = 1; j <= ny; ++j) {
            yc[j] = yc[j - 1] + h;
            h *= ratio;
        }
        // scale out the bisection remainder so the top lands exactly on Ly
        double scale = Ly / yc[ny];
        for (int j = 1; j <= ny; ++j)
            yc[j] *= scale;
    } else {
        for (int j = 0; j <= ny; ++j)
            yc[j] = Ly * static_cast<double>(j) / ny;
    }

    std::vector<double> xc(nx + 1);
    for (int i = 0; i <= nx; ++i)
        xc[i] = Lx * static_cast<double>(i) / nx;

    int ptsPerPlane = (nx + 1) * (ny + 1);
    m.nodes_.resize(2 * ptsPerPlane);
    auto nid = [&](int i, int j, int k) { return k * ptsPerPlane + j * (nx + 1) + i; };
    for (int j = 0; j <= ny; ++j)
        for (int i = 0; i <= nx; ++i) {
            m.nodes_[nid(i, j, 0)] = Vec3(xc[i], yc[j], 0.0);
            m.nodes_[nid(i, j, 1)] = Vec3(xc[i], yc[j], dz);
        }

    int nC = nx * ny;
    m.cells_.resize(nC);
    auto cid = [&](int i, int j) { return j * nx + i; };

    int nInternalH = nx * (ny - 1);
    int nInternalV = (nx - 1) * ny;
    m.nInternal_ = nInternalH + nInternalV;
    int nBnd = 2 * nx + 2 * ny;
    m.faces_.resize(m.nInternal_ + nBnd);

    int fi = 0;
    for (int j = 0; j < ny - 1; ++j)
        for (int i = 0; i < nx; ++i) {
            Face &f = m.faces_[fi];
            f.owner = cid(i, j);
            f.neighbor = cid(i, j + 1);
            f.center = Vec3(0.5 * (xc[i] + xc[i + 1]), yc[j + 1], 0.5 * dz);
            f.area = (xc[i + 1] - xc[i]) * dz;
            f.normal = Vec3(0, 1, 0);
            m.cells_[f.owner].faces.push_back(fi);
            m.cells_[f.neighbor].faces.push_back(fi);
            ++fi;
        }
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx - 1; ++i) {
            Face &f = m.faces_[fi];
            f.owner = cid(i, j);
            f.neighbor = cid(i + 1, j);
            f.center = Vec3(xc[i + 1], 0.5 * (yc[j] + yc[j + 1]), 0.5 * dz);
            f.area = (yc[j + 1] - yc[j]) * dz;
            f.normal = Vec3(1, 0, 0);
            m.cells_[f.owner].faces.push_back(fi);
            m.cells_[f.neighbor].faces.push_back(fi);
            ++fi;
        }

    Patch bottom;
    bottom.name = "bottom";
    bottom.type = "wall";
    for (int i = 0; i < nx; ++i) {
        Face &f = m.faces_[fi];
        f.owner = cid(i, 0);
        f.neighbor = -1;
        f.patchID = static_cast<int>(m.patches_.size());
        f.center = Vec3(0.5 * (xc[i] + xc[i + 1]), 0.0, 0.5 * dz);
        f.area = (xc[i + 1] - xc[i]) * dz;
        f.normal = Vec3(0, -1, 0);
        m.cells_[f.owner].faces.push_back(fi);
        bottom.faces.push_back(fi);
        ++fi;
    }
    m.patches_.push_back(std::move(bottom));

    // far-field top: NOT a wall (patch type reflects the physics, so any
    // consumer of the mesh's own wall distance sees only the plate)
    Patch top;
    top.name = "top";
    top.type = "patch";
    for (int i = 0; i < nx; ++i) {
        Face &f = m.faces_[fi];
        f.owner = cid(i, ny - 1);
        f.neighbor = -1;
        f.patchID = static_cast<int>(m.patches_.size());
        f.center = Vec3(0.5 * (xc[i] + xc[i + 1]), Ly, 0.5 * dz);
        f.area = (xc[i + 1] - xc[i]) * dz;
        f.normal = Vec3(0, 1, 0);
        m.cells_[f.owner].faces.push_back(fi);
        top.faces.push_back(fi);
        ++fi;
    }
    m.patches_.push_back(std::move(top));

    Patch inlet;
    inlet.name = "inlet";
    inlet.type = "inlet";
    for (int j = 0; j < ny; ++j) {
        Face &f = m.faces_[fi];
        f.owner = cid(0, j);
        f.neighbor = -1;
        f.patchID = static_cast<int>(m.patches_.size());
        f.center = Vec3(0.0, 0.5 * (yc[j] + yc[j + 1]), 0.5 * dz);
        f.area = (yc[j + 1] - yc[j]) * dz;
        f.normal = Vec3(-1, 0, 0);
        m.cells_[f.owner].faces.push_back(fi);
        inlet.faces.push_back(fi);
        ++fi;
    }
    m.patches_.push_back(std::move(inlet));

    Patch outlet;
    outlet.name = "outlet";
    outlet.type = "outlet";
    for (int j = 0; j < ny; ++j) {
        Face &f = m.faces_[fi];
        f.owner = cid(nx - 1, j);
        f.neighbor = -1;
        f.patchID = static_cast<int>(m.patches_.size());
        f.center = Vec3(Lx, 0.5 * (yc[j] + yc[j + 1]), 0.5 * dz);
        f.area = (yc[j + 1] - yc[j]) * dz;
        f.normal = Vec3(1, 0, 0);
        m.cells_[f.owner].faces.push_back(fi);
        outlet.faces.push_back(fi);
        ++fi;
    }
    m.patches_.push_back(std::move(outlet));

    m.ownerList_.resize(m.nInternal_);
    m.neighborList_.resize(m.nInternal_);
    for (int f = 0; f < m.nInternal_; ++f) {
        m.ownerList_[f] = m.faces_[f].owner;
        m.neighborList_[f] = m.faces_[f].neighbor;
    }
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            Cell &c = m.cells_[cid(i, j)];
            c.center = Vec3(0.5 * (xc[i] + xc[i + 1]), 0.5 * (yc[j] + yc[j + 1]), 0.5 * dz);
            c.volume = (xc[i + 1] - xc[i]) * (yc[j + 1] - yc[j]) * dz;
        }
    m.computeFaceGeometry();
    m.ComputeInterpolationWeights();

    std::cout << "Plate2D mesh: " << nx << "x" << ny << " (" << m.nCells() << " cells, "
              << m.nFaces() << " faces)\n";
    return m;
}

// Re-adaptive wall clustering overload
// Computes tanh stretch parameter so the first cell targets yPlusTarget.
Mesh Mesh::makeChannel2D(int nx, int ny, double Lx, double Ly, double Re, double yPlusTarget) {
    requireRectangle(nx, ny, Lx, Ly);
    requirePositive(Re, "channel Reynolds number");
    requirePositive(yPlusTarget, "mesh y-plus target");
    // Estimate friction velocity from flat-plate correlation
    // Cf ~ 0.058 * Re^(-0.2)
    // u_tau = sqrt(Cf/2) * U_ref   (U_ref = 1)
    double Cf = 0.058 * std::pow(Re, -0.2);
    double u_tau = std::sqrt(Cf / 2.0);

    // nu = U_ref * L_ref / Re, with U_ref = 1, L_ref = Ly/2
    double nu = (Ly / 2.0) / Re;

    // Target first cell height from y+ definition: y1 = y+ * nu / u_tau
    double y1_target = yPlusTarget * nu / u_tau;

    // Uniform spacing for comparison
    double y1_uniform = Ly / ny;

    double stretch = 0.0; // default: uniform mesh

    if (y1_target < y1_uniform) {
        // Bisect for stretch: larger stretch => smaller first cell
        // First cell height from tanh formula:
        //   y1(s) = 0.5 * Ly * (1 + tanh(s * (2.0/ny - 1)) / tanh(s))
        double sLo = 0.1, sHi = 20.0;

        // Check if target is reachable at max stretch
        auto firstCellHeight = [&](double s) {
            return 0.5 * Ly * (1.0 + std::tanh(s * (2.0 / ny - 1.0)) / std::tanh(s));
        };

        double y1_max_stretch = firstCellHeight(sHi);
        if (y1_target < y1_max_stretch) {
            std::cout << "  WARNING: y1_target=" << y1_target << " unreachable with ny=" << ny
                      << "; using max stretch=" << sHi << " (y1=" << y1_max_stretch << ")\n";
            stretch = sHi;
        } else {
            // Bisection: find stretch where firstCellHeight(s) == y1_target
            for (int iter = 0; iter < 100; ++iter) {
                double sMid = 0.5 * (sLo + sHi);
                double y1_mid = firstCellHeight(sMid);
                if (y1_mid > y1_target)
                    sLo = sMid; // need more stretching
                else
                    sHi = sMid;
                if (sHi - sLo < 1e-10)
                    break;
            }
            stretch = 0.5 * (sLo + sHi);
        }
    }
    // else: y1_target >= y1_uniform => uniform mesh is fine, stretch = 0

    std::cout << "  Adaptive mesh: Re=" << Re << " y+_target=" << yPlusTarget
              << " y1_target=" << y1_target << " stretch=" << stretch << "\n";

    // Build the mesh using the same logic as the original, but with computed stretch
    Mesh m;
    double dz = 1.0;

    // y-coordinates with tanh stretching
    std::vector<double> yc(ny + 1);
    if (stretch > 1e-12) {
        for (int j = 0; j <= ny; ++j) {
            double eta = static_cast<double>(j) / ny;
            yc[j] = 0.5 * Ly * (1.0 + std::tanh(stretch * (2.0 * eta - 1.0)) / std::tanh(stretch));
        }
    } else {
        for (int j = 0; j <= ny; ++j) {
            yc[j] = Ly * static_cast<double>(j) / ny;
        }
    }

    // uniform x-coordinates
    std::vector<double> xc(nx + 1);
    for (int i = 0; i <= nx; ++i) {
        xc[i] = Lx * static_cast<double>(i) / nx;
    }

    // nodes
    int ptsPerPlane = (nx + 1) * (ny + 1);
    m.nodes_.resize(2 * ptsPerPlane);
    auto nid = [&](int i, int j, int k) { return k * ptsPerPlane + j * (nx + 1) + i; };
    for (int j = 0; j <= ny; ++j)
        for (int i = 0; i <= nx; ++i) {
            m.nodes_[nid(i, j, 0)] = Vec3(xc[i], yc[j], 0.0);
            m.nodes_[nid(i, j, 1)] = Vec3(xc[i], yc[j], dz);
        }

    // cells
    int nC = nx * ny;
    m.cells_.resize(nC);
    auto cid = [&](int i, int j) { return j * nx + i; };

    // faces
    int nInternalH = nx * (ny - 1);
    int nInternalV = (nx - 1) * ny;
    m.nInternal_ = nInternalH + nInternalV;
    int nBnd = 2 * nx + 2 * ny;
    m.faces_.resize(m.nInternal_ + nBnd);

    int fi = 0;

    // internal horizontal faces
    for (int j = 0; j < ny - 1; ++j) {
        for (int i = 0; i < nx; ++i) {
            Face &f = m.faces_[fi];
            f.owner = cid(i, j);
            f.neighbor = cid(i, j + 1);
            double x0 = xc[i], x1 = xc[i + 1];
            f.center = Vec3(0.5 * (x0 + x1), yc[j + 1], 0.5 * dz);
            f.area = (x1 - x0) * dz;
            f.normal = Vec3(0, 1, 0);
            m.cells_[f.owner].faces.push_back(fi);
            m.cells_[f.neighbor].faces.push_back(fi);
            ++fi;
        }
    }

    // internal vertical faces
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx - 1; ++i) {
            Face &f = m.faces_[fi];
            f.owner = cid(i, j);
            f.neighbor = cid(i + 1, j);
            double y0 = yc[j], y1 = yc[j + 1];
            f.center = Vec3(xc[i + 1], 0.5 * (y0 + y1), 0.5 * dz);
            f.area = (y1 - y0) * dz;
            f.normal = Vec3(1, 0, 0);
            m.cells_[f.owner].faces.push_back(fi);
            m.cells_[f.neighbor].faces.push_back(fi);
            ++fi;
        }
    }

    // boundary faces — bottom wall
    Patch bottom;
    bottom.name = "bottom";
    bottom.type = "wall";
    for (int i = 0; i < nx; ++i) {
        Face &f = m.faces_[fi];
        f.owner = cid(i, 0);
        f.neighbor = -1;
        f.patchID = static_cast<int>(m.patches_.size());
        f.center = Vec3(0.5 * (xc[i] + xc[i + 1]), 0.0, 0.5 * dz);
        f.area = (xc[i + 1] - xc[i]) * dz;
        f.normal = Vec3(0, -1, 0);
        m.cells_[f.owner].faces.push_back(fi);
        bottom.faces.push_back(fi);
        ++fi;
    }
    m.patches_.push_back(std::move(bottom));

    // top wall
    Patch top;
    top.name = "top";
    top.type = "wall";
    for (int i = 0; i < nx; ++i) {
        Face &f = m.faces_[fi];
        f.owner = cid(i, ny - 1);
        f.neighbor = -1;
        f.patchID = static_cast<int>(m.patches_.size());
        f.center = Vec3(0.5 * (xc[i] + xc[i + 1]), Ly, 0.5 * dz);
        f.area = (xc[i + 1] - xc[i]) * dz;
        f.normal = Vec3(0, 1, 0);
        m.cells_[f.owner].faces.push_back(fi);
        top.faces.push_back(fi);
        ++fi;
    }
    m.patches_.push_back(std::move(top));

    // inlet
    Patch inlet;
    inlet.name = "inlet";
    inlet.type = "inlet";
    for (int j = 0; j < ny; ++j) {
        Face &f = m.faces_[fi];
        f.owner = cid(0, j);
        f.neighbor = -1;
        f.patchID = static_cast<int>(m.patches_.size());
        f.center = Vec3(0.0, 0.5 * (yc[j] + yc[j + 1]), 0.5 * dz);
        f.area = (yc[j + 1] - yc[j]) * dz;
        f.normal = Vec3(-1, 0, 0);
        m.cells_[f.owner].faces.push_back(fi);
        inlet.faces.push_back(fi);
        ++fi;
    }
    m.patches_.push_back(std::move(inlet));

    // outlet
    Patch outlet;
    outlet.name = "outlet";
    outlet.type = "outlet";
    for (int j = 0; j < ny; ++j) {
        Face &f = m.faces_[fi];
        f.owner = cid(nx - 1, j);
        f.neighbor = -1;
        f.patchID = static_cast<int>(m.patches_.size());
        f.center = Vec3(Lx, 0.5 * (yc[j] + yc[j + 1]), 0.5 * dz);
        f.area = (yc[j + 1] - yc[j]) * dz;
        f.normal = Vec3(1, 0, 0);
        m.cells_[f.owner].faces.push_back(fi);
        outlet.faces.push_back(fi);
        ++fi;
    }
    m.patches_.push_back(std::move(outlet));

    // owner/neighbor lists
    m.ownerList_.resize(m.nInternal_);
    m.neighborList_.resize(m.nInternal_);
    for (int f = 0; f < m.nInternal_; ++f) {
        m.ownerList_[f] = m.faces_[f].owner;
        m.neighborList_[f] = m.faces_[f].neighbor;
    }

    // cell centers and volumes
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            Cell &c = m.cells_[cid(i, j)];
            c.center = Vec3(0.5 * (xc[i] + xc[i + 1]), 0.5 * (yc[j] + yc[j + 1]), 0.5 * dz);
            c.volume = (xc[i + 1] - xc[i]) * (yc[j + 1] - yc[j]) * dz;
        }
    }
    m.computeFaceGeometry();
    m.ComputeInterpolationWeights();

    std::cout << "Channel2D mesh: " << nx << "x" << ny << " (" << m.nCells() << " cells, "
              << m.nFaces() << " faces)\n";
    return m;
}

// Backward-facing step (non-adaptive): default tanh stretch = 2.0
Mesh Mesh::makeBackwardFacingStep2D(int nx_up, int nx_down, int ny_up, int ny_down, double Lu,
                                    double Ld, double h_s, double H) {
    return makeBackwardFacingStep2D(nx_up, nx_down, ny_up, ny_down, Lu, Ld, h_s, H, 0.0, 1.0);
}

// Backward-facing step with Re-adaptive y+ clustering.
// Upper block: y in [h_s, H] with symmetric tanh (near both step lip and top wall).
// Lower block: y in [0, h_s] with one-sided tanh toward y=0 (bottom wall).
// Six named patches: top_wall, bottom_wall_up, step_face, bottom_wall_down, inlet, outlet.
Mesh Mesh::makeBackwardFacingStep2D(int nx_up, int nx_down, int ny_up, int ny_down, double Lu,
                                    double Ld, double h_s, double H, double Re,
                                    double yPlusTarget) {
    requireRectangle(nx_up, ny_up, Lu, H - h_s);
    requireRectangle(nx_down, ny_down, Ld, h_s);
    const auto nxWide = static_cast<std::int64_t>(nx_up) + nx_down;
    const auto nyWide = static_cast<std::int64_t>(ny_up) + ny_down;
    if (nxWide > std::numeric_limits<int>::max() || nyWide > std::numeric_limits<int>::max())
        throw std::invalid_argument("step mesh counts exceed the supported indexing range");
    requireMeshSize(static_cast<int>(nxWide), static_cast<int>(nyWide));
    requirePositive(Lu + Ld, "step mesh total length");
    requireOptionalReynolds(Re); // Zero selects the non-Reynolds factory's stretching.
    requirePositive(yPlusTarget, "mesh y-plus target");
    const int nx_tot = nx_up + nx_down;
    const double dz = 1.0;
    const double H_up = H - h_s;

    // --- x-coordinates ---
    std::vector<double> xc(nx_tot + 1);
    for (int i = 0; i <= nx_up; ++i)
        xc[i] = -Lu + Lu * i / nx_up;
    for (int i = 1; i <= nx_down; ++i)
        xc[nx_up + i] = Ld * i / nx_down;

    // --- y-coordinates: upper block [h_s, H], symmetric tanh toward both walls ---
    double stretch_up = 2.0;
    if (Re > 0) {
        double Cf = 0.058 * std::pow(Re, -0.2);
        double u_tau = std::sqrt(Cf / 2.0);
        double nu = H_up / Re;
        double y1t = yPlusTarget * nu / u_tau;
        double y1u = H_up / ny_up;
        if (y1t < y1u) {
            auto fc = [&](double s) {
                return 0.5 * H_up * (1.0 + std::tanh(s * (2.0 / ny_up - 1.0)) / std::tanh(s));
            };
            double sLo = 0.1, sHi = 20.0;
            if (y1t < fc(sHi)) {
                stretch_up = sHi;
            } else {
                for (int it = 0; it < 100; ++it) {
                    double sm = 0.5 * (sLo + sHi);
                    if (fc(sm) > y1t)
                        sLo = sm;
                    else
                        sHi = sm;
                    if (sHi - sLo < 1e-10)
                        break;
                }
                stretch_up = 0.5 * (sLo + sHi);
            }
        }
    }
    std::vector<double> yc_up(ny_up + 1);
    for (int j = 0; j <= ny_up; ++j) {
        double eta = static_cast<double>(j) / ny_up;
        yc_up[j] =
            h_s +
            0.5 * H_up * (1.0 + std::tanh(stretch_up * (2.0 * eta - 1.0)) / std::tanh(stretch_up));
    }

    // --- y-coordinates: lower block [0, h_s], one-sided tanh toward y=0 ---
    // Uses the same symmetric-tanh formula as the channel, applied to the lower half:
    //   yc_low[j] = 0.5*h_s * (1 + tanh(s*(2*j/ny_down - 1)) / tanh(s))
    // which clusters cells near both j=0 and j=ny_down; we accept clustering at
    // the h_s interface too (it is the shear-layer side and benefits from resolution).
    double stretch_low = 2.0;
    if (Re > 0) {
        double Cf = 0.058 * std::pow(Re, -0.2);
        double u_tau = std::sqrt(Cf / 2.0);
        double nu = h_s / Re;
        double y1t = yPlusTarget * nu / u_tau;
        double y1u = h_s / ny_down;
        if (y1t < y1u) {
            // fc(s) = first cell height (distance from y=0 to yc_low[1])
            auto fc = [&](double s) {
                return 0.5 * h_s * (1.0 + std::tanh(s * (2.0 / ny_down - 1.0)) / std::tanh(s));
            };
            double sLo = 0.1, sHi = 20.0;
            if (y1t < fc(sHi)) {
                stretch_low = sHi;
            } else {
                for (int it = 0; it < 100; ++it) {
                    double sm = 0.5 * (sLo + sHi);
                    if (fc(sm) > y1t)
                        sLo = sm;
                    else
                        sHi = sm;
                    if (sHi - sLo < 1e-10)
                        break;
                }
                stretch_low = 0.5 * (sLo + sHi);
            }
        }
    }
    std::vector<double> yc_low(ny_down + 1);
    for (int j = 0; j <= ny_down; ++j) {
        double eta = static_cast<double>(j) / ny_down;
        yc_low[j] =
            0.5 * h_s * (1.0 + std::tanh(stretch_low * (2.0 * eta - 1.0)) / std::tanh(stretch_low));
    }

    // Cell ID helpers
    auto cid_up = [&](int i, int j) { return j * nx_tot + i; };
    auto cid_low = [&](int i, int j) { return nx_tot * ny_up + j * nx_down + i; };

    // Face counts
    int nHup = nx_tot * (ny_up - 1);
    int nHlow = nx_down * (ny_down - 1);
    int nCouple = nx_down;
    int nVup = (nx_tot - 1) * ny_up;
    int nVlow = (nx_down - 1) * ny_down;
    int nInternal = nHup + nHlow + nCouple + nVup + nVlow;
    int nBnd = nx_tot + nx_up + ny_down + nx_down + ny_up + (ny_up + ny_down);

    Mesh m;
    int nC = nx_tot * ny_up + nx_down * ny_down;
    m.cells_.resize(nC);
    m.faces_.resize(nInternal + nBnd);
    m.nInternal_ = nInternal;

    int fi = 0;

    // Internal: horizontal upper block
    for (int j = 0; j < ny_up - 1; ++j) {
        for (int i = 0; i < nx_tot; ++i) {
            Face &f = m.faces_[fi];
            f.owner = cid_up(i, j);
            f.neighbor = cid_up(i, j + 1);
            f.center = Vec3(0.5 * (xc[i] + xc[i + 1]), yc_up[j + 1], 0.5 * dz);
            f.area = (xc[i + 1] - xc[i]) * dz;
            f.normal = Vec3(0, 1, 0);
            m.cells_[f.owner].faces.push_back(fi);
            m.cells_[f.neighbor].faces.push_back(fi);
            ++fi;
        }
    }
    // Internal: horizontal lower block
    for (int j = 0; j < ny_down - 1; ++j) {
        for (int i = 0; i < nx_down; ++i) {
            Face &f = m.faces_[fi];
            f.owner = cid_low(i, j);
            f.neighbor = cid_low(i, j + 1);
            f.center = Vec3(0.5 * (xc[nx_up + i] + xc[nx_up + i + 1]), yc_low[j + 1], 0.5 * dz);
            f.area = (xc[nx_up + i + 1] - xc[nx_up + i]) * dz;
            f.normal = Vec3(0, 1, 0);
            m.cells_[f.owner].faces.push_back(fi);
            m.cells_[f.neighbor].faces.push_back(fi);
            ++fi;
        }
    }
    // Internal: coupling faces at y=h_s (owner=upper j=0, normal downward toward lower)
    for (int i = 0; i < nx_down; ++i) {
        Face &f = m.faces_[fi];
        f.owner = cid_up(nx_up + i, 0);
        f.neighbor = cid_low(i, ny_down - 1);
        f.center = Vec3(0.5 * (xc[nx_up + i] + xc[nx_up + i + 1]), h_s, 0.5 * dz);
        f.area = (xc[nx_up + i + 1] - xc[nx_up + i]) * dz;
        f.normal = Vec3(0, -1, 0);
        m.cells_[f.owner].faces.push_back(fi);
        m.cells_[f.neighbor].faces.push_back(fi);
        ++fi;
    }
    // Internal: vertical upper block
    for (int j = 0; j < ny_up; ++j) {
        for (int i = 0; i < nx_tot - 1; ++i) {
            Face &f = m.faces_[fi];
            f.owner = cid_up(i, j);
            f.neighbor = cid_up(i + 1, j);
            f.center = Vec3(xc[i + 1], 0.5 * (yc_up[j] + yc_up[j + 1]), 0.5 * dz);
            f.area = (yc_up[j + 1] - yc_up[j]) * dz;
            f.normal = Vec3(1, 0, 0);
            m.cells_[f.owner].faces.push_back(fi);
            m.cells_[f.neighbor].faces.push_back(fi);
            ++fi;
        }
    }
    // Internal: vertical lower block
    for (int j = 0; j < ny_down; ++j) {
        for (int i = 0; i < nx_down - 1; ++i) {
            Face &f = m.faces_[fi];
            f.owner = cid_low(i, j);
            f.neighbor = cid_low(i + 1, j);
            f.center = Vec3(xc[nx_up + i + 1], 0.5 * (yc_low[j] + yc_low[j + 1]), 0.5 * dz);
            f.area = (yc_low[j + 1] - yc_low[j]) * dz;
            f.normal = Vec3(1, 0, 0);
            m.cells_[f.owner].faces.push_back(fi);
            m.cells_[f.neighbor].faces.push_back(fi);
            ++fi;
        }
    }

    // Boundary: top_wall
    {
        Patch p;
        p.name = "top_wall";
        p.type = "wall";
        for (int i = 0; i < nx_tot; ++i) {
            Face &f = m.faces_[fi];
            f.owner = cid_up(i, ny_up - 1);
            f.neighbor = -1;
            f.patchID = static_cast<int>(m.patches_.size());
            f.center = Vec3(0.5 * (xc[i] + xc[i + 1]), H, 0.5 * dz);
            f.area = (xc[i + 1] - xc[i]) * dz;
            f.normal = Vec3(0, 1, 0);
            m.cells_[f.owner].faces.push_back(fi);
            p.faces.push_back(fi);
            ++fi;
        }
        m.patches_.push_back(std::move(p));
    }
    // Boundary: bottom_wall_up (upstream floor at y=h_s)
    {
        Patch p;
        p.name = "bottom_wall_up";
        p.type = "wall";
        for (int i = 0; i < nx_up; ++i) {
            Face &f = m.faces_[fi];
            f.owner = cid_up(i, 0);
            f.neighbor = -1;
            f.patchID = static_cast<int>(m.patches_.size());
            f.center = Vec3(0.5 * (xc[i] + xc[i + 1]), h_s, 0.5 * dz);
            f.area = (xc[i + 1] - xc[i]) * dz;
            f.normal = Vec3(0, -1, 0);
            m.cells_[f.owner].faces.push_back(fi);
            p.faces.push_back(fi);
            ++fi;
        }
        m.patches_.push_back(std::move(p));
    }
    // Boundary: step_face (x=0, y in [0,h_s])
    {
        Patch p;
        p.name = "step_face";
        p.type = "wall";
        for (int j = 0; j < ny_down; ++j) {
            Face &f = m.faces_[fi];
            f.owner = cid_low(0, j);
            f.neighbor = -1;
            f.patchID = static_cast<int>(m.patches_.size());
            f.center = Vec3(0.0, 0.5 * (yc_low[j] + yc_low[j + 1]), 0.5 * dz);
            f.area = (yc_low[j + 1] - yc_low[j]) * dz;
            f.normal = Vec3(-1, 0, 0);
            m.cells_[f.owner].faces.push_back(fi);
            p.faces.push_back(fi);
            ++fi;
        }
        m.patches_.push_back(std::move(p));
    }
    // Boundary: bottom_wall_down (y=0)
    {
        Patch p;
        p.name = "bottom_wall_down";
        p.type = "wall";
        for (int i = 0; i < nx_down; ++i) {
            Face &f = m.faces_[fi];
            f.owner = cid_low(i, 0);
            f.neighbor = -1;
            f.patchID = static_cast<int>(m.patches_.size());
            f.center = Vec3(0.5 * (xc[nx_up + i] + xc[nx_up + i + 1]), 0.0, 0.5 * dz);
            f.area = (xc[nx_up + i + 1] - xc[nx_up + i]) * dz;
            f.normal = Vec3(0, -1, 0);
            m.cells_[f.owner].faces.push_back(fi);
            p.faces.push_back(fi);
            ++fi;
        }
        m.patches_.push_back(std::move(p));
    }
    // Boundary: inlet
    {
        Patch p;
        p.name = "inlet";
        p.type = "inlet";
        for (int j = 0; j < ny_up; ++j) {
            Face &f = m.faces_[fi];
            f.owner = cid_up(0, j);
            f.neighbor = -1;
            f.patchID = static_cast<int>(m.patches_.size());
            f.center = Vec3(-Lu, 0.5 * (yc_up[j] + yc_up[j + 1]), 0.5 * dz);
            f.area = (yc_up[j + 1] - yc_up[j]) * dz;
            f.normal = Vec3(-1, 0, 0);
            m.cells_[f.owner].faces.push_back(fi);
            p.faces.push_back(fi);
            ++fi;
        }
        m.patches_.push_back(std::move(p));
    }
    // Boundary: outlet (upper + lower)
    {
        Patch p;
        p.name = "outlet";
        p.type = "outlet";
        for (int j = 0; j < ny_up; ++j) {
            Face &f = m.faces_[fi];
            f.owner = cid_up(nx_tot - 1, j);
            f.neighbor = -1;
            f.patchID = static_cast<int>(m.patches_.size());
            f.center = Vec3(Ld, 0.5 * (yc_up[j] + yc_up[j + 1]), 0.5 * dz);
            f.area = (yc_up[j + 1] - yc_up[j]) * dz;
            f.normal = Vec3(1, 0, 0);
            m.cells_[f.owner].faces.push_back(fi);
            p.faces.push_back(fi);
            ++fi;
        }
        for (int j = 0; j < ny_down; ++j) {
            Face &f = m.faces_[fi];
            f.owner = cid_low(nx_down - 1, j);
            f.neighbor = -1;
            f.patchID = static_cast<int>(m.patches_.size());
            f.center = Vec3(Ld, 0.5 * (yc_low[j] + yc_low[j + 1]), 0.5 * dz);
            f.area = (yc_low[j + 1] - yc_low[j]) * dz;
            f.normal = Vec3(1, 0, 0);
            m.cells_[f.owner].faces.push_back(fi);
            p.faces.push_back(fi);
            ++fi;
        }
        m.patches_.push_back(std::move(p));
    }

    // Owner/neighbor lists
    m.ownerList_.resize(nInternal);
    m.neighborList_.resize(nInternal);
    for (int f = 0; f < nInternal; ++f) {
        m.ownerList_[f] = m.faces_[f].owner;
        m.neighborList_[f] = m.faces_[f].neighbor;
    }

    // Cell centers and volumes
    for (int j = 0; j < ny_up; ++j)
        for (int i = 0; i < nx_tot; ++i) {
            Cell &c = m.cells_[cid_up(i, j)];
            c.center = Vec3(0.5 * (xc[i] + xc[i + 1]), 0.5 * (yc_up[j] + yc_up[j + 1]), 0.5 * dz);
            c.volume = (xc[i + 1] - xc[i]) * (yc_up[j + 1] - yc_up[j]) * dz;
        }
    for (int j = 0; j < ny_down; ++j)
        for (int i = 0; i < nx_down; ++i) {
            Cell &c = m.cells_[cid_low(i, j)];
            c.center = Vec3(0.5 * (xc[nx_up + i] + xc[nx_up + i + 1]),
                            0.5 * (yc_low[j] + yc_low[j + 1]), 0.5 * dz);
            c.volume = (xc[nx_up + i + 1] - xc[nx_up + i]) * (yc_low[j + 1] - yc_low[j]) * dz;
        }

    m.computeFaceGeometry();
    m.ComputeInterpolationWeights();

    std::cout << "BackwardFacingStep2D: [" << nx_up << "+" << nx_down << "]x[" << ny_up << "+"
              << ny_down << "]"
              << " (" << nC << " cells, " << m.nFaces() << " faces)\n";
    return m;
}

Mesh::Mesh(const Mesh &other) {
    *this = other;
}
Mesh::Mesh(Mesh &&other) noexcept {
    *this = std::move(other);
}
Mesh &Mesh::operator=(const Mesh &other) {
    if (this == &other)
        return *this;
    invalidate();
    cells_ = other.cells_;
    faces_ = other.faces_;
    nodes_ = other.nodes_;
    patches_ = other.patches_;
    wallDist_ = other.wallDist_;
    ownerList_ = other.ownerList_;
    neighborList_ = other.neighborList_;
    nInternal_ = other.nInternal_;
    wallDistanceValid_ = other.wallDistanceValid_;
    return *this;
}
Mesh &Mesh::operator=(Mesh &&other) noexcept {
    if (this == &other)
        return *this;
    invalidate();
    cells_ = std::move(other.cells_);
    faces_ = std::move(other.faces_);
    nodes_ = std::move(other.nodes_);
    patches_ = std::move(other.patches_);
    wallDist_ = std::move(other.wallDist_);
    ownerList_ = std::move(other.ownerList_);
    neighborList_ = std::move(other.neighborList_);
    nInternal_ = other.nInternal_;
    wallDistanceValid_ = other.wallDistanceValid_;
    other.invalidate();
    other.nInternal_ = 0;
    return *this;
}
