#pragma once

#include <vector>
#include <array>
#include <string>
#include <unordered_map>
#include <stdexcept>
#include <cmath>
#include <atomic>
#include <cstdint>
#include <utility>

// 3D cartesian vector with vector operations
struct Vec3 {
    double x = 0.0, y = 0.0, z = 0.0;

    Vec3() = default;
    Vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {} // 3D vector constructor

    Vec3 operator+(const Vec3 &o) const { return {x + o.x, y + o.y, z + o.z}; } // vector addition
    Vec3 operator-(const Vec3 &o) const {
        return {x - o.x, y - o.y, z - o.z};
    } // vector subtraction
    Vec3 operator*(double s) const { return {x * s, y * s, z * s}; } // vector scalor multiplication
    Vec3 operator/(double s) const { return {x / s, y / s, z / s}; } // vector scalor division

    double dot(const Vec3 &o) const { return x * o.x + y * o.y + z * o.z; } // dot product

    Vec3 cross(const Vec3 &o) const { // cross product
        return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
    }
    double norm() const { return std::sqrt(x * x + y * y + z * z); } // vector norm
    double norm2() const { return x * x + y * y + z * z; }           // vector norm squared

    Vec3 unit() const {
        double n = norm(); // unit vector
        return (n > 1e-300) ? (*this) / n : Vec3{};
    }
};

inline Vec3 operator*(double s, const Vec3 &v) {
    return v * s;
} // vector scaling

// Mesh Topology Indices
using CellID = int;  // volumes
using FaceID = int;  // interface between cells
using NodeID = int;  // corner points
using PatchID = int; // boundary groups

// Face data
struct Face {
    CellID owner = -1;    // cell normal vector points away
    CellID neighbor = -1; // cell that outward-from-owner normal points towards

    // each face is connected to two cells, each initialized to -1 to indicate no cell assigned

    Vec3 center;        // face centroid
    Vec3 normal;        // outward-from-owner unit normal
    double area = 0.0;  // face area
    double delta = 0.0; // distance betwwen owner and neighbor cell centers
    Vec3 d;             // vector from owner to neighbor centers

    double weight = 0.5; // interpolation weight: phi_f = w*phi_P + (1-w)*phi_N
    // Periodic translation of the neighbour: on a wrap face the neighbour's
    // image adjacent to this face sits at cells[neighbor].center + shift, and
    // the face seen from the neighbour lies at center - shift. Zero on every
    // ordinary face. Set by the periodic generator so that recomputing the
    // geometry cannot collapse a wrap face onto the far side of the domain.
    Vec3 shift;

    // at boundaries, owner = #, neighbor = -1 since there is no neighboring cell
    bool isBoundary() const { return neighbor < 0; }

    // patch index for boundary faces
    PatchID patchID = -1;
};

// no node struct because a mesh node is just a Vec3

// Cell data
struct Cell {
    Vec3 center;
    double volume = 0.0;       // cell volume
    std::vector<FaceID> faces; // faces that bound this cell
};

// Patch data
struct Patch {
    // actual Boundary condition values will live in a different file (BoundaryCondition.hpp)
    std::string name;
    std::string type; // "wall", "inlet", "outlet", ...
    std::vector<FaceID> faces;
};

// Mesh
// Computational grid with cells, faces, nodes, and boundary patches
class Mesh {
  public:
    struct Stamp {
        std::uint64_t identity = 0, revision = 0;
        bool operator==(const Stamp &other) const {
            return identity == other.identity && revision == other.revision;
        }
        bool operator!=(const Stamp &other) const { return !(*this == other); }
    };
    Stamp stamp() const { return {identity_, revision_}; }
    void requireStamp(Stamp expected) const {
        if (stamp() != expected)
            throw std::logic_error(
                "stale mesh-dependent state; reconstruct it after changing the mesh");
    }
    bool wallDistanceValid() const { return wallDistanceValid_; }

  private:
    static std::uint64_t nextIdentity() {
        static std::atomic<std::uint64_t> next{1};
        return next.fetch_add(1, std::memory_order_relaxed);
    }
    std::uint64_t identity_ = nextIdentity(), revision_ = 0;
    bool wallDistanceValid_ = false;
    void invalidate() {
        ++revision_;
        wallDistanceValid_ = false;
    }

    std::vector<Cell> cells_;
    std::vector<Face> faces_;
    std::vector<Vec3> nodes_;
    std::vector<Patch> patches_;
    std::vector<double> wallDist_; // wallDist data -> populated by computeWallDistance
    std::vector<int> ownerList_;
    std::vector<int> neighborList_;

    int nInternal_ = 0; // numner of internal faces

    // Sub-loaders
    static Mesh loadOpenFOAM(const std::string &polyMeshDir);
    static Mesh loadBinary(const std::string &path);

    void computeFaceGeometry();
    void computeCellCentersAndVolumes();
    void ComputeInterpolationWeights();

    // shared body of the periodic generators: y1t > 0 sets the first-cell
    // height, otherwise a default tanh stretch of 2
    static Mesh buildPeriodic2D(const std::vector<double> &xNodes,
                                const std::vector<double> &yBottom, double yTop, int ny, double y1t,
                                const char *bottomName, const char *topName);

  public:
    Mesh() = default;
    Mesh(const Mesh &other);
    Mesh(Mesh &&other) noexcept;
    Mesh &operator=(const Mesh &other);
    Mesh &operator=(Mesh &&other) noexcept;

    // returns constructed mesh from loaded file (OpenFOAM format or custom binary format)
    static Mesh loadFromFile(const std::string &path);

    // test mesh (2D channel mesh)
    // domain [0,Lx] x [0,Ly]), useful for validation before full meshes.
    // With inlet/outlet patches this is a DEVELOPING channel; compare with
    // fully developed reference data on makePeriodicChannel2D instead.
    static Mesh makeChannel2D(int nx, int ny, double Lx, double Ly);

    // Re-adaptive wall clustering overload: computes tanh stretch parameter
    // so the first cell targets yPlusTarget based on Re.
    static Mesh makeChannel2D(int nx, int ny, double Lx, double Ly, double Re,
                              double yPlusTarget = 1.0);

    // One-sided wall-clustered plate mesh: all refinement at the bottom wall
    // (the top is a far-field patch, not a wall), so the near-wall growth
    // ratio at matched ny is about half the two-sided channel's; the
    // turbulent near-wall omega field needs that resolution.
    static Mesh makePlate2D(int nx, int ny, double Lx, double Ly, double Re,
                            double yPlusTarget = 1.0);

    // Backward-facing step: L-shaped domain with upstream channel + expanded duct.
    // Patches: inlet, outlet, top_wall, bottom_wall_up, step_face, bottom_wall_down.
    // nx_up/nx_down: cells in x for upstream / downstream region.
    // ny_up/ny_down: cells in y for upper (h_s..H) / lower (0..h_s) block.
    // Lu/Ld: upstream / downstream length; h_s: step height; H: total downstream height.
    static Mesh makeBackwardFacingStep2D(int nx_up, int nx_down, int ny_up, int ny_down, double Lu,
                                         double Ld, double h_s, double H);
    static Mesh makeBackwardFacingStep2D(int nx_up, int nx_down, int ny_up, int ny_down, double Lu,
                                         double Ld, double h_s, double H, double Re,
                                         double yPlusTarget = 1.0);

    // Streamwise-PERIODIC curved-bottom channel (the periodic-hills family).
    // The bottom boundary is a prescribed curve y_bottom(x) sampled at the
    // nx+1 x-node locations (first and last must match: one periodic column);
    // the top is flat at yTop. Periodicity is built as wrap-around INTERNAL
    // faces between the last and first cell columns (no boundary condition is
    // involved), with the face distance vector, delta and interpolation weight
    // computed against the periodic image of the neighbor. The flow must be
    // driven by a body force (SolverSettings::bodyForce): there is no inlet.
    // Patches: bottom_wall (curved, type wall), top_wall (flat, type wall).
    static Mesh makeCurvedChannelPeriodic2D(const std::vector<double> &xNodes,
                                            const std::vector<double> &yBottom, double yTop, int ny,
                                            double Re, double yPlusTarget = 1.0);

    // The canonical channel for comparison with FULLY DEVELOPED reference data:
    // a flat streamwise-periodic channel of height 2h (walls "bottom" at y = 0
    // and "top" at y = 2h, the names the channel observations use), wall
    // clustering sized for y1+ = yPlusTarget at Re_tau. Run it in u_tau = 1
    // units: nu = h / Re_tau and SolverSettings::bodyForce.x = 1 / h, which
    // the global balance f V = integral of wall shear turns into exactly
    // Re_tau, with no viscosity matching. The flow is x-invariant, so a few
    // columns suffice (nx >= 3). The inlet/outlet makeChannel2D channel is a
    // developing flow and is not interchangeable with a fully developed
    // periodic reference.
    static Mesh makePeriodicChannel2D(int nx, int ny, double Lx, double h, double ReTau,
                                      double yPlusTarget = 1.0);

    // returns count
    int nCells() const { return static_cast<int>(cells_.size()); }
    int nFaces() const { return static_cast<int>(faces_.size()); }
    int nNodes() const { return static_cast<int>(nodes_.size()); }
    int nPatches() const { return static_cast<int>(patches_.size()); }

    // returns specific object by ID
    const Cell &cell(CellID id) const { return cells_[id]; }
    const Face &face(FaceID id) const { return faces_[id]; }
    const Vec3 &node(NodeID id) const { return nodes_[id]; }
    const Patch &patch(PatchID id) const { return patches_[id]; }

    PatchID patchByName(const std::string &name) const; // patch lookup by name

    // Retype a boundary patch (e.g. "wall" -> "symmetry"), so BC factories and
    // computeWallDistance see the new type. Call computeWallDistance afterwards:
    // a patch retyped away from "wall" must stop contributing wall distances
    // (the Le-Moin BFS top boundary is free-slip, not a turbulence wall).
    void setPatchType(const std::string &name, const std::string &type);

    // iterators (range-based for loops )
    const std::vector<Cell> &cells() const { return cells_; }
    const std::vector<Face> &faces() const { return faces_; }
    const std::vector<Patch> &patches() const { return patches_; }

    // computes geometric quantites like interpolation weights, delta vectors, etc.
    // must be called after modifying topology. Uses the single cell-geometry
    // convention (exact volumes and centroids, see computeCellCentersAndVolumes),
    // so on a generator mesh it reproduces the generator's geometry to round-off.
    void computeGeometry();

    // distance from cell center to nearest wall (for SST blending)
    const std::vector<double> &wallDistance() const {
        if (!wallDistanceValid_)
            throw std::logic_error("stale or absent wall distance; call computeWallDistance()");
        return wallDist_;
    }
    void computeWallDistance();

    // topology access for linear solver

    int nInternalFaces() const { return nInternal_; } // returns number of internal faces
    // internal faces [0, nInternal), boundary faces [nInternal_, nFaces_)

    // owner/neighbor list used by LinearSolver for matrix-vector multiplcation
    const std::vector<int> &ownerList() const { return ownerList_; }
    const std::vector<int> &neighborList() const { return neighborList_; }

    // MSH2 binary storage preserves periodic translations. The format uses
    // native byte order, int32 indices and IEEE binary64 coordinates; readers
    // reject the opposite byte order. Legacy MSH1 nonperiodic files are read.
    // Wall distance is derived state and must be recomputed after import.
    void saveBinary(const std::string &path) const;
};