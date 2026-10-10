#pragma once

#include "Mesh.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <deque>
#include <limits>
#include <stdexcept>
#include <string>

// Field<T> -- cell-centered FVM field (control volume)
// this means that physical variables are stored as average values at the centroid of each mesh cell
// T = double   -> Scalar Field
// T = vec3     -> Vector Field
// for storing Field<T> quantities:
// cell-centered (interior) values and boundary-face values are stored in separate arrays.
//      interior_[0..nCells)              — cell values
//      boundary_[0..nBoundaryFaces)      — boundary face values

template <typename T> class Field {
  public:
    std::string name;

    Field() = default; // default constructor

    // storage allocator for mesh
    explicit Field(const Mesh &mesh, const std::string &n = "", T init = T{})
        : name(n), mesh_(&mesh), meshStamp_(mesh.stamp()), interior_(mesh.nCells(), init),
          boundary_(mesh.nFaces() - mesh.nInternalFaces(), init) {}

    // cell access via operator[]
    T &operator[](int ci) {
        requireValid();
        return interior_[ci];
    }
    const T &operator[](int ci) const {
        requireValid();
        return interior_[ci];
    }

    // boundary face access by local boundary index
    T &bnd(int bi) {
        requireValid();
        return boundary_[bi];
    }
    const T &bnd(int bi) const {
        requireValid();
        return boundary_[bi];
    }

    // boundary face access by global face ID
    T &bface(int fi) {
        requireValid();
        return boundary_[fi - mesh_->nInternalFaces()];
    }
    const T &bface(int fi) const {
        requireValid();
        return boundary_[fi - mesh_->nInternalFaces()];
    }

    // returns number of cell-centered values in field (= mesh.nCells)
    int size() const { return static_cast<int>(interior_.size()); }
    // returns number of boundary-face values (equals nFaces- nInternalFaces)
    int nBnd() const { return static_cast<int>(boundary_.size()); }

    // direct access to interior (cell-centered) storage.
    // returns reference to underlying vector for solver/linear operations.
    std::vector<T> &data() {
        if (mesh_)
            requireValid();
        return interior_;
    }
    const std::vector<T> &data() const {
        if (mesh_)
            requireValid();
        return interior_;
    }

    // dereferenecs stored pointer; returns reference to mesh
    const Mesh &mesh() const {
        requireValid();
        return *mesh_;
    }

    // Non-owning identity for validation; null for a default-constructed field.
    const Mesh *meshPtr() const noexcept { return mesh_; }

    void requireValid() const {
        if (!mesh_)
            throw std::logic_error("unbound field has no mesh");
        mesh_->requireStamp(meshStamp_);
    }

    // face interpolation
    // computes field value at a face
    T faceValue(FaceID fi) const {
        requireValid();
        const Face &f = mesh_->face(fi);
        if (f.isBoundary())
            return bface(fi); // returns stored boundary value (since no neighbor)
        return interior_[f.owner] * f.weight +
               interior_[f.neighbor] *
                   (1.0 - f.weight); // returns computed interpolated internal value
    }

    // norms
    // computes root-mean-square norm; handles both scalars and vectors
    double l2Norm() const {
        requireValid();
        double s = 0.0;
        for (auto &v : interior_)
            s += sqMag(v);
        return std::sqrt(s / std::max(1, size()));
    }
    // Maximum magnitude; nonfinite input must remain visible to callers.
    double linfNorm() const {
        requireValid();
        double m = 0.0;
        for (auto &v : interior_) {
            const double value = mag(v);
            if (!std::isfinite(value))
                return value;
            m = std::max(m, value);
        }
        return m;
    }

    // arithmetic helpers
    // sets every cell and boundary face value equal to val
    void setUniform(T val) {
        requireValid();
        std::fill(interior_.begin(), interior_.end(), val);
        std::fill(boundary_.begin(), boundary_.end(), val);
    }
    // forces all values into the range lo <= v <= hi
    void clamp(double lo, double hi) {
        requireValid();
        for (auto &v : interior_)
            doClamp(v, lo, hi);
        for (auto &v : boundary_)
            doClamp(v, lo, hi);
    }

  private:
    const Mesh *mesh_ = nullptr; // pointer to mesh
    Mesh::Stamp meshStamp_;
    std::vector<T> interior_; // stores cell-cenetered values
    std::vector<T> boundary_; // stores boundary face values

    // static helper functions
    // -> correct behavior for both scalar/vector field
    static double sqMag(double v) { return v * v; }
    static double sqMag(const Vec3 &v) { return v.norm2(); } // type polymorphism by overloading
    static double mag(double v) { return std::abs(v); }
    static double mag(const Vec3 &v) { return v.norm(); }
    static void doClamp(double &v, double lo, double hi) { v = std::max(lo, std::min(hi, v)); }
    static void doClamp(Vec3 &v, double lo, double hi) {
        v.x = std::max(lo, std::min(hi, v.x));
        v.y = std::max(lo, std::min(hi, v.y));
        v.z = std::max(lo, std::min(hi, v.z));
    }
};

using ScalarField = Field<double>;
using VectorField = Field<Vec3>;

// Green-Gauss gradient: (phi)_C = ∇(1/V_C) Σ_f (phi_f · S_f)
// where phi_f is the field value at face, S_f is the face area vector, and V_C is the cell volume
// computes the gradient of the scalar field at each cell center using Green-Gauss theorem
template <class T> inline void requireFieldStorage(const Mesh &mesh, const Field<T> &field) {
    if (field.meshPtr() != &mesh || field.size() != mesh.nCells() ||
        field.nBnd() != mesh.nFaces() - mesh.nInternalFaces())
        throw std::invalid_argument("field storage must match the mesh");
    field.requireValid();
}

inline void greenGaussGrad(const ScalarField &phi, VectorField &grad) {
    // input scalar field phi, output vector field -> each cell gets 3D vector gradient
    const Mesh &m = phi.mesh();
    requireFieldStorage(m, phi);
    requireFieldStorage(m, grad);
    grad.setUniform(Vec3{});

    // loops over internal faces
    for (int fi = 0; fi < m.nInternalFaces(); ++fi) {
        const Face &f = m.face(fi);
        double phiF =
            phi[f.owner] * f.weight + phi[f.neighbor] * (1.0 - f.weight); // compute face value
        Vec3 Sf = f.normal * f.area;                                      // get area vector
        grad[f.owner] = grad[f.owner] + Sf * phiF; // owner cell contribution (phi_f * S_f)
        grad[f.neighbor] =
            grad[f.neighbor] -
            Sf * phiF; // neighbor cell contribution
                       // need to correct orientation for GG grad, thus the minus sign
    }

    // loops over boundary faces
    for (int fi = m.nInternalFaces(); fi < m.nFaces(); ++fi) {
        const Face &f = m.face(fi);
        Vec3 Sf = f.normal * f.area;
        // add boundary conditions (boundary faces only contribute to the owner cell)
        grad[f.owner] = grad[f.owner] + Sf * phi.bface(fi);
    }

    // divide by volume
    for (int ci = 0; ci < m.nCells(); ++ci) {
        double V = m.cell(ci).volume;
        if (V > 1e-30)
            grad[ci] = grad[ci] / V;
    }
}

inline VectorField greenGaussGrad(const ScalarField &phi) {
    VectorField grad(phi.mesh(), "grad(" + phi.name + ")");
    greenGaussGrad(phi, grad);
    return grad;
}

// Velocity gradient (component-wise Green-Gauss)
struct VelocityGradients {
    VectorField dudx; // (du/dx, du/dy, du/dz)
    VectorField dvdx; // (dv/dx, dv/dy, dv/dz)
    VectorField dwdx; // (dw/dx, dw/dy, dw/dz)
};

// Odd-even (checkerboard) energy ratio of a cell field: the magnitude of the
// volume-weighted projection of the demeaned field onto the alternating
// two-colour mode of the cell-adjacency graph, normalized by the field and
// mode energies. The value is in [0,1]: an exact checkerboard scores one and
// a smooth field with no alternating component scores near zero.
//
// The mesh factories used by the collocated SIMPLE solver produce bipartite
// quadrilateral cell graphs, including their periodic wrap faces. Requiring
// bipartiteness is deliberate: the all-Neumann odd-even null mode being
// diagnosed is precisely this graph mode. The former neighbour-average
// residual also responded strongly to smooth curvature and boundary stencil
// imbalance (a solved smooth channel scored 0.59), so it was not a specific
// checkerboard measurement.
// Projection of a cell field onto an alternating mode of the cell graph,
// built by walking the graph and flipping the colour across every face for
// which flips(face) is true. Returns NaN if the rule is inconsistent on some
// cycle (the mode does not exist on this mesh).
template <class Flip>
inline double oddEvenProjection(const Mesh &mesh, const ScalarField &phi, Flip flips) {
    int n = mesh.nCells();
    if (n == 0)
        return 0.0;

    std::vector<std::vector<std::pair<int, bool>>> adjacency(n);
    for (int fi = 0; fi < mesh.nInternalFaces(); ++fi) {
        const Face &face = mesh.face(fi);
        const bool flip = flips(face);
        adjacency[face.owner].emplace_back(face.neighbor, flip);
        adjacency[face.neighbor].emplace_back(face.owner, flip);
    }

    std::vector<int> colour(n, -1);
    std::deque<int> queue;
    for (int seed = 0; seed < n; ++seed) {
        if (colour[seed] != -1)
            continue;
        colour[seed] = 0;
        queue.push_back(seed);
        while (!queue.empty()) {
            int ci = queue.front();
            queue.pop_front();
            for (const auto &[nb, flip] : adjacency[ci]) {
                int want = flip ? 1 - colour[ci] : colour[ci];
                if (colour[nb] == -1) {
                    colour[nb] = want;
                    queue.push_back(nb);
                } else if (colour[nb] != want) {
                    return std::numeric_limits<double>::quiet_NaN();
                }
            }
        }
    }

    double weight = 0.0, mean = 0.0, modeMean = 0.0;
    for (int ci = 0; ci < n; ++ci) {
        double w = std::max(mesh.cell(ci).volume, 0.0);
        double q = colour[ci] == 0 ? 1.0 : -1.0;
        weight += w;
        mean += w * phi[ci];
        modeMean += w * q;
    }
    if (weight <= 1e-300)
        return 0.0;
    mean /= weight;
    modeMean /= weight;

    double projection = 0.0, fieldEnergy = 0.0, modeEnergy = 0.0;
    for (int ci = 0; ci < n; ++ci) {
        double w = std::max(mesh.cell(ci).volume, 0.0);
        double v = phi[ci] - mean;
        double q = (colour[ci] == 0 ? 1.0 : -1.0) - modeMean;
        projection += w * v * q;
        fieldEnergy += w * v * v;
        modeEnergy += w * q * q;
    }
    if (fieldEnergy <= 1e-300 || modeEnergy <= 1e-300)
        return 0.0;
    return std::abs(projection) / std::sqrt(fieldEnergy * modeEnergy);
}

inline double oddEvenEnergyRatio(const Mesh &mesh, const ScalarField &phi) {
    double r = oddEvenProjection(mesh, phi, [](const Face &) { return true; });
    if (std::isnan(r))
        throw std::invalid_argument("oddEvenEnergyRatio requires a bipartite cell graph");
    return r;
}

// The two-colour (pi,pi) mode misses the commonest collocated-grid failure,
// an alternation along ONE direction that is smooth along the other. These
// directional modes flip colour only across faces whose normal points mainly
// along x (or y). NaN where a mode does not exist on the mesh (e.g. x on a
// periodic mesh with an odd number of columns).
struct OddEvenRatios {
    double twoColour = 0.0, x = 0.0, y = 0.0;
};

inline OddEvenRatios oddEvenEnergyRatios(const Mesh &mesh, const ScalarField &phi) {
    OddEvenRatios r;
    r.twoColour = oddEvenProjection(mesh, phi, [](const Face &) { return true; });
    r.x = oddEvenProjection(
        mesh, phi, [](const Face &f) { return std::abs(f.normal.x) >= std::abs(f.normal.y); });
    r.y = oddEvenProjection(
        mesh, phi, [](const Face &f) { return std::abs(f.normal.y) > std::abs(f.normal.x); });
    return r;
}

struct VelocityGradientWorkspace {
    ScalarField Ux, Uy, Uz;
    explicit VelocityGradientWorkspace(const Mesh &mesh)
        : Ux(mesh, "Ux"), Uy(mesh, "Uy"), Uz(mesh, "Uz") {}
};

inline void computeVelocityGradients(const VectorField &U, VelocityGradients &output,
                                     VelocityGradientWorkspace &workspace) {
    const Mesh &m = U.mesh();
    requireFieldStorage(m, U);
    for (const auto *gradient : {&output.dudx, &output.dvdx, &output.dwdx}) {
        if (gradient == &U)
            throw std::invalid_argument("gradient output cannot alias velocity input");
        requireFieldStorage(m, *gradient);
    }
    for (const auto *field : {&workspace.Ux, &workspace.Uy, &workspace.Uz})
        requireFieldStorage(m, *field);
    auto &Ux = workspace.Ux;
    auto &Uy = workspace.Uy;
    auto &Uz = workspace.Uz;

    for (int ci = 0; ci < m.nCells(); ++ci) {
        Ux[ci] = U[ci].x;
        Uy[ci] = U[ci].y;
        Uz[ci] = U[ci].z;
    }
    for (int fi = m.nInternalFaces(); fi < m.nFaces(); ++fi) {
        Ux.bface(fi) = U.bface(fi).x;
        Uy.bface(fi) = U.bface(fi).y;
        Uz.bface(fi) = U.bface(fi).z;
    }

    greenGaussGrad(Ux, output.dudx);
    greenGaussGrad(Uy, output.dvdx);
    greenGaussGrad(Uz, output.dwdx);
}

inline VelocityGradients computeVelocityGradients(const VectorField &U) {
    const Mesh &mesh = U.mesh();
    VelocityGradientWorkspace workspace(mesh);
    VelocityGradients output{VectorField(mesh, "grad(Ux)"), VectorField(mesh, "grad(Uy)"),
                             VectorField(mesh, "grad(Uz)")};
    computeVelocityGradients(U, output, workspace);
    return output;
}

// Velocity gradient can be decomposed into:
// Strain-Rate tensor S (symmetric part)
// Rotation tensor W (antisymmetric part)

// Strain Rate Magnitude
// |S| = sqrt(2 * S_ij * S_ij),  S_ij = 0.5(dUi/dxj + dUj/dxi)
// Caller-owned output must already have storage for the gradients' mesh.
// Interior values are overwritten; boundary values match the returning form's zeros.
inline void strainRateMagnitude(const VelocityGradients &vg, ScalarField &S) {
    const Mesh *mesh = vg.dudx.meshPtr();
    if (!mesh)
        throw std::invalid_argument("strainRateMagnitude requires mesh-bound gradients");
    const Mesh &m = *mesh;
    const auto matches = [&m](const auto &field) {
        return field.meshPtr() == &m && field.size() == m.nCells() &&
               field.nBnd() == m.nFaces() - m.nInternalFaces();
    };
    if (!matches(vg.dudx) || !matches(vg.dvdx) || !matches(vg.dwdx) || !matches(S))
        throw std::invalid_argument("strainRateMagnitude requires matching mesh and field storage");
    vg.dudx.requireValid();
    vg.dvdx.requireValid();
    vg.dwdx.requireValid();
    S.requireValid();
    for (int bi = 0; bi < S.nBnd(); ++bi)
        S.bnd(bi) = 0.0;
    for (int ci = 0; ci < m.nCells(); ++ci) {
        double S11 = vg.dudx[ci].x;
        double S22 = vg.dvdx[ci].y;
        double S33 = vg.dwdx[ci].z;
        double S12 = 0.5 * (vg.dudx[ci].y + vg.dvdx[ci].x);
        double S13 = 0.5 * (vg.dudx[ci].z + vg.dwdx[ci].x);
        double S23 = 0.5 * (vg.dvdx[ci].z + vg.dwdx[ci].y);
        S[ci] = std::sqrt(
            2.0 * (S11 * S11 + S22 * S22 + S33 * S33 + 2.0 * (S12 * S12 + S13 * S13 + S23 * S23)));
    }
}

inline ScalarField strainRateMagnitude(const VelocityGradients &vg) {
    const Mesh *mesh = vg.dudx.meshPtr();
    if (!mesh)
        throw std::invalid_argument("strainRateMagnitude requires mesh-bound gradients");
    ScalarField S(*mesh, "|S|");
    strainRateMagnitude(vg, S);
    return S;
}

// Vorticity Magnitude
// |Omega| = sqrt(2 * W_ij * W_ij),  W_ij = 0.5(dUi/dxj - dUj/dxi)
inline ScalarField vorticityMagnitude(const VelocityGradients &vg) {
    const Mesh &m = vg.dudx.mesh();
    ScalarField W(m, "|W|");
    for (int ci = 0; ci < m.nCells(); ++ci) {
        double W12 = 0.5 * (vg.dudx[ci].y - vg.dvdx[ci].x);
        double W13 = 0.5 * (vg.dudx[ci].z - vg.dwdx[ci].x);
        double W23 = 0.5 * (vg.dvdx[ci].z - vg.dwdx[ci].y);
        W[ci] = std::sqrt(2.0 * 2.0 * (W12 * W12 + W13 * W13 + W23 * W23));
    }
    return W;
}
