#pragma once

#include "Field.hpp"
#include "FlowFields.hpp"
#include <array>

// Small forward-mode scalar used only for coefficient partials. The primal
// assembly calls the same face kernels with double. Branches use ordinary
// primal values; no smoothing or frozen-branch finite differences are used.
namespace diagnostic {
struct Partial {
    double value = 0;
    std::array<double, 11> derivative{};
    Partial() = default;
    Partial(double v) : value(v) {} // NOLINT: scalar promotion in kernels
    Partial(double v, const double *d) : value(v) { std::copy(d, d + 11, derivative.begin()); }
    Partial &operator+=(const Partial &b) {
        value += b.value;
        for (int j = 0; j < 11; ++j)
            derivative[j] += b.derivative[j];
        return *this;
    }
};
inline Partial operator+(Partial a, const Partial &b) {
    return a += b;
}
inline Partial operator-(const Partial &a) {
    Partial r(-a.value);
    for (int j = 0; j < 11; ++j)
        r.derivative[j] = -a.derivative[j];
    return r;
}
inline Partial operator-(const Partial &a, const Partial &b) {
    return a + -b;
}
inline Partial operator*(const Partial &a, const Partial &b) {
    Partial r(a.value * b.value);
    for (int j = 0; j < 11; ++j)
        r.derivative[j] = a.derivative[j] * b.value + a.value * b.derivative[j];
    return r;
}
inline Partial operator/(const Partial &a, const Partial &b) {
    Partial r(a.value / b.value);
    for (int j = 0; j < 11; ++j)
        r.derivative[j] = (a.derivative[j] - r.value * b.derivative[j]) / b.value;
    return r;
}
inline double value(double x) {
    return x;
}
inline double value(const Partial &x) {
    return x.value;
}
template <class T> T positive(const T &x) {
    return value(x) > 0 ? x : T(0);
}

template <class T> struct FaceCoefficients {
    T owner, neighbor, upper, lower;
};
template <class T>
FaceCoefficients<T> momentumInternal(const Face &face, double nu, const T &nuO, const T &nuN,
                                     const T &flux) {
    const T effO = nu + nuO, effN = nu + nuN;
    const T effF = face.weight * effO + (1.0 - face.weight) * effN;
    const T diffusion = effF * face.area / std::max(face.delta, 1e-20);
    const T pos = positive(flux), neg = positive(-flux);
    return {diffusion + pos, diffusion + neg, -(diffusion + neg), -(diffusion + pos)};
}
template <class T>
T momentumBoundaryDiagonal(const Face &face, double nu, const T &nuT, const T &flux,
                           bool resolvedWall) {
    const T eff = resolvedWall ? T(nu) : nu + nuT;
    const T diffusion = eff * face.area / std::max(face.delta, 1e-20);
    // Preserve the outflow addition order for reproducible residuals.
    return value(flux) >= 0 ? flux + diffusion : diffusion;
}
template <class T>
T rhieChow(const Mesh &mesh, const Face &face, const ScalarField &p, const VectorField &gradP,
           const T &aO, const T &aN) {
    const int o = face.owner, n = face.neighbor;
    const T dO = std::abs(value(aO)) > 1e-30 ? mesh.cell(o).volume / aO : T(0);
    const T dN = std::abs(value(aN)) > 1e-30 ? mesh.cell(n).volume / aN : T(0);
    const T dF = face.weight * dO + (1.0 - face.weight) * dN;
    const Vec3 e = face.d / std::max(face.delta, 1e-30);
    const Vec3 g = gradP[o] * face.weight + gradP[n] * (1.0 - face.weight);
    return -dF * face.area * ((p[n] - p[o]) / std::max(face.delta, 1e-20) - g.dot(e));
}
inline double convectionIncrement(const Mesh &mesh, const Face &face, const ScalarField &phi,
                                  const VectorField &grad, double flux, bool limited) {
    const bool ownerUp = flux > 0;
    const int up = ownerUp ? face.owner : face.neighbor;
    const int dn = ownerUp ? face.neighbor : face.owner;
    const Vec3 &xU = mesh.cell(up).center;
    double phiF;
    if (!limited) {
        const Vec3 xf = ownerUp ? face.center : face.center - face.shift;
        phiF = phi[up] + grad[up].dot(xf - xU);
    } else {
        const Vec3 xD =
            ownerUp ? mesh.cell(dn).center + face.shift : mesh.cell(dn).center - face.shift;
        const double dphi = phi[dn] - phi[up];
        double psi = 0;
        if (std::abs(dphi) > 1e-300) {
            const double r = 2.0 * grad[up].dot(xD - xU) / dphi - 1.0;
            psi = r > 0 ? (r * r + r) / (r * r + 1.0) : 0;
        }
        phiF = phi[up] + 0.5 * psi * dphi;
    }
    return phiF - phi[up];
}
// The value and derivative diagnostic paths both execute this exact two-pass
// algorithm. The solve loop retains its historical arithmetic and stored flux.
template <class T>
std::vector<T> reconstructFlux(const Mesh &mesh, const FlowFields &f, double nu,
                               const std::vector<T> &viscosity) {
    std::vector<T> phi(mesh.nFaces()), interpolated(mesh.nFaces());
    for (int fi = 0; fi < mesh.nFaces(); ++fi) {
        const auto &face = mesh.face(fi);
        const Vec3 u = face.isBoundary() ? f.U.bface(fi)
                                         : f.U[face.owner] * face.weight +
                                               f.U[face.neighbor] * (1.0 - face.weight);
        phi[fi] = interpolated[fi] =
            (u.x * face.normal.x + u.y * face.normal.y + u.z * face.normal.z) * face.area;
    }
    std::vector<char> wall(mesh.nFaces(), 0);
    for (int pi = 0; pi < mesh.nPatches(); ++pi)
        if (mesh.patch(pi).type == "wall")
            for (int fi : mesh.patch(pi).faces)
                wall[fi] = 1;
    const auto gradient = greenGaussGrad(f.p);
    for (int pass = 0; pass < 2; ++pass) {
        std::vector<T> diagonal(mesh.nCells());
        for (int fi = 0; fi < mesh.nInternalFaces(); ++fi) {
            const auto &face = mesh.face(fi);
            const auto c = momentumInternal(face, nu, viscosity[face.owner],
                                            viscosity[face.neighbor], phi[fi]);
            diagonal[face.owner] += c.owner;
            diagonal[face.neighbor] += c.neighbor;
        }
        for (int fi = mesh.nInternalFaces(); fi < mesh.nFaces(); ++fi) {
            const auto &face = mesh.face(fi);
            diagonal[face.owner] +=
                momentumBoundaryDiagonal(face, nu, viscosity[face.owner], phi[fi], wall[fi] != 0);
        }
        phi = interpolated;
        for (int fi = 0; fi < mesh.nInternalFaces(); ++fi) {
            const auto &face = mesh.face(fi);
            phi[fi] +=
                rhieChow(mesh, face, f.p, gradient, diagonal[face.owner], diagonal[face.neighbor]);
        }
    }
    return phi;
}
} // namespace diagnostic
