#pragma once
#include <SIMPLESolver.hpp>
#include <algorithm>
#include <cmath>
#include <stdexcept>

// Target-stress production for the installed hills consumer. Production is
// clipped to [0, 10 betaStar max(k, 0) max(omega, 1e-20)].
class HillsTargetProduction final : public SIMPLESolver::ProductionPolicy {
  public:
    static constexpr const char *identity = "hills-limited-target-production-v1";
    static constexpr const char *sourcePatchSha256 =
        "76923aed2479666912f7394eba6238092a4451c46a04a7e6777a2317af7f4513";
    void kProduction(const SIMPLESolver::ProductionContext &context,
                     std::vector<double> &output) const override {
        const auto &f = context.fields;
        const auto &coeffs = context.coefficients;
        const auto &target = context.targetAnisotropy;
        if (target.size() != static_cast<size_t>(6 * context.mesh.nCells()))
            throw std::invalid_argument("hills production needs six target entries per cell");
        const auto g = computeVelocityGradients(f.U);
        output.resize(context.mesh.nCells());
        for (int i = 0; i < context.mesh.nCells(); ++i) {
            const double *b = target.data() + 6 * i;
            const double tr = g.dudx[i].x + g.dvdx[i].y + g.dwdx[i].z;
            const double contraction = b[0] * g.dudx[i].x + b[1] * g.dvdx[i].y +
                                       b[2] * g.dwdx[i].z + b[3] * (g.dudx[i].y + g.dvdx[i].x) +
                                       b[4] * (g.dudx[i].z + g.dwdx[i].x) +
                                       b[5] * (g.dvdx[i].z + g.dwdx[i].y) + tr / 3.;
            const double k = std::max(f.k[i], 0.);
            const double raw = -2. * k * contraction;
            const double cap = 10. * coeffs.betaStar * k * std::max(f.omega[i], 1e-20);
            if (!std::isfinite(raw) || !std::isfinite(cap))
                throw std::runtime_error("nonfinite target production");
            output[i] = std::max(0., std::min(raw, cap));
        }
    }
    double omegaProduction(const SIMPLESolver::ProductionContext &context,
                           int cell) const override {
        return context.fields.Pk[cell] / std::max(context.fields.nuT[cell], 1e-30);
    }
};
