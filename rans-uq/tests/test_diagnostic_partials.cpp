#include "SIMPLESolver.hpp"
#include "InferenceParameters.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

// This is the same maximum-error reduction used by the numerical controls.
// Independent nonfinite controls below must fail before values can disappear.
static double checkedMaximum(double current, double error) {
    if (!std::isfinite(current) || !std::isfinite(error))
        throw std::runtime_error("nonfinite value in numerical error reduction");
    return std::max(current, error);
}

static bool maximumOracleRejectsNonfinite() {
    for (double invalid :
         {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(),
          -std::numeric_limits<double>::infinity()}) {
        bool rejected = false;
        try {
            (void)checkedMaximum(0.0, invalid);
        } catch (const std::runtime_error &) {
            rejected = true;
        }
        if (!rejected)
            return false;
    }
    return checkedMaximum(0.25, 0.5) == 0.5 && checkedMaximum(0.5, 0.25) == 0.5;
}

// No data/RNG. Fixed nondimensional length/velocity/density are one;
// integrated equations and face fluxes use unit scales. Source is the test binary.
int main() {
    if (!maximumOracleRejectsNonfinite()) {
        std::cerr << "diagnostic maximum-error oracle accepts nonfinite values\n";
        return 5;
    }
    int failures = 0;
    for (bool periodic : {false, true})
        for (const char *scheme : {"upwind", "linearUpwind"}) {
            Mesh mesh = periodic ? Mesh::makePeriodicChannel2D(4, 5, 2.0, 0.5, 100.0)
                                 : Mesh::makeChannel2D(4, 5, 2.0, 1.0);
            mesh.computeWallDistance();
            auto bcs = FlowBoundaryConditions::channelDefaults(mesh, 1.0, 0.02, 2.0);
            SolverSettings settings;
            settings.verbose = false;
            settings.momentumConvection = settings.turbulenceConvection = scheme;
            settings.alphaU = settings.alphaK = settings.alphaOmega = 1.0;
            const double nu = 1e-4;
            SIMPLESolver solver(mesh, SSTModel{}, bcs, nu, settings);
            FlowFields fields(mesh);
            auto fill = [](const Vec3 &x, Vec3 &u, double &p, double &k, double &omega) {
                u = {0.7 + 0.22 * x.x + 0.31 * x.y * x.y, 0.23 + 0.17 * x.y + 0.07 * x.x * x.y, 0};
                p = 0.13 * x.x * x.x + 0.09 * x.y * x.y + 0.05 * x.x * x.y;
                k = 0.006 + 0.002 * x.x + 0.003 * x.y;
                omega = 1.2 + 0.13 * x.x + 0.41 * x.y;
            };
            for (int i = 0; i < mesh.nCells(); ++i)
                fill(mesh.cell(i).center, fields.U[i], fields.p[i], fields.k[i], fields.omega[i]);
            for (int fi = mesh.nInternalFaces(); fi < mesh.nFaces(); ++fi)
                fill(mesh.face(fi).center, fields.U.bface(fi), fields.p.bface(fi),
                     fields.k.bface(fi), fields.omega.bface(fi));
            const auto theta = InferenceParameterSet::toVector(SSTCoefficients{});
            const auto columns = solver.diagnosticParameterPartials(fields, SSTCoefficients{});
            const auto base = solver.diagnosticResidual(fields, SSTCoefficients{});
            if (base.transport != solver.assembleResidual(fields, SSTCoefficients{}) ||
                base.continuity != solver.continuityResidual(fields, SSTCoefficients{}))
                return 2;
            const auto wrapper = solver.assembleResidualSensitivity(fields, SSTCoefficients{});
            std::vector<std::vector<double>> analytic;
            for (int j = 0; j < 11; ++j) {
                if (wrapper[j] != columns[j].transport)
                    return 2;
                analytic.push_back(columns[j].equations());
                analytic.back().insert(analytic.back().end(), columns[j].faceFluxes.begin(),
                                       columns[j].faceFluxes.end());
            }
            auto packed = [](const SIMPLESolver::DiagnosticResidual &r) {
                auto values = r.equations();
                values.insert(values.end(), r.faceFluxes.begin(), r.faceFluxes.end());
                return values;
            };
            const auto r0 = packed(base);
            for (int j = 0; j < 11; ++j) {
                double worst = 0;
                double previousTaylor = -1;
                for (double relative : {2e-4, 1e-4, 5e-5}) {
                    const double h = relative * theta[j];
                    auto plus = theta, minus = theta;
                    plus[j] += h;
                    minus[j] -= h;
                    const auto rp = packed(
                        solver.diagnosticResidual(fields, InferenceParameterSet::fromVector(plus)));
                    const auto rm = packed(solver.diagnosticResidual(
                        fields, InferenceParameterSet::fromVector(minus)));
                    double taylor = 0;
                    for (size_t i = 0; i < rp.size(); ++i) {
                        const double fd = (rp[i] - rm[i]) / (2 * h);
                        const double tolerance =
                            1e-8 + 1e-5 * std::max(std::abs(fd), std::abs(analytic[j][i]));
                        taylor =
                            checkedMaximum(taylor, std::abs(rp[i] - r0[i] - h * analytic[j][i]));
                        if (j == 10 && (analytic[j][i] != 0 || rp[i] != rm[i]))
                            return 3;
                        worst = checkedMaximum(worst, std::abs(fd - analytic[j][i]) / tolerance);
                    }
                    // Quadratic one-sided Taylor remainder until roundoff dominates.
                    if (previousTaylor > 1e-12 && taylor > 0.4 * previousTaylor) {
                        std::cerr << "FAIL Taylor refinement column=" << j << '\n';
                        ++failures;
                    }
                    previousTaylor = taylor;
                }
                if (worst > 1) {
                    std::cerr << "FAIL diagnostic partial: periodic=" << periodic
                              << " scheme=" << scheme << " column=" << j
                              << " normalized error=" << worst << '\n';
                    ++failures;
                }
            }
            auto unsupported = settings;
            unsupported.useWallFunctions = true;
            SIMPLESolver gated(mesh, SSTModel{}, bcs, nu, unsupported);
            bool rejected = false;
            try {
                gated.diagnosticParameterPartials(fields, SSTCoefficients{});
            } catch (const std::logic_error &) {
                rejected = true;
            }
            if (!rejected)
                return 4;
        }
    return failures ? 1 : 0;
}
