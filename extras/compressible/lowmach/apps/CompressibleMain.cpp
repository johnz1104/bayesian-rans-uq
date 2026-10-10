#include "Mesh.hpp"
#include "SSTModel.hpp"
#include "ObservationOperator.hpp"
#include "InferenceParameters.hpp"
#include "IdealGasEOS.hpp"
#include "CompressibleBCs.hpp"
#include "CompressibleSIMPLESolver.hpp"
#include "CompressibleForwardModel.hpp"
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <string>

// Compressible channel flow validation:
//   Air at Re = 10,000 (based on half-height H/2),  Ma_in = 0.3
//   Uniform pressure p_ref, inlet temperature T_in = 300 K
//   Compares wall Cf and Nusselt number at the outlet section.
int main(int argc, char **argv) {
    // Geometry
    int nx = 40, ny = 30;
    int maxIterations = 3000;
    double divergenceLimit = 1e10;
    try {
        for (int i = 1; i < argc; ++i) {
            const std::string option = argv[i];
            if (option != "--nx" && option != "--ny" && option != "--max-iterations" &&
                option != "--divergence-limit")
                throw std::invalid_argument("unknown option: " + option);
            if (++i == argc)
                throw std::invalid_argument("missing value for " + option);
            const std::string value = argv[i];
            std::size_t consumed = 0;
            if (option == "--divergence-limit") {
                divergenceLimit = std::stod(value, &consumed);
                if (!std::isfinite(divergenceLimit) || divergenceLimit < 0)
                    throw std::invalid_argument("divergence limit must be finite and nonnegative");
            } else {
                const int count = std::stoi(value, &consumed);
                if (count < (option == "--max-iterations" ? 0 : 1))
                    throw std::invalid_argument("invalid count for " + option);
                if (option == "--nx")
                    nx = count;
                else if (option == "--ny")
                    ny = count;
                else
                    maxIterations = count;
            }
            if (consumed != value.size())
                throw std::invalid_argument("invalid value for " + option + ": " + value);
        }
    } catch (const std::exception &error) {
        std::cerr << error.what()
                  << "\nUsage: rans_sst_compressible [--nx N] [--ny N] "
                     "[--max-iterations N] [--divergence-limit X]\n";
        return 2;
    }
    double Lx = 10.0, H = 1.0; // channel: length x half-height

    // Thermodynamics
    IdealGasEOS eos;
    double T_in = 300.0;                      // inlet temperature [K]
    double p_ref = 101325.0;                  // reference pressure [Pa]
    double rho_in = eos.density(p_ref, T_in); // ~1.177 kg/m³
    double mu_in = eos.viscosity(T_in);       // ~1.716e-5 Pa·s
    double a_in = eos.soundSpeed(T_in);       // ~347 m/s
    double Ma_in = 0.1;                       // Ma=0.1 for validation stability
    double Uin = Ma_in * a_in;                // ~34.7 m/s
    double nu_in = mu_in / rho_in;            // kinematic viscosity
    double Re = rho_in * Uin * H / mu_in;     // ~2,370,000 (fully turbulent)

    std::cout << "Compressible channel flow validation\n";
    std::cout << "  Geometry: " << nx << " x " << ny << "  (Lx=" << Lx << ", H=" << H << ")\n";
    std::cout << "  T_in=" << T_in << " K  p_ref=" << p_ref << " Pa\n";
    std::cout << "  U_in=" << Uin << " m/s  Ma=" << Ma_in << "  Re=" << Re << "\n";

    // Mesh (uniform channel, wall-clustered if Re known)
    Mesh mesh = Mesh::makeChannel2D(nx, ny, Lx, H, Re, 1.0);
    mesh.computeWallDistance();
    std::cout << "  Mesh: " << mesh.nCells() << " cells, " << mesh.nFaces() << " faces\n";

    // Turbulence inlet conditions
    double Tu = 0.05;
    double kIn = 1.5 * (Uin * Tu) * (Uin * Tu);
    double omIn = kIn / (nu_in * 100.0);

    // Boundary conditions — use absolute outlet pressure (p_ref) so that interior
    // cells initialised at p_ref don't see a spurious 101325 Pa pressure gradient.
    CompressibleBoundaryConditions bcs =
        CompressibleBoundaryConditions::channelDefaults(mesh, Uin, T_in, p_ref, kIn, omIn);

    // Solver settings (relaxed for channel)
    SolverSettings settings;
    settings.maxIterations = maxIterations;
    settings.convergenceTol = 1e-3;
    settings.divergenceLimit = divergenceLimit;
    settings.alphaU = 0.5; // conservative for compressible
    settings.alphaP = 0.2;
    settings.alphaT = 0.7; // temperature under-relaxation
    settings.alphaK = 0.4;
    settings.alphaOmega = 0.4;
    settings.innerIterations = 300;
    settings.innerTolerance = 1e-4;
    settings.turbStartIter = 50; // let mean flow establish first
    settings.turbUpdateInterval = 2;
    settings.verbose = false;
    settings.reportInterval = 300;

    // Solve at Menter defaults
    SSTModel sst;
    CompressibleSIMPLESolver solver(mesh, sst, bcs, eos, settings);
    CompressibleFlowFields fields(mesh);
    solver.initUniform(fields, Vec3(Uin, 0, 0), p_ref, T_in, kIn, omIn);

    settings.verbose = true;
    CompressibleConvergenceHistory hist = solver.solve(fields);

    std::cout << "\nResult:\n";
    std::cout << "  Status: "
              << (hist.converged  ? "Converged"
                  : hist.diverged ? "Diverged"
                                  : "MaxIter")
              << "  (" << hist.finalIter << " iterations)\n";
    if (!hist.entries.empty()) {
        const auto &last = hist.entries.back();
        std::cout << "  Final residuals: Ux=" << last.Ux << "  p=" << last.p << "  k=" << last.k
                  << "  om=" << last.omega << "  T=" << last.T << "\n";
    }

    // Compute Cf from streamwise pressure gradient: τ_w = H * |dp/dx|, Cf = 2τ_w/(ρ U²)
    double p_in = 0, p_out_avg = 0;
    int n_in = 0, n_out = 0;
    for (int ci = 0; ci < mesh.nCells(); ++ci) {
        double cx = mesh.cell(ci).center.x;
        if (cx < Lx / nx) {
            p_in += fields.p[ci];
            ++n_in;
        }
        if (cx > Lx - Lx / nx) {
            p_out_avg += fields.p[ci];
            ++n_out;
        }
    }
    if (n_in > 0 && n_out > 0) {
        p_in /= n_in;
        p_out_avg /= n_out;
        double dp_dx = (p_in - p_out_avg) / Lx;
        // Full channel of height H with two walls: 2τ_w = H * dp/dx
        double tau_w = dp_dx * H / 2.0;
        double rho_bulk = eos.density(p_ref, T_in);
        double Cf_computed = 2.0 * tau_w / (rho_bulk * Uin * Uin);
        // Dean correlation based on hydraulic diameter (2H for channel)
        double Re_D = Re * 2.0;
        double Cf_dean = 0.073 * std::pow(Re_D, -0.25);
        std::cout << "  Cf (pressure gradient): " << Cf_computed << "\n";
        std::cout << "  Cf (Dean correlation):  " << Cf_dean << "  (Re_D=" << Re_D << ")\n";
    }

    // Print flow statistics
    double U_max = 0;
    for (int ci = 0; ci < mesh.nCells(); ++ci) {
        double umag = std::sqrt(fields.U[ci].x * fields.U[ci].x + fields.U[ci].y * fields.U[ci].y);
        U_max = std::max(U_max, umag);
    }
    double T_cent = fields.T[0];
    for (int ci = 0; ci < mesh.nCells(); ++ci)
        T_cent = std::max(T_cent, fields.T[ci]);
    std::cout << "  Max velocity:    " << U_max << " m/s\n";
    std::cout << "  Max temperature: " << T_cent << " K\n";
    std::cout << "  Max Ma ~ " << U_max / eos.soundSpeed(T_cent) << "\n";

    return hist.converged ? 0 : 1;
}
