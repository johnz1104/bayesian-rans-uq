#include "ParameterSensitivity.hpp"
#include "SIMPLEValidation.hpp"
#include "cfdr/BuildInfo.hpp"
#include <cmath>
#include <algorithm>
#include <vector>
#include <limits>
#include <stdexcept>

// Fixed-state partials of the 11 SST coefficients and experimental tangents.
// State actions use finite differences; no analytic state adjoint is supplied.

ParameterSensitivity::ParameterSensitivity(const Mesh &mesh, const ObservationOperator &obs,
                                           const FlowBoundaryConditions &bcs, double nu,
                                           const SolverSettings &settings, const Vec3 &Uinit,
                                           double pInit, double kInit, double omegaInit)
    : mesh_(mesh), obs_(obs), bcs_(bcs), nu_(nu), settings_(settings), Uinit_(Uinit), pInit_(pInit),
      kInit_(kInit), omegaInit_(omegaInit), state_(mesh) {
    simple_admission::problem(mesh_, bcs_, nu_, settings_);
    simple_admission::initial(Uinit_, pInit_, kInit_, omegaInit_);
    (void)obs_.prepare(mesh_);
}

SSTModel ParameterSensitivity::makeModel(const std::vector<double> &theta11) const {
    if (theta11.size() != 11)
        throw std::invalid_argument("Expected 11 finite positive SST coefficients");
    for (double value : theta11)
        if (!std::isfinite(value) || value <= 0.0)
            throw std::invalid_argument("Expected 11 finite positive SST coefficients");
    SSTModel sst(InferenceParameterSet::fromVector(theta11));
    sst.variant = static_cast<SSTVariant>(settings_.sstVariant);
    return sst;
}

EvaluationStatus ParameterSensitivity::solveState(const std::vector<double> &theta11) {
    mesh_.requireStamp(meshStamp_);
    hasState_ = false;
    stateTheta_.clear();
    stateFluxes_.clear();
    stateHistory_ = {};
    if (!InferenceParameterSet::all11().inBounds(theta11))
        return EvaluationStatus::InvalidParameters;
    SSTModel sst = makeModel(theta11);
    SIMPLESolver solver(mesh_, sst, bcs_, nu_, settings_);
    FlowFields f(mesh_);
    solver.initUniform(f, Uinit_, pInit_, kInit_, omegaInit_);
    ConvergenceHistory hist = solver.solve(f);
    if (hist.diverged)
        return EvaluationStatus::Diverged;
    if (hist.converged) {
        state_ = std::move(f);
        stateTheta_ = theta11;
        stateFluxes_ = solver.faceFluxes();
        stateHistory_ = hist;
        hasState_ = true;
        return EvaluationStatus::Converged;
    }
    return EvaluationStatus::Unconverged;
}

void ParameterSensitivity::requireState() const {
    mesh_.requireStamp(meshStamp_);
    if (!hasState_)
        throw std::logic_error("Sensitivity requires a converged base solveState");
}

void ParameterSensitivity::requireMatchingState(const std::vector<double> &theta11) const {
    requireState();
    if (theta11 != stateTheta_)
        throw std::invalid_argument(
            "Total sensitivity requires the coefficients of the converged base state");
}

void ParameterSensitivity::requireExperimentalTangent() const {
    if (!allowExperimentalTangents_)
        throw std::logic_error(
            "Analytic residual/tangent sensitivity is experimental: corrected Rhie-Chow "
            "derivatives are unverified; use warm FD or explicitly opt in");
}

std::vector<double> ParameterSensitivity::residual(const std::vector<double> &theta11) {
    requireState();
    SSTModel sst = makeModel(theta11);
    SIMPLESolver solver(mesh_, sst, bcs_, nu_, settings_);
    return solver.assembleResidual(state_, sst.coeffs);
}

std::vector<std::vector<double>>
ParameterSensitivity::dResidualDTheta(const std::vector<double> &theta11) {
    requireState();
    if (!settings_.legacyRhieChowGate)
        requireExperimentalTangent();
    SSTModel sst = makeModel(theta11);
    SIMPLESolver solver(mesh_, sst, bcs_, nu_, settings_);
    return solver.assembleResidualSensitivity(state_, sst.coeffs);
}

void ParameterSensitivity::recompute(const std::vector<double> &theta11, FlowFields &work,
                                     std::vector<SSTClosureSensitivity> *cs) const {
    requireState();
    SSTModel sst = makeModel(theta11);
    work = state_;
    sst.computeFields(mesh_, work.k, work.omega, work.U, nu_, work.nuT, work.F1, work.F2, work.Pk,
                      work.CDkw);
    // startup-only nuT floor has released at the converged states this path
    // evaluates, so the mirror applies only the non-negativity clamp and the
    // sensitivity carries no floor dead-zone
    for (int ci = 0; ci < mesh_.nCells(); ++ci)
        work.nuT[ci] = std::max(work.nuT[ci], 0.0);
    if (cs) {
        ScalarField Smag = strainRateMagnitude(computeVelocityGradients(work.U));
        const auto &wd = mesh_.wallDistance();
        cs->resize(mesh_.nCells());
        for (int ci = 0; ci < mesh_.nCells(); ++ci)
            (*cs)[ci] = sst.closureSensitivity(work.k[ci], work.omega[ci], Smag[ci], wd[ci], nu_,
                                               work.CDkw[ci], 0.0);
    }
}

std::vector<double> ParameterSensitivity::observe(const std::vector<double> &theta11) {
    FlowFields work(mesh_);
    recompute(theta11, work, nullptr);
    return obs_.evaluate(mesh_, work, nu_);
}

double ParameterSensitivity::logLik(const std::vector<double> &theta11) {
    FlowFields work(mesh_);
    recompute(theta11, work, nullptr);
    return obs_.logLikelihood(mesh_, work, nu_);
}

// Every current observable reads U/p and molecular viscosity only. At fixed
// primary state no SST coefficient enters g: its exact partial derivative is zero.
// Total solution derivatives are a separate calculation (warm FD / experimental tangent).
std::vector<std::vector<double>>
ParameterSensitivity::dObsDTheta(const std::vector<double> &theta11) {
    requireState();
    (void)makeModel(theta11); // validate the same coefficient contract as observe()
    return std::vector<std::vector<double>>(obs_.nObs(), std::vector<double>(11, 0.0));
}

// Helpers for experimental tangent approximations.

std::vector<int> ParameterSensitivity::cellColoring(int &nColors) const {
    const int nc = mesh_.nCells();
    const int nIF = mesh_.nInternalFaces();
    std::vector<std::vector<int>> adj(nc);
    for (int fi = 0; fi < nIF; ++fi) {
        const Face &face = mesh_.face(fi);
        adj[face.owner].push_back(face.neighbor);
        adj[face.neighbor].push_back(face.owner);
    }
    std::vector<int> color(nc, -1);
    nColors = 0;
    std::vector<char> used;
    for (int i = 0; i < nc; ++i) {
        used.assign(nColors, 0);
        for (int j : adj[i])
            if (color[j] >= 0)
                used[color[j]] = 1;
        int c = 0;
        while (c < (int)used.size() && used[c])
            ++c;
        color[i] = c;
        if (c + 1 > nColors)
            nColors = c + 1;
    }
    return color;
}

// Experimental frozen-pressure, Picard-reduced tangent. Central-difference
// state actions and analytic coefficient partials do not by themselves establish
// consistency with the fully coupled forward operator.
TangentGradientResult ParameterSensitivity::etaJacobianTangent(const std::vector<double> &theta11,
                                                               double krylovTol, int maxIter,
                                                               double fdStep) {
    requireMatchingState(theta11);
    requireExperimentalTangent();
    if (!std::isfinite(fdStep) || fdStep <= 0 || !std::isfinite(krylovTol) || krylovTol <= 0 ||
        maxIter < 0)
        throw std::invalid_argument("Invalid tangent step/tolerance/iteration cap");
    const int nc = mesh_.nCells();
    const int N = 4 * nc;
    const int nObsv = obs_.nObs();
    const int BUX = 0, BUY = nc, BK = 2 * nc, BOM = 3 * nc;

    TangentGradientResult out;
    out.dObsDTheta.assign(nObsv, std::vector<double>(11, 0.0));
    out.krylovIters.assign(11, 0);
    out.krylovRelRes.assign(11, 0.0);
    out.krylovConverged.assign(11, 0);
    if (!hasState_ || nc == 0)
        return out;

    SSTModel sst = makeModel(theta11);
    SIMPLESolver solver(mesh_, sst, bcs_, nu_, settings_);

    // RHS_j = −∂R/∂θ_j of the PICARD-REDUCED model: the transpose-stress
    // deferred correction is lagged in both U (operator switch
    // below) and theta (includeTransposeTheta = false), exactly like the
    // frozen pressure, so the reduced tangent differentiates a
    // self-consistent model. The full transpose coupling belongs to the
    // coupled tangent and warm-FD, where pressure removes the rigid-rotation
    // near-null mode the completed stress operator otherwise carries; the
    // residual bias of this reduction is characterised against warm-FD by the
    // direction tests, as the frozen-pressure bias always was.
    std::vector<std::vector<double>> dRdTheta =
        solver.assembleResidualSensitivity(state_, sst.coeffs,
                                           /*includeTransposeTheta=*/false);
    solver.setFreezeTransposeStress(true);

    // ---- per-DOF column scaling σ_i = max(|x_i|, floor_block) ---------------------------
    // The tangent unknown [Ux|Uy|k|ω] spans orders of magnitude (ω∼1e6 near walls, ∼1 in
    // the freestream).  Per-DOF |x_i| scaling makes every FD probe a uniform RELATIVE
    // perturbation, so the column-scaled operator Ã = J·diag(σ) is well posed.
    double rmsUx = 0.0;
    for (int ci = 0; ci < nc; ++ci)
        rmsUx += state_.U[ci].x * state_.U[ci].x;
    rmsUx = std::sqrt(rmsUx / std::max(nc, 1));
    const double floorU = std::max(1e-4 * rmsUx, 1e-12);
    const double floorK = std::max(settings_.kMin, 1e-12);
    const double floorW = std::max(settings_.omegaMin, 1e-12);
    std::vector<double> sigma(N);
    for (int ci = 0; ci < nc; ++ci) {
        sigma[BUX + ci] = std::max(std::abs(state_.U[ci].x), floorU);
        sigma[BUY + ci] = std::max(std::abs(state_.U[ci].y), floorU);
        sigma[BK + ci] = std::max(std::abs(state_.k[ci]), floorK);
        sigma[BOM + ci] = std::max(std::abs(state_.omega[ci]), floorW);
    }

    // residual at a state delta (assembleResidual copies the state and recomputes closure,
    // so the perturbation propagates through nuT/F1/F2/Pk/CDkw and Smag — full coupling).
    auto residualAt = [&](const std::vector<double> &d) -> std::vector<double> {
        FlowFields pert = state_;
        for (int ci = 0; ci < nc; ++ci) {
            pert.U[ci].x += d[BUX + ci];
            pert.U[ci].y += d[BUY + ci];
            pert.k[ci] += d[BK + ci];
            pert.omega[ci] += d[BOM + ci];
        }
        return solver.assembleResidual(pert, sst.coeffs);
    };

    // observable vector at a state delta (recompute closure exactly as observe()).
    auto observeAtDelta = [&](const std::vector<double> &d) -> std::vector<double> {
        FlowFields work = state_;
        for (int ci = 0; ci < nc; ++ci) {
            work.U[ci].x += d[BUX + ci];
            work.U[ci].y += d[BUY + ci];
            work.k[ci] += d[BK + ci];
            work.omega[ci] += d[BOM + ci];
        }
        sst.computeFields(mesh_, work.k, work.omega, work.U, nu_, work.nuT, work.F1, work.F2,
                          work.Pk, work.CDkw);
        for (int ci = 0; ci < nc; ++ci)
            work.nuT[ci] = std::max(work.nuT[ci], 0.0);
        return obs_.evaluate(mesh_, work, nu_);
    };

    // ---- colored-FD diagonal of Ã = J·diag(σ)  →  Jacobi (left) preconditioner ----------
    int nColors = 0;
    std::vector<int> color = cellColoring(nColors);
    out.nColors = nColors;
    std::vector<double> diagA(N, 0.0);
    std::vector<double> probe(N, 0.0);
    const double inv2s = 1.0 / (2.0 * fdStep);
    for (int f = 0; f < 4; ++f) {
        const int B = f * nc;
        for (int c = 0; c < nColors; ++c) {
            std::fill(probe.begin(), probe.end(), 0.0);
            for (int ci = 0; ci < nc; ++ci)
                if (color[ci] == c)
                    probe[B + ci] = fdStep * sigma[B + ci];
            std::vector<double> Rp = residualAt(probe);
            for (int ci = 0; ci < nc; ++ci)
                if (color[ci] == c)
                    probe[B + ci] = -fdStep * sigma[B + ci];
            std::vector<double> Rm = residualAt(probe);
            out.nResidualEvals += 2;
            for (int ci = 0; ci < nc; ++ci)
                if (color[ci] == c)
                    diagA[B + ci] = (Rp[B + ci] - Rm[B + ci]) * inv2s;
        }
    }
    std::vector<double> invDiag(N);
    for (int i = 0; i < N; ++i)
        invDiag[i] = (std::abs(diagA[i]) > 1e-300) ? 1.0 / diagA[i] : 1.0;

    // ---- left-preconditioned scaled operator  B ṽ = invDiag ∘ (J·(σ∘ṽ)) ----------------
    // Baking invDiag into the operator (not as a separate right-preconditioner) makes the
    // BiCGSTAB stopping norm the *balanced* residual: diag(B)=1, so the ω-block (residual
    // O(1e6)) cannot dominate ‖r‖ and starve the velocity tangent the QoIs depend on.
    std::vector<double> dplus(N), dminus(N);
    auto matvecB = [&](const std::vector<double> &vt, std::vector<double> &yv) {
        double vn = linalg::norm(vt);
        if (vn < 1e-300) {
            std::fill(yv.begin(), yv.end(), 0.0);
            return;
        }
        double eps = fdStep / vn;
        for (int i = 0; i < N; ++i) {
            double s = eps * sigma[i] * vt[i];
            dplus[i] = s;
            dminus[i] = -s;
        }
        std::vector<double> Rp = residualAt(dplus);
        std::vector<double> Rm = residualAt(dminus);
        out.nResidualEvals += 2;
        double inv2e = 1.0 / (2.0 * eps);
        for (int i = 0; i < N; ++i)
            yv[i] = invDiag[i] * (Rp[i] - Rm[i]) * inv2e;
    };

    std::vector<double> ones(N, 1.0), wt(N), w(N);
    for (int j = 0; j < 11; ++j) {
        double bn = 0.0;
        for (int i = 0; i < N; ++i)
            bn += dRdTheta[j][i] * dRdTheta[j][i];
        if (std::sqrt(bn) < 1e-300) {
            out.krylovConverged[j] = 1;
            continue;
        } // e.g. κ (∂R/∂κ≡0)

        std::vector<double> b(N);
        for (int i = 0; i < N; ++i)
            b[i] = -invDiag[i] * dRdTheta[j][i]; // invDiag ∘ (−∂R/∂θ_j)

        MatrixFreeResult kr = bicgstabMatrixFree(matvecB, b, wt, ones, maxIter, krylovTol);
        out.krylovIters[j] = kr.iterations;
        out.krylovRelRes[j] = kr.finalRes;
        out.krylovConverged[j] = kr.converged ? 1 : 0;
        for (int i = 0; i < N; ++i)
            w[i] = sigma[i] * wt[i]; // unscale: w = dU/dθ_j

        // directional observable derivative ∂η_i/∂θ_j = [g_i(x+h w) − g_i(x−h w)]/(2h),
        // h chosen so the largest RELATIVE state perturbation equals fdStep.
        double scale = 0.0;
        for (int i = 0; i < N; ++i)
            scale = std::max(scale, std::abs(w[i]) / sigma[i]);
        if (scale < 1e-300)
            continue; // w ≈ 0 → zero gradient column
        double hg = fdStep / scale;
        for (int i = 0; i < N; ++i) {
            dplus[i] = hg * w[i];
            dminus[i] = -hg * w[i];
        }
        std::vector<double> gP = observeAtDelta(dplus);
        std::vector<double> gM = observeAtDelta(dminus);
        double inv2h = 1.0 / (2.0 * hg);
        for (int i = 0; i < nObsv; ++i)
            out.dObsDTheta[i][j] = (gP[i] - gM[i]) * inv2h;
    }
    return out;
}

// Experimental pressure-coupled tangent: continuity and pressure-Poisson blocks.

std::vector<double> ParameterSensitivity::massFluxDivergence(const std::vector<double> &ux,
                                                             const std::vector<double> &uy,
                                                             bool homogeneous) const {
    const int nc = mesh_.nCells();
    const int nIF = mesh_.nInternalFaces();
    std::vector<double> div(nc, 0.0);
    for (int fi = 0; fi < nIF; ++fi) {
        const Face &face = mesh_.face(fi);
        const int o = face.owner, n = face.neighbor;
        const double w = face.weight;
        double Ufx = w * ux[o] + (1.0 - w) * ux[n];
        double Ufy = w * uy[o] + (1.0 - w) * uy[n];
        double mf = (Ufx * face.normal.x + Ufy * face.normal.y) * face.area;
        div[o] += mf;
        div[n] -= mf;
    }
    for (int pi = 0; pi < mesh_.nPatches(); ++pi) {
        const Patch &pat = mesh_.patch(pi);
        for (FaceID fi : pat.faces) {
            const Face &face = mesh_.face(fi);
            const int o = face.owner;
            double Ubx, Uby;
            if (pat.type == "outlet") { // zero-gradient: responds to interior (both modes)
                Ubx = ux[o];
                Uby = uy[o];
            } else if (pat.type == "wall") { // no-slip: zero flux (both modes)
                Ubx = 0.0;
                Uby = 0.0;
            } else { // inlet/other: fixed value (residual) or 0 (increment)
                if (homogeneous) {
                    Ubx = 0.0;
                    Uby = 0.0;
                } else {
                    Vec3 ub = state_.U.bface(fi);
                    Ubx = ub.x;
                    Uby = ub.y;
                }
            }
            div[o] += (Ubx * face.normal.x + Uby * face.normal.y) * face.area;
        }
    }
    return div;
}

LinearSystem ParameterSensitivity::assemblePressurePoisson(const std::vector<double> &aP) const {
    LinearSystem sys = makeSystem(mesh_);
    const int nIF = mesh_.nInternalFaces();
    for (int fi = 0; fi < nIF; ++fi) {
        const Face &face = mesh_.face(fi);
        const int o = face.owner, n = face.neighbor;
        double Sf = face.area, delta = std::max(face.delta, 1e-20);
        double Vo = mesh_.cell(o).volume, Vn = mesh_.cell(n).volume;
        double dPo = (std::abs(aP[o]) > 1e-30) ? Vo / aP[o] : 0.0;
        double dPn = (std::abs(aP[n]) > 1e-30) ? Vn / aP[n] : 0.0;
        double dPf = face.weight * dPo + (1.0 - face.weight) * dPn;
        double coeff = dPf * Sf / delta;
        sys.diag[o] += coeff;
        sys.diag[n] += coeff;
        sys.upper[fi] = -coeff;
        sys.lower[fi] = -coeff;
    }
    for (int pi = 0; pi < mesh_.nPatches(); ++pi) { // outlet Dirichlet p'=0
        const Patch &pat = mesh_.patch(pi);
        if (pat.type != "outlet")
            continue;
        for (FaceID fi : pat.faces) {
            const Face &face = mesh_.face(fi);
            const int o = face.owner;
            double Sf = face.area, delta = std::max(face.delta, 1e-20);
            double dPo = (std::abs(aP[o]) > 1e-30) ? mesh_.cell(o).volume / aP[o] : 0.0;
            sys.diag[o] += dPo * Sf / delta;
        }
    }
    return sys;
}

// Experimental [Ux|Uy|k|omega|p] tangent. Wall-omega rows and row/column
// scaling condition the linear problem, but parameter dependence of the
// corrected Rhie-Chow continuity RHS remains incomplete. An explicit FGMRES
// residual certifies only this linear solve, not a total forward derivative.
TangentGradientResult
ParameterSensitivity::etaJacobianTangentCoupled(const std::vector<double> &theta11,
                                                double krylovTol, int maxIter, double fdStep) {
    requireMatchingState(theta11);
    requireExperimentalTangent();
    if (!std::isfinite(fdStep) || fdStep <= 0 || !std::isfinite(krylovTol) || krylovTol <= 0 ||
        maxIter < 0)
        throw std::invalid_argument("Invalid tangent step/tolerance/iteration cap");
    const int nc = mesh_.nCells();
    const int N4 = 4 * nc, N5 = 5 * nc;
    const int nObsv = obs_.nObs();
    const int BUX = 0, BUY = nc, BK = 2 * nc, BOM = 3 * nc, BP = 4 * nc;

    TangentGradientResult out;
    out.dObsDTheta.assign(nObsv, std::vector<double>(11, 0.0));
    out.krylovIters.assign(11, 0);
    out.krylovRelRes.assign(11, 0.0);
    out.krylovConverged.assign(11, 0);
    if (!hasState_ || nc == 0)
        return out;

    SSTModel sst = makeModel(theta11);
    SIMPLESolver solver(mesh_, sst, bcs_, nu_, settings_);
    std::vector<std::vector<double>> dRdTheta =
        solver.assembleResidualSensitivity(state_, sst.coeffs); // 11 × N4

    // The converged wall cell satisfies its prescribed interior omega value,
    // not the free PDE row returned by assembleResidual. Use the same
    // geometry-qualified value as the primal. The experimental tangent uses
    // R_wall = omega - omega_prescribed and its beta1 derivative; wall-function
    // blending is outside this historical tangent's supported domain.
    std::vector<int> pinCell;
    std::vector<double> pinBC;
    const double beta1 = sst.coeffs.beta1;
    if (!settings_.useWallFunctions) {
        for (int pi = 0; pi < mesh_.nPatches(); ++pi) {
            const Patch &pat = mesh_.patch(pi);
            if (pat.type != "wall")
                continue;
            for (FaceID fi : pat.faces) {
                int o = mesh_.face(fi).owner;
                double y1 = std::max(mesh_.face(fi).delta, 1e-20);
                double omBC = solver.resolvedWallOmega(y1);
                pinCell.push_back(o);
                pinBC.push_back(omBC);
                for (int j = 0; j < 11; ++j)
                    dRdTheta[j][BOM + o] = 0.0;
                dRdTheta[2][BOM + o] = omBC / beta1; // ∂R_ω,wall/∂β1 = −∂ω_BC/∂β1 = ω_BC/β1
            }
        }
    }

    // ---- per-DOF column scaling σ (5 blocks; pressure scaled by its RMS) ---------------
    double rmsUx = 0.0, rmsP = 0.0;
    for (int ci = 0; ci < nc; ++ci) {
        rmsUx += state_.U[ci].x * state_.U[ci].x;
        rmsP += state_.p[ci] * state_.p[ci];
    }
    rmsUx = std::sqrt(rmsUx / std::max(nc, 1));
    rmsP = std::sqrt(rmsP / std::max(nc, 1));
    const double floorU = std::max(1e-4 * rmsUx, 1e-12);
    const double floorK = std::max(settings_.kMin, 1e-12);
    const double floorW = std::max(settings_.omegaMin, 1e-12);
    const double floorP = std::max(1e-4 * std::max(rmsP, 0.5 * rmsUx * rmsUx), 1e-12);
    std::vector<double> sigma(N5);
    for (int ci = 0; ci < nc; ++ci) {
        sigma[BUX + ci] = std::max(std::abs(state_.U[ci].x), floorU);
        sigma[BUY + ci] = std::max(std::abs(state_.U[ci].y), floorU);
        sigma[BK + ci] = std::max(std::abs(state_.k[ci]), floorK);
        sigma[BOM + ci] = std::max(std::abs(state_.omega[ci]), floorW);
        sigma[BP + ci] = std::max(std::abs(state_.p[ci]), floorP);
    }
    // Row scaling ρ: the wall-ω Dirichlet rows carry an O(1e7) RHS (∂ω_BC/∂β1) that would
    // dominate the FGMRES residual norm and starve the velocity tangent the QoIs feel.  Scale
    // those rows by 1/σ_ω so they sit at O(1) — purely a residual-norm rebalance (the solution
    // ṽ is unchanged), and consistent with the identity-pinned ω preconditioner block.
    std::vector<double> rho(N5, 1.0);
    for (int o : pinCell)
        rho[BOM + o] = 1.0 / sigma[BOM + o];

    auto augResidualAt = [&](const std::vector<double> &d) -> std::vector<double> {
        FlowFields pert = state_;
        for (int ci = 0; ci < nc; ++ci) {
            pert.U[ci].x += d[BUX + ci];
            pert.U[ci].y += d[BUY + ci];
            pert.k[ci] += d[BK + ci];
            pert.omega[ci] += d[BOM + ci];
            pert.p[ci] += d[BP + ci];
        }
        // Re-apply the velocity/pressure BCs to the perturbed interior so the boundary faces
        // RESPOND (outlet velocity extrapolates ⇒ continuity can close; wall/outlet pressure
        // Neumann/Dirichlet ⇒ greenGaussGrad(p) is right near walls — the momentum↔pressure
        // coupling the QoIs feel).  assembleResidual otherwise freezes the boundary values.
        applyVelocityBC(pert.U, mesh_, bcs_);
        applyPressureBC(pert.p, mesh_, bcs_);
        std::vector<double> R = solver.assembleResidual(pert, sst.coeffs);
        R.resize(N5);
        // the continuity row with the solver's own face fluxes (steady
        // Rhie-Chow flux), so the tangent differentiates the operator the
        // converged state solves
        std::vector<double> div = solver.continuityResidual(pert, sst.coeffs);
        for (int ci = 0; ci < nc; ++ci)
            R[BP + ci] = div[ci];
        for (size_t p = 0; p < pinCell.size(); ++p) // wall-ω Dirichlet residual
            R[BOM + pinCell[p]] = pert.omega[pinCell[p]] - pinBC[p];
        return R;
    };
    auto observeAtDelta = [&](const std::vector<double> &d) -> std::vector<double> {
        FlowFields work = state_;
        for (int ci = 0; ci < nc; ++ci) {
            work.U[ci].x += d[BUX + ci];
            work.U[ci].y += d[BUY + ci];
            work.k[ci] += d[BK + ci];
            work.omega[ci] += d[BOM + ci];
            work.p[ci] += d[BP + ci];
        }
        sst.computeFields(mesh_, work.k, work.omega, work.U, nu_, work.nuT, work.F1, work.F2,
                          work.Pk, work.CDkw);
        for (int ci = 0; ci < nc; ++ci)
            work.nuT[ci] = std::max(work.nuT[ci], 0.0);
        return obs_.evaluate(mesh_, work, nu_);
    };

    // ---- physics-based block-SIMPLE preconditioner blocks (assembled ONCE) --------------
    LinearSystem Amom, Ak, Aom;
    std::vector<double> aPp;
    solver.assemblePreconBlocks(state_, sst.coeffs, Amom, Ak, Aom, aPp);
    // pin the wall-ω rows of the ω preconditioner block to identity (consistent with the
    // Dirichlet residual): diag = −1, row off-diagonals zeroed ⇒ −A_om⁻¹ acts as identity there.
    if (!pinCell.empty()) {
        std::vector<char> pinned(nc, 0);
        for (int o : pinCell) {
            pinned[o] = 1;
            Aom.diag[o] = -1.0;
        }
        for (int f = 0; f < Aom.nIF; ++f) {
            if (pinned[Aom.own[f]])
                Aom.upper[f] = 0.0;
            if (pinned[Aom.nbr[f]])
                Aom.lower[f] = 0.0;
        }
    }
    LinearSystem Lp = assemblePressurePoisson(aPp);
    BiCGSTABSolver blockSolver;
    AMGSolver pSolver;
    auto setPrimeBC = [&](ScalarField &pf) {
        for (int pi = 0; pi < mesh_.nPatches(); ++pi) {
            const Patch &pat = mesh_.patch(pi);
            if (pat.type == "outlet")
                continue;
            for (FaceID fi : pat.faces)
                pf.bface(fi) = pf[mesh_.face(fi).owner];
        }
    };

    // M_phys⁻¹ r → z (physical 5-block).  z_block = −A_block⁻¹ r_block (since A_block ≈
    // −∂R/∂block), then the SIMPLE pressure Schur on the velocity predictor.  Loose block solves
    // (tol 0.1).
    auto Mphys = [&](const std::vector<double> &r) -> std::vector<double> {
        std::vector<double> z(N5, 0.0);
        std::vector<double> rb(nc), zb(nc, 0.0);
        for (int ci = 0; ci < nc; ++ci)
            rb[ci] = r[BOM + ci];
        Aom.source = rb;
        std::fill(zb.begin(), zb.end(), 0.0);
        blockSolver.solve(Aom, zb, 400, 1e-4);
        for (int ci = 0; ci < nc; ++ci)
            z[BOM + ci] = -zb[ci];
        for (int ci = 0; ci < nc; ++ci)
            rb[ci] = r[BK + ci];
        Ak.source = rb;
        std::fill(zb.begin(), zb.end(), 0.0);
        blockSolver.solve(Ak, zb, 400, 1e-4);
        for (int ci = 0; ci < nc; ++ci)
            z[BK + ci] = -zb[ci];

        std::vector<double> zuhx(nc, 0.0), zuhy(nc, 0.0), rux(nc), ruy(nc);
        for (int ci = 0; ci < nc; ++ci) {
            rux[ci] = r[BUX + ci];
            ruy[ci] = r[BUY + ci];
        }
        Amom.source = rux;
        blockSolver.solve(Amom, zuhx, 400, 1e-4);
        Amom.source = ruy;
        blockSolver.solve(Amom, zuhy, 400, 1e-4);
        for (int ci = 0; ci < nc; ++ci) {
            zuhx[ci] = -zuhx[ci];
            zuhy[ci] = -zuhy[ci];
        }

        std::vector<double> div = massFluxDivergence(zuhx, zuhy, /*homogeneous=*/true);
        std::vector<double> prhs(nc);
        for (int ci = 0; ci < nc; ++ci)
            prhs[ci] = -div[ci] + r[BP + ci];
        Lp.source = prhs;
        std::vector<double> zp(nc, 0.0);
        pSolver.solve(Lp, zp, 500, 1e-8);
        ScalarField zpF(mesh_, "zp");
        for (int ci = 0; ci < nc; ++ci)
            zpF[ci] = zp[ci];
        setPrimeBC(zpF);
        VectorField gradZp = greenGaussGrad(zpF);
        for (int ci = 0; ci < nc; ++ci) {
            double rap = mesh_.cell(ci).volume / (std::abs(aPp[ci]) > 1e-30 ? aPp[ci] : 1.0);
            z[BUX + ci] = zuhx[ci] - rap * gradZp[ci].x;
            z[BUY + ci] = zuhy[ci] - rap * gradZp[ci].y;
            z[BP + ci] = zp[ci];
        }
        return z;
    };
    // preconditioner for the row-scaled, column-scaled operator B = ρ∘Ã:  M⁻¹ ≈ B⁻¹ = Ã⁻¹∘ρ⁻¹,
    // so un-row-scale (÷ρ), apply the physical block-SIMPLE M_phys⁻¹, then un-column-scale (÷σ).
    auto precond = [&](const std::vector<double> &r, std::vector<double> &z) {
        std::vector<double> ru(N5);
        for (int i = 0; i < N5; ++i)
            ru[i] = r[i] / rho[i];
        std::vector<double> zp = Mphys(ru);
        for (int i = 0; i < N5; ++i)
            z[i] = zp[i] / sigma[i];
    };

    // Central-difference sigma-scaled augmented state action (approximate Jv).
    std::vector<double> dplus(N5), dminus(N5);
    auto matvec = [&](const std::vector<double> &vt, std::vector<double> &yv) {
        double vn = linalg::norm(vt);
        if (vn < 1e-300) {
            std::fill(yv.begin(), yv.end(), 0.0);
            return;
        }
        double eps = fdStep / vn;
        for (int i = 0; i < N5; ++i) {
            double s = eps * sigma[i] * vt[i];
            dplus[i] = s;
            dminus[i] = -s;
        }
        std::vector<double> Rp = augResidualAt(dplus);
        std::vector<double> Rm = augResidualAt(dminus);
        out.nResidualEvals += 2;
        double inv2e = 1.0 / (2.0 * eps);
        for (int i = 0; i < N5; ++i)
            yv[i] = rho[i] * (Rp[i] - Rm[i]) * inv2e;
    };

    std::vector<double> vt(N5), w(N5);
    FGMRESWorkspace krylovWorkspace;
    for (int j = 0; j < 11; ++j) {
        double bn = 0.0;
        for (int i = 0; i < N4; ++i)
            bn += dRdTheta[j][i] * dRdTheta[j][i];
        if (std::sqrt(bn) < 1e-300) {
            out.krylovConverged[j] = 1;
            continue;
        } // κ
        std::vector<double> b(N5, 0.0);
        for (int i = 0; i < N4; ++i)
            b[i] = -dRdTheta[j][i]; // continuity RHS = 0
        for (int i = 0; i < N5; ++i)
            b[i] *= rho[i]; // row-balance wall-ω

        std::fill(vt.begin(), vt.end(), 0.0);
        MatrixFreeResult kr = fgmresMatrixFree(matvec, precond, b, vt, coupledKrylovRestart_,
                                               maxIter, krylovTol, &krylovWorkspace);
        out.krylovIters[j] = kr.iterations;
        out.krylovRelRes[j] = kr.finalRes;
        out.krylovConverged[j] = kr.converged ? 1 : 0;
        for (int i = 0; i < N5; ++i)
            w[i] = sigma[i] * vt[i];

        // directional-derivative step from the U/k/ω blocks ONLY: the pressure tangent dp can be
        // large in magnitude but the QoIs do not read p, so including it would shrink hg until the
        // velocity perturbation fell below FD roundoff (collapsing the gradient to ~0).
        double scale = 0.0;
        for (int i = 0; i < N4; ++i)
            scale = std::max(scale, std::abs(w[i]) / sigma[i]);
        if (scale < 1e-300)
            continue;
        double hg = fdStep / scale;
        for (int i = 0; i < N5; ++i) {
            dplus[i] = hg * w[i];
            dminus[i] = -hg * w[i];
        }
        std::vector<double> gP = observeAtDelta(dplus);
        std::vector<double> gM = observeAtDelta(dminus);
        double inv2h = 1.0 / (2.0 * hg);
        for (int i = 0; i < nObsv; ++i)
            out.dObsDTheta[i][j] = (gP[i] - gM[i]) * inv2h;
    }
    return out;
}

// Total forward response by central differences of re-converged warm solves.
TangentGradientResult
ParameterSensitivity::etaJacobianWarmFD(const std::vector<double> &theta11, double hRel,
                                        double hFloor, int warmMaxIter, double warmTol,
                                        const std::vector<double> &absoluteSteps) {
    requireMatchingState(theta11);
    if (!std::isfinite(hRel) || hRel <= 0 || !std::isfinite(hFloor) || hFloor <= 0 ||
        warmMaxIter < 0 || !std::isfinite(warmTol) || warmTol < 0)
        throw std::invalid_argument("Invalid warm finite-difference step/tolerance/iteration cap");
    if (!absoluteSteps.empty() &&
        (absoluteSteps.size() != 11 ||
         !std::all_of(absoluteSteps.begin(), absoluteSteps.end(),
                      [](double h) { return std::isfinite(h) && h > 0; })))
        throw std::invalid_argument(
            "finite-difference steps require eleven positive finite entries");
    const int nObsv = obs_.nObs();
    TangentGradientResult out;
    out.dObsDTheta.assign(nObsv, std::vector<double>(11, std::numeric_limits<double>::quiet_NaN()));
    out.logLikGradient.assign(11, std::numeric_limits<double>::quiet_NaN());
    out.krylovIters.assign(11, 0);
    out.krylovRelRes.assign(11, 0.0);
    out.krylovConverged.assign(11, 0);
    out.finiteDifferenceSteps.resize(11);
    out.perturbedSolveTolerances.resize(11);
    out.positiveStatus.assign(11, EvaluationStatus::Unknown);
    out.negativeStatus.assign(11, EvaluationStatus::Unknown);

    // Perturbations start from the common base; their convergence is checked separately.
    // Without an explicit warmTol the re-solve tolerance is tied to the step:
    // the convergence criterion bounds the remaining RELATIVE change of the
    // fields, and a central difference of an O(hRel) response needs that
    // error well below hRel, not merely below the base solve's tolerance.
    SolverSettings warm = settings_;
    if (warmMaxIter > 0)
        warm.maxIterations = warmMaxIter;

    // β* (8) and a1 (9) are the only coefficients nonlinear in the closure — smaller step.
    auto stepFor = [&](int j, double tj) {
        if (!absoluteSteps.empty())
            return absoluteSteps[j];
        double hr = (j == 8 || j == 9) ? std::min(hRel, 1e-5) : hRel;
        return std::max(hr * std::abs(tj), hFloor);
    };
    // Re-solve SIMPLE at theta_pert WARM-STARTED from the fixed converged state, then observe
    // (closure recomputed exactly as observe()) and also score the Gaussian log-likelihood —
    // both η and logL from the SAME warm solve, so ∂η/∂θ and ∂logL/∂θ cost one warm-FD pass.
    auto etaWarm = [&](const std::vector<double> &tp, bool &ok, double &logL, int &iterations,
                       double &worstMeasure, EvaluationStatus &status) -> std::vector<double> {
        if (!std::all_of(tp.begin(), tp.end(),
                         [](double x) { return std::isfinite(x) && x > 0; })) {
            status = EvaluationStatus::InvalidParameters;
            ok = false;
            logL = std::numeric_limits<double>::quiet_NaN();
            worstMeasure = std::numeric_limits<double>::infinity();
            return {};
        }
        SSTModel sstp = makeModel(tp);
        SIMPLESolver solverp(mesh_, sstp, bcs_, nu_, warm);
        FlowFields f = state_; // WARM start
        applyVelocityBC(f.U, mesh_, bcs_);
        applyPressureBC(f.p, mesh_, bcs_);
        applyKBC(f.k, mesh_, bcs_);
        applyOmegaBC(f.omega, mesh_, bcs_, nu_, sstp.coeffs.beta1);
        ConvergenceHistory hist = solverp.solve(f);
        iterations = hist.finalIter;
        worstMeasure = std::isfinite(hist.finalWorstMeasure)
                           ? hist.finalWorstMeasure
                           : std::numeric_limits<double>::infinity();
        out.nResidualEvals += hist.finalIter; // total SIMPLE iterations (cost proxy)
        // a perturbed state that never converged would difference a transient
        status = hist.diverged    ? EvaluationStatus::Diverged
                 : hist.converged ? EvaluationStatus::Converged
                                  : EvaluationStatus::Unconverged;
        ok = status == EvaluationStatus::Converged;
        if (!ok) {
            logL = std::numeric_limits<double>::quiet_NaN();
            return {};
        }
        sstp.computeFields(mesh_, f.k, f.omega, f.U, nu_, f.nuT, f.F1, f.F2, f.Pk, f.CDkw);
        for (int ci = 0; ci < mesh_.nCells(); ++ci)
            f.nuT[ci] = std::max(f.nuT[ci], 0.0);
        auto predictions = obs_.evaluate(mesh_, f, nu_);
        logL = obs_.logLikelihood(predictions);
        return predictions;
    };

    for (int j = 0; j < 11; ++j) {
        double hj = stepFor(j, theta11[j]);
        std::vector<double> tp = theta11, tm = theta11;
        tp[j] += hj;
        tm[j] -= hj;
        hj = 0.5 * (tp[j] - tm[j]);
        if (!(hj > 0) || !std::isfinite(hj))
            throw std::invalid_argument("finite-difference step is not representable");
        out.finiteDifferenceSteps[j] = hj;
        const double actualRelative = hj / std::max(std::abs(theta11[j]), hFloor);
        warm.convergenceTol =
            warmTol > 0 ? warmTol : std::min(settings_.convergenceTol, 0.01 * actualRelative);
        out.perturbedSolveTolerances[j] = warm.convergenceTol;
        bool okp = true, okm = true;
        double lp = 0.0, lm = 0.0;
        int ip = 0, im = 0;
        double rp = 0.0, rm = 0.0;
        std::vector<double> ep = etaWarm(tp, okp, lp, ip, rp, out.positiveStatus[j]);
        std::vector<double> em = etaWarm(tm, okm, lm, im, rm, out.negativeStatus[j]);
        out.krylovIters[j] = ip + im;
        out.krylovRelRes[j] = std::max(rp, rm);
        out.krylovConverged[j] = (okp && okm) ? 1 : 0;
        if (!(okp && okm))
            continue; // NaN column plus failure flag, never a plausible zero
        for (int i = 0; i < nObsv; ++i)
            out.dObsDTheta[i][j] = (ep[i] - em[i]) / (2.0 * hj);
        out.logLikGradient[j] = (lp - lm) / (2.0 * hj); // ∂logL/∂θ_j (Gaussian)
    }
    return out;
}

SteadySensitivityBase ParameterSensitivity::steadyBase() const {
    requireState();
    return {state_,    stateFluxes_, stateHistory_, stateTheta_,
            settings_, bcs_,         nu_,           cfdr::sourceDigest};
}

std::vector<double> ParameterSensitivity::steadyStateVector() const {
    requireState();
    SIMPLESolver solver(mesh_, makeModel(stateTheta_), bcs_, nu_, settings_);
    const auto faces = solver.freeFluxFaces();
    const int nc = mesh_.nCells();
    std::vector<double> q(5 * nc + faces.size());
    double pMean = 0, volume = 0;
    bool gauge = true;
    for (const auto &bc : bcs_.pressureBC)
        if (bc.type == BCType::OutletPressure || bc.type == BCType::Dirichlet)
            gauge = false;
    if (gauge)
        for (int i = 0; i < nc; ++i) {
            pMean += state_.p[i] * mesh_.cell(i).volume;
            volume += mesh_.cell(i).volume;
        }
    if (gauge)
        pMean /= volume;
    for (int i = 0; i < nc; ++i) {
        q[i] = state_.U[i].x;
        q[nc + i] = state_.U[i].y;
        q[2 * nc + i] = state_.k[i];
        q[3 * nc + i] = state_.omega[i];
        q[4 * nc + i] = state_.p[i] - pMean;
    }
    for (size_t i = 0; i < faces.size(); ++i)
        q[5 * nc + i] = stateFluxes_[faces[i]];
    return q;
}

FlowFields ParameterSensitivity::mappedSteadyFields(const std::vector<double> &q,
                                                    const std::vector<double> &theta) const {
    requireState();
    const auto sst = makeModel(theta);
    SIMPLESolver solver(mesh_, sst, bcs_, nu_, settings_);
    const int nc = mesh_.nCells();
    if (q.size() != 5 * static_cast<size_t>(nc) + solver.freeFluxFaces().size() ||
        !std::all_of(q.begin(), q.end(), [](double x) { return std::isfinite(x); }))
        throw std::invalid_argument("augmented state has invalid size or nonfinite values");
    auto f = state_;
    for (int i = 0; i < nc; ++i) {
        f.U[i] = {q[i], q[nc + i], 0};
        f.k[i] = q[2 * nc + i];
        f.omega[i] = q[3 * nc + i];
        f.p[i] = q[4 * nc + i];
    }
    applyVelocityBC(f.U, mesh_, bcs_);
    applyPressureBC(f.p, mesh_, bcs_);
    applyKBC(f.k, mesh_, bcs_);
    applyOmegaBC(f.omega, mesh_, bcs_, nu_, sst.coeffs.beta1);
    return f;
}

SIMPLESolver::SteadyResidual
ParameterSensitivity::steadyResidualAt(const std::vector<double> &q,
                                       const std::vector<double> &theta) const {
    auto f = mappedSteadyFields(q, theta);
    SIMPLESolver solver(mesh_, makeModel(theta), bcs_, nu_, settings_);
    return solver.steadyResidual(f, InferenceParameterSet::fromVector(theta),
                                 {q.begin() + 5 * mesh_.nCells(), q.end()});
}

std::vector<int> ParameterSensitivity::steadyBranches(const std::vector<double> &q,
                                                      const std::vector<double> &theta) const {
    auto f = mappedSteadyFields(q, theta);
    const auto sst = makeModel(theta);
    ScalarField strain(mesh_, "branch_strain");
    sst.computeFields(mesh_, f.k, f.omega, f.U, nu_, f.nuT, f.F1, f.F2, f.Pk, f.CDkw, strain);
    std::vector<int> branches;
    for (int i = 0; i < mesh_.nCells(); ++i) {
        const double k = std::max(f.k[i], 0.0), w = std::max(f.omega[i], 1e-20);
        const double y = std::max(mesh_.wallDistance()[i], 1e-20), S = strain[i];
        const double t1 = std::sqrt(k) / (theta[8] * w * y), t2 = 500 * nu_ / (y * y * w);
        const double t3 = 4 * theta[5] * k / (std::max(f.CDkw[i], 1e-20) * y * y);
        const double raw = theta[9] * k / std::max(std::max(theta[9] * w, S * f.F2[i]), 1e-20);
        for (bool b :
             {f.k[i] > 0, f.omega[i] > 1e-20, f.CDkw[i] > 1e-20, t1 >= t2, std::max(t1, t2) <= t3,
              2 * t1 >= t2, theta[9] * w >= S * f.F2[i],
              std::max(theta[9] * w, S * f.F2[i]) >= 1e-20, raw >= 0,
              raw * S * S <= 10 * theta[8] * k * w,
              S * S <= 10 * theta[8] * k * w / std::max(raw, 1e-30), f.k[i] > settings_.kMin,
              f.k[i]<settings_.kMax, f.omega[i]> settings_.omegaMin, f.omega[i] < 1e15})
            branches.push_back(b);
    }
    return branches;
}

void ParameterSensitivity::requireSteadyPair(const std::vector<double> &plus,
                                             const std::vector<double> &minus,
                                             const std::vector<double> &thetaPlus,
                                             const std::vector<double> &thetaMinus) const {
    const auto base = steadyStateVector();
    if (plus.size() != base.size() || minus.size() != base.size())
        throw std::invalid_argument("steady perturbation shape mismatch");
    for (size_t i = 5 * mesh_.nCells(); i < base.size(); ++i) {
        const double change = std::max(std::abs(plus[i] - base[i]), std::abs(minus[i] - base[i]));
        const double roundoff =
            32 * std::numeric_limits<double>::epsilon() * std::max(1.0, std::abs(base[i]));
        if (std::abs(base[i]) <= std::max(roundoff, 2 * change))
            throw std::domain_error("free-face flux lacks the required ordinary-derivative margin");
    }
    const auto branches = steadyBranches(base, stateTheta_);
    if (steadyBranches(plus, thetaPlus) != branches ||
        steadyBranches(minus, thetaMinus) != branches)
        throw std::domain_error(
            "SST or clamp branch changed across the ordinary two-sided perturbation");
}

std::vector<double> ParameterSensitivity::steadyStateAction(const std::vector<double> &direction,
                                                            double step) const {
    const auto base = steadyStateVector();
    if (direction.size() != base.size() || !std::isfinite(step) || step <= 0 ||
        !std::all_of(direction.begin(), direction.end(), [](double x) { return std::isfinite(x); }))
        throw std::invalid_argument(
            "steady state action requires a finite matching direction and positive step");
    const double norm = linalg::norm(direction);
    std::vector<double> result(base.size());
    if (norm == 0)
        return result;
    const double h = step / norm;
    auto plus = base, minus = base;
    for (size_t i = 0; i < base.size(); ++i) {
        plus[i] += h * direction[i];
        minus[i] -= h * direction[i];
    }
    requireSteadyPair(plus, minus, stateTheta_, stateTheta_);
    const auto rp = steadyResidualAt(plus, stateTheta_).equations;
    const auto rm = steadyResidualAt(minus, stateTheta_).equations;
    for (size_t i = 0; i < result.size(); ++i)
        result[i] = (rp[i] - rm[i]) / (2 * h);
    return result;
}

size_t ParameterSensitivity::steadyKrylovBasisBytes(int restart) const {
    return FGMRESWorkspace::basisBytes(steadyStateVector().size(), restart);
}

SteadyTangentResult
ParameterSensitivity::etaJacobianSteady(const SteadySensitivityOptions &options) {
    requireState();
    if (options.restart < 1 || options.maxIterations < 0 || !std::isfinite(options.actionStep) ||
        options.actionStep <= 0 || !std::isfinite(options.parameterRelativeStep) ||
        options.parameterRelativeStep <= 0 || !std::isfinite(options.krylovTolerance) ||
        options.krylovTolerance <= 0 || options.krylovTolerance > 1e-8)
        throw std::invalid_argument(
            "invalid steady-sensitivity steps, limits or residual tolerance");
    for (const auto &scales : {options.stateScales, options.equationScales})
        for (double scale : scales)
            if (!std::isfinite(scale) || scale <= 0)
                throw std::invalid_argument("steady scales must be fixed positive finite values");
    const auto base = steadyStateVector();
    const auto residual = steadyResidualAt(base, stateTheta_);
    if (residual.pressureGauge)
        for (const auto &observable : obs_.observables())
            if (observable.type == ObsType::PressureTap)
                throw std::logic_error(
                    "periodic steady sensitivities initially require gauge-invariant observations");
    const size_t n = base.size();
    const int nc = mesh_.nCells();
    SteadyTangentResult result;
    result.krylovBasisBytes = steadyKrylovBasisBytes(options.restart);
    result.removedContinuity = residual.removedContinuity;
    result.branchQualified.assign(11, 0);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    auto &out = result.derivatives;
    out.dObsDTheta.assign(obs_.nObs(), std::vector<double>(11, nan));
    out.logLikGradient.assign(11, nan);
    out.krylovConverged.assign(11, 0);
    out.krylovIters.assign(11, 0);
    out.krylovRelRes.assign(11, std::numeric_limits<double>::infinity());
    result.stateTangents.assign(11, std::vector<double>(n, nan));
    std::vector<double> columnScale(n), rowScale(n);
    for (size_t i = 0; i < n; ++i) {
        const size_t block = std::min(i / static_cast<size_t>(nc), size_t{5});
        columnScale[i] = options.stateScales[block];
        rowScale[i] = 1 / options.equationScales[block];
        const double scaled = std::abs(residual.equations[i] * rowScale[i]);
        result.normalizedBaseResidual = std::isfinite(scaled)
                                            ? std::max(result.normalizedBaseResidual, scaled)
                                            : std::numeric_limits<double>::infinity();
    }
    if (!std::isfinite(result.normalizedBaseResidual) || result.normalizedBaseResidual > 1e-8) {
        result.qualification = "steady base residual exceeds 1e-8; tangent remains experimental";
        return result;
    }
    try {
        requireSteadyPair(base, base, stateTheta_, stateTheta_);
    } catch (const std::domain_error &error) {
        result.qualification = error.what();
        return result;
    }
    FGMRESWorkspace workspace;
    auto matvec = [&](const std::vector<double> &v, std::vector<double> &y) {
        const double norm = linalg::norm(v);
        if (norm == 0) {
            std::fill(y.begin(), y.end(), 0);
            return;
        }
        const double h = options.actionStep / norm;
        auto plus = base, minus = base;
        for (size_t i = 0; i < n; ++i) {
            plus[i] += h * columnScale[i] * v[i];
            minus[i] -= h * columnScale[i] * v[i];
        }
        requireSteadyPair(plus, minus, stateTheta_, stateTheta_);
        const auto rp = steadyResidualAt(plus, stateTheta_).equations;
        const auto rm = steadyResidualAt(minus, stateTheta_).equations;
        out.nResidualEvals += 2;
        for (size_t i = 0; i < n; ++i)
            y[i] = rowScale[i] * (rp[i] - rm[i]) / (2 * h);
    };
    const auto identity = [](const std::vector<double> &r, std::vector<double> &z) { z = r; };
    for (int j = 0; j < 11; ++j) {
        try {
            auto tp = stateTheta_, tm = stateTheta_;
            const double h = options.parameterRelativeStep * stateTheta_[j];
            tp[j] += h;
            tm[j] -= h;
            requireSteadyPair(base, base, tp, tm);
            const auto rp = steadyResidualAt(base, tp).equations;
            const auto rm = steadyResidualAt(base, tm).equations;
            out.nResidualEvals += 2;
            std::vector<double> rhs(n), tangent(n, 0);
            for (size_t i = 0; i < n; ++i)
                rhs[i] = -rowScale[i] * (rp[i] - rm[i]) / (2 * h);
            const auto kr =
                fgmresMatrixFree(matvec, identity, rhs, tangent, options.restart,
                                 options.maxIterations, options.krylovTolerance, &workspace);
            out.krylovIters[j] = kr.iterations;
            out.krylovRelRes[j] = kr.finalRes;
            if (!kr.converged || kr.finalRes > 1e-8)
                continue;
            for (size_t i = 0; i < n; ++i)
                tangent[i] *= columnScale[i];
            auto plus = base, minus = base;
            for (size_t i = 0; i < n; ++i) {
                plus[i] += h * tangent[i];
                minus[i] -= h * tangent[i];
            }
            requireSteadyPair(plus, minus, tp, tm);
            const auto gp = obs_.evaluate(mesh_, mappedSteadyFields(plus, tp), nu_);
            const auto gm = obs_.evaluate(mesh_, mappedSteadyFields(minus, tm), nu_);
            for (int i = 0; i < obs_.nObs(); ++i)
                out.dObsDTheta[i][j] = (gp[i] - gm[i]) / (2 * h);
            out.logLikGradient[j] = (obs_.logLikelihood(gp) - obs_.logLikelihood(gm)) / (2 * h);
            out.krylovConverged[j] = result.branchQualified[j] = 1;
            result.stateTangents[j] = std::move(tangent);
        } catch (const std::domain_error &error) {
            result.qualification = error.what();
        }
    }
    if (result.qualification.empty())
        result.qualification =
            "experimental: external dense-Jacobian, full-FD and tighter-solve refinement required";
    return result;
}
