#include "LinearSolver.hpp"
#include "MatrixFreeSolver.hpp"
#include <iostream>
#include <limits>
#include <functional>

static LinearSystem matrix(int n, bool asymmetric) {
    LinearSystem A(n, std::max(0, n - 1));
    for (int i = 0; i < n; ++i) {
        A.diag[i] = 5 + (i % 3);
        A.source[i] = .3 + .11 * i;
    }
    for (int f = 0; f < A.nIF; ++f) {
        A.own[f] = f;
        A.nbr[f] = f + 1;
        A.upper[f] = -1;
        A.lower[f] = asymmetric ? -.7 : -1;
    }
    return A;
}

template <class Solver> int check(bool asymmetric) {
    Solver reused;
    size_t retained = 0;
    for (int n : {8, 32, 7, 64, 4, 0, 11})
        for (int mode = 0; mode < 5; ++mode) {
            auto A = matrix(n, asymmetric);
            std::vector<double> x(n, .17), freshX = x;
            int cap = mode == 1 ? 0 : 200;
            if (mode == 2) {
                std::fill(x.begin(), x.end(), 1);
                freshX = x;
                A.matvec(x, A.source); // exact caller initial solution
            }
            if (mode == 3) { // breakdown followed by a valid changed system
                std::fill(A.diag.begin(), A.diag.end(), 0);
                std::fill(A.upper.begin(), A.upper.end(), 0);
                std::fill(A.lower.begin(), A.lower.end(), 0);
            }
            if (mode == 4)
                for (double &a : A.diag)
                    a += 2;
            Solver fresh;
            const auto a = reused.solve(A, x, cap, 1e-11);
            const auto b = fresh.solve(A, freshX, cap, 1e-11);
            if (x != freshX || a.iterations != b.iterations || a.converged != b.converged ||
                a.initialRes != b.initialRes || a.finalRes != b.finalRes)
                return 1;
            if (a.converged) {
                std::vector<double> r(n);
                A.residual(x, r);
                if (linalg::norm(r) > 1e-10 * std::max(1.0, linalg::norm(A.source)))
                    return 2;
            }
            if (reused.workspaceCapacityBytes() < retained)
                return 3;
            retained = reused.workspaceCapacityBytes();
        }
    if (!retained)
        return 4;
    std::cout << reused.name() << " retained bytes=" << retained << '\n';
    return 0;
}
static void admission() {
    for (const std::string name : {"PCG", "BiCGSTAB", "GaussSeidel", "AMG"}) {
        auto solver = makeSolver(name);
        const auto reject =
            [&](const std::function<void(LinearSystem &, std::vector<double> &)> &edit, int cap = 2,
                double tol = 1e-8) {
                auto A = matrix(2, false);
                std::vector<double> x(2, 0.);
                edit(A, x);
                const auto saved = x;
                bool rejected = false;
                try {
                    (void)solver->solve(A, x, cap, tol);
                } catch (const std::invalid_argument &) {
                    rejected = true;
                }
                const bool unchanged =
                    x.size() == saved.size() &&
                    std::equal(x.begin(), x.end(), saved.begin(), [](double a, double b) {
                        return a == b || (std::isnan(a) && std::isnan(b));
                    });
                if (!rejected || !unchanged)
                    throw std::runtime_error(name +
                                             " accepted invalid storage/control or changed x");
            };
        reject([](auto &, auto &x) { x.pop_back(); });
        reject([](auto &, auto &x) { x.push_back(0.); });
        reject([](auto &A, auto &) { A.diag.pop_back(); });
        reject([](auto &A, auto &) { A.source.pop_back(); });
        reject([](auto &A, auto &) { A.upper.clear(); });
        reject([](auto &A, auto &) { A.lower.clear(); });
        reject([](auto &A, auto &) { A.own.clear(); });
        reject([](auto &A, auto &) { A.nbr.clear(); });
        reject([](auto &A, auto &) { A.own[0] = -1; });
        reject([](auto &A, auto &) { A.nbr[0] = A.nCells; });
        reject([](auto &A, auto &) { A.nCells = -1; });
        reject([](auto &A, auto &) { A.nIF = -1; });
        reject([](auto &, auto &) {}, -1);
        for (double bad : {0., -1., std::numeric_limits<double>::quiet_NaN(),
                           std::numeric_limits<double>::infinity()})
            reject([](auto &, auto &) {}, 2, bad);
        for (double bad :
             {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {
            reject([bad](auto &, auto &x) { x[0] = bad; });
            reject([bad](auto &A, auto &) { A.diag[0] = bad; });
            reject([bad](auto &A, auto &) { A.source[0] = bad; });
            reject([bad](auto &A, auto &) { A.upper[0] = bad; });
            reject([bad](auto &A, auto &) { A.lower[0] = bad; });
        }
        LinearSystem A(2, 0);
        A.diag = {2., 4.};
        A.source = {6., -8.};
        std::vector<double> x(2, 0.);
        const auto result = solver->solve(A, x, 4, 1e-12);
        if (!result.converged || std::abs(x[0] - 3.) > 1e-12 || std::abs(x[1] + 2.) > 1e-12)
            throw std::runtime_error(name + " failed independent diagonal control after rejection");
    }
}

static void matrixFreeAdmission() {
    const std::vector<double> rhs{1., -2.};
    const auto reject = [&](std::vector<double> x, std::vector<double> inverse, int cap,
                            double tolerance) {
        int calls = 0;
        const auto identity = [&](const auto &in, auto &out) {
            ++calls;
            out = in;
        };
        const auto saved = x;
        bool rejected = false;
        try {
            (void)bicgstabMatrixFree(identity, rhs, x, inverse, cap, tolerance);
        } catch (const std::invalid_argument &) {
            rejected = true;
        }
        if (!rejected || x != saved || calls != 0)
            throw std::runtime_error(
                "matrix-free BiCGSTAB accepted malformed input or called the operator");
    };
    for (size_t size : {size_t{0}, size_t{1}, size_t{3}}) {
        reject(std::vector<double>(size, 0.), std::vector<double>(2, 1.), 4, 1e-10);
        reject(std::vector<double>(2, 0.), std::vector<double>(size, 1.), 4, 1e-10);
    }
    reject(std::vector<double>(2, 0.), std::vector<double>(2, 1.), -1, 1e-10);
    for (double bad : {0., -1., std::numeric_limits<double>::quiet_NaN(),
                       std::numeric_limits<double>::infinity()})
        reject(std::vector<double>(2, 0.), std::vector<double>(2, 1.), 4, bad);
    std::vector<double> empty;
    const auto identity = [](const auto &in, auto &out) { out = in; };
    const auto result = bicgstabMatrixFree(identity, empty, empty, empty, 0, 1e-10);
    if (!result.converged || result.finalRes != 0. || result.initialRes != 0.)
        throw std::runtime_error("empty matrix-free system must remain an exact solution");
}

static void breakdownOutcomes() {
    const auto zero = [](const auto &x, auto &y) { y.assign(x.size(), 0.); };
    const std::vector<double> rhs{1., -2.}, inverse(2, 1.);
    for (bool gmres : {true, false})
        for (int cap : {0, 4}) {
            std::vector<double> x(2, 0.);
            const auto result = gmres ? gmresMatrixFree(zero, rhs, x, 2, cap, 1e-12)
                                      : bicgstabMatrixFree(zero, rhs, x, inverse, cap, 1e-12);
            if (result.converged || std::abs(result.finalRes - 1.) > 1e-14 ||
                std::abs(result.initialRes - std::sqrt(5.)) > 1e-14 || result.iterations > cap)
                throw std::runtime_error("legacy Krylov breakdown/cap reported a false residual");
        }
}

static void residualSemantics() {
    int calls = 0;
    const auto identity = [&](const auto &x, auto &y) {
        ++calls;
        y = x;
    };
    const std::vector<double> rhs{1., -2.}, inverse(2, 1.);
    for (bool gmres : {true, false}) {
        calls = 0;
        std::vector<double> x{3., -4.};
        const auto capped = gmres ? gmresMatrixFree(identity, rhs, x, 2, 0, 1e-12)
                                  : bicgstabMatrixFree(identity, rhs, x, inverse, 0, 1e-12);
        if (capped.converged || capped.iterations != 0 || capped.matvecs != calls ||
            std::abs(capped.finalRes - std::sqrt(8. / 5.)) > 1e-14 ||
            std::abs(capped.initialRes - std::sqrt(8.)) > 1e-14 ||
            x != std::vector<double>({3., -4.}))
            throw std::runtime_error("legacy Krylov cap changed the guess or residual semantics");
        for (bool zeroRhs : {false, true}) {
            calls = 0;
            const auto b = zeroRhs ? std::vector<double>(2, 0.) : rhs;
            x = zeroRhs ? std::vector<double>{.5, -.5} : rhs;
            const auto solved = gmres ? gmresMatrixFree(identity, b, x, 2, 4, 1e-12)
                                      : bicgstabMatrixFree(identity, b, x, inverse, 4, 1e-12);
            if (!solved.converged || solved.finalRes > 1e-14 || std::abs(x[0] - b[0]) > 1e-14 ||
                std::abs(x[1] - b[1]) > 1e-14 || solved.matvecs != calls)
                throw std::runtime_error("legacy Krylov exact/zero RHS control failed");
        }
    }
}

static void publicHelpers() {
    CSRMatrix csr;
    csr.nRows = 3;
    csr.rowPtr = {0, 2, 5, 7};
    csr.colIdx = {0, 1, 0, 1, 2, 1, 2};
    csr.values = {4., -1., -2., 5., -1., -1., 3.};
    csr.diag = {4., 5., 3.};
    const std::vector<double> exact{1., 2., -1.}, rhs{2., 9., -5.};
    std::vector<double> product(3), residual(3);
    csr.matvec(exact, product);
    csr.residual(exact, rhs, residual);
    if (product != rhs || residual != std::vector<double>(3, 0.))
        throw std::runtime_error("CSR products disagree with the hand-computed tridiagonal system");
    const auto matvec = [&](const auto &x, auto &y) { csr.matvec(x, y); };
    const std::vector<double> inverse{.25, .2, 1. / 3.};
    for (bool warm : {false, true})
        for (bool gmres : {false, true}) {
            std::vector<double> x(3, warm ? .5 : 0.);
            const auto result = gmres ? gmresMatrixFree(matvec, rhs, x, 3, 20, 1e-12)
                                      : bicgstabMatrixFree(matvec, rhs, x, inverse, 20, 1e-12);
            csr.residual(x, rhs, residual);
            if (!result.converged || result.iterations <= 0 || result.matvecs <= 0 ||
                linalg::norm(residual) > 1e-10 || std::abs(x[0] - exact[0]) > 1e-10 ||
                std::abs(x[1] - exact[1]) > 1e-10 || std::abs(x[2] - exact[2]) > 1e-10)
                throw std::runtime_error(
                    "legacy Krylov solve failed independent true-residual control");
        }
}

int main() {
    matrixFreeAdmission();
    breakdownOutcomes();
    residualSemantics();
    publicHelpers();
    admission();
    const int pcg = check<PCGSolver>(false);
    const int bicg = check<BiCGSTABSolver>(true);
    return pcg ? pcg : bicg;
}
