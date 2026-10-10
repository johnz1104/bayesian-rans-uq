#include "ExactRiemann.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

static void require(bool ok, const char *message) {
    if (!ok)
        throw std::runtime_error(message);
}
template <class F> static void rejects(F f, const char *message) {
    bool rejected = false;
    try {
        f();
    } catch (const std::exception &) {
        rejected = true;
    }
    require(rejected, message);
}
int main(int argc, char **argv) {
    try {
        require(argc == 2, "one contract required");
        const std::string name = argv[1];
        const dbns::RiemannState1D valid{1, 0, 1};
        double p = 0, u = 0;
        const dbns::ExactRiemann solver;
        if (name == "invalid") {
            for (double v : {0., -1., std::numeric_limits<double>::quiet_NaN(),
                             std::numeric_limits<double>::infinity()}) {
                rejects([&] { solver.starState({v, 0, 1}, valid, p, u); },
                        "invalid density accepted");
                rejects([&] { solver.starState(valid, {1, 0, v}, p, u); },
                        "invalid pressure accepted");
                rejects([&] { dbns::ExactRiemann(v).starState(valid, valid, p, u); },
                        "invalid gamma accepted");
            }
            rejects([&] { dbns::ExactRiemann(1).starState(valid, valid, p, u); },
                    "gamma one accepted");
            rejects([&] { solver.sample(valid, valid, std::numeric_limits<double>::quiet_NaN()); },
                    "nonfinite sampling speed accepted");
        } else if (name == "vacuum") {
            // At p*=0 the two pressure functions sum to -2(aL+aR)/(gamma-1).
            const double critical = 4 * std::sqrt(1.4) / .4;
            for (double gap : {critical, critical + 1})
                rejects([&] { solver.starState({1, -gap / 2, 1}, {1, gap / 2, 1}, p, u); },
                        "vacuum solution silently replaced by a positive pressure floor");
        } else if (name == "unresolved") {
            // Analytic symmetric two-rarefaction pressure is 1e-16. The
            // inherited Newton pressure floor is 1e-12, so returning that
            // value as an exact root must be rejected by a residual check.
            const double v = 2 * std::sqrt(1.4) / .4 * (1 - std::pow(1e-16, .4 / 2.8));
            rejects([&] { solver.starState({1, -v, 1}, {1, v, 1}, p, u); },
                    "stagnation at a pressure floor was accepted as an exact root");
        } else
            require(false, "unknown contract");
        std::cout << name << " reference contract passed\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
