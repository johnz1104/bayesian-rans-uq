#include "LinearSolver.hpp"
#include <iostream>
#include <stdexcept>
#include <string>

int main() {
    for (const std::string name : {"PCG", "BiCGSTAB", "GaussSeidel", "AMG"}) {
        if (makeSolver(name)->name() != name)
            return 1;
    }
    for (const std::string name : {"", "not-a-solver", "bicgstab"}) {
        bool rejected = false;
        try {
            (void)makeSolver(name);
        } catch (const std::invalid_argument &) {
            rejected = true;
        }
        if (!rejected) {
            std::cerr << "unknown solver silently accepted: '" << name << "'\n";
            return 2;
        }
    }
    std::cout << "four explicit solver selections accepted; unknown names rejected\n";
}
