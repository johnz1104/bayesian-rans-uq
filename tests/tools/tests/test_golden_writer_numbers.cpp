#include "GoldenWriter.hpp"
#include <iostream>
#include <limits>

using namespace cfdr::golden;

int main() {
    int failures = 0;
    for (double value :
         {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(),
          -std::numeric_limits<double>::infinity()}) {
        for (int shape = 0; shape != 3; ++shape) {
            JObj object;
            object.integer("preserved", 7);
            bool rejected = false;
            try {
                if (shape == 0)
                    object.num("invalid", value);
                else if (shape == 1)
                    object.arr("invalid", {1, value});
                else
                    object.mat("invalid", {{1}, {value}});
            } catch (const std::invalid_argument &) {
                rejected = true;
            }
            if (!rejected || object.dump() != "{\n  \"preserved\": 7\n}") {
                std::cerr << "FAIL: nonfinite value accepted/replaced, shape=" << shape << '\n';
                ++failures;
            }
        }
    }
    JObj finite;
    finite.num("scalar", 1.25).arr("array", {0, -2}).mat("matrix", {{3}, {4}});
    const std::string expected = "{\n  \"scalar\": 1.25,\n  \"array\": [0, -2],\n"
                                 "  \"matrix\": [\n    [3],\n    [4]]\n}";
    if (finite.dump() != expected) {
        std::cerr << "FAIL: finite scalar/array/matrix representation changed\n";
        ++failures;
    }
    if (failures == 0)
        std::cout << "nine nonfinite writer controls rejected; finite representation unchanged\n";
    return failures == 0 ? 0 : 1;
}
