#include "BoundaryCondition.hpp"
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char **argv) {
    if (argc != 3)
        return 2;
    const std::string field = argv[1], shape = argv[2];
    Mesh mesh = Mesh::makeChannel2D(3, 4, 1.0, 1.0);
    mesh.computeWallDistance();
    auto bc = FlowBoundaryConditions::channelDefaults(mesh, 1.0, 1e-4, 1.0);
    const int inlet = mesh.patchByName("inlet");
    const size_t count = mesh.patch(inlet).faces.size();
    auto &boundaries = field == "velocity" ? bc.velocityBC : field == "k" ? bc.kBC : bc.omegaBC;
    const size_t size = shape == "short" ? 1 : shape == "long" ? count + 1 : count;
    if (field == "velocity")
        boundaries[inlet].vecProfile.assign(size, {2.0, 0.0, 0.0});
    else
        boundaries[inlet].scalarProfile.assign(size, 2.0);
    if (shape == "patches")
        boundaries.resize(1);
    VectorField velocity(mesh, "U", {0.0, 0.0, 0.0});
    ScalarField scalar(mesh, "scalar", 0.0);
    bool rejected = false;
    try {
        if (field == "velocity")
            applyVelocityBC(velocity, mesh, bc);
        else if (field == "k")
            applyKBC(scalar, mesh, bc);
        else
            applyOmegaBC(scalar, mesh, bc, 0.01);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    if (shape != "valid") {
        if (!rejected) {
            std::cerr << field << ' ' << shape << " was not rejected\n";
            return 1;
        }
        for (int face = 0; face < scalar.nBnd(); ++face)
            if (scalar.bnd(face) != 0.0 || velocity.bnd(face).norm2() != 0.0)
                return 3;
    } else {
        if (rejected)
            return 4;
        for (int face : mesh.patch(inlet).faces)
            if ((field == "velocity" ? velocity.bface(face).x : scalar.bface(face)) != 2.0)
                return 5;
    }
    std::cout << field << ' ' << shape << " passed\n";
}
