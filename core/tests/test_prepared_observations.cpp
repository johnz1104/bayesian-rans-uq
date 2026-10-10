#include "ObservationOperator.hpp"
#include <iostream>
#include <limits>

int main() {
    for (int kind = 0; kind < 4; ++kind) {
        Mesh mesh =
            kind == 0 ? Mesh::makeChannel2D(4, 5, 2, 1)
            : kind == 1
                ? Mesh::makeCurvedChannelPeriodic2D({0, .3, .8, 1.4, 2}, {0, .08, .04, -.04, 0}, 1,
                                                    5, 50)
                : Mesh::makeChannel2D(1, 1, 1, kind == 2 ? 1 : std::numeric_limits<double>::min());
        mesh.computeWallDistance();
        std::string wallName;
        for (const auto &patch : mesh.patches())
            if (patch.type == "wall") {
                wallName = patch.name;
                break;
            }
        ObservationOperator obs;
        for (const Vec3 point :
             {Vec3{.3, .37, 0}, Vec3{-.1, .5, 0}, Vec3{2.1, .9, 0}, Vec3{0, 0, 0}}) {
            obs.addVelocityProfile(point, 0, 0, 1);
            obs.addVelocityProfile(point, 1, 0, 1);
            obs.addPressureTap(point, 0, 1, 1);
        }
        obs.addSkinFriction(wallName, {.63, 0, 0}, 0, 1, 1);
        obs.addDrag(wallName, 0, 1, 1, 1);
        obs.addSeparationPoint(wallName, 0, 1);
        obs.addReattachmentLength(wallName, 0, 1);
        const auto prepared = obs.prepare(mesh);
        FlowFields f(mesh);
        for (int i = 0; i < mesh.nCells(); ++i) {
            const auto p = mesh.cell(i).center;
            f.U[i] = {1 + p.x + 2 * p.y, -.1 + .3 * p.y, 0};
            f.p[i] = .3 + p.x - .2 * p.y;
        }
        for (int fi = mesh.nInternalFaces(); fi < mesh.nFaces(); ++fi) {
            const auto p = mesh.face(fi).center;
            f.U.bface(fi) = {1 + p.x + 2 * p.y, -.1 + .3 * p.y, 0};
            f.p.bface(fi) = .3 + p.x - .2 * p.y;
        }
        if (kind >= 2) {
            const auto gradient = leastSquaresGradient2D(
                mesh, 0, [&](int cell) { return f.U[cell].x; },
                [&](int face) { return f.U.bface(face).x; });
            // The smallest normal positive height is admitted by the mesh
            // factory but places the weighted y moment below the stencil's
            // singularity threshold. Preserve the zero-gradient fallback
            // witness without constructing an invalid zero-height mesh.
            if (kind == 3 && gradient.norm2() != 0.)
                return 6;
            if (kind == 2 &&
                (std::abs(gradient.x - 1.) > 1e-12 || std::abs(gradient.y - 2.) > 1e-12))
                return 7;
        }
        for (int pass = 0; pass < 4; ++pass) {
            std::vector<double> out(obs.nObs(), -999);
            prepared.evaluate(f, .01, out);
            const auto direct = obs.evaluate(mesh, f, .01);
            if (kind == 3 && (out[0] != f.U[0].x || direct[0] != f.U[0].x))
                return 8; // Both paths must use the cell value when the fit is singular.
            if (out != direct || prepared.evaluate(f, .01) != direct) {
                for (size_t i = 0; i < out.size(); ++i)
                    if (out[i] != direct[i])
                        std::cerr << "prepared mismatch kind=" << kind << " pass=" << pass
                                  << " obs=" << i << " difference=" << out[i] - direct[i] << '\n';
                return 1;
            }
            // Field and boundary changes invalidate every value/gradient,
            // but no geometric search or preparation needs to run again.
            for (int i = 0; i < mesh.nCells(); ++i)
                f.U[i].x = mesh.cell(i).center.x - .4 - .2 * pass;
            for (int fi = mesh.nInternalFaces(); fi < mesh.nFaces(); ++fi)
                f.p.bface(fi) += .01;
        }
        bool rejected = false;
        try {
            prepared.evaluate(f, .01, f.p.data());
        } catch (const std::invalid_argument &) {
            rejected = true;
        }
        if (!rejected)
            return 2;
        const auto count = prepared.nObs();
        obs.addVelocityProfile({.2, .2, 0}, 0, 0, 1);
        if (prepared.nObs() != count || obs.prepare(mesh).nObs() != count + 1)
            return 3;
        mesh.computeGeometry();
        rejected = false;
        try {
            prepared.evaluate(f, .01);
        } catch (const std::logic_error &) {
            rejected = true;
        }
        if (!rejected)
            return 4;
    }
    std::cout << "prepared/direct exact equality, dynamic values and stale/alias guards passed\n";
}
