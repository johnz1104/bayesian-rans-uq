#pragma once

#include "Mesh.hpp"
#include "DBNSSolver.hpp"
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>

// ---------------------------------------------------------------------------
// Wall observation operator for the density-based compressible solver.
//
// Follows the same pattern as the core ObservationOperator (which maps the flow
// state to predicted measurements), but extended to the compressible wall
// quantities the SBLI heat-flux program needs and which the incompressible
// operator cannot form because it has no temperature field:
//   - skin friction  Cf = tau_w / (0.5 rho_inf U_inf^2)
//   - wall pressure  Cp = (p_w - p_inf) / (0.5 rho_inf U_inf^2)
//   - wall heat flux q_w = lambda_eff (T_1 - T_w)/delta        [W/m^2]
//   - Stanton number St  = q_w / (rho_inf U_inf Cp (T_aw - T_w))
// with recovery temperature T_aw = T_inf (1 + r (gamma-1)/2 M_inf^2) and
// recovery factor r (sqrt(Pr) laminar, cbrt(Pr) turbulent).
// ---------------------------------------------------------------------------

namespace dbns {

// Freestream reference conditions for non-dimensionalisation.
struct ReferenceState {
    double rho = 1.0;
    double U = 1.0;
    double T = 300.0;
    double p = 101325.0;
    double recoveryFactor = 0.0; // 0 -> sqrt(Pr) chosen automatically
};

// Per-wall-face observation record.
struct WallRecord {
    std::vector<double> x; // face-centre streamwise coordinate
    std::vector<double> Cf;
    std::vector<double> Cp;
    std::vector<double> qw; // wall heat flux [W/m^2]
    std::vector<double> St; // Stanton number
};

class DBNSObservation {
  public:
    DBNSObservation(const DBNSSolver &solver, const ReferenceState &ref)
        : solver_(solver), ref_(ref) {}

    // Compute the wall records on a named patch (ordered by face index).
    WallRecord wall(const std::string &patchName, double wallTemp) const {
        return wallProfile(patchName, {}, wallTemp);
    }

    // Per-face wall temperatures (the interaction data's measured
    // wall-temperature row); entries beyond the vector fall back to wallTemp.
    WallRecord wallProfile(const std::string &patchName, const std::vector<double> &wallTemps,
                           double wallTemp) const {
        const Mesh &mesh = solver_.mesh();
        const IdealGasEOS &eos = solver_.eos();
        PatchID pid = mesh.patchByName(patchName);
        const Patch &pat = mesh.patch(pid);

        double dynP = 0.5 * ref_.rho * ref_.U * ref_.U;
        double Minf = ref_.U / std::sqrt(eos.gamma * eos.R * ref_.T);
        double r = ref_.recoveryFactor > 0.0 ? ref_.recoveryFactor : std::sqrt(eos.Pr);
        double Taw = ref_.T * (1.0 + r * 0.5 * (eos.gamma - 1.0) * Minf * Minf);

        WallRecord rec;
        size_t j = 0;
        for (FaceID fi : pat.faces) {
            const Face &f = mesh.face(fi);
            int P = f.owner;
            double Tw = (j < wallTemps.size()) ? wallTemps[j] : wallTemp;
            ++j;
            Primitive V = solver_.primitive(P);
            double T1 = V.p / (V.rho * eos.R);
            // use the solver's own viscosity (honours constMu) so the wall
            // observation is consistent with the field it reads. Transport
            // at a resolved wall face is MOLECULAR: the eddy viscosity
            // vanishes at the wall, and the owner-cell muT would overestimate
            // the wall shear and heat flux (immaterial at y+ 0.05, where
            // muT/mu is of order 1e-15 to 1e-12). This matches the solver's
            // wall-flux convention.
            double muLam = solver_.laminarViscosity(P);
            double delta = std::max(f.delta, 1e-12);

            // wall-tangential velocity magnitude at the near-wall cell
            Vec3 Uc{V.u, V.v, 0.0};
            double un = Uc.dot(f.normal);
            Vec3 Ut = Uc - f.normal * un;
            double tau_w = muLam * Ut.norm() / delta;
            double Cf = tau_w / std::max(dynP, 1e-30);

            double Cp = (V.p - ref_.p) / std::max(dynP, 1e-30);

            // wall heat flux: conduction from the near-wall cell to the wall
            double lamWall = eos.Cp() * muLam / eos.Pr;
            double qw = lamWall * (T1 - Tw) / delta;

            double denomSt = ref_.rho * ref_.U * eos.Cp() * (Taw - Tw);
            double St = (std::abs(denomSt) > 1e-30) ? qw / denomSt : 0.0;

            rec.x.push_back(f.center.x);
            rec.Cf.push_back(Cf);
            rec.Cp.push_back(Cp);
            rec.qw.push_back(qw);
            rec.St.push_back(St);
        }
        return rec;
    }

  private:
    const DBNSSolver &solver_;
    ReferenceState ref_;
};

} // namespace dbns
