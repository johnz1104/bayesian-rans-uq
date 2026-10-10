"""Reject unprepared meshes at the Python boundary before SST can dereference them."""

import pytest

from cfdr import _core as c


@pytest.mark.parametrize("kind", ["forward", "compressible", "sensitivity"])
def test_unprepared_mesh_is_rejected_before_solve(kind):
    mesh = c.Mesh.make_channel_2d(2, 4, 1.0, 1.0)
    params = c.InferenceParameterSet.a1_betaStar()
    obs = c.ObservationOperator()
    bc = c.FlowBoundaryConditions.channel_defaults(mesh, 1.0, 1e-4, 1.0)
    with pytest.raises(ValueError, match="compute_wall_distance"):
        if kind == "forward":
            c.ForwardModel(mesh, params, obs, bc, 0.01)
        elif kind == "sensitivity":
            c.ParameterSensitivity(mesh, obs, bc, 0.01)
        else:
            cbc = c.CompressibleBoundaryConditions.channel_defaults(
                mesh, 1.0, 300.0, 101325.0, 1e-4, 1.0
            )
            c.CompressibleForwardModel(mesh, params, obs, cbc, c.IdealGasEOS())


def test_retyping_invalidates_preparation_and_existing_solver():
    mesh = c.Mesh.make_channel_2d(3, 4, 1.0, 1.0)
    mesh.compute_wall_distance()
    bc = c.FlowBoundaryConditions.channel_defaults(mesh, 1.0, 1e-4, 1.0)
    solver = c.SIMPLESolver(mesh, c.SSTCoefficients(), bc, 0.01)
    fields = c.FlowFields(mesh)
    mesh.set_patch_type("top", "symmetry")
    with pytest.raises(ValueError, match="compute_wall_distance"):
        c.ForwardModel(
            mesh, c.InferenceParameterSet.a1_betaStar(), c.ObservationOperator(), bc, 0.01
        )
    mesh.compute_wall_distance()
    with pytest.raises(RuntimeError, match="stale"):
        solver.init_uniform(fields, c.Vec3(1, 0, 0), 0, 1e-4, 1)
    rebuilt = c.SIMPLESolver(mesh, c.SSTCoefficients(), bc, 0.01)
    rebuilt.init_uniform(c.FlowFields(mesh), c.Vec3(1, 0, 0), 0, 1e-4, 1)
