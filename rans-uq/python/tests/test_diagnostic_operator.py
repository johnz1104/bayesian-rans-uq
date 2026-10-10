"""Owned diagnostic output and the supported-configuration boundary."""

import gc

import pytest

from cfdr import _core as core


def test_diagnostic_output_owns_all_blocks():
    mesh = core.Mesh.make_channel_2d(3, 4, 1.0, 1.0)
    mesh.compute_wall_distance()
    bcs = core.FlowBoundaryConditions.channel_defaults(mesh, 1.0, 0.01, 1.0)
    settings = core.SolverSettings()
    settings.momentum_convection = "upwind"
    settings.turbulence_convection = "linearUpwind"
    theta = core.SSTCoefficients()
    solver = core.SIMPLESolver(mesh, theta, bcs, 0.001, settings)
    fields = core.FlowFields(mesh)
    solver.init_uniform(fields, core.Vec3(1, 0.2, 0), 0, 0.01, 1)
    result = solver.diagnostic_residual(fields, theta)
    columns = solver.diagnostic_parameter_partials(fields, theta)
    assert len(result.transport) == 4 * mesh.n_cells()
    assert len(result.continuity) == mesh.n_cells()
    assert len(result.face_fluxes) == mesh.n_faces()
    assert len(columns) == 11
    assert not any(columns[10].transport + columns[10].continuity + columns[10].face_fluxes)
    saved = result.transport
    del fields, solver, mesh
    gc.collect()
    assert result.transport == saved


def test_limited_diagnostic_is_explicitly_unsupported():
    mesh = core.Mesh.make_channel_2d(3, 4, 1.0, 1.0)
    mesh.compute_wall_distance()
    bcs = core.FlowBoundaryConditions.channel_defaults(mesh, 1.0, 0.01, 1.0)
    theta = core.SSTCoefficients()
    solver = core.SIMPLESolver(mesh, theta, bcs, 0.001)
    fields = core.FlowFields(mesh)
    with pytest.raises(RuntimeError, match="upwind"):
        solver.diagnostic_residual(fields, theta)
