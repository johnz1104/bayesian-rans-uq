"""Ownership regressions: weakrefs test retention without dereferencing freed data."""

import gc
import weakref

import pytest

from cfdr import _core as c


def model(compressible=False):
    mesh = c.Mesh.make_channel_2d(2, 4, 1.0, 1.0)
    mesh.compute_wall_distance()

    params = c.InferenceParameterSet.a1_betaStar()
    observations = c.ObservationOperator()
    settings = c.SolverSettings()
    settings.max_iterations = 1
    settings.verbose = False
    if compressible:
        bcs = c.CompressibleBoundaryConditions.channel_defaults(
            mesh, 1.0, 300.0, 101325.0, 1e-4, 1.0
        )
        fm = c.CompressibleForwardModel(mesh, params, observations, bcs, c.IdealGasEOS(), settings)
    else:
        bcs = c.FlowBoundaryConditions.channel_defaults(mesh, 1.0, 1e-4, 1.0)
        fm = c.ForwardModel(mesh, params, observations, bcs, 0.01, settings)
    return mesh, fm


@pytest.mark.parametrize("compressible", [False, True])
def test_forward_model_retains_borrowed_mesh(compressible):
    mesh, fm = model(compressible)
    reference = weakref.ref(mesh)
    del mesh
    gc.collect()
    assert reference() is not None
    del fm
    gc.collect()
    assert reference() is None


@pytest.mark.parametrize("compressible", [False, True])
def test_parameter_view_retains_parent(compressible):
    _mesh, fm = model(compressible)
    reference = weakref.ref(fm)
    params = fm.param_set()
    del fm
    gc.collect()
    assert reference() is not None
    assert params.n_active() == 2
    del params
    gc.collect()
    assert reference() is None
