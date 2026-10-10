"""Prepared definitions borrow a live mesh and reread every field value."""

import gc

import numpy as np
import pytest

from cfdr import _core as c


def test_prepared_observations_own_definitions_and_keep_mesh_alive():
    mesh = c.Mesh.make_channel_2d(3, 4, 1.0, 1.0)
    mesh.compute_wall_distance()
    obs = c.ObservationOperator()
    obs.add_velocity_profile(c.Vec3(0.5, 0.3, 0), 0, 0, 1)
    prepared = obs.prepare(mesh)
    fields = c.FlowFields(mesh)
    snapshot = fields.snapshot()
    snapshot["U"][:] = (2, 0, 0)
    snapshot["boundary"]["U"][:] = (2, 0, 0)
    fields.restore(snapshot)
    assert prepared.evaluate(fields, 0.01) == obs.evaluate(mesh, fields, 0.01)
    obs.add_velocity_profile(c.Vec3(0.8, 0.3, 0), 0, 0, 1)
    del obs
    gc.collect()
    assert prepared.n_obs() == 1
    np.testing.assert_array_equal(prepared.evaluate(fields, 0.01), [2])
    mesh.compute_geometry()
    with pytest.raises(RuntimeError, match="stale"):
        prepared.evaluate(fields, 0.01)


def test_prepared_survives_original_mesh_name_deletion():
    mesh = c.Mesh.make_channel_2d(1, 1, 1.0, 1.0)
    fields = c.FlowFields(mesh)
    obs = c.ObservationOperator()
    obs.add_velocity_profile(c.Vec3(0.5, 0.5, 0), 0, 0, 1)
    prepared = obs.prepare(mesh)
    del mesh, obs
    gc.collect()
    assert prepared.evaluate(fields, 0.01) == [0]
