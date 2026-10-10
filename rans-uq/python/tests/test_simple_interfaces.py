"""Python ownership and explicit-state access for the native SIMPLE interfaces."""

import gc
import weakref

import numpy as np

from cfdr import PeriodicChannelConfig, SolverConfig
from cfdr import _core as c


def test_simple_target_and_observer_outputs_are_owned():
    mesh = PeriodicChannelConfig(nx=4, ny=8).mesh()
    settings = SolverConfig(max_iterations=2).native()
    settings.turb_start_iter = 10
    bc = c.FlowBoundaryConditions.channel_defaults(mesh, 0.0, 0.0, 1.0)
    solver = c.SIMPLESolver(mesh, c.SSTCoefficients(), bc, 0.01, settings)
    fields = c.FlowFields(mesh)
    solver.init_uniform(fields, c.Vec3(0, 0, 0), 0.0, 0.0, 1.0)
    target = np.zeros((mesh.n_cells(), 6))
    solver.set_target_anisotropy(target)
    target[:] = np.nan  # Must not mutate the native target.
    snapshots = []
    history = solver.solve(fields, source_observer=snapshots.append)
    assert not history.diverged
    assert len(snapshots) == 2
    assert snapshots[0]["stage"] == "pre_momentum_x"
    assert snapshots[0]["fields"]["U"].flags.owndata
    assert snapshots[0]["fields"]["boundary"]["U"].flags.owndata
    snapshots[0]["fields"]["U"][:] = 20.0
    assert not np.any(snapshots[1]["fields"]["U"] == 20.0)
    flux = solver.face_fluxes()
    diagnostic = solver.diagnose_momentum(fields, c.SSTCoefficients(), flux)
    assert np.isfinite(diagnostic["residual"]).all()
    numeric = fields.snapshot()
    numeric["U"][:] = 3.0
    fields.restore(numeric)
    assert np.all(fields.snapshot()["U"] == 3.0)
    reference = weakref.ref(mesh)
    del mesh, target, solver
    gc.collect()
    assert reference() is not None  # FlowFields keeps its own borrowed mesh alive.
    del fields
    gc.collect()
    assert reference() is None
    assert np.isfinite(diagnostic["residual"]).all()
