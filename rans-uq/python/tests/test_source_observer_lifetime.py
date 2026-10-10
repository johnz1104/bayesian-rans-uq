"""Python callback captures must not create invisible native ownership cycles."""

import gc
import weakref

import pytest

from cfdr import PeriodicChannelConfig, SolverConfig
from cfdr import _core as c


def make_solver():
    mesh = PeriodicChannelConfig(nx=4, ny=8).mesh()
    settings = SolverConfig(max_iterations=2).native()
    settings.turb_start_iter = 10
    bc = c.FlowBoundaryConditions.channel_defaults(mesh, 0.0, 0.0, 1.0)
    solver = c.SIMPLESolver(mesh, c.SSTCoefficients(), bc, 0.01, settings)
    fields = c.FlowFields(mesh)
    solver.init_uniform(fields, c.Vec3(0, 0, 0), 0.0, 0.0, 1.0)
    return solver, fields


def test_callback_capturing_solver_is_collectable():
    def scope():
        solver, fields = make_solver()

        def observer(snapshot):
            assert solver.face_fluxes().shape[0] > 0

        solver.solve(fields, source_observer=observer)
        return weakref.ref(solver)

    reference = scope()
    gc.collect()
    assert reference() is None


def test_observer_exception_clears_native_callback():
    def scope():
        solver, fields = make_solver()

        def observer(snapshot):
            assert solver.face_fluxes().shape[0] > 0
            raise RuntimeError("observer interrupted solve")

        callback_ref = weakref.ref(observer)
        with pytest.raises(RuntimeError, match="observer interrupted"):
            solver.solve(fields, source_observer=observer)
        del observer
        gc.collect()
        assert callback_ref() is None
        # A later solve cannot invoke the callback from the failed call.
        assert not solver.solve(fields).diverged
        return weakref.ref(solver)

    reference = scope()
    gc.collect()
    assert reference() is None


@pytest.mark.parametrize("nested_observer", [False, True])
def test_recursive_solve_does_not_replace_or_clear_outer_observer(nested_observer):
    solver, fields = make_solver()
    iterations = []
    nested_calls = []

    def observer(snapshot):
        iterations.append(snapshot["iteration"])
        with pytest.raises(RuntimeError, match="recursive solve"):
            if nested_observer:
                solver.solve(fields, source_observer=nested_calls.append)
            else:
                solver.solve(fields)

    assert not solver.solve(fields, source_observer=observer).diverged
    assert len(iterations) == 2
    assert nested_calls == []
    # Both the active-solve guard and observer scope must reset after exit.
    later = []
    assert not solver.solve(fields, source_observer=later.append).diverged
    assert len(later) == 2
