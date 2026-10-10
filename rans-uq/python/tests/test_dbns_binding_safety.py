"""Unsafe native inputs run in a child process so regressions cannot crash pytest."""

import subprocess
import sys

import numpy as np
import pytest

from cfdr import _core as c

SETUP = r"""
import sys
import numpy as np
from cfdr import _core as c
mesh = c.Mesh.make_channel_2d(4, 4, 1.0, 1.0)
settings = c.DBNSSettings()
settings.max_iterations = 1
settings.verbose = False
settings.viscous = True
settings.turbulent = False
settings.const_mu = 1e-4

def solver(profile=None):
    bcs = c.DBNSBoundaryConditions()
    for patch in mesh.patch_names():
        spec = c.DBNSBoundarySpec()
        spec.kind = c.DBNSBoundaryKind.Extrapolate
        if patch == "bottom":
            spec.kind = c.DBNSBoundaryKind.NoSlipIsothermal
            spec.wall_temp = 300.0
            if profile is not None:
                spec.set_wall_temp_profile(profile)
        bcs.set(patch, spec)
    s = c.DBNSSolver(mesh, c.IdealGasEOS(), c.SSTCoefficients(), bcs, settings)
    s.init_uniform(c.Primitive(1.0, 0.0, 0.0, 287.0 * 300.0, 0.0, 0.0))
    return s
"""


def run_child(code, *args):
    result = subprocess.run(
        [sys.executable, "-c", SETUP + code, *map(str, args)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("index", [-1, 16])
def test_primitive_rejects_out_of_range_cell(index):
    run_child(
        r"""
s = solver()
try:
    s.primitive(int(sys.argv[1]))
except IndexError:
    pass
else:
    raise AssertionError("invalid cell did not raise IndexError")
""",
        index,
    )


def test_primitive_preserves_valid_cells():
    mesh = c.Mesh.make_channel_2d(2, 4, 1.0, 1.0)
    solver = c.DBNSSolver(
        mesh,
        c.IdealGasEOS(),
        c.SSTCoefficients(),
        c.DBNSBoundaryConditions(),
        c.DBNSSettings(),
    )
    solver.init_uniform(c.Primitive(1.2, 2.0, 3.0, 90000.0, 0.0, 0.0))
    for cell in (0, mesh.n_cells() - 1):
        state = solver.primitive(cell)
        assert (state.rho, state.u, state.v, state.p) == pytest.approx((1.2, 2, 3, 90000))


@pytest.mark.parametrize("column", [4, 5])
def test_laminar_dbns_rejects_turbulence_state(column):
    # Total energy includes rho*k, so a laminar k would become heat when the
    # solver clears it. Laminar states must carry zero k and omega.
    mesh = c.Mesh.make_channel_2d(2, 2, 1.0, 1.0)
    settings = c.DBNSSettings()
    solver = c.DBNSSolver(
        mesh, c.IdealGasEOS(), c.SSTCoefficients(), c.DBNSBoundaryConditions(), settings
    )
    state = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    solver.init_uniform(c.Primitive(*state))
    state[column] = 0.5
    with pytest.raises(ValueError, match="laminar"):
        solver.init_uniform(c.Primitive(*state))
    field = np.tile(state, (mesh.n_cells(), 1))
    with pytest.raises(ValueError, match="laminar"):
        solver.init_field(field)
    assert solver.primitive(0).p == 1.0

    settings.turbulent = True
    turbulent = c.DBNSSolver(
        mesh, c.IdealGasEOS(), c.SSTCoefficients(), c.DBNSBoundaryConditions(), settings
    )
    turbulent.init_field(field)
    assert (turbulent.primitive(0).k, turbulent.primitive(0).omega) == tuple(state[4:])


@pytest.mark.parametrize("layout", ["stride", "reverse", "empty"])
@pytest.mark.parametrize("route", ["observation", "boundary"])
def test_wall_temperature_views_match_contiguous_values(layout, route):
    run_child(
        r"""
layout, route = sys.argv[1:]
base = np.arange(8, dtype=float) + 305.0
values = base[::2] if layout == "stride" else base[3::-1]
if layout == "empty":
    values = base[:0]
contiguous = np.ascontiguousarray(values)
if route == "observation":
    s = solver()
    s.prepare_properties()
    obs = c.DBNSObservation(s, c.ReferenceState())
    actual = obs.wall_profile("bottom", values, 300.0)
    expected = obs.wall_profile("bottom", contiguous, 300.0)
    if layout == "empty":
        expected = obs.wall("bottom", 300.0)
else:
    a, b = solver(values), solver(None if layout == "empty" else contiguous)
    a.solve()
    b.solve()
    actual, expected = a.fields(), b.fields()
for name in actual:
    np.testing.assert_array_equal(actual[name], expected[name])
""",
        layout,
        route,
    )


def test_dbns_rejects_retyped_mesh():
    run_child(r"""
s = solver()
mesh.set_patch_type("top", "symmetry")
for action in [lambda: s.primitive(0), s.prepare_properties, s.solve]:
    try:
        action()
    except RuntimeError as error:
        assert "stale" in str(error)
    else:
        raise AssertionError("stale DBNS state was used")
rebuilt = solver()
assert rebuilt.n_cells() == mesh.n_cells()
""")


@pytest.mark.parametrize("route", ["tensors", "heat", "mask"])
def test_correction_dimensions_reject_narrowing_before_copy(route):
    run_child(
        r"""
s = c.DBNSSolver(mesh, c.IdealGasEOS(), c.SSTCoefficients(),
                 c.DBNSBoundaryConditions(), settings)
n = s.n_cells()
db = np.zeros((n, 3, 3))
b = db.copy()
dq = np.zeros((n, 2))
mask = np.zeros(n, dtype=bool)
s.set_target_correction(db, b, dq, False, mask)

# Logical dimensions differ from the native count by 2**32. These read-only
# zero-stride views use only tiny backing arrays; no large payload is allocated.
rows = 2**32 + n
route = sys.argv[1]
if route == "tensors":
    db = b = np.broadcast_to(np.zeros((1, 3, 3)), (rows, 3, 3))
    view = db
elif route == "heat":
    dq = np.broadcast_to(np.zeros((1, 2)), (rows, 2))
    view = dq
else:
    mask = np.broadcast_to(np.zeros(1, dtype=bool), (rows,))
    view = mask
assert view.strides[0] == 0 and not view.flags.writeable
try:
    s.set_target_correction(db, b, dq, False, mask)
except ValueError as error:
    assert "n_cells" in str(error)
else:
    raise AssertionError("oversized correction dimension was silently narrowed")
""",
        route,
    )
