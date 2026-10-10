"""Package configuration, identity, and owned field exports."""

import gc
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError
from importlib.resources import files

import numpy as np
import pytest
from test_binding_lifetimes import model

from cfdr import PeriodicChannelConfig, SolverConfig
from cfdr import _core as c


def test_configuration_is_frozen_and_maps_convection():
    config = SolverConfig(max_iterations=3, convection="upwind")
    with pytest.raises(FrozenInstanceError):
        config.max_iterations = 4
    native = config.native()
    assert native.max_iterations == 3
    assert native.momentum_convection == native.turbulence_convection == "upwind"
    assert PeriodicChannelConfig(nx=4, ny=8).mesh().n_cells() == 32
    with pytest.raises(ValueError):
        PeriodicChannelConfig(nx=2)
    with pytest.raises(ValueError):
        PeriodicChannelConfig(re_tau=0)


def test_build_identity_is_explicit():
    info = c.build_info()
    assert info["version"] == "0.1.0"
    assert len(info["source_digest"]) == 64
    assert info["compiler"]
    assert info["revision"]


@pytest.mark.parametrize("compressible", [False, True])
def test_last_fields_are_owned_complete_interiors(compressible):
    mesh, fm = model(compressible)
    fm.evaluate(fm.param_set().pack(c.SSTCoefficients()))
    fields = fm.last_fields()
    assert {"U", "p", "k", "omega", "nuT", "F1", "F2", "Pk", "CDkw"} <= fields.keys()
    original = fields["U"].copy()
    assert all(array.flags.owndata for array in fields.values())
    fields["U"][:] = 123
    assert np.array_equal(fm.last_fields()["U"], original)
    del mesh, fm
    gc.collect()
    assert np.all(fields["U"] == 123)


def test_missing_extension_fails_loudly(tmp_path):
    # A source-only copy deliberately has no extension and must not fall back.
    package = tmp_path / "cfdr"
    package.mkdir()
    source = files("cfdr").joinpath("__init__.py")
    (package / "__init__.py").write_text(source.read_text())
    env = {**os.environ, "PYTHONPATH": str(tmp_path)}
    result = subprocess.run(
        [sys.executable, "-c", "import cfdr"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "requires its compiled _core extension" in result.stderr


def test_dbns_rejects_wrong_primitive_shape():
    mesh = c.Mesh.make_channel_2d(2, 4, 1, 1)
    solver = c.DBNSSolver(
        mesh,
        c.IdealGasEOS(),
        c.SSTCoefficients(),
        c.DBNSBoundaryConditions(),
        c.DBNSSettings(),
    )
    with pytest.raises(ValueError, match="n_cells"):
        solver.init_field(np.zeros((1, 4)))


@pytest.mark.parametrize("value", [True, 2.5])
def test_solver_config_rejects_noninteger_iteration_counts(value):
    with pytest.raises(TypeError, match="integer"):
        SolverConfig(max_iterations=value)


@pytest.mark.parametrize("value", [0, -1])
def test_solver_config_rejects_nonpositive_iteration_counts(value):
    with pytest.raises(ValueError, match="positive"):
        SolverConfig(max_iterations=value)


@pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf])
def test_solver_config_rejects_invalid_tolerance(value):
    with pytest.raises(ValueError, match="finite and positive"):
        SolverConfig(convergence_tol=value)


def test_solver_config_rejects_unknown_convection():
    with pytest.raises(ValueError, match="unknown convection"):
        SolverConfig(convection="unsupported")


@pytest.mark.parametrize("name", ["nx", "ny"])
@pytest.mark.parametrize("value", [True, 2.5])
def test_channel_config_rejects_noninteger_counts(name, value):
    with pytest.raises(TypeError, match="integer"):
        PeriodicChannelConfig(**{name: value})


@pytest.mark.parametrize("name", ["length", "half_height", "re_tau", "y_plus_target"])
@pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf])
def test_channel_config_rejects_invalid_physical_controls(name, value):
    with pytest.raises(ValueError, match="finite and positive"):
        PeriodicChannelConfig(**{name: value})
