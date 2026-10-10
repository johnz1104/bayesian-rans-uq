"""Per-evaluation diagnostics must outlive later calls and released solver storage."""

import gc

import numpy as np
from test_binding_lifetimes import model

from cfdr import _core as c


def test_complete_snapshot_and_diagnostics_survive_later_failures():
    mesh, fm = model()
    theta = fm.param_set().pack(c.SSTCoefficients())
    fm.set_retain_last_fields(False)
    fm.cache().set_byte_budget(0)
    result = fm.evaluate_with_diagnostics(theta, snapshot=True)
    assert result.error == ""
    assert result.result.status == c.EvaluationStatus.Unconverged
    assert not fm.has_last_fields()
    snapshot = result.snapshot
    assert snapshot["U"].shape == (mesh.n_cells(), 3)
    assert snapshot["boundary"]["U"].shape == (mesh.n_faces() - mesh.n_internal_faces(), 3)
    assert snapshot["face_fluxes"].shape == (mesh.n_faces(),)
    assert snapshot["source_digest"] == c.build_info()["source_digest"]
    invalid = fm.evaluate_with_diagnostics([np.nan, 0.09], snapshot=True)
    assert invalid.result.status == c.EvaluationStatus.InvalidParameters
    assert invalid.snapshot is None
    assert not invalid.injection.active
    assert fm.cache().retained_bytes() == 0
    prior = snapshot["U"].copy()
    del mesh, fm, result
    gc.collect()
    assert np.array_equal(snapshot["U"], prior)
    assert snapshot["U"].flags.owndata
    assert snapshot["boundary"]["U"].flags.owndata
