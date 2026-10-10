"""Configured SST variants must reach both Python construction routes."""

import numpy as np

from cfdr import _core as c


def test_direct_simple_honors_sst_variant():
    # Compare wrapper routing, not physical accuracy: the native forward model
    # explicitly selects the configured closure. Active turbulence distinguishes
    # the variants even in this deliberately iteration-limited cold solve.
    mesh = c.Mesh.make_channel_2d(3, 8, 2.0, 1.0)
    mesh.compute_wall_distance()
    bc = c.FlowBoundaryConditions.channel_defaults(mesh, 1.0, 0.02, 3.0)
    parameters = c.InferenceParameterSet.a1_betaStar()
    observations = c.ObservationOperator()
    snapshots = []
    references = []
    for variant in (0, 1, 2):
        settings = c.SolverSettings()
        settings.max_iterations = 20
        settings.verbose = False
        settings.turb_start_iter = 0
        settings.turb_update_interval = 1
        settings.nut_floor_iters = 0
        settings.sst_variant = variant
        direct = c.SIMPLESolver(mesh, c.SSTCoefficients(), bc, 0.01, settings)
        fields = c.FlowFields(mesh)
        direct.init_uniform(fields, c.Vec3(1, 0, 0), 0.0, 0.02, 3.0)
        history = direct.solve(fields)
        model = c.ForwardModel(
            mesh, parameters, observations, bc, 0.01, settings, c.Vec3(1, 0, 0), 0.0, 0.02, 3.0
        )
        result = model.evaluate([0.31, 0.09])
        assert not history.diverged
        assert not history.converged
        assert result.status == c.EvaluationStatus.Unconverged
        actual, expected = fields.snapshot(), model.last_fields()
        for name in ("U", "p", "k", "omega", "nuT", "F1", "F2"):
            assert np.isfinite(actual[name]).all()
            assert np.isfinite(expected[name]).all()
        snapshots.append(actual)
        references.append(expected)

    # KOmega changes the equations on this fixture; ignoring the selector makes
    # these arrays exactly equal. This guards against two shared default routes.
    assert np.any(snapshots[2]["nuT"] != snapshots[0]["nuT"])
    for alternative in snapshots[1:]:
        assert np.any(alternative["nuT"] != snapshots[0]["nuT"])
    for actual, expected in zip(snapshots, references, strict=True):
        for name in ("U", "p", "k", "omega", "nuT", "F1", "F2"):
            np.testing.assert_array_equal(actual[name], expected[name])
