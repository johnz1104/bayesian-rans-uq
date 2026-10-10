"""Cold batch equivalence, failures, owned arrays and native GIL release."""

import gc
import sys
import threading

import numpy as np
import pytest

from cfdr import _core as c


def problem(iterations=8):
    mesh = c.Mesh.make_channel_2d(3, 4, 1.0, 1.0)
    mesh.compute_wall_distance()
    params = c.InferenceParameterSet.a1_betaStar()
    obs = c.ObservationOperator()
    obs.add_velocity_profile(c.Vec3(0.5, 0.4, 0), 0, 0, 1)
    bc = c.FlowBoundaryConditions.channel_defaults(mesh, 1.0, 0.02, 3.0)
    settings = c.SolverSettings()
    settings.verbose = False
    settings.max_iterations = iterations
    return mesh, params, obs, bc, 0.01, settings


def test_batch_matches_serial_cold_and_owns_inputs_and_results():
    inputs = problem()
    theta = np.array([[0.31, 0.09], [0.35, 0.1], [-1, 0.09], [np.nan, 0.1], [0.3, 0.08]])
    batch = c.ForwardBatch(*inputs)
    expected = []
    for row in theta:
        model = c.ForwardModel(*inputs)
        model.cache().set_byte_budget(0)
        expected.append(model.evaluate(row))
    one, two = batch.evaluate(theta), batch.evaluate(theta, workers=2)
    np.testing.assert_array_equal(one.predictions, two.predictions)
    assert one.statuses == two.statuses == [row.status for row in expected]
    np.testing.assert_array_equal(one.iteration_counts, [row.simple_iters for row in expected])
    np.testing.assert_array_equal(one.log_likelihoods, [row.log_lik for row in expected])
    for index, row in enumerate(expected):
        if row.predictions:
            np.testing.assert_array_equal(one.predictions[index], row.predictions)
        else:
            assert np.isnan(one.predictions[index]).all()
    assert one.errors[2] and one.errors[3]
    assert one.convergence_mask.dtype == np.bool_
    assert not one.convergence_mask.any()
    saved = two.predictions
    assert saved.flags.owndata
    inputs[0].compute_geometry()
    inputs[2].add_velocity_profile(c.Vec3(0.1, 0.2, 0), 0, 0, 1)
    del inputs, model, expected
    gc.collect()
    np.testing.assert_array_equal(batch.evaluate(theta, 2).predictions, saved)
    saved[0, 0] = 123
    assert batch.evaluate(theta).predictions[0, 0] != 123
    del batch, one, two
    gc.collect()
    assert saved[0, 0] == 123


def test_empty_malformed_and_recovery():
    batch = c.ForwardBatch(*problem(0))
    result = batch.evaluate(np.empty((0, 2)), 2)
    assert result.predictions.shape == (0, 1)
    assert result.convergence_mask.shape == (0,)
    for malformed in (np.ones(2), np.ones((3, 1)), np.ones((1, 2, 1))):
        with pytest.raises(ValueError, match="shape"):
            batch.evaluate(malformed, 2)
    with pytest.raises(ValueError, match="workers"):
        batch.evaluate(np.ones((1, 2)), 0)
    result = batch.evaluate([[0.31, 0.09]])
    assert result.statuses == [c.EvaluationStatus.Unconverged]
    assert np.isfinite(result.predictions).all()
    assert result.iteration_counts[0] == 0


def test_invalid_batch_configuration_fails_at_construction():
    inputs = problem(0)
    inputs[-1].momentum_convection = "invalid-scheme"
    with pytest.raises(ValueError, match="convection"):
        c.ForwardBatch(*inputs)


def test_ordinary_row_errors_are_records():
    inputs = problem(0)
    ordinary = c.ForwardBatch(*inputs).evaluate([[0.31, 0.09], [0.3, 0.08]], 2)
    assert ordinary.statuses == [c.EvaluationStatus.Unconverged] * 2
    assert np.isfinite(ordinary.predictions).all()

    # Finite, admitted measurements can still overflow the computed likelihood.
    # Each failed row must retain its explicit status, error and output position.
    obs = c.ObservationOperator()
    obs.add_velocity_profile(c.Vec3(0.5, 0.4, 0), 0, 1e308, 1)
    mesh, params, _, bc, nu, settings = inputs
    errors = c.ForwardBatch(mesh, params, obs, bc, nu, settings).evaluate(
        [[0.31, 0.09], [0.3, 0.08]], 2
    )
    assert errors.statuses == [c.EvaluationStatus.Unknown] * 2
    assert all("finite range" in message for message in errors.errors)
    assert np.isnan(errors.predictions).all()


def test_divergence_is_a_row_record():
    inputs = problem(30)
    inputs[-1].divergence_limit = 0
    diverged = c.ForwardBatch(*inputs).evaluate([[0.31, 0.09]], 1)
    assert diverged.statuses == [c.EvaluationStatus.DivergenceDetected]
    assert np.isnan(diverged.predictions).all()


def test_native_batch_releases_gil():
    batch = c.ForwardBatch(*problem(100))
    ready, go, stop = threading.Event(), threading.Event(), threading.Event()
    progress = [0]

    def other_python_thread():
        ready.set()
        go.wait()
        while not stop.is_set():
            progress[0] += 1

    thread = threading.Thread(target=other_python_thread)
    thread.start()
    ready.wait()
    old_interval = sys.getswitchinterval()
    try:
        # Avoid an incidental Python timeslice between go.set and native entry.
        sys.setswitchinterval(1.0)
        go.set()
        batch.evaluate(np.tile([0.31, 0.09], (8, 1)), workers=2)
        assert progress[0] > 0
    finally:
        stop.set()
        sys.setswitchinterval(old_interval)
        thread.join(timeout=5)
    assert not thread.is_alive()


def test_converged_batch_mask_and_likelihood_match_cold_reference():
    inputs = problem(5000)
    theta = np.array([[0.31, 0.09], [0.32, 0.095]])
    batch = c.ForwardBatch(*inputs)
    result = batch.evaluate(theta, 2)
    assert result.convergence_mask.all()
    for i, row in enumerate(theta):
        cold = c.ForwardModel(*inputs)
        cold.cache().set_byte_budget(0)
        reference = cold.evaluate(row)
        assert result.statuses[i] == reference.status == c.EvaluationStatus.Converged
        assert result.log_likelihoods[i] == reference.log_lik
        np.testing.assert_array_equal(result.predictions[i], reference.predictions)
