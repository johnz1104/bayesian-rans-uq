"""Synthetic solve batches with hand-calculated statistical answers."""

import itertools
import math

import numpy as np
import pytest

from cfdr import _core as c
from cfdr.statistics import (
    Evaluation,
    EvaluationSet,
    Sample,
    conformal_quantile,
    empirical_crps,
    evaluate_batch,
    evaluate_scalar,
    independent_uniform_design,
    posterior_weights,
    predictive_summary,
    randomized_ranks,
    rank_histograms,
    resample_posterior,
    sample_rng,
    split_conformal_intervals,
)


def record(sample_id, *, status="Converged", likelihood=0.0, prediction=(0.0,), theta=(0.31, 0.09)):
    return Evaluation(sample_id, theta, status, prediction, likelihood, 1, "analytic", "")


def problem(iterations=0):
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


def test_native_admission_keeps_failed_ids_and_aligned_statuses():
    # Capped rows must not be labelled converged, and failed rows must keep their IDs.
    samples = (Sample(8, (0.31, 0.09)), Sample(3, (-1.0, 0.09)), Sample(9, (0.3, 0.08)))
    inputs = problem()
    scalar = evaluate_scalar(samples, lambda: c.ForwardModel(*inputs))
    one = evaluate_batch(samples, c.ForwardBatch(*inputs), workers=1)
    two = evaluate_batch(samples[::-1], c.ForwardBatch(*inputs), workers=2)
    assert scalar.failure_count == one.failure_count == two.failure_count == 3
    assert tuple(r.sample_id for r in scalar.records) == (8, 3, 9)
    assert tuple(r.status for r in scalar.records) == (
        "Unconverged",
        "InvalidParameters",
        "Unconverged",
    )
    for expected, actual, reverse in zip(scalar.records, one.records, two.records[::-1]):
        assert expected.sample_id == actual.sample_id == reverse.sample_id
        assert expected.status == actual.status == reverse.status
        assert expected.log_likelihood is actual.log_likelihood is reverse.log_likelihood is None
        np.testing.assert_array_equal(expected.predictions, actual.predictions)
        np.testing.assert_array_equal(actual.predictions, reverse.predictions)
        assert expected.backend == "cfdr._core.ForwardModel"
        assert actual.backend == "cfdr._core.ForwardBatch"
    with pytest.raises(ValueError, match="unresolved"):
        posterior_weights(one)


def test_likelihood_magnitude_never_admits_failure_or_rejects_valid_value():
    valid = EvaluationSet((record(1, likelihood=-2e30),))
    np.testing.assert_array_equal(posterior_weights(valid), [1.0])
    failed = record(2, status="Unconverged", likelihood=-1e30)
    with pytest.raises(ValueError, match="unresolved"):
        posterior_weights(EvaluationSet((valid.records[0], failed)))
    with pytest.raises(ValueError, match="finite"):
        posterior_weights(EvaluationSet((record(1, likelihood=float("nan")),)))


def test_importance_weights_and_resampling_use_ids():
    records = (record(5, likelihood=math.log(3)), record(2, theta=(0.32, 0.095)))
    np.testing.assert_allclose(posterior_weights(EvaluationSet(records)), [0.75, 0.25])
    a = resample_posterior(EvaluationSet(records), [10, 11, 12], root_seed=17)
    b = resample_posterior(EvaluationSet(records[::-1]), [12, 10, 11], root_seed=17)
    assert {s.sample_id: s.theta for s in a} == {s.sample_id: s.theta for s in b}


def test_native_converged_rows_match_scalar_cold_batch_and_reordering():
    inputs = problem(5000)
    samples = (Sample(8, (0.31, 0.09)), Sample(3, (0.32, 0.095)))
    scalar = evaluate_scalar(samples, lambda: c.ForwardModel(*inputs))
    batch = evaluate_batch(samples[::-1], c.ForwardBatch(*inputs), workers=2)
    assert scalar.failure_count == batch.failure_count == 0
    for expected, actual in zip(scalar.records, batch.records[::-1]):
        assert expected.status == actual.status == "Converged"
        assert expected.log_likelihood == actual.log_likelihood
        assert expected.iterations == actual.iterations
        assert expected.predictions == actual.predictions


def test_duplicate_sample_ids_are_rejected():
    with pytest.raises(ValueError, match="unique"):
        EvaluationSet((record(2), record(2)))
    with pytest.raises(ValueError, match="unique"):
        independent_uniform_design([2, 2], [0], [1], root_seed=1)


def test_design_streams_do_not_use_global_rng_or_row_order():
    np.random.seed(55)
    expected = np.random.random()
    np.random.seed(55)
    first = independent_uniform_design([8, 2, 5], [0, 2], [1, 3], root_seed=812)
    assert tuple(sample.sample_id for sample in first) == (8, 2, 5)
    assert np.random.random() == expected
    np.random.random(17)
    second = independent_uniform_design([5, 8, 2], [0, 2], [1, 3], root_seed=812)
    assert {s.sample_id: s.theta for s in first} == {s.sample_id: s.theta for s in second}
    assert sample_rng(812, 8, "design").random() != sample_rng(812, 8, "prediction").random()
    assert all(0 <= s.theta[0] < 1 and 2 <= s.theta[1] < 3 for s in first)


@pytest.mark.parametrize(
    "scores,alpha,expected",
    [
        ([1, 2, 3, 4], 0.5, 3),
        ([7], 0.1, math.inf),
        ([7], 0.5, 7),
        ([1, 1, 2, 2], 0.5, 2),
    ],
)
def test_conformal_exact_order_statistic(scores, alpha, expected):
    assert conformal_quantile(scores, alpha) == expected


@pytest.mark.parametrize(
    "scores,alpha",
    [
        ([], 0.1),
        ([1, np.nan], 0.5),
        ([1, np.inf], 0.5),
        ([1], 0),
        ([1], 1),
        ([1], np.nan),
        ([-1], 0.5),
    ],
)
def test_conformal_rejects_invalid_domains(scores, alpha):
    with pytest.raises(ValueError):
        conformal_quantile(scores, alpha)


def test_ordinary_split_coverage_on_exact_exchangeable_sweep_labels():
    # Fixed predictor f=0, one held-out calibration point, one test point.
    # Enumerating the six permutations: 3/6 tests have test <= calibration.
    covered = 0
    for labels in itertools.permutations([0.0, 1.0, 2.0]):
        lower, upper = split_conformal_intervals([labels[0]], [0], [0], alpha=0.5)
        covered += lower[0] <= labels[1] <= upper[0]
    assert covered == 3


def test_rank_histogram_counts_discrete_cardinality_per_parameter():
    ranks = np.tile(np.arange(21), 100)[:, None]
    diagnostic = rank_histograms(ranks, n_draws=20, bins=20)
    np.testing.assert_array_equal(diagnostic.counts, diagnostic.expected_counts)
    assert sorted(diagnostic.expected_counts[0]) == [100] * 19 + [200]
    duplicated = rank_histograms(np.repeat(ranks, 2, axis=1), n_draws=20, bins=20)
    np.testing.assert_array_equal(duplicated.counts, np.repeat(diagnostic.counts, 2, axis=0))
    assert not hasattr(diagnostic, "pvalue")


def test_rank_ties_are_randomized_by_trial_id_and_parameter():
    ids = tuple(range(60))
    truth, draws = np.zeros((60, 2)), np.zeros((60, 5, 2))
    ranks = randomized_ranks(truth, draws, ids, root_seed=914)
    assert set(ranks[:, 0]) == set(range(6))
    assert not np.array_equal(ranks[:, 0], ranks[:, 1])
    reverse = randomized_ranks(truth[::-1], draws[::-1], ids[::-1], root_seed=914)
    np.testing.assert_array_equal(ranks, reverse[::-1])
    untied = randomized_ranks([[2, 0]], [[[1, 1], [3, 2], [4, 3]]], [3], root_seed=914)
    np.testing.assert_array_equal(untied, [[1, 0]])


def test_prediction_labels_and_variance_components():
    deterministic = EvaluationSet((record(1, prediction=(4.0,)), record(2, prediction=(4.0,))))
    latent = predictive_summary(deterministic, root_seed=4)
    assert latent.latent_variance == latent.total_variance == (0.0,)
    assert not latent.includes_surrogate_noise
    assert not latent.includes_discrepancy
    assert not latent.includes_observation_noise
    observed = predictive_summary(deterministic, root_seed=4, observation_sigma=(100.0,))
    assert observed.observation_variance == observed.total_variance == (10000.0,)
    assert observed.includes_observation_noise
    assert observed.samples != latent.samples
    mixed = EvaluationSet((record(1, prediction=(0.0,)), record(2, prediction=(2.0,))))
    components = predictive_summary(
        mixed,
        root_seed=4,
        observation_sigma=(2.0,),
        surrogate_variance=(3.0,),
        discrepancy_variance=(5.0,),
    )
    assert components.latent_mean == (1.0,)
    assert components.latent_variance == (1.0,)
    assert components.total_variance == (13.0,)
    assert components.includes_surrogate_noise and components.includes_discrepancy


def test_empirical_crps_has_finite_forecast_denominator():
    # E|X-1|=1; 0.5 E|X-X'|=0.5 for X uniform on {0,2}.
    assert empirical_crps([0, 2], 1) == 0.5
    assert empirical_crps([0], 1) == 1
    assert empirical_crps([1e16, 1e16 + 2], 1e16) == 0.5
    with pytest.raises(ValueError):
        empirical_crps([0, np.nan], 1)


@pytest.mark.parametrize("error", ["solver exception", "invalid native diagnostic"])
def test_success_status_with_error_is_unresolved(error):
    evaluation = Evaluation(3, (0.31, 0.09), "Converged", (2.0,), 0.0, 1, "analytic", error)
    evaluations = EvaluationSet((evaluation,))
    assert not evaluation.admitted
    assert evaluations.failure_count == 1
    for action in (
        lambda: posterior_weights(evaluations),
        lambda: predictive_summary(evaluations, root_seed=17),
    ):
        with pytest.raises(ValueError, match="unresolved"):
            action()


@pytest.mark.parametrize(
    "prediction,likelihood",
    [([], 0.0), ([np.nan], 0.0), ([np.inf], 0.0), ([2.0], np.nan), ([2.0], np.inf)],
)
def test_scalar_admission_rejects_invalid_native_success(prediction, likelihood):
    from types import SimpleNamespace

    class ControlledModel(c.ForwardModel):
        def evaluate_with_diagnostics(self, theta):
            assert theta == (0.31, 0.09)
            return SimpleNamespace(
                result=SimpleNamespace(
                    status=c.EvaluationStatus.Converged,
                    predictions=prediction,
                    log_lik=likelihood,
                    simple_iters=1,
                ),
                error="",
            )

    evaluations = evaluate_scalar((Sample(9, (0.31, 0.09)),), lambda: ControlledModel(*problem()))
    assert evaluations.failure_count == 1
    result = evaluations.records[0]
    assert result.sample_id == 9
    assert result.status == "Unknown"
    assert result.log_likelihood is None
    assert result.error == "nonfinite or missing successful output"
    with pytest.raises(ValueError, match="unresolved"):
        posterior_weights(evaluations)


def test_scalar_admission_keeps_a_valid_native_success():
    from types import SimpleNamespace

    class ControlledModel(c.ForwardModel):
        def evaluate_with_diagnostics(self, theta):
            return SimpleNamespace(
                result=SimpleNamespace(
                    status=c.EvaluationStatus.Converged,
                    predictions=[2.0],
                    log_lik=-3.0,
                    simple_iters=4,
                ),
                error="",
            )

    evaluations = evaluate_scalar((Sample(9, (0.31, 0.09)),), lambda: ControlledModel(*problem()))
    assert evaluations.failure_count == 0
    assert evaluations.records[0].predictions == (2.0,)
    assert evaluations.records[0].log_likelihood == -3.0
    assert evaluations.records[0].iterations == 4
    np.testing.assert_array_equal(posterior_weights(evaluations), [1.0])


@pytest.mark.parametrize("bad", [-1, True, 1.5])
def test_statistics_rejects_invalid_identifiers(bad):
    with pytest.raises(ValueError, match="nonnegative integer"):
        Sample(bad, (0.31, 0.09))
    with pytest.raises(ValueError, match="nonnegative integer"):
        sample_rng(bad, 3, "design")


def test_rng_requires_an_explicit_purpose():
    with pytest.raises(ValueError, match="purpose must be nonempty"):
        sample_rng(1, 2, "")


@pytest.mark.parametrize("lower,upper", [([], []), ([1], [0]), ([0], [np.inf]), ([0], [1, 2])])
def test_design_rejects_invalid_bounds(lower, upper):
    with pytest.raises(ValueError, match="uniform bounds"):
        independent_uniform_design([3], lower, upper, root_seed=1)


def test_evaluation_requires_known_status_and_backend():
    with pytest.raises(ValueError, match="unrecognized evaluation status"):
        record(1, status="success")
    with pytest.raises(ValueError, match="backend must be explicit"):
        Evaluation(1, (0.31, 0.09), "Converged", (1.0,), 0.0, 1, "", "")


@pytest.mark.parametrize("samples", [(Sample(1, ()),), (Sample(1, (0.3,)), Sample(2, (0.3, 0.09)))])
def test_evaluation_rejects_incompatible_parameter_vectors(samples):
    with pytest.raises(ValueError, match="equal nonzero lengths"):
        evaluate_scalar(samples, lambda: c.ForwardModel(*problem()))


def test_evaluation_requires_native_backends_and_positive_workers():
    with pytest.raises(TypeError, match="model_factory must construct"):
        evaluate_scalar((Sample(1, (0.31, 0.09)),), object)
    with pytest.raises(TypeError, match="batch must be"):
        evaluate_batch((), object())
    batch = c.ForwardBatch(*problem())
    with pytest.raises(ValueError, match="worker count must be positive"):
        evaluate_batch((), batch, workers=0)
    assert evaluate_batch((), batch).records == ()


def test_batch_admission_rejects_a_lost_native_row():
    from types import SimpleNamespace

    class MissingRowBatch(c.ForwardBatch):
        def evaluate(self, theta, workers):
            return SimpleNamespace(rows=[])

    with pytest.raises(RuntimeError, match="different row count"):
        evaluate_batch((Sample(1, (0.31, 0.09)),), MissingRowBatch(*problem()))


def test_posterior_requires_nonempty_resolved_predictions():
    with pytest.raises(ValueError, match="at least one evaluation"):
        posterior_weights(EvaluationSet(()))
    for prediction in ((), (np.nan,)):
        with pytest.raises(ValueError, match="successful predictions must be finite"):
            predictive_summary(EvaluationSet((record(1, prediction=prediction),)), root_seed=1)


@pytest.mark.parametrize(
    "component", ["observation_sigma", "surrogate_variance", "discrepancy_variance"]
)
@pytest.mark.parametrize("value", [(-1.0,), (np.inf,), (0.0, 1.0)])
def test_predictive_noise_requires_finite_nonnegative_output_dimensions(component, value):
    with pytest.raises(ValueError, match="noise components must be finite nonnegative"):
        predictive_summary(EvaluationSet((record(1),)), root_seed=1, **{component: value})


def test_predictive_variance_overflow_is_an_explicit_failure():
    evaluations = EvaluationSet((record(1),))
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match="observation variance must be finite"):
            predictive_summary(evaluations, root_seed=1, observation_sigma=(1e308,))
        with pytest.raises(
            ValueError, match="predictive samples and total variance must be finite"
        ):
            predictive_summary(
                evaluations,
                root_seed=1,
                surrogate_variance=(1e308,),
                discrepancy_variance=(1e308,),
            )


def test_conformal_calibration_vectors_must_align():
    with pytest.raises(ValueError, match="target and prediction shapes must match"):
        split_conformal_intervals([1, 2], [0], [0], alpha=0.5)


def test_crps_rejects_nonfinite_truth_and_arithmetic_overflow():
    with pytest.raises(ValueError, match="truth must be finite"):
        empirical_crps([0], np.nan)
    with (
        np.errstate(over="ignore", invalid="ignore"),
        pytest.raises(ValueError, match="CRPS arithmetic must remain finite"),
    ):
        empirical_crps([-1e308, 1e308], -1e308)


@pytest.mark.parametrize(
    "truth,draws,ids",
    [([], [], []), ([[0]], [[[0]]], []), ([[0]], [[[np.nan]]], [1]), ([[0]], [[]], [1])],
)
def test_rank_inputs_require_aligned_finite_trials(truth, draws, ids):
    with pytest.raises(ValueError, match="must align"):
        randomized_ranks(truth, draws, ids, root_seed=1)


@pytest.mark.parametrize("ranks", [[], [[-1]], [[3]], [[1.5]]])
def test_rank_histograms_reject_invalid_discrete_ranks(ranks):
    with pytest.raises(ValueError, match="nonempty integer"):
        rank_histograms(ranks, n_draws=2)
