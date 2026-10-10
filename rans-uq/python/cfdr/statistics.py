"""Small, status-aware statistics for independent native CFD ensembles.

Importance sampling assumes the design is drawn from the prior. Solver failure
is unresolved probability mass, so posterior construction fails until every
evaluation succeeds. Split-conformal intervals require a predictor fitted
independently of exchangeable calibration/test units. Rank histograms are
descriptive per-parameter diagnostics and do not report independence-based
p-values. No GP, MCMC, pooled cross-conformal or KOH implementation is implied.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import _core

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _identifier(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _ids(values: Sequence[int]) -> tuple[int, ...]:
    result = tuple(values)
    for value in result:
        _identifier(value, "sample ID")
    if len(set(result)) != len(result):
        raise ValueError("sample IDs must be unique")
    return result


def sample_rng(root_seed: int, sample_id: int, purpose: str) -> np.random.Generator:
    """PCG64 stream keyed by root seed, scientific ID and an explicit purpose.

    SHA-256 encodes the complete identity without Python's process-randomized
    hash. Decimal integer fields have explicit separators; scheduling and worker
    IDs never enter the stream. Reproduction also records NumPy's version.
    """
    _identifier(root_seed, "root seed")
    _identifier(sample_id, "sample ID")
    if not purpose:
        raise ValueError("RNG purpose must be nonempty")
    identity = f"cfdr-statistics-v1:{root_seed}:{sample_id}:{purpose}"
    digest = hashlib.sha256(identity.encode("utf-8")).digest()
    words = [int.from_bytes(digest[i : i + 4], "little") for i in range(0, 32, 4)]
    seed = np.random.SeedSequence(words)
    return np.random.Generator(np.random.PCG64(seed))


@dataclass(frozen=True)
class Sample:
    sample_id: int
    theta: tuple[float, ...]

    def __post_init__(self) -> None:
        _identifier(self.sample_id, "sample ID")


def independent_uniform_design(
    sample_ids: Sequence[int], lower: ArrayLike, upper: ArrayLike, *, root_seed: int
) -> tuple[Sample, ...]:
    """Independent uniform prior draws; this is not a Latin hypercube."""
    ids = _ids(sample_ids)
    lo, hi = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if (
        lo.ndim != 1
        or lo.size == 0
        or hi.shape != lo.shape
        or not np.isfinite(lo).all()
        or not np.isfinite(hi).all()
        or np.any(hi <= lo)
        or not np.isfinite(hi - lo).all()
    ):
        raise ValueError("uniform bounds must be finite nonempty vectors with lower < upper")
    _identifier(root_seed, "root seed")
    return tuple(
        Sample(i, tuple(float(x) for x in sample_rng(root_seed, i, "prior-design").uniform(lo, hi)))
        for i in ids
    )


@dataclass(frozen=True)
class Evaluation:
    sample_id: int
    theta: tuple[float, ...]
    status: str
    predictions: tuple[float, ...]
    log_likelihood: float | None
    iterations: int
    backend: str
    error: str

    def __post_init__(self) -> None:
        _identifier(self.sample_id, "sample ID")
        _identifier(self.iterations, "iteration count")
        if self.status not in {
            "Converged",
            "Unconverged",
            "InvalidParameters",
            "DivergenceDetected",
            "Unknown",
        }:
            raise ValueError("unrecognized evaluation status")
        if not self.backend:
            raise ValueError("evaluation backend must be explicit")

    @property
    def admitted(self) -> bool:
        return self.status == "Converged" and not self.error


@dataclass(frozen=True)
class EvaluationSet:
    records: tuple[Evaluation, ...]

    def __post_init__(self) -> None:
        _ids(tuple(record.sample_id for record in self.records))

    @property
    def failure_count(self) -> int:
        return sum(not record.admitted for record in self.records)


def _native_record(sample: Sample, result: Any, backend: str, error: str = "") -> Evaluation:
    status = str(result.status.name)
    predictions = tuple(float(x) for x in result.predictions)
    # Invalid native batch rows use NaN placeholders; scalar rows use empty
    # vectors. Both mean there is no prediction, while the failed record stays.
    if status in {"InvalidParameters", "DivergenceDetected", "Unknown"}:
        predictions = ()
    likelihood = float(result.log_lik) if status == "Converged" else None
    if status == "Converged" and (
        not predictions
        or not all(math.isfinite(x) for x in predictions)
        or likelihood is None
        or not math.isfinite(likelihood)
    ):
        status, likelihood, error = "Unknown", None, "nonfinite or missing successful output"
    return Evaluation(
        sample.sample_id,
        sample.theta,
        status,
        predictions,
        likelihood,
        int(result.simple_iters),
        backend,
        error,
    )


def _samples(samples: Sequence[Sample]) -> tuple[Sample, ...]:
    result = tuple(samples)
    _ids(tuple(sample.sample_id for sample in result))
    if result and (
        not result[0].theta or any(len(s.theta) != len(result[0].theta) for s in result)
    ):
        raise ValueError("sample parameter vectors must have equal nonzero lengths")
    return result


def evaluate_scalar(samples: Sequence[Sample], model_factory: Callable[[], Any]) -> EvaluationSet:
    """Evaluate with a fresh native SIMPLE model and cold cache for every ID."""
    records = []
    for sample in _samples(samples):
        model = model_factory()
        if not isinstance(model, _core.ForwardModel):
            raise TypeError("model_factory must construct cfdr._core.ForwardModel")
        model.cache().set_byte_budget(0)
        # Native diagnostic evaluation retains ordinary row errors and status.
        result = model.evaluate_with_diagnostics(sample.theta)
        records.append(
            _native_record(sample, result.result, "cfdr._core.ForwardModel", result.error)
        )
    return EvaluationSet(tuple(records))


def evaluate_batch(samples: Sequence[Sample], batch: Any, *, workers: int = 1) -> EvaluationSet:
    """Run native cold ForwardBatch; native failures remain aligned with IDs."""
    rows = _samples(samples)
    if not isinstance(batch, _core.ForwardBatch):
        raise TypeError("batch must be cfdr._core.ForwardBatch")
    _identifier(workers, "worker count")
    if workers < 1:
        raise ValueError("worker count must be positive")
    if not rows:
        return EvaluationSet(())
    result = batch.evaluate(np.asarray([sample.theta for sample in rows], dtype=float), workers)
    if len(result.rows) != len(rows):
        raise RuntimeError("native batch returned a different row count")
    return EvaluationSet(
        tuple(
            _native_record(sample, native.result, "cfdr._core.ForwardBatch", native.error)
            for sample, native in zip(rows, result.rows, strict=True)
        )
    )


def _resolved(evaluations: EvaluationSet) -> None:
    if not evaluations.records:
        raise ValueError("at least one evaluation is required")
    if evaluations.failure_count:
        raise ValueError(
            f"{evaluations.failure_count} unresolved evaluations; posterior is undefined"
        )


def posterior_weights(evaluations: EvaluationSet) -> FloatArray:
    """Normalized likelihood weights for an independent prior design.

    All rows must be explicitly successful. A finite failure sentinel can
    exceed a valid likelihood, so magnitude is never an admission rule.
    """
    _resolved(evaluations)
    likelihoods = []
    for record in evaluations.records:
        value = record.log_likelihood
        if value is None or not math.isfinite(value):
            raise ValueError("successful log likelihoods must be finite")
        likelihoods.append(value)
    largest = max(likelihoods)
    weights = [math.exp(value - largest) for value in likelihoods]
    return np.asarray(weights, dtype=float) / math.fsum(weights)


def resample_posterior(
    evaluations: EvaluationSet, sample_ids: Sequence[int], *, root_seed: int
) -> tuple[Sample, ...]:
    """IID multinomial draws from the finite importance posterior.

    The canonical input-ID order and per-output-ID streams make resampling
    invariant to input order, output order, chunking and worker assignment.
    """
    ids = _ids(sample_ids)
    ordered = EvaluationSet(tuple(sorted(evaluations.records, key=lambda row: row.sample_id)))
    weights = posterior_weights(ordered)
    _identifier(root_seed, "root seed")
    return tuple(
        Sample(
            i,
            ordered.records[
                int(
                    sample_rng(root_seed, i, "posterior-resample").choice(
                        len(ordered.records), p=weights
                    )
                )
            ].theta,
        )
        for i in ids
    )


@dataclass(frozen=True)
class PredictiveSummary:
    sample_ids: tuple[int, ...]
    samples: tuple[tuple[float, ...], ...]
    latent_mean: tuple[float, ...]
    latent_variance: tuple[float, ...]
    surrogate_variance: tuple[float, ...]
    discrepancy_variance: tuple[float, ...]
    observation_variance: tuple[float, ...]
    total_variance: tuple[float, ...]
    includes_surrogate_noise: bool
    includes_discrepancy: bool
    includes_observation_noise: bool


def predictive_summary(
    evaluations: EvaluationSet,
    *,
    root_seed: int,
    observation_sigma: ArrayLike | None = None,
    surrogate_variance: ArrayLike | None = None,
    discrepancy_variance: ArrayLike | None = None,
) -> PredictiveSummary:
    """Equal-weight latent empirical mixture plus specified Gaussian components.

    Component noise is independent between samples, observations and components;
    this API does not represent correlated GP discrepancy. Total variance uses
    the exact variance of this empirical mixture (ddof=0), not the random noisy
    sample variance. Observation sigma has output units; variance inputs have
    squared output units. Every output labels its included components.
    """
    _resolved(evaluations)
    _identifier(root_seed, "root seed")
    latent = np.asarray([record.predictions for record in evaluations.records], dtype=float)
    if latent.ndim != 2 or latent.shape[1] == 0 or not np.isfinite(latent).all():
        raise ValueError("successful predictions must be finite equal-length nonempty vectors")
    dimension = latent.shape[1]

    def component(value: ArrayLike | None) -> FloatArray:
        result = np.zeros(dimension) if value is None else np.asarray(value, dtype=float)
        if result.shape != (dimension,) or not np.isfinite(result).all() or np.any(result < 0):
            raise ValueError("noise components must be finite nonnegative output-length vectors")
        return result

    observation = component(observation_sigma) ** 2
    surrogate, discrepancy = component(surrogate_variance), component(discrepancy_variance)
    if not np.isfinite(observation).all():
        raise ValueError("observation variance must be finite")
    samples = latent.copy()
    for i, record in enumerate(evaluations.records):
        for purpose, variance in (
            ("surrogate-noise", surrogate),
            ("discrepancy", discrepancy),
            ("observation-noise", observation),
        ):
            if np.any(variance):
                samples[i] += sample_rng(root_seed, record.sample_id, purpose).normal(
                    0, np.sqrt(variance), dimension
                )
    canonical = latent[np.argsort([record.sample_id for record in evaluations.records])]
    mean = canonical.mean(axis=0)
    variance = canonical.var(axis=0, ddof=0)
    total = variance + surrogate + discrepancy + observation
    if not np.isfinite(samples).all() or not np.isfinite(total).all():
        raise ValueError("predictive samples and total variance must be finite")
    return PredictiveSummary(
        tuple(record.sample_id for record in evaluations.records),
        tuple(tuple(float(x) for x in row) for row in samples),
        tuple(mean),
        tuple(variance),
        tuple(surrogate),
        tuple(discrepancy),
        tuple(observation),
        tuple(total),
        bool(np.any(surrogate)),
        bool(np.any(discrepancy)),
        bool(np.any(observation)),
    )


def _finite_vector(values: ArrayLike, name: str) -> FloatArray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or result.size == 0 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite nonempty vector")
    return result


def conformal_quantile(scores: ArrayLike, alpha: float) -> float:
    """The ceil((n+1)(1-alpha))th smallest score, or +inf when k=n+1."""
    values = _finite_vector(scores, "conformal scores")
    if not math.isfinite(alpha) or not 0 < alpha < 1 or np.any(values < 0):
        raise ValueError("alpha must lie in (0,1) and scores must be nonnegative")
    k = math.ceil((values.size + 1) * (1 - alpha))
    return math.inf if k > values.size else float(np.partition(values, k - 1)[k - 1])


def split_conformal_intervals(
    calibration_y: ArrayLike,
    calibration_prediction: ArrayLike,
    test_prediction: ArrayLike,
    *,
    alpha: float,
) -> tuple[FloatArray, FloatArray]:
    """Ordinary absolute-residual split conformal, with marginal coverage >=1-alpha.

    The guarantee is conditional on a predictor trained without calibration/test
    data and exchangeable calibration/test units. The caller owns that split;
    spatial cells or correlated posterior draws are not independent replicates.
    No fold fitting or pooling takes place in this function.
    """
    truth = _finite_vector(calibration_y, "calibration targets")
    fitted = _finite_vector(calibration_prediction, "calibration predictions")
    test = _finite_vector(test_prediction, "test predictions")
    if truth.shape != fitted.shape:
        raise ValueError("calibration target and prediction shapes must match")
    radius = conformal_quantile(np.abs(truth - fitted), alpha)
    return test - radius, test + radius


def empirical_crps(samples: ArrayLike, truth: float) -> float:
    """CRPS of the finite equal-weight empirical forecast, with M squared denominator.

    This is not the fair estimator of an infinite-ensemble score. Sorting avoids
    constructing the quadratic pair-distance array.
    """
    values = np.sort(_finite_vector(samples, "forecast samples"))
    if not math.isfinite(truth):
        raise ValueError("truth must be finite")
    n = values.size
    coefficients = 2 * np.arange(1, n + 1) - n - 1
    centered = values - truth
    answer = float(np.mean(np.abs(centered)) - np.dot(coefficients, centered) / (n * n))
    if not math.isfinite(answer):
        raise ValueError("CRPS arithmetic must remain finite")
    return answer


def randomized_ranks(
    truth: ArrayLike, draws: ArrayLike, trial_ids: Sequence[int], *, root_seed: int
) -> IntArray:
    """Per-parameter ranks: # draws below truth plus uniform integer in [0,# ties]."""
    actual, samples = np.asarray(truth, dtype=float), np.asarray(draws, dtype=float)
    ids = _ids(trial_ids)
    _identifier(root_seed, "root seed")
    if (
        actual.ndim != 2
        or samples.ndim != 3
        or actual.shape[0] != len(ids)
        or samples.shape[0] != len(ids)
        or samples.shape[2] != actual.shape[1]
        or actual.size == 0
        or samples.shape[1] == 0
        or not np.isfinite(actual).all()
        or not np.isfinite(samples).all()
    ):
        raise ValueError("truth [trial,parameter] and draws [trial,draw,parameter] must align")
    ranks: IntArray = np.sum(samples < actual[:, None, :], axis=1, dtype=np.int64)
    ties = np.sum(samples == actual[:, None, :], axis=1)
    for i, sample_id in enumerate(ids):
        for parameter in range(actual.shape[1]):
            ranks[i, parameter] += sample_rng(
                root_seed, sample_id, f"rank-tie:{parameter}"
            ).integers(int(ties[i, parameter]) + 1)
    return ranks


@dataclass(frozen=True)
class RankHistograms:
    counts: IntArray
    expected_counts: FloatArray
    rank_to_bin: IntArray


def rank_histograms(ranks: ArrayLike, *, n_draws: int, bins: int = 20) -> RankHistograms:
    """Descriptive histograms with the exact discrete-uniform bin cardinalities.

    Rows are parameters, not flattened dependent columns. No p-value is supplied:
    uniform marginal ranks do not establish independence of trials or columns.
    """
    _identifier(n_draws, "draw count")
    _identifier(bins, "bin count")
    values = np.asarray(ranks)
    if (
        n_draws < 1
        or bins < 1
        or values.ndim != 2
        or values.size == 0
        or not np.issubdtype(values.dtype, np.integer)
        or np.any(values < 0)
        or np.any(values > n_draws)
    ):
        raise ValueError("ranks must be a nonempty integer [trial,parameter] array in [0,n_draws]")
    bins = min(bins, n_draws + 1)
    mapping = np.arange(n_draws + 1, dtype=np.int64) * bins // (n_draws + 1)
    cardinality = np.bincount(mapping, minlength=bins)
    counts = np.asarray(
        [np.bincount(mapping[values[:, j]], minlength=bins) for j in range(values.shape[1])],
        dtype=np.int64,
    )
    expected = np.tile(cardinality * (values.shape[0] / (n_draws + 1)), (values.shape[1], 1))
    return RankHistograms(counts, expected, mapping)
