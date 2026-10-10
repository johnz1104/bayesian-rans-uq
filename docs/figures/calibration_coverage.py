"""Calibrate SST against direct simulation and test whether its uncertainty is honest.

Calibrates the SST coefficients a1 and beta* against the Lee and Moser (J. Fluid Mech.
774, 2015) direct numerical simulation of a turbulent channel at Re_tau = 182.088, on the
periodic channel used by the grid-convergence test in rans-uq/tests. Thirty points
across the channel are split into thirds: one third enters the likelihood, one third
calibrates a split-conformal correction, and one third is held out to test the 90%
ranges. The likelihood noise is the simulation's own statistical uncertainty of the mean.
Because that noise is tiny, the likelihood is sharply peaked: a coarse uniform design
locates its region, a pattern search finds the best fit, and the posterior is formed from
uniform prior draws in a small box centered there, checked to hold the best draw inside.

Download LM_Channel_0180_mean_prof.dat and LM_Channel_0180_mean_stdev.dat from
https://turbulence.oden.utexas.edu (they are not distributed with this repository).
The script writes the main plot (calibration_coverage_*.png), a supporting image
(calibration_support_*.png) and the predictive-distribution figure
(calibration_distributions_*.png, plus a PDF and SVG of the light version), each in a
light and a dark version. That figure draws each point's exact predictive density, a
mixture of the posterior draws blurred by the Gaussian measurement noise, and shows the
split-conformal intervals as intervals, never as densities. The solves take about an
hour on two threads; --cache keeps their results so the images can be redrawn without
solving again. Requires cfdr, NumPy and matplotlib:

    python docs/figures/calibration_coverage.py LM_Channel_0180_mean_prof.dat \
        LM_Channel_0180_mean_stdev.dat docs/figures --cache calibration.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator

from cfdr import _core as c
from cfdr import statistics as stats

RE_TAU = 182.088
HALF_CELLS = 64
SEED = 20261010
ALPHA = 0.1
WORKERS = 2
STAGE1_BOX = ((0.28, 0.084), (0.34, 0.096))  # (a1, beta*) lower and upper corners
SEARCH_STEP = (0.004, 0.001)  # first pattern-search step; halved three times

# Categorical slots 1-3 of the validated reference palette with its ink and chrome tokens.
THEMES = {
    "light": {
        "surface": "#fcfcfb", "ink": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
        "grid": "#e1e0d9", "axis": "#c3c2b7", "border": "#0b0b0b1a",
        "reference": "#2a78d6", "model": "#eb6834", "recalibrated": "#1baf7a",
    },
    "dark": {
        "surface": "#1a1a19", "ink": "#ffffff", "secondary": "#c3c2b7", "muted": "#898781",
        "grid": "#2c2c2a", "axis": "#383835", "border": "#ffffff1a",
        "reference": "#3987e5", "model": "#d95926", "recalibrated": "#199e70",
    },
}  # fmt: skip


def stations(y_plus: np.ndarray) -> np.ndarray:
    """Indices of 30 distinct simulation grid points spread logarithmically over y+ 1-175."""
    candidates = [int(i) for i in np.flatnonzero((y_plus >= 1.0) & (y_plus <= 175.0))]
    picked: list[int] = []
    for target in np.geomspace(1.0, 175.0, 30):
        free = [i for i in candidates if i not in picked]
        picked.append(min(free, key=lambda i: abs(math.log(y_plus[i] / target))))
    return np.asarray(sorted(picked))


def forward_batch(y: np.ndarray, truth: np.ndarray, sigma: np.ndarray) -> c.ForwardBatch:
    """Native batch on the verified periodic channel; sigma[i] = inf drops point i."""
    first_height = 1 + math.tanh(2 * (1 / HALF_CELLS - 1)) / math.tanh(2)
    mesh = c.Mesh.make_periodic_channel_2d(
        3, 2 * HALF_CELLS, 1.0, 1.0, RE_TAU, first_height * RE_TAU
    )
    mesh.compute_wall_distance()
    observations = c.ObservationOperator()
    for yi, ui, si in zip(y, truth, sigma, strict=True):
        # Points outside the likelihood get a huge sigma: their term is constant.
        observations.add_velocity_profile(
            c.Vec3(0.5, float(yi), 0.0), 0, float(ui), float(si) if np.isfinite(si) else 1e6
        )
    settings = c.SolverSettings()
    settings.max_iterations = 150000
    settings.convergence_tol = 1e-8
    settings.alpha_u = 0.5
    settings.body_force = c.Vec3(1.0, 0.0, 0.0)
    settings.verbose = False
    boundaries = c.FlowBoundaryConditions.channel_defaults(mesh, 1.0, 1.0, 10.0)
    parameters = c.InferenceParameterSet.a1_betaStar()
    return c.ForwardBatch(
        mesh,
        parameters,
        observations,
        boundaries,
        1 / RE_TAU,
        settings,
        c.Vec3(15.0, 0.0, 0.0),
        0.0,
        1.0,
        10.0,
    )


def evaluate(batch: c.ForwardBatch, samples: tuple[stats.Sample, ...]) -> stats.EvaluationSet:
    evaluations = stats.evaluate_batch(samples, batch, workers=WORKERS)
    if evaluations.failure_count:
        raise RuntimeError(f"{evaluations.failure_count} solves failed; the posterior is undefined")
    return evaluations


def pattern_search(
    batch: c.ForwardBatch, start: tuple[float, ...], start_log_likelihood: float
) -> tuple[np.ndarray, np.ndarray, list]:
    """Move a 3x3 stencil to its best point; halve the step when the center wins (3 times)."""
    center, best, step = np.asarray(start), start_log_likelihood, np.asarray(SEARCH_STEP)
    path, halvings = [[*center, best]], 0
    while halvings < 3:
        offsets = [(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1) if (i, j) != (0, 0)]
        rows = tuple(stats.Sample(k, tuple(center + step * o)) for k, o in enumerate(offsets))
        winner = max(evaluate(batch, rows).records, key=lambda r: r.log_likelihood)
        if winner.log_likelihood > best:
            center, best = np.asarray(winner.theta), winner.log_likelihood
            path.append([*center, best])
        else:
            step, halvings = step / 2, halvings + 1
    return center, step, path


def calibrate(mean_file: Path, stdev_file: Path) -> dict:
    """Run both design stages and every statistic the figure shows."""
    mean = np.loadtxt(mean_file, comments="%")
    stdev = np.loadtxt(stdev_file, comments="%")
    if not np.allclose(mean[:, 1], stdev[:, 1]):
        raise ValueError("mean and uncertainty files must share the wall-normal grid")
    index = stations(mean[:, 1])
    y_plus, truth, sigma = mean[index, 1], mean[index, 2], stdev[index, 2]
    group = np.arange(30) % 3  # 0: likelihood, 1: conformal calibration, 2: held out
    likelihood_sigma = np.where(group == 0, sigma, np.inf)
    batch = forward_batch(y_plus / RE_TAU, truth, likelihood_sigma)

    lower, upper = STAGE1_BOX
    stage1 = evaluate(batch, stats.independent_uniform_design(range(24), lower, upper,
                                                              root_seed=SEED))  # fmt: skip
    first = max(stage1.records, key=lambda r: r.log_likelihood)
    optimum, step, path = pattern_search(batch, first.theta, first.log_likelihood)
    lower2, upper2 = tuple(optimum - 2 * step), tuple(optimum + 2 * step)
    stage2 = evaluate(batch, stats.independent_uniform_design(range(100, 148), lower2, upper2,
                                                              root_seed=SEED + 1))  # fmt: skip

    weights = stats.posterior_weights(stage2)
    theta = np.array([r.theta for r in stage2.records])
    span = np.asarray(upper2) - np.asarray(lower2)
    top_draw = theta[int(np.argmax(weights))]
    top_on_edge = bool(np.any((top_draw - lower2 < 0.1 * span) | (upper2 - top_draw < 0.1 * span)))
    draws = stats.resample_posterior(stage2, range(1000, 1400), root_seed=SEED + 2)
    # Each draw repeats an evaluated coefficient vector; the solver is deterministic,
    # so its predictions are reused instead of solved again.
    by_theta = {r.theta: r for r in stage2.records}
    posterior = stats.EvaluationSet(
        tuple(
            stats.Evaluation(d.sample_id, d.theta, "Converged", by_theta[d.theta].predictions,
                             by_theta[d.theta].log_likelihood, by_theta[d.theta].iterations,
                             by_theta[d.theta].backend, "")
            for d in draws
        )
    )  # fmt: skip
    predictive = stats.predictive_summary(posterior, root_seed=SEED + 3, observation_sigma=sigma)
    samples = np.asarray(predictive.samples)
    low, high = np.quantile(samples, [ALPHA / 2, 1 - ALPHA / 2], axis=0)
    prediction = np.asarray(predictive.latent_mean)

    # Split conformal on the log scale, so the correction is a relative one.
    cal, test = group == 1, group == 2
    log_lo, log_hi = stats.split_conformal_intervals(
        np.log(truth[cal]), np.log(prediction[cal]), np.log(prediction), alpha=ALPHA
    )
    return {
        "y_plus": y_plus.tolist(), "truth": truth.tolist(), "sigma": sigma.tolist(),
        "group": group.tolist(), "prediction": prediction.tolist(),
        "standard_low": low.tolist(), "standard_high": high.tolist(),
        "recalibrated_low": np.exp(log_lo).tolist(), "recalibrated_high": np.exp(log_hi).tolist(),
        "latent_variance": list(predictive.latent_variance),
        "observation_variance": list(predictive.observation_variance),
        "standard_covered": int(np.sum((truth >= low) & (truth <= high) & test)),
        "recalibrated_covered": int(np.sum((truth >= np.exp(log_lo)) & (truth <= np.exp(log_hi))
                                           & test)),
        "held_out": int(test.sum()),
        "effective_sample_size": float(1 / np.sum(weights**2)),
        "max_weight_draw_on_box_edge": top_on_edge,
        "search_path": path,
        "stage1_box": [list(lower), list(upper)], "stage2_box": [list(lower2), list(upper2)],
        "posterior_mean_theta": (weights @ theta).tolist(),
        "max_weight_theta": list(theta[int(np.argmax(weights))]),
        "solves": len(stage1.records) + 8 * (len(path) - 1 + 3) + len(stage2.records),
        "dns_profile": mean[(mean[:, 1] >= 0.5) & (mean[:, 1] < RE_TAU)][:, 1:3].tolist(),
        "center_speed": float(mean[np.argmax(mean[:, 1]), 2]),
        # The predictive distribution itself: each posterior draw's noise-free prediction
        # (blurred by Gaussian measurement noise of standard deviation sigma) and the noisy
        # predictive samples whose quantiles give the standard 90% ranges.
        "posterior_latent": [list(record.predictions) for record in posterior.records],
        "predictive_samples": samples.tolist(),
    }  # fmt: skip


def derived(r: dict) -> dict:
    """Arrays and summary numbers both images draw from."""
    truth, prediction = np.asarray(r["truth"]), np.asarray(r["prediction"])
    std = np.stack([r["standard_low"], r["standard_high"]])
    rec = np.stack([r["recalibrated_low"], r["recalibrated_high"]])
    return {
        "share": np.asarray(r["y_plus"]) / RE_TAU, "truth": truth, "prediction": prediction,
        "test": np.flatnonzero(np.asarray(r["group"]) == 2),
        "std_rel": 100 * (std - truth) / truth, "rec_rel": 100 * (rec - truth) / truth,
        "std_in": (truth >= std[0]) & (truth <= std[1]),
        "rec_in": (truth >= rec[0]) & (truth <= rec[1]),
        "noise": 100 * float(np.median(np.asarray(r["sigma"]) / truth)),
        "model_error": 100 * float(np.max(np.abs(prediction - truth) / truth)),
    }  # fmt: skip


def styled_figure(t: dict, size: tuple[float, float]) -> plt.Figure:
    plt.rcParams.update({
        "font.family": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 12,
        "text.color": t["ink"], "axes.labelcolor": t["secondary"],
        "xtick.color": t["muted"], "ytick.color": t["muted"],
        "xtick.labelcolor": t["secondary"], "ytick.labelcolor": t["secondary"],
        "xtick.labelsize": 11, "ytick.labelsize": 11,
    })  # fmt: skip
    fig = plt.figure(figsize=size)
    fig.patch.set_facecolor(t["surface"])
    return fig


def chart(fig: plt.Figure, t: dict, rect: tuple[float, float, float, float]) -> plt.Axes:
    ax = fig.add_axes(rect)
    ax.set_facecolor(t["surface"])
    ax.grid(True, color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for edge in ("top", "right"):
        ax.spines[edge].set_visible(False)
    for edge in ("left", "bottom"):
        ax.spines[edge].set_color(t["axis"])
    ax.tick_params(length=0, pad=5)
    return ax


def canvas(fig: plt.Figure, rect: tuple[float, float, float, float], xlim, ylim) -> plt.Axes:
    ax = fig.add_axes(rect)
    ax.set_axis_off()
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    return ax


def truth_formatter(v: float, _: object) -> str:
    return "truth" if abs(v) < 1e-9 else f"{v:+.0f}%".replace("-", "−")


def render_main(r: dict, theme: str, output: Path) -> None:
    """The one plot: each nominal 90% range at the ten test points against the truth."""
    t, d = THEMES[theme], derived(r)
    order, n = d["test"], len(d["test"])
    fig = styled_figure(t, (10, 6.0))
    fig.text(0.035, 0.955, "A close fit doesn't mean reliable uncertainty", fontsize=21,
             fontweight="bold", va="top")  # fmt: skip
    fig.text(0.035, 0.885, "Each bar is a range the turbulence model says it is “90% sure” "
             "contains the true flow speed at one point in a channel.\nA model that knew its own "
             "accuracy would be right about 9 times in 10.", fontsize=12,
             color=t["secondary"], va="top", linespacing=1.4)  # fmt: skip
    ax = chart(fig, t, (0.11, 0.19, 0.86, 0.6))
    for k, i in enumerate(order):
        ax.plot([k - 0.16] * 2, d["std_rel"][:, i], color=t["model"], linewidth=4.5,
                solid_capstyle="round")  # fmt: skip
        ax.plot([k + 0.16] * 2, d["rec_rel"][:, i], color=t["recalibrated"], linewidth=7,
                solid_capstyle="round")  # fmt: skip
    ax.axhline(0, color=t["reference"], linewidth=2.4)
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(
        min(d["rec_rel"][0, order].min(), -10) - 1.5, max(d["rec_rel"][1, order].max(), 5) + 8.5
    )
    ax.yaxis.set_major_locator(FixedLocator([-10, -5, 0, 5]))
    ax.yaxis.set_major_formatter(truth_formatter)
    ax.set_xticks(range(n))
    ax.set_xticklabels([f"{100 * d['share'][i]:.2g}%" for i in order])
    ax.grid(False, axis="x")
    ax.set_xlabel("Test point: distance from the wall, as a share of the way to the center line")
    ax.set_ylabel("Claimed range vs. true speed")
    # Direct labels, which double as the legend.
    top = ax.get_ylim()[1]
    std_hits, rec_hits = int(d["std_in"][order].sum()), int(d["rec_in"][order].sum())
    for y, color, width, text in (
        (top - 1.8, t["model"], 4.5,
         f"Standard calibration: right {std_hits} time{'s' if std_hits != 1 else ''} in {n}"),
        (top - 4.6, t["recalibrated"], 7,
         f"Recalibrated (widened using held-back data): right {rec_hits} times in {n}"),
    ):  # fmt: skip
        ax.plot([-0.45, -0.13], [y] * 2, color=color, linewidth=width, solid_capstyle="round")
        ax.text(0.0, y, text, va="center", fontsize=12, color=t["ink"])
    # Two callouts teach how to read a bar: one that misses, one that contains the truth.
    # The miss nearest the center line: its pointer reaches it without crossing a bar.
    miss = max(k for k, i in enumerate(order) if not d["std_in"][i])
    miss_i = order[miss]
    callout = {"fontsize": 11, "color": t["ink"], "va": "center",
               "arrowprops": {"arrowstyle": "-", "color": t["muted"], "lw": 0.9}}  # fmt: skip
    ax.annotate("Misses: the true speed\nis outside this range",
                xy=(miss - 0.16, d["std_rel"][0, miss_i]), xytext=(miss - 1.75, -8.6),
                **callout)  # fmt: skip
    hit_i = order[0]
    ax.annotate("Contains the true speed", xy=(0.16, d["rec_rel"][0, hit_i] + 0.4),
                xytext=(0.45, -8.6), **callout)  # fmt: skip
    fig.text(0.035, 0.04, "Test points were used for neither fitting nor recalibration. "
             "Ten correlated points cannot establish general 90% coverage.", fontsize=10.5,
             color=t["secondary"], va="center")  # fmt: skip
    fig.savefig(output, dpi=170, facecolor=t["surface"])
    plt.close(fig)


def render_support(r: dict, theme: str, output: Path) -> None:
    """Supporting panels: the fit, coverage counts, and noise against model error."""
    t, d = THEMES[theme], derived(r)
    order, n = d["test"], len(d["test"])
    center = r["center_speed"]
    fig = styled_figure(t, (12.8, 4.6))
    for x, label in ((0.045, "The fitted model looks accurate"),
                     (0.38, "How often the “90% sure” ranges are right"), (0.79, "Why the ranges miss")):  # fmt: skip
        fig.text(x, 0.93, label, fontsize=15, fontweight="bold", va="center")

    profile = chart(fig, t, (0.075, 0.17, 0.235, 0.64))
    dns = np.asarray(r["dns_profile"])
    line = {"linewidth": 2.4, "solid_capstyle": "round", "dash_capstyle": "round"}
    profile.plot(dns[:, 0] / RE_TAU, 100 * dns[:, 1] / center, color=t["reference"],
                 label="Direct simulation", **line)  # fmt: skip
    profile.plot(d["share"], 100 * d["prediction"] / center, color=t["model"],
                 linestyle=(0, (5, 2.2)), label="Calibrated SST", **line)  # fmt: skip
    profile.plot(d["share"][order], 100 * d["truth"][order] / center, "o", markersize=7,
                 markerfacecolor=t["surface"], markeredgecolor=t["reference"],
                 markeredgewidth=1.8, linestyle="none", label="Test points", zorder=5)  # fmt: skip
    profile.set_xscale("log")
    profile.set_xlim(4e-3, 1.1)
    profile.set_ylim(0, 108)
    profile.xaxis.set_major_locator(FixedLocator([1e-2, 1e-1, 1]))
    profile.xaxis.set_minor_locator(NullLocator())
    profile.xaxis.set_minor_formatter(NullFormatter())
    profile.set_xticklabels(["1%", "10%", "100%"])
    profile.yaxis.set_major_locator(FixedLocator([0, 50, 100]))
    profile.set_yticklabels(["0%", "50%", "100%"])
    profile.set_xlabel("Distance from the wall")
    profile.set_ylabel("Flow speed")
    profile.legend(loc="lower right", frameon=False, fontsize=10.5, handlelength=1.8,
                   labelcolor=t["ink"], borderaxespad=0.1)  # fmt: skip

    dots = canvas(fig, (0.445, 0.2, 0.235, 0.6), (-0.6, n - 0.4), (-1.1, 2.0))
    for y, color, inside, name in ((1.25, t["model"], d["std_in"], "Standard"),
                                   (0.25, t["recalibrated"], d["rec_in"], "Recalibrated")):  # fmt: skip
        for k, i in enumerate(order):
            dots.plot(k, y, "o", markersize=14, markeredgewidth=2.2, color=color,
                      markerfacecolor=color if inside[i] else t["surface"])  # fmt: skip
        dots.text(-0.9, y, name, ha="right", va="center", fontsize=12, color=t["ink"])
        dots.text(n - 0.1, y, f"{int(inside[order].sum())}/{n}", ha="left", va="center",
                  fontsize=24, fontweight="bold")  # fmt: skip
    dots.text(-0.6, -0.75, "Filled: contains the true speed.  Hollow: misses it.\nA model that knew its accuracy would score about 9/10.",
              fontsize=10.5, color=t["secondary"], va="center")  # fmt: skip

    error, noise = d["model_error"], d["noise"]
    why = canvas(fig, (0.79, 0.17, 0.19, 0.62), (-0.75, 2.05), (0, error * 1.32))
    for x, value, color, label in ((0, noise, t["reference"], "assumed\ndata noise"),
                                   (1.35, error, t["model"], "model's own\nlargest error")):  # fmt: skip
        height = max(value, error * 0.012)
        why.add_patch(FancyBboxPatch((x - 0.3, 0), 0.6, height,
                                     boxstyle="round,pad=0,rounding_size=0.06",
                                     facecolor=color, edgecolor="none",
                                     mutation_aspect=error / 2.2))  # fmt: skip
        why.text(x, height + error * 0.04, f"{value:.2g}%", ha="center", va="bottom",
                 fontsize=22, fontweight="bold")  # fmt: skip
        why.text(x, -error * 0.04, label, ha="center", va="top", fontsize=11.5,
                 color=t["secondary"], linespacing=1.3)  # fmt: skip
    why.text(0, error * 0.66, f"{error / noise:.0f}×", ha="center", va="center", fontsize=30,
             fontweight="bold")  # fmt: skip
    why.text(0, error * 0.54, "larger than the\nassumed noise", ha="center", va="top",
             fontsize=11, color=t["secondary"], linespacing=1.3)  # fmt: skip
    fig.savefig(output, dpi=150, facecolor=t["surface"])
    plt.close(fig)


def predictive_density(r: dict, i: int, values: np.ndarray) -> np.ndarray:
    """Exact predictive density at point i: posterior draws blurred by Gaussian noise."""
    latent = np.asarray(r["posterior_latent"])[:, i]
    sigma = np.asarray(r["sigma"])[i]
    z = (values[:, None] - latent[None, :]) / sigma
    return np.exp(-0.5 * z**2).mean(axis=1) / (sigma * np.sqrt(2 * np.pi))


def check_against_samples(r: dict) -> None:
    """The standard 90% ranges must be the 5%/95% quantiles of the stored samples."""
    samples = np.asarray(r["predictive_samples"])
    low, high = np.quantile(samples, [ALPHA / 2, 1 - ALPHA / 2], axis=0)
    if not (np.allclose(low, r["standard_low"]) and np.allclose(high, r["standard_high"])):
        raise ValueError("stored ranges do not match the stored predictive samples")


def render_distributions(r: dict, theme: str, outputs: list[Path]) -> None:
    """Bayesian predictive distributions against the truth, and the conformal intervals."""
    check_against_samples(r)
    t, d = THEMES[theme], derived(r)
    order, n = d["test"], len(d["test"])
    truth, prediction = d["truth"], d["prediction"]
    center = r["center_speed"]
    rel = lambda v, i: 100 * (v - truth[i]) / truth[i]
    fig = styled_figure(t, (13.5, 7.4))
    fig.text(0.03, 0.955, "The model was highly certain, and missed the truth", fontsize=21,
             fontweight="bold", va="top")  # fmt: skip
    fig.text(0.03, 0.89, "Calibrated SST predictions at ten test points across a turbulent "
             "channel, compared with direct simulation (the truth). Recalibration\nwidens the "
             "90% range; it does not move the predictions or change their probability "
             "distributions.", fontsize=12, color=t["secondary"], va="top",
             linespacing=1.4)  # fmt: skip
    heading = {"loc": "left", "fontsize": 13, "fontweight": "bold", "pad": 8}
    a = chart(fig, t, (0.075, 0.5, 0.5, 0.27))
    b = chart(fig, t, (0.075, 0.15, 0.5, 0.27))
    c = chart(fig, t, (0.64, 0.15, 0.33, 0.62))

    # A: the actual predictive densities as violins, on one shared density scale.
    grids, dens = {}, {}
    for i in order:
        sd = np.sqrt(r["latent_variance"][i] + r["observation_variance"][i])
        grid = np.linspace(prediction[i] - 6 * sd, prediction[i] + 6 * sd, 400)
        grids[i], dens[i] = grid, predictive_density(r, i, grid) * truth[i] / 100
    scale = 0.38 / max(dens[i].max() for i in order)
    for k, i in enumerate(order):
        y = rel(grids[i], i)
        a.fill_betweenx(y, k - scale * dens[i], k + scale * dens[i], color=t["model"],
                        alpha=0.35, linewidth=0)  # fmt: skip
        a.plot([k, k], [rel(r["standard_low"][i], i), rel(r["standard_high"][i], i)],
               color=t["model"], linewidth=3, solid_capstyle="butt")  # fmt: skip
    # B: the same predictions with the conformal intervals computed at each test point.
    lo = np.array([rel(r["recalibrated_low"][i], i) for i in order])
    hi = np.array([rel(r["recalibrated_high"][i], i) for i in order])
    b.fill_between(range(n), lo, hi, color=t["recalibrated"], alpha=0.13, linewidth=0)
    b.plot(range(n), lo, color=t["recalibrated"], linewidth=1, alpha=0.6)
    b.plot(range(n), hi, color=t["recalibrated"], linewidth=1, alpha=0.6)
    for k in range(n):
        b.plot([k, k], [lo[k], hi[k]], color=t["recalibrated"], linewidth=5,
               solid_capstyle="round")  # fmt: skip
    for ax, covered, label in (
        (a, int(d["std_in"][order].sum()), "A   Original Bayesian 90% ranges"),
        (b, int(d["rec_in"][order].sum()), "B   Same predictions, recalibrated 90% ranges"),
    ):  # fmt: skip
        means = [rel(prediction[i], i) for i in order]
        ax.plot(range(n), means, "o", color=t["model"], markersize=6, markeredgecolor=t["surface"],
                markeredgewidth=1.5, zorder=5)  # fmt: skip
        ax.axhline(0, color=t["reference"], linewidth=2.2, zorder=4)
        ax.set_title(label, **heading)
        ax.text(0.99, 0.95, f"Observed: {covered} of {n} contain the truth (nominal 90%)",
                transform=ax.transAxes, ha="right", va="top", fontsize=11.5,
                color=t["ink"])  # fmt: skip
        ax.set_xlim(-0.6, n - 0.4)
        ax.set_ylim(-11.5, 12.5)
        ax.yaxis.set_major_locator(FixedLocator([-10, -5, 0, 5]))
        ax.yaxis.set_major_formatter(truth_formatter)
        ax.set_xticks(range(n))
        ax.set_xticklabels([f"{100 * d['share'][i]:.2g}%" for i in order])
        ax.grid(False, axis="x")
        ax.set_ylabel("Difference from\nthe true speed")
    a.tick_params(labelbottom=False)
    b.set_xlabel("Test point: distance from the wall, as a share of the way to the center line")
    b.text(0.01, 0.95, "Shading between points is for readability;\nintervals were computed "
           "only at the ten test points.", transform=b.transAxes, fontsize=10, va="top",
           color=t["secondary"])  # fmt: skip
    k = int(np.argmin([rel(prediction[i], i) for i in order]))
    a.annotate("Each sliver is the model's whole probability distribution:\nvery narrow, and "
               "mostly off the truth", xy=(k - 0.3, rel(prediction[order[k]], order[k])),
               xytext=(-0.45, -9.6), fontsize=10.5, color=t["ink"], va="center",
               arrowprops={"arrowstyle": "-", "color": t["ink"], "lw": 0.8})  # fmt: skip

    # C: one test point where the Bayesian range misses and the conformal one contains it,
    # chosen as the clearest case: the largest of min(miss distance, conformal margin).
    std_lo, std_hi = np.asarray(r["standard_low"]), np.asarray(r["standard_high"])
    rec_lo, rec_hi = np.asarray(r["recalibrated_low"]), np.asarray(r["recalibrated_high"])
    candidates = [i for i in order if not d["std_in"][i] and d["rec_in"][i]]
    clarity = lambda i: min(
        min(abs(truth[i] - std_lo[i]), abs(truth[i] - std_hi[i])),
        min(truth[i] - rec_lo[i], rec_hi[i] - truth[i]),
    )
    j = max(candidates, key=clarity)
    to_pct = 100 / center
    span = (rec_lo[j] * to_pct, rec_hi[j] * to_pct)
    margin = 0.25 * (span[1] - span[0])
    xs = np.linspace(span[0] - margin, span[1] + margin, 2000)
    density = predictive_density(r, j, xs / to_pct) / to_pct
    peak = density.max()
    c.axvspan(*span, color=t["recalibrated"], alpha=0.12, linewidth=0, zorder=1)
    for edge in span:
        c.axvline(edge, color=t["recalibrated"], linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
    c.fill_between(xs, density, color=t["model"], alpha=0.3, linewidth=0, zorder=3)
    c.plot(xs, density, color=t["model"], linewidth=2, zorder=4)
    c.axvline(truth[j] * to_pct, color=t["reference"], linewidth=2.2, linestyle=(0, (5, 3)),
              zorder=5)  # fmt: skip
    width = xs[-1] - xs[0]
    std_span = (std_lo[j] * to_pct, std_hi[j] * to_pct)
    c.plot(std_span, [-0.06 * peak] * 2, color=t["model"], linewidth=5, solid_capstyle="butt")
    c.text(std_span[1] + 0.015 * width, -0.06 * peak, "Original 90% Bayesian interval",
           va="center", fontsize=10.5, color=t["ink"])  # fmt: skip
    c.plot(span, [-0.25 * peak] * 2, color=t["recalibrated"], linewidth=5, solid_capstyle="butt")
    c.text(sum(span) / 2, -0.16 * peak, "Recalibrated 90% conformal interval\n(shaded above: "
           "a range, not a probability curve)", ha="center", va="center", fontsize=10.5,
           color=t["ink"], linespacing=1.3,
           bbox={"boxstyle": "round,pad=0.25", "facecolor": t["surface"], "edgecolor": "none",
                 "alpha": 0.85})  # fmt: skip
    c.set_ylim(-0.3 * peak, 1.12 * peak)
    c.set_xlim(xs[0], xs[-1])
    c.annotate("High confidence,\nincorrect\nprediction", xy=(prediction[j] * to_pct, 0.75 * peak),
               xytext=(prediction[j] * to_pct + 0.05 * width, 0.85 * peak), fontsize=11,
               color=t["ink"], va="center",
               arrowprops={"arrowstyle": "-", "color": t["ink"], "lw": 0.9})  # fmt: skip
    c.text(truth[j] * to_pct, 1.05 * peak, "Observed truth", ha="center", fontsize=11,
           color=t["ink"])  # fmt: skip
    c.set_title("C   One test point up close", **heading)
    c.set_xlabel(f"Flow speed at the test point {100 * d['share'][j]:.0f}% of the way to the "
                 "center line\n(% of the center-line speed)")  # fmt: skip
    c.set_ylabel("Probability density (Bayesian prediction)")
    c.yaxis.set_major_locator(FixedLocator([0]))
    c.yaxis.set_major_formatter(lambda v, _: "0")
    c.grid(False, axis="y")

    fig.text(0.03, 0.035, "Coverage shown is observed on these ten test points. The conformal 90% "
             "guarantee is marginal and assumes exchangeable points; points along one channel "
             "are correlated, so this sample cannot confirm it.", fontsize=10,
             color=t["secondary"], va="center")  # fmt: skip
    for output in outputs:
        fig.savefig(output, dpi=200, facecolor=t["surface"])
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mean", type=Path, help="LM_Channel_0180_mean_prof.dat")
    parser.add_argument("stdev", type=Path, help="LM_Channel_0180_mean_stdev.dat")
    parser.add_argument("output_dir", type=Path, help="directory for the two images")
    parser.add_argument("--cache", type=Path, help="JSON file that stores or reuses the solves")
    args = parser.parse_args()

    if args.cache and args.cache.exists():
        result = json.loads(args.cache.read_text())
    else:
        result = calibrate(args.mean, args.stdev)
        if args.cache:
            args.cache.write_text(json.dumps(result, indent=1))
    for theme in THEMES:
        render_main(result, theme, args.output_dir / f"calibration_coverage_{theme}.png")
        render_support(result, theme, args.output_dir / f"calibration_support_{theme}.png")
        if "posterior_latent" in result:
            outputs = [args.output_dir / f"calibration_distributions_{theme}.png"]
            if theme == "light":
                outputs += [args.output_dir / "calibration_distributions.pdf",
                            args.output_dir / "calibration_distributions.svg"]  # fmt: skip
            render_distributions(result, theme, outputs)

    keys = ("solves", "effective_sample_size", "max_weight_draw_on_box_edge", "standard_covered",
            "recalibrated_covered", "held_out", "posterior_mean_theta", "stage2_box")  # fmt: skip
    for key in keys:
        print(f"{key}: {result[key]}")
    truth, test = np.asarray(result["truth"]), np.asarray(result["group"]) == 2
    share = 100 * np.asarray(result["y_plus"]) / RE_TAU
    print("share | truth U+ | standard 90% range | recalibrated 90% range | coefficient var share")
    for i in np.flatnonzero(test):
        latent, obs = result["latent_variance"][i], result["observation_variance"][i]
        print(f"{share[i]:.1f}% | {truth[i]:.3f} | {result['standard_low'][i]:.3f}-"
              f"{result['standard_high'][i]:.3f} | {result['recalibrated_low'][i]:.3f}-"
              f"{result['recalibrated_high'][i]:.3f} | {latent / (latent + obs):.2f}")  # fmt: skip


if __name__ == "__main__":
    main()
