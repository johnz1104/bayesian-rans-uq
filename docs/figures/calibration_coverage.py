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
The solves take about half an hour on two threads; --cache keeps their results so the
figure can be redrawn without solving again. Requires cfdr, NumPy and matplotlib:

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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
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
        "grid": "#e1e0d9", "axis": "#c3c2b7",
        "reference": "#2a78d6", "model": "#eb6834", "recalibrated": "#1baf7a",
    },
    "dark": {
        "surface": "#1a1a19", "ink": "#ffffff", "secondary": "#c3c2b7", "muted": "#898781",
        "grid": "#2c2c2a", "axis": "#383835",
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
    }  # fmt: skip


def render(r: dict, theme: str, output: Path) -> None:
    t = THEMES[theme]
    share = np.asarray(r["y_plus"]) / RE_TAU
    truth, prediction = np.asarray(r["truth"]), np.asarray(r["prediction"])
    center = r["center_speed"]
    test = np.asarray(r["group"]) == 2
    std_lo, std_hi = np.asarray(r["standard_low"]), np.asarray(r["standard_high"])
    rec_lo, rec_hi = np.asarray(r["recalibrated_low"]), np.asarray(r["recalibrated_high"])

    plt.rcParams.update({
        "font.family": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 11,
        "text.color": t["ink"], "axes.labelcolor": t["secondary"],
        "xtick.color": t["muted"], "ytick.color": t["muted"],
        "xtick.labelcolor": t["secondary"], "ytick.labelcolor": t["secondary"],
    })  # fmt: skip
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(8, 7.4), sharex=True, gridspec_kw={"height_ratios": [1.9, 1.3]}
    )
    fig.patch.set_facecolor(t["surface"])
    for ax in (top, bottom):
        ax.set_facecolor(t["surface"])
        ax.grid(True, color=t["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(t["axis"])
        ax.tick_params(length=0, pad=6)
    line = {"linewidth": 2.2, "solid_capstyle": "round", "solid_joinstyle": "round"}

    dns = np.asarray(r["dns_profile"])
    top.fill_between(share, 100 * rec_lo / center, 100 * rec_hi / center,
                     color=t["recalibrated"], alpha=0.16, linewidth=0)  # fmt: skip
    top.plot(dns[:, 0] / RE_TAU, 100 * dns[:, 1] / center, color=t["reference"], **line)
    top.plot(share, 100 * prediction / center, color=t["model"], **line)
    top.plot(share[test], 100 * truth[test] / center, "o", markersize=8,
             markerfacecolor=t["surface"], markeredgecolor=t["reference"],
             markeredgewidth=2, zorder=5)  # fmt: skip
    top.set_ylim(0, 108)
    top.yaxis.set_major_locator(FixedLocator([0, 25, 50, 75, 100]))
    top.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    top.set_ylabel("Average flow speed\n(share of reference center-line speed)")
    top.legend(
        handles=[
            Line2D([], [], color=t["reference"], **line, label="Reference: direct simulation"),
            Line2D([], [], color=t["model"], **line, label="Calibrated model"),
            Patch(color=t["recalibrated"], alpha=0.35, label="Recalibrated 90% range"),
            Line2D([], [], marker="o", linestyle="none", markersize=8,
                   markerfacecolor=t["surface"], markeredgecolor=t["reference"],
                   markeredgewidth=2, label="Held-out test points"),
        ],
        loc="upper left", frameon=False, labelcolor=t["ink"], handlelength=1.6,
    )  # fmt: skip

    # Each held-out point: where each claimed 90% range lies relative to the true speed.
    offset = 1.09
    std_rel = 100 * (np.stack([std_lo, std_hi]) - truth) / truth
    rec_rel = 100 * (np.stack([rec_lo, rec_hi]) - truth) / truth
    for i in np.flatnonzero(test):
        bottom.plot([share[i] / offset] * 2, std_rel[:, i], color=t["model"], linewidth=4.5,
                    solid_capstyle="round")  # fmt: skip
        bottom.plot([share[i] * offset] * 2, rec_rel[:, i], color=t["recalibrated"],
                    linewidth=4.5, solid_capstyle="round")  # fmt: skip
    bottom.axhline(0, color=t["reference"], linewidth=1.6)
    bottom.text(0.0042, 0.6, "true speed", color=t["secondary"], fontsize=10, va="bottom")
    bottom.set_ylabel("Claimed 90% range\n(relative to the true speed)")
    bottom.set_ylim(min(rec_rel[0, test].min(), std_rel[0, test].min(), -10) - 2,
                    max(rec_rel[1, test].max(), 4) + 9)  # fmt: skip
    bottom.yaxis.set_major_formatter(
        lambda v, _: "0%" if abs(v) < 1e-9 else f"{v:+.0f}%".replace("-", "−")
    )
    n = r["held_out"]
    bottom.legend(
        handles=[
            Line2D([], [], color=t["model"], linewidth=4.5, solid_capstyle="round",
                   label=f"Standard calibration: contains the truth at "
                         f"{r['standard_covered']} of {n} points"),
            Line2D([], [], color=t["recalibrated"], linewidth=4.5, solid_capstyle="round",
                   label=f"Recalibrated: contains the truth at "
                         f"{r['recalibrated_covered']} of {n} points"),
        ],
        loc="upper left", frameon=False, labelcolor=t["ink"], handlelength=1.4,
    )  # fmt: skip

    bottom.set_xscale("log")
    bottom.set_xlim(4e-3, 1)
    bottom.xaxis.set_major_locator(FixedLocator([1e-2, 1e-1, 1]))
    bottom.xaxis.set_minor_locator(NullLocator())
    bottom.xaxis.set_minor_formatter(NullFormatter())
    bottom.set_xticklabels(["1%", "10%", "100%\n(center line)"])
    bottom.set_xlabel("Distance from the wall, as a share of the way to the channel center "
                      "(log scale)")  # fmt: skip

    fig.suptitle(
        f"Standard calibration claims 90% confidence but is right "
        f"{r['standard_covered']} time{'s' if r['standard_covered'] != 1 else ''} in {n}",
        x=0.015, y=0.98, ha="left", fontsize=15, fontweight="bold", color=t["ink"],
    )  # fmt: skip
    fig.text(0.015, 0.937, "The SST coefficients a1 and β* are calibrated against direct "
             "simulation of a turbulent channel (Re$_\\tau$ = 182).\nThe claimed range "
             "reflects only the data's tiny noise, not the model's own error, so it misses.\n"
             "Recalibrating on points held back from the fit widens the range until it "
             "matches.",
             ha="left", va="top", fontsize=10, color=t["secondary"])  # fmt: skip
    fig.subplots_adjust(left=0.13, right=0.93, top=0.83, bottom=0.11, hspace=0.1)
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
        render(result, theme, args.output_dir / f"calibration_coverage_{theme}.png")

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
