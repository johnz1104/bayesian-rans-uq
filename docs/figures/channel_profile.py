"""Plot the mean flow speed in a turbulent channel: SST model against direct simulation.

The SST k-omega profile is computed with cfdr on the periodic channel used by the
grid-convergence test in rans-uq/tests (Re_tau = 182.088, unit half-height and friction
velocity). It is compared with the direct numerical simulation of Lee and Moser
(J. Fluid Mech. 774, 2015), whose mean profile is not distributed with this repository:
download LM_Channel_0180_mean_prof.dat from https://turbulence.oden.utexas.edu and pass
its path. The script writes a light and a dark version of the figure and prints the
values shown in the README table.

Requires the cfdr package, NumPy and matplotlib:

    python docs/figures/channel_profile.py LM_Channel_0180_mean_prof.dat docs/figures
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator

from cfdr import _core as c

RE_TAU = 182.088

# Colors: categorical slots 1 and 2 of the validated reference palette, with its ink
# and chrome tokens, stepped separately for the light and dark surfaces.
THEMES = {
    "light": {
        "surface": "#fcfcfb", "ink": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
        "grid": "#e1e0d9", "axis": "#c3c2b7", "reference": "#2a78d6", "model": "#eb6834",
    },
    "dark": {
        "surface": "#1a1a19", "ink": "#ffffff", "secondary": "#c3c2b7", "muted": "#898781",
        "grid": "#2c2c2a", "axis": "#383835", "reference": "#3987e5", "model": "#d95926",
    },
}  # fmt: skip


def sst_profile(y_plus: np.ndarray, half_cells: int) -> np.ndarray:
    """Converged SST streamwise velocity, in wall units, at the given wall distances."""
    first_height = 1 + math.tanh(2 * (1 / half_cells - 1)) / math.tanh(2)
    mesh = c.Mesh.make_periodic_channel_2d(
        3, 2 * half_cells, 1.0, 1.0, RE_TAU, first_height * RE_TAU
    )
    mesh.compute_wall_distance()
    observations = c.ObservationOperator()
    observations.add_skin_friction("bottom", c.Vec3(0.5, 0.0, 0.0), 2.0, 1.0, 1.0)
    for y in y_plus / RE_TAU:
        observations.add_velocity_profile(c.Vec3(0.5, float(y), 0.0), 0, 0.0, 1.0)
    settings = c.SolverSettings()
    settings.max_iterations = 150000
    settings.convergence_tol = 1e-8
    settings.alpha_u = 0.5
    settings.body_force = c.Vec3(1.0, 0.0, 0.0)
    settings.verbose = False
    parameters = c.InferenceParameterSet.a1_betaStar()
    boundaries = c.FlowBoundaryConditions.channel_defaults(mesh, 1.0, 1.0, 10.0)
    model = c.ForwardModel(
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
    result = model.evaluate(parameters.pack(c.SSTCoefficients()))
    if result.status != c.EvaluationStatus.Converged:
        raise RuntimeError(f"channel solve did not converge: {result.status.name}")
    predictions = np.asarray(result.predictions)
    # With unit friction velocity, the wall shear must balance the unit body force.
    if abs(predictions[0] / 2 - 1) > 1e-5:
        raise RuntimeError("wall shear does not balance the applied body force")
    return predictions[1:]


def render(
    share: np.ndarray, reference: np.ndarray, model: np.ndarray, theme: str, output: Path
) -> None:
    """Draw the two-panel figure in one theme; distances are shares of the half-height."""
    t = THEMES[theme]
    error = 100 * (model - reference) / reference
    worst = int(np.argmin(error))
    plt.rcParams.update({
        "font.family": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 11, "text.color": t["ink"], "axes.labelcolor": t["secondary"],
        "xtick.color": t["muted"], "ytick.color": t["muted"],
        "xtick.labelcolor": t["secondary"], "ytick.labelcolor": t["secondary"],
    })  # fmt: skip
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(8, 6.6), sharex=True, gridspec_kw={"height_ratios": [2.3, 1]}
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
    top.plot(share, reference, color=t["reference"], label="Reference: direct simulation",
             **line)  # fmt: skip
    top.plot(share, model, color=t["model"], label="Turbulence model (this library)", **line)
    top.fill_between(share, model, reference, where=model < reference, color=t["model"],
                     alpha=0.12, linewidth=0)  # fmt: skip
    top.set_ylim(0, 104)
    top.yaxis.set_major_locator(FixedLocator([0, 25, 50, 75, 100]))
    top.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    top.set_ylabel("Average flow speed\n(share of reference center-line speed)")
    top.legend(loc="upper left", frameon=False, handlelength=1.6, borderaxespad=0.2,
               labelcolor=t["ink"])  # fmt: skip
    widest = int(np.argmax(reference - model))
    top.annotate("Shaded gap: the model's error",
                 xy=(share[widest], (reference[widest] + model[widest]) / 2),
                 xytext=(0.14, 48), color=t["secondary"], fontsize=10.5,
                 arrowprops={"arrowstyle": "-", "color": t["muted"], "linewidth": 0.9})  # fmt: skip
    label_at = int(np.argmin(np.abs(share - 0.4)))
    top.text(share[label_at], reference[label_at] + 3, "Reference", ha="right", va="bottom",
             color=t["ink"], fontsize=10.5)  # fmt: skip
    top.text(share[label_at], model[label_at] - 3, "Model", ha="left", va="top", color=t["ink"],
             fontsize=10.5)  # fmt: skip

    bottom.plot(share, error, color=t["model"], **line)
    bottom.fill_between(share, error, 0, color=t["model"], alpha=0.12, linewidth=0)
    bottom.axhline(0, color=t["axis"], linewidth=1)
    bottom.plot(share[worst], error[worst], "o", markersize=8, color=t["model"],
                markeredgecolor=t["surface"], markeredgewidth=2, zorder=5)  # fmt: skip
    bottom.annotate(f"{abs(error[worst]):.1f}% too slow",
                    xy=(share[worst], error[worst]), xytext=(10, -2),
                    textcoords="offset points", va="center", color=t["ink"],
                    fontsize=10.5)  # fmt: skip
    bottom.text(0.0012, 1.6, "Accurate in the thin layer next to the wall",
                color=t["secondary"], fontsize=10.5, va="bottom")  # fmt: skip
    bottom.set_ylim(-10, 5)
    bottom.yaxis.set_major_locator(FixedLocator([-10, -5, 0, 5]))
    bottom.set_yticklabels(["−10%", "−5%", "0%", "+5%"])
    bottom.set_ylabel("Model error")

    bottom.set_xscale("log")
    bottom.set_xlim(1e-3, 1)
    bottom.xaxis.set_major_locator(FixedLocator([1e-3, 1e-2, 1e-1, 1]))
    bottom.xaxis.set_minor_locator(NullLocator())
    bottom.xaxis.set_minor_formatter(NullFormatter())
    bottom.set_xticklabels(["0.1%", "1%", "10%", "100%\n(center line)"])
    bottom.set_xlabel("Distance from the wall, as a share of the way to the channel center "
                      "(log scale)")  # fmt: skip

    fig.suptitle(
        f"A standard turbulence model predicts the flow up to {abs(error[worst]):.1f}% too slow",
        x=0.015,
        y=0.975,
        ha="left",
        fontsize=15,
        fontweight="bold",
        color=t["ink"],
    )
    fig.text(0.015, 0.925, "Average flow speed across half of a turbulent channel "
             "(Re$_\\tau$ = 182), from the wall to the center line. Reference: direct "
             "numerical\nsimulation, which resolves every eddy (Lee & Moser, 2015). "
             "Model: SST k–ω with its standard coefficients.",
             ha="left", va="top", fontsize=10, color=t["secondary"])  # fmt: skip
    fig.subplots_adjust(left=0.13, right=0.93, top=0.84, bottom=0.13, hspace=0.12)
    fig.savefig(output, dpi=200, facecolor=t["surface"])
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dns", type=Path, help="LM_Channel_0180_mean_prof.dat")
    parser.add_argument("output_dir", type=Path, help="directory for the two images")
    parser.add_argument("--half-cells", type=int, default=64, help="cells per half-channel")
    args = parser.parse_args()

    dns = np.loadtxt(args.dns, comments="%")
    keep = (dns[:, 1] >= 0.1) & (dns[:, 1] < RE_TAU)  # the plot starts at 0.1% of the way
    y_plus, u_dns = dns[keep, 1], dns[keep, 2]
    u_center = dns[np.argmax(dns[:, 1]), 2]

    u_sst = sst_profile(y_plus, args.half_cells)
    u_coarse = sst_profile(y_plus, args.half_cells // 2)
    print(f"grid check: max |U+(N) - U+(N/2)| = {np.max(np.abs(u_sst - u_coarse)):.2e}")

    share = y_plus / RE_TAU
    reference, model = 100 * u_dns / u_center, 100 * u_sst / u_center
    for theme in THEMES:
        render(share, reference, model, theme, args.output_dir / f"channel_profile_{theme}.png")

    print("share of way to center | reference | model | model error")
    worst_share = share[int(np.argmin((model - reference) / reference))]
    for target in (0.005, 0.02, 0.05, worst_share, 0.3, 0.6, 1.0):
        i = int(np.argmin(np.abs(share - target)))
        err = 100 * (u_sst[i] - u_dns[i]) / u_dns[i]
        print(f"{100 * share[i]:.1f}% | {reference[i]:.1f}% | {model[i]:.1f}% | {err:+.1f}%")


if __name__ == "__main__":
    main()
