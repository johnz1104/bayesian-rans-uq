"""Draw the flow over a backward-facing step as the SST turbulence model computes it.

Solves the uniform-inlet backward-facing step at Re_h = 5100 (step height h = 1, channel
height 6h, slip top) with the same mesh and settings as the bfs_lemoin_uniform_inlet
reference solution in tests/golden. The top panel shows the average flow speed with
streamlines and the recirculation bubble; the bottom panel shows the turbulent kinetic
energy, which is how a RANS model represents turbulence. The inlet is uniform and not
matched to a measured case, so the reattachment length is not a validated prediction.

Writes step_flow_light.png and step_flow_dark.png. The solve takes about a minute; --cache
keeps the fields so the figure can be redrawn without solving again. Requires cfdr, NumPy
and matplotlib:

    python docs/figures/step_flow.py docs/figures --cache step_flow.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib.tri import LinearTriInterpolator, Triangulation

from cfdr import _core as c

H_STEP, HEIGHT, RE, U_BULK = 1.0, 6.0, 5100.0, 1.0
VIEW = (-4.0, 16.0, 0.0, 3.0)  # x and y limits, in step heights

# Ink and chrome tokens of the reference palette. Both fields read like an image in either
# theme: dark for low values and bright for high ones (the palette's blue ramp for speed,
# a one-hue orange ramp for turbulence).
BLUE = ["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"]
THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "secondary": "#52514e",
              "muted": "#898781", "solid": "#c3c2b7", "accent": "#eb6834"},
    "dark": {"surface": "#1a1a19", "ink": "#ffffff", "secondary": "#c3c2b7",
             "muted": "#898781", "solid": "#383835", "accent": "#d95926"},
}  # fmt: skip


def solve() -> dict:
    """The reference step case; returns cell centers, velocity, k and reattachment."""
    nu = U_BULK * H_STEP / RE
    mesh = c.Mesh.make_backward_facing_step_2d(40, 48, 24, 18, 10.0, 22.0, H_STEP, HEIGHT, RE, 1.0)
    mesh.set_patch_type("top_wall", "symmetry")
    mesh.compute_wall_distance()
    k_in = 1.5 * (U_BULK * 0.05) ** 2
    omega_in = k_in / (nu * 100.0)
    boundaries = c.FlowBoundaryConditions.bfs_defaults(mesh, U_BULK, k_in, omega_in)
    settings = c.SolverSettings()
    for name, value in {
        "max_iterations": 12000, "convergence_tol": 1e-4, "divergence_limit": 1e8,
        "alpha_u": 0.3, "alpha_p": 0.2, "alpha_k": 0.4, "alpha_omega": 0.4,
        "inner_iterations": 300, "inner_tolerance": 1e-4, "turb_start_iter": 30,
        "turb_update_interval": 2, "verbose": False,
    }.items():  # fmt: skip
        setattr(settings, name, value)
    solver = c.SIMPLESolver(mesh, c.SSTCoefficients(), boundaries, nu, settings)
    fields = c.FlowFields(mesh)
    solver.init_uniform(fields, c.Vec3(U_BULK, 0.0, 0.0), 0.0, k_in, omega_in)
    history = solver.solve(fields)
    if not history.converged:
        raise RuntimeError("the step solve did not converge")
    observations = c.ObservationOperator()
    observations.add_reattachment_length("bottom_wall_down", 6.28, 0.5)
    snapshot = fields.snapshot()
    # Boundary-face values come from the solver's own boundary conditions (zero velocity
    # and zero k on walls), so the drawn field reaches the walls without invented data.
    faces = np.asarray(mesh.face_centers())[mesh.n_internal_faces() :, :2]
    boundary = snapshot["boundary"]
    return {
        "centers": np.vstack([np.asarray(mesh.cell_centers())[:, :2], faces]),
        "velocity": np.vstack([np.asarray(snapshot["U"])[:, :2], np.asarray(boundary["U"])[:, :2]]),
        "k": np.concatenate([np.asarray(snapshot["k"]), np.asarray(boundary["k"])]),
        "reattachment": float(observations.evaluate(mesh, fields, nu)[0]),
        "iterations": int(history.final_iteration),
    }


def triangulation(points: np.ndarray) -> Triangulation:
    """Delaunay on cell centers, with triangles that cut through the step removed."""
    tri = Triangulation(points[:, 0], points[:, 1])
    corners = points[tri.triangles]  # (triangles, 3, 2)

    def solid(p: np.ndarray) -> np.ndarray:
        return (p[..., 0] < -1e-9) & (p[..., 1] < H_STEP - 1e-9)

    midpoints = 0.5 * (corners + np.roll(corners, 1, axis=1))
    tri.set_mask(solid(corners.mean(axis=1)) | solid(midpoints).any(axis=1))
    return tri


def render(data: dict, theme: str, output: Path) -> None:
    t = THEMES[theme]
    plt.rcParams.update({
        "font.family": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 12,
        "text.color": t["ink"], "axes.labelcolor": t["secondary"],
        "xtick.color": t["muted"], "ytick.color": t["muted"],
        "xtick.labelcolor": t["secondary"], "ytick.labelcolor": t["secondary"],
        "xtick.labelsize": 11, "ytick.labelsize": 11,
    })  # fmt: skip
    points, velocity, k = data["centers"], data["velocity"], data["k"]
    tri = triangulation(points)
    speed = np.hypot(velocity[:, 0], velocity[:, 1]) / U_BULK
    speed_map = LinearSegmentedColormap.from_list("speed", BLUE)
    turbulence_map = plt.get_cmap("Oranges_r")
    label_box = {"boxstyle": "round,pad=0.3,rounding_size=0.3", "facecolor": t["surface"],
                 "edgecolor": "none", "alpha": 0.85}  # fmt: skip
    leader = {"arrowstyle": "-", "color": t["ink"], "lw": 0.9, "alpha": 0.7}

    fig = plt.figure(figsize=(12, 6.6))
    fig.patch.set_facecolor(t["surface"])
    fig.text(0.03, 0.955, "Flow over a step, as the turbulence model sees it", fontsize=21,
             fontweight="bold", va="top")  # fmt: skip
    fig.text(0.03, 0.885, "Computed with this library: the SST k–ω model at a Reynolds number "
             "of 5100. The flow separates at the step, recirculates, and\nreattaches "
             "downstream. A turbulence model does not resolve individual eddies; it predicts "
             "where turbulence is and how strong it is (bottom).", fontsize=12,
             color=t["secondary"], va="top", linespacing=1.4)  # fmt: skip

    panels = [fig.add_axes((0.06, 0.45, 0.8, 0.3)), fig.add_axes((0.06, 0.1, 0.8, 0.3))]
    xg, yg = np.meshgrid(np.linspace(VIEW[0], VIEW[1], 700), np.linspace(VIEW[2], VIEW[3], 120))
    ug = LinearTriInterpolator(tri, velocity[:, 0])(xg, yg)
    vg = LinearTriInterpolator(tri, velocity[:, 1])(xg, yg)
    for ax in panels:
        ax.set_facecolor(t["surface"])
        ax.set_xlim(VIEW[0], VIEW[1])
        ax.set_ylim(VIEW[2], VIEW[3])
        ax.set_aspect("equal")
        for edge in ("top", "right"):
            ax.spines[edge].set_visible(False)
        for edge in ("left", "bottom"):
            ax.spines[edge].set_color(t["solid"])
        ax.tick_params(length=0, pad=5)
        ax.set_yticks([0, 1, 2, 3])
        ax.set_xticks([-4, 0, 4, 8, 12, 16])

    top, bottom = panels
    shade = top.tricontourf(tri, speed, levels=np.linspace(0, 1.05, 22), cmap=speed_map)
    halo = [patheffects.withStroke(linewidth=1.8, foreground="#0d366b", alpha=0.55)]
    flow_lines = top.streamplot(xg, yg, ug, vg, density=(4.0, 1.1), color="#ffffff",
                                linewidth=0.7, arrowsize=0.7)  # fmt: skip
    seeds = np.column_stack([np.linspace(0.6, 5.5, 7), np.full(7, 0.35)])
    bubble_lines = top.streamplot(xg, yg, ug, vg, start_points=seeds, color="#ffffff",
                                  linewidth=0.7, arrowsize=0.7, broken_streamlines=False)  # fmt: skip
    for lines in (flow_lines, bubble_lines):
        lines.lines.set_path_effects(halo)
    x_r = data["reattachment"]
    top.plot(x_r, 0.0, marker="^", markersize=10, color=t["accent"], clip_on=False, zorder=6)
    top.annotate(f"Reattaches {x_r:.1f} step\nheights downstream", xy=(x_r, 0.05),
                 xytext=(x_r + 1.6, 1.75), fontsize=11, color=t["ink"], bbox=label_box,
                 arrowprops=leader)  # fmt: skip
    top.annotate("Recirculation bubble", xy=(2.6, 0.55), xytext=(1.2, 2.2), fontsize=11,
                 color=t["ink"], bbox=label_box, arrowprops=leader)  # fmt: skip
    top.annotate("flow", xy=(-2.0, 2.2), xytext=(-3.7, 2.2), fontsize=11, color=t["ink"],
                 va="center", bbox=label_box,
                 arrowprops={"arrowstyle": "->", "color": t["ink"], "lw": 1.2})  # fmt: skip
    top.set_title("Average flow speed and streamlines", loc="left", fontsize=13.5,
                  fontweight="bold", pad=8)  # fmt: skip

    k_scaled = k / U_BULK**2
    heat = bottom.tricontourf(tri, k_scaled, levels=np.linspace(0, k_scaled.max(), 22),
                              cmap=turbulence_map)  # fmt: skip
    bottom.set_title("Turbulence intensity the model predicts", loc="left", fontsize=13.5,
                     fontweight="bold", pad=8)  # fmt: skip
    bottom.annotate("Shear layer shed from the step", xy=(3.0, 0.95), xytext=(5.0, 2.2),
                    fontsize=11, color=t["ink"], bbox=label_box, arrowprops=leader)  # fmt: skip
    bottom.set_xlabel("Distance downstream of the step, in step heights")
    for ax in panels:
        ax.add_patch(Rectangle((VIEW[0], 0.0), -VIEW[0], H_STEP, facecolor=t["solid"],
                               edgecolor="none", zorder=4))  # fmt: skip
        ax.text(-2.0, 0.5, "step", ha="center", va="center", fontsize=11,
                color=t["secondary"], zorder=5)  # fmt: skip
        ax.set_ylabel("Height")

    for ax, mappable, label, ticks in (
        (top, shade, "Speed\n(share of inflow)", [0, 0.5, 1.0]),
        (bottom, heat, "Turbulent kinetic\nenergy, k / U²", None),
    ):  # fmt: skip
        box = ax.get_position()
        cax = fig.add_axes((0.885, box.y0 + 0.02, 0.012, box.height - 0.04))
        bar = fig.colorbar(mappable, cax=cax, ticks=ticks)
        bar.outline.set_visible(False)
        bar.ax.tick_params(length=0, labelsize=10, colors=t["secondary"])
        if ticks is None:
            bar.ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        bar.ax.set_title(label, fontsize=10, color=t["secondary"], loc="left", pad=8)

    fig.text(0.03, 0.03, "Uniform inflow, not matched to a measured case; the reattachment "
             "length is not a validated prediction.", fontsize=10.5, color=t["secondary"],
             va="center")  # fmt: skip
    fig.savefig(output, dpi=160, facecolor=t["surface"])
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output_dir", type=Path, help="directory for the two images")
    parser.add_argument("--cache", type=Path, help="NPZ file that stores or reuses the fields")
    args = parser.parse_args()
    if args.cache and args.cache.exists():
        data = dict(np.load(args.cache))
        data["reattachment"] = float(data["reattachment"])
    else:
        data = solve()
        if args.cache:
            np.savez(args.cache, **data)
    for theme in THEMES:
        render(data, theme, args.output_dir / f"step_flow_{theme}.png")
    print(f"reattachment x/h = {data['reattachment']:.3f}")


if __name__ == "__main__":
    main()
