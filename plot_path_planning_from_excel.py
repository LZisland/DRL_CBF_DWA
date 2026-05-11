import argparse
from pathlib import Path

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from cbf_dwa_concave_obstacles import (
    build_concave_obstacles,
    draw_box_from_segment as draw_concave_box,
)
from cbf_dwa_elongated_obstacles import (
    build_elongated_obstacles,
    draw_box_from_segment as draw_elongated_box,
)


FIGURE_CONFIG = {
    "concave": {
        "trajectory": "concave_uav_trajectory.xlsx",
        "output_png": "concave_path_planning.png",
        "output_pdf": "concave_path_planning.pdf",
    },
    "elongated": {
        "trajectory": "elongated_uav_trajectory.xlsx",
        "output_png": "elongated_path_planning.png",
        "output_pdf": "elongated_path_planning.pdf",
    },
}


def configure_ieee_style():
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "mathtext.fontset": "stix",
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "lines.linewidth": 1.1,
            "axes.linewidth": 0.6,
            "grid.linewidth": 0.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def make_sphere(center, radius, n=80):
    u = np.linspace(0.0, 2.0 * np.pi, n)
    v = np.linspace(0.0, np.pi, n)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


def draw_envelope_spheres(ax, obstacles):
    for obs in obstacles:
        x, y, z = make_sphere(obs[:3], obs[3])
        ax.plot_surface(
            x,
            y,
            z,
            color="#D99058",
            alpha=0.16,
            linewidth=0.0,
            shade=False,
            antialiased=True,
        )
        ax.scatter(
            obs[0],
            obs[1],
            obs[2],
            color="#B3352B",
            s=7,
            depthshade=False,
        )


def draw_obstacle_geometry(ax, scenario, geometry):
    if scenario == "concave":
        for segments in geometry:
            for start, end in segments:
                draw_concave_box(ax, start, end, 0.28, 0.22, color="0.05")
    else:
        for start, end in geometry:
            draw_elongated_box(ax, start, end, 0.23, 0.20, color="0.05")


def load_scenario_geometry(scenario):
    if scenario == "concave":
        geometry, obstacles = build_concave_obstacles()
    elif scenario == "elongated":
        geometry, obstacles = build_elongated_obstacles()
    else:
        raise ValueError(f"Unsupported scenario: {scenario}")
    return geometry, obstacles


def set_dashed_3d_grid(ax):
    ax.grid(True)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis._axinfo["grid"]["linestyle"] = "--"
        axis._axinfo["grid"]["linewidth"] = 0.4
        axis._axinfo["grid"]["color"] = (0.65, 0.65, 0.65, 0.55)


def build_legend_handles():
    """
    Build custom legend handles so that the legend symbols are visually
    consistent with the objects shown in the figure.
    """
    original_obstacle_handle = Rectangle(
        (0, 0),
        1,
        1,
        facecolor="none",
        edgecolor="0.05",
        linewidth=1.0,
        label="Original obstacle",
    )

    envelope_sphere_handle = Line2D(
        [0],
        [0],
        marker="o",
        linestyle="none",
        markerfacecolor="#D99058",
        markeredgecolor="none",
        alpha=0.16,
        markersize=7,
        label="Envelope spheres",
    )

    trajectory_handle = Line2D(
        [0],
        [0],
        color="tab:blue",
        linewidth=1.4,
        label="DT-CBF DWA",
    )

    start_handle = Line2D(
        [0],
        [0],
        marker="^",
        linestyle="none",
        markerfacecolor="black",
        markeredgecolor="black",
        markersize=5,
        label="Start",
    )

    goal_handle = Line2D(
        [0],
        [0],
        marker="o",
        linestyle="none",
        markerfacecolor="red",
        markeredgecolor="red",
        markersize=5.5,
        label="Goal",
    )

    return [
        original_obstacle_handle,
        envelope_sphere_handle,
        trajectory_handle,
        start_handle,
        goal_handle,
    ]


def plot_path_planning(scenario, trajectory_path, output_png, output_pdf):
    configure_ieee_style()

    trajectory_df = pd.read_excel(trajectory_path)
    required = {"position_x", "position_y", "position_z"}
    missing = required.difference(trajectory_df.columns)
    if missing:
        raise ValueError(f"{trajectory_path} is missing columns: {sorted(missing)}")

    geometry, obstacles = load_scenario_geometry(scenario)
    trajectory = trajectory_df[["position_x", "position_y", "position_z"]].to_numpy(dtype=float)

    fig = plt.figure(figsize=(3.5, 3.05))
    ax = fig.add_subplot(111, projection="3d")

    # White background
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    ax.xaxis.pane.set_facecolor("white")
    ax.yaxis.pane.set_facecolor("white")
    ax.zaxis.pane.set_facecolor("white")

    # Draw original obstacles and their spherical envelopes
    draw_obstacle_geometry(ax, scenario, geometry)
    draw_envelope_spheres(ax, obstacles)

    # UAV trajectory
    ax.plot(
        trajectory[:, 0],
        trajectory[:, 1],
        trajectory[:, 2],
        color="tab:blue",
        linewidth=1.4,
    )

    # Start point
    ax.scatter(
        trajectory[0, 0],
        trajectory[0, 1],
        trajectory[0, 2],
        color="black",
        s=24,
        marker="^",
        linewidth=0.4,
        depthshade=False,
    )

    # Goal point
    ax.scatter(
        trajectory[-1, 0],
        trajectory[-1, 1],
        trajectory[-1, 2],
        color="red",
        s=36,
        marker="o",
        linewidth=0.4,
        depthshade=False,
    )

    # Axis settings
    ax.set_xlabel("X (m)", labelpad=-2)
    ax.set_ylabel("Y (m)", labelpad=-2)
    ax.set_zlabel("Z (m)", labelpad=4)
    ax.set_xlim(0.0, 12.0)
    ax.set_ylim(0.0, 12.0)
    ax.set_zlim(0.0, 12.0)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=65, azim=-135)
    ax.tick_params(axis="both", which="major", pad=-2)

    # Dashed grid
    set_dashed_3d_grid(ax)

    # Custom legend
    legend_handles = build_legend_handles()
    ax.legend(handles=legend_handles, loc="best", frameon=True)

    fig.subplots_adjust(
        left=0.02,
        right=0.90,
        bottom=0.02,
        top=0.98,
    )

    fig.savefig(output_png, dpi=600, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(output_pdf, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    print(f"Saved figure: {output_png}")
    print(f"Saved figure: {output_pdf}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate IEEE-style path-planning figures from UAV trajectory Excel files."
    )
    parser.add_argument(
        "--scenario",
        choices=["concave", "elongated", "all"],
        default="all",
        help="Scenario to plot. Default: all.",
    )
    parser.add_argument(
        "--trajectory",
        help="Trajectory Excel file for a single scenario.",
    )
    parser.add_argument(
        "--output",
        help="Output PNG path for a single scenario.",
    )
    args = parser.parse_args()

    scenarios = ["concave", "elongated"] if args.scenario == "all" else [args.scenario]

    for scenario in scenarios:
        config = FIGURE_CONFIG[scenario]
        trajectory = args.trajectory or config["trajectory"]
        output_png = args.output or config["output_png"]
        output_pdf = (
            str(Path(output_png).with_suffix(".pdf"))
            if args.output
            else config["output_pdf"]
        )

        if not Path(trajectory).exists():
            print(f"Skipped missing trajectory file: {trajectory}")
            continue

        plot_path_planning(scenario, trajectory, output_png, output_pdf)


if __name__ == "__main__":
    main()