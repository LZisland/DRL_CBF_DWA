import argparse
from pathlib import Path

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_PLOTS = [
    (
        "concave_obstacle_distances.xlsx",
        "concave_obstacle_distances.pdf",
    ),
    (
        "elongated_obstacle_distances.xlsx",
        "elongated_obstacle_distances.pdf",
    ),
]


def plot_distance_excel(excel_path, figure_path):
    df = pd.read_excel(excel_path)
    sphere_columns = [col for col in df.columns if col.startswith("sphere_")]
    if not sphere_columns:
        raise ValueError(f"No sphere distance columns found in {excel_path}")

    if "min_clearance_to_safe_boundary" not in df.columns:
        raise ValueError(
            f"{excel_path} does not contain min_clearance_to_safe_boundary. "
            "Regenerate it with the simulation script first."
        )

    if "min_envelope_surface_distance" not in df.columns:
        raise ValueError(
            f"{excel_path} does not contain min_envelope_surface_distance. "
            "Regenerate it with the simulation script first."
        )

    quadrotor_radius = (
        df["min_envelope_surface_distance"] - df["min_clearance_to_safe_boundary"]
    ).to_numpy(dtype=float).reshape(-1, 1)
    sphere_margins = df[sphere_columns].to_numpy(dtype=float) - quadrotor_radius
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    for idx in range(sphere_margins.shape[1]):
        ax.plot(df["time"], sphere_margins[:, idx], linewidth=0.8, alpha=0.38)

    min_flat_index = int(np.nanargmin(sphere_margins))
    min_row, min_col = np.unravel_index(min_flat_index, sphere_margins.shape)
    min_time = float(df["time"].iloc[min_row])
    min_value = float(sphere_margins[min_row, min_col])
    ax.scatter(min_time, min_value, color="black", s=12, zorder=5, label="minimum distance to obstacle")
    ax.annotate(
        f"{min_value:.3f} m",
        xy=(min_time, min_value),
        xytext=(4, 5),
        textcoords="offset points",
        fontsize=7,
        color="black",
    )

    y_max = max(0.1, float(np.nanmax(sphere_margins)) * 1.08)
    ax.set_xlim(0.0, float(df["time"].iloc[-1]))
    ax.set_ylim(0.0, y_max)
    ax.set_xlabel("time(s)")
    ax.set_ylabel("distance(m)")
    ax.grid(True, alpha=0.30)
    ax.legend(frameon=True, fontsize=7)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Distance figure saved to: {figure_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate safety-margin figures from obstacle-distance Excel files."
    )
    parser.add_argument("--excel", help="Input Excel file. If omitted, default Excel files are used.")
    parser.add_argument("--output", help="Output image path for --excel.")
    args = parser.parse_args()

    if args.excel:
        output = args.output or f"{Path(args.excel).stem}.pdf"
        plot_distance_excel(args.excel, output)
        return

    for excel_path, figure_path in DEFAULT_PLOTS:
        if Path(excel_path).exists():
            plot_distance_excel(excel_path, figure_path)
        else:
            print(f"Skipped missing Excel file: {excel_path}")


if __name__ == "__main__":
    main()
