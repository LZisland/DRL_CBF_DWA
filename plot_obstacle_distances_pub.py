"""IEEE Trans publication-quality safety-margin figure.

Upgraded from plot_obstacle_distances_from_excel.py with:
  - editable SVG/PDF text (svg.fonttype=none, pdf.fonttype=42)
  - unified colour family, safety threshold, statistics annotation
  - multi-format export (SVG + PDF + TIFF)
"""

import argparse
from pathlib import Path

import matplotlib as mpl
mpl.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════════════════════
# IEEE Transactions style — Times New Roman, editable vector text
# ═══════════════════════════════════════════════════════════════════════════════

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "serif"],
    "mathtext.fontset": "stix",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "legend.frameon": False,
    "lines.linewidth": 1.0,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
})

# ═══════════════════════════════════════════════════════════════════════════════
# Colour & design tokens
# ═══════════════════════════════════════════════════════════════════════════════

OBSTACLE_ALPHA = 0.68
CLEARANCE_COLOUR = "#1B3A5C"    # darker blue for the min-clearance summary line
THRESHOLD_COLOUR = "#767676"    # neutral mid-grey
MIN_POINT_COLOUR = "#B64342"    # red accent for the closest approach
STATS_BOX_EDGE = "#CFCECE"

DEFAULT_PLOTS = [
    ("concave_obstacle_distances.xlsx", "figures/concave_obstacle_distances_pub"),
    ("elongated_obstacle_distances.xlsx", "figures/elongated_obstacle_distances_pub"),
]


def plot_distance_excel(excel_path, output_stem):
    df = pd.read_excel(excel_path)

    sphere_columns = [col for col in df.columns if col.startswith("sphere_")]
    if not sphere_columns:
        raise ValueError(f"No sphere distance columns in {excel_path}")

    required = {"min_clearance_to_safe_boundary", "min_envelope_surface_distance"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{excel_path} missing columns: {sorted(missing)}")

    # Per-timestep UAV radius (constant 0.3 m in practice)
    quadrotor_radius = (
        df["min_envelope_surface_distance"] - df["min_clearance_to_safe_boundary"]
    ).to_numpy(dtype=float).reshape(-1, 1)

    # Distance from UAV surface to each obstacle surface.
    sphere_margins = df[sphere_columns].to_numpy(dtype=float) - quadrotor_radius
    time = df["time"].to_numpy(dtype=float)

    # Worst-case clearance at each timestep
    min_clearance = df["min_clearance_to_safe_boundary"].to_numpy(dtype=float)

    # ── Figure ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(3.5, 2.6))

    # ── Per-obstacle margin lines (faint, unified colour) ──────────────────
    colour_map = plt.get_cmap("tab20")
    for idx in range(sphere_margins.shape[1]):
        colour = colour_map(idx % colour_map.N)
        ax.plot(time, sphere_margins[:, idx],
                color=colour, alpha=OBSTACLE_ALPHA, linewidth=0.7)

    # Dummy artists for compact legend
    dummy_obstacle = mpl.lines.Line2D(
        [], [], color="0.35", alpha=0.9, linewidth=0.8,
        label="Distance to obstacle"
    )
    dummy_clearance = mpl.lines.Line2D(
        [], [], color=CLEARANCE_COLOUR, linewidth=0.9,
        label="Worst-case clearance"
    )

    # ── Worst-case clearance summary line ──────────────────────────────────
    ax.plot(time, min_clearance, color=CLEARANCE_COLOUR, linewidth=0.9,
            zorder=3)

    # ── Collision boundary ─────────────────────────────────────────────────
    ax.axhline(y=0, color=THRESHOLD_COLOUR, linestyle="--", linewidth=0.7,
               alpha=0.65, zorder=2)
    dummy_threshold = mpl.lines.Line2D(
        [], [], color=THRESHOLD_COLOUR, linestyle="--", linewidth=0.7, alpha=0.65,
        label="Collision boundary"
    )

    # ── Global minimum marker ──────────────────────────────────────────────
    min_flat_index = int(np.nanargmin(sphere_margins))
    min_row, min_col = np.unravel_index(min_flat_index, sphere_margins.shape)
    min_time = float(time[min_row])
    min_value = float(sphere_margins[min_row, min_col])

    ax.scatter(min_time, min_value,
               color=MIN_POINT_COLOUR, s=22, zorder=6,
               edgecolors="white", linewidths=0.5)
    dummy_min = mpl.lines.Line2D(
        [], [], marker="o", color=MIN_POINT_COLOUR, markeredgecolor="white",
        markeredgewidth=0.4, markersize=5, linestyle="",
        label=f"Closest approach ({min_value:.3f} m)"
    )

    # ── Minimum annotation with arrow ──────────────────────────────────────
    ax.annotate(
        f"{min_value:.3f} m",
        xy=(min_time, min_value),
        xytext=(10, 12),
        textcoords="offset points",
        fontsize=6.5, fontweight="bold", color=MIN_POINT_COLOUR,
        arrowprops=dict(
            arrowstyle="->", lw=0.6, color=MIN_POINT_COLOUR, alpha=0.7,
            shrinkA=3, shrinkB=2,
        ),
    )

    # ── Statistics annotation ──────────────────────────────────────────────
    # ── Axis formatting ────────────────────────────────────────────────────
    # Cap y-axis to the worst-case clearance range so the safety-critical
    # region (0–1 m) is legible.  Per-obstacle lines from distant obstacles
    # that extend beyond this ceiling are clipped — they lie far above the
    # collision boundary and pose no safety concern.
    distance_max = float(np.nanmax(sphere_margins))
    y_max = max(0.5, distance_max * 1.08)
    ax.set_xlim(0.0, float(time[-1]))
    ax.set_ylim(0.0, y_max)

    ax.set_xlabel("time (s)")
    ax.set_ylabel("distance (m)")

    ax.xaxis.set_major_locator(plt.MaxNLocator(6))
    ax.yaxis.set_major_locator(plt.MaxNLocator(6))

    ax.tick_params(axis="both", which="major", pad=1.5)

    # ── Legend ─────────────────────────────────────────────────────────────
    ax.legend(
        handles=[dummy_min, dummy_clearance, dummy_obstacle, dummy_threshold],
        loc="upper right",
        frameon=True,
        fancybox=False,
        edgecolor=STATS_BOX_EDGE,
        fontsize=6.5,
        handlelength=1.3,
        handletextpad=0.5,
        borderpad=0.4,
        labelspacing=0.35,
    )

    fig.tight_layout(pad=0.3)

    # ── Multi-format export ────────────────────────────────────────────────
    out_dir = Path(output_stem).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []
    for fmt, dpi in [("png", 600), ("svg", None), ("pdf", None), ("tiff", 600)]:
        output_path = Path(f"{output_stem}.{fmt}")
        kw = {"bbox_inches": "tight", "pad_inches": 0.02}
        if dpi:
            kw["dpi"] = dpi
        fig.savefig(output_path, **kw)
        saved_files.append(output_path.resolve())

    plt.close(fig)
    print("Saved figures:")
    for saved_file in saved_files:
        print(f"  {saved_file}")


def main():
    parser = argparse.ArgumentParser(
        description="IEEE Trans publication-quality safety-margin figure."
    )
    parser.add_argument("--excel", help="Input Excel file path.")
    parser.add_argument("--output", help="Output stem (without extension).")
    args = parser.parse_args()

    if args.excel:
        output = args.output or f"./figures/{Path(args.excel).stem}"
        plot_distance_excel(args.excel, output)
        return

    for excel_path, figure_stem in DEFAULT_PLOTS:
        if Path(excel_path).exists():
            plot_distance_excel(excel_path, figure_stem)
        else:
            print(f"Skipped (missing): {excel_path}")


if __name__ == "__main__":
    main()
