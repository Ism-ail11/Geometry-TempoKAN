"""Render four publication-style composite figures from verified tables."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np
import pandas as pd

from Datasets.prepare import CLASSES
from .registry import DATA_NAMES, NAMES, PRIMARY, TASKS


INK, MUTED, GRID = "#172b45", "#536780", "#e0e8f0"
PALETTE = {"geometry_kan": "#087e77", "geometry_mlp": "#8253a0", "geometry_axes": "#52677e",
           "temporal_cnn": "#be642b", "kan_mixer": "#385fab", "kan_fe": "#99720d", "kan_har": "#b04c72"}
MARKERS = {"geometry_kan": "o", "geometry_axes": "+", "temporal_cnn": "s", "kan_mixer": "D",
           "kan_fe": "^", "kan_har": "v", "geometry_mlp": "P"}
COMPARISON = (PRIMARY, "temporal_cnn", "kan_mixer", "kan_fe", "kan_har")

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10.4, "text.color": INK,
    "axes.labelcolor": INK, "axes.titlecolor": INK, "axes.edgecolor": GRID,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "xtick.color": MUTED, "ytick.color": MUTED, "legend.frameon": False,
    "figure.facecolor": "#fbfcfe", "axes.facecolor": "white",
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
})


def canvas(title, subtitle, *, size=(13, 9.2), shape=(2, 2), **layout):
    fig, axes = plt.subplots(*shape, figsize=size, gridspec_kw=layout)
    fig.text(0.067, 0.95, title, fontsize=23, weight="bold", va="top")
    fig.text(0.068, 0.902, subtitle, fontsize=10.6, color=MUTED, va="top")
    return fig, axes.ravel()


def panel_title(ax, i, dataset, regime, detail=""):
    title = DATA_NAMES[dataset]
    if regime == "pace":
        title += " · " + (detail or "unseen walking pace")
    ax.set_title(f"({'abcd'[i]})  {title}", loc="left", fontsize=13.1, weight="bold", pad=11)


def footer(fig, text):
    fig.text(0.068, 0.027, text, fontsize=8.6, color=MUTED)


def line_style(ax):
    ax.tick_params(axis="both", length=0)
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


def legend(fig, ax, *, columns=3):
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.54, 0.055),
               ncol=columns, fontsize=9.4, columnspacing=1.9)


def learning_curves(history, scores):
    fig, axes = canvas("Learning across held-out validation folds",
                      "Median validation macro-F1 by epoch over the folds and training seeds still active.",
                      left=0.10, right=0.96, top=0.837, bottom=0.17, hspace=0.47, wspace=0.24)
    for i, (ax, (ds, regime)) in enumerate(zip(axes, TASKS)):
        panel_title(ax, i, ds, regime, "recorded pace protocol")
        protocol = scores.loc[(scores.dataset == ds) & (scores.regime == regime)].iloc[0]
        seeds = [int(seed) for seed in protocol.seeds.split()]
        rows = history.loc[(history.dataset == ds) & (history.regime == regime) & history.seed.isin(seeds)]
        fits = protocol.folds * len(seeds)
        for model in COMPARISON:
            data = rows.loc[rows.model == model].groupby("epoch").validation_macro_f1
            median = data.median()
            x, y = median.index.to_numpy(), median.to_numpy()
            lo, hi = data.quantile(0.25).to_numpy(), data.quantile(0.75).to_numpy()
            active = data.count().to_numpy()
            stop = np.count_nonzero(active == fits)
            shade = PALETTE[model]
            ax.plot(x[:stop], y[:stop], color=shade, lw=2.7 if model == PRIMARY else 1.65,
                    label=NAMES[model], marker=MARKERS[model], ms=4 if model == PRIMARY else 3)
            if stop < len(x):
                tail = slice(max(0, stop-1), None)
                ax.plot(x[tail], y[tail], "--", color=shade, lw=1.35, alpha=0.8)
            if model == PRIMARY:
                ax.fill_between(x, lo, hi, color=shade, alpha=0.15, linewidth=0)
        ax.set_xlim(0.8, 14.4)
        ax.set_xticks([1, 4, 7, 10, 14])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation macro-F1")
        line_style(ax)
    legend(fig, axes[0])
    footer(fig, "Teal shading spans the Geometry model’s interquartile range. Dashed segments use fewer fits after early stopping.")
    return fig


def shift_curves(rotation, pace, sweeps):
    fig, axes = canvas("Recognition under sensor and walking-speed shifts",
                      "Models share held-out participants and perturbations. Teal shading shows the 95% participant interval.",
                      size=(16.1, 12.9), shape=(3, 3), left=0.093, right=0.966, top=0.851, bottom=0.144,
                      hspace=0.58, wspace=0.34)
    angles = [0, 15, 30, 60]
    angle_labels = [f"{angle}°" for angle in angles]
    pace = pace.assign(speed_order=pace.condition.map({"pace_slow": 0, "pace_medium": 1, "pace_fast": 2}))
    joint = sweeps.loc[(sweeps.dataset == "pamap2") & (sweeps.family == "rotation_with_outage")]
    gaps = sweeps.loc[(sweeps.dataset == "selfback") & (sweeps.family == "sample_gap")]
    panels = [
        (f"{label} · Mounting rotation", rotation.loc[rotation.dataset == ds],
         "angle", angles, angle_labels, "Simulated rotation")
        for ds, label in DATA_NAMES.items()
    ]
    panels.extend((
        ("SELFBACK · Recorded walking pace", pace, "speed_order", [0, 1, 2],
         ["Slow", "Medium", "Fast"], "Recorded pace"),
        ("PAMAP2 · Rotation with IMU loss", joint, "level", angles, angle_labels, "Rotation with one IMU absent"),
        ("SELFBACK · Missing samples", gaps, "level", [0, 8, 16, 32],
         ["0%", "6.25%", "12.5%", "25%"], "Contiguous fraction of window interpolated"),
    ))
    panels.extend(
        (f"{label} · Gaussian noise", sweeps.loc[(sweeps.dataset == ds) & (sweeps.family == "gaussian_noise")],
         "level", [0, .05, .1, .2, .35], ["0", "0.05", "0.10", "0.20", "0.35"], "Noise SD / training channel SD")
        for ds, label in DATA_NAMES.items()
    )
    axes[3].axvspan(0.76, 1.24, color="#e7edf5", alpha=0.72, zorder=0)
    for i, (ax, (title, ready, x_column, ticks, labels, axis_label)) in enumerate(zip(axes, panels, strict=True)):
        ax.set_title(f"({'abcdefghi'[i]})  {title}", loc="left", fontsize=12.0, weight="bold", pad=10)
        ax.set_xticks(ticks, labels)
        ax.set_xlabel(axis_label)
        for model in (*COMPARISON, "geometry_axes"):
            series = ready.loc[ready.model == model].sort_values(x_column)
            x, y = series[x_column].to_numpy(), series.macro_f1.to_numpy()
            shade = PALETTE[model]
            ax.plot(x, y, color=shade, marker=MARKERS[model], alpha=1 if model == PRIMARY else 0.86,
                    lw=2.7 if model == PRIMARY else 1.5, ms=6 if model == PRIMARY else 4,
                    label=NAMES[model], ls=":" if model == "geometry_axes" else "-")
            if model == PRIMARY:
                ax.fill_between(x, series.low.to_numpy(), series.high.to_numpy(), color=shade, alpha=0.13, linewidth=0)
        ax.set_ylabel("Test macro-F1")
        line_style(ax)
    for row in (axes[:3], axes[6:]):
        y_min = min(ax.get_ylim()[0] for ax in row)
        y_max = min(1.0, max(ax.get_ylim()[1] for ax in row))
        for ax in row:
            ax.set_ylim(y_min, y_max)
    legend(fig, axes[0])
    footer(fig, "Sensor simulations use seed 42. Recorded pace combines seeds 42 and 77. The shaded medium pace was used in training.")
    return fig


def confusions(counts):
    fig, axes = canvas("Which activities are confused?",
                      "Geometry TempoKAN on held-out participants. Each row pools predictions from both matched seeds.",
                      size=(13.1, 11.9), left=0.139, right=0.97, top=0.855, bottom=0.175, hspace=0.51, wspace=0.42)
    palette = LinearSegmentedColormap.from_list("recognition", ["#f4f7fc", "#b9dce3", "#5b9aab", "#246473", "#172e53"])
    replacements = {"Nordic walking": "Nordic walk", "Vacuum cleaning": "Vacuum", "Rope jumping": "Rope jump"}
    for i, (ax, (ds, regime)) in enumerate(zip(axes, TASKS)):
        panel_title(ax, i, ds, regime, "fast walking test")
        classes = CLASSES[ds]
        labels = [replacements.get(name, name) for name in classes]
        condition = "pace_fast" if regime == "pace" else "clean"
        block = counts.loc[(counts.dataset == ds) & (counts.regime == regime) &
                           (counts.model == PRIMARY) & (counts.condition == condition)]
        matrix = block.pivot(index="actual", columns="predicted", values="row_fraction").reindex(
            index=classes, columns=classes).to_numpy()
        image = ax.imshow(matrix, cmap=palette, vmin=0, vmax=1, interpolation="nearest", aspect="auto")
        for row, col in np.ndindex(matrix.shape):
            value = matrix[row, col]
            if row == col or value >= 0.045:
                ax.text(col, row, f"{100*value:.0f}", ha="center", va="center",
                        color="white" if value > 0.60 else INK, fontsize=8.5 if len(labels) < 10 else 7.3,
                        weight="bold" if row == col else "normal")
        coordinates = np.arange(len(labels))
        ax.set_xticks(coordinates, labels, rotation=33, ha="right", rotation_mode="anchor", fontsize=8.2)
        ax.set_yticks(coordinates, labels, fontsize=8.2)
        ax.set_xlabel("Predicted activity")
        ax.set_ylabel("Recorded activity")
        ax.tick_params(axis="both", length=0)
        ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.33)
        ax.tick_params(which="minor", bottom=False, left=False)
    cax = fig.add_axes((0.37, 0.071, 0.28, 0.013))
    bar = fig.colorbar(image, cax=cax, orientation="horizontal", ticks=np.linspace(0, 1, 5))
    bar.ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    bar.set_label("Share of the recorded activity")
    footer(fig, "Cells show row percentages. The fast-walking panel includes fast walking and all six nonwalking activities.")
    return fig


def benchmark_bars(scores):
    fig, axes = canvas("Macro-F1 on matched participant splits",
                      "Pooled held-out results from two matched training seeds. Whiskers show 95% participant-bootstrap intervals.",
                      size=(14.6, 9.3), left=0.207, right=0.961, top=0.837, bottom=0.10, hspace=0.46, wspace=0.54)
    shown = (PRIMARY, "geometry_mlp", "temporal_cnn", "kan_mixer", "kan_fe", "kan_har")
    for i, (ax, (ds, regime)) in enumerate(zip(axes, TASKS)):
        panel_title(ax, i, ds, regime)
        block = scores.loc[(scores.dataset == ds) & (scores.regime == regime)].set_index("model").loc[list(shown)]
        values = block.macro_f1.to_numpy()
        y = np.arange(len(shown))
        ax.barh(y, values, color=[PALETTE[model] for model in shown], height=0.58, edgecolor="white", linewidth=0.8)
        for j, row in enumerate(block.itertuples()):
            ax.hlines(j, row.low, row.high, color=INK, linewidth=1.15)
            ax.vlines([row.low, row.high], j-0.095, j+0.095, color=INK, linewidth=1.0)
            ax.scatter(row.macro_f1, j, s=18, color="white", edgecolor=INK, linewidth=0.7, zorder=3)
            ax.text(max(row.macro_f1, row.high)+0.018, j, f"{row.macro_f1:.2%}", ha="left", va="center",
                    fontsize=8.4, weight="bold" if row.Index == PRIMARY else "normal")
        ax.set_yticks(y, [NAMES[key] for key in shown], fontsize=8.4)
        ax.set_xlim(0, 1.15)
        ax.invert_yaxis()
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
        ax.set_xlabel("Test macro-F1")
        ax.xaxis.grid(color=GRID, linewidth=0.65)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", length=0)
    footer(fig, "The SELFBACK unseen-pace endpoint averages slow and fast macro-F1. The release includes all four ablation controls.")
    return fig


def render(root):
    scores = pd.read_csv(root / "Benchmark" / "Matched_Seeds.csv")
    history = pd.read_csv(root / "Training" / "Epoch_History.csv")
    rotation = pd.read_csv(root / "Robustness" / "Rotation.csv")
    pace = pd.read_csv(root / "Robustness" / "Matched_Pace.csv")
    sweeps = pd.read_csv(root / "Robustness" / "Stress_Sweeps.csv")
    counts = pd.read_csv(root / "Diagnostics" / "Matched_Confusions.csv")
    plots = (
        ("01_Learning_Curves", learning_curves, (history, scores)),
        ("02_Robustness_Curves", shift_curves, (rotation, pace, sweeps)),
        ("03_Confusion_Matrices", confusions, (counts,)),
        ("04_Benchmark_Bar_Chart", benchmark_bars, (scores,)),
    )
    folder = root / "Figures"
    folder.mkdir(parents=True, exist_ok=True)
    for filename, function, inputs in plots:
        fig = function(*inputs)
        for ext in ("pdf", "svg", "png"):
            fig.savefig(folder / f"{filename}.{ext}", dpi=200 if ext == "png" else 240)
        plt.close(fig)
        print(f"Rendered {filename}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=Path("Results"))
    render(p.parse_args().results)


if __name__ == "__main__":
    main()
