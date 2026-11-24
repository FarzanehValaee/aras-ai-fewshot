#!/usr/bin/env python3
"""Publication-ready bar chart of Spider execution accuracy (3×3 paper summary)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

ROOT = Path("/var/www/aras-ai-data/benchmark/paper_runs")
DEFAULT_SUMMARY = ROOT / "summary_3x3.json"
DEFAULT_STEM = ROOT / "accuracy_comparison"

# Display order and paper labels.
METHODS = (
    ("nofew", "No-Shot\n(Baseline)"),
    ("jaccard", "Jaccard\nFew-Shot"),
    ("skeleton", "Skeleton Few-Shot\n(Ours)"),
)

# Muted academic blues/teal (print-friendly, colorblind-safer).
BAR_COLORS = ("#6BAED6", "#3182BD", "#1B9E77")
EDGE_COLOR = "#1B2838"
ERROR_COLOR = "#2C3E50"


def load_metrics(path: Path) -> tuple[list[str], list[float], list[float], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    modes = payload.get("modes") or {}
    labels: list[str] = []
    means: list[float] = []
    stdevs: list[float] = []
    for key, label in METHODS:
        block = modes.get(key)
        if not isinstance(block, dict):
            raise SystemExit(f"missing mode {key!r} in {path}")
        mean = block.get("mean")
        stdev = block.get("stdev")
        if mean is None:
            accs = block.get("accuracies") or [
                run.get("accuracy_pct") for run in (block.get("runs") or [])
            ]
            accs = [float(x) for x in accs if x is not None]
            if not accs:
                raise SystemExit(f"no accuracy values for mode {key!r}")
            mean = float(np.mean(accs))
            stdev = float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0
        labels.append(label)
        means.append(float(mean))
        stdevs.append(float(stdev or 0.0))
    model = str(payload.get("model") or "model")
    return labels, means, stdevs, model


def y_limits(means: list[float], stdevs: list[float]) -> tuple[float, float]:
    low = min(m - s for m, s in zip(means, stdevs))
    high = max(m + s for m, s in zip(means, stdevs))
    # Zoom to gaps while keeping room for value labels / error caps.
    ymin = max(0.0, np.floor((low - 4.0) / 2.0) * 2.0)
    ymax = min(100.0, np.ceil((high + 4.0) / 2.0) * 2.0)
    if ymax - ymin < 12:
        mid = 0.5 * (ymin + ymax)
        ymin, ymax = mid - 6, mid + 6
    return float(ymin), float(ymax)


def plot_comparison(
    labels: list[str],
    means: list[float],
    stdevs: list[float],
    *,
    model: str,
    out_stem: Path,
) -> list[Path]:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "axes.linewidth": 0.9,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    x = np.arange(len(labels))
    bars = ax.bar(
        x,
        means,
        width=0.62,
        color=BAR_COLORS,
        edgecolor=EDGE_COLOR,
        linewidth=0.8,
        yerr=stdevs,
        capsize=5,
        error_kw={
            "ecolor": ERROR_COLOR,
            "elinewidth": 1.1,
            "capthick": 1.1,
            "zorder": 3,
        },
        zorder=2,
    )

    ymin, ymax = y_limits(means, stdevs)
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel("Execution Accuracy (%)")
    ax.set_xlabel("Method")
    ax.set_xticks(x, labels)
    ax.set_title("Spider Dev Execution Accuracy by Retrieval Method")
    ax.yaxis.set_major_locator(MultipleLocator(2))
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.45, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # Exact mean labels above error-bar caps.
    for bar, mean, stdev in zip(bars, means, stdevs):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            mean + stdev + 0.35,
            f"{mean:.2f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color=EDGE_COLOR,
        )

    ax.text(
        0.99,
        0.02,
        f"model: {model}  ·  mean±stdev over 3 seeds",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#4A5568",
    )

    fig.tight_layout()
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for ext in (".png", ".pdf"):
        path = out_stem.with_suffix(ext)
        fig.savefig(path, bbox_inches="tight")
        written.append(path)
    plt.close(fig)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY,
        help="Path to summary_3x3.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_STEM,
        help="Output path stem (writes .png and .pdf)",
    )
    args = parser.parse_args()
    if not args.summary.exists():
        raise SystemExit(f"missing summary file: {args.summary}")

    labels, means, stdevs, model = load_metrics(args.summary)
    paths = plot_comparison(
        labels, means, stdevs, model=model, out_stem=args.out
    )
    print("methods:", ", ".join(METHODS[i][0] for i in range(len(means))))
    print("means:", ", ".join(f"{m:.4f}" for m in means))
    print("stdevs:", ", ".join(f"{s:.4f}" for s in stdevs))
    for path in paths:
        print(f"wrote {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
