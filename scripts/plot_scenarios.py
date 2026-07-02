#!/usr/bin/env python3
"""Render paper-style RMSE-vs-SNR figures from
``experiments/runs/scenarios_*/{scenario}/results.csv``.

Produces:

  * One 2×2 grid figure with all four scenarios (Basic / Moderate /
    Advanced 1 OOD / Advanced 2 Crowded), log-y RMSE-vs-SNR with all
    four methods overlaid (raw_esprit, raw_root_music, reconunet_esprit,
    reconunet_root_music).  Mirrors paper Figs. 6, 8, 9, 10.
  * Per-scenario PNGs alongside each scenario's CSV.

Usage::

    python scripts/plot_scenarios.py \\
        --results-dir experiments/runs/scenarios_paper_v2/

    # or compare two runs side by side:
    python scripts/plot_scenarios.py \\
        --results-dir experiments/runs/scenarios_paper_v2/ \\
                      experiments/runs/scenarios_legacy_2025_fixed/ \\
        --labels "paper-faithful 2026-04-28" "legacy 2025-09-29"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import pandas as pd


SCENARIO_ORDER = ["basic", "moderate", "advanced1_ood", "advanced2_crowded"]
SCENARIO_TITLES = {
    "basic":             "Basic (1 source, no multipath)",
    "moderate":          "Moderate (2 src + 1 coherent replica)",
    "advanced1_ood":     "Advanced 1 / OOD (1 src + 6 replicas)",
    "advanced2_crowded": "Advanced 2 / Crowded (4 src + 3 replicas)",
}
METHOD_STYLE = {
    "raw_esprit":           dict(color="#1f77b4", linestyle="--", marker="s",
                                 label="ESPRIT (raw SCM)"),
    "raw_root_music":       dict(color="#ff7f0e", linestyle="--", marker="o",
                                 label="Root-MUSIC (raw SCM)"),
    "reconunet_esprit":     dict(color="#2ca02c", linestyle="-",  marker="s",
                                 label="ReconUNet + ESPRIT"),
    "reconunet_root_music": dict(color="#d62728", linestyle="-",  marker="o",
                                 label="ReconUNet + Root-MUSIC"),
}


def _load_run(results_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load every ``<scenario>/results.csv`` under ``results_dir``."""
    out: Dict[str, pd.DataFrame] = {}
    for scen in SCENARIO_ORDER:
        csv = results_dir / scen / "results.csv"
        if csv.exists():
            out[scen] = pd.read_csv(csv).sort_values(["method", "snr_db"])
    if not out:
        raise SystemExit(
            f"No <scenario>/results.csv found under {results_dir}.  "
            f"Expected one of {SCENARIO_ORDER!r}."
        )
    return out


def _plot_grid(runs: List[Dict[str, pd.DataFrame]],
               labels: List[str],
               output_path: Path) -> None:
    """2×2 grid of RMSE-vs-SNR, one panel per scenario.

    When ``len(runs) > 1``, the same method is drawn on each panel for
    every run, with the run label appended to the legend entry and a
    progressively dimmer alpha so the last run is solid and earlier runs
    faded.  Useful for direct paper-faithful-vs-legacy comparisons.
    """
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8), sharex=True)
    n_runs = len(runs)
    for ax, scen in zip(axes.ravel(), SCENARIO_ORDER):
        for run_idx, (run, label) in enumerate(zip(runs, labels)):
            if scen not in run:
                continue
            df = run[scen]
            alpha = 0.35 + 0.65 * (run_idx + 1) / n_runs
            for method, style in METHOD_STYLE.items():
                sub = df[df["method"] == method]
                if sub.empty:
                    continue
                lbl = style["label"] if n_runs == 1 else f"{style['label']} — {label}"
                ax.plot(sub["snr_db"], sub["rmse_deg"],
                        color=style["color"],
                        linestyle=style["linestyle"],
                        marker=style["marker"],
                        markersize=5,
                        linewidth=1.4,
                        alpha=alpha,
                        label=lbl)
        ax.set_yscale("log")
        ax.set_title(SCENARIO_TITLES[scen], fontsize=11)
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("RMSPE (deg)")
        ax.grid(True, which="both", alpha=0.25)
    # One shared legend below the grid.
    handles, labels_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=2 if n_runs == 1 else 1,
               bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=9)
    fig.suptitle("RMSE vs SNR across paper §IV scenarios", fontsize=13, y=0.99)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_one(scen: str, runs: List[Dict[str, pd.DataFrame]],
              labels: List[str], output_path: Path) -> None:
    """Single-scenario PNG (used only for the first run)."""
    if scen not in runs[0]:
        return
    df = runs[0][scen]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for method, style in METHOD_STYLE.items():
        sub = df[df["method"] == method]
        if sub.empty:
            continue
        ax.plot(sub["snr_db"], sub["rmse_deg"],
                color=style["color"], linestyle=style["linestyle"],
                marker=style["marker"], markersize=5, linewidth=1.4,
                label=style["label"])
    ax.set_yscale("log")
    ax.set_title(SCENARIO_TITLES[scen])
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("RMSPE (deg)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="plot_scenarios",
                                description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", "-r", required=True, type=Path, nargs="+",
                   help="One or more directories containing "
                        "<scenario>/results.csv.")
    p.add_argument("--labels", "-l", default=None, nargs="+",
                   help="Human-readable label per --results-dir (default: dir basename).")
    p.add_argument("--output-dir", "-o", default=None, type=Path,
                   help="Where to write the figures.  Default: the first --results-dir.")
    args = p.parse_args(argv)

    if args.labels and len(args.labels) != len(args.results_dir):
        raise SystemExit("--labels must have the same length as --results-dir")
    labels = args.labels or [d.name for d in args.results_dir]
    output_dir = args.output_dir or args.results_dir[0]
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = [_load_run(d) for d in args.results_dir]

    grid_path = output_dir / "rmse_vs_snr_grid.png"
    _plot_grid(runs, labels, grid_path)
    print(f"wrote {grid_path}")

    # Per-scenario PNGs only for the first run (the canonical one).
    for scen in SCENARIO_ORDER:
        single_path = (args.results_dir[0] / scen / "rmse_vs_snr.png")
        _plot_one(scen, runs, labels, single_path)
        if single_path.exists():
            print(f"wrote {single_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
