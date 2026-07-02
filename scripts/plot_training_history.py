#!/usr/bin/env python3
"""Render training-curve figures from
``experiments/runs/<run>/checkpoints/history.json``.

The trainer (``reconunet.cli.train``) writes one JSON record per epoch
with fields: ``epoch, train, val, val_rmspe_deg, lr``.  This script reads
that file and produces:

  * A 3-panel figure: train/val composite loss (left), val RMSPE in
    degrees (middle), learning-rate schedule (right).  The "best epoch"
    (lowest val loss — the same signal early-stopping tracks) is marked
    with a vertical dashed line on every panel.
  * A standalone single-panel "loss only" figure that mirrors paper Fig. 5.

Usage::

    python scripts/plot_training_history.py \\
        --history experiments/runs/reconunet_paper/checkpoints/history.json

    # overlay multiple runs:
    python scripts/plot_training_history.py \\
        --history experiments/runs/reconunet_paper/checkpoints/history.json \\
                  experiments/runs/reconunet/checkpoints/history.json \\
        --labels "paper-faithful 2026-04-28" "legacy mini" \\
        --output-dir experiments/runs/training_comparison/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt


def _load_history(path: Path) -> Dict[str, list]:
    with path.open() as fh:
        rows = json.load(fh)
    return {
        "epoch":         [r["epoch"]          for r in rows],
        "train":         [r["train"]          for r in rows],
        "val":           [r["val"]            for r in rows],
        "val_rmspe_deg": [r["val_rmspe_deg"]  for r in rows],
        "lr":            [r["lr"]             for r in rows],
    }


def _best_epoch(h: Dict[str, list]) -> int:
    """Return the epoch index that minimised val loss — the signal the
    trainer's early-stopping tracks."""
    return h["epoch"][min(range(len(h["val"])), key=lambda i: h["val"][i])]


_TRAIN_COLOR = "#1f77b4"   # matplotlib tab:blue
_VAL_COLOR   = "#ff7f0e"   # matplotlib tab:orange


def _plot_three_panel(runs: List[Dict[str, list]], labels: List[str],
                      output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    n_runs = len(runs)
    for run_idx, (h, label) in enumerate(zip(runs, labels)):
        # When overlaying multiple runs, distinguish them by linestyle
        # rather than colour so train stays blue and val stays orange.
        ls = ["-", "--", ":", "-."][run_idx % 4]
        run_label = "" if n_runs == 1 else f" — {label}"
        best = _best_epoch(h)
        # --- Loss panel: train (blue) + val (orange) ------------------------
        axes[0].plot(h["epoch"], h["train"], color=_TRAIN_COLOR, linestyle=ls,
                     linewidth=1.6, label=f"train{run_label}")
        axes[0].plot(h["epoch"], h["val"],   color=_VAL_COLOR,   linestyle=ls,
                     linewidth=1.6, label=f"validation{run_label}")
        axes[0].axvline(best, color="grey", linestyle="--",
                        linewidth=0.8, alpha=0.5)
        # --- val_rmspe panel ------------------------------------------------
        axes[1].plot(h["epoch"], h["val_rmspe_deg"], color=_VAL_COLOR,
                     linestyle=ls, linewidth=1.6,
                     label=f"validation{run_label}")
        axes[1].axvline(best, color="grey", linestyle="--",
                        linewidth=0.8, alpha=0.5)
        # --- LR panel -------------------------------------------------------
        axes[2].plot(h["epoch"], h["lr"], color="black",
                     linestyle=ls, linewidth=1.4, drawstyle="steps-post",
                     label=label if n_runs > 1 else None)

    axes[0].set_yscale("log")
    axes[0].set_title("Composite loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend(fontsize=9)

    axes[1].set_yscale("log")
    axes[1].set_title("Validation RMSPE")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("val RMSPE (deg)")
    axes[1].grid(True, which="both", alpha=0.25)
    if n_runs > 1:
        axes[1].legend(fontsize=9)

    axes[2].set_yscale("log")
    axes[2].set_title("Learning rate (ReduceLROnPlateau)")
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("lr")
    axes[2].grid(True, which="both", alpha=0.25)
    if n_runs > 1:
        axes[2].legend(fontsize=9)

    fig.suptitle("Training history", fontsize=13, y=1.00)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_loss_only(runs: List[Dict[str, list]], labels: List[str],
                    output_path: Path) -> None:
    """Paper Fig. 5 style: train (blue) + val (orange) vs epoch."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    n_runs = len(runs)
    for run_idx, (h, label) in enumerate(zip(runs, labels)):
        ls = ["-", "--", ":", "-."][run_idx % 4]
        run_label = "" if n_runs == 1 else f" — {label}"
        ax.plot(h["epoch"], h["train"], color=_TRAIN_COLOR, linestyle=ls,
                linewidth=1.6, label=f"train{run_label}")
        ax.plot(h["epoch"], h["val"],   color=_VAL_COLOR,   linestyle=ls,
                linewidth=1.6, label=f"validation{run_label}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("composite loss")
    ax.set_title("Total loss over epochs")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="plot_training_history",
                                description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--history", "-H", required=True, type=Path, nargs="+",
                   help="One or more history.json files.")
    p.add_argument("--labels", "-l", default=None, nargs="+",
                   help="Label per --history (default: parent dir name).")
    p.add_argument("--output-dir", "-o", default=None, type=Path,
                   help="Where to write the figures.  "
                        "Default: parent of the first history.json.")
    args = p.parse_args(argv)

    if args.labels and len(args.labels) != len(args.history):
        raise SystemExit("--labels must have the same length as --history")
    labels = args.labels or [
        h.parent.parent.name + ("/" + h.parent.name if h.parent.name != "checkpoints" else "")
        for h in args.history
    ]
    output_dir = args.output_dir or args.history[0].parent.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = [_load_history(h) for h in args.history]

    three_panel = output_dir / "training_curves.png"
    loss_only   = output_dir / "training_loss.png"
    _plot_three_panel(runs, labels, three_panel)
    _plot_loss_only(runs, labels, loss_only)

    # Print a one-line summary per run.
    print(f"wrote {three_panel}")
    print(f"wrote {loss_only}")
    print()
    for h, label in zip(runs, labels):
        best = _best_epoch(h)
        i = h["epoch"].index(best)
        print(f"  {label}:  best epoch={best}  val={h['val'][i]:.4f}  "
              f"val_rmspe={h['val_rmspe_deg'][i]:.3f}°  "
              f"final_lr={h['lr'][-1]:.1e}  total_epochs={len(h['epoch'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
