#!/usr/bin/env python3
"""Re-derive every table and figure of the revision results from the tracked
CSVs alone — no GPU, no checkpoints, no rendered scenes required.

Inputs (all tracked in git, written by the three evaluation scripts):

  experiments/runs/eval_coherent_20260910/
    paper_testset/paper_testset_by_K.csv   compare_paper_testset.py
    scenario_sweep/scenario_sweep.csv      scenario_sweep.py
    table2_mild/table2_full.csv            reproduce_table2.py
    table2_harsh/table2_full.csv           reproduce_table2.py --scenarios-root data/scenes/scenarios_harsh

Outputs (``--output-dir``, default ``<eval-dir>/derived/``):

  table2_mild_0dB.md, table2_harsh_0dB.md   Table II view: pooled RMSE (deg) at 0 dB,
                                            methods x scenarios, best per column marked
  paper_testset_by_K.md                     pooled RMSE and median per method per K
  scenario_sweep_grid.png                   2x2 RMSE-vs-SNR grid (log-y) with the CRLB
  table2_<preset>_vs_snr_<scenario>.md      full RMSE-vs-SNR tables behind the grid

Usage::

    python scripts/analysis/render_results_from_csv.py
    python scripts/analysis/render_results_from_csv.py --snr 5 --output-dir /tmp/tables
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
EVAL = REPO / "experiments/runs/eval_coherent_20260910"

SCEN_ORDER = ["basic", "moderate", "advanced1_ood", "advanced2_crowded"]
SCEN_LABEL = {"basic": "Basic (K=1)", "moderate": "Moderate (K=2 +1)",
              "advanced1_ood": "OOD (K=1 +6)", "advanced2_crowded": "Crowded (K=4 +3)"}
METHOD_ORDER = ["Bartlett", "MVDR", "MUSIC", "Root-MUSIC", "ESPRIT", "Unitary-ESPRIT",
                "ReconUNet+Root-MUSIC", "ReconUNet+MUSIC", "ReconUNet+ESPRIT", "ReconUNet+Unitary-ESPRIT",
                "SubspaceNet", "SubViT", "DA-MUSIC", "CRLB"]
BOUNDS = {"CRLB"}


def _md(df: pd.DataFrame, fmt: str = "{:.2f}", mark_best_cols: bool = False, skip_rows=()) -> str:
    """Minimal GitHub-markdown table (avoids a `tabulate` dependency)."""
    cols = list(df.columns)
    best = {}
    if mark_best_cols:
        body = df.drop(index=[r for r in skip_rows if r in df.index], errors="ignore")
        best = {c: body[c].idxmin() for c in cols if pd.api.types.is_numeric_dtype(body[c])}
    lines = ["| " + " | ".join([df.index.name or ""] + [str(c) for c in cols]) + " |",
             "|" + "---|" * (len(cols) + 1)]
    for idx, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            s = fmt.format(v) if isinstance(v, (int, float)) and pd.notna(v) else ("" if pd.isna(v) else str(v))
            if best.get(c) == idx:
                s = f"**{s}**"
            cells.append(s)
        lines.append("| " + " | ".join([str(idx)] + cells) + " |")
    return "\n".join(lines) + "\n"


def table2_view(csv: Path, snr_db: float) -> pd.DataFrame:
    df = pd.read_csv(csv)
    t = df[df["snr_db"] == snr_db].pivot(index="method", columns="scenario", values="rmse_deg")
    t = t.reindex([m for m in METHOD_ORDER if m in t.index])
    t = t[[s for s in SCEN_ORDER if s in t.columns]].rename(columns=SCEN_LABEL)
    t.index.name = f"pooled RMSE (deg) @ {snr_db:g} dB"
    return t


def table2_vs_snr(csv: Path, scenario: str) -> pd.DataFrame:
    df = pd.read_csv(csv)
    t = df[df["scenario"] == scenario].pivot(index="method", columns="snr_db", values="rmse_deg")
    t = t.reindex([m for m in METHOD_ORDER if m in t.index])
    t.columns = [f"{c:g} dB" for c in t.columns]
    t.index.name = f"{SCEN_LABEL.get(scenario, scenario)}: pooled RMSE (deg)"
    return t


def per_k_tables(csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(csv).set_index("K")
    methods = sorted({c.rsplit("_", 1)[0] for c in df.columns if c.endswith("_mean")},
                     key=lambda m: ["R-MUSIC", "ESPRIT", "SubspaceNet", "SubViT", "DA-MUSIC", "ReconUNet"].index(m)
                     if m in ["R-MUSIC", "ESPRIT", "SubspaceNet", "SubViT", "DA-MUSIC", "ReconUNet"] else 99)
    mean = df[[f"{m}_mean" for m in methods]].rename(columns=lambda c: c[:-5]).T
    med = df[[f"{m}_med" for m in methods]].rename(columns=lambda c: c[:-4]).T
    for t, name in ((mean, "pooled RMSE (deg) by true K"), (med, "median per-scene RMSPE (deg) by true K")):
        t.columns = [f"K={c}" if c != "all" else "all K" for c in t.columns]
        t.index.name = name
    return mean, med


def sweep_figure(csv: Path, out: Path) -> None:
    df = pd.read_csv(csv)
    methods = [c for c in df.columns if c not in ("scenario", "snr_db", "n", "crlb_deg")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for ax, sc in zip(axes.ravel(), SCEN_ORDER):
        d = df[df["scenario"] == sc].sort_values("snr_db")
        for m in methods:
            ax.semilogy(d["snr_db"], d[m], marker="o", ms=3.5, lw=1.4,
                        label=m, color="#1f4e79" if m == "ReconUNet" else None,
                        zorder=3 if m == "ReconUNet" else 2)
        ax.semilogy(d["snr_db"], d["crlb_deg"], "k--", lw=1, label="CRLB")
        ax.set_title(SCEN_LABEL[sc]); ax.grid(True, which="both", alpha=0.3)
        ax.set_ylabel("RMSE (deg)")
    for ax in axes[1]:
        ax.set_xlabel("SNR (dB)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)
    fig.suptitle("Mean direct-path RMSE vs SNR, paper §IV scenarios (corrected renderer, 2026-09-10)")
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", type=Path, default=EVAL)
    ap.add_argument("--output-dir", "-o", type=Path, default=None)
    ap.add_argument("--snr", type=float, default=0.0, help="SNR (dB) for the Table II view")
    a = ap.parse_args()
    out = a.output_dir or (a.eval_dir / "derived")
    out.mkdir(parents=True, exist_ok=True)

    for preset in ("mild", "harsh"):
        csv = a.eval_dir / f"table2_{preset}" / "table2_full.csv"
        if not csv.exists():
            print(f"[skip] {csv} missing"); continue
        t = table2_view(csv, a.snr)
        md = _md(t, mark_best_cols=True, skip_rows=BOUNDS)
        (out / f"table2_{preset}_{a.snr:g}dB.md").write_text(md)
        print(f"\n## Table II — {preset} imperfections, {a.snr:g} dB\n\n{md}")
        for sc in SCEN_ORDER:
            (out / f"table2_{preset}_vs_snr_{sc}.md").write_text(_md(table2_vs_snr(csv, sc)))

    csv = a.eval_dir / "paper_testset" / "paper_testset_by_K.csv"
    if csv.exists():
        mean, med = per_k_tables(csv)
        md = _md(mean, mark_best_cols=True) + "\n" + _md(med, mark_best_cols=True)
        (out / "paper_testset_by_K.md").write_text(md)
        print(f"\n## Paper test split, per true K\n\n{md}")

    csv = a.eval_dir / "scenario_sweep" / "scenario_sweep.csv"
    if csv.exists():
        sweep_figure(csv, out / "scenario_sweep_grid.png")
        print(f"\nfigure -> {out / 'scenario_sweep_grid.png'}")
    print(f"\nall outputs in {out}")


if __name__ == "__main__":
    main()
