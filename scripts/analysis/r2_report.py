#!/usr/bin/env python3
"""Sensors revision round 2 (2026-09-30) helpers for the R2 pipeline.

    r2_report.py train-summary --run-dir experiments/runs/reconunet_c_paper --status-log S -o CSV
    r2_report.py export --eval-dir E --sweeps-dir SW --run-dir RUN --dest docs/revision_r2

train-summary  ReconUNet-C training outcome (epochs, best epoch, best validation loss, validation
               RMSPE, wall-clock) next to the released full ReconUNet (R1) for reference.
export         copies the R2 result CSVs into --dest under the R1 file names, derives compact
               tables (Table II/III ReconUNet vs ReconUNet-C back ends at every SNR; bandwidth sweep
               wide), and writes HEADLINE.md with the headline numbers.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from revision_facts import md_table, status_elapsed  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def hist_row(label: str, run_dir: Path, elapsed: str) -> dict | None:
    h = run_dir / "checkpoints" / "history.json"
    if not h.exists(): return None
    rows = json.load(h.open())
    if not rows: return None
    best = min(rows, key=lambda r: r["val"])
    ep = len(rows); mins = float(elapsed.replace("min", "")) if elapsed else float("nan")
    return {"model": label, "epochs_run": ep, "best_epoch": best["epoch"], "best_val_loss": round(best["val"], 5),
            "val_rmspe_at_best_deg": round(best["val_rmspe_deg"], 3), "min_val_rmspe_deg": round(min(r["val_rmspe_deg"] for r in rows), 3),
            "last_lr": f"{rows[-1]['lr']:.1e}", "early_stopped": ep < 300, "wall_clock_min": elapsed.replace("min", ""),
            "min_per_epoch": round(mins / ep, 2) if ep else ""}


def train_summary(a) -> int:
    rows = [hist_row("ReconUNet-C (R2, L_rec only)", a.run_dir, status_elapsed(a.status_log).get("train_reconunet_c", "")),
            hist_row("ReconUNet (R1 released, composite loss)", REPO / "experiments/runs/reconunet_paper",
                     status_elapsed(REPO / "experiments/runs/revision_retrain_20260906/status.log").get("reconunet", ""))]
    rows = [r for r in rows if r]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(a.output, index=False)
    print(pd.DataFrame(rows).to_string(index=False)); print(f"[r2] wrote {a.output}")
    return 0


def _rel(p: Path) -> str:
    return str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p)


def _cp(src: Path, dst: Path, log: list):
    if src.exists():
        shutil.copy2(src, dst); log.append(f"| `{dst.name}` | `{_rel(src)}` |")
    else:
        log.append(f"| `{dst.name}` | _missing: {_rel(src)}_ |")


def export(a) -> int:
    E, SW, RUN, D = a.eval_dir, a.sweeps_dir, a.run_dir, a.dest
    D.mkdir(parents=True, exist_ok=True)
    log = ["| file | source |", "|---|---|"]
    for src, name in [(E / "paper_testset_r2/paper_testset_by_K.csv", "paper_testset_by_K.csv"),
                      (E / "paper_testset_r2/paper_testset_r2_ci.csv", "paper_testset_ci.csv"),
                      (E / "scenario_sweep_r2/scenario_sweep.csv", "scenario_sweep.csv"),
                      (E / "scenario_sweep_r2/scenario_sweep_r2_ci.csv", "scenario_sweep_ci.csv"),
                      (E / "scenario_sweep_r2/scenario_sweep_grid.png", "scenario_sweep_grid.png"),
                      (E / "table2_mild_r2/table2_full.csv", "table2_mild_full.csv"),
                      (E / "table2_mild_r2/table2_mild_r2_ci.csv", "table2_mild_ci.csv"),
                      (E / "table2_harsh_r2/table2_full.csv", "table2_harsh_full.csv"),
                      (E / "table2_harsh_r2/table2_harsh_r2_ci.csv", "table2_harsh_ci.csv"),
                      (SW / "snapshot_sweep.csv", "snapshot_sweep.csv"), (SW / "separation_sweep.csv", "separation_sweep.csv"),
                      (SW / "music_verification.csv", "music_verification.csv"),
                      (SW / "cost_summary_r2.csv", "cost_summary.csv"), (SW / "cost_latency_r2.csv", "cost_latency.csv"),
                      (SW / "bandwidth_sweep.csv", "bandwidth_sweep.csv"), (SW / "bandwidth_sweep_coupled.csv", "bandwidth_sweep_coupled.csv"),
                      (SW / "bandwidth_autocorr.csv", "bandwidth_autocorr.csv"), (SW / "bandwidth_autocorr_coupled.csv", "bandwidth_autocorr_coupled.csv"),
                      (RUN / "reconunet_c_training.csv", "reconunet_c_training.csv")]:
        _cp(src, D / name, log)
    # Table II / III: ReconUNet vs ReconUNet-C, every back end, every SNR (pooled RMSE, deg)
    for preset in ("mild", "harsh"):
        f = E / f"table2_{preset}_r2/table2_full.csv"
        if f.exists():
            df = pd.read_csv(f)
            keep = [m for m in df.method.unique() if m.startswith("ReconUNet")] + ["Root-MUSIC", "CRLB"]
            w = df[df.method.isin(keep)].pivot_table(index=["scenario", "snr_db"], columns="method", values="rmse_deg").reset_index()
            w = w[["scenario", "snr_db"] + [m for m in keep if m in w.columns]]
            out = D / f"table2_{preset}_reconunet_backends_by_snr.csv"; w.to_csv(out, index=False)
            log.append(f"| `{out.name}` | derived from `table2_{preset}_full.csv` |")
    # Bandwidth sweep, wide (pooled RMSE, deg)
    for suf in ("", "_coupled"):
        f = SW / f"bandwidth_sweep{suf}.csv"
        if f.exists():
            df = pd.read_csv(f, dtype={"bw_frac": str})
            w = df.pivot_table(index=["scenario", "bw_frac"], columns="method", values="rmse_deg", sort=False).reset_index()
            out = D / f"bandwidth_sweep{suf}_wide.csv"; w.to_csv(out, index=False)
            log.append(f"| `{out.name}` | derived from `bandwidth_sweep{suf}.csv` |")
    (D / "README.md").write_text(
        "# Revision round 2 (2026-09-30) — result CSVs\n\nCopied by `scripts/analysis/r2_report.py export` at the end of "
        "`scripts/run_revision_r2_pipeline.sh`. Same column schema as the R1 CSVs; the covariance-only network is the "
        "method `ReconUNet-C` (Table II/III: `ReconUNet-C+<back end>`), next to the full `ReconUNet`. DA-MUSIC = v2 ensemble "
        "(`experiments/runs/damusic_paper/ensemble_v2`). CI files: percentile bootstrap over scenes, 1000 resamples, seed 20260920.\n\n"
        + "\n".join(log) + "\n")
    headline(D)
    print(f"[r2] exported to {D}")
    return 0


def headline(D: Path) -> None:
    L = ["# R2 headline numbers (2026-09-30)", ""]
    ci = D / "paper_testset_ci.csv"
    if ci.exists():
        df = pd.read_csv(ci); df = df[df.scenario == "all K"].sort_values("rmse_deg")
        L += ["## Paper test split, pooled over K = 1..4", "",
              "| method | pooled RMSE (°) | 95 % CI | median RMSPE (°) | 95 % CI | scenes |", "|---|---|---|---|---|---|"]
        for _, r in df.iterrows():
            L.append(f"| {r.method} | {r.rmse_deg:.2f} | [{r.rmse_ci_lo:.2f}, {r.rmse_ci_hi:.2f}] | {r.median_rmspe_deg:.2f} | "
                     f"[{r.median_ci_lo:.2f}, {r.median_ci_hi:.2f}] | {r.n_scenes} |")
        L.append("")
    tr = D / "reconunet_c_training.csv"
    if tr.exists():
        L += ["## Training", "", md_table(pd.read_csv(tr).to_dict("records"), fmt="{:g}"), ""]
    for suf, title in (("", "replica delays fixed at the training bandwidth 0.05 (decoupled, primary)"),
                       ("_coupled", "replica delays scale with 1/bw (coupled, secondary)")):
        f = D / f"bandwidth_sweep{suf}_wide.csv"
        if f.exists():
            L += [f"## Bandwidth sweep, pooled RMSE (°), 0 dB — {title}", "", md_table(pd.read_csv(f, dtype={"bw_frac": str}).to_dict("records"), fmt="{:.2f}"), ""]
    ac = D / "bandwidth_autocorr.csv"
    if ac.exists():
        df = pd.read_csv(ac, dtype={"bw_frac": str})
        df = df[df.scenario == df.scenario.iloc[0]][["bw_frac"] + [f"r{l}" for l in range(8)] + ["gamma_mean"]]
        L += ["## Measured source autocorrelation |r(l)| and direct/replica |γ| (decoupled delays, first scenario)", "",
              md_table(df.to_dict("records"), fmt="{:.3f}"), ""]
    (D / "HEADLINE.md").write_text("\n".join(L))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train-summary"); t.add_argument("--run-dir", type=Path, required=True)
    t.add_argument("--status-log", type=Path, required=True); t.add_argument("--output", "-o", type=Path, required=True)
    e = sub.add_parser("export")
    for k in ("--eval-dir", "--sweeps-dir", "--run-dir", "--dest"): e.add_argument(k, type=Path, required=True)
    a = ap.parse_args()
    for k in ("run_dir", "eval_dir", "sweeps_dir", "dest", "status_log", "output"):
        if getattr(a, k, None) is not None: setattr(a, k, getattr(a, k).resolve())
    return train_summary(a) if a.cmd == "train-summary" else export(a)


if __name__ == "__main__":
    raise SystemExit(main())
