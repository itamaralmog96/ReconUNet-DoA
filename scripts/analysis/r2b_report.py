#!/usr/bin/env python3
"""Sensors revision R2b (2026-10-01) helpers: ReconUNet-CB (randomised source bandwidth).

    r2b_report.py train-summary --status-log S -o CSV
    r2b_report.py export --eval-dir E --sweeps-dir SW --run-dir RUN --dest docs/revision_r2b

train-summary  ReconUNet-CB training outcome next to ReconUNet-C (R2) and the full ReconUNet (R1).
export         copies the R2b CSVs into --dest (R2 file names), derives wide tables and writes
               HEADLINE.md, including an automatically generated plain statement (thresholds stated
               in the text) of whether randomised-bandwidth training removed the wideband failure.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r2_report import _cp, hist_row  # noqa: E402
from revision_facts import md_table, status_elapsed  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
ORDER = ["Root-MUSIC", "ReconUNet", "ReconUNet-C", "ReconUNet-CB", "SubspaceNet", "DA-MUSIC", "SubViT"]
WIDE_BW = ["0.2", "0.4", "white"]


def train_summary(a) -> int:
    rows = [hist_row("ReconUNet-CB (R2b, L_rec only, randomised bw)", REPO / "experiments/runs/reconunet_cb_paper",
                     status_elapsed(a.status_log).get("train_reconunet_cb", "")),
            hist_row("ReconUNet-C (R2, L_rec only, bw 0.05)", REPO / "experiments/runs/reconunet_c_paper",
                     status_elapsed(REPO / "experiments/runs/revision_r2_20260930/status.log").get("train_reconunet_c", "")),
            hist_row("ReconUNet (R1 released, composite loss, bw 0.05)", REPO / "experiments/runs/reconunet_paper",
                     status_elapsed(REPO / "experiments/runs/revision_retrain_20260906/status.log").get("reconunet", ""))]
    rows = [r for r in rows if r]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(a.output, index=False)
    print(pd.DataFrame(rows).to_string(index=False)); print(f"[r2b] wrote {a.output}")
    return 0


def wide(f: Path, index: list[str]) -> pd.DataFrame | None:
    if not f.exists(): return None
    df = pd.read_csv(f, dtype={"bw_frac": str})
    w = df.pivot_table(index=index, columns="method", values="rmse_deg", sort=False).reset_index()
    return w[index + [m for m in ORDER if m in w.columns]]


def export(a) -> int:
    E, SW, RUN, D = a.eval_dir, a.sweeps_dir, a.run_dir, a.dest
    D.mkdir(parents=True, exist_ok=True)
    log = ["| file | source |", "|---|---|"]
    for src, name in [(E / "paper_testset_r2b/paper_testset_by_K.csv", "paper_testset_by_K.csv"),
                      (E / "paper_testset_r2b/paper_testset_r2b_ci.csv", "paper_testset_ci.csv"),
                      (E / "paper_testset_bwrand_r2b/paper_testset_by_K.csv", "paper_testset_bwrand_by_K.csv"),
                      (E / "paper_testset_bwrand_r2b/paper_testset_bwrand_r2b_ci.csv", "paper_testset_bwrand_ci.csv"),
                      (E / "scenario_sweep_r2b/scenario_sweep.csv", "scenario_sweep.csv"),
                      (E / "scenario_sweep_r2b/scenario_sweep_r2b_ci.csv", "scenario_sweep_ci.csv"),
                      (E / "scenario_sweep_r2b/scenario_sweep_grid.png", "scenario_sweep_grid.png"),
                      (E / "table2_mild_r2b/table2_full.csv", "table2_mild_full.csv"),
                      (E / "table2_mild_r2b/table2_mild_r2b_ci.csv", "table2_mild_ci.csv"),
                      (E / "table2_harsh_r2b/table2_full.csv", "table2_harsh_full.csv"),
                      (E / "table2_harsh_r2b/table2_harsh_r2b_ci.csv", "table2_harsh_ci.csv"),
                      (SW / "snapshot_sweep.csv", "snapshot_sweep.csv"), (SW / "separation_sweep.csv", "separation_sweep.csv"),
                      (SW / "cost_summary_r2.csv", "cost_summary.csv"), (SW / "cost_latency_r2.csv", "cost_latency.csv"),
                      (SW / "bandwidth_sweep.csv", "bandwidth_sweep.csv"), (SW / "bandwidth_sweep_coupled.csv", "bandwidth_sweep_coupled.csv"),
                      (SW / "bandwidth_sweep_m5dB.csv", "bandwidth_sweep_m5dB.csv"), (SW / "bandwidth_sweep_p10dB.csv", "bandwidth_sweep_p10dB.csv"),
                      (SW / "bandwidth_autocorr.csv", "bandwidth_autocorr.csv"), (SW / "bandwidth_autocorr_coupled.csv", "bandwidth_autocorr_coupled.csv"),
                      (RUN / "reconunet_cb_training.csv", "reconunet_cb_training.csv")]:
        _cp(src, D / name, log)
    for preset in ("mild", "harsh"):
        f = E / f"table2_{preset}_r2b/table2_full.csv"
        if f.exists():
            df = pd.read_csv(f)
            keep = [m for m in df.method.unique() if m.startswith("ReconUNet")] + ["Root-MUSIC", "CRLB"]
            w = df[df.method.isin(keep)].pivot_table(index=["scenario", "snr_db"], columns="method", values="rmse_deg").reset_index()
            w = w[["scenario", "snr_db"] + [m for m in keep if m in w.columns]]
            out = D / f"table2_{preset}_reconunet_backends_by_snr.csv"; w.to_csv(out, index=False)
            log.append(f"| `{out.name}` | derived from `table2_{preset}_full.csv` |")
    for suf in ("", "_coupled", "_m5dB", "_p10dB"):
        w = wide(SW / f"bandwidth_sweep{suf}.csv", ["scenario", "bw_frac"])
        if w is not None:
            out = D / f"bandwidth_sweep{suf}_wide.csv"; w.to_csv(out, index=False)
            log.append(f"| `{out.name}` | derived from `bandwidth_sweep{suf}.csv` |")
    (D / "README.md").write_text(
        "# Revision R2b (2026-10-01) — result CSVs\n\nCopied by `scripts/analysis/r2b_report.py export` at the end of "
        "`scripts/run_revision_r2b_pipeline.sh`. Same column schema as R1/R2; `ReconUNet-CB` = the covariance-only network "
        "(same architecture and loss as `ReconUNet-C`) trained from scratch on the randomised-source-bandwidth corpus "
        "(`configs/data/paper_corpus_bwrand.yaml`). `paper_testset_bwrand_*` = the paper test split re-rendered with each "
        "scene's own drawn bandwidth (matched wideband). DA-MUSIC = v2 ensemble. CIs: percentile bootstrap over scenes, "
        "1000 resamples, seed 20260920.\n\n" + "\n".join(log) + "\n")
    headline(D)
    print(f"[r2b] exported to {D}")
    return 0


def _ci_table(path: Path, title: str) -> list[str]:
    if not path.exists(): return []
    df = pd.read_csv(path); df = df[df.scenario == "all K"].sort_values("rmse_deg")
    L = [f"## {title}", "", "| method | pooled RMSE (°) | 95 % CI | median RMSPE (°) | 95 % CI | scenes |", "|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        L.append(f"| {r.method} | {r.rmse_deg:.2f} | [{r.rmse_ci_lo:.2f}, {r.rmse_ci_hi:.2f}] | {r.median_rmspe_deg:.2f} | "
                 f"[{r.median_ci_lo:.2f}, {r.median_ci_hi:.2f}] | {r.n_scenes} |")
    return L + [""]


def _by_k(path: Path) -> list[str]:
    if not path.exists(): return []
    df = pd.read_csv(path)
    ms = [m for m in ["R-MUSIC", "ESPRIT"] + ORDER if f"{m}_mean" in df.columns]
    rows = [{"K": r["K"], "n": r["n"], **{m: r[f"{m}_mean"] for m in ms}} for _, r in df.iterrows()]
    return ["Pooled RMSE (°) by K:", "", md_table(rows, fmt="{:.2f}"), ""]


def statement(D: Path) -> list[str]:
    """Plain statement, generated from the numbers with explicit thresholds."""
    bw = D / "bandwidth_sweep.csv"; ci = D / "paper_testset_ci.csv"
    if not bw.exists(): return []
    df = pd.read_csv(bw, dtype={"bw_frac": str})
    g = lambda m, bws: df[(df.method == m) & df.bw_frac.isin(bws)].set_index(["scenario", "bw_frac"]).rmse_deg
    cb, c, rm = g("ReconUNet-CB", WIDE_BW), g("ReconUNet-C", WIDE_BW), g("Root-MUSIC", WIDE_BW)
    if cb.empty or c.empty: return []
    cb_ci_lo = df[(df.method == "ReconUNet-CB") & df.bw_frac.isin(WIDE_BW)].set_index(["scenario", "bw_frac"]).rmse_ci_lo
    n = len(cb); beats_rm = int((cb <= rm.reindex(cb.index)).sum()); sig_worse = int((cb_ci_lo > rm.reindex(cb.index)).sum())
    if beats_rm == n: verdict = "removed"
    elif cb.mean() < 0.5 * c.mean(): verdict = "substantially reduced but did not fully remove"
    else: verdict = "did not remove"
    t = (f"**Wideband (bw ≥ 0.2 and white sources, {n} cells: 3 scenarios × 3 bandwidths, decoupled delays, 0 dB).** "
         f"Randomised-bandwidth training {verdict} the wideband failure: the mean pooled RMSE over these cells is "
         f"{cb.mean():.2f}° for ReconUNet-CB vs {c.mean():.2f}° for ReconUNet-C and {rm.mean():.2f}° for raw Root-MUSIC; "
         f"ReconUNet-CB is at or below Root-MUSIC in {beats_rm}/{n} cells (its CI lies entirely above Root-MUSIC in {sig_worse}/{n}). ")
    at = lambda m: df[(df.method == m) & (df.bw_frac == "0.05")].set_index("scenario").rmse_deg
    cb05, c05 = at("ReconUNet-CB"), at("ReconUNet-C")
    t += ("**At the training bandwidth 0.05** the sweep RMSE is " +
          ", ".join(f"{s} {cb05[s]:.2f}° vs {c05[s]:.2f}°" for s in cb05.index if s in c05.index) + " (ReconUNet-CB vs ReconUNet-C)")
    if ci.exists():
        p = pd.read_csv(ci); p = p[p.scenario == "all K"].set_index("method")
        if {"ReconUNet-CB", "ReconUNet-C"} <= set(p.index):
            x, y = p.loc["ReconUNet-CB"], p.loc["ReconUNet-C"]
            t += (f"; on the paper test split (rendered at bw 0.05) pooled RMSE {x.rmse_deg:.2f}° [{x.rmse_ci_lo:.2f}, {x.rmse_ci_hi:.2f}] "
                  f"vs {y.rmse_deg:.2f}° [{y.rmse_ci_lo:.2f}, {y.rmse_ci_hi:.2f}], median RMSPE {x.median_rmspe_deg:.2f}° vs {y.median_rmspe_deg:.2f}°")
    t += (". _(Generated by `r2b_report.py` from the CSVs: 'removed' = ReconUNet-CB ≤ Root-MUSIC in every wideband cell; "
          "'substantially reduced' = mean wideband RMSE below half of ReconUNet-C's.)_")
    return ["## Plain statement", "", t, ""]


def headline(D: Path) -> None:
    L = ["# R2b headline numbers (2026-10-01) — ReconUNet-CB (randomised source bandwidth)", ""]
    L += statement(D)
    L += _ci_table(D / "paper_testset_ci.csv", "Paper test split (rendered at bw 0.05), pooled over K = 1..4")
    L += _ci_table(D / "paper_testset_bwrand_ci.csv", "Matched wideband: paper test split re-rendered at each scene's drawn bandwidth, pooled over K")
    L += _by_k(D / "paper_testset_bwrand_by_K.csv")
    tr = D / "reconunet_cb_training.csv"
    if tr.exists():
        L += ["## Training", "", md_table(pd.read_csv(tr).to_dict("records"), fmt="{:g}"), ""]
    for suf, title in (("", "0 dB, replica delays fixed at the training bandwidth 0.05 (decoupled, primary)"),
                       ("_coupled", "0 dB, replica delays ∝ 1/bw (coupled, secondary)"),
                       ("_m5dB", "−5 dB, decoupled"), ("_p10dB", "+10 dB, decoupled")):
        f = D / f"bandwidth_sweep{suf}_wide.csv"
        if f.exists():
            L += [f"## Bandwidth sweep, pooled RMSE (°) — {title}", "",
                  md_table(pd.read_csv(f, dtype={"bw_frac": str}).to_dict("records"), fmt="{:.2f}"), ""]
    co = D / "cost_summary.csv"
    if co.exists():
        L += ["## Cost", "", md_table(pd.read_csv(co).to_dict("records"), fmt="{:.4g}"), ""]
    (D / "HEADLINE.md").write_text("\n".join(L))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train-summary"); t.add_argument("--status-log", type=Path, required=True); t.add_argument("--output", "-o", type=Path, required=True)
    e = sub.add_parser("export")
    for k in ("--eval-dir", "--sweeps-dir", "--run-dir", "--dest"): e.add_argument(k, type=Path, required=True)
    a = ap.parse_args()
    for k in ("eval_dir", "sweeps_dir", "run_dir", "dest", "status_log", "output"):
        if getattr(a, k, None) is not None: setattr(a, k, getattr(a, k).resolve())
    return train_summary(a) if a.cmd == "train-summary" else export(a)


if __name__ == "__main__":
    raise SystemExit(main())
