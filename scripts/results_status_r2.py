#!/usr/bin/env python3
"""Rewrite RESULTS_STATUS.md for the revision round-2 pipeline (2026-09-30): per-stage state
(done / failed / running / pending) from experiments/runs/revision_r2_20260930/status.log,
ReconUNet-C training progress, links to the R2 result files, and — once exported — the
headline numbers (docs/revision_r2/HEADLINE.md).  Called every cycle by
scripts/autopush_watcher.sh (STATUS_SCRIPT) and at the end of the pipeline, where
``--final`` also writes docs/revision_r2/FINAL_STATUS.md.
"""
from __future__ import annotations

import argparse
import os
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from results_status import GH, TS, parse_status, pid_alive  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DATE = os.environ.get("R2_DATE", "20260930")
RUN = REPO / f"experiments/runs/revision_r2_{DATE}"
E = f"experiments/runs/eval_r2_{DATE}"
SW = f"experiments/runs/sweeps_r2_{DATE}"
CKD = REPO / "experiments/runs/reconunet_c_paper/checkpoints"
# Overridden by scripts/results_status_r2b.py for the R2b pipeline.
TITLE = "# Results status — revision round 2 (ReconUNet-C + bandwidth sweep), started 2026-09-30"
MODEL, TRAIN_STAGE, DOCS = "ReconUNet-C", "train_reconunet_c", "docs/revision_r2"
REF = "Full ReconUNet (R1) reached 2.174° val RMSPE at epoch 84 of 109."
STAGES = [("train_reconunet_c", "Part A: train ReconUNet-C, full scale (≤300 epochs, patience 25)"),
          ("train_summary", "Part A: training summary CSV"),
          ("eval_paper_testset_r2", "Part B1: paper test split by K (+errors)"),
          ("eval_scenario_sweep_r2", "Part B2: scenario sweep, 4 scenarios × 9 SNRs (+errors)"),
          ("eval_table2_mild_r2", "Part B3: Table II mild, all back ends, all SNRs"),
          ("eval_table2_harsh_r2", "Part B3: Table III harsh, all back ends, all SNRs"),
          ("bootstrap_ci_r2", "Part B1-3: bootstrap 95 % CIs"),
          ("snapshot_sweep_r2", "Part B4: snapshot sweep"),
          ("separation_sweep_r2", "Part B4: separation sweep + resolution probability"),
          ("cost_r2", "Part B5: cost (params, MMACs, GPU/CPU latency, training time)"),
          ("music_verification_r2", "Part B6: grid-MUSIC verification incl. ReconUNet-C"),
          ("figures_r2", "Part B7: MUSIC-spectrum figure with ReconUNet-C"),
          ("bw_sweep_decoupled", "Part C: bandwidth sweep, delays fixed at bw 0.05 (primary)"),
          ("bw_sweep_coupled", "Part C: bandwidth sweep, delays ∝ 1/bw (secondary)"),
          ("bw_plot", "Part C: quick-look figure"),
          ("export_r2", "Copy CSVs to docs/revision_r2 + HEADLINE.md"),
          ("facts_r2", "Part D: append R2 sections to docs/revision_facts.md")]
RESULTS = [("Headline numbers", "docs/revision_r2/HEADLINE.md"), ("All R2 CSVs (folder)", "docs/revision_r2"),
           ("ReconUNet-C training history", "experiments/runs/reconunet_c_paper/checkpoints/history.json"),
           ("Paper test split by K", f"{E}/paper_testset_r2/paper_testset_by_K.csv"),
           ("Scenario sweep", f"{E}/scenario_sweep_r2/scenario_sweep.csv"),
           ("Table II mild", f"{E}/table2_mild_r2/table2_full.csv"), ("Table III harsh", f"{E}/table2_harsh_r2/table2_full.csv"),
           ("Snapshot sweep", f"{SW}/snapshot_sweep.csv"), ("Separation sweep", f"{SW}/separation_sweep.csv"),
           ("Cost", f"{SW}/cost_summary_r2.csv"), ("Bandwidth sweep", f"{SW}/bandwidth_sweep.csv"),
           ("Bandwidth sweep (coupled)", f"{SW}/bandwidth_sweep_coupled.csv"), ("Bandwidth autocorrelation", f"{SW}/bandwidth_autocorr.csv"),
           ("R2 figures", "docs/figs_revision/r2"), ("Revision facts", "docs/revision_facts.md"),
           ("Pipeline status log", f"experiments/runs/revision_r2_{DATE}/status.log")]


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--final", action="store_true"); a = ap.parse_args()
    st, done = parse_status(RUN / "status.log")
    alive = pid_alive(RUN / "pipeline.pid")
    now = dt.datetime.now(dt.timezone.utc)
    n_done = sum(1 for k, _ in STAGES if st.get(k, {}).get("rc") == 0)
    n_fail = sum(1 for k, _ in STAGES if st.get(k, {}).get("rc") not in (None, 0))
    L = [TITLE, "",
         f"**Last update:** {now.strftime('%Y-%m-%d %H:%M UTC')} · **pipeline process:** {'running' if alive else 'not running'} · "
         f"**log says:** {'PIPELINE DONE' if done else 'in progress' if alive else 'stopped before completion' if st else 'not started'} · "
         f"**stages:** {n_done}/{len(STAGES)} done, {n_fail} failed", ""]
    h = CKD / "history.json"
    if h.exists():
        H = json.load(h.open()); best = min(H, key=lambda r: r["val"])
        s = st.get(TRAIN_STAGE, {})
        el = ""
        if "start" in s:
            t0 = dt.datetime.strptime(s["start"], TS).replace(tzinfo=dt.timezone.utc)
            t1 = dt.datetime.strptime(s["end"], TS).replace(tzinfo=dt.timezone.utc) if "end" in s else now
            mins = (t1 - t0).total_seconds() / 60; el = f", {mins:.0f} min elapsed ({mins / max(len(H), 1):.2f} min/epoch)"
        L += [f"## {MODEL} training", "",
              f"Epoch {len(H)}/300{el}; best epoch {best['epoch']} (val loss {best['val']:.5f}, val RMSPE {best['val_rmspe_deg']:.3f}°); "
              f"last: val RMSPE {H[-1]['val_rmspe_deg']:.3f}°, LR {H[-1]['lr']:.1e}; early stop after 25 epochs without improvement "
              f"(currently {len(H) - best['epoch']}). {REF}", ""]
    L += ["## Stages", "", "| stage | what | state | started (UTC) | ended | elapsed |", "|---|---|---|---|---|---|"]
    for k, what in STAGES:
        d = st.get(k, {})
        state = ("✅ done" if d.get("rc") == 0 else f"❌ failed rc={d['rc']}" if "rc" in d else "🔄 running" if "start" in d
                 else ("⏭ not run" if done or not alive and st else "⏳ pending"))
        L.append(f"| `{k}` | {what} | {state} | {d.get('start', '')} | {d.get('end', '')} | {d.get('elapsed', '')} |")
    L += ["", "## Results (links)", "", "| result | status | link |", "|---|---|---|"]
    for label, rel in RESULTS:
        p = REPO / rel; ok = p.exists()
        when = dt.datetime.fromtimestamp(p.stat().st_mtime, dt.timezone.utc).strftime("%m-%d %H:%M") if ok else ""
        L.append(f"| {label} | {'✅ ' + when if ok else '⏳ pending'} | [{rel}]({GH}{rel}) |")
    L.append("")
    hl = REPO / DOCS / "HEADLINE.md"
    if hl.exists():
        L += ["#" + hl.read_text(), ""]
    if (RUN / "status.log").exists():
        L += ["## Master status log (tail)", "", "```"] + (RUN / "status.log").read_text().splitlines()[-15:] + ["```", ""]
    txt = "\n".join(L)
    (REPO / "RESULTS_STATUS.md").write_text(txt)
    if a.final:
        (REPO / DOCS).mkdir(parents=True, exist_ok=True)
        (REPO / DOCS / "FINAL_STATUS.md").write_text(txt.replace("# Results status", "# Final status", 1))
    print(f"[status-r2] RESULTS_STATUS.md rewritten ({n_done}/{len(STAGES)} done, {n_fail} failed, alive={alive}, done={done})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
