#!/usr/bin/env python3
"""Rewrite RESULTS_STATUS.md — the phone-readable dashboard of the 2026-09-20 revision
pipeline: which runs are done / running / failed / pending, last update, ETA, and
direct GitHub links to every result CSV, the facts document and the figures.

Reads experiments/runs/revision2_20260920/status.log (START/END lines written by
scripts/run_revision2_pipeline.sh), the pipeline pid file, the DA-MUSIC v2 histories
and the presence of result files.  Called by scripts/autopush_watcher.sh every cycle.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUN = REPO / "experiments/runs/revision2_20260920"
GH = "https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/"
ABL = "experiments/runs/ablation_20260920"
E = "experiments/runs/eval_coherent_20260910"
SW = "experiments/runs/sweeps_20260920"
VARIANTS = ["01_full", "02_rec_only", "03_rec_proj", "04_no_dom", "05_no_eig", "06_single_lag", "07_relu", "08_no_evd_heads", "09_fixed_imperf"]
STAGES = ([(f"abl_train_{v}", f"Ablation train {v}") for v in VARIANTS] +
          [("abl_eval", "Ablation evaluation → ablation_results.csv / route_comparison.csv"), ("facts_ablation", "Append ablation numbers to revision_facts.md")] +
          [(f"damusic_k{k}_v2", f"DA-MUSIC K={k} continuation (≤600 epochs)") for k in (2, 3, 4)] +
          [("eval_paper_testset_v2", "Eval v2: paper test split"), ("eval_scenario_sweep_v2", "Eval v2: scenario sweep"),
           ("eval_table2_mild_v2", "Eval v2: Table II mild"), ("eval_table2_harsh_v2", "Eval v2: Table II harsh"),
           ("facts_damusic", "Append DA-MUSIC v2 numbers to revision_facts.md"),
           ("bootstrap_ci", "Bootstrap 95 % CIs (*_ci.csv)"), ("snapshot_sweep", "Snapshot sweep T ∈ {8…512}"),
           ("separation_sweep", "Separation sweep Δθ ∈ {2…15}°"), ("fbss_baseline", "FBSS Root-MUSIC baseline"),
           ("revision_figures", "Revision figures (vector PDF)"), ("music_verification", "Grid-MUSIC verification"),
           ("facts_extras", "Append sweep/FBSS numbers to revision_facts.md")])
RESULTS = [("Ablation results (9 variants × eval sets)", f"{ABL}/ablation_results.csv"),
           ("Route comparison (01_full, 08_no_evd_heads)", f"{ABL}/route_comparison.csv"),
           ("Ablation status log", f"{ABL}/status.log"),
           ("DA-MUSIC v2: paper test split by K", f"{E}/paper_testset_v2/paper_testset_by_K.csv"),
           ("DA-MUSIC v2: scenario sweep", f"{E}/scenario_sweep_v2/scenario_sweep.csv"),
           ("DA-MUSIC v2: Table II mild", f"{E}/table2_mild_v2/table2_full.csv"),
           ("DA-MUSIC v2: Table II harsh", f"{E}/table2_harsh_v2/table2_full.csv"),
           ("Bootstrap CIs: Table II mild", f"{E}/table2_mild_v2/table2_mild_v2_ci.csv"),
           ("Bootstrap CIs: Table II harsh", f"{E}/table2_harsh_v2/table2_harsh_v2_ci.csv"),
           ("Bootstrap CIs: scenario sweep", f"{E}/scenario_sweep_v2/scenario_sweep_v2_ci.csv"),
           ("Bootstrap CIs: paper test split", f"{E}/paper_testset_v2/paper_testset_v2_ci.csv"),
           ("Snapshot sweep", f"{SW}/snapshot_sweep.csv"), ("Separation sweep", f"{SW}/separation_sweep.csv"),
           ("FBSS baseline", f"{SW}/fbss_baseline.csv"), ("MUSIC verification", f"{SW}/music_verification.csv"),
           ("Revision facts (docs/revision_facts.md)", "docs/revision_facts.md"),
           ("Revision figures folder", "docs/figs_revision"),
           ("Original 2026-09-10 results + HTML report", E),
           ("Pipeline master status log", "experiments/runs/revision2_20260920/status.log"),
           ("Auto-push log", "experiments/runs/autopush.log")]
TS = "%Y-%m-%d %H:%M:%S"


def parse_status(path: Path):
    st = {}
    if not path.exists(): return st, False
    done = False
    for line in path.read_text().splitlines():
        m = re.match(r"(\S+ \S+) START (\S+)", line)
        if m: st.setdefault(m.group(2), {})["start"] = m.group(1); continue
        m = re.match(r"(\S+ \S+) END (\S+) rc=(\d+)(?: elapsed=(\S+))?", line)
        if m:
            d = st.setdefault(m.group(2), {}); d["end"] = m.group(1); d["rc"] = int(m.group(3)); d["elapsed"] = m.group(4) or ""
        if "PIPELINE DONE" in line: done = True
    return st, done


def pid_alive(pidfile: Path) -> bool:
    try:
        pid = int(pidfile.read_text().split()[0]); os.kill(pid, 0); return True
    except Exception:
        return False


def damusic_progress():
    rows = []
    for k in (2, 3, 4):
        h = REPO / f"experiments/runs/damusic_paper/k{k}_v2/checkpoints/history.json"
        b = REPO / f"experiments/runs/damusic_paper/k{k}/checkpoints/history.json"
        if h.exists():
            H = json.load(h.open()); B = json.load(b.open()) if b.exists() else []
            best = min(H, key=lambda r: r["val"]); best_v1 = min(B, key=lambda r: r["val"]) if B else None
            rows.append((k, len(H), best["epoch"], best["val_rmspe_deg"], best_v1["val_rmspe_deg"] if best_v1 else float("nan"), H[-1]["lr"]))
    return rows


def main() -> int:
    st, done = parse_status(RUN / "status.log")
    alive = pid_alive(RUN / "pipeline.pid")
    now = dt.datetime.now(dt.timezone.utc)
    L = ["# Results status — revision pipeline 2026-09-20", "",
         f"**Last update:** {now.strftime('%Y-%m-%d %H:%M UTC')} · **pipeline process:** {'running' if alive else 'not running'}"
         f" · **pipeline log says:** {'PIPELINE DONE' if done else 'in progress' if alive else 'stopped before completion' if st else 'not started'}", "",
         "This page is rewritten by the auto-push watcher every 30 minutes. Links point at the files on `main`.", ""]
    n_done = sum(1 for k, _ in STAGES if st.get(k, {}).get("rc") == 0)
    n_fail = sum(1 for k, _ in STAGES if st.get(k, {}).get("rc") not in (None, 0))
    L += [f"**Progress:** {n_done}/{len(STAGES)} stages done, {n_fail} failed.", ""]
    # ETA
    running = [k for k, _ in STAGES if "start" in st.get(k, {}) and "end" not in st.get(k, {})]
    eta = []
    for k in running:
        started = dt.datetime.strptime(st[k]["start"], TS).replace(tzinfo=dt.timezone.utc)
        el = (now - started).total_seconds() / 60
        if k.startswith("abl_train"): eta.append(f"`{k}` running {el:.0f} min (a variant takes ≈ 12–15 min)")
        elif k.startswith("damusic"):
            kk = int(k[len("damusic_k")])
            h = REPO / f"experiments/runs/damusic_paper/k{kk}_v2/checkpoints/history.json"
            ep = len(json.load(h.open())) if h.exists() else 300
            eta.append(f"`{k}` at epoch {ep}/600 after {el:.0f} min; worst case {(600 - ep) * 46 / 60:.0f} more min at 46 s/epoch, sooner if early stopping (patience 25) triggers")
        else: eta.append(f"`{k}` running {el:.0f} min")
    pend = [k for k, _ in STAGES if k not in st]
    if not done:
        rem_abl = sum(1 for k in pend if k.startswith("abl_train")) * 13
        rem_dm = sum(1 for k in pend if k.startswith("damusic")) * 230
        eta.append(f"Pending: {len(pend)} stages; rough worst-case remaining ≈ {rem_abl + rem_dm + 60:.0f} min "
                   f"({sum(1 for k in pend if k.startswith('abl_train'))} ablation variants ≈ {rem_abl} min run two at a time → ≈ {rem_abl // 2} min, "
                   f"{sum(1 for k in pend if k.startswith('damusic'))} DA-MUSIC runs ≤ {rem_dm} min, evals + extras ≈ 60 min).")
    L += ["## ETA", ""] + [f"- {e}" for e in (eta or ["- nothing running"])] + [""]
    # results
    L += ["## Results (links)", "", "| result | status | link |", "|---|---|---|"]
    for label, rel in RESULTS:
        p = REPO / rel; ok = p.exists()
        when = dt.datetime.fromtimestamp(p.stat().st_mtime, dt.timezone.utc).strftime("%m-%d %H:%M") if ok else ""
        L.append(f"| {label} | {'✅ ' + when if ok else '⏳ pending'} | [{rel}]({GH}{rel}) |")
    L.append("")
    # stages
    L += ["## Stages", "", "| stage | what | state | started (UTC) | ended | elapsed |", "|---|---|---|---|---|---|"]
    for k, what in STAGES:
        d = st.get(k, {})
        state = ("✅ done" if d.get("rc") == 0 else f"❌ failed rc={d['rc']}" if "rc" in d else "🔄 running" if "start" in d
                 else ("⏭ not run" if done else "⏳ pending"))
        L.append(f"| `{k}` | {what} | {state} | {d.get('start', '')} | {d.get('end', '')} | {d.get('elapsed', '')} |")
    L.append("")
    # latest numbers
    ab = REPO / ABL / "ablation_results.csv"
    if ab.exists():
        import csv
        rows = [r for r in csv.DictReader(ab.open()) if r["eval_set"] == "paper_test"]
        if rows:
            L += ["## Latest numbers: ablation on the paper test split (pooled RMSE °, median RMSPE °, rel. cov. error, subspace dist, eigengap err)", "",
                  "| variant | RMSE | median | cov_err | proj_dist | gap_err | n |", "|---|---|---|---|---|---|---|"]
            for r in rows:
                L.append(f"| {r['variant']} | {float(r['pooled_rmse_deg']):.2f} | {float(r['median_rmspe_deg']):.2f} | {float(r['cov_err_frob']):.3f} | "
                         f"{float(r['subspace_dist_projF']):.3f} | {float(r['eigengap_err']):.3f} | {r['n_scenes']} |")
            L.append("")
    dm = damusic_progress()
    if dm:
        L += ["## Latest numbers: DA-MUSIC continuation (validation RMSPE at the best epoch, v1 = 300-epoch cap)", "",
              "| K | epochs so far | best epoch | best val RMSPE v2 (°) | best val RMSPE v1 (°) | current LR |", "|---|---|---|---|---|---|"]
        for k, n, be, v2, v1, lr in dm:
            L.append(f"| {k} | {n} | {be} | {v2:.3f} | {v1:.3f} | {lr:.1e} |")
        L.append("")
    if (RUN / "status.log").exists():
        L += ["## Master status log (tail)", "", "```"] + (RUN / "status.log").read_text().splitlines()[-12:] + ["```", ""]
    (REPO / "RESULTS_STATUS.md").write_text("\n".join(L))
    print(f"[status] RESULTS_STATUS.md rewritten ({n_done}/{len(STAGES)} done, alive={alive}, done={done})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
