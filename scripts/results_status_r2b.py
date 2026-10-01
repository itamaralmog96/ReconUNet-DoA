#!/usr/bin/env python3
"""RESULTS_STATUS.md for the R2b pipeline (ReconUNet-CB, randomised source bandwidth, 2026-10-01):
scripts/results_status_r2.py with the R2b stages, paths and headline (docs/revision_r2b/)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import results_status_r2 as R  # noqa: E402

D = os.environ.get("R2B_DATE", "20261001")
R.RUN = R.REPO / f"experiments/runs/revision_r2b_{D}"
R.CKD = R.REPO / "experiments/runs/reconunet_cb_paper/checkpoints"
R.TITLE = "# Results status — revision R2b (ReconUNet-CB, randomised source bandwidth), started 2026-10-01"
R.REF = ("ReconUNet-C (R2) reached 1.330° val RMSPE at epoch 182 of 207 — on the bw-0.05 validation split, so not directly "
         "comparable with ReconUNet-CB's randomised-bandwidth validation split.")
R.MODEL, R.TRAIN_STAGE, R.DOCS = "ReconUNet-CB", "train_reconunet_cb", "docs/revision_r2b"
E, SW = f"experiments/runs/eval_r2b_{D}", f"experiments/runs/sweeps_r2b_{D}"
R.STAGES = [("train_reconunet_cb", "Part B: train ReconUNet-CB from scratch on the randomised-bandwidth corpus (≤300 epochs, patience 25)"),
            ("train_summary", "Part B: training summary CSV"),
            ("bw_sweep_decoupled", "C1: bandwidth sweep 0 dB, decoupled (primary), all methods"),
            ("bw_sweep_coupled", "C1: bandwidth sweep 0 dB, coupled"),
            ("bw_sweep_m5dB", "C1: bandwidth sweep −5 dB, decoupled (Root-MUSIC, ReconUNet-C, ReconUNet-CB)"),
            ("bw_sweep_p10dB", "C1: bandwidth sweep +10 dB, decoupled (Root-MUSIC, ReconUNet-C, ReconUNet-CB)"),
            ("bw_plot", "C1: quick-look figure"),
            ("eval_paper_testset_r2b", "C2: paper test split by K (+errors)"),
            ("eval_scenario_sweep_r2b", "C3: scenario sweep"),
            ("eval_table2_mild_r2b", "C4: Table II mild"), ("eval_table2_harsh_r2b", "C4: Table III harsh"),
            ("eval_paper_testset_bwrand_r2b", "C6: paper test split re-rendered at per-scene random bandwidth"),
            ("bootstrap_ci_r2b", "C2-C6: bootstrap 95 % CIs"),
            ("snapshot_sweep_r2b", "C5: snapshot sweep"), ("separation_sweep_r2b", "C5: separation sweep"),
            ("cost_r2b", "C7: cost incl. ReconUNet-CB"),
            ("export_r2b", "Copy CSVs to docs/revision_r2b + HEADLINE.md"),
            ("facts_r2b", "Append R2b sections to docs/revision_facts.md")]
R.RESULTS = [("Headline numbers", "docs/revision_r2b/HEADLINE.md"), ("All R2b CSVs (folder)", "docs/revision_r2b"),
             ("ReconUNet-CB training history", "experiments/runs/reconunet_cb_paper/checkpoints/history.json"),
             ("Bandwidth sweep (decoupled)", f"{SW}/bandwidth_sweep.csv"), ("Bandwidth sweep (coupled)", f"{SW}/bandwidth_sweep_coupled.csv"),
             ("Bandwidth sweep −5 dB", f"{SW}/bandwidth_sweep_m5dB.csv"), ("Bandwidth sweep +10 dB", f"{SW}/bandwidth_sweep_p10dB.csv"),
             ("Paper test split", f"{E}/paper_testset_r2b/paper_testset_by_K.csv"),
             ("Matched wideband test split", f"{E}/paper_testset_bwrand_r2b/paper_testset_by_K.csv"),
             ("Scenario sweep", f"{E}/scenario_sweep_r2b/scenario_sweep.csv"),
             ("Table II mild", f"{E}/table2_mild_r2b/table2_full.csv"), ("Table III harsh", f"{E}/table2_harsh_r2b/table2_full.csv"),
             ("Snapshot sweep", f"{SW}/snapshot_sweep.csv"), ("Separation sweep", f"{SW}/separation_sweep.csv"),
             ("Cost", f"{SW}/cost_summary_r2.csv"), ("R2b figures", "docs/figs_revision/r2b"),
             ("Revision facts", "docs/revision_facts.md"), ("Pipeline status log", f"experiments/runs/revision_r2b_{D}/status.log")]

if __name__ == "__main__":
    raise SystemExit(R.main())
