# Results status — revision round 2 (ReconUNet-C + bandwidth sweep), started 2026-09-30

**Last update:** 2026-09-30 20:15 UTC · **pipeline process:** running · **log says:** in progress · **stages:** 0/17 done, 0 failed

## ReconUNet-C training

Epoch 175/300, 451 min elapsed (2.57 min/epoch); best epoch 160 (val loss 0.07002, val RMSPE 1.333°); last: val RMSPE 1.334°, LR 4.0e-06; early stop after 25 epochs without improvement (currently 15). Full ReconUNet (R1) reached 2.174° val RMSPE at epoch 84 of 109.

## Stages

| stage | what | state | started (UTC) | ended | elapsed |
|---|---|---|---|---|---|
| `train_reconunet_c` | Part A: train ReconUNet-C, full scale (≤300 epochs, patience 25) | 🔄 running | 2026-09-30 12:45:06 |  |  |
| `train_summary` | Part A: training summary CSV | ⏳ pending |  |  |  |
| `eval_paper_testset_r2` | Part B1: paper test split by K (+errors) | ⏳ pending |  |  |  |
| `eval_scenario_sweep_r2` | Part B2: scenario sweep, 4 scenarios × 9 SNRs (+errors) | ⏳ pending |  |  |  |
| `eval_table2_mild_r2` | Part B3: Table II mild, all back ends, all SNRs | ⏳ pending |  |  |  |
| `eval_table2_harsh_r2` | Part B3: Table III harsh, all back ends, all SNRs | ⏳ pending |  |  |  |
| `bootstrap_ci_r2` | Part B1-3: bootstrap 95 % CIs | ⏳ pending |  |  |  |
| `snapshot_sweep_r2` | Part B4: snapshot sweep | ⏳ pending |  |  |  |
| `separation_sweep_r2` | Part B4: separation sweep + resolution probability | ⏳ pending |  |  |  |
| `cost_r2` | Part B5: cost (params, MMACs, GPU/CPU latency, training time) | ⏳ pending |  |  |  |
| `music_verification_r2` | Part B6: grid-MUSIC verification incl. ReconUNet-C | ⏳ pending |  |  |  |
| `figures_r2` | Part B7: MUSIC-spectrum figure with ReconUNet-C | ⏳ pending |  |  |  |
| `bw_sweep_decoupled` | Part C: bandwidth sweep, delays fixed at bw 0.05 (primary) | ⏳ pending |  |  |  |
| `bw_sweep_coupled` | Part C: bandwidth sweep, delays ∝ 1/bw (secondary) | ⏳ pending |  |  |  |
| `bw_plot` | Part C: quick-look figure | ⏳ pending |  |  |  |
| `export_r2` | Copy CSVs to docs/revision_r2 + HEADLINE.md | ⏳ pending |  |  |  |
| `facts_r2` | Part D: append R2 sections to docs/revision_facts.md | ⏳ pending |  |  |  |

## Results (links)

| result | status | link |
|---|---|---|
| Headline numbers | ⏳ pending | [docs/revision_r2/HEADLINE.md](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/docs/revision_r2/HEADLINE.md) |
| All R2 CSVs (folder) | ✅ 09-30 12:45 | [docs/revision_r2](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/docs/revision_r2) |
| ReconUNet-C training history | ✅ 09-30 20:14 | [experiments/runs/reconunet_c_paper/checkpoints/history.json](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/experiments/runs/reconunet_c_paper/checkpoints/history.json) |
| Paper test split by K | ⏳ pending | [experiments/runs/eval_r2_20260930/paper_testset_r2/paper_testset_by_K.csv](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/experiments/runs/eval_r2_20260930/paper_testset_r2/paper_testset_by_K.csv) |
| Scenario sweep | ⏳ pending | [experiments/runs/eval_r2_20260930/scenario_sweep_r2/scenario_sweep.csv](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/experiments/runs/eval_r2_20260930/scenario_sweep_r2/scenario_sweep.csv) |
| Table II mild | ⏳ pending | [experiments/runs/eval_r2_20260930/table2_mild_r2/table2_full.csv](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/experiments/runs/eval_r2_20260930/table2_mild_r2/table2_full.csv) |
| Table III harsh | ⏳ pending | [experiments/runs/eval_r2_20260930/table2_harsh_r2/table2_full.csv](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/experiments/runs/eval_r2_20260930/table2_harsh_r2/table2_full.csv) |
| Snapshot sweep | ⏳ pending | [experiments/runs/sweeps_r2_20260930/snapshot_sweep.csv](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/experiments/runs/sweeps_r2_20260930/snapshot_sweep.csv) |
| Separation sweep | ⏳ pending | [experiments/runs/sweeps_r2_20260930/separation_sweep.csv](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/experiments/runs/sweeps_r2_20260930/separation_sweep.csv) |
| Cost | ⏳ pending | [experiments/runs/sweeps_r2_20260930/cost_summary_r2.csv](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/experiments/runs/sweeps_r2_20260930/cost_summary_r2.csv) |
| Bandwidth sweep | ⏳ pending | [experiments/runs/sweeps_r2_20260930/bandwidth_sweep.csv](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/experiments/runs/sweeps_r2_20260930/bandwidth_sweep.csv) |
| Bandwidth sweep (coupled) | ⏳ pending | [experiments/runs/sweeps_r2_20260930/bandwidth_sweep_coupled.csv](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/experiments/runs/sweeps_r2_20260930/bandwidth_sweep_coupled.csv) |
| Bandwidth autocorrelation | ⏳ pending | [experiments/runs/sweeps_r2_20260930/bandwidth_autocorr.csv](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/experiments/runs/sweeps_r2_20260930/bandwidth_autocorr.csv) |
| R2 figures | ✅ 09-30 12:45 | [docs/figs_revision/r2](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/docs/figs_revision/r2) |
| Revision facts | ✅ 09-30 12:44 | [docs/revision_facts.md](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/docs/revision_facts.md) |
| Pipeline status log | ✅ 09-30 12:45 | [experiments/runs/revision_r2_20260930/status.log](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/experiments/runs/revision_r2_20260930/status.log) |

## Master status log (tail)

```
2026-09-30 12:45:06 PIPELINE START pid=3820944 gpu=NVIDIA RTX 2000 Ada Generation git=f674ff2
2026-09-30 12:45:06 START train_reconunet_c  (DOA_env/bin/python -m reconunet.cli.train --config configs/train/reconunet_c_paper.yaml)
```
