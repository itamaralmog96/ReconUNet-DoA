# Final status — revision R2b (ReconUNet-CB, randomised source bandwidth), started 2026-10-01

**Last update:** 2026-10-01 17:52 UTC · **pipeline process:** running · **log says:** PIPELINE DONE · **stages:** 18/18 done, 0 failed

## ReconUNet-CB training

Epoch 204/300, 530 min elapsed (2.60 min/epoch); best epoch 179 (val loss 0.08128, val RMSPE 1.458°); last: val RMSPE 1.457°, LR 1.0e-06; early stop after 25 epochs without improvement (currently 25). ReconUNet-C (R2) reached 1.330° val RMSPE at epoch 182 of 207 — on the bw-0.05 validation split, so not directly comparable with ReconUNet-CB's randomised-bandwidth validation split.

## Stages

| stage | what | state | started (UTC) | ended | elapsed |
|---|---|---|---|---|---|
| `train_reconunet_cb` | Part B: train ReconUNet-CB from scratch on the randomised-bandwidth corpus (≤300 epochs, patience 25) | ✅ done | 2026-10-01 08:51:07 | 2026-10-01 17:40:57 | 529min |
| `train_summary` | Part B: training summary CSV | ✅ done | 2026-10-01 17:40:57 | 2026-10-01 17:40:58 | 0min |
| `bw_sweep_decoupled` | C1: bandwidth sweep 0 dB, decoupled (primary), all methods | ✅ done | 2026-10-01 17:40:58 | 2026-10-01 17:41:29 | 0min |
| `bw_sweep_coupled` | C1: bandwidth sweep 0 dB, coupled | ✅ done | 2026-10-01 17:41:29 | 2026-10-01 17:42:00 | 0min |
| `bw_sweep_m5dB` | C1: bandwidth sweep −5 dB, decoupled (Root-MUSIC, ReconUNet-C, ReconUNet-CB) | ✅ done | 2026-10-01 17:42:00 | 2026-10-01 17:42:13 | 0min |
| `bw_sweep_p10dB` | C1: bandwidth sweep +10 dB, decoupled (Root-MUSIC, ReconUNet-C, ReconUNet-CB) | ✅ done | 2026-10-01 17:42:13 | 2026-10-01 17:42:27 | 0min |
| `bw_plot` | C1: quick-look figure | ✅ done | 2026-10-01 17:42:27 | 2026-10-01 17:42:30 | 0min |
| `eval_paper_testset_r2b` | C2: paper test split by K (+errors) | ✅ done | 2026-10-01 17:42:30 | 2026-10-01 17:42:53 | 0min |
| `eval_scenario_sweep_r2b` | C3: scenario sweep | ✅ done | 2026-10-01 17:42:53 | 2026-10-01 17:43:53 | 1min |
| `eval_table2_mild_r2b` | C4: Table II mild | ✅ done | 2026-10-01 17:43:53 | 2026-10-01 17:47:03 | 3min |
| `eval_table2_harsh_r2b` | C4: Table III harsh | ✅ done | 2026-10-01 17:47:03 | 2026-10-01 17:50:14 | 3min |
| `eval_paper_testset_bwrand_r2b` | C6: paper test split re-rendered at per-scene random bandwidth | ✅ done | 2026-10-01 17:50:14 | 2026-10-01 17:50:36 | 0min |
| `bootstrap_ci_r2b` | C2-C6: bootstrap 95 % CIs | ✅ done | 2026-10-01 17:50:36 | 2026-10-01 17:51:07 | 0min |
| `snapshot_sweep_r2b` | C5: snapshot sweep | ✅ done | 2026-10-01 17:51:07 | 2026-10-01 17:51:19 | 0min |
| `separation_sweep_r2b` | C5: separation sweep | ✅ done | 2026-10-01 17:51:19 | 2026-10-01 17:51:36 | 0min |
| `cost_r2b` | C7: cost incl. ReconUNet-CB | ✅ done | 2026-10-01 17:51:36 | 2026-10-01 17:51:40 | 0min |
| `export_r2b` | Copy CSVs to docs/revision_r2b + HEADLINE.md | ✅ done | 2026-10-01 17:51:40 | 2026-10-01 17:51:41 | 0min |
| `facts_r2b` | Append R2b sections to docs/revision_facts.md | ✅ done | 2026-10-01 17:51:41 | 2026-10-01 17:52:00 | 0min |

## Results (links)

| result | status | link |
|---|---|---|
| Headline numbers | ✅ 10-01 17:51 | [docs/revision_r2b/HEADLINE.md](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/docs/revision_r2b/HEADLINE.md) |
| All R2b CSVs (folder) | ✅ 10-01 17:52 | [docs/revision_r2b](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/docs/revision_r2b) |
| ReconUNet-CB training history | ✅ 10-01 17:40 | [experiments/runs/reconunet_cb_paper/checkpoints/history.json](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/experiments/runs/reconunet_cb_paper/checkpoints/history.json) |
| Bandwidth sweep (decoupled) | ✅ 10-01 17:41 | [experiments/runs/sweeps_r2b_20261001/bandwidth_sweep.csv](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/experiments/runs/sweeps_r2b_20261001/bandwidth_sweep.csv) |
| Bandwidth sweep (coupled) | ✅ 10-01 17:41 | [experiments/runs/sweeps_r2b_20261001/bandwidth_sweep_coupled.csv](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/experiments/runs/sweeps_r2b_20261001/bandwidth_sweep_coupled.csv) |
| Bandwidth sweep −5 dB | ✅ 10-01 17:42 | [experiments/runs/sweeps_r2b_20261001/bandwidth_sweep_m5dB.csv](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/experiments/runs/sweeps_r2b_20261001/bandwidth_sweep_m5dB.csv) |
| Bandwidth sweep +10 dB | ✅ 10-01 17:42 | [experiments/runs/sweeps_r2b_20261001/bandwidth_sweep_p10dB.csv](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/experiments/runs/sweeps_r2b_20261001/bandwidth_sweep_p10dB.csv) |
| Paper test split | ✅ 10-01 17:42 | [experiments/runs/eval_r2b_20261001/paper_testset_r2b/paper_testset_by_K.csv](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/experiments/runs/eval_r2b_20261001/paper_testset_r2b/paper_testset_by_K.csv) |
| Matched wideband test split | ✅ 10-01 17:50 | [experiments/runs/eval_r2b_20261001/paper_testset_bwrand_r2b/paper_testset_by_K.csv](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/experiments/runs/eval_r2b_20261001/paper_testset_bwrand_r2b/paper_testset_by_K.csv) |
| Scenario sweep | ✅ 10-01 17:43 | [experiments/runs/eval_r2b_20261001/scenario_sweep_r2b/scenario_sweep.csv](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/experiments/runs/eval_r2b_20261001/scenario_sweep_r2b/scenario_sweep.csv) |
| Table II mild | ✅ 10-01 17:47 | [experiments/runs/eval_r2b_20261001/table2_mild_r2b/table2_full.csv](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/experiments/runs/eval_r2b_20261001/table2_mild_r2b/table2_full.csv) |
| Table III harsh | ✅ 10-01 17:50 | [experiments/runs/eval_r2b_20261001/table2_harsh_r2b/table2_full.csv](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/experiments/runs/eval_r2b_20261001/table2_harsh_r2b/table2_full.csv) |
| Snapshot sweep | ✅ 10-01 17:51 | [experiments/runs/sweeps_r2b_20261001/snapshot_sweep.csv](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/experiments/runs/sweeps_r2b_20261001/snapshot_sweep.csv) |
| Separation sweep | ✅ 10-01 17:51 | [experiments/runs/sweeps_r2b_20261001/separation_sweep.csv](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/experiments/runs/sweeps_r2b_20261001/separation_sweep.csv) |
| Cost | ✅ 10-01 17:51 | [experiments/runs/sweeps_r2b_20261001/cost_summary_r2.csv](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/experiments/runs/sweeps_r2b_20261001/cost_summary_r2.csv) |
| R2b figures | ✅ 10-01 17:42 | [docs/figs_revision/r2b](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/docs/figs_revision/r2b) |
| Revision facts | ✅ 10-01 17:52 | [docs/revision_facts.md](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/docs/revision_facts.md) |
| Pipeline status log | ✅ 10-01 17:52 | [experiments/runs/revision_r2b_20261001/status.log](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/experiments/runs/revision_r2b_20261001/status.log) |

## R2b headline numbers (2026-10-01) — ReconUNet-CB (randomised source bandwidth)

## Plain statement

**Wideband (bw ≥ 0.2 and white sources, 9 cells: 3 scenarios × 3 bandwidths, decoupled delays, 0 dB).** Randomised-bandwidth training removed the wideband failure: the mean pooled RMSE over these cells is 1.14° for ReconUNet-CB vs 22.55° for ReconUNet-C and 5.81° for raw Root-MUSIC; ReconUNet-CB is at or below Root-MUSIC in 9/9 cells (its CI lies entirely above Root-MUSIC in 0/9). **At the training bandwidth 0.05** the sweep RMSE is moderate 0.78° vs 0.61°, crowded 2.57° vs 2.45°, moderate3 0.77° vs 0.67° (ReconUNet-CB vs ReconUNet-C); on the paper test split (rendered at bw 0.05) pooled RMSE 4.96° [4.51, 5.38] vs 4.23° [3.78, 4.67], median RMSPE 0.63° vs 0.58°. _(Generated by `r2b_report.py` from the CSVs: 'removed' = ReconUNet-CB ≤ Root-MUSIC in every wideband cell; 'substantially reduced' = mean wideband RMSE below half of ReconUNet-C's.)_

## Paper test split (rendered at bw 0.05), pooled over K = 1..4

| method | pooled RMSE (°) | 95 % CI | median RMSPE (°) | 95 % CI | scenes |
|---|---|---|---|---|---|
| ReconUNet-C | 4.23 | [3.78, 4.67] | 0.58 | [0.57, 0.59] | 12000 |
| ReconUNet-CB | 4.96 | [4.51, 5.38] | 0.63 | [0.61, 0.64] | 12000 |
| ReconUNet | 5.79 | [5.34, 6.22] | 0.94 | [0.93, 0.96] | 12000 |
| DA-MUSIC | 7.91 | [7.58, 8.22] | 3.17 | [3.12, 3.22] | 12000 |
| SubspaceNet | 8.37 | [8.03, 8.69] | 1.70 | [1.67, 1.74] | 12000 |
| ESPRIT | 11.00 | [10.55, 11.47] | 2.15 | [2.07, 2.23] | 12000 |
| R-MUSIC | 12.43 | [11.93, 12.91] | 0.80 | [0.78, 0.83] | 12000 |
| SubViT | 14.95 | [14.58, 15.31] | 0.58 | [0.57, 0.59] | 12000 |

## Matched wideband: paper test split re-rendered at each scene's drawn bandwidth, pooled over K

| method | pooled RMSE (°) | 95 % CI | median RMSPE (°) | 95 % CI | scenes |
|---|---|---|---|---|---|
| ReconUNet-CB | 4.80 | [4.36, 5.23] | 0.58 | [0.57, 0.59] | 12000 |
| DA-MUSIC | 7.65 | [7.34, 7.95] | 3.10 | [3.04, 3.16] | 12000 |
| ESPRIT | 10.59 | [10.10, 11.08] | 1.57 | [1.51, 1.63] | 12000 |
| SubspaceNet | 11.26 | [10.94, 11.61] | 2.13 | [2.09, 2.17] | 12000 |
| R-MUSIC | 11.60 | [11.12, 12.04] | 0.71 | [0.69, 0.73] | 12000 |
| SubViT | 14.57 | [14.21, 14.95] | 0.57 | [0.56, 0.58] | 12000 |
| ReconUNet | 15.20 | [14.83, 15.58] | 1.69 | [1.64, 1.75] | 12000 |
| ReconUNet-C | 15.45 | [15.09, 15.88] | 1.32 | [1.28, 1.38] | 12000 |

Pooled RMSE (°) by K:

| K | n | R-MUSIC | ESPRIT | ReconUNet | ReconUNet-C | ReconUNet-CB | SubspaceNet | DA-MUSIC | SubViT |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 3000 | 23.55 | 22.87 | 23.14 | 23.85 | 12.10 | 16.30 | 15.47 | 14.50 |
| 2 | 3000 | 13.69 | 11.49 | 16.88 | 18.81 | 4.77 | 11.17 | 7.97 | 15.13 |
| 3 | 3000 | 9.25 | 8.20 | 14.39 | 14.33 | 2.24 | 10.58 | 6.27 | 15.06 |
| 4 | 3000 | 6.34 | 5.75 | 12.07 | 11.13 | 2.40 | 10.21 | 5.02 | 13.93 |
| all | 12000 | 11.60 | 10.59 | 15.20 | 15.45 | 4.80 | 11.26 | 7.65 | 14.57 |


## Training

| model | epochs_run | best_epoch | best_val_loss | val_rmspe_at_best_deg | min_val_rmspe_deg | last_lr | early_stopped | wall_clock_min | min_per_epoch |
|---|---|---|---|---|---|---|---|---|---|
| ReconUNet-CB (R2b, L_rec only, randomised bw) | 204 | 179 | 0.08128 | 1.458 | 1.442 | 1e-06 | True | 529 | 2.59 |
| ReconUNet-C (R2, L_rec only, bw 0.05) | 207 | 182 | 0.07001 | 1.33 | 1.323 | 1e-06 | True | 531 | 2.57 |
| ReconUNet (R1 released, composite loss, bw 0.05) | 109 | 84 | 0.42371 | 2.174 | 2.018 | 8.2e-06 | True | 287 | 2.63 |


## Bandwidth sweep, pooled RMSE (°) — 0 dB, replica delays fixed at the training bandwidth 0.05 (decoupled, primary)

| scenario | bw_frac | Root-MUSIC | ReconUNet | ReconUNet-C | ReconUNet-CB | SubspaceNet | DA-MUSIC | SubViT |
|---|---|---|---|---|---|---|---|---|
| moderate | 0.01 | 5.91 | 2.01 | 2.05 | 0.96 | 5.33 | 6.72 | 10.07 |
| moderate | 0.02 | 5.93 | 1.85 | 0.82 | 0.81 | 4.36 | 4.78 | 7.37 |
| moderate | 0.05 | 5.91 | 1.79 | 0.61 | 0.78 | 2.92 | 3.60 | 8.72 |
| moderate | 0.1 | 5.67 | 1.94 | 1.65 | 0.67 | 3.73 | 3.33 | 7.09 |
| moderate | 0.2 | 5.91 | 14.08 | 22.70 | 0.53 | 5.87 | 3.12 | 5.35 |
| moderate | 0.4 | 5.63 | 24.65 | 26.42 | 0.53 | 8.10 | 2.98 | 5.74 |
| moderate | white | 5.27 | 32.71 | 30.88 | 0.53 | 11.64 | 2.84 | 6.91 |
| crowded | 0.01 | 9.51 | 9.04 | 5.93 | 5.57 | 11.27 | 8.33 | 20.25 |
| crowded | 0.02 | 9.13 | 5.82 | 3.49 | 3.39 | 9.20 | 7.41 | 17.95 |
| crowded | 0.05 | 8.87 | 4.80 | 2.45 | 2.57 | 7.91 | 6.65 | 16.32 |
| crowded | 0.1 | 8.80 | 5.00 | 3.01 | 2.24 | 8.58 | 6.14 | 14.39 |
| crowded | 0.2 | 8.78 | 14.52 | 16.67 | 1.75 | 10.74 | 5.83 | 14.13 |
| crowded | 0.4 | 8.02 | 22.50 | 19.88 | 2.27 | 11.61 | 5.41 | 14.24 |
| crowded | white | 7.56 | 22.28 | 20.52 | 2.81 | 17.56 | 5.55 | 14.26 |
| moderate3 | 0.01 | 4.86 | 4.69 | 3.37 | 3.61 | 7.68 | 6.29 | 15.21 |
| moderate3 | 0.02 | 4.54 | 2.82 | 2.04 | 1.95 | 5.66 | 4.64 | 11.54 |
| moderate3 | 0.05 | 4.64 | 2.90 | 0.67 | 0.77 | 5.46 | 3.79 | 10.96 |
| moderate3 | 0.1 | 4.46 | 3.13 | 1.63 | 0.89 | 5.55 | 3.47 | 8.04 |
| moderate3 | 0.2 | 3.70 | 12.48 | 19.29 | 0.62 | 8.94 | 3.37 | 7.95 |
| moderate3 | 0.4 | 3.75 | 24.72 | 21.97 | 0.58 | 10.48 | 3.25 | 7.41 |
| moderate3 | white | 3.72 | 25.89 | 24.57 | 0.60 | 14.10 | 3.32 | 8.03 |


## Bandwidth sweep, pooled RMSE (°) — 0 dB, replica delays ∝ 1/bw (coupled, secondary)

| scenario | bw_frac | Root-MUSIC | ReconUNet | ReconUNet-C | ReconUNet-CB | SubspaceNet | DA-MUSIC | SubViT |
|---|---|---|---|---|---|---|---|---|
| moderate | 0.01 | 5.64 | 2.43 | 1.22 | 0.93 | 5.85 | 6.48 | 10.15 |
| moderate | 0.02 | 6.91 | 2.12 | 2.09 | 0.78 | 4.13 | 4.81 | 7.93 |
| moderate | 0.05 | 5.91 | 1.79 | 0.61 | 0.78 | 2.92 | 3.60 | 8.72 |
| moderate | 0.1 | 5.80 | 1.34 | 0.90 | 0.70 | 3.58 | 3.02 | 8.47 |
| moderate | 0.2 | 7.29 | 13.76 | 24.95 | 0.91 | 7.20 | 3.18 | 8.22 |
| moderate | 0.4 | 6.81 | 25.61 | 25.34 | 1.35 | 8.61 | 3.04 | 7.91 |
| moderate | white | 5.27 | 32.71 | 30.88 | 0.53 | 11.64 | 2.84 | 6.91 |
| crowded | 0.01 | 10.25 | 9.01 | 6.41 | 5.50 | 11.45 | 8.24 | 19.71 |
| crowded | 0.02 | 9.43 | 5.79 | 3.10 | 3.49 | 9.14 | 7.09 | 17.08 |
| crowded | 0.05 | 8.87 | 4.80 | 2.45 | 2.57 | 7.91 | 6.65 | 16.32 |
| crowded | 0.1 | 8.77 | 5.45 | 3.09 | 2.96 | 7.99 | 5.99 | 16.31 |
| crowded | 0.2 | 10.00 | 14.19 | 17.61 | 2.77 | 11.20 | 6.37 | 16.49 |
| crowded | 0.4 | 9.90 | 22.43 | 19.68 | 3.31 | 11.68 | 6.53 | 16.68 |
| crowded | white | 7.56 | 22.28 | 20.52 | 2.81 | 17.56 | 5.55 | 14.26 |
| moderate3 | 0.01 | 5.28 | 5.88 | 3.91 | 2.45 | 7.72 | 6.62 | 14.86 |
| moderate3 | 0.02 | 4.51 | 2.92 | 1.66 | 1.20 | 5.29 | 4.55 | 10.93 |
| moderate3 | 0.05 | 4.64 | 2.90 | 0.67 | 0.77 | 5.46 | 3.79 | 10.96 |
| moderate3 | 0.1 | 4.34 | 3.13 | 1.87 | 0.70 | 5.26 | 3.57 | 9.63 |
| moderate3 | 0.2 | 4.85 | 12.99 | 20.06 | 0.67 | 9.33 | 3.54 | 9.79 |
| moderate3 | 0.4 | 4.94 | 24.60 | 21.86 | 0.98 | 9.90 | 3.53 | 9.68 |
| moderate3 | white | 3.72 | 25.89 | 24.57 | 0.60 | 14.10 | 3.32 | 8.03 |


## Bandwidth sweep, pooled RMSE (°) — −5 dB, decoupled

| scenario | bw_frac | Root-MUSIC | ReconUNet-C | ReconUNet-CB |
|---|---|---|---|---|
| moderate | 0.01 | 5.58 | 2.87 | 1.11 |
| moderate | 0.02 | 5.73 | 0.92 | 0.96 |
| moderate | 0.05 | 5.88 | 0.65 | 0.80 |
| moderate | 0.1 | 5.61 | 1.67 | 0.70 |
| moderate | 0.2 | 6.13 | 22.94 | 0.59 |
| moderate | 0.4 | 6.02 | 26.23 | 0.58 |
| moderate | white | 5.28 | 31.37 | 0.60 |
| crowded | 0.01 | 10.35 | 6.10 | 5.26 |
| crowded | 0.02 | 9.48 | 3.64 | 3.31 |
| crowded | 0.05 | 9.01 | 2.54 | 2.57 |
| crowded | 0.1 | 9.00 | 3.11 | 2.26 |
| crowded | 0.2 | 8.83 | 16.66 | 1.78 |
| crowded | 0.4 | 8.57 | 20.13 | 2.30 |
| crowded | white | 7.92 | 20.94 | 2.96 |
| moderate3 | 0.01 | 5.13 | 3.42 | 3.50 |
| moderate3 | 0.02 | 5.23 | 1.03 | 1.80 |
| moderate3 | 0.05 | 4.93 | 1.95 | 1.76 |
| moderate3 | 0.1 | 4.71 | 1.66 | 0.95 |
| moderate3 | 0.2 | 3.83 | 18.97 | 0.64 |
| moderate3 | 0.4 | 3.54 | 22.12 | 0.65 |
| moderate3 | white | 3.92 | 25.37 | 0.67 |


## Bandwidth sweep, pooled RMSE (°) — +10 dB, decoupled

| scenario | bw_frac | Root-MUSIC | ReconUNet-C | ReconUNet-CB |
|---|---|---|---|---|
| moderate | 0.01 | 6.01 | 1.77 | 0.95 |
| moderate | 0.02 | 6.01 | 0.81 | 0.81 |
| moderate | 0.05 | 6.00 | 0.60 | 0.80 |
| moderate | 0.1 | 5.89 | 1.65 | 0.55 |
| moderate | 0.2 | 5.90 | 22.59 | 0.52 |
| moderate | 0.4 | 5.62 | 25.98 | 0.52 |
| moderate | white | 5.28 | 30.36 | 0.54 |
| crowded | 0.01 | 9.05 | 5.97 | 5.54 |
| crowded | 0.02 | 8.99 | 3.37 | 3.51 |
| crowded | 0.05 | 8.81 | 2.68 | 2.57 |
| crowded | 0.1 | 8.85 | 3.12 | 2.24 |
| crowded | 0.2 | 8.64 | 16.85 | 1.74 |
| crowded | 0.4 | 8.29 | 19.93 | 2.31 |
| crowded | white | 7.53 | 20.69 | 2.54 |
| moderate3 | 0.01 | 4.19 | 3.40 | 3.36 |
| moderate3 | 0.02 | 4.19 | 1.30 | 1.96 |
| moderate3 | 0.05 | 4.53 | 0.64 | 0.81 |
| moderate3 | 0.1 | 4.34 | 1.61 | 0.67 |
| moderate3 | 0.2 | 3.70 | 19.12 | 0.63 |
| moderate3 | 0.4 | 3.86 | 21.68 | 0.56 |
| moderate3 | white | 3.86 | 24.12 | 0.59 |


## Cost

| model | class | trainable parameters | MMACs per scene | GPU fwd b1 ms/scene | GPU fwd+Root-MUSIC b1 ms/scene | CPU fwd b1 ms | GPU fwd b1024 ms/scene | GPU fwd+Root-MUSIC b1024 ms/scene | epochs | best_epoch | train_wallclock_min |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ReconUNet | EVDCovarianceReconstructionUNet | 337842 | 7.782 | 1.2 | 1.773 | 6.046 | 0.03229 | 0.06985 | 109 | 84 | 287 |
| ReconUNet-C | CovarianceOnlyReconstructionUNet | 323070 | 7.128 | 0.7837 | 1.357 | 0.4839 | 0.0136 | 0.05635 | 207 | 182 | 531 |
| ReconUNet-CB | CovarianceOnlyReconstructionUNet | 323070 | 7.128 | 0.7831 | 1.353 | 0.4667 | 0.01339 | 0.05637 | 204 | 179 | 529 |



## Master status log (tail)

```
  rc=0; for d in paper_testset_r2b scenario_sweep_r2b table2_mild_r2b table2_harsh_r2b paper_testset_bwrand_r2b; do
    if [ -f experiments/runs/eval_r2b_20261001/$d/errors.npz ]; then DOA_env/bin/python scripts/analysis/bootstrap_ci.py experiments/runs/eval_r2b_20261001/$d/errors.npz -o experiments/runs/eval_r2b_20261001/$d/${d}_ci.csv --table $d || rc=1; else echo "missing $d"; rc=1; fi
  done; exit $rc)
2026-10-01 17:51:07 END bootstrap_ci_r2b rc=0 elapsed=0min
2026-10-01 17:51:07 START snapshot_sweep_r2b  (DOA_env/bin/python scripts/analysis/snapshot_sweep.py -o experiments/runs/sweeps_r2b_20261001 --reconunet-c experiments/runs/reconunet_c_paper/checkpoints/best.pt --reconunet-cb experiments/runs/reconunet_cb_paper/checkpoints/best.pt)
2026-10-01 17:51:19 END snapshot_sweep_r2b rc=0 elapsed=0min
2026-10-01 17:51:19 START separation_sweep_r2b  (DOA_env/bin/python scripts/analysis/separation_sweep.py -o experiments/runs/sweeps_r2b_20261001 --reconunet-c experiments/runs/reconunet_c_paper/checkpoints/best.pt --reconunet-cb experiments/runs/reconunet_cb_paper/checkpoints/best.pt --damusic-dir experiments/runs/damusic_paper/ensemble_v2)
2026-10-01 17:51:36 END separation_sweep_r2b rc=0 elapsed=0min
2026-10-01 17:51:36 START cost_r2b  (DOA_env/bin/python scripts/analysis/cost_r2.py -o experiments/runs/sweeps_r2b_20261001 --reconunet-c experiments/runs/reconunet_c_paper/checkpoints/best.pt --status-log experiments/runs/revision_r2_20260930/status.log --reconunet-cb experiments/runs/reconunet_cb_paper/checkpoints/best.pt --status-log-cb experiments/runs/revision_r2b_20261001/status.log)
2026-10-01 17:51:40 END cost_r2b rc=0 elapsed=0min
2026-10-01 17:51:40 START export_r2b  (DOA_env/bin/python scripts/analysis/r2b_report.py export --eval-dir experiments/runs/eval_r2b_20261001 --sweeps-dir experiments/runs/sweeps_r2b_20261001 --run-dir experiments/runs/revision_r2b_20261001 --dest docs/revision_r2b)
2026-10-01 17:51:41 END export_r2b rc=0 elapsed=0min
2026-10-01 17:51:41 START facts_r2b  (facts_r2b)
2026-10-01 17:52:00 END facts_r2b rc=0 elapsed=0min
2026-10-01 17:52:00 PIPELINE DONE (18 stages rc=0, 0 failed)
```
