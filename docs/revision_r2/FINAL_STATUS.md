# Final status — revision round 2 (ReconUNet-C + bandwidth sweep), started 2026-09-30

**Last update:** 2026-09-30 21:45 UTC · **pipeline process:** running · **log says:** PIPELINE DONE · **stages:** 17/17 done, 0 failed

## ReconUNet-C training

Epoch 207/300, 532 min elapsed (2.57 min/epoch); best epoch 182 (val loss 0.07001, val RMSPE 1.330°); last: val RMSPE 1.332°, LR 1.0e-06; early stop after 25 epochs without improvement (currently 25). Full ReconUNet (R1) reached 2.174° val RMSPE at epoch 84 of 109.

## Stages

| stage | what | state | started (UTC) | ended | elapsed |
|---|---|---|---|---|---|
| `train_reconunet_c` | Part A: train ReconUNet-C, full scale (≤300 epochs, patience 25) | ✅ done | 2026-09-30 12:45:06 | 2026-09-30 21:36:36 | 531min |
| `train_summary` | Part A: training summary CSV | ✅ done | 2026-09-30 21:36:36 | 2026-09-30 21:36:37 | 0min |
| `eval_paper_testset_r2` | Part B1: paper test split by K (+errors) | ✅ done | 2026-09-30 21:36:37 | 2026-09-30 21:37:00 | 0min |
| `eval_scenario_sweep_r2` | Part B2: scenario sweep, 4 scenarios × 9 SNRs (+errors) | ✅ done | 2026-09-30 21:37:00 | 2026-09-30 21:37:58 | 0min |
| `eval_table2_mild_r2` | Part B3: Table II mild, all back ends, all SNRs | ✅ done | 2026-09-30 21:37:58 | 2026-09-30 21:40:34 | 2min |
| `eval_table2_harsh_r2` | Part B3: Table III harsh, all back ends, all SNRs | ✅ done | 2026-09-30 21:40:34 | 2026-09-30 21:43:08 | 2min |
| `bootstrap_ci_r2` | Part B1-3: bootstrap 95 % CIs | ✅ done | 2026-09-30 21:43:08 | 2026-09-30 21:43:31 | 0min |
| `snapshot_sweep_r2` | Part B4: snapshot sweep | ✅ done | 2026-09-30 21:43:31 | 2026-09-30 21:43:43 | 0min |
| `separation_sweep_r2` | Part B4: separation sweep + resolution probability | ✅ done | 2026-09-30 21:43:43 | 2026-09-30 21:43:59 | 0min |
| `cost_r2` | Part B5: cost (params, MMACs, GPU/CPU latency, training time) | ✅ done | 2026-09-30 21:43:59 | 2026-09-30 21:44:03 | 0min |
| `music_verification_r2` | Part B6: grid-MUSIC verification incl. ReconUNet-C | ✅ done | 2026-09-30 21:44:03 | 2026-09-30 21:44:07 | 0min |
| `figures_r2` | Part B7: MUSIC-spectrum figure with ReconUNet-C | ✅ done | 2026-09-30 21:44:07 | 2026-09-30 21:44:09 | 0min |
| `bw_sweep_decoupled` | Part C: bandwidth sweep, delays fixed at bw 0.05 (primary) | ✅ done | 2026-09-30 21:44:09 | 2026-09-30 21:44:38 | 0min |
| `bw_sweep_coupled` | Part C: bandwidth sweep, delays ∝ 1/bw (secondary) | ✅ done | 2026-09-30 21:44:38 | 2026-09-30 21:45:07 | 0min |
| `bw_plot` | Part C: quick-look figure | ✅ done | 2026-09-30 21:45:07 | 2026-09-30 21:45:10 | 0min |
| `export_r2` | Copy CSVs to docs/revision_r2 + HEADLINE.md | ✅ done | 2026-09-30 21:45:10 | 2026-09-30 21:45:11 | 0min |
| `facts_r2` | Part D: append R2 sections to docs/revision_facts.md | ✅ done | 2026-09-30 21:45:11 | 2026-09-30 21:45:28 | 0min |

## Results (links)

| result | status | link |
|---|---|---|
| Headline numbers | ✅ 09-30 21:45 | [docs/revision_r2/HEADLINE.md](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/docs/revision_r2/HEADLINE.md) |
| All R2 CSVs (folder) | ✅ 09-30 21:45 | [docs/revision_r2](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/docs/revision_r2) |
| ReconUNet-C training history | ✅ 09-30 21:36 | [experiments/runs/reconunet_c_paper/checkpoints/history.json](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/experiments/runs/reconunet_c_paper/checkpoints/history.json) |
| Paper test split by K | ✅ 09-30 21:36 | [experiments/runs/eval_r2_20260930/paper_testset_r2/paper_testset_by_K.csv](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/experiments/runs/eval_r2_20260930/paper_testset_r2/paper_testset_by_K.csv) |
| Scenario sweep | ✅ 09-30 21:37 | [experiments/runs/eval_r2_20260930/scenario_sweep_r2/scenario_sweep.csv](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/experiments/runs/eval_r2_20260930/scenario_sweep_r2/scenario_sweep.csv) |
| Table II mild | ✅ 09-30 21:40 | [experiments/runs/eval_r2_20260930/table2_mild_r2/table2_full.csv](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/experiments/runs/eval_r2_20260930/table2_mild_r2/table2_full.csv) |
| Table III harsh | ✅ 09-30 21:43 | [experiments/runs/eval_r2_20260930/table2_harsh_r2/table2_full.csv](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/experiments/runs/eval_r2_20260930/table2_harsh_r2/table2_full.csv) |
| Snapshot sweep | ✅ 09-30 21:43 | [experiments/runs/sweeps_r2_20260930/snapshot_sweep.csv](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/experiments/runs/sweeps_r2_20260930/snapshot_sweep.csv) |
| Separation sweep | ✅ 09-30 21:43 | [experiments/runs/sweeps_r2_20260930/separation_sweep.csv](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/experiments/runs/sweeps_r2_20260930/separation_sweep.csv) |
| Cost | ✅ 09-30 21:44 | [experiments/runs/sweeps_r2_20260930/cost_summary_r2.csv](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/experiments/runs/sweeps_r2_20260930/cost_summary_r2.csv) |
| Bandwidth sweep | ✅ 09-30 21:44 | [experiments/runs/sweeps_r2_20260930/bandwidth_sweep.csv](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/experiments/runs/sweeps_r2_20260930/bandwidth_sweep.csv) |
| Bandwidth sweep (coupled) | ✅ 09-30 21:45 | [experiments/runs/sweeps_r2_20260930/bandwidth_sweep_coupled.csv](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/experiments/runs/sweeps_r2_20260930/bandwidth_sweep_coupled.csv) |
| Bandwidth autocorrelation | ✅ 09-30 21:44 | [experiments/runs/sweeps_r2_20260930/bandwidth_autocorr.csv](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/experiments/runs/sweeps_r2_20260930/bandwidth_autocorr.csv) |
| R2 figures | ✅ 09-30 21:45 | [docs/figs_revision/r2](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/docs/figs_revision/r2) |
| Revision facts | ✅ 09-30 21:45 | [docs/revision_facts.md](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/docs/revision_facts.md) |
| Pipeline status log | ✅ 09-30 21:45 | [experiments/runs/revision_r2_20260930/status.log](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/experiments/runs/revision_r2_20260930/status.log) |

## R2 headline numbers (2026-09-30)

## Paper test split, pooled over K = 1..4

| method | pooled RMSE (°) | 95 % CI | median RMSPE (°) | 95 % CI | scenes |
|---|---|---|---|---|---|
| ReconUNet-C | 4.23 | [3.76, 4.67] | 0.58 | [0.57, 0.59] | 12000 |
| ReconUNet | 5.79 | [5.35, 6.18] | 0.94 | [0.93, 0.96] | 12000 |
| DA-MUSIC | 7.91 | [7.58, 8.25] | 3.17 | [3.12, 3.22] | 12000 |
| SubspaceNet | 8.37 | [8.02, 8.70] | 1.70 | [1.67, 1.74] | 12000 |
| ESPRIT | 11.00 | [10.54, 11.48] | 2.15 | [2.07, 2.24] | 12000 |
| R-MUSIC | 12.43 | [11.96, 12.88] | 0.80 | [0.78, 0.83] | 12000 |
| SubViT | 14.95 | [14.59, 15.30] | 0.58 | [0.57, 0.59] | 12000 |

## Training

| model | epochs_run | best_epoch | best_val_loss | val_rmspe_at_best_deg | min_val_rmspe_deg | last_lr | early_stopped | wall_clock_min | min_per_epoch |
|---|---|---|---|---|---|---|---|---|---|
| ReconUNet-C (R2, L_rec only) | 207 | 182 | 0.07001 | 1.33 | 1.323 | 1e-06 | True | 531 | 2.57 |
| ReconUNet (R1 released, composite loss) | 109 | 84 | 0.42371 | 2.174 | 2.018 | 8.2e-06 | True | 287 | 2.63 |


## Bandwidth sweep, pooled RMSE (°), 0 dB — replica delays fixed at the training bandwidth 0.05 (decoupled, primary)

| scenario | bw_frac | Root-MUSIC | ReconUNet | ReconUNet-C | SubspaceNet | DA-MUSIC | SubViT |
|---|---|---|---|---|---|---|---|
| moderate | 0.01 | 5.91 | 2.01 | 2.05 | 5.33 | 6.72 | 10.07 |
| moderate | 0.02 | 5.93 | 1.85 | 0.82 | 4.36 | 4.78 | 7.37 |
| moderate | 0.05 | 5.91 | 1.79 | 0.61 | 2.92 | 3.60 | 8.72 |
| moderate | 0.1 | 5.67 | 1.94 | 1.65 | 3.73 | 3.33 | 7.09 |
| moderate | 0.2 | 5.91 | 14.08 | 22.70 | 5.87 | 3.12 | 5.35 |
| moderate | 0.4 | 5.63 | 24.65 | 26.42 | 8.10 | 2.98 | 5.74 |
| moderate | white | 5.27 | 32.71 | 30.88 | 11.64 | 2.84 | 6.91 |
| crowded | 0.01 | 9.51 | 9.04 | 5.93 | 11.27 | 8.33 | 20.25 |
| crowded | 0.02 | 9.13 | 5.82 | 3.49 | 9.20 | 7.41 | 17.95 |
| crowded | 0.05 | 8.87 | 4.80 | 2.45 | 7.91 | 6.65 | 16.32 |
| crowded | 0.1 | 8.80 | 5.00 | 3.01 | 8.58 | 6.14 | 14.39 |
| crowded | 0.2 | 8.78 | 14.52 | 16.67 | 10.74 | 5.83 | 14.13 |
| crowded | 0.4 | 8.02 | 22.50 | 19.88 | 11.61 | 5.41 | 14.24 |
| crowded | white | 7.56 | 22.28 | 20.52 | 17.56 | 5.55 | 14.26 |
| moderate3 | 0.01 | 4.86 | 4.69 | 3.37 | 7.68 | 6.29 | 15.21 |
| moderate3 | 0.02 | 4.54 | 2.82 | 2.04 | 5.66 | 4.64 | 11.54 |
| moderate3 | 0.05 | 4.64 | 2.90 | 0.67 | 5.46 | 3.79 | 10.96 |
| moderate3 | 0.1 | 4.46 | 3.13 | 1.63 | 5.55 | 3.47 | 8.04 |
| moderate3 | 0.2 | 3.70 | 12.48 | 19.29 | 8.94 | 3.37 | 7.95 |
| moderate3 | 0.4 | 3.75 | 24.72 | 21.97 | 10.48 | 3.25 | 7.41 |
| moderate3 | white | 3.72 | 25.89 | 24.57 | 14.10 | 3.32 | 8.03 |


## Bandwidth sweep, pooled RMSE (°), 0 dB — replica delays scale with 1/bw (coupled, secondary)

| scenario | bw_frac | Root-MUSIC | ReconUNet | ReconUNet-C | SubspaceNet | DA-MUSIC | SubViT |
|---|---|---|---|---|---|---|---|
| moderate | 0.01 | 5.64 | 2.43 | 1.22 | 5.85 | 6.48 | 10.15 |
| moderate | 0.02 | 6.91 | 2.12 | 2.09 | 4.13 | 4.81 | 7.93 |
| moderate | 0.05 | 5.91 | 1.79 | 0.61 | 2.92 | 3.60 | 8.72 |
| moderate | 0.1 | 5.80 | 1.34 | 0.90 | 3.58 | 3.02 | 8.47 |
| moderate | 0.2 | 7.29 | 13.76 | 24.95 | 7.20 | 3.18 | 8.22 |
| moderate | 0.4 | 6.81 | 25.61 | 25.34 | 8.61 | 3.04 | 7.91 |
| moderate | white | 5.27 | 32.71 | 30.88 | 11.64 | 2.84 | 6.91 |
| crowded | 0.01 | 10.25 | 9.01 | 6.41 | 11.45 | 8.24 | 19.71 |
| crowded | 0.02 | 9.43 | 5.79 | 3.10 | 9.14 | 7.09 | 17.08 |
| crowded | 0.05 | 8.87 | 4.80 | 2.45 | 7.91 | 6.65 | 16.32 |
| crowded | 0.1 | 8.77 | 5.45 | 3.09 | 7.99 | 5.99 | 16.31 |
| crowded | 0.2 | 10.00 | 14.19 | 17.61 | 11.20 | 6.37 | 16.49 |
| crowded | 0.4 | 9.90 | 22.43 | 19.68 | 11.68 | 6.53 | 16.68 |
| crowded | white | 7.56 | 22.28 | 20.52 | 17.56 | 5.55 | 14.26 |
| moderate3 | 0.01 | 5.28 | 5.88 | 3.91 | 7.72 | 6.62 | 14.86 |
| moderate3 | 0.02 | 4.51 | 2.92 | 1.66 | 5.29 | 4.55 | 10.93 |
| moderate3 | 0.05 | 4.64 | 2.90 | 0.67 | 5.46 | 3.79 | 10.96 |
| moderate3 | 0.1 | 4.34 | 3.13 | 1.87 | 5.26 | 3.57 | 9.63 |
| moderate3 | 0.2 | 4.85 | 12.99 | 20.06 | 9.33 | 3.54 | 9.79 |
| moderate3 | 0.4 | 4.94 | 24.60 | 21.86 | 9.90 | 3.53 | 9.68 |
| moderate3 | white | 3.72 | 25.89 | 24.57 | 14.10 | 3.32 | 8.03 |


## Measured source autocorrelation |r(l)| and direct/replica |γ| (decoupled delays, first scenario)

| bw_frac | r0 | r1 | r2 | r3 | r4 | r5 | r6 | r7 | gamma_mean |
|---|---|---|---|---|---|---|---|---|---|
| 0.01 | 1.000 | 1.000 | 1.000 | 0.999 | 0.998 | 0.997 | 0.996 | 0.994 | 0.996 |
| 0.02 | 1.000 | 0.999 | 0.997 | 0.994 | 0.989 | 0.983 | 0.975 | 0.967 | 0.978 |
| 0.05 | 1.000 | 0.996 | 0.985 | 0.967 | 0.941 | 0.909 | 0.871 | 0.827 | 0.887 |
| 0.1 | 1.000 | 0.984 | 0.937 | 0.862 | 0.763 | 0.647 | 0.519 | 0.388 | 0.618 |
| 0.2 | 1.000 | 0.935 | 0.756 | 0.505 | 0.239 | 0.086 | 0.175 | 0.229 | 0.394 |
| 0.4 | 1.000 | 0.758 | 0.239 | 0.165 | 0.197 | 0.061 | 0.135 | 0.087 | 0.251 |
| white | 1.000 | 0.039 | 0.039 | 0.040 | 0.039 | 0.038 | 0.039 | 0.039 | 0.129 |



## Master status log (tail)

```
2026-09-30 21:44:03 START music_verification_r2  (bash -c DOA_env/bin/python scripts/analysis/music_verification.py -o experiments/runs/sweeps_r2_20260930 --reconunet-c experiments/runs/reconunet_c_paper/checkpoints/best.pt > experiments/runs/sweeps_r2_20260930/music_verification.md)
2026-09-30 21:44:07 END music_verification_r2 rc=0 elapsed=0min
2026-09-30 21:44:07 START figures_r2  (DOA_env/bin/python scripts/analysis/revision_figures.py --reconunet-c experiments/runs/reconunet_c_paper/checkpoints/best.pt --out-dir docs/figs_revision/r2)
2026-09-30 21:44:09 END figures_r2 rc=0 elapsed=0min
2026-09-30 21:44:09 START bw_sweep_decoupled  (DOA_env/bin/python scripts/analysis/bandwidth_sweep.py --mode decoupled -o experiments/runs/sweeps_r2_20260930 --damusic-dir experiments/runs/damusic_paper/ensemble_v2 --reconunet-c experiments/runs/reconunet_c_paper/checkpoints/best.pt)
2026-09-30 21:44:38 END bw_sweep_decoupled rc=0 elapsed=0min
2026-09-30 21:44:38 START bw_sweep_coupled  (DOA_env/bin/python scripts/analysis/bandwidth_sweep.py --mode coupled -o experiments/runs/sweeps_r2_20260930 --damusic-dir experiments/runs/damusic_paper/ensemble_v2 --reconunet-c experiments/runs/reconunet_c_paper/checkpoints/best.pt)
2026-09-30 21:45:07 END bw_sweep_coupled rc=0 elapsed=0min
2026-09-30 21:45:07 START bw_plot  (DOA_env/bin/python scripts/analysis/bandwidth_sweep.py --mode plot -o experiments/runs/sweeps_r2_20260930 --fig-dir docs/figs_revision/r2)
2026-09-30 21:45:10 END bw_plot rc=0 elapsed=0min
2026-09-30 21:45:10 START export_r2  (DOA_env/bin/python scripts/analysis/r2_report.py export --eval-dir experiments/runs/eval_r2_20260930 --sweeps-dir experiments/runs/sweeps_r2_20260930 --run-dir experiments/runs/revision_r2_20260930 --dest docs/revision_r2)
2026-09-30 21:45:11 END export_r2 rc=0 elapsed=0min
2026-09-30 21:45:11 START facts_r2  (facts_r2)
2026-09-30 21:45:28 END facts_r2 rc=0 elapsed=0min
2026-09-30 21:45:28 PIPELINE DONE (17 stages rc=0, 0 failed)
```
