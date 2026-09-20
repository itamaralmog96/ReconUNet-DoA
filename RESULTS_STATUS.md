# Results status — revision pipeline 2026-09-20

**Last update:** 2026-09-20 12:49 UTC · **pipeline process:** running · **pipeline log says:** in progress

This page is rewritten by the auto-push watcher every 30 minutes. Links point at the files on `main`.

**Progress:** 12/26 stages done, 0 failed.

## ETA

- `damusic_k3_v2` at epoch 377/600 after 60 min; worst case 171 more min at 46 s/epoch, sooner if early stopping (patience 25) triggers
- Pending: 13 stages; rough worst-case remaining ≈ 290 min (0 ablation variants ≈ 0 min run two at a time → ≈ 0 min, 1 DA-MUSIC runs ≤ 230 min, evals + extras ≈ 60 min).

## Results (links)

| result | status | link |
|---|---|---|
| Ablation results (9 variants × eval sets) | ✅ 09-20 09:22 | [experiments/runs/ablation_20260920/ablation_results.csv](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/experiments/runs/ablation_20260920/ablation_results.csv) |
| Route comparison (01_full, 08_no_evd_heads) | ✅ 09-20 09:22 | [experiments/runs/ablation_20260920/route_comparison.csv](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/experiments/runs/ablation_20260920/route_comparison.csv) |
| Ablation status log | ✅ 09-20 09:22 | [experiments/runs/ablation_20260920/status.log](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/experiments/runs/ablation_20260920/status.log) |
| DA-MUSIC v2: paper test split by K | ⏳ pending | [experiments/runs/eval_coherent_20260910/paper_testset_v2/paper_testset_by_K.csv](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/experiments/runs/eval_coherent_20260910/paper_testset_v2/paper_testset_by_K.csv) |
| DA-MUSIC v2: scenario sweep | ⏳ pending | [experiments/runs/eval_coherent_20260910/scenario_sweep_v2/scenario_sweep.csv](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/experiments/runs/eval_coherent_20260910/scenario_sweep_v2/scenario_sweep.csv) |
| DA-MUSIC v2: Table II mild | ⏳ pending | [experiments/runs/eval_coherent_20260910/table2_mild_v2/table2_full.csv](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/experiments/runs/eval_coherent_20260910/table2_mild_v2/table2_full.csv) |
| DA-MUSIC v2: Table II harsh | ⏳ pending | [experiments/runs/eval_coherent_20260910/table2_harsh_v2/table2_full.csv](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/experiments/runs/eval_coherent_20260910/table2_harsh_v2/table2_full.csv) |
| Bootstrap CIs: Table II mild | ⏳ pending | [experiments/runs/eval_coherent_20260910/table2_mild_v2/table2_mild_v2_ci.csv](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/experiments/runs/eval_coherent_20260910/table2_mild_v2/table2_mild_v2_ci.csv) |
| Bootstrap CIs: Table II harsh | ⏳ pending | [experiments/runs/eval_coherent_20260910/table2_harsh_v2/table2_harsh_v2_ci.csv](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/experiments/runs/eval_coherent_20260910/table2_harsh_v2/table2_harsh_v2_ci.csv) |
| Bootstrap CIs: scenario sweep | ⏳ pending | [experiments/runs/eval_coherent_20260910/scenario_sweep_v2/scenario_sweep_v2_ci.csv](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/experiments/runs/eval_coherent_20260910/scenario_sweep_v2/scenario_sweep_v2_ci.csv) |
| Bootstrap CIs: paper test split | ⏳ pending | [experiments/runs/eval_coherent_20260910/paper_testset_v2/paper_testset_v2_ci.csv](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/experiments/runs/eval_coherent_20260910/paper_testset_v2/paper_testset_v2_ci.csv) |
| Snapshot sweep | ⏳ pending | [experiments/runs/sweeps_20260920/snapshot_sweep.csv](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/experiments/runs/sweeps_20260920/snapshot_sweep.csv) |
| Separation sweep | ⏳ pending | [experiments/runs/sweeps_20260920/separation_sweep.csv](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/experiments/runs/sweeps_20260920/separation_sweep.csv) |
| FBSS baseline | ⏳ pending | [experiments/runs/sweeps_20260920/fbss_baseline.csv](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/experiments/runs/sweeps_20260920/fbss_baseline.csv) |
| MUSIC verification | ⏳ pending | [experiments/runs/sweeps_20260920/music_verification.csv](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/experiments/runs/sweeps_20260920/music_verification.csv) |
| Revision facts (docs/revision_facts.md) | ✅ 09-20 09:22 | [docs/revision_facts.md](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/docs/revision_facts.md) |
| Revision figures folder | ⏳ pending | [docs/figs_revision](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/docs/figs_revision) |
| Original 2026-09-10 results + HTML report | ✅ 09-10 11:25 | [experiments/runs/eval_coherent_20260910](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/experiments/runs/eval_coherent_20260910) |
| Pipeline master status log | ✅ 09-20 11:49 | [experiments/runs/revision2_20260920/status.log](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/experiments/runs/revision2_20260920/status.log) |
| Auto-push log | ✅ 09-20 12:19 | [experiments/runs/autopush.log](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/experiments/runs/autopush.log) |

## Stages

| stage | what | state | started (UTC) | ended | elapsed |
|---|---|---|---|---|---|
| `abl_train_01_full` | Ablation train 01_full | ✅ done | 2026-09-20 07:49:16 | 2026-09-20 08:08:03 | 18min |
| `abl_train_02_rec_only` | Ablation train 02_rec_only | ✅ done | 2026-09-20 07:49:16 | 2026-09-20 08:08:05 | 18min |
| `abl_train_03_rec_proj` | Ablation train 03_rec_proj | ✅ done | 2026-09-20 08:08:05 | 2026-09-20 08:27:07 | 19min |
| `abl_train_04_no_dom` | Ablation train 04_no_dom | ✅ done | 2026-09-20 08:08:05 | 2026-09-20 08:27:11 | 19min |
| `abl_train_05_no_eig` | Ablation train 05_no_eig | ✅ done | 2026-09-20 08:27:11 | 2026-09-20 08:45:39 | 18min |
| `abl_train_06_single_lag` | Ablation train 06_single_lag | ✅ done | 2026-09-20 08:27:11 | 2026-09-20 08:43:01 | 15min |
| `abl_train_07_relu` | Ablation train 07_relu | ✅ done | 2026-09-20 08:45:39 | 2026-09-20 09:04:21 | 18min |
| `abl_train_08_no_evd_heads` | Ablation train 08_no_evd_heads | ✅ done | 2026-09-20 08:45:39 | 2026-09-20 09:04:19 | 18min |
| `abl_train_09_fixed_imperf` | Ablation train 09_fixed_imperf | ✅ done | 2026-09-20 09:04:21 | 2026-09-20 09:20:12 | 15min |
| `abl_eval` | Ablation evaluation → ablation_results.csv / route_comparison.csv | ✅ done | 2026-09-20 09:20:12 | 2026-09-20 09:22:15 | 2min |
| `facts_ablation` | Append ablation numbers to revision_facts.md | ✅ done | 2026-09-20 09:22:15 | 2026-09-20 09:22:17 | 0min |
| `damusic_k2_v2` | DA-MUSIC K=2 continuation (≤600 epochs) | ✅ done | 2026-09-20 09:22:17 | 2026-09-20 11:49:26 | 147min |
| `damusic_k3_v2` | DA-MUSIC K=3 continuation (≤600 epochs) | 🔄 running | 2026-09-20 11:49:26 |  |  |
| `damusic_k4_v2` | DA-MUSIC K=4 continuation (≤600 epochs) | ⏳ pending |  |  |  |
| `eval_paper_testset_v2` | Eval v2: paper test split | ⏳ pending |  |  |  |
| `eval_scenario_sweep_v2` | Eval v2: scenario sweep | ⏳ pending |  |  |  |
| `eval_table2_mild_v2` | Eval v2: Table II mild | ⏳ pending |  |  |  |
| `eval_table2_harsh_v2` | Eval v2: Table II harsh | ⏳ pending |  |  |  |
| `facts_damusic` | Append DA-MUSIC v2 numbers to revision_facts.md | ⏳ pending |  |  |  |
| `bootstrap_ci` | Bootstrap 95 % CIs (*_ci.csv) | ⏳ pending |  |  |  |
| `snapshot_sweep` | Snapshot sweep T ∈ {8…512} | ⏳ pending |  |  |  |
| `separation_sweep` | Separation sweep Δθ ∈ {2…15}° | ⏳ pending |  |  |  |
| `fbss_baseline` | FBSS Root-MUSIC baseline | ⏳ pending |  |  |  |
| `revision_figures` | Revision figures (vector PDF) | ⏳ pending |  |  |  |
| `music_verification` | Grid-MUSIC verification | ⏳ pending |  |  |  |
| `facts_extras` | Append sweep/FBSS numbers to revision_facts.md | ⏳ pending |  |  |  |

## Latest numbers: ablation on the paper test split (pooled RMSE °, median RMSPE °, rel. cov. error, subspace dist, eigengap err)

| variant | RMSE | median | cov_err | proj_dist | gap_err | n |
|---|---|---|---|---|---|---|
| 01_full | 9.22 | 1.28 | 0.357 | 0.534 | 0.128 | 12000 |
| 02_rec_only | 7.27 | 1.00 | 0.288 | 0.380 | 0.134 | 12000 |
| 03_rec_proj | 7.03 | 1.01 | 0.286 | 0.376 | 0.134 | 12000 |
| 04_no_dom | 9.43 | 1.20 | 0.365 | 0.504 | 0.121 | 12000 |
| 05_no_eig | 7.37 | 1.06 | 0.291 | 0.417 | 0.156 | 12000 |
| 06_single_lag | 9.87 | 1.27 | 0.361 | 0.551 | 0.140 | 12000 |
| 07_relu | 8.54 | 1.18 | 0.332 | 0.494 | 0.105 | 12000 |
| 08_no_evd_heads | 6.05 | 0.84 | 0.217 | 0.265 | 0.204 | 12000 |
| 09_fixed_imperf | 9.27 | 1.27 | 0.355 | 0.531 | 0.119 | 12000 |

## Latest numbers: DA-MUSIC continuation (validation RMSPE at the best epoch, v1 = 300-epoch cap)

| K | epochs so far | best epoch | best val RMSPE v2 (°) | best val RMSPE v1 (°) | current LR |
|---|---|---|---|---|---|
| 2 | 491 | 466 | 4.983 | 5.120 | 1.0e-06 |
| 3 | 377 | 377 | 4.422 | 4.566 | 1.7e-05 |

## Master status log (tail)

```
2026-09-20 09:04:21 END abl_train_07_relu rc=0 elapsed=18min
2026-09-20 09:04:21 START abl_train_09_fixed_imperf  (DOA_env/bin/python -m reconunet.cli.train --config configs/train/ablation_09_fixed_imperf.yaml)
2026-09-20 09:20:12 END abl_train_09_fixed_imperf rc=0 elapsed=15min
2026-09-20 09:20:12 START abl_eval  (DOA_env/bin/python scripts/analysis/ablation_eval.py --ablation-dir experiments/runs/ablation_20260920)
2026-09-20 09:22:15 END abl_eval rc=0 elapsed=2min
2026-09-20 09:22:15 START facts_ablation  (bash -c 
  DOA_env/bin/python scripts/analysis/revision_facts.py append-csv --title 'Ablation — reduced protocol (10 % seeded subset, 40 epochs), Root-MUSIC back end' --csv experiments/runs/ablation_20260920/ablation_results.csv      --note 'Columns: pooled RMSE (paper eq. 31), median per-scene RMSPE, relative covariance error ||R_hat-R*||_F/||R*||_F, leading-K projector distance ||P_hat-P*||_F, relative eigengap error |gap_hat-gap*|/gap* (all from eigh(R_hat)); eval sets: paper test split (3000 scenes per K), Moderate and Crowded at 0 dB (1000 scenes), and for 01/09 the paper test split rendered with the single fixed imperfection realisation used to train 09.' &&
  DOA_env/bin/python scripts/analysis/revision_facts.py append-csv --title 'Route comparison — covariance route (eigh of R_hat) vs subspace route (EVD-head eigenvectors)' --csv experiments/runs/ablation_20260920/route_comparison.csv)
2026-09-20 09:22:17 END facts_ablation rc=0 elapsed=0min
2026-09-20 09:22:17 START damusic_k2_v2  (DOA_env/bin/python -m reconunet.cli.train --config configs/train/damusic_paper_k2_v2.yaml)
2026-09-20 11:49:26 END damusic_k2_v2 rc=0 elapsed=147min
2026-09-20 11:49:26 START damusic_k3_v2  (DOA_env/bin/python -m reconunet.cli.train --config configs/train/damusic_paper_k3_v2.yaml)
```
