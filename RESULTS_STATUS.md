# Results status — revision pipeline 2026-09-20

**Last update:** 2026-09-20 07:46 UTC · **pipeline process:** not running · **pipeline log says:** not started

This page is rewritten by the auto-push watcher every 30 minutes. Links point at the files on `main`.

**Progress:** 0/26 stages done, 0 failed.

## ETA

- Pending: 26 stages; rough worst-case remaining ≈ 867 min (9 ablation variants ≈ 117 min run two at a time → ≈ 58 min, 3 DA-MUSIC runs ≤ 690 min, evals + extras ≈ 60 min).

## Results (links)

| result | status | link |
|---|---|---|
| Ablation results (9 variants × eval sets) | ⏳ pending | [experiments/runs/ablation_20260920/ablation_results.csv](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/experiments/runs/ablation_20260920/ablation_results.csv) |
| Route comparison (01_full, 08_no_evd_heads) | ⏳ pending | [experiments/runs/ablation_20260920/route_comparison.csv](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/experiments/runs/ablation_20260920/route_comparison.csv) |
| Ablation status log | ⏳ pending | [experiments/runs/ablation_20260920/status.log](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/experiments/runs/ablation_20260920/status.log) |
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
| Revision facts (docs/revision_facts.md) | ⏳ pending | [docs/revision_facts.md](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/docs/revision_facts.md) |
| Revision figures folder | ⏳ pending | [docs/figs_revision](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/docs/figs_revision) |
| Original 2026-09-10 results + HTML report | ✅ 09-10 11:25 | [experiments/runs/eval_coherent_20260910](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/experiments/runs/eval_coherent_20260910) |
| Pipeline master status log | ⏳ pending | [experiments/runs/revision2_20260920/status.log](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/experiments/runs/revision2_20260920/status.log) |
| Auto-push log | ⏳ pending | [experiments/runs/autopush.log](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/experiments/runs/autopush.log) |

## Stages

| stage | what | state | started (UTC) | ended | elapsed |
|---|---|---|---|---|---|
| `abl_train_01_full` | Ablation train 01_full | ⏳ pending |  |  |  |
| `abl_train_02_rec_only` | Ablation train 02_rec_only | ⏳ pending |  |  |  |
| `abl_train_03_rec_proj` | Ablation train 03_rec_proj | ⏳ pending |  |  |  |
| `abl_train_04_no_dom` | Ablation train 04_no_dom | ⏳ pending |  |  |  |
| `abl_train_05_no_eig` | Ablation train 05_no_eig | ⏳ pending |  |  |  |
| `abl_train_06_single_lag` | Ablation train 06_single_lag | ⏳ pending |  |  |  |
| `abl_train_07_relu` | Ablation train 07_relu | ⏳ pending |  |  |  |
| `abl_train_08_no_evd_heads` | Ablation train 08_no_evd_heads | ⏳ pending |  |  |  |
| `abl_train_09_fixed_imperf` | Ablation train 09_fixed_imperf | ⏳ pending |  |  |  |
| `abl_eval` | Ablation evaluation → ablation_results.csv / route_comparison.csv | ⏳ pending |  |  |  |
| `facts_ablation` | Append ablation numbers to revision_facts.md | ⏳ pending |  |  |  |
| `damusic_k2_v2` | DA-MUSIC K=2 continuation (≤600 epochs) | ⏳ pending |  |  |  |
| `damusic_k3_v2` | DA-MUSIC K=3 continuation (≤600 epochs) | ⏳ pending |  |  |  |
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
