# Results status — revision R2b (ReconUNet-CB, randomised source bandwidth), started 2026-10-01

**Last update:** 2026-10-01 10:51 UTC · **pipeline process:** running · **log says:** in progress · **stages:** 0/18 done, 0 failed

## ReconUNet-CB training

Epoch 46/300, 120 min elapsed (2.61 min/epoch); best epoch 46 (val loss 0.09056, val RMSPE 1.560°); last: val RMSPE 1.560°, LR 1.0e-04; early stop after 25 epochs without improvement (currently 0). ReconUNet-C (R2) reached 1.330° val RMSPE at epoch 182 of 207 — on the bw-0.05 validation split, so not directly comparable with ReconUNet-CB's randomised-bandwidth validation split.

## Stages

| stage | what | state | started (UTC) | ended | elapsed |
|---|---|---|---|---|---|
| `train_reconunet_cb` | Part B: train ReconUNet-CB from scratch on the randomised-bandwidth corpus (≤300 epochs, patience 25) | 🔄 running | 2026-10-01 08:51:07 |  |  |
| `train_summary` | Part B: training summary CSV | ⏳ pending |  |  |  |
| `bw_sweep_decoupled` | C1: bandwidth sweep 0 dB, decoupled (primary), all methods | ⏳ pending |  |  |  |
| `bw_sweep_coupled` | C1: bandwidth sweep 0 dB, coupled | ⏳ pending |  |  |  |
| `bw_sweep_m5dB` | C1: bandwidth sweep −5 dB, decoupled (Root-MUSIC, ReconUNet-C, ReconUNet-CB) | ⏳ pending |  |  |  |
| `bw_sweep_p10dB` | C1: bandwidth sweep +10 dB, decoupled (Root-MUSIC, ReconUNet-C, ReconUNet-CB) | ⏳ pending |  |  |  |
| `bw_plot` | C1: quick-look figure | ⏳ pending |  |  |  |
| `eval_paper_testset_r2b` | C2: paper test split by K (+errors) | ⏳ pending |  |  |  |
| `eval_scenario_sweep_r2b` | C3: scenario sweep | ⏳ pending |  |  |  |
| `eval_table2_mild_r2b` | C4: Table II mild | ⏳ pending |  |  |  |
| `eval_table2_harsh_r2b` | C4: Table III harsh | ⏳ pending |  |  |  |
| `eval_paper_testset_bwrand_r2b` | C6: paper test split re-rendered at per-scene random bandwidth | ⏳ pending |  |  |  |
| `bootstrap_ci_r2b` | C2-C6: bootstrap 95 % CIs | ⏳ pending |  |  |  |
| `snapshot_sweep_r2b` | C5: snapshot sweep | ⏳ pending |  |  |  |
| `separation_sweep_r2b` | C5: separation sweep | ⏳ pending |  |  |  |
| `cost_r2b` | C7: cost incl. ReconUNet-CB | ⏳ pending |  |  |  |
| `export_r2b` | Copy CSVs to docs/revision_r2b + HEADLINE.md | ⏳ pending |  |  |  |
| `facts_r2b` | Append R2b sections to docs/revision_facts.md | ⏳ pending |  |  |  |

## Results (links)

| result | status | link |
|---|---|---|
| Headline numbers | ⏳ pending | [docs/revision_r2b/HEADLINE.md](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/docs/revision_r2b/HEADLINE.md) |
| All R2b CSVs (folder) | ✅ 10-01 08:51 | [docs/revision_r2b](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/docs/revision_r2b) |
| ReconUNet-CB training history | ✅ 10-01 10:50 | [experiments/runs/reconunet_cb_paper/checkpoints/history.json](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/experiments/runs/reconunet_cb_paper/checkpoints/history.json) |
| Bandwidth sweep (decoupled) | ⏳ pending | [experiments/runs/sweeps_r2b_20261001/bandwidth_sweep.csv](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/experiments/runs/sweeps_r2b_20261001/bandwidth_sweep.csv) |
| Bandwidth sweep (coupled) | ⏳ pending | [experiments/runs/sweeps_r2b_20261001/bandwidth_sweep_coupled.csv](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/experiments/runs/sweeps_r2b_20261001/bandwidth_sweep_coupled.csv) |
| Bandwidth sweep −5 dB | ⏳ pending | [experiments/runs/sweeps_r2b_20261001/bandwidth_sweep_m5dB.csv](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/experiments/runs/sweeps_r2b_20261001/bandwidth_sweep_m5dB.csv) |
| Bandwidth sweep +10 dB | ⏳ pending | [experiments/runs/sweeps_r2b_20261001/bandwidth_sweep_p10dB.csv](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/experiments/runs/sweeps_r2b_20261001/bandwidth_sweep_p10dB.csv) |
| Paper test split | ⏳ pending | [experiments/runs/eval_r2b_20261001/paper_testset_r2b/paper_testset_by_K.csv](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/experiments/runs/eval_r2b_20261001/paper_testset_r2b/paper_testset_by_K.csv) |
| Matched wideband test split | ⏳ pending | [experiments/runs/eval_r2b_20261001/paper_testset_bwrand_r2b/paper_testset_by_K.csv](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/experiments/runs/eval_r2b_20261001/paper_testset_bwrand_r2b/paper_testset_by_K.csv) |
| Scenario sweep | ⏳ pending | [experiments/runs/eval_r2b_20261001/scenario_sweep_r2b/scenario_sweep.csv](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/experiments/runs/eval_r2b_20261001/scenario_sweep_r2b/scenario_sweep.csv) |
| Table II mild | ⏳ pending | [experiments/runs/eval_r2b_20261001/table2_mild_r2b/table2_full.csv](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/experiments/runs/eval_r2b_20261001/table2_mild_r2b/table2_full.csv) |
| Table III harsh | ⏳ pending | [experiments/runs/eval_r2b_20261001/table2_harsh_r2b/table2_full.csv](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/experiments/runs/eval_r2b_20261001/table2_harsh_r2b/table2_full.csv) |
| Snapshot sweep | ⏳ pending | [experiments/runs/sweeps_r2b_20261001/snapshot_sweep.csv](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/experiments/runs/sweeps_r2b_20261001/snapshot_sweep.csv) |
| Separation sweep | ⏳ pending | [experiments/runs/sweeps_r2b_20261001/separation_sweep.csv](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/experiments/runs/sweeps_r2b_20261001/separation_sweep.csv) |
| Cost | ⏳ pending | [experiments/runs/sweeps_r2b_20261001/cost_summary_r2.csv](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/experiments/runs/sweeps_r2b_20261001/cost_summary_r2.csv) |
| R2b figures | ✅ 10-01 08:51 | [docs/figs_revision/r2b](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/docs/figs_revision/r2b) |
| Revision facts | ✅ 10-01 08:50 | [docs/revision_facts.md](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/docs/revision_facts.md) |
| Pipeline status log | ✅ 10-01 08:51 | [experiments/runs/revision_r2b_20261001/status.log](https://github.com/itamaralmog96/ReconUNet-DoA/blob/main/experiments/runs/revision_r2b_20261001/status.log) |

## Master status log (tail)

```
2026-10-01 08:51:07 PIPELINE START pid=3978057 gpu=NVIDIA RTX 2000 Ada Generation git=607feb8
2026-10-01 08:51:07 START train_reconunet_cb  (DOA_env/bin/python -m reconunet.cli.train --config configs/train/reconunet_cb_paper.yaml)
```
