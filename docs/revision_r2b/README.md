# Revision R2b (2026-10-01) — result CSVs

Copied by `scripts/analysis/r2b_report.py export` at the end of `scripts/run_revision_r2b_pipeline.sh`. Same column schema as R1/R2; `ReconUNet-CB` = the covariance-only network (same architecture and loss as `ReconUNet-C`) trained from scratch on the randomised-source-bandwidth corpus (`configs/data/paper_corpus_bwrand.yaml`). `paper_testset_bwrand_*` = the paper test split re-rendered with each scene's own drawn bandwidth (matched wideband). DA-MUSIC = v2 ensemble. CIs: percentile bootstrap over scenes, 1000 resamples, seed 20260920.

| file | source |
|---|---|
| `paper_testset_by_K.csv` | `experiments/runs/eval_r2b_20261001/paper_testset_r2b/paper_testset_by_K.csv` |
| `paper_testset_ci.csv` | `experiments/runs/eval_r2b_20261001/paper_testset_r2b/paper_testset_r2b_ci.csv` |
| `paper_testset_bwrand_by_K.csv` | `experiments/runs/eval_r2b_20261001/paper_testset_bwrand_r2b/paper_testset_by_K.csv` |
| `paper_testset_bwrand_ci.csv` | `experiments/runs/eval_r2b_20261001/paper_testset_bwrand_r2b/paper_testset_bwrand_r2b_ci.csv` |
| `scenario_sweep.csv` | `experiments/runs/eval_r2b_20261001/scenario_sweep_r2b/scenario_sweep.csv` |
| `scenario_sweep_ci.csv` | `experiments/runs/eval_r2b_20261001/scenario_sweep_r2b/scenario_sweep_r2b_ci.csv` |
| `scenario_sweep_grid.png` | `experiments/runs/eval_r2b_20261001/scenario_sweep_r2b/scenario_sweep_grid.png` |
| `table2_mild_full.csv` | `experiments/runs/eval_r2b_20261001/table2_mild_r2b/table2_full.csv` |
| `table2_mild_ci.csv` | `experiments/runs/eval_r2b_20261001/table2_mild_r2b/table2_mild_r2b_ci.csv` |
| `table2_harsh_full.csv` | `experiments/runs/eval_r2b_20261001/table2_harsh_r2b/table2_full.csv` |
| `table2_harsh_ci.csv` | `experiments/runs/eval_r2b_20261001/table2_harsh_r2b/table2_harsh_r2b_ci.csv` |
| `snapshot_sweep.csv` | `experiments/runs/sweeps_r2b_20261001/snapshot_sweep.csv` |
| `separation_sweep.csv` | `experiments/runs/sweeps_r2b_20261001/separation_sweep.csv` |
| `cost_summary.csv` | `experiments/runs/sweeps_r2b_20261001/cost_summary_r2.csv` |
| `cost_latency.csv` | `experiments/runs/sweeps_r2b_20261001/cost_latency_r2.csv` |
| `bandwidth_sweep.csv` | `experiments/runs/sweeps_r2b_20261001/bandwidth_sweep.csv` |
| `bandwidth_sweep_coupled.csv` | `experiments/runs/sweeps_r2b_20261001/bandwidth_sweep_coupled.csv` |
| `bandwidth_sweep_m5dB.csv` | `experiments/runs/sweeps_r2b_20261001/bandwidth_sweep_m5dB.csv` |
| `bandwidth_sweep_p10dB.csv` | `experiments/runs/sweeps_r2b_20261001/bandwidth_sweep_p10dB.csv` |
| `bandwidth_autocorr.csv` | `experiments/runs/sweeps_r2b_20261001/bandwidth_autocorr.csv` |
| `bandwidth_autocorr_coupled.csv` | `experiments/runs/sweeps_r2b_20261001/bandwidth_autocorr_coupled.csv` |
| `reconunet_cb_training.csv` | `experiments/runs/revision_r2b_20261001/reconunet_cb_training.csv` |
| `table2_mild_reconunet_backends_by_snr.csv` | derived from `table2_mild_full.csv` |
| `table2_harsh_reconunet_backends_by_snr.csv` | derived from `table2_harsh_full.csv` |
| `bandwidth_sweep_wide.csv` | derived from `bandwidth_sweep.csv` |
| `bandwidth_sweep_coupled_wide.csv` | derived from `bandwidth_sweep_coupled.csv` |
| `bandwidth_sweep_m5dB_wide.csv` | derived from `bandwidth_sweep_m5dB.csv` |
| `bandwidth_sweep_p10dB_wide.csv` | derived from `bandwidth_sweep_p10dB.csv` |
