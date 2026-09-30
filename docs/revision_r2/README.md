# Revision round 2 (2026-09-30) — result CSVs

Copied by `scripts/analysis/r2_report.py export` at the end of `scripts/run_revision_r2_pipeline.sh`. Same column schema as the R1 CSVs; the covariance-only network is the method `ReconUNet-C` (Table II/III: `ReconUNet-C+<back end>`), next to the full `ReconUNet`. DA-MUSIC = v2 ensemble (`experiments/runs/damusic_paper/ensemble_v2`). CI files: percentile bootstrap over scenes, 1000 resamples, seed 20260920.

| file | source |
|---|---|
| `paper_testset_by_K.csv` | `experiments/runs/eval_r2_20260930/paper_testset_r2/paper_testset_by_K.csv` |
| `paper_testset_ci.csv` | `experiments/runs/eval_r2_20260930/paper_testset_r2/paper_testset_r2_ci.csv` |
| `scenario_sweep.csv` | `experiments/runs/eval_r2_20260930/scenario_sweep_r2/scenario_sweep.csv` |
| `scenario_sweep_ci.csv` | `experiments/runs/eval_r2_20260930/scenario_sweep_r2/scenario_sweep_r2_ci.csv` |
| `scenario_sweep_grid.png` | `experiments/runs/eval_r2_20260930/scenario_sweep_r2/scenario_sweep_grid.png` |
| `table2_mild_full.csv` | `experiments/runs/eval_r2_20260930/table2_mild_r2/table2_full.csv` |
| `table2_mild_ci.csv` | `experiments/runs/eval_r2_20260930/table2_mild_r2/table2_mild_r2_ci.csv` |
| `table2_harsh_full.csv` | `experiments/runs/eval_r2_20260930/table2_harsh_r2/table2_full.csv` |
| `table2_harsh_ci.csv` | `experiments/runs/eval_r2_20260930/table2_harsh_r2/table2_harsh_r2_ci.csv` |
| `snapshot_sweep.csv` | `experiments/runs/sweeps_r2_20260930/snapshot_sweep.csv` |
| `separation_sweep.csv` | `experiments/runs/sweeps_r2_20260930/separation_sweep.csv` |
| `music_verification.csv` | `experiments/runs/sweeps_r2_20260930/music_verification.csv` |
| `cost_summary.csv` | `experiments/runs/sweeps_r2_20260930/cost_summary_r2.csv` |
| `cost_latency.csv` | `experiments/runs/sweeps_r2_20260930/cost_latency_r2.csv` |
| `bandwidth_sweep.csv` | `experiments/runs/sweeps_r2_20260930/bandwidth_sweep.csv` |
| `bandwidth_sweep_coupled.csv` | `experiments/runs/sweeps_r2_20260930/bandwidth_sweep_coupled.csv` |
| `bandwidth_autocorr.csv` | `experiments/runs/sweeps_r2_20260930/bandwidth_autocorr.csv` |
| `bandwidth_autocorr_coupled.csv` | `experiments/runs/sweeps_r2_20260930/bandwidth_autocorr_coupled.csv` |
| `reconunet_c_training.csv` | `experiments/runs/revision_r2_20260930/reconunet_c_training.csv` |
| `table2_mild_reconunet_backends_by_snr.csv` | derived from `table2_mild_full.csv` |
| `table2_harsh_reconunet_backends_by_snr.csv` | derived from `table2_harsh_full.csv` |
| `bandwidth_sweep_wide.csv` | derived from `bandwidth_sweep.csv` |
| `bandwidth_sweep_coupled_wide.csv` | derived from `bandwidth_sweep_coupled.csv` |
