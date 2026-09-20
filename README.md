# ReconUNet

Code, learned baselines, trained-model recipes and result tables for

> **ReconUNet: Learning to Reconstruct Clean Covariance Matrix under Low SNR, Multipath, and Array Imperfections**
> Itamar Almog and Anthony J. Weiss, School of Electrical Engineering, Tel Aviv University.
> Under review at *MDPI Sensors* (manuscript sensors-4536109).

ReconUNet is a data-driven front-end for direction-of-arrival (DoA) estimation. It ingests a stack of
τ = 8 multi-lag autocorrelation matrices of an N = 8 uniform linear array and reconstructs a clean,
mismatch-compensated covariance matrix together with an explicit eigendecomposition. Unmodified classical
estimators (MUSIC, Root-MUSIC, ESPRIT, Unitary-ESPRIT, MVDR, Bartlett) then run on the repaired statistics.
The model is trained on ~2 × 10⁶ synthetic paired scenes covering low SNR, coherent multipath and
gain/phase/coupling/position array errors.

This repository also contains the three learned baselines used in the revision, SubspaceNet, DA-MUSIC and
SubViT (DOA-ViT), all trained on the *same* corpus with the *same* optimiser schedule, and every number
behind the revision's tables as tracked CSV files.

---

## Contents

1. [Repository layout](#repository-layout)
2. [Installation](#installation)
3. [Generating the data](#generating-the-data)
4. [Training each model](#training-each-model)
5. [Evaluating and reproducing every table and figure](#evaluating-and-reproducing-every-table-and-figure)
6. [Trained checkpoints](#trained-checkpoints)
7. [Headline results](#headline-results)
8. [A note on the renderer fix](#a-note-on-the-renderer-fix)
9. [Tests](#tests)
10. [Citing](#citing)
11. [License](#license)

---

## Repository layout

```
configs/
  data/          scene-manifest configs: paper_corpus.yaml (training), paper_corpus_eval.yaml,
                 scenarios/{basic,moderate,advanced1_ood,advanced2_crowded}.yaml (mild errors),
                 scenarios_harsh/… (5× the error magnitudes)
  train/         *_paper.yaml: the four paper-faithful training recipes (+ smoke variants)
  eval/          configs for the generic reconunet-evaluate CLI
src/reconunet/
  data/          SceneManifest (compact .npy scene tables), SceneRenderer (lazy, deterministic
                 rendering of snapshots, imperfections, coherent multipath), collates per model
  models/        deep_learning/EVDUNet.py = ReconUNet; classical estimators; third_party/ adapters
                 for SubspaceNet, DA-MUSIC and SubViT (upstream code is vendored as git submodules)
  cli/           reconunet-generate, reconunet-train, reconunet-evaluate(-scenarios)
  evaluation/    batched classical estimators, stochastic CRLB, unified harness
scripts/analysis/
  compare_paper_testset.py    per-K and pooled RMSE of all methods on the paper test split
  scenario_sweep.py           RMSE vs SNR across the four §IV scenarios (+ CRLB) → CSV + figure
  reproduce_table2.py         Table II: pooled RMSE per scenario/SNR for classical, aided and learned
  render_results_from_csv.py  re-derives every table and figure from the tracked CSVs (no GPU)
scripts/run_revision_retrain_sequence.sh   trains all seven models in sequence on one GPU
experiments/runs/eval_coherent_20260910/   TRACKED results: CSVs, sweep figure, standalone HTML report
experiments/runs/legacy/checkpoints/       TRACKED: the EVD-UNet weights behind the submitted manuscript
experiments/paper_figures/                 result JSONs and plots of the submitted manuscript
docs/PAPER_FAITHFUL_DATASET.md             what the corpus reproduces exactly, and its bug-fix log
docs/PAPER_FAITHFUL_TRAINING.md            hyper-parameter audit against the paper, and its bug-fix log
third_party/                               git submodules (SubspaceNet, DOA_est_Master) + patches/
tests/unit/                                pytest suite
```

## Installation

Tested with Python 3.12, PyTorch 2.11 (CUDA 13.0) and NumPy 2.4 on Ubuntu; the package declares
`python >= 3.9`. A CUDA GPU is needed for training; evaluation of the CSVs needs none.

```bash
git clone --recurse-submodules https://github.com/itamaralmog96/ReconUNet-DoA.git
cd ReconUNet-DoA

python3 -m venv .venv && source .venv/bin/activate
# Install the PyTorch build matching your CUDA driver first, e.g.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
pip install -e ".[dev]"          # installs reconunet + the reconunet-* CLIs + pytest

python -c "import torch; print(torch.cuda.is_available())"
```

**Submodules.** The baselines import upstream code vendored as submodules:

| Path | Upstream | Pinned at |
|---|---|---|
| `third_party/subspacenet` | [ShlezingerLab/SubspaceNet](https://github.com/ShlezingerLab/SubspaceNet) | upstream `f2ef464` + one line: [`third_party/patches/0001-…cpu-before-numpy.patch`](third_party/patches/) |
| `third_party/doa_est_master` | [zzb-nice/DOA_est_Master](https://github.com/zzb-nice/DOA_est_Master) (DOA-ViT / SubViT) | `4116a85` |

`.gitmodules` points the SubspaceNet submodule at a fork that carries the patched commit. If that fork
is unavailable, reproduce it from upstream:

```bash
git clone https://github.com/ShlezingerLab/SubspaceNet.git third_party/subspacenet
git -C third_party/subspacenet checkout f2ef464
git -C third_party/subspacenet am ../patches/0001-subspacenet-move-covariance-to-cpu-before-numpy.patch
```

The patch only moves a CUDA tensor to the CPU before a NumPy conversion; DA-MUSIC and SubspaceNet model
code is used **unmodified** (deviations in the forward pass are documented in the adapters' docstrings).

## Generating the data

Scenes are stored as compact manifests (~96 bytes per scene). Snapshots, array imperfections and
multipath replicas are rendered lazily and deterministically from the manifest at load time, so the
whole 2 M-scene corpus is ~190 MB on disk and generates in about two minutes.

```bash
# Training corpus + val/test splits (paper §IV): 2,000,000 / 50,000 / 50,000 scenes
reconunet-generate --config configs/data/paper_corpus.yaml

# The four §IV stress scenarios, mild and harsh imperfections
# (1000 fixed angle configurations × 9 SNR levels from -20 to 20 dB each)
for s in basic moderate advanced1_ood advanced2_crowded; do
  reconunet-generate --config configs/data/scenarios/$s.yaml
  reconunet-generate --config configs/data/scenarios_harsh/$s.yaml
done
```

Key corpus parameters (all in `configs/data/paper_corpus.yaml`): N = 8 ULA at λ/2, 2.45 GHz narrowband,
T = 512 snapshots, τ = 8 lags, K ∈ {1, 2, 3, 4} direct sources with ≥ 10° separation in a ±60° sector,
0 to 3 coherent multipath replicas with sub-sample delays, training SNR ∈ [−10, 10] dB, "mild"
imperfections (0.1 dB gain, 1° phase, 0.02 coupling, 1 % position error), sources band-limited to
0.05 fs so replicas are coherent (|γ| ≈ 0.9). Master seed 20260420; changing it invalidates every
checkpoint. `docs/PAPER_FAITHFUL_DATASET.md` lists what is exact and what is approximated.

## Training each model

All four models share one recipe from the paper: Adam, LR 1e-4, weight decay 1e-5, batch 2048, up to 300
epochs, ReduceLROnPlateau (factor 0.7, patience 5), early stopping with patience 25 on validation loss,
gradient-norm clipping at 1.0, seed 20260420.

```bash
reconunet-train --config configs/train/reconunet_paper.yaml       # ReconUNet (EVD-UNet)
reconunet-train --config configs/train/subspacenet_paper.yaml     # SubspaceNet + Root-MUSIC head
reconunet-train --config configs/train/subvit_paper.yaml          # SubViT, published capacity (D=768, 6 layers, 12 heads)
for K in 1 2 3 4; do                                              # DA-MUSIC: one fixed-head model per K
  reconunet-train --config configs/train/damusic_paper_k$K.yaml   #   (data.k_filter selects the K subset)
done

# …or everything in sequence, detached, with a status log (this is how the released models were made):
setsid nohup bash scripts/run_revision_retrain_sequence.sh > /dev/null 2>&1 &
tail -f experiments/runs/revision_retrain_20260906/status.log
```

Each run writes `experiments/runs/<model>_paper/checkpoints/{best,last}.pt`, `history.json` and
TensorBoard events under `tb/`. `best.pt` is selected on validation loss. Run the models one at a time:
the eigendecomposition heads of SubspaceNet and DA-MUSIC are starved badly by any concurrent GPU job.

Measured on a single NVIDIA RTX 2000 Ada (16 GB), 8 data-loader workers:

| Model | Params | Epochs run (best) | Wall-clock | Val RMSPE at best |
|---|---:|---:|---:|---:|
| ReconUNet | 0.34 M | 109 (84) | 4.8 h | 2.02° |
| DA-MUSIC K=1 / 2 / 3 / 4 | 0.015 M each | 207 / 300 / 300 / 300 | 2.6 / 3.8 / 3.8 / 3.8 h | 8.24 / 5.13 / 4.60 / 4.13° |
| SubspaceNet | 0.042 M | 83 (58) | 30.3 h | 4.04° |
| SubViT | 42.6 M | 161 (136) | 43.9 h | 6.45° |

The DA-MUSIC K ≥ 2 runs reached the 300-epoch cap without triggering early stopping.

## Evaluating and reproducing every table and figure

### From the tracked CSVs (no GPU, no checkpoints, no scenes)

Everything reported for the revision is derived from four CSV files under
`experiments/runs/eval_coherent_20260910/`, which are tracked in git:

| File | Written by | Contents |
|---|---|---|
| `table2_mild/table2_full.csv` | `reproduce_table2.py` | pooled RMSE (deg, paper eq. 31) per scenario × SNR × method, mild imperfections; long form |
| `table2_harsh/table2_full.csv` | `reproduce_table2.py --scenarios-root data/scenes/scenarios_harsh` | same, harsh imperfections |
| `scenario_sweep/scenario_sweep.csv` | `scenario_sweep.py` | mean direct-path RMSE vs SNR per scenario for Root-MUSIC, SubspaceNet, SubViT, DA-MUSIC, ReconUNet, plus the stochastic CRLB |
| `paper_testset/paper_testset_by_K.csv` | `compare_paper_testset.py` | pooled RMSE and median per-scene RMSPE per method for K = 1..4 and all K on the paper test split (3000 scenes per K) |

```bash
python scripts/analysis/render_results_from_csv.py            # → experiments/runs/eval_coherent_20260910/derived/
```

produces, from the CSVs alone:

* **Table II** (0 dB view, mild and harsh): `derived/table2_{mild,harsh}_0dB.md`; pass `--snr 5` etc. for other levels.
  Full RMSE-vs-SNR tables per scenario: `derived/table2_<preset>_vs_snr_<scenario>.md`.
* **RMSE-vs-SNR figure** (2 × 2 grid of the four scenarios with the CRLB): `derived/scenario_sweep_grid.png`.
  The figure written at evaluation time is tracked as `scenario_sweep/scenario_sweep_grid.png`.
* **Per-K test-split table** (pooled RMSE and medians): `derived/paper_testset_by_K.md`.

The standalone report `experiments/runs/eval_coherent_20260910/reconunet_revision_results.html` renders
the same numbers with the notes for the manuscript; open it locally in a browser.

### Re-running the evaluation

With the data generated and checkpoints in place (see [Trained checkpoints](#trained-checkpoints)):

```bash
E=experiments/runs/eval_coherent_20260910
python scripts/analysis/compare_paper_testset.py -o $E/paper_testset
python scripts/analysis/scenario_sweep.py        -o $E/scenario_sweep
python scripts/analysis/reproduce_table2.py      -o $E/table2_mild
python scripts/analysis/reproduce_table2.py      -o $E/table2_harsh --scenarios-root data/scenes/scenarios_harsh
# or: bash experiments/runs/eval_coherent_20260910/run_evals.sh
```

Each script takes `--reconunet/--subspacenet/--subvit <best.pt>` and `--damusic-dir <dir with k1..k4/>`
to point at other checkpoints; DA-MUSIC is skipped with a warning when its per-K checkpoints are absent.
The four scripts together take about six minutes on the RTX 2000 Ada.

### The submitted manuscript's figures

The submitted manuscript was produced with the project's original signal generator, before this
rewrite. Its result JSONs and plots are kept under `experiments/paper_figures/evaluation/`, and the
weights behind them are tracked in git at
`experiments/runs/legacy/checkpoints/evd_unet_denoising_model_20250929_015132.pth`.
`scripts/analysis/run_legacy_paper_table.py` re-runs the legacy per-scenario evaluation against a
checkpoint and writes a `summary_0dB.csv` in the Table II layout.

## Trained checkpoints

Checkpoints are excluded from git (`*.pt`, `*.pth`) except the legacy EVD-UNet above. The revision
models are published as follows; download them to the paths the evaluation scripts expect by default.

| Model | File to place at | Size | Where |
|---|---|---:|---|
| ReconUNet | `experiments/runs/reconunet_paper/checkpoints/best.pt` | 3.9 MB | GitHub Release `revision-r1-checkpoints` |
| SubspaceNet | `experiments/runs/subspacenet_paper/checkpoints/best.pt` | 0.5 MB | GitHub Release `revision-r1-checkpoints` |
| DA-MUSIC, K = 1..4 | `experiments/runs/damusic_paper/k<K>/checkpoints/best.pt` | 4 × 0.18 MB | GitHub Release `revision-r1-checkpoints` |
| SubViT (42.6 M params) | `experiments/runs/subvit_paper/checkpoints/best.pt` | 512 MB | Zenodo (DOI to be added on release; too large for a GitHub Release asset without LFS) |

Each `best.pt` is a dict with `model` (state dict), `cfg` (the full training YAML), `epoch`,
`val_loss` and `best_val`; the adapters and evaluation scripts rebuild the architecture from `cfg`.

## Headline results

Pooled RMSE in degrees at 0 dB, mild imperfections, from `table2_mild/table2_full.csv`
(bounds excluded from the bold-best marking; full tables via `render_results_from_csv.py`):

| Method | Basic (K=1) | Moderate (K=2 +1) | OOD (K=1 +6) | Crowded (K=4 +3) |
|---|---:|---:|---:|---:|
| Root-MUSIC | **0.25** | 5.91 | 40.95 | 8.87 |
| ESPRIT | 0.33 | 5.94 | 35.72 | 8.14 |
| Unitary-ESPRIT | 0.41 | 5.69 | 24.30 | 8.04 |
| ReconUNet + Root-MUSIC | 0.37 | 1.79 | 22.89 | 4.80 |
| ReconUNet + ESPRIT | 0.40 | **1.78** | 24.20 | 4.61 |
| ReconUNet + Unitary-ESPRIT | 0.41 | 2.63 | **21.33** | **4.00** |
| SubspaceNet | 0.54 | 2.92 | 21.49 | 7.91 |
| SubViT | 0.38 | 8.72 | 23.60 | 16.32 |
| DA-MUSIC | 2.13 | 3.71 | 24.28 | 6.82 |
| CRLB | 0.12 | 0.14 | 0.12 | 0.21 |

Paper test split, all K pooled (12,000 scenes, SNR ∈ [−10, 10] dB): ReconUNet 5.79°, DA-MUSIC 8.04°,
SubspaceNet 8.37°, ESPRIT 11.00°, Root-MUSIC 12.43°, SubViT 14.95°; ReconUNet is best in every K bucket.
SubViT has the sharpest median (0.58°) and the heaviest tail. The OOD scenario (seven coherent arrivals
on eight sensors) is a degrees-of-freedom limit where no method recovers.

## A note on the renderer fix

In July 2026 an audit of this rewritten code base found that its scene renderer produced multipath
replicas that were *not* coherent with the direct path (white sources, so a one-sample delay already
decorrelated the copy). The original signal generator used for the submitted manuscript band-limits
every source to 10 % of Nyquist and never had this problem; **no result in the submitted manuscript was
affected**. The fix (2026-09-06) restores band-limited sources and exact fractional delays, after which
direct/replica correlation is |γ| ≈ 0.9 as the paper's Section II-B derives. All learned models were
retrained on the corrected renderer and every number in this repository comes from those retrains. The
full account, with measurements, is the 2026-09-06 entry in `docs/PAPER_FAITHFUL_DATASET.md`; the
discarded runs are kept under `experiments/runs/archive_incoherent_renderer_20260906/` locally and are
not part of the release.

## Tests

```bash
pytest tests/unit -q
```

`tests/unit/test_scene_pipeline.py` includes regression tests for source band-limiting and replica
coherence; `tests/unit/test_damusic_adapter.py` covers the DA-MUSIC adapter. One pre-existing test
(`test_collates_agree_on_labels`) compares padded label tensors with `torch.equal`, which treats NaN
padding as unequal; it fails independently of any change here and is documented in the dataset notes.

## Citing

See [`CITATION.cff`](CITATION.cff) (GitHub's "Cite this repository" button). Until the article is
published, please cite the manuscript as

```bibtex
@article{almog2026reconunet,
  author  = {Almog, Itamar and Weiss, Anthony J.},
  title   = {ReconUNet: Learning to Reconstruct Clean Covariance Matrix under Low {SNR}, Multipath, and Array Imperfections},
  journal = {Sensors},
  year    = {2026},
  note    = {Manuscript sensors-4536109, under review}
}
```

## License

MIT, see [`LICENSE`](LICENSE). The vendored SubspaceNet and DOA_est_Master trees keep their own
upstream licenses.
