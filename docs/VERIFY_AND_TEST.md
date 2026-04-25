# Verify & test the data-generation pipeline

**Goal.** Before spending a single GPU-day training ReconUNet / SubspaceNet /
SubViT on a corpus, be sure the corpus is what you *think* it is:

* the right number of scenes in train / val / test,
* the right distribution of K (sources), SNR, angles,
* the array-imperfection preset you selected actually applied,
* the rendered snapshots and covariances actually match a ULA steering model,
* a well-understood *classical* estimator (MUSIC, Root-MUSIC, ESPRIT,
  Unitary-ESPRIT, MVDR, Beamformer) can solve the easy scenes.

That last point is the one most worth leaning on.  If MUSIC with the
true K, mild imperfections, and 20 dB SNR cannot recover the angles to
within a fraction of a degree, your data-generation pipeline is buggy —
no amount of training will save you.

This doc assumes the layout you already have on the GPU box:

```
ReconUNet/
├── src/reconunet/
│   ├── cli/                    # reconunet-generate, -train, -evaluate
│   ├── data/
│   │   ├── scene_manifest.py   # 96-byte fixed-width scene records
│   │   └── scene_renderer.py   # X = A @ s + n, plane-wave ULA
│   ├── signalgen/
│   │   ├── array_processing.py # legacy ArrayModel (used by classic DOA)
│   │   └── signal_generator.py # legacy generator (multipath lives here)
│   └── models/classic/         # MUSIC, RootMUSIC, ESPRIT, etc.
├── scripts/verify/             # <-- the new helper scripts this doc uses
├── configs/data/verify/        # <-- targeted verification configs
└── docs/VERIFY_AND_TEST.md     # <-- you are here
```

---

## 0. What the new SceneRenderer does and does *not* simulate

It helps to know the forward model up front so you don't spend time
debugging a feature that isn't implemented.

**Supported today (`reconunet.data.scene_renderer`)**

| Knob | Where it lives | Values |
| --- | --- | --- |
| Array geometry | `ManifestMeta.array_type` | `ULA` (only ULA is actually used; URA / TRIANGULAR are reserved) |
| Number of elements `M` | `ManifestMeta.M` | integer, paper uses `8` |
| Snapshots per scene `T` | `ManifestMeta.T` | integer, paper uses `512` |
| Sampling rate | `ManifestMeta.fs_Hz` | Hz |
| Decorrelation lag `tau` | `ManifestMeta.tau` | integer, paper uses `8` |
| Element spacing | `ManifestMeta.element_spacing_lambda` | `0.5` by default |
| Angle range | `ManifestMeta.angle_range_deg` | e.g. `[-60, 60]` |
| SNR range | `ManifestMeta.snr_range_db` | e.g. `[-10, 20]` |
| K (sources) choices | config `sampling.k_choices` | e.g. `[1, 2, 3]` |
| Min. source separation | config `sampling.min_separation_deg` | e.g. `3.0` |
| Array imperfections preset | config `sampling.array_errors` | `"none"`, `"mild"`, `"harsh"` |
| Modulation | `Scene.modulation` (default `NARROWBAND`) | `NARROWBAND`, `WIDEBAND`, `BPSK`, `QPSK` |

Each `Scene` row is 96 bytes fixed-width.  The imperfection preset is
materialised per-scene as four scalars:

| Preset | `gain_err_dB` | `phase_err_deg` | `mutual_coupling` | `position_err_pct` |
| --- | --- | --- | --- | --- |
| `none`  | 0.0 | 0.0 | 0.00 | 0.0 |
| `mild`  | 0.1 | 1.0 | 0.02 | 1.0 |
| `harsh` | 0.5 | 5.0 | 0.10 | 5.0 |

**Multipath (v1.1+):** supported as a legacy-faithful port of
`signal_generator.add_multipath`.  Each scene can carry `num_multipath`
additional scatterers with random AoAs, delays, and complex gains — see
§3.5 for the full contract.  Gated on `Scene.has_multipath`; the default
for old manifests is `False`, so non-multipath behaviour is byte-
identical to v1.0.

**Not supported in the new pipeline:**

* **URA / triangular / circular arrays.** The enum exists but the
  forward model is ULA-only today.
* **Frequency-domain (truly wideband) modelling.** `WIDEBAND` draws
  complex-Gaussian samples exactly like `NARROWBAND`; the "wideband"
  label is symbolic for now.
* **Near-field / curved wavefront.** Plane-wave only.

Keep this table in mind when you design the test matrix below.

---

## 1. Where each verification artefact lives

Everything you need is in three places:

```
configs/data/verify/             # YAML configs for targeted generations
  ├── tiny.yaml                  # 1k/200/200 smoke-test (fast)
  ├── snr_sweep.yaml             # widen SNR range to probe SNR floor
  ├── k_sweep.yaml               # K ∈ {1,2,3,4,5}
  ├── errors_none.yaml           # perfectly calibrated array
  ├── errors_mild.yaml           # paper default
  ├── errors_harsh.yaml          # stress test
  └── modulation_qpsk.yaml       # QPSK signals

scripts/verify/
  ├── inspect_manifest.py        # summarise + histogram a .npy manifest
  ├── visualize_scene.py         # render ONE scene + plot cov / spectra
  └── run_classic_on_manifest.py # sweep classic DOA algorithms

docs/VERIFY_AND_TEST.md          # this file
```

Each script is self-contained and uses only `numpy`, `matplotlib`,
`tqdm`, and the in-tree reconunet package.  No GPU required.

---

## 2. The 30-second sanity loop

Before anything fancy, run this.  It takes about a minute on a laptop
and catches 90 % of mistakes.

```bash
cd ReconUNet
source .venv/bin/activate          # or: conda activate reconunet

# (a) generate a tiny corpus
reconunet-generate -c configs/data/verify/tiny.yaml --overwrite

# (b) summarise its distributions
python scripts/verify/inspect_manifest.py data/scenes/verify/tiny/train.npy

# (c) render one scene and plot its covariance + MUSIC spectrum
python scripts/verify/visualize_scene.py data/scenes/verify/tiny/train.npy \
    --index 0 --out experiments/verify/scene_0.png

# (d) run classical MUSIC on 100 scenes, report RMSPE
python scripts/verify/run_classic_on_manifest.py \
    data/scenes/verify/tiny/val.npy --algo music --n 100 --true-k
```

Expected signals that the pipeline is healthy:

* `inspect_manifest` shows K spread across your `k_choices`, SNRs
  spread roughly uniformly over `snr_range_db`, angles inside
  `angle_range_deg`, and (very important) seeds are *all unique* across
  train/val/test.
* `visualize_scene` shows a covariance matrix whose top-left block is
  dominated by the source signal (Hermitian, positive-semidef), and a
  MUSIC spectrum with sharp peaks at the ground-truth angles.
* `run_classic_on_manifest` with `--algo music --true-k` at 20 dB,
  mild imperfections, 3 sources with `min_separation_deg=3`, reports
  RMSPE well under 1°.  If it can't, your data is broken.

---

## 3. Per-mode test recipes

Each recipe = `(config → generate → inspect → visualise → classic)`.
Run these in the order listed; each builds on the last.

### 3.1 Source-count / interference sweep

**What it exercises.** The renderer's support for multiple incoherent
sources; the SceneManifest's sampling of K from `k_choices`; the
classical algorithms' ability to resolve close sources.

```bash
reconunet-generate -c configs/data/verify/k_sweep.yaml --overwrite
python scripts/verify/inspect_manifest.py data/scenes/verify/k_sweep/test.npy
python scripts/verify/run_classic_on_manifest.py \
    data/scenes/verify/k_sweep/test.npy \
    --algo music --n 500 --true-k \
    --group-by K --out experiments/verify/k_sweep_music.csv
```

You should see RMSPE grow monotonically with K (degrees of freedom
limit: MUSIC on an M-element array can resolve at most M-1 sources, so
for M=8 we expect noticeable degradation at K≥5).

If you *don't* see the expected degradation, the most common cause is
a mis-set `min_separation_deg` — e.g. the generator is refusing to
sample K=5 scenes with adequate separation given `angle_range_deg=
[-60, 60]`, and silently falling back to lower K.  Check the K
histogram.

### 3.2 SNR sweep

**What it exercises.** The additive-noise term in `X = A @ s + n`, and
how the classical algorithms' detection threshold behaves.

```bash
reconunet-generate -c configs/data/verify/snr_sweep.yaml --overwrite
python scripts/verify/run_classic_on_manifest.py \
    data/scenes/verify/snr_sweep/test.npy \
    --algo music,rootmusic,esprit,mvdr \
    --n 2000 --true-k \
    --group-by snr_db --out experiments/verify/snr_sweep.csv
```

Expected shape: RMSPE drops roughly as `1/√SNR` once you clear the
SNR threshold (often near 0 dB for MUSIC on an 8-element ULA with
T=512, K=3).  Below threshold, estimates degrade rapidly and the
curves go flat around the σ of a uniform prior over the angle range.

### 3.3 Array-imperfection sweep

**What it exercises.** The `array_errors` preset, in particular how
much the classical algorithms' performance floor is lifted.

```bash
for preset in none mild harsh; do
  reconunet-generate -c configs/data/verify/errors_${preset}.yaml --overwrite
  python scripts/verify/run_classic_on_manifest.py \
      data/scenes/verify/errors_${preset}/test.npy \
      --algo music --n 1000 --true-k \
      --group-by snr_db --out experiments/verify/errors_${preset}.csv
done
```

Expected: at high SNR the `none` curve keeps dropping, but `mild`
asymptotes around ~0.2° and `harsh` flattens near 1–2° — classical
MUSIC is model-based and is hurt by imperfections exactly as you'd
expect.  This is the whole motivation for ReconUNet; if the preset
doesn't change this asymptote, the imperfection fields are not being
applied to the steering matrix.

### 3.4 Modulation sweep

**What it exercises.** The `modulation` field in the `Scene` record
and whether the renderer samples source symbols from the right
distribution.

```bash
reconunet-generate -c configs/data/verify/modulation_qpsk.yaml --overwrite
python scripts/verify/visualize_scene.py \
    data/scenes/verify/modulation_qpsk/train.npy --index 0 \
    --out experiments/verify/qpsk_scene.png --show-constellation
```

For `BPSK` the source-signal IQ plot should show two tight clusters at
±1.  For `QPSK`, four tight clusters at `±1±j`.  For `NARROWBAND`, a
fuzzy circular cloud (complex Gaussian).  If the constellation for
BPSK looks Gaussian, the modulation enum is being ignored somewhere.

### 3.5 Multipath sweep (legacy-faithful port)

**What it exercises.**  The v1.1 multipath block inside
`SceneRenderer.render()` (`_render_with_multipath`), a port of the
legacy `signalgen/signal_generator.py::add_multipath`.  For each scene
with `has_multipath=True`, the renderer appends `num_multipath`
additional plane-wave components to the direct-path sources; each has

* **AoA**  `U[0, 2π)` rad
* **delay τ**  `U(0, 1/BW)` (uniform) or `Exp((1/BW)/2)` (exponential),
   where `BW = fs × 0.05` mirrors the legacy default
* **dB-loss**  uniform:  `10·(τ/τ_max) + U(1, 2)`;
                exponential: `10·(1 − e^{-τ/τ_max}) + Exp(2.5)`
* **phase**  `N(π, π/4)`
* **complex gain**  `10^{-dB/20} · √P · exp[j(φ − 2π f_c τ)]`
* **sample-shift**  `N_k = int(τ · fs · factor/2)`, source 0 is sliced
  to produce the delayed copy

These formulas are line-for-line the legacy ones (lines 308–322 and 332
of `signal_generator.py`).

```bash
reconunet-generate -c configs/data/verify/multipath.yaml --overwrite
python scripts/verify/inspect_manifest.py \
    data/scenes/verify/multipath/test.npy
python scripts/verify/visualize_scene.py \
    data/scenes/verify/multipath/test.npy --index 0 \
    --out experiments/verify/multipath_scene.png
python scripts/verify/run_classic_on_manifest.py \
    data/scenes/verify/multipath/test.npy \
    --algo music --n 500 --fixed-k 3 \
    --group-by snr_db --out experiments/verify/multipath.csv
```

**How to read the results.**

* `inspect_manifest` will show `has_multipath ≈ 1` on every row and
  `num_multipath` spread across `[0, max_paths − K]`.  For
  `multipath.yaml` with `K=2, max_paths=3`, that's 0 or 1 per scene.
* `visualize_scene` MUSIC spectrum shows extra peaks at the multipath
  AoAs.  Because the legacy draws AoAs uniformly in azimuth (full
  circle, not bounded by `angle_range_deg`), they can land outside
  the direct-path range — that's faithful to the legacy, not a bug.
* `run_classic_on_manifest --fixed-k 3` tells MUSIC to expect three
  sources, so it will lock onto two direct + one multipath peak.  With
  `--true-k` (which uses `scene.K=2`), it will treat the multipath as
  an unmodelled interferer and RMSPE on the direct sources rises.
  Compare against `errors_mild.yaml` (same imperfections, no
  multipath) to quantify the multipath penalty for classical MUSIC —
  that's your classical baseline for the ReconUNet-vs-multipath story.

**Config knobs** (under `sampling:` in the YAML):

| key | default | maps to legacy |
| --- | --- | --- |
| `enable_multipath` | `false` | `SignalConfig.enable_multipath` |
| `max_paths` | `3` | `add_multipath(max_paths=...)` |
| `num_multipath_components` | `null` | `add_multipath(num_multipath_components=...)` |
| `multipath_distribution` | `uniform` | `add_multipath(multipath_distribution=...)` |
| `mp_max_delay_factor` | `10.0` | `add_multipath(max_delay_factor=...)` |

**Caveats — be honest about the differences.**

* The legacy uses bandlimited-white-noise source signals (rectangular
  filter in the frequency domain).  The v1.1 port uses an
  FFT-bandlimited-to-Nyquist approximation for the multipath branch
  specifically; the direct-source path is still iid complex Gaussian
  (unchanged).  For narrow-band tests the difference is negligible,
  but for a strictly-bandlimited scenario the legacy and the new
  renderer diverge in the sample-level autocorrelation of the
  delayed copies.  Non-multipath scenes render byte-identically to
  v1.0.
* The legacy multipath AoA draw is `U[0, 2π)` in azimuth with
  endfire-0 steering.  The new renderer uses broadside-0 with
  `sin(θ)` steering — statistically equivalent on the ULA manifold
  (same distribution of projected sines), but individual angles are
  not bitwise-identical to a legacy run with the same seed.
* Direct-path SNR is preserved: noise variance is set against the
  unit-power direct sources, not against the combined direct +
  reflection power, so `scene.snr_db` keeps the same meaning with or
  without multipath — just like the legacy's `add_noise` (which also
  computes `signal_power` on direct signals only).

---

## 4. Running the classical DOA models (`reconunet.models.classic`)

The classical models (`MUSIC`, `RootMUSIC`, `ESPRIT`, `UnitaryESPRIT`,
`MVDR`, `Beamformer`) all inherit from `BaseDOAModel` and share one
interface:

```python
from reconunet.signalgen.array_processing import ArrayConfig, ArrayModel
from reconunet.models.classic import MUSIC, RootMUSIC, ESPRIT

# 1. Build an ArrayModel that matches the manifest's ULA geometry.
cfg = ArrayConfig(
    array_type       = "linear",
    num_elements     = meta.M,
    carrier_freq     = 2.45e9,        # anything consistent with meta.fs_Hz
    element_spacing  = meta.element_spacing_lambda,
    enable_gain_phase_errors = False, # we feed imperfect data directly,
    enable_mutual_coupling   = False, # the steering matrix should be *nominal*
    position_error_std = 0.0,
)
array = ArrayModel(cfg, seed=0)

# 2. Render a scene to get snapshots X ∈ C^{M×T} and covariance R ∈ C^{M×M}.
#    The renderer seeds itself from scene.seed for reproducibility.
from reconunet.data.scene_renderer import SceneRenderer
renderer = SceneRenderer(meta=meta)
res = renderer.render(scene)
X, R = res.snapshots, res.covariance

# 3. Feed into a classic model.
model = MUSIC(array_model=array, num_sources=int(scene.K))  # or None to auto-estimate
model.set_received_data(X)           # or: model.set_received_covariance(R)
angles_deg, spectrum = model.estimate_doa()
```

**Three things worth calling out:**

1. **Nominal vs. calibrated steering matrix.** The classic models need
   the *nominal* steering matrix — the one the system *thinks* it has
   — to expose the penalty of mis-calibration.  That's why the
   snippet above disables imperfections on the `ArrayModel` even
   though the rendered snapshots include them.  If you flip that, your
   classical estimator is implicitly calibrated and the whole point of
   the imperfection sweep is lost.

2. **Angle convention.** `BaseDOAModel.scan_angles_deg` defaults to
   `np.arange(0, 360)`.  The manifest stores angles in radians with
   broadside = 0° (range `[-60°, 60°]` for the paper).  Convert with
   ```python
   theta_mani_deg = np.rad2deg(scene.angles_rad[:scene.K])
   theta_classic_deg = theta_mani_deg + 90.0   # or use a symmetric scan
   ```
   `scripts/verify/run_classic_on_manifest.py` does this for you.

3. **True K vs. auto-estimated K.** When you pass `num_sources=None`
   the model uses MDL / AIC / eigen-ratio heuristics to guess K.
   During data-quality verification you want to compare against ground
   truth, so pass the manifest's `scene.K` explicitly.  The
   `--true-k` flag in `run_classic_on_manifest.py` does this.

### 4.1 Recommended comparison grid

| Algorithm | Needs K? | Notes |
| --- | --- | --- |
| `Beamformer` | no | Conventional, low resolution, high sidelobes |
| `MVDR` | no | Capon, higher resolution, needs invertible R |
| `MUSIC` | yes | Subspace, the standard workhorse |
| `RootMUSIC` | yes | ULA-only, closed-form, fastest |
| `ESPRIT` | yes | ULA-only, very low-variance at high SNR |
| `UnitaryESPRIT` | yes | Real-valued ESPRIT, 2× cheaper |

For the SNR sweep and imperfection sweep, report Root-MUSIC as the
strongest classical baseline.  For the K sweep and the source-
coherence tests, report MUSIC (it's the model most affected by both).

### 4.2 Comparing classic vs. deep-learning models in one plot

The unified harness already supports this.  In your eval YAML,
declare classical baselines alongside ReconUNet / SubspaceNet / SubViT
and mark them `is_classic: true` so the harness knows to skip
checkpoint loading:

```yaml
models:
  - name: music_truek
    adapter_class: reconunet.models.third_party.classic_adapter.ClassicAdapter
    init: { algo: music, num_sources_from_scene_K: true }
    is_classic: true
  - name: rootmusic_truek
    adapter_class: reconunet.models.third_party.classic_adapter.ClassicAdapter
    init: { algo: rootmusic, num_sources_from_scene_K: true }
    is_classic: true
  - name: reconunet
    class_path: reconunet.models.deep_learning.EVDUNet.EVDCovarianceReconstructionUNet
    init: { M: 8, tau: 8 }
    checkpoint: experiments/runs/reconunet/checkpoints/best.pt
    collate: reconunet
```

> NB: `ClassicAdapter` is a thin adapter you may need to write.  The
> `run_classic_on_manifest.py` script handles the call pattern
> end-to-end without the harness, so for data-QA work you don't need
> the adapter.  Add it only if you want classical curves on the same
> `rmse_vs_snr.png` as the neural models.

---

## 5. What "good" looks like — a concrete pass/fail table

Run the full verification suite before kicking off the full-corpus
training.  For each row, treat the ">" column as a hard fail.

| Test | Config | Expected RMSPE | Fail if |
| --- | --- | --- | --- |
| MUSIC, K=3, 20 dB, none | `errors_none.yaml` | < 0.1° | > 0.5° |
| MUSIC, K=3, 20 dB, mild | `errors_mild.yaml` | 0.2–0.4° | > 1.0° |
| MUSIC, K=3, 20 dB, harsh | `errors_harsh.yaml` | 0.8–1.5° | > 3.0° |
| MUSIC, K=3, 0 dB, mild | `errors_mild.yaml` | 1–3° | > 8° |
| MUSIC, K=5, 20 dB, mild | `k_sweep.yaml` | 1–3° | > 8° |
| Root-MUSIC, K=3, 20 dB, mild | `errors_mild.yaml` | 0.15–0.35° | > 1.0° |
| ESPRIT, K=3, 20 dB, mild | `errors_mild.yaml` | 0.15–0.35° | > 1.0° |

Angles are in the manifest's broadside-0 convention.  All numbers are
means over at least 500 scenes.  If any row fails, stop and debug —
it's much cheaper now than after 8 hours on the RTX 2000.

---

## 6. Pre-flight checklist before launching a full training run

```
[ ] reconunet-generate ran cleanly on configs/data/verify/tiny.yaml
[ ] inspect_manifest shows K/SNR/angle distributions match the config
[ ] visualize_scene renders a recognisable covariance + MUSIC spectrum
[ ] run_classic_on_manifest with --true-k passes the table in §5
[ ] all three splits (train/val/test) have unique seed sets
[ ] experiments/runs/ is gitignored and has enough free disk
[ ] python -c "import torch; print(torch.cuda.is_available())" → True
[ ] nvidia-smi shows the RTX 2000 Ada with 16 GB free
```

Once that's green, launch the real run per `TRAINING.md`.
