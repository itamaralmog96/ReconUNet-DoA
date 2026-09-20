# Paper-faithful dataset generation

Reference: `docs/paper.pdf` (Almog & Weiss, *ReconUNet: Learning to
Reconstruct Clean Covariance Matrix under Low SNR, Multipath, and Array
Imperfections*, IEEE Trans. Signal Processing, 2026), §IV "Training
Approach and Experimental Setup".

This note records, for anyone re-running the corpus, exactly which
paper settings the on-disk manifest format reproduces and which it
approximates.  Use it together with `configs/data/paper_corpus.yaml`
(training corpus) and `configs/data/paper_corpus_eval.yaml` (wider SNR
evaluation corpus).

## What `paper_corpus.yaml` reproduces exactly

The following items map 1-to-1 onto fields the sampler already
supports, and the on-disk manifest is byte-identical to what the
paper describes:

* Array: ULA, N=8, d=λ/2, f_c = 2.45 GHz narrowband (paper §IV
  "Signal model and geometry").
* Snapshot count: T = 512 per scene.
* Lag depth: τ = 8.
* Angle range: [-60°, +60°] in the broadside-0° convention used by
  `SceneRenderer`, which is the same physical sector the paper writes
  as θ ∈ [30°, 150°] in the array-axis (cosine) convention.
* Source counts: K ∈ {1, 2, 3, 4}, sampled uniformly — equivalent to
  "one main source plus I ∈ {0, 1, 2, 3} equal-power interferers".
* Min separation between direct sources: 10°.
* Modulation: NARROWBAND.
* Imperfection preset: "mild" (gain 0.1 dB, phase 1°, mutual coupling
  0.02, position 1% of λ).
* Training-time SNR range: [-10, +10] dB.
* Total corpus size: 2 × 10⁶ paired examples (with 50k val / 50k
  test).
* Train/val/test seed split: deterministic via
  `numpy.random.SeedSequence.spawn(3)` from the master `rng_seed`.

## Approximations and known gaps

Two parts of the paper's protocol cannot be expressed cleanly in the
existing `SceneManifest` row dtype.  Both are conservative — they err
toward harder data, not easier:

### Gap A: per-scene perfect ↔ imperfect randomization

Paper §IV says "we render both perfect and imperfect arrays, with
imperfection parameters randomized per scene".  In the current
manifest, `array_errors` is a single global preset chosen once at
generation time (one of `none` / `mild` / `harsh`).  There is no way
to flag a row as "this one is the perfect-array twin of the next
one", and the sampler cannot draw a per-scene mix.

**Workaround that recovers the paper's behaviour** at the cost of
double generation:

```bash
# 1) Perfect-array half (1M scenes).
sed 's/array_errors: mild/array_errors: none/'  \
    configs/data/paper_corpus.yaml > /tmp/paper_perfect.yaml
sed -i 's/data\/scenes\/paper\//data\/scenes\/paper_perfect\//g'  \
    /tmp/paper_perfect.yaml
sed -i 's/2_000_000/1_000_000/' /tmp/paper_perfect.yaml
reconunet-generate --config /tmp/paper_perfect.yaml

# 2) Imperfect-array half (1M scenes), already lives at
#    data/scenes/paper/ from the default config.  Reduce its size first.
sed 's/2_000_000/1_000_000/' configs/data/paper_corpus.yaml \
    > /tmp/paper_imperfect.yaml
reconunet-generate --config /tmp/paper_imperfect.yaml

# 3) Concatenate at DataLoader time (small change inside SceneDataset).
```

A cleaner long-term fix is a v1.2 schema field `array_errors_preset`
stored per-row, drawn from `{none, mild}` at sampling time — small
change to `_ROW_DTYPE` (4-byte slot already padded), `Scene` dataclass,
`SceneManifest.random` and `SceneRenderer`.  Out of scope for the
verification work; flagged here so it isn't lost.

### Gap B: multipath-count distribution

Paper §IV says "M ∈ {0, 1, 2, 3} narrowband replicas with small delays
and independent departure angles".  The intended distribution is flat
on `{0, 1, 2, 3}` regardless of K.

The legacy sampler instead caps the replica count by
`max_additional = max_paths − K`, then draws
`U[0, max_additional]`.  Setting `max_paths: 7` in
`paper_corpus.yaml` gives the right ceiling at K=4 (max_additional =
3, matches the paper) but lets the count grow up to 6 at K=1.

**Net effect**: low-K scenes get *more* coherent replicas than the
paper, not fewer.  This is the conservative direction — every classic
estimator in the comparison sees harder data, the network sees a
slightly broader training distribution.  When reporting against the
paper's tables you would want either to (a) post-filter the manifest
to drop rows with `K + num_multipath > 7`, or (b) extend the sampler
to take an explicit `n_multipath_choices` list (the matching change is
~10 lines in `SceneManifest.random`).

## Switching between corpora

The repository now contains three ready-to-use data configs:

| Config                                      | What it is                                                           |
| ------------------------------------------- | -------------------------------------------------------------------- |
| `configs/data/tiny.yaml`                    | 12k-scene smoke test (~2 s).  K=3, no multipath, mild errors.       |
| `configs/data/shared_manifest.yaml`         | Simplified 2M corpus.  K=3 only, no multipath, mild errors.          |
| `configs/data/paper_corpus.yaml`            | **Paper-faithful training**: K∈{1..4}, multipath, 10° sep, train SNR. |
| `configs/data/paper_corpus_eval.yaml`       | **Paper-faithful evaluation**: same but SNR ∈ [-20, +20] dB.        |

Generate any of them with:

```bash
reconunet-generate --config configs/data/<name>.yaml
```

Each config writes to a different `data/scenes/<name>/` subdirectory
(the `manifest.train_path` / `val_path` / `test_path` keys), so you
can keep multiple corpora on disk side by side without overwriting.

To point training at a particular corpus, edit
`configs/train/<model>.yaml` to reference the matching manifest path,
or pass `--data <yaml>` if your training entrypoint supports it.

## Quick verification before committing to 2M scenes

After generating, sanity-check the manifest with the existing verify
runners (no GPU required):

```bash
# 1) Sample inspection — visualise ten random scenes' MUSIC spectra.
python scripts/verify/visualize_scene.py \
    data/scenes/paper/test.npy --n 10

# 2) Per-bucket classical RMSPE against ground truth.
python scripts/verify/run_classic_on_manifest.py \
    data/scenes/paper/test.npy \
    --algo music,rootmusic,esprit,mvdr \
    --n 5000 --true-k \
    --group-by snr_db \
    --snr-bins "-10,-5,0,5,10"
```

The expected order-of-magnitude floor for `mild` + multipath at 0 dB
is RMSPE ≈ 1°-2° for MUSIC/RootMUSIC and ≈ 3°-5° for the ESPRIT
family, matching the unaided baselines in Fig. 6 / Table I of the
paper.

## Bug-fix log

### 2026-04-25 — Imperfections aligned with paper §II-C

Before: `SceneRenderer.render` used Gaussian gain/phase draws, a 1-D
fractional axial-spacing perturbation for position error, and a mutual
coupling matrix that populated only the nearest-neighbour off-diagonals
(`c_1` only).  This neither matched the paper's distributions nor the
legacy `signalgen/array_processing.py` reference implementation.

After: per-scene draws now reproduce paper §II-C eqs. (21)–(25)
exactly:

| Quantity | Paper formula | Renderer draw |
|---|---|---|
| Gain `g_n` | `U[10^(-Δ/20), 10^(+Δ/20)]` (eq. 23) | `rng.uniform(gmin, gmax, size=M)` |
| Phase `φ_n` | `U[-Φ°, +Φ°]` (eq. 23) | `np.deg2rad(rng.uniform(-Φ, +Φ, size=M))` |
| Position `ε_n` | `N(0, σ_pos² · I_2)` in λ units (eq. 21, 2-D) | `rng.normal(0, σ_pos, size=(M, 2))` |
| Coupling `c_k` | `c_k = γ^k · v_k`,  `γ = ρ·exp(jφ_c)`,  `v_k ~ U(1−δ/2, 1+δ/2)` (eqs. 24-25) | full Toeplitz `M = I + E`, populated for all `k ∈ {1,…,M−1}` |

Internal coupling defaults: `φ_c = -100°`, `δ = 0.9` (matching the
legacy `signalgen/array_processing.py` defaults; not exposed on `Scene`
yet — coupling is parameterised by the scalar `mutual_coupling = ρ`).

`_steering_vector` now accepts 2-D `position_err` in λ units and
applies the paper's 2-D phase

  `phase = -2π · ((n·d + ε_x) · sin θ + ε_y · cos θ)`

while still accepting 1-D `position_err` for backward compatibility
with the older `_build_verify_array.py` notebooks.

Verification: `tests/unit/verify_imperfection_distributions.py` runs
4 000 mild scenes and checks empirical mean/variance/range against
the paper formulas, plus structural checks on the coupling matrix
(symmetry, diagonal-of-1, all `(M-1)` off-diagonals follow `ρ^k`
geometric decay) and reproducibility (same `scene.seed` → byte-
identical covariance, snapshots, steering).  Sample numerics:

```
gain:  sample mean=1.00006 (th=1.00007),  var=4.438e-05 (th=4.418e-05)
phase: sample var=0.3300  (th=1/3 = 0.3333)
pos x: sample std=0.01002 (th=0.01000)
|c_1|: 2.041e-02   theory ρ^1 = 2.000e-02
|c_7|: 1.302e-12   theory ρ^7 = 1.280e-12
```

Clean target `R_clean = A_ideal A_ideal^H` is unaffected (it never
saw the imperfections).  All 6 non-fragile `tests/unit/test_scene_pipeline.py`
tests still pass.  The pre-existing
`test_collates_agree_on_labels` failure is a `torch.equal(NaN, NaN)`
fragility unrelated to this change (verified by running with the
imperfection-model edits stashed — same failure).

### 2026-09-06 — Band-limited sources restored: multipath is coherent again

**Scope — what this defect did and did not affect.**  The incoherent-replica
defect described below existed *only* in the rewritten renderer of this
repository (`src/reconunet/data/scene_renderer.py`, the v1.1 port).  It
affected only runs made in this rewritten project for the post-submission
revision work — the reviewer-requested baseline comparisons (SubspaceNet,
SubViT, DA-MUSIC) and the ReconUNet retrains alongside them.  Every one of
those runs was discarded (kept for the record under
`experiments/runs/archive_incoherent_renderer_20260906/`) and redone on the
corrected renderer; the retrain finished 2026-09-10 and the results live in
`experiments/runs/eval_coherent_20260910/`.  **No result in the submitted
manuscript (MDPI *Sensors*, sensors-4536109) was produced with the defective
renderer.**  Every number, table and figure in the submission came from the
original signal generator (`signalgen` / `SignalConfig`), which band-limits
each source to 10 % of Nyquist and therefore always produced coherent
multipath replicas — exactly the behaviour this fix restores.

**Symptom (found by the 2026-07-02 publication audit, blocker B1).**  The
renderer's multipath replicas were *covariance-incoherent*: direct/replica
correlation at lag 0 was |γ| ≈ 0.04 and the noise-free signal covariance of
a 1-source + 2-replica scene had λ₂/λ₁ ≈ 0.4 — no rank collapse.  Paper
§II-B derives |γ| → 1 and an exactly rank-1 covariance.  In addition ~80 %
of delays saturated at the clip, making same-scene replicas identical
copies of each other, and the non-zero lags of the autocorrelation stack
carried no signal (|r_s(ℓ)|/r_s(0) ≈ 0.04 = the 1/√T estimation floor).

**Cause.**  The v1.1 port kept the legacy delay logic but switched the
source model to white (full-band) complex Gaussian samples, whose coherence
time is one sample.  The legacy `SignalConfig` band-limited every source to
10 % of Nyquist (0.05·fs, coherence time ≈ 20 samples), so a delay of a few
samples left the replica highly correlated.  The port also applied the
legacy *upsampled-domain* shift `int(τ·fs·factor/2)` as an original-domain
integer shift (factor = 10 ⇒ 10× too long ⇒ clipping).

**Fix (`scene_renderer.py`, `scene_manifest.py`, `cli/generate_scenes.py`).**
* New `ManifestMeta.source_bw_frac` (default **0.05**, legacy value).  Every
  source waveform — direct paths, all modulations — is band-limited with a
  rectangular baseband mask via `_bandlimit_sources` and renormalised to
  unit power, so the SNR bookkeeping is unchanged.  `0`/`None` restores
  white sources.
* Replicas are the *fractionally* delayed direct waveform via an FFT phase
  ramp (`_fractional_delay`), with the legacy effective delay τ·fs/2 samples
  at 1/factor resolution (0–10 samples for uniform τ ≤ 1/bw).
* `RenderResult.mp_delay_samples` exposes the per-replica delays.

**Measured after the fix** (400 scenes, K=1 + 2 replicas, 40 dB):
direct/replica |γ| mean 0.89 / median 0.92 / min 0.58 (theory: sinc(bw·d) ∈
[0.64, 1], mean ≈ 0.87); λ₂/λ₁ median 0.025; |r_s(ℓ)|/r_s(0) for ℓ=1..7 =
0.99→0.81 (theory sinc(0.05·ℓ)); unit source power exact; empirical SNR
within 0.1 dB of target.  Regression tests: `tests/unit/test_scene_pipeline.py`
(`test_sources_are_bandlimited_and_unit_power`,
`test_multipath_replicas_are_highly_correlated`, …).

**Consequences.**  Rendering is lazy from the manifests, so **no manifest
or corpus is regenerated** and on-disk `meta.yaml` files without the new key
pick up the 0.05 default.  All observations change (band-limiting applies to
every scene, not only multipath ones), so **every trained checkpoint
(ReconUNet, SubspaceNet, SubViT) is stale and must be retrained**, and all
evaluation tables must be regenerated.  The non-zero lags of the τ-stack now
genuinely carry the signal subspace, which is the premise of the multi-lag
input in the paper.  Multipath replica AoAs for a given seed differ from
before (the rng consumption order inside the multipath branch changed).
