# Verify the generated signals are what you expect

A focused, copy-pasteable runbook for **signal-level** sanity checks on a
freshly generated manifest.  For the full test matrix (training,
evaluation, imperfections, multipath sweeps) see `VERIFY_AND_TEST.md`.

The goal here is narrow: **given a `.npy` manifest, confirm the rendered
X, R, and A actually look like a ULA observing K plane waves with the
SNR and angles we wrote into the scene record.**

All commands assume you are in `ReconUNet/` with the venv active.

---

## 0. Generate a small, known-good corpus

Do the checks on a corpus you understand.  The `tiny` config is
deliberately small, narrow-band, no multipath, mild imperfections —
exactly what the classical estimators should solve in their sleep.

```bash
reconunet-generate -c configs/data/verify/tiny.yaml --overwrite
MANIFEST=data/scenes/verify/tiny/train.npy
```

---

## 1. Manifest-level sanity (metadata, distributions, seeds)

```bash
python scripts/verify/inspect_manifest.py "$MANIFEST"
```

**Pass criteria**

| Check | Expected |
| --- | --- |
| Row size | `dtype.itemsize == 96` bytes |
| Number of rows | Matches `sampling.n_train` in the config |
| K histogram | All bins in `sampling.k_choices` non-empty, roughly uniform |
| SNR histogram | Spread ≈ uniform over `meta.snr_range_db` |
| Angle histogram | All samples inside `meta.angle_range_deg`, no pile-up at the edges |
| Seeds | Every seed is unique, and **disjoint** across train / val / test |
| Imperfection preset | Matches the YAML (`none` → all four error scalars = 0; `mild` → 0.1/1.0/0.02/1.0; etc.) |
| Multipath flag | `has_multipath == False` for a non-multipath config |

A failure in any of those means the manifest is wrong before you even
render anything — stop, fix the config, regenerate.

---

## 2. Single-scene render sanity

Pick one scene and render it.  This is the fastest way to spot a
dimensionality / dtype / sign-convention bug.

```bash
python scripts/verify/visualize_scene.py "$MANIFEST" --index 0 \
    --out experiments/verify/scene_0.png
```

Or, inline, in a Python REPL:

```python
import numpy as np
from reconunet.data.scene_manifest import SceneManifest
from reconunet.data.scene_renderer import SceneRenderer

m = SceneManifest.load("data/scenes/verify/tiny/train.npy")
renderer = SceneRenderer(m.meta)
scene = m[0]
X, A, s, angles_rad = renderer.render(scene)   # (M,T), (M,K), (K,T), (K,)

M, T = m.meta.M, m.meta.T
assert X.shape == (M, T),  f"X shape {X.shape} != ({M},{T})"
assert A.shape == (M, scene.k + scene.num_multipath)
assert s.shape == (scene.k + scene.num_multipath, T)
assert np.isfinite(X).all(),  "NaN/Inf in X"
```

**Pass criteria**

| Invariant | Test | Expected |
| --- | --- | --- |
| Shape | `X.shape == (M, T)` | True |
| Finite | `np.isfinite(X).all()` | True |
| Dtype | `X.dtype == np.complex64` | True |
| Hermitian R | `R = X @ X.conj().T / T`; `np.max(np.abs(R − R.conj().T))` | `< 1e-6` (float32) |
| PSD R | `np.linalg.eigvalsh(R).min()` | `≥ -1e-6` |
| Signal-subspace rank | count of eigenvalues > `10 × σ²_noise` | `== K` (for a no-multipath scene) |

If rank ≠ K you almost always have either (a) the wrong number of
columns in A, (b) correlated source draws, or (c) the noise
variance computed against the wrong signal power.

---

## 3. SNR bookkeeping

The scalar in `scene.snr_db` must match what you can measure from `X`.

```python
P_signal = np.mean(np.abs(A @ s) ** 2)              # per-sample, per-antenna
P_noise  = np.mean(np.abs(X - A @ s) ** 2)
snr_measured = 10 * np.log10(P_signal / P_noise)
print(f"requested {scene.snr_db:.2f} dB, measured {snr_measured:.2f} dB")
```

**Pass criterion (per-source SNR convention — what the renderer uses):**
the measured SNR should equal

```
snr_measured ≈ scene.snr_db + 10·log10(K)
```

**This is not a bug.**  The renderer sets `noise_var = 1 / 10^(snr_db/10)`
against **unit-power per-source** signals, so when K independent sources
sum at each element the per-element total signal power is K, giving a
`+10·log10(K)` bias between the scalar stored on the scene and what a
naive `P_signal / P_noise` measurement reports.  This matches the legacy
`signal_generator.add_noise` ("noise is scaled against one source") and
the DOA-literature "per-source SNR" convention.

If you want the stored scalar to measure as-is, subtract the bias:

```python
snr_per_source = snr_measured - 10 * np.log10(scene.n_sources)
# now |snr_per_source − scene.snr_db| < 0.5 dB averaged over ~32 scenes
```

Spotting a real bug: a bias of exactly **3 dB** (complex-vs-real noise
power confusion) or a bias that varies with SNR level (rather than with
K) would be a real bug — at that point the noise-variance formula or
the complex-Gaussian scaling has drifted from the legacy.

> Note for multipath scenes: the legacy convention (ported in v1.1)
> computes the noise variance against **direct-path** signal power.
> So `scene.snr_db` is the direct-path per-source SNR, and measured
> total signal power will be higher than `K · P_per_source` by the
> multipath sum.

---

## 4. Steering matrix matches nominal ULA

With `errors=none` the rendered `A` should equal the textbook ULA
steering at the ground-truth angles exactly (up to float32 round-off).

```python
# scene with errors_none
d_over_lambda = m.meta.element_spacing_lambda
ant = np.arange(M)[:, None]
A_nominal = np.exp(-1j * 2 * np.pi * d_over_lambda *
                   ant * np.sin(angles_rad[None, :scene.k]))
err = np.max(np.abs(A[:, :scene.k] - A_nominal))
print(f"|A - A_nominal|_max = {err:.2e}")     # expect < 1e-5
```

With `errors=mild` or `harsh`, the residual `A − A_nominal` should
**not** be zero, and its spectrum should broadly match the preset's
per-antenna gain/phase/position scales.  If harsh and none produce the
same A, the imperfection fields are not being read by the renderer.

---

## 5. Classical MUSIC on a clean scene must nail it

This is the single most valuable sanity check.  If MUSIC with the
correct K cannot solve an `errors_none`, 20 dB, K=2, well-separated
scene to within a fraction of a degree, *your data is broken* — no
model will learn from it.

```bash
python scripts/verify/run_classic_on_manifest.py "$MANIFEST" \
    --algo music --n 200 --true-k \
    --group-by snr_db --out experiments/verify/music_tiny.csv
```

**Pass criterion:**

| SNR | preset | Expected RMSPE (degrees) |
| --- | --- | --- |
| 20 dB | none  | `< 0.1°` |
| 20 dB | mild  | `< 0.5°` |
| 20 dB | harsh | `< 2°` |
|  0 dB | mild  | `< 3°` |

If the `errors=none, 20 dB` row is not sub-degree, stop and debug
before trusting anything downstream.

---

## 6. Reproducibility

Same seed → identical X, byte-for-byte.

```python
X1, *_ = renderer.render(m[0])
X2, *_ = renderer.render(m[0])
print("determinism:", np.max(np.abs(X1 - X2)))   # expect 0.0
```

If this is not zero, there is an un-seeded RNG somewhere in the path
and every run of your training pipeline is seeing a different X for
the same scene.  Fix immediately.

---

## 7. Multipath scenes (only if `has_multipath=True`)

For a multipath manifest (`configs/data/verify/multipath.yaml`):

```python
scene = m_mp[0]
assert scene.has_multipath
X, A, s, ang = renderer.render(scene)
assert A.shape[1] == scene.k + scene.num_multipath
R = X @ X.conj().T / T
# Expect: K + num_multipath large eigenvalues, not just K
eigs = np.linalg.eigvalsh(R)[::-1]
print("top eigs:", eigs[:scene.k + scene.num_multipath + 2])
```

**Pass criteria**

* `A.shape[1] == K + num_multipath`.
* `R`'s eigenspectrum has `K + num_multipath` eigenvalues clearly
  above the noise floor.
* `scene.snr_db` still equals the direct-path SNR (see §3 note).
* Classical MUSIC with `true-k = K` **should degrade** vs the no-
  multipath baseline — this is the point of the test, not a bug.

---

## 8. Cross-scene statistical sanity (batch check)

Run this over a few thousand scenes to catch rare issues (NaN bursts,
angle wrap-around, etc.):

```python
from tqdm import tqdm
bad = 0
for i in tqdm(range(len(m))):
    X, A, s, _ = renderer.render(m[i])
    if not np.isfinite(X).all():
        bad += 1
print(f"non-finite scenes: {bad} / {len(m)}")
```

`bad` must be zero.

---

## 9. One-liner green-light script

When all of the above is passing, you can collapse the loop into a
single "green-light" check that you run before every big
regeneration:

```bash
reconunet-generate -c configs/data/verify/tiny.yaml --overwrite && \
python scripts/verify/inspect_manifest.py data/scenes/verify/tiny/train.npy && \
python scripts/verify/run_classic_on_manifest.py \
    data/scenes/verify/tiny/val.npy --algo music --n 200 --true-k
```

Exit status 0 and sub-degree RMSPE at 20 dB mild-imperfections = signals
are as expected.  Proceed to train.
