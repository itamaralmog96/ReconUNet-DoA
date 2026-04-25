# Paper-faithful training configuration

Reference: `docs/paper.pdf` (Almog & Weiss, *ReconUNet: Learning to
Reconstruct Clean Covariance Matrix under Low SNR, Multipath, and Array
Imperfections*, IEEE Trans. Signal Processing, 2026), §IV "Training
Approach and Experimental Setup", subsection "Exact training
hyperparameters" and the surrounding discussion of the composite loss.

This note is the training-side companion to `PAPER_FAITHFUL_DATASET.md`.
It records, item by item, which paper hyperparameters are reproduced
exactly by `configs/train/{reconunet,subspacenet,subvit}_paper.yaml`,
which are approximated, and which are not currently expressible without a
small change to `src/reconunet/cli/train.py` or the model itself.

## Audit table — ReconUNet

| Setting (paper §IV)                          | Paper value          | Legacy `reconunet.yaml` | Paper-faithful `reconunet_paper.yaml` | Status                                                              |
| -------------------------------------------- | -------------------- | ----------------------- | ------------------------------------- | ------------------------------------------------------------------- |
| Optimiser                                    | Adam                 | `adam`                  | `adam`                                | Match                                                               |
| Learning rate                                | 1e-4                 | 1e-3                    | 1e-4                                  | Match                                                               |
| Weight decay                                 | 1e-5                 | 1e-6                    | 1e-5                                  | Match                                                               |
| LR scheduler                                 | ReduceLROnPlateau    | `reduce_on_plateau`     | `reduce_on_plateau`                   | Match                                                               |
| Scheduler factor                             | 0.7                  | 0.5                     | 0.7                                   | Match                                                               |
| Scheduler patience (epochs)                  | 5                    | 4                       | 5                                     | Match                                                               |
| Batch size                                   | 2048                 | 4096                    | 2048                                  | Match                                                               |
| Max epochs                                   | 300                  | 80                      | 300                                   | Match                                                               |
| Early-stopping patience (val loss)           | 25                   | 15                      | 25                                    | Match                                                               |
| Gradient clipping (global ℓ₂)                | 1.0                  | 1.0                     | 1.0                                   | Match                                                               |
| Sensors N                                    | 8                    | 8                       | 8                                     | Match                                                               |
| Lag depth τ                                  | 8                    | 8                       | 8                                     | Match                                                               |
| Activation                                   | anti-rectifier       | `anti_rectifier`        | `anti_rectifier`                      | Match                                                               |
| Dropout                                      | enabled              | `use_dropout: true`     | `use_dropout: true`                   | Match (rate 0.2 hard-coded; paper does not specify a number)       |
| SCM head: Hermitian symmetrisation           | yes                  | hard-coded              | hard-coded                            | Match (`gram_diagonal_overload`, EVDUNet.py line 33)               |
| SCM head: diagonal loading ε                 | 1                    | hard-coded `eps=1.0`    | hard-coded `eps=1.0`                  | Match (EVDUNet.py line 251)                                         |
| Loss term L_eig (eigenvalue MSE)             | yes (weight 1)       | `eigval_weight: 1.0`    | `eigval_weight: 1.0`                  | Match                                                               |
| Loss term L_rec (Frobenius / N²)             | yes (weight 1)       | `reconstruction_weight: 0.1`, no N² | `reconstruction_weight: 1.0`, ÷N² | Match (legacy under-weighted and missing N² normalisation)         |
| Loss term L_proj (leading-K projector)       | yes (weight 1)       | proxy via `eigvec_weight` | `proj_weight: 1.0` (leading-K)     | Match — see bug-fix log entry "L_dom + leading-K L_proj"          |
| Loss term L_dom (dominant eigvec, sign-rob.) | yes (weight 1)       | absent                  | `dom_weight: 1.0`                     | Match — see bug-fix log entry "L_dom + leading-K L_proj"          |
| Master seed                                  | n/a (deterministic)  | absent                  | `seed: 20260420`                      | Match the data master seed                                          |

Bottom line: the *optimiser schedule, batching, scheduler, model
architecture, activation, SCM head, and the full L = w_eig·L_eig +
w_proj·L_proj + w_dom·L_dom + w_rec·L_rec composite loss now reproduce
the paper exactly.*  The only remaining single-GPU caveat is SubViT's
batch-size ceiling (gap (c) below).

## Bug-fix log

### 2026-04-25 — `SceneDataset` crashed with multipath enabled (fixed)

**Symptom.** Generating any manifest with `enable_multipath: true` (e.g.
`configs/data/paper_corpus.yaml`) caused the very first
`SceneDataset.__getitem__` call to raise

```
ValueError: could not broadcast input array from shape (5,) into shape (2,)
```

so the paper-faithful training run died before it could touch the
optimiser.

**Root cause.** `SceneRenderer.render` populated
`RenderResult.angles_rad` with the *concatenation* of the K direct-path
angles and the N multipath-replica AoAs (length K+N), even though the
dataclass docstring promised `[K] float64`.  `SceneDataset.__getitem__`
then attempted

```python
angles_padded[: scene.n_sources] = result.angles_rad.astype(np.float32)
```

which is a length-K slot on the LHS and a length-(K+N) array on the RHS,
hence the broadcast error.  The renderer's `steering` and
`source_signals` fields were similarly K+N when multipath was on, but
the docstring claimed K — only `angles_rad` triggered a hard crash, the
rest were silent shape-contract violations.

**Fix.** Honour the documented contract on the public API and expose the
multipath AoAs as a separate field:

* `RenderResult.angles_rad` is now **always** length K (direct-path
  angles only) — this is the labelled DoA ground truth.
* `RenderResult.angles_rad_multipath` is new, length N (or empty if
  multipath is disabled).  It carries the U[0, 2π) replica AoAs for
  diagnostics / evaluation but is *not* a regression target.
* `RenderResult.steering` and `RenderResult.source_signals` keep their
  length-(K+N) layout (they're what produced the corrupted snapshots),
  but their docstring now says so explicitly: `[M, K+N]` and `[K+N, T]`.

**Verification.** Three scenes — multipath off (K=2,N=0), the previously
crashing K=2 N=3 case, and K=4 N=2 — all now flow through the dataset
without the broadcast error:

```
Case 1 (no multipath):  angles_rad=(2,)  angles_rad_multipath=(0,)  steering=(8, 2)   OK
Case 2 (K=2, N=3):       angles_rad=(2,)  angles_rad_multipath=(3,)  steering=(8, 5)   OK
                         angles_rad - deg2rad(scene.angles_deg[:K]) = 0.0   (exact match)
Case 3 (K=4, N=2):       angles_rad=(4,)  angles_rad_multipath=(2,)  steering=(8, 6)   OK
```

This unblocks `paper_corpus.yaml` (which sets `enable_multipath: true`)
and any downstream training run.

### 2026-04-25 — supervision target was the corrupted SCM, not the clean one (fixed)

**Symptom.** Before this fix, `_compute_loss_native` in
`src/reconunet/cli/train.py` did

```python
K_true = batch.get("covariance")
```

and `batch["covariance"]` was set by `ReconUNetCollate` to the *same*
covariance the network sees as input — i.e. the corrupted SCM rendered
by `SceneRenderer.render`.  All four loss terms (`L_eig`, `L_proj`
proxy, `L_dom`, `L_rec`) were therefore being computed against the
corrupted SCM instead of the paper's clean ideal SCM.  Effectively the
network was being asked to learn the EVD of its own input, i.e.
identity.  That is **not** the paper supervision (Almog & Weiss 2026
§IV "Dataset generation" — "paired clean ↔ corrupted training").

**Root cause.** Two-fold:

1. `RenderResult` only carried `covariance` (the corrupted sample
   covariance from the rendered noisy/multipath/error-laden snapshots),
   so the dataset literally had nothing else to expose as a target.
2. `ReconUNetCollate` then forwarded that single field as the supervision
   target, with no separate clean-target slot.

**Fix.** Three minimal changes wire a parallel clean target through the
existing pipeline without breaking any caller:

* `SceneRenderer.render(scene)` now also returns
  `covariance_clean = A_ideal @ A_ideal^H`, computed analytically from
  *only* the K direct-path angles with a perfect array (gain=1, phase=0,
  position=0, no mutual coupling, no multipath, no noise).  This is a
  Hermitian PSD matrix of rank K with eigvals
  `[e₁, …, e_K, 0, …, 0]` — exactly the canonical "ideal" SCM.
* `CanonicalSample`, `ReconUNetCollate`, and `SubViTCollate` all carry
  `covariance_clean` as a separate field; `covariance` (corrupted) is
  preserved unchanged.
* `train.py:_compute_loss_native` now reads
  `K_true = batch.get("covariance_clean", batch.get("covariance"))` —
  preferring the clean target, falling back to the corrupted SCM only
  when an old collate is in play (so legacy runs keep working).

**Verification.** A synthetic 2-source / SNR=0 dB / mild errors / 3
multipath replicas scene produces:

```
clean covariance eigvals (descending)     : [8.25, 7.75, 0, 0, 0, 0, 0, 0]
  rank > 1e-3                             : 2 / 8        (= K)
corrupted covariance eigvals (descending) : [13.96, 9.24, 3.82, 1.16, 1.10, 1.00, 0.96, 0.89]
  rank > 1e-3                             : 8 / 8        (full, as expected)
|| corrupted - clean ||_F                 : 8.55         (>> 0)
|| renderer.covariance_clean - A_ideal A_ideal^H ||_F : 0.0   (analytic match)
```

So the network now sees a real denoising signal (clean target ≠ input).
Re-running `reconunet-train --config configs/train/reconunet_paper.yaml`
will pick this up automatically — no config change required.

**Why an analytic clean target rather than a paired clean *render*.**
A second render at infinite SNR with the same source seeds would also
work, but introduces O(1/√T) sample variance in the target.  Using the
ensemble form `A_ideal A_ideal^H` is deterministic, exactly rank-K,
and matches the noise-free / error-free / multipath-free SCM the paper
defines for the L_eig / L_proj / L_dom / L_rec composite.

### 2026-04-25 — L_dom + leading-K L_proj + N²-normalised L_rec (fixed)

**Symptom.** Even after the clean-target fix above, four observable
deviations from paper §IV "Composite loss" remained in
`_compute_loss_native`:

1. `L_dom` (sign-/phase-robust dominant-eigenvector distance) was
   absent — `dom_weight` in the YAML had no key to bind to and was
   silently 0.
2. `L_proj` was a *column-wise* full-eigenbasis similarity
   `1 - mean_k |<v_k_pred, v_k_true>|^2` rather than the paper's
   leading-K projector difference
   `||V_S V_S^H (pred) - V_S V_S^H (true)||_F^2`.
3. `batch["n_sources"]` was already produced by `ReconUNetCollate` but
   never read in the loss, so the per-scene K (variable across the
   `k_choices=[1,2,3,4]` corpus) could not enter the L_proj signal-
   subspace mask.
4. `L_rec` was the raw squared Frobenius norm `||K_recon - K_true||_F^2`
   instead of the paper's `(1/N²) ||...||_F^2`, making the term scale
   with N² and dwarf the other three at N=8.

**Root cause.** The previous loss block only used the predicted-vs-true
eigvec matrices through a global column-wise inner product, which
discards both the eigenvalue ordering needed for L_dom and the
per-sample K cutoff needed for the leading-K projector.  N²
normalisation was simply omitted.

**Fix.** Single rewrite of the EVDUNet-output branch in
`_compute_loss_native`:

* `K_true = batch.get("covariance_clean")` with fallback to
  `batch.get("covariance")`.
* `L_rec  = (1/N²) · per-sample sum |K_recon - K_clean|² , then mean over batch`.
* `eigh(K_true_dev)` followed by `flip(dims=[-1])` produces
  descending-sorted eigenvalues `w_true` and the matching eigenvector
  matrix `V_true`.
* `L_eig  = MSE(eigvals_pred, w_true)` (descending order on both sides).
* Per-sample mask `(col_idx < n_src.unsqueeze(1))` of shape `[B, 1, N]`
  zeroes out noise-subspace columns of both `eigvecs_pred` and
  `V_true`; the masked outer product collapses to the K-column
  projector exactly.
* `L_proj = (1/N²) · per-sample ||P_pred - P_true||_F², then mean over batch`.
* `L_dom  = (1 - |<v1_pred, v1_true>|).mean()`, which is invariant to a
  global complex phase on `v1_pred` (and therefore to its sign).
* Loss-key mapping documented in the function docstring:
  `eigval_weight → L_eig`, `proj_weight → L_proj`,
  `eigvec_weight → L_proj` (legacy alias), `dom_weight → L_dom`,
  `reconstruction_weight → L_rec`.
* `configs/train/reconunet_paper.yaml` now exposes
  `proj_weight: 1.0` and `dom_weight: 1.0` directly, replacing the
  `eigvec_weight` proxy.  Legacy configs that still use `eigvec_weight`
  keep working — the alias maps to the same paper-faithful L_proj.

**Verification.** A synthetic mixed-K batch
(`n_sources = [1, 2, 3, 4]`, M=8) at
`tests/unit/verify_composite_loss.py` exercises the new code path
end-to-end:

```
Test 1: prediction == truth
  -> total loss 5.96e-08         (~0, float roundoff in L_dom)
  -> L_eig = L_proj = L_rec = 0  (exact match)

Test 2: perturbed prediction
  -> L_eig=6.6e-3, L_proj=1.0e-3, L_dom=8.3e-3, L_rec=1.8e-2
  -> all four positive, finite, well-conditioned magnitudes
  -> .backward() produces non-zero gradients on K_recon, eigvals, eigvecs

Test 3: per-sample K mask is honoured
  -> proj with K=[1,2,3,4]: 1.01e-3
  -> proj with K=[0,0,0,0]: 0.0       (V_S empty → projector is zero)

Test 4: legacy eigvec_weight aliases L_proj
  -> metrics["eigvec"] == metrics["proj"] (max abs diff < 1e-6)
```

So all four paper terms are now active, the per-scene K is honoured
without a Python loop, gradients flow, and old configs that referenced
`eigvec_weight: 1.0` still produce identical numerics for that term.

## Approximations and gaps

> **Update — 2026-04-25**: gaps (a) "L_proj proxy" and (b) "L_dom not
> implemented" are now closed; see the bug-fix log entry "L_dom +
> leading-K L_proj + N²-normalised L_rec" below.  They are kept here as
> historical context only — `reconunet_paper.yaml` no longer falls back
> to the column-wise proxy or omits the dominant-eigvector term.

### Gap (a) — RESOLVED: L_proj now uses the leading-K signal-subspace projector

Paper §IV defines

```
L_proj = (1 / N^2) || P_S(V_pred) - P_S(V_true) ||_F^2
```

with `P_S(V) = V_S V_S^H` = projector onto the **K signal eigenvectors**,
where K is per-scene (`K ∈ {1, 2, 3, 4}` in `paper_corpus.yaml`).

`src/reconunet/cli/train.py::_compute_loss_native` previously computed a
column-wise subspace distance averaged over **all M** eigenvectors —
fine in the well-separated high-SNR regime, wrong in the paper sense.
It now builds the projector explicitly with a per-sample K mask sourced
from `batch["n_sources"]`, using a fully vectorised broadcast (no
Python `for b in range(B)` loop):

```python
n_src    = batch["n_sources"].to(eigvecs_pred.device, dtype=torch.long)
col_idx  = torch.arange(N, device=eigvecs_pred.device).unsqueeze(0)
mask     = (col_idx < n_src.unsqueeze(1)).to(eigvecs_pred.real.dtype).unsqueeze(1)
Vs_pred  = eigvecs_pred * mask                # zero out noise-subspace columns
Vs_true  = V_true * mask
P_pred   = Vs_pred @ Vs_pred.conj().transpose(-2, -1)
P_true   = Vs_true @ Vs_true.conj().transpose(-2, -1)
l_proj   = ((P_pred - P_true).abs() ** 2).sum(dim=(-1, -2)) / (N * N)
```

The new `proj_weight` knob in `configs/train/reconunet_paper.yaml`
controls this term; `eigvec_weight` is kept as a legacy alias for the
same loss key so existing configs keep working.

### Gap (b) — RESOLVED: L_dom (sign- and phase-robust dominant eigenvector loss) is now wired in

Paper §IV §"Composite loss" introduces

```
L_dom = 1 - max_{s in {-1, +1}} | < s · v1_pred , v1_true > |
```

i.e. a sign-robust angular distance between the **single dominant**
eigenvector (largest eigenvalue) of the predicted and true covariance.
For complex eigenvectors the maximisation over the sign `s ∈ {±1}`
generalises to a maximisation over a global complex phase
`e^{iθ}, θ ∈ [0, 2π)`, which is taken automatically by `.abs()` on the
inner product:

```python
v1_pred = eigvecs_pred[..., :, 0]                     # [B, N] complex
v1_true = V_true[..., :, 0]
inner   = (v1_pred.conj() * v1_true).sum(dim=-1)      # [B] complex
l_dom   = (1.0 - inner.abs()).mean()                   # scalar in [0, 1]
```

The new `dom_weight` knob in `configs/train/reconunet_paper.yaml`
weights this term (paper value 1.0).  It is reported to stabilise the
early epochs when `L_proj` is still dominated by mis-ordered
noise-subspace columns.

### Gap (c): SubViT batch size cannot reach 2048 on a single GPU

The paper trains all three baselines at batch=2048.  SubViT's ViT
backbone makes that infeasible at single-GPU memory budgets typical of
this thesis (≈ 24 GB).  `subvit_paper.yaml` therefore sets
`batch_size: 512` and the trainer does **not** currently support
gradient accumulation.

**Workaround**: implement a 4-line `accumulate_grad_batches` knob in
`train.py`'s inner loop (call `optim.step()` only every N micro-batches,
multiply `loss` by `1/N` before backward).  For the verification work I
recommend either (i) running SubViT at batch=512 and noting the
deviation in the eval table, or (ii) running on a multi-GPU node where
batch=2048 fits.

## What `reconunet_paper.yaml` is intended to replace

The legacy `configs/train/reconunet.yaml` was tuned for thesis-time
quick iteration: a 4× larger batch, a 10× higher LR, a tighter scheduler
(factor 0.5, patience 4), 80 epochs, early-stop patience 15, and
`reconstruction_weight: 0.1`.  None of those match the paper.

Keep the legacy file in place — it remains useful for fast smoke runs
on the simplified `shared_manifest.yaml`.  When reproducing the paper's
numbers, point the trainer at the new `*_paper.yaml` configs:

```bash
# Generate the paper-faithful training corpus first (≈ 2M scenes).
reconunet-generate --config configs/data/paper_corpus.yaml

# Train the three baselines with the paper's optimiser schedule.
reconunet-train    --config configs/train/reconunet_paper.yaml
reconunet-train    --config configs/train/subspacenet_paper.yaml
reconunet-train    --config configs/train/subvit_paper.yaml
```

## Quick verification before committing to a 300-epoch run

Before kicking off a multi-day run, dry-run with a smaller epoch budget
and the tiny corpus to confirm the YAML parses and the model converges:

```bash
# 10-epoch dry run on the 12k-scene smoke corpus.
reconunet-generate --config configs/data/tiny.yaml
reconunet-train    --config configs/train/reconunet_paper.yaml \
                   --epochs 10
```

If the validation RMSPE drops below ≈ 5° within 5 epochs on the tiny
corpus the optimiser schedule is wired correctly; full convergence to
the paper's <1° at 0 dB will only happen on the full
`paper_corpus.yaml` run.
