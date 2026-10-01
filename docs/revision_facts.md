# Revision facts — ReconUNet (Sensors sensors-4536109)

Generated 2026-09-20 07:46 UTC from repository state `e448159` by `scripts/analysis/revision_facts.py`. Every number below is read from the code, the configs, the checkpoints or the run logs of this repository; nothing is copied from the manuscript. The nine headings follow the fact list agreed for the revision (reconstructed here as: loss, architecture, training, data, scenarios, source count, metrics, cost, baselines).

## 1. Loss weights and composite loss

Composite loss (trainer `_compute_loss_native`, target = clean covariance R\* = A A^H of the K direct paths, unit-power sources, perfect array):

    L = w_eig·L_eig + w_proj·L_proj + w_dom·L_dom + w_rec·L_rec
    L_rec  = (1/N²)·‖R̂ − R*‖²_F
    L_eig  = mean_k (λ̂_k − λ*_k)²          (both sorted descending)
    L_proj = (1/N²)·‖V̂_S V̂_S^H − V*_S V*_S^H‖²_F   (leading-K columns, per-sample K)
    L_dom  = 1 − |⟨v̂_1, v*_1⟩|             (phase- and sign-invariant)

| weight | value | term |
|---|---|---|
| eigval_weight | 1 | L_eig |
| proj_weight | 1 | L_proj |
| dom_weight | 1 | L_dom |
| reconstruction_weight | 1 | L_rec |

All four weights are 1.0 in the configuration that trained the released model (`configs/train/reconunet_paper.yaml`). Ablation variants change only the set of active terms.

## 2. Architecture and model size

ReconUNet = `EVDCovarianceReconstructionUNet(tau=8, M=8, activation=anti_rectifier, dropout)`; input is the lag stack [τ=8, 2N=16, N=8] (real/imag stacked), outputs (λ̂ [N], V̂ [N×N] complex, R̂ = V̂ diag(λ̂) V̂^H).

Top-level modules and parameter counts:

| module | params |
|---|---|
| base_unet | 321329 |
| tau_compression | 9 |
| eigenvalue_head | 9480 |
| eigenvector_head | 5185 |

Sub-blocks of `base_unet` (the covariance U-Net):

| module | type | params |
|---|---|---|
| activation | ReLU | 0 |
| dropout | Dropout | 0 |
| enc_conv1 | Conv2d | 1168 |
| enc_bn1 | BatchNorm2d | 64 |
| enc_conv1_2 | Conv2d | 4624 |
| enc_bn1_2 | BatchNorm2d | 64 |
| enc_conv2 | Conv2d | 9248 |
| enc_bn2 | BatchNorm2d | 128 |
| enc_conv2_2 | Conv2d | 18464 |
| enc_bn2_2 | BatchNorm2d | 128 |
| enc_conv3 | Conv2d | 36928 |
| enc_bn3 | BatchNorm2d | 256 |
| enc_conv3_2 | Conv2d | 73792 |
| enc_bn3_2 | BatchNorm2d | 256 |
| pool1 | MaxPool2d | 0 |
| pool2 | MaxPool2d | 0 |
| pool3 | MaxPool2d | 0 |
| bottleneck1 | Conv2d | 4624 |
| bn_bottleneck1 | BatchNorm2d | 64 |
| bottleneck2 | Conv2d | 18464 |
| bn_bottleneck2 | BatchNorm2d | 128 |
| bottleneck3 | Conv2d | 73792 |
| bn_bottleneck3 | BatchNorm2d | 256 |
| upconv3 | ConvTranspose2d | 16416 |
| dec_conv3 | Conv2d | 27680 |
| dec_bn3 | BatchNorm2d | 128 |
| dec_conv3_2 | Conv2d | 18464 |
| dec_bn3_2 | BatchNorm2d | 128 |
| upconv4 | ConvTranspose2d | 4112 |
| dec_conv4 | Conv2d | 6928 |
| dec_bn4 | BatchNorm2d | 64 |
| dec_conv4_2 | Conv2d | 4624 |
| dec_bn4_2 | BatchNorm2d | 64 |
| final_conv | Conv2d | 264 |
| tau_to_single_conv | Conv2d | 9 |

Parameter counts from the checkpoints:

| model | trainable parameters |
|---|---|
| ReconUNet (released) | 337842 |
| SubspaceNet | 41761 |
| SubViT | 42642674 |
| DA-MUSIC (per K, each) | 15187 |
| EVD-UNet of the submitted manuscript | 0 |

## 3. Training protocol and outcomes

| model | optimizer | lr | weight decay | batch | max epochs | scheduler | early stop | grad clip | AMP | loss keys | seed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ReconUNet | adam | 0.0001 | 1e-05 | 2048 | 300 | reduce_on_plateau ×0.7 / 5 | 25 | 1 | True | eigval_weight, proj_weight, dom_weight, reconstruction_weight | 20260420 |
| SubspaceNet | adam | 0.0001 | 1e-05 | 2048 | 300 | reduce_on_plateau ×0.7 / 5 | 25 | 1 | True | rmspel_weight | 20260420 |
| SubViT | adam | 0.0001 | 1e-05 | 2048 | 300 | reduce_on_plateau ×0.7 / 5 | 25 | 1 | False | bce_weight | 20260420 |
| DA-MUSIC K=1 | adam | 0.0001 | 1e-05 | 2048 | 300 | reduce_on_plateau ×0.7 / 5 | 25 | 1 | False | rmspel_weight | 20260420 |

Shared schedule from the paper: Adam, LR 1e-4, weight decay 1e-5, batch 2048, ≤300 epochs, ReduceLROnPlateau(0.7, 5), early stopping on validation loss with patience 25, gradient-norm clip 1.0, seed 20260420. Checkpoint selection = minimum validation loss (`best.pt`). AMP is force-disabled for complex tensors, so every model trains in fp32. DA-MUSIC trains one fixed-head model per K on the K-subset of the same corpus (`data.k_filter`).

Outcomes of the 2026-09-06 → 2026-09-10 retrain on the corrected renderer (NVIDIA RTX 2000 Ada 16 GB, 8 loader workers):

| run | epochs_run | best_epoch(val loss) | val_rmspe_at_best(deg) | last_val_rmspe(deg) | last_lr | min_val_rmspe(deg) | wall-clock |
|---|---|---|---|---|---|---|---|
| ReconUNet | 109 | 84 | 2.174 | 2.022 | 8.2e-06 | 2.018 | 287min |
| SubspaceNet | 83 | 58 | 3.991 | 4.042 | 5.8e-06 | 3.967 | 1817min |
| SubViT | 161 | 136 | 6.439 | 6.451 | 1.0e-06 | 6.426 | 2633min |
| DA-MUSIC K=1 | 207 | 182 | 8.213 | 8.236 | 2.0e-06 | 8.209 | 157min |
| DA-MUSIC K=2 | 300 | 296 | 5.12 | 5.134 | 4.9e-05 | 5.11 | 228min |
| DA-MUSIC K=3 | 300 | 298 | 4.566 | 4.595 | 1.0e-04 | 4.566 | 229min |
| DA-MUSIC K=4 | 300 | 300 | 4.132 | 4.132 | 7.0e-05 | 4.132 | 230min |

DA-MUSIC K=2,3,4 reached the 300-epoch cap without early stopping (continuation runs to 600 epochs are appended below when finished).

## 4. Dataset and imperfection facts

| quantity | value |
|---|---|
| array | ULA, N=8, d=0.5 λ, f_c=2.45e+09 Hz narrowband |
| snapshots T | 512 |
| lag depth τ | 8 |
| source bandwidth | 0.05·fs (10 % of Nyquist) → coherence time ≈ 20 samples |
| direct sources K | [1, 2, 3, 4] uniform |
| min separation | 10.0° |
| DoA sector | [-60.0, 60.0] broadside convention (= [30°,150°] array-axis convention) |
| training SNR | [-10.0, 10.0] dB (evaluation sweeps −20…20 dB) |
| multipath | enabled, ≤ 7−K coherent replicas per scene, count U{0..max}, delays 'uniform' with mp_max_delay_factor=10.0 |
| imperfection preset | mild |
| corpus | 2,000,000 train / 50,000 val / 50,000 test scenes |
| master seed | 20260420 |
| manifest row | 96 bytes (seed, K, angles, SNR, error magnitudes, multipath params); rendering is lazy and deterministic from the seed |

Imperfection presets (per-scene draws, paper §II-C eqs. 21–25; `SceneManifest.random` / `SceneRenderer.render`):

| preset | gain error | phase error | mutual coupling |γ| | position error |
|---|---|---|---|---|
| mild | ±0.1 dB, g_n ~ U[10^(−0.1/20), 10^(+0.1/20)] | φ_n ~ U[−1°, +1°] | 0.02 | ε_n ~ N(0, (0.01 λ)² I₂) (1 % of λ, 2-D) |
| harsh | ±0.5 dB | U[−5°, +5°] | 0.1 | N(0, (0.05 λ)² I₂) (5 % of λ) |

Mutual coupling is a reciprocal Toeplitz matrix I + E with unit-step coefficient γ = |γ|·e^{j(−100°)} and per-diagonal jitter v_k ~ U[0.55, 1.45] (legacy `coupling_variation` 0.9); composite operator H = M·G as in paper eq. (26).

Empirical composition of the 2 M-scene training manifest (share of scenes by K, and within each K the fraction with r coherent replicas):

| K | scenes | share | replicas=0 | replicas=1 | replicas=2 | replicas=3 | replicas=4 | replicas=5 | replicas=6 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 500193 | 0.250 | 0.143 | 0.144 | 0.143 | 0.144 | 0.142 | 0.143 | 0.142 |
| 2 | 499413 | 0.250 | 0.166 | 0.168 | 0.166 | 0.167 | 0.167 | 0.166 | 0.000 |
| 3 | 499943 | 0.250 | 0.201 | 0.200 | 0.200 | 0.199 | 0.200 | 0.000 | 0.000 |
| 4 | 500451 | 0.250 | 0.249 | 0.251 | 0.250 | 0.250 | 0.000 | 0.000 | 0.000 |

SNR in the training manifest: min -10.00, max 10.00 dB (uniform). Error magnitudes per scene: gain [0.0999755859375] dB, phase [1.0]°, coupling [0.0200042724609375], position [1.0] %.

Measured on 300 multipath scenes: replica delays 0.10–9.90 samples (mean 4.86); direct/replica waveform correlation |γ| mean 0.886, median 0.920, min 0.520 (theory sinc(bw·delay); rank-1 coherent model of paper §II-B).

## 5. Scenario definitions (Section IV stress tests)

| preset | scenario | K direct | coherent replicas | angle configs | SNR levels (dB) | errors | seed |
|---|---|---|---|---|---|---|---|
| mild | advanced1_ood | 1 | 6 | 1000 | [-20, -15, -10, -5, 0, 5, 10, 15, 20] | mild | 20260424 |
| mild | advanced2_crowded | 4 | 3 | 1000 | [-20, -15, -10, -5, 0, 5, 10, 15, 20] | mild | 20260425 |
| mild | basic | 1 | 0 | 1000 | [-20, -15, -10, -5, 0, 5, 10, 15, 20] | mild | 20260422 |
| mild | moderate | 2 | 1 | 1000 | [-20, -15, -10, -5, 0, 5, 10, 15, 20] | mild | 20260423 |
| harsh | advanced1_ood | 1 | 6 | 1000 | [-20, -15, -10, -5, 0, 5, 10, 15, 20] | harsh | 20260424 |
| harsh | advanced2_crowded | 4 | 3 | 1000 | [-20, -15, -10, -5, 0, 5, 10, 15, 20] | harsh | 20260425 |
| harsh | basic | 1 | 0 | 1000 | [-20, -15, -10, -5, 0, 5, 10, 15, 20] | harsh | 20260422 |
| harsh | moderate | 2 | 1 | 1000 | [-20, -15, -10, -5, 0, 5, 10, 15, 20] | harsh | 20260423 |

Each angle configuration (seed, angles, imperfection draw, multipath layout) is reused at every SNR; only the noise power changes (`SceneManifest.fixed_angles_snr_sweep`). The corpus 'Moderate' scenario is 2 direct + 1 replica; the submitted Table II caption said 3 + 1 — the manuscript text must be made consistent with the data.

## 6. How the source count K is defined and supplied

* K is the number of **direct-path** sources of a scene (`n_sources`); coherent multipath replicas are never counted in K and are not labelled targets. K is assumed **known** for every method (classical, aided and learned), per scene.
* Classical estimators (Bartlett, MVDR, MUSIC, Root-MUSIC, ESPRIT, Unitary-ESPRIT) and ReconUNet-aided estimators receive K directly; MUSIC-type methods split the eigenbasis into K signal / N−K noise vectors.
* SubspaceNet: variable-K forward — the adapter passes the true per-sample K to the differentiable Root-MUSIC head (published code fixes M at build time).
* SubViT (DOA-ViT): K strongest local maxima of the 121-point spatial spectrum, per sample.
* DA-MUSIC: the published head is `Linear(hidden, M)`, so one model per K is trained and evaluation dispatches each scene to the model of its true K.
* Under coherent multipath the signal-subspace rank stays K (|γ|≈0.9 per replica), which is what the covariance reconstruction exploits; raw subspace methods see a rank-deficient / leaked spectrum.

## 7. Metric definitions

* **Pooled RMSE (paper eq. 31)** = sqrt( mean over all scenes and all K sources of the squared angle error in degrees ). Estimates and truths are sorted (optimal permutation for scalars); errors are **not** wrapped (sin θ is injective on [−90°, 90°] in the broadside convention).
* **Median RMSPE** = median over scenes of sqrt(mean_k e_k²) — robust companion reported in the test-split table.
* **Resolution probability** (separation sweep) = P(|θ̂_i − θ_i| < Δθ/2 for both sources).
* **CRLB** = stochastic (unconditional) Gaussian-signal bound per scene (`stochastic_crlb_deg`, M, T, angles, SNR, d/λ), RMS over scenes.
* **Bootstrap 95 % CI** = percentile bootstrap over scenes, 1000 resamples, seed 20260920 (`bootstrap_ci.py`), for pooled RMSE and median RMSPE.
* Angle convention everywhere: broadside 0°, sector ±60°; the paper's [30°, 150°] array-axis angles are the same physical directions.

## 8. Computational cost

Inference latency measured 2026-09-20 07:46 UTC on NVIDIA RTX 2000 Ada Generation (median of repeats, GPU-synchronised; lag-stack formation included where the pipeline needs it; K = 1 test scenes):

| pipeline | batch | ms per scene | ms per batch | peak GPU MiB |
|---|---|---|---|---|
| ReconUNet forward only (GPU) | 1 | 1.191 | 1.191 | 179.576 |
| ReconUNet + Root-MUSIC (GPU net, CPU eigh/roots) | 1 | 1.834 | 1.834 | 179.580 |
| Root-MUSIC on raw SCM (CPU eigh + roots) | 1 | 0.258 | 0.258 | 173.571 |
| eigh of the 8×8 SCM alone (CPU) | 1 | 0.011 | 0.011 | 173.571 |
| SubspaceNet + Root-MUSIC head (GPU) | 1 | 1.842 | 1.842 | 173.838 |
| SubViT (GPU) | 1 | 1.527 | 1.527 | 173.888 |
| DA-MUSIC K=1 (GPU) | 1 | 0.755 | 0.755 | 182.236 |
| ReconUNet forward only (CPU, 20 threads) | 1 | 72.108 | 72.108 |  |
| ReconUNet forward only (GPU) | 1024 | 0.032 | 32.804 | 307.567 |
| ReconUNet + Root-MUSIC (GPU net, CPU eigh/roots) | 1024 | 0.071 | 72.687 | 311.567 |
| Root-MUSIC on raw SCM (CPU eigh + roots) | 1024 | 0.097 | 99.422 | 177.567 |
| eigh of the 8×8 SCM alone (CPU) | 1024 | 0.003 | 3.016 | 177.567 |
| SubspaceNet + Root-MUSIC head (GPU) | 1024 | 0.722 | 739.625 | 268.567 |
| SubViT (GPU) | 1024 | 0.141 | 144.748 | 504.067 |
| DA-MUSIC K=1 (GPU) | 1024 | 0.013 | 13.377 | 889.961 |
| ReconUNet forward only (CPU, 20 threads) | 1024 | 0.195 | 199.258 |  |

Multiply-accumulates per scene (forward hooks on Conv/Linear/GRU layers; counting rule in parentheses):

| model | MMACs per scene |
|---|---|
| ReconUNet | 7.78 |
| SubspaceNet (conv/linear only; eigh + roots excluded) | 3.29 |
| SubViT (linear/conv only; attention products excluded) | 382.40 |
| DA-MUSIC K=1 (GRU formula + linear; MUSIC spectrum excluded) | 0.79 |

Reading for the manuscript: for N = 8 the eigendecomposition of the sample covariance costs microseconds per scene and is far cheaper than the U-Net pass; ReconUNet's cost is dominated by the network and amortises well in batches (see the per-scene column at batch 1024).

Training wall-clock of the released models (single RTX 2000 Ada, from `revision_retrain_20260906/status.log`): reconunet 287min, damusic_k1 157min, damusic_k2 228min, damusic_k3 229min, damusic_k4 230min, subspacenet 1817min, subvit 2633min.

## 9. Baseline implementation notes

* **SubspaceNet** (Shmuel et al.): vendored upstream tree at commit `50f26ec` (= ShlezingerLab/SubspaceNet `f2ef464` + a one-line CUDA→CPU fix, `third_party/patches/`). Adapter `SubspaceNetAdapter(M=4, tau=8, diff_method=root_music)`; the differentiable Root-MUSIC head receives the true per-sample K. Training loss = RMSPE (upstream objective), our shared optimiser schedule.

* **SubViT / DOA-ViT** (Zhou et al.): vendored `third_party/doa_est_master`; published capacity M=8, K_max=4, grid_size=121, angle_range_deg=[-60.0, 60.0], embed_dim=768, depth=6, num_heads=12; training objective grid-BCE on the spatial spectrum; validation/checkpoint criterion = sorted RMSPE; per-sample-K peak picking.

* **DA-MUSIC** (Merkofer et al., ICASSP 2022): upstream `DeepAugmentedMUSIC` from the SubspaceNet tree, unmodified weights/architecture; one model per K. Deviations of the adapter's forward pass from the vendored code (from the adapter docstring):

```text
(all documented, none change parameters)
--------------------------------------------------------------------------------
1. **Batched** MUSIC spectrum.  Upstream loops per sample and per grid angle
   (``pre_MUSIC`` / ``spectrum_calculation``: 2048 × 361 tiny matmuls per
   batch) which is unusable at 2M samples; here it is one ``einsum``.
2. **Hermitian PSD surrogate + robust eigh** (``hermitian_psd=True``,
   default).  Upstream calls ``torch.linalg.eig`` on the *general* learned
   matrix and takes eigenvector columns ``M:`` in LAPACK's unspecified order;
   ``eig``'s backward is also numerically fragile.  We form
   ``R_z = K_x^H K_x + εI`` (SubspaceNet's ``gram_diagonal_overload``) and use
   the project's phase-safe differentiable noise projector — the same
   machinery the SubspaceNet baseline runs on, so both learned baselines share
   one eigendecomposition path.  ``hermitian_psd=False`` gives a batched
   version of the upstream ``eig`` route (noise subspace = eigenvectors with
   the smallest |λ|) for reference.
3. **Time-major GRU input.**  Upstream reshapes ``[B, 2N, T] → [B, T, 2N]`` with
   ``.view`` (a memory reinterpretation, not a transpose), which scrambles the
   time axis.  We use the intended ``transpose`` so the GRU sees one 2N-vector
   per snapshot, as the DA-MUSIC paper describes.
4. Grid steering vectors / angle grid are registered as **buffers** so they
   follow ``model.to(device)`` (upstream keeps them as plain CPU tensors).

Everything else — ``BatchNorm1d(T)``, GRU sizes, ``fc → fc1 → fc2 → fc2 → fc3``
(upstream applies ``fc2`` twice; kept verbatim), ReLUs, Xavier init — is the
vendored module's own parameters and ordering.
```

* **Classical estimators** (`reconunet/evaluation/classical_batched.py`): Bartlett, MVDR and MUSIC on a 1° grid over [−60°, 60°] with three-point parabolic peak refinement and local-maximum selection; Root-MUSIC (roots inside and closest to the unit circle), ESPRIT and Unitary-ESPRIT; all fp64 CPU eigendecompositions for robustness. The same K is supplied to every estimator.


<!-- AUTO-APPEND BELOW: result sections are appended by the pipeline -->

## Ablation — reduced protocol (10 % seeded subset, 40 epochs), Root-MUSIC back end

_Appended 2026-09-20 09:22 UTC from `experiments/runs/ablation_20260920/ablation_results.csv`._

Columns: pooled RMSE (paper eq. 31), median per-scene RMSPE, relative covariance error ||R_hat-R*||_F/||R*||_F, leading-K projector distance ||P_hat-P*||_F, relative eigengap error |gap_hat-gap*|/gap* (all from eigh(R_hat)); eval sets: paper test split (3000 scenes per K), Moderate and Crowded at 0 dB (1000 scenes), and for 01/09 the paper test split rendered with the single fixed imperfection realisation used to train 09.

| variant | eval_set | pooled_rmse_deg | median_rmspe_deg | cov_err_frob | subspace_dist_projF | eigengap_err | n_scenes |
|---|---|---|---|---|---|---|---|
| 01_full | paper_test | 9.224 | 1.282 | 0.357 | 0.534 | 0.128 | 12000 |
| 01_full | moderate_0dB | 4.045 | 0.820 | 0.270 | 0.366 | 0.085 | 1000 |
| 01_full | crowded_0dB | 9.388 | 2.002 | 0.384 | 0.779 | 0.218 | 1000 |
| 01_full | paper_test_fixed_imperf | 9.251 | 1.261 | 0.356 | 0.533 | 0.127 | 12000 |
| 02_rec_only | paper_test | 7.275 | 1.001 | 0.288 | 0.380 | 0.134 | 12000 |
| 02_rec_only | moderate_0dB | 3.686 | 0.611 | 0.211 | 0.231 | 0.085 | 1000 |
| 02_rec_only | crowded_0dB | 7.015 | 1.664 | 0.326 | 0.569 | 0.208 | 1000 |
| 03_rec_proj | paper_test | 7.032 | 1.008 | 0.286 | 0.376 | 0.134 | 12000 |
| 03_rec_proj | moderate_0dB | 3.631 | 0.607 | 0.209 | 0.227 | 0.087 | 1000 |
| 03_rec_proj | crowded_0dB | 6.709 | 1.659 | 0.326 | 0.569 | 0.217 | 1000 |
| 04_no_dom | paper_test | 9.426 | 1.202 | 0.365 | 0.504 | 0.121 | 12000 |
| 04_no_dom | moderate_0dB | 4.145 | 0.741 | 0.266 | 0.338 | 0.071 | 1000 |
| 04_no_dom | crowded_0dB | 9.999 | 2.053 | 0.398 | 0.723 | 0.213 | 1000 |
| 05_no_eig | paper_test | 7.373 | 1.055 | 0.291 | 0.417 | 0.156 | 12000 |
| 05_no_eig | moderate_0dB | 1.845 | 0.628 | 0.207 | 0.255 | 0.130 | 1000 |
| 05_no_eig | crowded_0dB | 7.196 | 1.705 | 0.329 | 0.631 | 0.229 | 1000 |
| 06_single_lag | paper_test | 9.870 | 1.274 | 0.361 | 0.551 | 0.140 | 12000 |
| 06_single_lag | moderate_0dB | 6.076 | 0.789 | 0.270 | 0.367 | 0.100 | 1000 |
| 06_single_lag | crowded_0dB | 9.537 | 2.034 | 0.384 | 0.795 | 0.238 | 1000 |
| 07_relu | paper_test | 8.542 | 1.178 | 0.332 | 0.494 | 0.105 | 12000 |
| 07_relu | moderate_0dB | 3.348 | 0.742 | 0.247 | 0.334 | 0.065 | 1000 |
| 07_relu | crowded_0dB | 8.445 | 1.906 | 0.367 | 0.727 | 0.200 | 1000 |
| 08_no_evd_heads | paper_test | 6.050 | 0.845 | 0.217 | 0.265 | 0.204 | 12000 |
| 08_no_evd_heads | moderate_0dB | 1.892 | 0.520 | 0.144 | 0.153 | 0.121 | 1000 |
| 08_no_evd_heads | crowded_0dB | 4.713 | 1.338 | 0.223 | 0.393 | 0.280 | 1000 |
| 09_fixed_imperf | paper_test | 9.266 | 1.268 | 0.355 | 0.531 | 0.119 | 12000 |
| 09_fixed_imperf | moderate_0dB | 4.538 | 0.811 | 0.266 | 0.364 | 0.066 | 1000 |
| 09_fixed_imperf | crowded_0dB | 9.742 | 1.995 | 0.382 | 0.769 | 0.206 | 1000 |
| 09_fixed_imperf | paper_test_fixed_imperf | 9.235 | 1.228 | 0.351 | 0.524 | 0.119 | 12000 |

## Route comparison — covariance route (eigh of R_hat) vs subspace route (EVD-head eigenvectors)

_Appended 2026-09-20 09:22 UTC from `experiments/runs/ablation_20260920/route_comparison.csv`._

| variant | eval_set | route | batch_size | rmse_deg | median_rmspe_deg | latency_ms_per_scene | orthogonality_residual | n_scenes | note |
|---|---|---|---|---|---|---|---|---|---|
| 01_full | paper_test | covariance | 1024 | 9.224 | 1.282 | 0.135 | 0.000 | 12000 | eigh(R_hat) -> Root-MUSIC |
| 01_full | paper_test | subspace | 1024 | 9.224 | 1.282 | 0.121 | 0.000 | 12000 | EVD-head eigenvectors used directly (no eigh) |
| 01_full | paper_test | covariance | 1 |  |  | 1.628 |  | 256 | batch-1 latency only |
| 01_full | paper_test | subspace | 1 |  |  | 1.472 |  | 256 | batch-1 latency only |
| 01_full | moderate_0dB | covariance | 1024 | 4.045 | 0.820 | 0.075 | 0.000 | 1000 | eigh(R_hat) -> Root-MUSIC |
| 01_full | moderate_0dB | subspace | 1024 | 4.045 | 0.820 | 0.160 | 0.000 | 1000 | EVD-head eigenvectors used directly (no eigh) |
| 01_full | moderate_0dB | covariance | 1 |  |  | 1.613 |  | 256 | batch-1 latency only |
| 01_full | moderate_0dB | subspace | 1 |  |  | 1.493 |  | 256 | batch-1 latency only |
| 08_no_evd_heads | paper_test | covariance | 1024 | 6.050 | 0.845 | 0.153 | 0.000 | 12000 | eigh(R_hat) -> Root-MUSIC |
| 08_no_evd_heads | paper_test | subspace | 1024 | 6.050 | 0.845 | 0.101 | 0.000 | 12000 | no EVD heads: eigenvectors from eigh(R_hat); routes coincide by construction |
| 08_no_evd_heads | paper_test | covariance | 1 |  |  | 1.166 |  | 256 | batch-1 latency only |
| 08_no_evd_heads | paper_test | subspace | 1 |  |  | 1.058 |  | 256 | batch-1 latency only |
| 08_no_evd_heads | moderate_0dB | covariance | 1024 | 1.892 | 0.520 | 0.060 | 0.000 | 1000 | eigh(R_hat) -> Root-MUSIC |
| 08_no_evd_heads | moderate_0dB | subspace | 1024 | 1.892 | 0.520 | 0.063 | 0.000 | 1000 | no EVD heads: eigenvectors from eigh(R_hat); routes coincide by construction |
| 08_no_evd_heads | moderate_0dB | covariance | 1 |  |  | 1.169 |  | 256 | batch-1 latency only |
| 08_no_evd_heads | moderate_0dB | subspace | 1 |  |  | 1.067 |  | 256 | batch-1 latency only |

## DA-MUSIC K=2,3,4 continuation to 600 epochs

_Appended 2026-09-20 19:26 UTC._

Exact continuation of the K = 2, 3, 4 runs from their epoch-300 `last.pt` (model, Adam moments, scheduler, best/stale restored), cap 600 epochs, patience 25; checkpoints under `experiments/runs/damusic_paper/k<K>_v2/`.

| run | epochs_run | best_epoch(val loss) | val_rmspe_at_best(deg) | last_val_rmspe(deg) | last_lr | min_val_rmspe(deg) | v1 (300-epoch cap) val RMSPE at best |
|---|---|---|---|---|---|---|---|
| k2_v2 | 491 | 466 | 4.983 | 4.975 | 1.0e-06 | 4.973 | 5.12 |
| k3_v2 | 600 | 595 | 4.39 | 4.388 | 1.0e-06 | 4.388 | 4.566 |
| k4_v2 | 582 | 557 | 3.889 | 3.888 | 1.0e-06 | 3.888 | 4.132 |

## Paper test split with the DA-MUSIC v2 ensemble (pooled RMSE / median per K, all methods)

_Appended 2026-09-20 19:27 UTC from `experiments/runs/eval_coherent_20260910/paper_testset_v2/paper_testset_by_K.csv`._

| K | n | R-MUSIC_med | R-MUSIC_mean | ESPRIT_med | ESPRIT_mean | SubspaceNet_med | SubspaceNet_mean | SubViT_med | SubViT_mean | DA-MUSIC_med | DA-MUSIC_mean | ReconUNet_med | ReconUNet_mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 3000 | 1.142 | 26.366 | 4.938 | 23.446 | 1.416 | 13.616 | 0.527 | 13.867 | 4.087 | 16.250 | 0.749 | 12.494 |
| 2 | 3000 | 0.792 | 14.206 | 2.602 | 12.490 | 1.435 | 8.168 | 0.531 | 15.236 | 2.903 | 8.312 | 0.838 | 6.044 |
| 3 | 3000 | 0.766 | 9.708 | 1.819 | 8.788 | 1.737 | 7.568 | 0.594 | 15.917 | 3.150 | 6.380 | 0.983 | 3.728 |
| 4 | 3000 | 0.673 | 6.409 | 1.146 | 5.411 | 2.063 | 7.244 | 0.671 | 14.309 | 3.051 | 5.043 | 1.119 | 3.999 |
| all | 12000 | 0.799 | 12.433 | 2.152 | 11.002 | 1.703 | 8.370 | 0.578 | 14.952 | 3.175 | 7.913 | 0.943 | 5.786 |

## Table II (mild) at 0 dB with the DA-MUSIC v2 ensemble

_Appended 2026-09-20 19:27 UTC from `experiments/runs/eval_coherent_20260910/table2_mild_v2/table2_full.csv` (snr_db=0.0)._

| scenario | snr_db | method | rmse_deg | n |
|---|---|---|---|---|
| basic | 0.000 | Bartlett | 3.964 | 1000 |
| basic | 0.000 | MVDR | 3.561 | 1000 |
| basic | 0.000 | MUSIC | 3.972 | 1000 |
| basic | 0.000 | Root-MUSIC | 0.251 | 1000 |
| basic | 0.000 | ESPRIT | 0.332 | 1000 |
| basic | 0.000 | Unitary-ESPRIT | 0.406 | 1000 |
| basic | 0.000 | ReconUNet+Root-MUSIC | 0.373 | 1000 |
| basic | 0.000 | ReconUNet+MUSIC | 0.422 | 1000 |
| basic | 0.000 | ReconUNet+ESPRIT | 0.398 | 1000 |
| basic | 0.000 | ReconUNet+Unitary-ESPRIT | 0.408 | 1000 |
| basic | 0.000 | SubspaceNet | 0.540 | 1000 |
| basic | 0.000 | SubViT | 0.380 | 1000 |
| basic | 0.000 | DA-MUSIC | 2.134 | 1000 |
| basic | 0.000 | CRLB | 0.120 | 1000 |
| moderate | 0.000 | Bartlett | 12.917 | 1000 |
| moderate | 0.000 | MVDR | 11.528 | 1000 |
| moderate | 0.000 | MUSIC | 10.391 | 1000 |
| moderate | 0.000 | Root-MUSIC | 5.915 | 1000 |
| moderate | 0.000 | ESPRIT | 5.938 | 1000 |
| moderate | 0.000 | Unitary-ESPRIT | 5.691 | 1000 |
| moderate | 0.000 | ReconUNet+Root-MUSIC | 1.790 | 1000 |
| moderate | 0.000 | ReconUNet+MUSIC | 4.033 | 1000 |
| moderate | 0.000 | ReconUNet+ESPRIT | 1.776 | 1000 |
| moderate | 0.000 | ReconUNet+Unitary-ESPRIT | 2.630 | 1000 |
| moderate | 0.000 | SubspaceNet | 2.918 | 1000 |
| moderate | 0.000 | SubViT | 8.724 | 1000 |
| moderate | 0.000 | DA-MUSIC | 3.600 | 1000 |
| moderate | 0.000 | CRLB | 0.137 | 1000 |
| advanced1_ood | 0.000 | Bartlett | 28.718 | 1000 |
| advanced1_ood | 0.000 | MVDR | 25.060 | 1000 |
| advanced1_ood | 0.000 | MUSIC | 29.926 | 1000 |
| advanced1_ood | 0.000 | Root-MUSIC | 40.950 | 1000 |
| advanced1_ood | 0.000 | ESPRIT | 35.720 | 1000 |
| advanced1_ood | 0.000 | Unitary-ESPRIT | 24.296 | 1000 |
| advanced1_ood | 0.000 | ReconUNet+Root-MUSIC | 22.889 | 1000 |
| advanced1_ood | 0.000 | ReconUNet+MUSIC | 22.889 | 1000 |
| advanced1_ood | 0.000 | ReconUNet+ESPRIT | 24.196 | 1000 |
| advanced1_ood | 0.000 | ReconUNet+Unitary-ESPRIT | 21.327 | 1000 |
| advanced1_ood | 0.000 | SubspaceNet | 21.495 | 1000 |
| advanced1_ood | 0.000 | SubViT | 23.599 | 1000 |
| advanced1_ood | 0.000 | DA-MUSIC | 24.276 | 1000 |
| advanced1_ood | 0.000 | CRLB | 0.120 | 1000 |
| advanced2_crowded | 0.000 | Bartlett | 21.003 | 1000 |
| advanced2_crowded | 0.000 | MVDR | 15.723 | 1000 |
| advanced2_crowded | 0.000 | MUSIC | 12.561 | 1000 |
| advanced2_crowded | 0.000 | Root-MUSIC | 8.871 | 1000 |
| advanced2_crowded | 0.000 | ESPRIT | 8.145 | 1000 |
| advanced2_crowded | 0.000 | Unitary-ESPRIT | 8.044 | 1000 |
| advanced2_crowded | 0.000 | ReconUNet+Root-MUSIC | 4.804 | 1000 |
| advanced2_crowded | 0.000 | ReconUNet+MUSIC | 8.162 | 1000 |
| advanced2_crowded | 0.000 | ReconUNet+ESPRIT | 4.613 | 1000 |
| advanced2_crowded | 0.000 | ReconUNet+Unitary-ESPRIT | 4.005 | 1000 |
| advanced2_crowded | 0.000 | SubspaceNet | 7.912 | 1000 |
| advanced2_crowded | 0.000 | SubViT | 16.321 | 1000 |
| advanced2_crowded | 0.000 | DA-MUSIC | 6.652 | 1000 |
| advanced2_crowded | 0.000 | CRLB | 0.210 | 1000 |

## Table II (harsh) at 0 dB with the DA-MUSIC v2 ensemble

_Appended 2026-09-20 19:27 UTC from `experiments/runs/eval_coherent_20260910/table2_harsh_v2/table2_full.csv` (snr_db=0.0)._

| scenario | snr_db | method | rmse_deg | n |
|---|---|---|---|---|
| basic | 0.000 | Bartlett | 6.140 | 1000 |
| basic | 0.000 | MVDR | 5.906 | 1000 |
| basic | 0.000 | MUSIC | 5.734 | 1000 |
| basic | 0.000 | Root-MUSIC | 1.149 | 1000 |
| basic | 0.000 | ESPRIT | 1.579 | 1000 |
| basic | 0.000 | Unitary-ESPRIT | 3.645 | 1000 |
| basic | 0.000 | ReconUNet+Root-MUSIC | 1.203 | 1000 |
| basic | 0.000 | ReconUNet+MUSIC | 1.211 | 1000 |
| basic | 0.000 | ReconUNet+ESPRIT | 1.210 | 1000 |
| basic | 0.000 | ReconUNet+Unitary-ESPRIT | 1.216 | 1000 |
| basic | 0.000 | SubspaceNet | 1.282 | 1000 |
| basic | 0.000 | SubViT | 1.216 | 1000 |
| basic | 0.000 | DA-MUSIC | 2.744 | 1000 |
| basic | 0.000 | CRLB | 0.120 | 1000 |
| moderate | 0.000 | Bartlett | 14.500 | 1000 |
| moderate | 0.000 | MVDR | 12.275 | 1000 |
| moderate | 0.000 | MUSIC | 11.699 | 1000 |
| moderate | 0.000 | Root-MUSIC | 6.827 | 1000 |
| moderate | 0.000 | ESPRIT | 6.201 | 1000 |
| moderate | 0.000 | Unitary-ESPRIT | 6.760 | 1000 |
| moderate | 0.000 | ReconUNet+Root-MUSIC | 2.357 | 1000 |
| moderate | 0.000 | ReconUNet+MUSIC | 5.402 | 1000 |
| moderate | 0.000 | ReconUNet+ESPRIT | 2.194 | 1000 |
| moderate | 0.000 | ReconUNet+Unitary-ESPRIT | 3.018 | 1000 |
| moderate | 0.000 | SubspaceNet | 4.734 | 1000 |
| moderate | 0.000 | SubViT | 14.851 | 1000 |
| moderate | 0.000 | DA-MUSIC | 4.585 | 1000 |
| moderate | 0.000 | CRLB | 0.137 | 1000 |
| advanced1_ood | 0.000 | Bartlett | 30.220 | 1000 |
| advanced1_ood | 0.000 | MVDR | 27.974 | 1000 |
| advanced1_ood | 0.000 | MUSIC | 30.670 | 1000 |
| advanced1_ood | 0.000 | Root-MUSIC | 41.958 | 1000 |
| advanced1_ood | 0.000 | ESPRIT | 38.633 | 1000 |
| advanced1_ood | 0.000 | Unitary-ESPRIT | 25.550 | 1000 |
| advanced1_ood | 0.000 | ReconUNet+Root-MUSIC | 25.307 | 1000 |
| advanced1_ood | 0.000 | ReconUNet+MUSIC | 25.298 | 1000 |
| advanced1_ood | 0.000 | ReconUNet+ESPRIT | 26.989 | 1000 |
| advanced1_ood | 0.000 | ReconUNet+Unitary-ESPRIT | 23.794 | 1000 |
| advanced1_ood | 0.000 | SubspaceNet | 23.245 | 1000 |
| advanced1_ood | 0.000 | SubViT | 26.097 | 1000 |
| advanced1_ood | 0.000 | DA-MUSIC | 25.522 | 1000 |
| advanced1_ood | 0.000 | CRLB | 0.120 | 1000 |
| advanced2_crowded | 0.000 | Bartlett | 21.828 | 1000 |
| advanced2_crowded | 0.000 | MVDR | 18.253 | 1000 |
| advanced2_crowded | 0.000 | MUSIC | 15.558 | 1000 |
| advanced2_crowded | 0.000 | Root-MUSIC | 9.342 | 1000 |
| advanced2_crowded | 0.000 | ESPRIT | 8.390 | 1000 |
| advanced2_crowded | 0.000 | Unitary-ESPRIT | 9.072 | 1000 |
| advanced2_crowded | 0.000 | ReconUNet+Root-MUSIC | 6.136 | 1000 |
| advanced2_crowded | 0.000 | ReconUNet+MUSIC | 9.151 | 1000 |
| advanced2_crowded | 0.000 | ReconUNet+ESPRIT | 5.410 | 1000 |
| advanced2_crowded | 0.000 | ReconUNet+Unitary-ESPRIT | 4.931 | 1000 |
| advanced2_crowded | 0.000 | SubspaceNet | 8.823 | 1000 |
| advanced2_crowded | 0.000 | SubViT | 18.420 | 1000 |
| advanced2_crowded | 0.000 | DA-MUSIC | 7.503 | 1000 |
| advanced2_crowded | 0.000 | CRLB | 0.210 | 1000 |

## MUSIC verification

_Appended 2026-09-20 19:28 UTC._

Basic scenario (K = 1, no multipath, mild imperfections), 1000 scenes per SNR. MUSIC uses the paper's 1° scan grid on [-60°, 60°]; 'refined' adds the three-point parabolic peak refinement; Root-MUSIC is gridless. A spurious peak is a scene whose selected maximum is more than 1°/3°/5° from the true angle.

| SNR (dB) | estimator | median abs err (°) | pooled RMSE (°) | mean abs err (°) | frac > 1° | frac > 3° | frac > 5° | n |
|---|---|---|---|---|---|---|---|---|
| 20 | MUSIC-grid | 0.269 | 0.371 | 0.305 | 0.0020 | 0.0000 | 0.0000 | 1000 |
| 20 | MUSIC-refined | 0.192 | 5.302 | 0.561 | 0.0060 | 0.0060 | 0.0060 | 1000 |
| 20 | Root-MUSIC | 0.147 | 0.229 | 0.182 | 0.0000 | 0.0000 | 0.0000 | 1000 |
| 0 | MUSIC-grid | 0.270 | 0.380 | 0.310 | 0.0060 | 0.0000 | 0.0000 | 1000 |
| 0 | MUSIC-refined | 0.188 | 3.972 | 0.450 | 0.0060 | 0.0050 | 0.0050 | 1000 |
| 0 | Root-MUSIC | 0.163 | 0.251 | 0.198 | 0.0000 | 0.0000 | 0.0000 | 1000 |

Reading: at 20 dB the refined grid-MUSIC median error is 0.192° against 0.147° for Root-MUSIC, i.e. the estimator itself is fine; the pooled RMSE gap (5.302° vs 0.229°) is driven by the 0.60 % of scenes whose selected peak is spurious (> 3°), which RMSE squares. Without refinement the grid alone contributes a 0.269° median quantisation error.

Data: `experiments/runs/sweeps_20260920/music_verification.csv`.

## FBSS Root-MUSIC baseline at 0 dB (Moderate, Crowded; mild)

_Appended 2026-09-20 19:28 UTC from `experiments/runs/sweeps_20260920/fbss_baseline.csv` (snr_db=0.0)._

| scenario | snr_db | method | rmse_deg | median_rmspe_deg | n |
|---|---|---|---|---|---|
| moderate | 0.000 | Root-MUSIC | 5.915 | 0.486 | 1000 |
| moderate | 0.000 | ReconUNet+Root-MUSIC | 1.790 | 0.616 | 1000 |
| moderate | 0.000 | FBSS(L=5)+Root-MUSIC | 8.536 | 0.888 | 1000 |
| moderate | 0.000 | FBSS(L=6)+Root-MUSIC | 6.450 | 0.664 | 1000 |
| moderate | 0.000 | FBSS(L=7)+Root-MUSIC | 6.312 | 0.609 | 1000 |
| advanced2_crowded | 0.000 | Root-MUSIC | 8.871 | 1.291 | 1000 |
| advanced2_crowded | 0.000 | ReconUNet+Root-MUSIC | 4.804 | 1.430 | 1000 |
| advanced2_crowded | 0.000 | FBSS(L=5)+Root-MUSIC | 12.420 | 4.697 | 1000 |
| advanced2_crowded | 0.000 | FBSS(L=6)+Root-MUSIC | 11.304 | 3.129 | 1000 |
| advanced2_crowded | 0.000 | FBSS(L=7)+Root-MUSIC | 9.610 | 2.225 | 1000 |

## Snapshot sweep, Moderate at 0 dB

_Appended 2026-09-20 19:28 UTC from `experiments/runs/sweeps_20260920/snapshot_sweep.csv`._

| T | sampling | method | rmse_deg | median_rmspe_deg | n |
|---|---|---|---|---|---|
| 8 | window | Root-MUSIC | 27.038 | 13.711 | 1000 |
| 8 | window | ReconUNet |  |  | 0 |
| 8 | window | SubspaceNet |  |  | 0 |
| 8 | both | CRLB | 1.661 |  | 1000 |
| 8 | decimated | Root-MUSIC | 8.302 | 1.260 | 1000 |
| 8 | decimated | ReconUNet |  |  | 0 |
| 8 | decimated | SubspaceNet |  |  | 0 |
| 16 | window | Root-MUSIC | 20.592 | 2.458 | 1000 |
| 16 | window | ReconUNet | 25.941 | 17.300 | 1000 |
| 16 | window | SubspaceNet | 21.481 | 9.916 | 1000 |
| 16 | both | CRLB | 0.831 |  | 1000 |
| 16 | decimated | Root-MUSIC | 7.718 | 0.906 | 1000 |
| 16 | decimated | ReconUNet | 34.192 | 27.631 | 1000 |
| 16 | decimated | SubspaceNet | 23.058 | 10.677 | 1000 |
| 32 | window | Root-MUSIC | 11.931 | 1.161 | 1000 |
| 32 | window | ReconUNet | 20.403 | 3.149 | 1000 |
| 32 | window | SubspaceNet | 17.576 | 3.970 | 1000 |
| 32 | both | CRLB | 0.415 |  | 1000 |
| 32 | decimated | Root-MUSIC | 5.944 | 0.716 | 1000 |
| 32 | decimated | ReconUNet | 29.470 | 23.286 | 1000 |
| 32 | decimated | SubspaceNet | 15.684 | 3.820 | 1000 |
| 64 | window | Root-MUSIC | 7.019 | 0.752 | 1000 |
| 64 | window | ReconUNet | 12.653 | 1.289 | 1000 |
| 64 | window | SubspaceNet | 12.238 | 2.265 | 1000 |
| 64 | both | CRLB | 0.208 |  | 1000 |
| 64 | decimated | Root-MUSIC | 5.781 | 0.612 | 1000 |
| 64 | decimated | ReconUNet | 26.482 | 21.263 | 1000 |
| 64 | decimated | SubspaceNet | 10.132 | 2.096 | 1000 |
| 128 | window | Root-MUSIC | 5.713 | 0.572 | 1000 |
| 128 | window | ReconUNet | 4.596 | 0.881 | 1000 |
| 128 | window | SubspaceNet | 6.645 | 1.517 | 1000 |
| 128 | both | CRLB | 0.104 |  | 1000 |
| 128 | decimated | Root-MUSIC | 5.546 | 0.560 | 1000 |
| 128 | decimated | ReconUNet | 16.356 | 2.572 | 1000 |
| 128 | decimated | SubspaceNet | 7.117 | 1.397 | 1000 |
| 256 | window | Root-MUSIC | 5.875 | 0.503 | 1000 |
| 256 | window | ReconUNet | 1.871 | 0.699 | 1000 |
| 256 | window | SubspaceNet | 3.629 | 1.189 | 1000 |
| 256 | both | CRLB | 0.052 |  | 1000 |
| 256 | decimated | Root-MUSIC | 5.908 | 0.513 | 1000 |
| 256 | decimated | ReconUNet | 1.843 | 0.704 | 1000 |
| 256 | decimated | SubspaceNet | 3.818 | 1.029 | 1000 |
| 512 | window | Root-MUSIC | 5.915 | 0.486 | 1000 |
| 512 | window | ReconUNet | 1.790 | 0.616 | 1000 |
| 512 | window | SubspaceNet | 2.918 | 1.035 | 1000 |
| 512 | both | CRLB | 0.026 |  | 1000 |
| 512 | decimated | Root-MUSIC | 5.915 | 0.486 | 1000 |
| 512 | decimated | ReconUNet | 1.790 | 0.616 | 1000 |
| 512 | decimated | SubspaceNet | 2.918 | 1.035 | 1000 |

## Separation sweep, K=2 at 0 and −5 dB (RMSE, median, resolution probability)

_Appended 2026-09-20 19:28 UTC from `experiments/runs/sweeps_20260920/separation_sweep.csv`._

| sep_deg | snr_db | method | rmse_deg | median_rmspe_deg | resolution_prob | n |
|---|---|---|---|---|---|---|
| 2.000 | 0.000 | Root-MUSIC | 29.201 | 18.528 | 0.068 | 1000 |
| 2.000 | 0.000 | ESPRIT | 13.417 | 2.394 | 0.128 | 1000 |
| 2.000 | 0.000 | ReconUNet | 34.761 | 20.303 | 0.000 | 1000 |
| 2.000 | 0.000 | SubspaceNet | 25.365 | 18.833 | 0.001 | 1000 |
| 2.000 | 0.000 | DA-MUSIC | 9.018 | 7.137 | 0.001 | 1000 |
| 2.000 | 0.000 | CRLB | 4.851 |  |  | 1000 |
| 2.000 | -5.000 | Root-MUSIC | 34.382 | 25.212 | 0.001 | 1000 |
| 2.000 | -5.000 | ESPRIT | 32.171 | 21.999 | 0.014 | 1000 |
| 2.000 | -5.000 | ReconUNet | 35.292 | 20.248 | 0.001 | 1000 |
| 2.000 | -5.000 | SubspaceNet | 22.815 | 20.053 | 0.004 | 1000 |
| 2.000 | -5.000 | DA-MUSIC | 8.983 | 7.181 | 0.001 | 1000 |
| 2.000 | -5.000 | CRLB | 40.267 |  |  | 1000 |
| 4.000 | 0.000 | Root-MUSIC | 0.911 | 0.712 | 0.930 | 1000 |
| 4.000 | 0.000 | ESPRIT | 1.123 | 0.860 | 0.869 | 1000 |
| 4.000 | 0.000 | ReconUNet | 32.156 | 18.383 | 0.041 | 1000 |
| 4.000 | 0.000 | SubspaceNet | 23.546 | 16.580 | 0.018 | 1000 |
| 4.000 | 0.000 | DA-MUSIC | 7.681 | 6.013 | 0.007 | 1000 |
| 4.000 | 0.000 | CRLB | 0.534 |  |  | 1000 |
| 4.000 | -5.000 | Root-MUSIC | 24.308 | 2.481 | 0.316 | 1000 |
| 4.000 | -5.000 | ESPRIT | 11.022 | 1.941 | 0.361 | 1000 |
| 4.000 | -5.000 | ReconUNet | 30.876 | 17.580 | 0.040 | 1000 |
| 4.000 | -5.000 | SubspaceNet | 21.620 | 19.078 | 0.019 | 1000 |
| 4.000 | -5.000 | DA-MUSIC | 7.661 | 6.066 | 0.006 | 1000 |
| 4.000 | -5.000 | CRLB | 3.313 |  |  | 1000 |
| 6.000 | 0.000 | Root-MUSIC | 0.557 | 0.443 | 1.000 | 1000 |
| 6.000 | 0.000 | ESPRIT | 0.688 | 0.534 | 0.999 | 1000 |
| 6.000 | 0.000 | ReconUNet | 15.997 | 2.598 | 0.441 | 1000 |
| 6.000 | 0.000 | SubspaceNet | 20.214 | 12.065 | 0.086 | 1000 |
| 6.000 | 0.000 | DA-MUSIC | 6.046 | 4.984 | 0.042 | 1000 |
| 6.000 | 0.000 | CRLB | 0.180 |  |  | 1000 |
| 6.000 | -5.000 | Root-MUSIC | 5.533 | 0.834 | 0.952 | 1000 |
| 6.000 | -5.000 | ESPRIT | 1.480 | 1.020 | 0.926 | 1000 |
| 6.000 | -5.000 | ReconUNet | 15.616 | 2.629 | 0.446 | 1000 |
| 6.000 | -5.000 | SubspaceNet | 18.057 | 12.077 | 0.104 | 1000 |
| 6.000 | -5.000 | DA-MUSIC | 6.157 | 5.099 | 0.043 | 1000 |
| 6.000 | -5.000 | CRLB | 0.916 |  |  | 1000 |
| 8.000 | 0.000 | Root-MUSIC | 0.421 | 0.332 | 1.000 | 1000 |
| 8.000 | 0.000 | ESPRIT | 0.569 | 0.434 | 1.000 | 1000 |
| 8.000 | 0.000 | ReconUNet | 2.825 | 1.323 | 0.951 | 1000 |
| 8.000 | 0.000 | SubspaceNet | 14.374 | 4.025 | 0.393 | 1000 |
| 8.000 | 0.000 | DA-MUSIC | 4.624 | 3.974 | 0.275 | 1000 |
| 8.000 | 0.000 | CRLB | 0.090 |  |  | 1000 |
| 8.000 | -5.000 | Root-MUSIC | 0.731 | 0.530 | 1.000 | 1000 |
| 8.000 | -5.000 | ESPRIT | 0.926 | 0.700 | 0.998 | 1000 |
| 8.000 | -5.000 | ReconUNet | 3.374 | 1.377 | 0.942 | 1000 |
| 8.000 | -5.000 | SubspaceNet | 13.824 | 3.771 | 0.417 | 1000 |
| 8.000 | -5.000 | DA-MUSIC | 4.874 | 4.149 | 0.245 | 1000 |
| 8.000 | -5.000 | CRLB | 0.403 |  |  | 1000 |
| 10.000 | 0.000 | Root-MUSIC | 0.345 | 0.261 | 1.000 | 1000 |
| 10.000 | 0.000 | ESPRIT | 0.504 | 0.372 | 1.000 | 1000 |
| 10.000 | 0.000 | ReconUNet | 1.159 | 0.941 | 1.000 | 1000 |
| 10.000 | 0.000 | SubspaceNet | 11.704 | 2.235 | 0.705 | 1000 |
| 10.000 | 0.000 | DA-MUSIC | 3.301 | 2.896 | 0.783 | 1000 |
| 10.000 | 0.000 | CRLB | 0.055 |  |  | 1000 |
| 10.000 | -5.000 | Root-MUSIC | 0.534 | 0.404 | 1.000 | 1000 |
| 10.000 | -5.000 | ESPRIT | 0.735 | 0.547 | 1.000 | 1000 |
| 10.000 | -5.000 | ReconUNet | 1.199 | 0.942 | 0.999 | 1000 |
| 10.000 | -5.000 | SubspaceNet | 9.870 | 2.207 | 0.737 | 1000 |
| 10.000 | -5.000 | DA-MUSIC | 3.519 | 3.120 | 0.729 | 1000 |
| 10.000 | -5.000 | CRLB | 0.232 |  |  | 1000 |
| 15.000 | 0.000 | Root-MUSIC | 0.267 | 0.205 | 1.000 | 1000 |
| 15.000 | 0.000 | ESPRIT | 0.381 | 0.267 | 1.000 | 1000 |
| 15.000 | 0.000 | ReconUNet | 0.675 | 0.532 | 1.000 | 1000 |
| 15.000 | 0.000 | SubspaceNet | 5.986 | 1.176 | 0.920 | 1000 |
| 15.000 | 0.000 | DA-MUSIC | 1.881 | 1.468 | 0.998 | 1000 |
| 15.000 | 0.000 | CRLB | 0.023 |  |  | 1000 |
| 15.000 | -5.000 | Root-MUSIC | 0.378 | 0.289 | 1.000 | 1000 |
| 15.000 | -5.000 | ESPRIT | 0.507 | 0.374 | 1.000 | 1000 |
| 15.000 | -5.000 | ReconUNet | 0.713 | 0.556 | 1.000 | 1000 |
| 15.000 | -5.000 | SubspaceNet | 7.453 | 1.207 | 0.956 | 1000 |
| 15.000 | -5.000 | DA-MUSIC | 2.020 | 1.636 | 0.998 | 1000 |
| 15.000 | -5.000 | CRLB | 0.092 |  |  | 1000 |

## Bootstrap 95 % CIs, Table II mild at 0 dB

_Appended 2026-09-20 19:28 UTC from `experiments/runs/eval_coherent_20260910/table2_mild_v2/table2_mild_v2_ci.csv` (snr_db=0.0)._

_(empty)_

## R2 (2026-09-30) — ReconUNet-C training (full scale)

_Appended 2026-09-30 21:45 UTC from `experiments/runs/revision_r2_20260930/reconunet_c_training.csv`._

ReconUNet-C = CovarianceOnlyReconstructionUNet (no eigen heads) trained with L_rec only on the full 2 M / 50 k corpus with the paper schedule (configs/train/reconunet_c_paper.yaml); checkpoint = minimum validation loss. Validation losses of the two rows are NOT comparable (L_rec alone vs the 4-term composite loss); validation RMSPE is (Root-MUSIC on R_hat, true K). Wall-clock from the R2 status log (single RTX 2000 Ada).

| model | epochs_run | best_epoch | best_val_loss | val_rmspe_at_best_deg | min_val_rmspe_deg | last_lr | early_stopped | wall_clock_min | min_per_epoch |
|---|---|---|---|---|---|---|---|---|---|
| ReconUNet-C (R2, L_rec only) | 207 | 182 | 0.070 | 1.330 | 1.323 | 0.000 | True | 531 | 2.570 |
| ReconUNet (R1 released, composite loss) | 109 | 84 | 0.424 | 2.174 | 2.018 | 0.000 | True | 287 | 2.630 |

## R2 (2026-09-30) — Paper test split by K, ReconUNet-C added (median RMSPE _med / pooled RMSE _mean, deg)

_Appended 2026-09-30 21:45 UTC from `experiments/runs/eval_r2_20260930/paper_testset_r2/paper_testset_by_K.csv`._

Same scenes and code path as the R1 table (DA-MUSIC v2 ensemble); ReconUNet-C uses the same Root-MUSIC back end with the true per-scene K.

| K | n | R-MUSIC_med | R-MUSIC_mean | ESPRIT_med | ESPRIT_mean | SubspaceNet_med | SubspaceNet_mean | SubViT_med | SubViT_mean | DA-MUSIC_med | DA-MUSIC_mean | ReconUNet_med | ReconUNet_mean | ReconUNet-C_med | ReconUNet-C_mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 3000 | 1.142 | 26.366 | 4.938 | 23.446 | 1.416 | 13.616 | 0.527 | 13.867 | 4.087 | 16.250 | 0.749 | 12.494 | 0.490 | 10.487 |
| 2 | 3000 | 0.792 | 14.206 | 2.602 | 12.490 | 1.435 | 8.168 | 0.531 | 15.236 | 2.903 | 8.312 | 0.838 | 6.044 | 0.518 | 4.170 |
| 3 | 3000 | 0.766 | 9.708 | 1.819 | 8.788 | 1.737 | 7.568 | 0.594 | 15.917 | 3.150 | 6.380 | 0.983 | 3.728 | 0.598 | 2.354 |
| 4 | 3000 | 0.673 | 6.409 | 1.146 | 5.411 | 2.063 | 7.244 | 0.671 | 14.309 | 3.051 | 5.043 | 1.119 | 3.999 | 0.664 | 2.106 |
| all | 12000 | 0.799 | 12.433 | 2.152 | 11.002 | 1.703 | 8.370 | 0.578 | 14.952 | 3.175 | 7.913 | 0.943 | 5.786 | 0.581 | 4.232 |

## R2 (2026-09-30) — Paper test split, bootstrap 95 % CIs

_Appended 2026-09-30 21:45 UTC from `experiments/runs/eval_r2_20260930/paper_testset_r2/paper_testset_r2_ci.csv`._

| scenario | method | n_scenes | rmse_deg | rmse_ci_lo | rmse_ci_hi | median_rmspe_deg | median_ci_lo | median_ci_hi |
|---|---|---|---|---|---|---|---|---|
| K=1 | R-MUSIC | 3000 | 26.366 | 24.569 | 28.002 | 1.142 | 1.058 | 1.230 |
| K=1 | ESPRIT | 3000 | 23.446 | 21.775 | 25.196 | 4.938 | 4.631 | 5.318 |
| K=1 | SubspaceNet | 3000 | 13.616 | 12.170 | 15.069 | 1.416 | 1.345 | 1.484 |
| K=1 | SubViT | 3000 | 13.867 | 12.414 | 15.362 | 0.527 | 0.510 | 0.557 |
| K=1 | DA-MUSIC | 3000 | 16.250 | 14.981 | 17.613 | 4.087 | 3.846 | 4.333 |
| K=1 | ReconUNet | 3000 | 12.494 | 10.962 | 14.008 | 0.749 | 0.696 | 0.793 |
| K=1 | ReconUNet-C | 3000 | 10.487 | 8.905 | 12.038 | 0.490 | 0.466 | 0.522 |
| K=2 | R-MUSIC | 3000 | 14.206 | 13.300 | 15.045 | 0.792 | 0.738 | 0.848 |
| K=2 | ESPRIT | 3000 | 12.490 | 11.382 | 13.428 | 2.602 | 2.423 | 2.767 |
| K=2 | SubspaceNet | 3000 | 8.168 | 7.437 | 8.913 | 1.435 | 1.385 | 1.489 |
| K=2 | SubViT | 3000 | 15.236 | 14.296 | 16.167 | 0.531 | 0.507 | 0.549 |
| K=2 | DA-MUSIC | 3000 | 8.312 | 7.697 | 8.927 | 2.903 | 2.784 | 2.986 |
| K=2 | ReconUNet | 3000 | 6.044 | 5.118 | 6.952 | 0.838 | 0.804 | 0.866 |
| K=2 | ReconUNet-C | 3000 | 4.170 | 3.206 | 5.051 | 0.518 | 0.498 | 0.539 |
| K=3 | R-MUSIC | 3000 | 9.708 | 9.054 | 10.359 | 0.766 | 0.716 | 0.809 |
| K=3 | ESPRIT | 3000 | 8.788 | 8.093 | 9.490 | 1.819 | 1.713 | 1.947 |
| K=3 | SubspaceNet | 3000 | 7.568 | 7.035 | 8.035 | 1.737 | 1.687 | 1.792 |
| K=3 | SubViT | 3000 | 15.917 | 15.222 | 16.558 | 0.594 | 0.571 | 0.613 |
| K=3 | DA-MUSIC | 3000 | 6.380 | 6.023 | 6.776 | 3.150 | 3.053 | 3.228 |
| K=3 | ReconUNet | 3000 | 3.728 | 3.185 | 4.260 | 0.983 | 0.948 | 1.012 |
| K=3 | ReconUNet-C | 3000 | 2.354 | 1.782 | 2.895 | 0.598 | 0.579 | 0.616 |
| K=4 | R-MUSIC | 3000 | 6.409 | 5.941 | 6.892 | 0.673 | 0.642 | 0.701 |
| K=4 | ESPRIT | 3000 | 5.411 | 4.907 | 5.896 | 1.146 | 1.087 | 1.209 |
| K=4 | SubspaceNet | 3000 | 7.244 | 6.867 | 7.627 | 2.063 | 2.002 | 2.107 |
| K=4 | SubViT | 3000 | 14.309 | 13.829 | 14.796 | 0.671 | 0.643 | 0.703 |
| K=4 | DA-MUSIC | 3000 | 5.043 | 4.821 | 5.272 | 3.051 | 2.983 | 3.119 |
| K=4 | ReconUNet | 3000 | 3.999 | 3.527 | 4.475 | 1.119 | 1.087 | 1.142 |
| K=4 | ReconUNet-C | 3000 | 2.106 | 1.674 | 2.494 | 0.664 | 0.644 | 0.680 |
| all K | R-MUSIC | 12000 | 12.433 | 11.956 | 12.882 | 0.799 | 0.777 | 0.825 |
| all K | ESPRIT | 12000 | 11.002 | 10.535 | 11.479 | 2.152 | 2.071 | 2.236 |
| all K | SubspaceNet | 12000 | 8.370 | 8.024 | 8.699 | 1.703 | 1.670 | 1.738 |
| all K | SubViT | 12000 | 14.952 | 14.586 | 15.302 | 0.578 | 0.569 | 0.588 |
| all K | DA-MUSIC | 12000 | 7.913 | 7.582 | 8.246 | 3.175 | 3.122 | 3.224 |
| all K | ReconUNet | 12000 | 5.786 | 5.346 | 6.184 | 0.943 | 0.925 | 0.961 |
| all K | ReconUNet-C | 12000 | 4.232 | 3.758 | 4.673 | 0.581 | 0.570 | 0.590 |

## R2 (2026-09-30) — Scenario sweep (pooled RMSE vs SNR, deg), ReconUNet-C added

_Appended 2026-09-30 21:45 UTC from `experiments/runs/eval_r2_20260930/scenario_sweep_r2/scenario_sweep.csv`._

| scenario | snr_db | n | crlb_deg | Root-MUSIC | SubspaceNet | SubViT | DA-MUSIC | ReconUNet | ReconUNet-C |
|---|---|---|---|---|---|---|---|---|---|
| basic | -20.000 | 1000 | 4.148 | 40.119 | 40.772 | 36.116 | 36.594 | 44.484 | 37.000 |
| basic | -15.000 | 1000 | 1.413 | 4.755 | 15.102 | 11.715 | 30.119 | 4.394 | 5.590 |
| basic | -10.000 | 1000 | 0.535 | 0.580 | 0.941 | 0.666 | 8.309 | 0.678 | 0.627 |
| basic | -5.000 | 1000 | 0.237 | 0.317 | 0.554 | 0.439 | 2.784 | 0.449 | 0.372 |
| basic | 0.000 | 1000 | 0.120 | 0.251 | 0.540 | 0.380 | 2.134 | 0.373 | 0.288 |
| basic | 5.000 | 1000 | 0.065 | 0.234 | 0.554 | 0.381 | 2.031 | 0.350 | 0.261 |
| basic | 10.000 | 1000 | 0.036 | 0.230 | 0.567 | 0.374 | 2.009 | 0.344 | 0.252 |
| basic | 15.000 | 1000 | 0.020 | 0.229 | 0.572 | 0.373 | 2.004 | 0.343 | 0.250 |
| basic | 20.000 | 1000 | 0.011 | 0.229 | 0.574 | 0.376 | 2.003 | 0.343 | 0.249 |
| moderate | -20.000 | 1000 | 4.894 | 31.688 | 32.029 | 34.892 | 28.988 | 31.954 | 26.459 |
| moderate | -15.000 | 1000 | 1.660 | 12.872 | 15.875 | 34.777 | 20.790 | 5.240 | 8.851 |
| moderate | -10.000 | 1000 | 0.623 | 6.596 | 4.892 | 21.258 | 6.009 | 1.871 | 1.693 |
| moderate | -5.000 | 1000 | 0.273 | 5.880 | 2.975 | 11.503 | 3.902 | 1.807 | 0.654 |
| moderate | 0.000 | 1000 | 0.137 | 5.915 | 2.918 | 8.724 | 3.600 | 1.790 | 0.608 |
| moderate | 5.000 | 1000 | 0.074 | 5.913 | 3.219 | 7.255 | 3.552 | 1.784 | 0.597 |
| moderate | 10.000 | 1000 | 0.041 | 6.001 | 3.219 | 6.423 | 3.536 | 1.782 | 0.595 |
| moderate | 15.000 | 1000 | 0.023 | 6.002 | 3.765 | 6.127 | 3.529 | 1.781 | 0.595 |
| moderate | 20.000 | 1000 | 0.013 | 6.002 | 3.219 | 5.708 | 3.525 | 1.780 | 0.596 |
| advanced1_ood | -20.000 | 1000 | 4.160 | 47.510 | 38.979 | 37.255 | 36.354 | 49.003 | 41.625 |
| advanced1_ood | -15.000 | 1000 | 1.417 | 41.614 | 28.088 | 34.065 | 33.559 | 28.872 | 26.925 |
| advanced1_ood | -10.000 | 1000 | 0.537 | 41.189 | 20.919 | 26.038 | 25.440 | 22.323 | 24.016 |
| advanced1_ood | -5.000 | 1000 | 0.238 | 41.171 | 21.639 | 24.156 | 24.175 | 22.636 | 20.994 |
| advanced1_ood | 0.000 | 1000 | 0.120 | 40.950 | 21.495 | 23.599 | 24.276 | 23.040 | 20.808 |
| advanced1_ood | 5.000 | 1000 | 0.065 | 41.110 | 21.688 | 23.814 | 24.267 | 22.851 | 20.420 |
| advanced1_ood | 10.000 | 1000 | 0.036 | 40.876 | 21.965 | 23.898 | 24.294 | 22.870 | 20.551 |
| advanced1_ood | 15.000 | 1000 | 0.020 | 41.034 | 21.969 | 24.042 | 24.296 | 22.816 | 20.750 |
| advanced1_ood | 20.000 | 1000 | 0.011 | 41.022 | 21.875 | 24.094 | 24.295 | 22.931 | 20.749 |
| advanced2_crowded | -20.000 | 1000 | 8.352 | 21.024 | 21.729 | 30.317 | 21.173 | 18.432 | 18.355 |
| advanced2_crowded | -15.000 | 1000 | 2.798 | 14.658 | 13.553 | 32.972 | 12.058 | 7.918 | 5.407 |
| advanced2_crowded | -10.000 | 1000 | 1.023 | 9.887 | 9.167 | 22.358 | 7.286 | 5.384 | 3.053 |
| advanced2_crowded | -5.000 | 1000 | 0.432 | 9.006 | 8.140 | 17.578 | 6.711 | 4.874 | 2.543 |
| advanced2_crowded | 0.000 | 1000 | 0.210 | 8.871 | 7.912 | 16.321 | 6.652 | 4.804 | 2.445 |
| advanced2_crowded | 5.000 | 1000 | 0.112 | 8.929 | 7.914 | 16.345 | 6.647 | 4.766 | 2.627 |
| advanced2_crowded | 10.000 | 1000 | 0.062 | 8.814 | 7.903 | 16.191 | 6.645 | 4.765 | 2.676 |
| advanced2_crowded | 15.000 | 1000 | 0.034 | 8.814 | 7.964 | 16.063 | 6.644 | 4.787 | 2.677 |
| advanced2_crowded | 20.000 | 1000 | 0.019 | 8.814 | 7.912 | 16.223 | 6.644 | 4.787 | 2.677 |

## R2 (2026-09-30) — Scenario sweep bootstrap 95 % CIs at 0 dB (all SNRs in the CSV)

_Appended 2026-09-30 21:45 UTC from `experiments/runs/eval_r2_20260930/scenario_sweep_r2/scenario_sweep_r2_ci.csv` (snr_db=0)._

| scenario | method | n_scenes | rmse_deg | rmse_ci_lo | rmse_ci_hi | median_rmspe_deg | median_ci_lo | median_ci_hi |
|---|---|---|---|---|---|---|---|---|
| basic | Root-MUSIC | 1000 | 0.251 | 0.239 | 0.262 | 0.163 | 0.152 | 0.174 |
| basic | SubspaceNet | 1000 | 0.540 | 0.512 | 0.566 | 0.323 | 0.290 | 0.345 |
| basic | SubViT | 1000 | 0.380 | 0.365 | 0.395 | 0.270 | 0.257 | 0.290 |
| basic | DA-MUSIC | 1000 | 2.134 | 2.041 | 2.219 | 1.440 | 1.329 | 1.551 |
| basic | ReconUNet | 1000 | 0.373 | 0.353 | 0.393 | 0.238 | 0.219 | 0.253 |
| basic | ReconUNet-C | 1000 | 0.288 | 0.274 | 0.303 | 0.183 | 0.169 | 0.196 |
| moderate | Root-MUSIC | 2000 | 5.915 | 4.036 | 7.574 | 0.290 | 0.269 | 0.309 |
| moderate | SubspaceNet | 2000 | 2.918 | 2.353 | 3.490 | 0.845 | 0.798 | 0.886 |
| moderate | SubViT | 2000 | 8.724 | 6.862 | 10.442 | 0.328 | 0.313 | 0.340 |
| moderate | DA-MUSIC | 2000 | 3.600 | 3.206 | 4.061 | 1.799 | 1.713 | 1.882 |
| moderate | ReconUNet | 2000 | 1.790 | 0.981 | 2.752 | 0.459 | 0.433 | 0.486 |
| moderate | ReconUNet-C | 2000 | 0.608 | 0.566 | 0.656 | 0.279 | 0.264 | 0.299 |
| advanced1_ood | Root-MUSIC | 1000 | 40.950 | 38.410 | 43.552 | 3.918 | 3.467 | 4.572 |
| advanced1_ood | SubspaceNet | 1000 | 21.495 | 19.299 | 23.689 | 3.262 | 3.029 | 3.590 |
| advanced1_ood | SubViT | 1000 | 23.599 | 21.054 | 26.047 | 1.137 | 1.009 | 1.257 |
| advanced1_ood | DA-MUSIC | 1000 | 24.276 | 22.129 | 26.435 | 7.751 | 6.976 | 8.717 |
| advanced1_ood | ReconUNet | 1000 | 23.040 | 20.211 | 25.594 | 1.875 | 1.725 | 2.027 |
| advanced1_ood | ReconUNet-C | 1000 | 20.808 | 18.094 | 23.378 | 1.175 | 1.078 | 1.277 |
| advanced2_crowded | Root-MUSIC | 4000 | 8.871 | 8.239 | 9.448 | 0.361 | 0.343 | 0.379 |
| advanced2_crowded | SubspaceNet | 4000 | 7.912 | 7.399 | 8.395 | 1.887 | 1.811 | 1.962 |
| advanced2_crowded | SubViT | 4000 | 16.321 | 15.634 | 17.005 | 0.557 | 0.540 | 0.579 |
| advanced2_crowded | DA-MUSIC | 4000 | 6.652 | 6.310 | 6.986 | 2.948 | 2.798 | 3.054 |
| advanced2_crowded | ReconUNet | 4000 | 4.804 | 4.176 | 5.406 | 0.894 | 0.862 | 0.925 |
| advanced2_crowded | ReconUNet-C | 4000 | 2.445 | 1.895 | 2.980 | 0.540 | 0.521 | 0.564 |

## R2 (2026-09-30) — Table II (mild) at 0 dB with ReconUNet-C back ends, bootstrap 95 % CIs

_Appended 2026-09-30 21:45 UTC from `experiments/runs/eval_r2_20260930/table2_mild_r2/table2_mild_r2_ci.csv` (snr_db=0)._

| scenario | method | n_scenes | rmse_deg | rmse_ci_lo | rmse_ci_hi | median_rmspe_deg | median_ci_lo | median_ci_hi |
|---|---|---|---|---|---|---|---|---|
| basic | Bartlett | 1000 | 3.964 | 0.957 | 6.467 | 0.162 | 0.152 | 0.175 |
| basic | MVDR | 1000 | 3.561 | 1.399 | 5.086 | 0.165 | 0.156 | 0.179 |
| basic | MUSIC | 1000 | 3.972 | 0.975 | 6.478 | 0.188 | 0.174 | 0.203 |
| basic | Root-MUSIC | 1000 | 0.251 | 0.239 | 0.263 | 0.163 | 0.152 | 0.175 |
| basic | ESPRIT | 1000 | 0.332 | 0.315 | 0.348 | 0.207 | 0.193 | 0.221 |
| basic | Unitary-ESPRIT | 1000 | 0.406 | 0.372 | 0.443 | 0.225 | 0.212 | 0.241 |
| basic | ReconUNet+Root-MUSIC | 1000 | 0.373 | 0.353 | 0.392 | 0.238 | 0.222 | 0.254 |
| basic | ReconUNet+MUSIC | 1000 | 0.422 | 0.402 | 0.440 | 0.268 | 0.246 | 0.287 |
| basic | ReconUNet+ESPRIT | 1000 | 0.398 | 0.378 | 0.419 | 0.257 | 0.239 | 0.276 |
| basic | ReconUNet+Unitary-ESPRIT | 1000 | 0.408 | 0.384 | 0.433 | 0.257 | 0.238 | 0.281 |
| basic | ReconUNet-C+Root-MUSIC | 1000 | 0.288 | 0.273 | 0.304 | 0.182 | 0.167 | 0.197 |
| basic | ReconUNet-C+MUSIC | 1000 | 4.972 | 0.352 | 7.852 | 0.245 | 0.231 | 0.258 |
| basic | ReconUNet-C+ESPRIT | 1000 | 0.289 | 0.274 | 0.304 | 0.181 | 0.169 | 0.201 |
| basic | ReconUNet-C+Unitary-ESPRIT | 1000 | 0.289 | 0.275 | 0.304 | 0.181 | 0.169 | 0.201 |
| basic | SubspaceNet | 1000 | 0.540 | 0.513 | 0.567 | 0.323 | 0.290 | 0.347 |
| basic | SubViT | 1000 | 0.380 | 0.365 | 0.395 | 0.270 | 0.257 | 0.289 |
| basic | DA-MUSIC | 1000 | 2.134 | 2.045 | 2.224 | 1.440 | 1.331 | 1.550 |
| moderate | Bartlett | 1000 | 12.917 | 11.411 | 14.342 | 0.748 | 0.668 | 0.821 |
| moderate | MVDR | 1000 | 11.528 | 9.894 | 13.037 | 0.503 | 0.459 | 0.549 |
| moderate | MUSIC | 1000 | 10.391 | 8.822 | 11.937 | 0.511 | 0.477 | 0.551 |
| moderate | Root-MUSIC | 1000 | 5.915 | 3.830 | 7.647 | 0.486 | 0.456 | 0.524 |
| moderate | ESPRIT | 1000 | 5.938 | 4.096 | 7.626 | 1.767 | 1.618 | 1.919 |
| moderate | Unitary-ESPRIT | 1000 | 5.691 | 4.874 | 6.562 | 1.217 | 1.089 | 1.337 |
| moderate | ReconUNet+Root-MUSIC | 1000 | 1.790 | 0.971 | 2.755 | 0.617 | 0.581 | 0.645 |
| moderate | ReconUNet+MUSIC | 1000 | 4.033 | 1.917 | 5.854 | 0.630 | 0.597 | 0.656 |
| moderate | ReconUNet+ESPRIT | 1000 | 1.776 | 0.995 | 2.743 | 0.632 | 0.608 | 0.658 |
| moderate | ReconUNet+Unitary-ESPRIT | 1000 | 2.630 | 1.048 | 4.049 | 0.642 | 0.612 | 0.670 |
| moderate | ReconUNet-C+Root-MUSIC | 1000 | 0.608 | 0.560 | 0.655 | 0.369 | 0.353 | 0.389 |
| moderate | ReconUNet-C+MUSIC | 1000 | 2.083 | 0.645 | 3.555 | 0.412 | 0.394 | 0.444 |
| moderate | ReconUNet-C+ESPRIT | 1000 | 0.610 | 0.566 | 0.660 | 0.371 | 0.355 | 0.391 |
| moderate | ReconUNet-C+Unitary-ESPRIT | 1000 | 0.610 | 0.562 | 0.661 | 0.371 | 0.355 | 0.389 |
| moderate | SubspaceNet | 1000 | 2.918 | 2.303 | 3.512 | 1.035 | 0.986 | 1.091 |
| moderate | SubViT | 1000 | 8.724 | 6.904 | 10.398 | 0.380 | 0.363 | 0.402 |
| moderate | DA-MUSIC | 1000 | 3.600 | 3.179 | 4.131 | 2.143 | 2.036 | 2.270 |
| advanced1_ood | Bartlett | 1000 | 28.718 | 26.499 | 31.006 | 2.670 | 2.450 | 2.865 |
| advanced1_ood | MVDR | 1000 | 25.060 | 22.516 | 27.264 | 1.691 | 1.551 | 1.864 |
| advanced1_ood | MUSIC | 1000 | 29.926 | 27.503 | 32.137 | 2.840 | 2.598 | 3.048 |
| advanced1_ood | Root-MUSIC | 1000 | 40.950 | 38.362 | 43.667 | 3.918 | 3.467 | 4.569 |
| advanced1_ood | ESPRIT | 1000 | 35.720 | 32.929 | 38.506 | 12.618 | 11.414 | 13.887 |
| advanced1_ood | Unitary-ESPRIT | 1000 | 24.296 | 22.717 | 25.686 | 10.310 | 9.310 | 11.613 |
| advanced1_ood | ReconUNet+Root-MUSIC | 1000 | 22.889 | 20.358 | 25.453 | 1.874 | 1.722 | 2.026 |
| advanced1_ood | ReconUNet+MUSIC | 1000 | 22.889 | 20.130 | 25.434 | 1.860 | 1.704 | 2.028 |
| advanced1_ood | ReconUNet+ESPRIT | 1000 | 24.196 | 21.194 | 26.844 | 2.007 | 1.857 | 2.188 |
| advanced1_ood | ReconUNet+Unitary-ESPRIT | 1000 | 21.327 | 18.873 | 23.720 | 1.987 | 1.861 | 2.112 |
| advanced1_ood | ReconUNet-C+Root-MUSIC | 1000 | 20.808 | 17.927 | 23.261 | 1.175 | 1.081 | 1.278 |
| advanced1_ood | ReconUNet-C+MUSIC | 1000 | 20.805 | 18.038 | 23.456 | 1.206 | 1.093 | 1.309 |
| advanced1_ood | ReconUNet-C+ESPRIT | 1000 | 21.031 | 18.159 | 23.708 | 1.192 | 1.080 | 1.296 |
| advanced1_ood | ReconUNet-C+Unitary-ESPRIT | 1000 | 20.553 | 17.753 | 22.947 | 1.192 | 1.098 | 1.304 |
| advanced1_ood | SubspaceNet | 1000 | 21.495 | 19.144 | 23.832 | 3.262 | 3.049 | 3.589 |
| advanced1_ood | SubViT | 1000 | 23.599 | 21.184 | 25.891 | 1.137 | 1.007 | 1.260 |
| advanced1_ood | DA-MUSIC | 1000 | 24.276 | 21.968 | 26.373 | 7.751 | 6.889 | 8.793 |
| advanced2_crowded | Bartlett | 1000 | 21.003 | 19.935 | 22.250 | 13.510 | 12.370 | 14.623 |
| advanced2_crowded | MVDR | 1000 | 15.723 | 14.787 | 16.554 | 2.530 | 2.006 | 3.274 |
| advanced2_crowded | MUSIC | 1000 | 12.561 | 11.907 | 13.145 | 1.639 | 1.466 | 1.933 |
| advanced2_crowded | Root-MUSIC | 1000 | 8.871 | 8.015 | 9.670 | 1.291 | 1.146 | 1.395 |
| advanced2_crowded | ESPRIT | 1000 | 8.145 | 7.233 | 9.045 | 2.802 | 2.553 | 3.101 |
| advanced2_crowded | Unitary-ESPRIT | 1000 | 8.044 | 7.481 | 8.614 | 2.746 | 2.513 | 3.073 |
| advanced2_crowded | ReconUNet+Root-MUSIC | 1000 | 4.804 | 4.035 | 5.533 | 1.431 | 1.361 | 1.501 |
| advanced2_crowded | ReconUNet+MUSIC | 1000 | 8.162 | 7.415 | 8.865 | 1.524 | 1.422 | 1.603 |
| advanced2_crowded | ReconUNet+ESPRIT | 1000 | 4.613 | 3.658 | 5.480 | 1.505 | 1.435 | 1.593 |
| advanced2_crowded | ReconUNet+Unitary-ESPRIT | 1000 | 4.005 | 3.436 | 4.530 | 1.518 | 1.436 | 1.601 |
| advanced2_crowded | ReconUNet-C+Root-MUSIC | 1000 | 2.446 | 1.778 | 3.104 | 0.903 | 0.873 | 0.948 |
| advanced2_crowded | ReconUNet-C+MUSIC | 1000 | 3.872 | 3.152 | 4.599 | 0.919 | 0.878 | 0.964 |
| advanced2_crowded | ReconUNet-C+ESPRIT | 1000 | 2.663 | 1.846 | 3.437 | 0.919 | 0.876 | 0.961 |
| advanced2_crowded | ReconUNet-C+Unitary-ESPRIT | 1000 | 2.494 | 1.866 | 3.104 | 0.920 | 0.875 | 0.959 |
| advanced2_crowded | SubspaceNet | 1000 | 7.912 | 7.369 | 8.448 | 2.786 | 2.657 | 2.935 |
| advanced2_crowded | SubViT | 1000 | 16.321 | 15.555 | 17.105 | 5.159 | 1.768 | 6.791 |
| advanced2_crowded | DA-MUSIC | 1000 | 6.652 | 6.266 | 7.049 | 4.126 | 3.961 | 4.327 |

## R2 (2026-09-30) — Table II (mild): ReconUNet vs ReconUNet-C, every back end, every SNR (pooled RMSE, deg)

_Appended 2026-09-30 21:45 UTC from `docs/revision_r2/table2_mild_reconunet_backends_by_snr.csv`._

| scenario | snr_db | ReconUNet+Root-MUSIC | ReconUNet+MUSIC | ReconUNet+ESPRIT | ReconUNet+Unitary-ESPRIT | ReconUNet-C+Root-MUSIC | ReconUNet-C+MUSIC | ReconUNet-C+ESPRIT | ReconUNet-C+Unitary-ESPRIT | Root-MUSIC | CRLB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| advanced1_ood | -20.000 | 49.003 | 48.065 | 54.184 | 41.590 | 41.580 | 41.406 | 42.223 | 40.718 | 47.510 | 4.160 |
| advanced1_ood | -15.000 | 28.872 | 28.593 | 30.038 | 25.609 | 26.925 | 27.025 | 27.204 | 26.441 | 41.614 | 1.417 |
| advanced1_ood | -10.000 | 22.324 | 22.666 | 23.317 | 21.421 | 24.016 | 24.017 | 23.956 | 23.607 | 41.189 | 0.537 |
| advanced1_ood | -5.000 | 22.636 | 22.433 | 23.928 | 21.408 | 20.994 | 20.991 | 21.413 | 20.921 | 41.171 | 0.238 |
| advanced1_ood | 0.000 | 22.889 | 22.889 | 24.196 | 21.327 | 20.808 | 20.805 | 21.031 | 20.553 | 40.950 | 0.120 |
| advanced1_ood | 5.000 | 22.851 | 22.683 | 24.080 | 21.271 | 20.420 | 20.415 | 20.619 | 20.125 | 41.110 | 0.065 |
| advanced1_ood | 10.000 | 22.871 | 22.683 | 23.924 | 21.334 | 20.551 | 20.549 | 20.889 | 20.284 | 40.876 | 0.036 |
| advanced1_ood | 15.000 | 22.816 | 23.049 | 23.974 | 21.365 | 20.751 | 20.748 | 20.641 | 20.283 | 41.034 | 0.020 |
| advanced1_ood | 20.000 | 22.931 | 23.033 | 24.318 | 21.384 | 20.749 | 20.747 | 21.099 | 20.276 | 41.022 | 0.011 |
| advanced2_crowded | -20.000 | 18.334 | 25.555 | 18.937 | 15.696 | 18.360 | 18.604 | 18.151 | 16.889 | 21.024 | 8.352 |
| advanced2_crowded | -15.000 | 7.918 | 12.272 | 7.737 | 6.584 | 5.407 | 6.680 | 5.498 | 5.132 | 14.658 | 2.798 |
| advanced2_crowded | -10.000 | 5.384 | 8.659 | 5.555 | 4.444 | 3.053 | 4.034 | 3.062 | 2.963 | 9.887 | 1.023 |
| advanced2_crowded | -5.000 | 4.874 | 8.311 | 4.739 | 4.124 | 2.543 | 3.838 | 2.757 | 2.572 | 9.006 | 0.432 |
| advanced2_crowded | 0.000 | 4.804 | 8.162 | 4.613 | 4.005 | 2.446 | 3.872 | 2.663 | 2.494 | 8.871 | 0.210 |
| advanced2_crowded | 5.000 | 4.766 | 8.202 | 4.402 | 3.965 | 2.627 | 3.856 | 2.490 | 2.371 | 8.929 | 0.112 |
| advanced2_crowded | 10.000 | 4.765 | 8.298 | 4.402 | 3.939 | 2.677 | 3.743 | 2.515 | 2.337 | 8.814 | 0.062 |
| advanced2_crowded | 15.000 | 4.787 | 8.340 | 4.410 | 3.924 | 2.677 | 3.718 | 2.537 | 2.348 | 8.814 | 0.034 |
| advanced2_crowded | 20.000 | 4.787 | 8.385 | 4.237 | 3.917 | 2.677 | 3.759 | 2.552 | 2.360 | 8.814 | 0.019 |
| basic | -20.000 | 44.485 | 44.244 | 49.647 | 36.933 | 37.000 | 37.077 | 37.368 | 36.435 | 40.119 | 4.148 |
| basic | -15.000 | 4.394 | 5.256 | 4.984 | 3.316 | 5.590 | 5.591 | 5.572 | 5.736 | 4.755 | 1.413 |
| basic | -10.000 | 0.678 | 0.698 | 0.707 | 0.714 | 0.627 | 3.564 | 0.630 | 0.630 | 0.580 | 0.535 |
| basic | -5.000 | 0.449 | 0.484 | 0.472 | 0.481 | 0.372 | 3.773 | 0.374 | 0.374 | 0.317 | 0.237 |
| basic | 0.000 | 0.373 | 0.422 | 0.398 | 0.408 | 0.288 | 4.972 | 0.289 | 0.289 | 0.251 | 0.120 |
| basic | 5.000 | 0.350 | 0.400 | 0.376 | 0.387 | 0.261 | 4.973 | 0.261 | 0.261 | 0.234 | 0.065 |
| basic | 10.000 | 0.344 | 0.395 | 0.371 | 0.382 | 0.252 | 3.644 | 0.253 | 0.253 | 0.230 | 0.036 |
| basic | 15.000 | 0.343 | 0.394 | 0.370 | 0.381 | 0.250 | 3.644 | 0.250 | 0.250 | 0.229 | 0.020 |
| basic | 20.000 | 0.343 | 0.394 | 0.369 | 0.381 | 0.249 | 0.985 | 0.249 | 0.250 | 0.229 | 0.011 |
| moderate | -20.000 | 31.946 | 31.939 | 32.929 | 26.738 | 26.523 | 25.966 | 26.858 | 25.773 | 31.688 | 4.894 |
| moderate | -15.000 | 5.240 | 10.041 | 3.847 | 4.865 | 8.851 | 9.031 | 8.688 | 8.368 | 12.872 | 1.660 |
| moderate | -10.000 | 1.871 | 4.098 | 1.867 | 1.883 | 1.693 | 3.366 | 1.683 | 1.687 | 6.596 | 0.623 |
| moderate | -5.000 | 1.807 | 4.043 | 1.788 | 2.581 | 0.654 | 2.099 | 0.656 | 0.659 | 5.880 | 0.273 |
| moderate | 0.000 | 1.790 | 4.033 | 1.776 | 2.630 | 0.608 | 2.083 | 0.610 | 0.610 | 5.915 | 0.137 |
| moderate | 5.000 | 1.784 | 4.028 | 1.776 | 2.561 | 0.597 | 2.081 | 0.601 | 0.600 | 5.913 | 0.074 |
| moderate | 10.000 | 1.782 | 4.026 | 1.777 | 2.503 | 0.595 | 2.082 | 0.599 | 0.598 | 6.001 | 0.041 |
| moderate | 15.000 | 1.781 | 4.025 | 1.779 | 2.457 | 0.595 | 2.083 | 0.599 | 0.599 | 6.002 | 0.023 |
| moderate | 20.000 | 1.780 | 4.025 | 1.779 | 2.428 | 0.596 | 2.083 | 0.600 | 0.599 | 6.002 | 0.013 |

## R2 (2026-09-30) — Table III (harsh) at 0 dB with ReconUNet-C back ends, bootstrap 95 % CIs

_Appended 2026-09-30 21:45 UTC from `experiments/runs/eval_r2_20260930/table2_harsh_r2/table2_harsh_r2_ci.csv` (snr_db=0)._

| scenario | method | n_scenes | rmse_deg | rmse_ci_lo | rmse_ci_hi | median_rmspe_deg | median_ci_lo | median_ci_hi |
|---|---|---|---|---|---|---|---|---|
| basic | Bartlett | 1000 | 6.140 | 2.945 | 8.831 | 0.767 | 0.708 | 0.821 |
| basic | MVDR | 1000 | 5.906 | 3.129 | 8.469 | 0.770 | 0.718 | 0.828 |
| basic | MUSIC | 1000 | 5.734 | 2.330 | 8.351 | 0.763 | 0.704 | 0.817 |
| basic | Root-MUSIC | 1000 | 1.149 | 1.097 | 1.198 | 0.760 | 0.710 | 0.813 |
| basic | ESPRIT | 1000 | 1.579 | 1.489 | 1.666 | 1.020 | 0.941 | 1.086 |
| basic | Unitary-ESPRIT | 1000 | 3.645 | 3.188 | 4.081 | 1.349 | 1.244 | 1.468 |
| basic | ReconUNet+Root-MUSIC | 1000 | 1.203 | 1.154 | 1.260 | 0.836 | 0.779 | 0.868 |
| basic | ReconUNet+MUSIC | 1000 | 1.211 | 1.154 | 1.262 | 0.825 | 0.746 | 0.898 |
| basic | ReconUNet+ESPRIT | 1000 | 1.210 | 1.153 | 1.268 | 0.851 | 0.773 | 0.897 |
| basic | ReconUNet+Unitary-ESPRIT | 1000 | 1.216 | 1.158 | 1.269 | 0.849 | 0.785 | 0.896 |
| basic | ReconUNet-C+Root-MUSIC | 1000 | 1.178 | 1.121 | 1.234 | 0.773 | 0.714 | 0.830 |
| basic | ReconUNet-C+MUSIC | 1000 | 1.497 | 1.144 | 2.005 | 0.766 | 0.706 | 0.809 |
| basic | ReconUNet-C+ESPRIT | 1000 | 1.178 | 1.120 | 1.236 | 0.767 | 0.705 | 0.829 |
| basic | ReconUNet-C+Unitary-ESPRIT | 1000 | 1.179 | 1.125 | 1.237 | 0.768 | 0.709 | 0.831 |
| basic | SubspaceNet | 1000 | 1.282 | 1.221 | 1.343 | 0.829 | 0.763 | 0.880 |
| basic | SubViT | 1000 | 1.216 | 1.157 | 1.281 | 0.783 | 0.730 | 0.850 |
| basic | DA-MUSIC | 1000 | 2.744 | 2.611 | 2.867 | 1.867 | 1.716 | 1.988 |
| moderate | Bartlett | 1000 | 14.500 | 13.068 | 15.838 | 1.389 | 1.297 | 1.467 |
| moderate | MVDR | 1000 | 12.275 | 10.813 | 13.716 | 1.165 | 1.100 | 1.249 |
| moderate | MUSIC | 1000 | 11.699 | 10.245 | 13.141 | 1.193 | 1.129 | 1.269 |
| moderate | Root-MUSIC | 1000 | 6.827 | 4.938 | 8.563 | 1.191 | 1.132 | 1.253 |
| moderate | ESPRIT | 1000 | 6.201 | 4.547 | 7.792 | 2.388 | 2.248 | 2.514 |
| moderate | Unitary-ESPRIT | 1000 | 6.760 | 6.005 | 7.540 | 2.736 | 2.506 | 2.983 |
| moderate | ReconUNet+Root-MUSIC | 1000 | 2.357 | 1.523 | 3.283 | 1.132 | 1.088 | 1.199 |
| moderate | ReconUNet+MUSIC | 1000 | 5.402 | 3.717 | 7.075 | 1.151 | 1.106 | 1.222 |
| moderate | ReconUNet+ESPRIT | 1000 | 2.194 | 1.557 | 3.124 | 1.172 | 1.093 | 1.228 |
| moderate | ReconUNet+Unitary-ESPRIT | 1000 | 3.018 | 1.586 | 4.337 | 1.166 | 1.100 | 1.237 |
| moderate | ReconUNet-C+Root-MUSIC | 1000 | 2.136 | 1.408 | 3.039 | 1.039 | 0.999 | 1.087 |
| moderate | ReconUNet-C+MUSIC | 1000 | 5.601 | 3.753 | 7.293 | 1.051 | 1.004 | 1.102 |
| moderate | ReconUNet-C+ESPRIT | 1000 | 2.141 | 1.413 | 3.104 | 1.047 | 1.003 | 1.091 |
| moderate | ReconUNet-C+Unitary-ESPRIT | 1000 | 2.140 | 1.404 | 3.118 | 1.047 | 1.000 | 1.090 |
| moderate | SubspaceNet | 1000 | 4.734 | 3.426 | 5.997 | 1.545 | 1.468 | 1.625 |
| moderate | SubViT | 1000 | 14.851 | 13.201 | 16.462 | 1.156 | 1.078 | 1.235 |
| moderate | DA-MUSIC | 1000 | 4.585 | 3.939 | 5.466 | 2.650 | 2.500 | 2.808 |
| advanced1_ood | Bartlett | 1000 | 30.220 | 28.076 | 32.476 | 3.056 | 2.889 | 3.409 |
| advanced1_ood | MVDR | 1000 | 27.974 | 25.363 | 30.354 | 2.191 | 1.973 | 2.378 |
| advanced1_ood | MUSIC | 1000 | 30.670 | 28.253 | 32.825 | 3.222 | 2.952 | 3.617 |
| advanced1_ood | Root-MUSIC | 1000 | 41.958 | 39.407 | 44.413 | 4.785 | 4.311 | 5.378 |
| advanced1_ood | ESPRIT | 1000 | 38.633 | 35.679 | 41.314 | 14.228 | 13.107 | 15.430 |
| advanced1_ood | Unitary-ESPRIT | 1000 | 25.550 | 24.110 | 27.003 | 12.170 | 10.966 | 13.354 |
| advanced1_ood | ReconUNet+Root-MUSIC | 1000 | 25.307 | 22.631 | 27.706 | 2.277 | 2.100 | 2.514 |
| advanced1_ood | ReconUNet+MUSIC | 1000 | 25.298 | 22.702 | 27.903 | 2.320 | 2.136 | 2.522 |
| advanced1_ood | ReconUNet+ESPRIT | 1000 | 26.989 | 24.053 | 29.789 | 2.519 | 2.271 | 2.809 |
| advanced1_ood | ReconUNet+Unitary-ESPRIT | 1000 | 23.794 | 21.164 | 26.159 | 2.471 | 2.214 | 2.698 |
| advanced1_ood | ReconUNet-C+Root-MUSIC | 1000 | 22.387 | 19.546 | 24.806 | 1.755 | 1.600 | 1.952 |
| advanced1_ood | ReconUNet-C+MUSIC | 1000 | 22.560 | 19.776 | 24.930 | 1.787 | 1.634 | 1.951 |
| advanced1_ood | ReconUNet-C+ESPRIT | 1000 | 22.423 | 19.677 | 24.922 | 1.790 | 1.615 | 1.957 |
| advanced1_ood | ReconUNet-C+Unitary-ESPRIT | 1000 | 22.099 | 19.648 | 24.665 | 1.796 | 1.620 | 1.965 |
| advanced1_ood | SubspaceNet | 1000 | 23.245 | 21.143 | 25.123 | 3.961 | 3.590 | 4.315 |
| advanced1_ood | SubViT | 1000 | 26.097 | 23.964 | 28.377 | 2.177 | 1.986 | 2.369 |
| advanced1_ood | DA-MUSIC | 1000 | 25.522 | 23.225 | 27.466 | 9.437 | 8.444 | 10.264 |
| advanced2_crowded | Bartlett | 1000 | 21.828 | 20.665 | 23.090 | 13.646 | 12.693 | 14.811 |
| advanced2_crowded | MVDR | 1000 | 18.253 | 17.155 | 19.348 | 3.197 | 2.694 | 4.320 |
| advanced2_crowded | MUSIC | 1000 | 15.558 | 14.641 | 16.514 | 2.497 | 2.204 | 2.886 |
| advanced2_crowded | Root-MUSIC | 1000 | 9.342 | 8.514 | 10.115 | 2.094 | 2.002 | 2.207 |
| advanced2_crowded | ESPRIT | 1000 | 8.390 | 7.545 | 9.251 | 3.503 | 3.330 | 3.854 |
| advanced2_crowded | Unitary-ESPRIT | 1000 | 9.072 | 8.522 | 9.647 | 4.407 | 4.071 | 4.721 |
| advanced2_crowded | ReconUNet+Root-MUSIC | 1000 | 6.136 | 5.382 | 6.904 | 1.975 | 1.890 | 2.052 |
| advanced2_crowded | ReconUNet+MUSIC | 1000 | 9.151 | 8.385 | 9.982 | 2.090 | 1.998 | 2.201 |
| advanced2_crowded | ReconUNet+ESPRIT | 1000 | 5.410 | 4.586 | 6.265 | 2.081 | 2.006 | 2.185 |
| advanced2_crowded | ReconUNet+Unitary-ESPRIT | 1000 | 4.931 | 4.304 | 5.550 | 2.030 | 1.943 | 2.117 |
| advanced2_crowded | ReconUNet-C+Root-MUSIC | 1000 | 4.152 | 3.359 | 4.867 | 1.588 | 1.511 | 1.655 |
| advanced2_crowded | ReconUNet-C+MUSIC | 1000 | 5.159 | 4.412 | 5.875 | 1.584 | 1.519 | 1.679 |
| advanced2_crowded | ReconUNet-C+ESPRIT | 1000 | 4.029 | 3.213 | 4.772 | 1.589 | 1.504 | 1.642 |
| advanced2_crowded | ReconUNet-C+Unitary-ESPRIT | 1000 | 3.819 | 3.099 | 4.447 | 1.583 | 1.509 | 1.634 |
| advanced2_crowded | SubspaceNet | 1000 | 8.823 | 8.271 | 9.390 | 3.418 | 3.289 | 3.571 |
| advanced2_crowded | SubViT | 1000 | 18.420 | 17.678 | 19.247 | 11.513 | 9.927 | 13.433 |
| advanced2_crowded | DA-MUSIC | 1000 | 7.503 | 7.094 | 7.928 | 4.599 | 4.418 | 4.797 |

## R2 (2026-09-30) — Table III (harsh): ReconUNet vs ReconUNet-C, every back end, every SNR (pooled RMSE, deg)

_Appended 2026-09-30 21:45 UTC from `docs/revision_r2/table2_harsh_reconunet_backends_by_snr.csv`._

| scenario | snr_db | ReconUNet+Root-MUSIC | ReconUNet+MUSIC | ReconUNet+ESPRIT | ReconUNet+Unitary-ESPRIT | ReconUNet-C+Root-MUSIC | ReconUNet-C+MUSIC | ReconUNet-C+ESPRIT | ReconUNet-C+Unitary-ESPRIT | Root-MUSIC | CRLB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| advanced1_ood | -20.000 | 49.921 | 49.507 | 55.164 | 43.386 | 42.955 | 42.872 | 43.232 | 42.236 | 48.819 | 4.160 |
| advanced1_ood | -15.000 | 30.501 | 30.726 | 32.383 | 28.000 | 28.888 | 29.013 | 29.472 | 28.351 | 41.841 | 1.417 |
| advanced1_ood | -10.000 | 25.455 | 25.444 | 27.378 | 24.398 | 24.732 | 24.726 | 24.755 | 24.335 | 41.360 | 0.537 |
| advanced1_ood | -5.000 | 25.404 | 24.816 | 26.226 | 23.598 | 23.652 | 23.654 | 23.811 | 23.549 | 41.484 | 0.238 |
| advanced1_ood | 0.000 | 25.307 | 25.298 | 26.989 | 23.794 | 22.387 | 22.560 | 22.423 | 22.099 | 41.958 | 0.120 |
| advanced1_ood | 5.000 | 25.189 | 25.407 | 26.503 | 24.122 | 22.374 | 22.266 | 22.393 | 22.104 | 41.857 | 0.065 |
| advanced1_ood | 10.000 | 25.294 | 25.461 | 26.383 | 24.221 | 22.328 | 22.248 | 22.292 | 22.001 | 41.994 | 0.036 |
| advanced1_ood | 15.000 | 25.382 | 25.348 | 26.307 | 24.253 | 22.286 | 22.179 | 22.450 | 21.969 | 41.998 | 0.020 |
| advanced1_ood | 20.000 | 25.382 | 25.192 | 26.267 | 24.269 | 22.199 | 22.092 | 22.386 | 22.012 | 42.041 | 0.011 |
| advanced2_crowded | -20.000 | 19.214 | 26.645 | 19.507 | 16.094 | 18.346 | 18.684 | 18.448 | 17.463 | 21.343 | 8.352 |
| advanced2_crowded | -15.000 | 8.798 | 12.787 | 8.551 | 7.186 | 7.078 | 8.012 | 7.083 | 6.556 | 15.379 | 2.798 |
| advanced2_crowded | -10.000 | 6.284 | 9.588 | 5.911 | 5.268 | 4.355 | 5.484 | 4.479 | 4.236 | 10.735 | 1.023 |
| advanced2_crowded | -5.000 | 6.122 | 9.166 | 5.903 | 5.012 | 4.120 | 5.367 | 3.957 | 3.934 | 9.409 | 0.432 |
| advanced2_crowded | 0.000 | 6.136 | 9.151 | 5.410 | 4.931 | 4.152 | 5.159 | 4.029 | 3.819 | 9.342 | 0.210 |
| advanced2_crowded | 5.000 | 6.208 | 9.229 | 5.523 | 4.928 | 4.069 | 5.137 | 3.908 | 3.773 | 9.191 | 0.112 |
| advanced2_crowded | 10.000 | 6.024 | 9.214 | 5.496 | 4.907 | 4.078 | 5.132 | 3.949 | 3.693 | 9.100 | 0.062 |
| advanced2_crowded | 15.000 | 6.025 | 9.229 | 5.484 | 4.899 | 4.117 | 5.087 | 3.784 | 3.617 | 9.110 | 0.034 |
| advanced2_crowded | 20.000 | 6.027 | 9.247 | 5.479 | 4.894 | 3.999 | 5.150 | 3.862 | 3.610 | 9.110 | 0.019 |
| basic | -20.000 | 44.948 | 44.797 | 50.119 | 37.720 | 38.165 | 38.148 | 38.312 | 37.274 | 41.159 | 4.148 |
| basic | -15.000 | 7.110 | 7.340 | 7.570 | 5.644 | 7.635 | 6.755 | 6.773 | 6.788 | 5.573 | 1.413 |
| basic | -10.000 | 1.321 | 1.334 | 1.331 | 1.340 | 2.181 | 4.134 | 2.182 | 2.183 | 1.252 | 0.535 |
| basic | -5.000 | 1.222 | 1.231 | 1.228 | 1.235 | 1.209 | 1.522 | 1.210 | 1.210 | 1.161 | 0.237 |
| basic | 0.000 | 1.203 | 1.211 | 1.210 | 1.216 | 1.178 | 1.497 | 1.178 | 1.179 | 1.149 | 0.120 |
| basic | 5.000 | 1.200 | 1.204 | 1.207 | 1.213 | 1.170 | 3.677 | 1.170 | 1.170 | 1.147 | 0.065 |
| basic | 10.000 | 1.200 | 1.202 | 1.208 | 1.213 | 1.169 | 3.677 | 1.169 | 1.170 | 1.147 | 0.036 |
| basic | 15.000 | 1.201 | 1.202 | 1.208 | 1.214 | 1.169 | 3.677 | 1.170 | 1.173 | 1.148 | 0.020 |
| basic | 20.000 | 1.201 | 1.202 | 1.209 | 1.214 | 1.169 | 3.676 | 1.172 | 1.181 | 1.148 | 0.011 |
| moderate | -20.000 | 32.243 | 32.697 | 33.462 | 27.378 | 28.282 | 27.605 | 28.339 | 27.228 | 32.036 | 4.894 |
| moderate | -15.000 | 6.533 | 10.772 | 6.574 | 5.989 | 9.984 | 10.919 | 9.896 | 9.465 | 14.870 | 1.660 |
| moderate | -10.000 | 2.706 | 5.054 | 2.611 | 2.708 | 2.417 | 5.626 | 2.431 | 2.431 | 7.491 | 0.623 |
| moderate | -5.000 | 2.480 | 5.047 | 2.297 | 2.430 | 2.175 | 5.334 | 2.182 | 2.179 | 7.221 | 0.273 |
| moderate | 0.000 | 2.357 | 5.402 | 2.194 | 3.018 | 2.136 | 5.601 | 2.141 | 2.140 | 6.827 | 0.137 |
| moderate | 5.000 | 2.354 | 5.428 | 2.187 | 2.992 | 2.133 | 5.549 | 2.138 | 2.136 | 6.876 | 0.074 |
| moderate | 10.000 | 2.353 | 5.427 | 2.191 | 2.993 | 2.132 | 5.547 | 2.137 | 2.134 | 6.877 | 0.041 |
| moderate | 15.000 | 2.353 | 5.457 | 2.195 | 2.997 | 2.132 | 5.546 | 2.137 | 2.133 | 6.877 | 0.023 |
| moderate | 20.000 | 2.352 | 5.457 | 2.199 | 2.999 | 2.132 | 5.546 | 2.136 | 2.133 | 6.877 | 0.013 |

## R2 (2026-09-30) — Snapshot sweep, Moderate at 0 dB, ReconUNet-C added

_Appended 2026-09-30 21:45 UTC from `experiments/runs/sweeps_r2_20260930/snapshot_sweep.csv`._

| T | sampling | method | rmse_deg | median_rmspe_deg | n |
|---|---|---|---|---|---|
| 8 | window | Root-MUSIC | 27.038 | 13.711 | 1000 |
| 8 | window | ReconUNet |  |  | 0 |
| 8 | window | SubspaceNet |  |  | 0 |
| 8 | window | ReconUNet-C |  |  | 0 |
| 8 | both | CRLB | 1.661 |  | 1000 |
| 8 | decimated | Root-MUSIC | 8.302 | 1.260 | 1000 |
| 8 | decimated | ReconUNet |  |  | 0 |
| 8 | decimated | SubspaceNet |  |  | 0 |
| 8 | decimated | ReconUNet-C |  |  | 0 |
| 16 | window | Root-MUSIC | 20.592 | 2.458 | 1000 |
| 16 | window | ReconUNet | 25.941 | 17.300 | 1000 |
| 16 | window | SubspaceNet | 21.481 | 9.916 | 1000 |
| 16 | window | ReconUNet-C | 28.630 | 18.479 | 1000 |
| 16 | both | CRLB | 0.831 |  | 1000 |
| 16 | decimated | Root-MUSIC | 7.718 | 0.906 | 1000 |
| 16 | decimated | ReconUNet | 34.192 | 27.631 | 1000 |
| 16 | decimated | SubspaceNet | 23.058 | 10.677 | 1000 |
| 16 | decimated | ReconUNet-C | 38.161 | 30.549 | 1000 |
| 32 | window | Root-MUSIC | 11.931 | 1.161 | 1000 |
| 32 | window | ReconUNet | 20.403 | 3.149 | 1000 |
| 32 | window | SubspaceNet | 17.576 | 3.970 | 1000 |
| 32 | window | ReconUNet-C | 22.013 | 4.110 | 1000 |
| 32 | both | CRLB | 0.415 |  | 1000 |
| 32 | decimated | Root-MUSIC | 5.944 | 0.716 | 1000 |
| 32 | decimated | ReconUNet | 29.470 | 23.286 | 1000 |
| 32 | decimated | SubspaceNet | 15.684 | 3.820 | 1000 |
| 32 | decimated | ReconUNet-C | 34.392 | 24.680 | 1000 |
| 64 | window | Root-MUSIC | 7.019 | 0.752 | 1000 |
| 64 | window | ReconUNet | 12.653 | 1.289 | 1000 |
| 64 | window | SubspaceNet | 12.238 | 2.265 | 1000 |
| 64 | window | ReconUNet-C | 11.730 | 1.193 | 1000 |
| 64 | both | CRLB | 0.208 |  | 1000 |
| 64 | decimated | Root-MUSIC | 5.781 | 0.612 | 1000 |
| 64 | decimated | ReconUNet | 26.482 | 21.263 | 1000 |
| 64 | decimated | SubspaceNet | 10.132 | 2.096 | 1000 |
| 64 | decimated | ReconUNet-C | 30.346 | 18.371 | 1000 |
| 128 | window | Root-MUSIC | 5.713 | 0.572 | 1000 |
| 128 | window | ReconUNet | 4.596 | 0.881 | 1000 |
| 128 | window | SubspaceNet | 6.645 | 1.517 | 1000 |
| 128 | window | ReconUNet-C | 4.823 | 0.695 | 1000 |
| 128 | both | CRLB | 0.104 |  | 1000 |
| 128 | decimated | Root-MUSIC | 5.546 | 0.560 | 1000 |
| 128 | decimated | ReconUNet | 16.356 | 2.572 | 1000 |
| 128 | decimated | SubspaceNet | 7.117 | 1.397 | 1000 |
| 128 | decimated | ReconUNet-C | 23.684 | 4.047 | 1000 |
| 256 | window | Root-MUSIC | 5.875 | 0.503 | 1000 |
| 256 | window | ReconUNet | 1.871 | 0.699 | 1000 |
| 256 | window | SubspaceNet | 3.629 | 1.189 | 1000 |
| 256 | window | ReconUNet-C | 1.694 | 0.481 | 1000 |
| 256 | both | CRLB | 0.052 |  | 1000 |
| 256 | decimated | Root-MUSIC | 5.908 | 0.513 | 1000 |
| 256 | decimated | ReconUNet | 1.843 | 0.704 | 1000 |
| 256 | decimated | SubspaceNet | 3.818 | 1.029 | 1000 |
| 256 | decimated | ReconUNet-C | 2.922 | 0.548 | 1000 |
| 512 | window | Root-MUSIC | 5.915 | 0.486 | 1000 |
| 512 | window | ReconUNet | 1.790 | 0.616 | 1000 |
| 512 | window | SubspaceNet | 2.918 | 1.035 | 1000 |
| 512 | window | ReconUNet-C | 0.608 | 0.369 | 1000 |
| 512 | both | CRLB | 0.026 |  | 1000 |
| 512 | decimated | Root-MUSIC | 5.915 | 0.486 | 1000 |
| 512 | decimated | ReconUNet | 1.790 | 0.616 | 1000 |
| 512 | decimated | SubspaceNet | 2.918 | 1.035 | 1000 |
| 512 | decimated | ReconUNet-C | 0.608 | 0.369 | 1000 |

## R2 (2026-09-30) — Separation sweep, K=2 at 0 and −5 dB (RMSE, median, resolution probability), ReconUNet-C added

_Appended 2026-09-30 21:45 UTC from `experiments/runs/sweeps_r2_20260930/separation_sweep.csv`._

| sep_deg | snr_db | method | rmse_deg | median_rmspe_deg | resolution_prob | n |
|---|---|---|---|---|---|---|
| 2.000 | 0.000 | Root-MUSIC | 29.201 | 18.528 | 0.068 | 1000 |
| 2.000 | 0.000 | ESPRIT | 13.417 | 2.394 | 0.128 | 1000 |
| 2.000 | 0.000 | ReconUNet | 34.761 | 20.303 | 0.000 | 1000 |
| 2.000 | 0.000 | SubspaceNet | 25.365 | 18.833 | 0.001 | 1000 |
| 2.000 | 0.000 | DA-MUSIC | 9.366 | 7.194 | 0.003 | 1000 |
| 2.000 | 0.000 | ReconUNet-C | 26.149 | 6.579 | 0.090 | 1000 |
| 2.000 | 0.000 | CRLB | 4.851 |  |  | 1000 |
| 2.000 | -5.000 | Root-MUSIC | 34.382 | 25.212 | 0.001 | 1000 |
| 2.000 | -5.000 | ESPRIT | 32.171 | 21.999 | 0.014 | 1000 |
| 2.000 | -5.000 | ReconUNet | 35.292 | 20.248 | 0.001 | 1000 |
| 2.000 | -5.000 | SubspaceNet | 22.815 | 20.053 | 0.004 | 1000 |
| 2.000 | -5.000 | DA-MUSIC | 9.296 | 7.263 | 0.003 | 1000 |
| 2.000 | -5.000 | ReconUNet-C | 24.376 | 5.501 | 0.081 | 1000 |
| 2.000 | -5.000 | CRLB | 40.267 |  |  | 1000 |
| 4.000 | 0.000 | Root-MUSIC | 0.911 | 0.712 | 0.930 | 1000 |
| 4.000 | 0.000 | ESPRIT | 1.123 | 0.860 | 0.869 | 1000 |
| 4.000 | 0.000 | ReconUNet | 32.156 | 18.383 | 0.041 | 1000 |
| 4.000 | 0.000 | SubspaceNet | 23.546 | 16.580 | 0.018 | 1000 |
| 4.000 | 0.000 | DA-MUSIC | 7.907 | 6.098 | 0.006 | 1000 |
| 4.000 | 0.000 | ReconUNet-C | 15.861 | 2.034 | 0.380 | 1000 |
| 4.000 | 0.000 | CRLB | 0.534 |  |  | 1000 |
| 4.000 | -5.000 | Root-MUSIC | 24.308 | 2.481 | 0.316 | 1000 |
| 4.000 | -5.000 | ESPRIT | 11.022 | 1.941 | 0.361 | 1000 |
| 4.000 | -5.000 | ReconUNet | 30.876 | 17.580 | 0.040 | 1000 |
| 4.000 | -5.000 | SubspaceNet | 21.620 | 19.078 | 0.019 | 1000 |
| 4.000 | -5.000 | DA-MUSIC | 7.851 | 6.120 | 0.006 | 1000 |
| 4.000 | -5.000 | ReconUNet-C | 15.791 | 2.048 | 0.368 | 1000 |
| 4.000 | -5.000 | CRLB | 3.313 |  |  | 1000 |
| 6.000 | 0.000 | Root-MUSIC | 0.557 | 0.443 | 1.000 | 1000 |
| 6.000 | 0.000 | ESPRIT | 0.688 | 0.534 | 0.999 | 1000 |
| 6.000 | 0.000 | ReconUNet | 15.997 | 2.598 | 0.441 | 1000 |
| 6.000 | 0.000 | SubspaceNet | 20.214 | 12.065 | 0.086 | 1000 |
| 6.000 | 0.000 | DA-MUSIC | 6.162 | 5.030 | 0.042 | 1000 |
| 6.000 | 0.000 | ReconUNet-C | 5.315 | 1.681 | 0.730 | 1000 |
| 6.000 | 0.000 | CRLB | 0.180 |  |  | 1000 |
| 6.000 | -5.000 | Root-MUSIC | 5.533 | 0.834 | 0.952 | 1000 |
| 6.000 | -5.000 | ESPRIT | 1.480 | 1.020 | 0.926 | 1000 |
| 6.000 | -5.000 | ReconUNet | 15.616 | 2.629 | 0.446 | 1000 |
| 6.000 | -5.000 | SubspaceNet | 18.057 | 12.077 | 0.104 | 1000 |
| 6.000 | -5.000 | DA-MUSIC | 6.243 | 5.035 | 0.045 | 1000 |
| 6.000 | -5.000 | ReconUNet-C | 5.766 | 1.632 | 0.712 | 1000 |
| 6.000 | -5.000 | CRLB | 0.916 |  |  | 1000 |
| 8.000 | 0.000 | Root-MUSIC | 0.421 | 0.332 | 1.000 | 1000 |
| 8.000 | 0.000 | ESPRIT | 0.569 | 0.434 | 1.000 | 1000 |
| 8.000 | 0.000 | ReconUNet | 2.825 | 1.323 | 0.951 | 1000 |
| 8.000 | 0.000 | SubspaceNet | 14.374 | 4.025 | 0.393 | 1000 |
| 8.000 | 0.000 | DA-MUSIC | 4.671 | 3.961 | 0.282 | 1000 |
| 8.000 | 0.000 | ReconUNet-C | 1.322 | 1.063 | 0.987 | 1000 |
| 8.000 | 0.000 | CRLB | 0.090 |  |  | 1000 |
| 8.000 | -5.000 | Root-MUSIC | 0.731 | 0.530 | 1.000 | 1000 |
| 8.000 | -5.000 | ESPRIT | 0.926 | 0.700 | 0.998 | 1000 |
| 8.000 | -5.000 | ReconUNet | 3.374 | 1.377 | 0.942 | 1000 |
| 8.000 | -5.000 | SubspaceNet | 13.824 | 3.771 | 0.417 | 1000 |
| 8.000 | -5.000 | DA-MUSIC | 4.888 | 4.071 | 0.254 | 1000 |
| 8.000 | -5.000 | ReconUNet-C | 1.332 | 1.069 | 0.987 | 1000 |
| 8.000 | -5.000 | CRLB | 0.403 |  |  | 1000 |
| 10.000 | 0.000 | Root-MUSIC | 0.345 | 0.261 | 1.000 | 1000 |
| 10.000 | 0.000 | ESPRIT | 0.504 | 0.372 | 1.000 | 1000 |
| 10.000 | 0.000 | ReconUNet | 1.159 | 0.941 | 1.000 | 1000 |
| 10.000 | 0.000 | SubspaceNet | 11.704 | 2.235 | 0.705 | 1000 |
| 10.000 | 0.000 | DA-MUSIC | 3.299 | 2.900 | 0.790 | 1000 |
| 10.000 | 0.000 | ReconUNet-C | 0.673 | 0.502 | 1.000 | 1000 |
| 10.000 | 0.000 | CRLB | 0.055 |  |  | 1000 |
| 10.000 | -5.000 | Root-MUSIC | 0.534 | 0.404 | 1.000 | 1000 |
| 10.000 | -5.000 | ESPRIT | 0.735 | 0.547 | 1.000 | 1000 |
| 10.000 | -5.000 | ReconUNet | 1.199 | 0.942 | 0.999 | 1000 |
| 10.000 | -5.000 | SubspaceNet | 9.870 | 2.207 | 0.737 | 1000 |
| 10.000 | -5.000 | DA-MUSIC | 3.505 | 3.100 | 0.732 | 1000 |
| 10.000 | -5.000 | ReconUNet-C | 0.719 | 0.534 | 1.000 | 1000 |
| 10.000 | -5.000 | CRLB | 0.232 |  |  | 1000 |
| 15.000 | 0.000 | Root-MUSIC | 0.267 | 0.205 | 1.000 | 1000 |
| 15.000 | 0.000 | ESPRIT | 0.381 | 0.267 | 1.000 | 1000 |
| 15.000 | 0.000 | ReconUNet | 0.675 | 0.532 | 1.000 | 1000 |
| 15.000 | 0.000 | SubspaceNet | 5.986 | 1.176 | 0.920 | 1000 |
| 15.000 | 0.000 | DA-MUSIC | 1.907 | 1.463 | 0.998 | 1000 |
| 15.000 | 0.000 | ReconUNet-C | 0.338 | 0.258 | 1.000 | 1000 |
| 15.000 | 0.000 | CRLB | 0.023 |  |  | 1000 |
| 15.000 | -5.000 | Root-MUSIC | 0.378 | 0.289 | 1.000 | 1000 |
| 15.000 | -5.000 | ESPRIT | 0.507 | 0.374 | 1.000 | 1000 |
| 15.000 | -5.000 | ReconUNet | 0.713 | 0.556 | 1.000 | 1000 |
| 15.000 | -5.000 | SubspaceNet | 7.453 | 1.207 | 0.956 | 1000 |
| 15.000 | -5.000 | DA-MUSIC | 2.039 | 1.627 | 0.998 | 1000 |
| 15.000 | -5.000 | ReconUNet-C | 0.423 | 0.335 | 1.000 | 1000 |
| 15.000 | -5.000 | CRLB | 0.092 |  |  | 1000 |

## R2 (2026-09-30) — Cost: ReconUNet-C vs ReconUNet (same measurement as item 8)

_Appended 2026-09-30 21:45 UTC from `experiments/runs/sweeps_r2_20260930/cost_summary_r2.csv`._

| model | class | trainable parameters | MMACs per scene | GPU fwd b1 ms/scene | GPU fwd+Root-MUSIC b1 ms/scene | CPU fwd b1 ms | GPU fwd b1024 ms/scene | GPU fwd+Root-MUSIC b1024 ms/scene | epochs | best_epoch | train_wallclock_min |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ReconUNet | EVDCovarianceReconstructionUNet | 337842 | 7.782 | 1.155 | 1.741 | 2.548 | 0.032 | 0.071 | 109 | 84 | 287 |
| ReconUNet-C | CovarianceOnlyReconstructionUNet | 323070 | 7.128 | 0.758 | 1.330 | 0.487 | 0.014 | 0.058 | 207 | 182 | 531 |

## R2 (2026-09-30) — Cost: latency detail

_Appended 2026-09-30 21:45 UTC from `experiments/runs/sweeps_r2_20260930/cost_latency_r2.csv`._

| pipeline | batch | ms per scene | ms per batch | peak GPU MiB |
|---|---|---|---|---|
| ReconUNet forward only (GPU) | 1 | 1.155 | 1.155 | 16.456 |
| ReconUNet U-Net only, base_unet (GPU) | 1 | 0.616 | 0.616 | 10.766 |
| ReconUNet + Root-MUSIC (GPU net, CPU eigh/roots) | 1 | 1.741 | 1.741 | 16.460 |
| ReconUNet forward only (CPU, 20 threads) | 1 | 2.548 | 2.548 |  |
| ReconUNet forward only (GPU) | 1024 | 0.032 | 33.075 | 144.447 |
| ReconUNet U-Net only, base_unet (GPU) | 1024 | 0.011 | 10.810 | 144.447 |
| ReconUNet + Root-MUSIC (GPU net, CPU eigh/roots) | 1024 | 0.071 | 72.721 | 148.447 |
| ReconUNet-C forward only (GPU) | 1 | 0.758 | 0.758 | 10.702 |
| ReconUNet-C U-Net only, base_unet (GPU) | 1 | 0.631 | 0.631 | 10.702 |
| ReconUNet-C + Root-MUSIC (GPU net, CPU eigh/roots) | 1 | 1.330 | 1.330 | 10.706 |
| ReconUNet-C forward only (CPU, 20 threads) | 1 | 0.487 | 0.487 |  |
| ReconUNet-C forward only (GPU) | 1024 | 0.014 | 13.858 | 144.383 |
| ReconUNet-C U-Net only, base_unet (GPU) | 1024 | 0.009 | 9.528 | 144.383 |
| ReconUNet-C + Root-MUSIC (GPU net, CPU eigh/roots) | 1024 | 0.058 | 58.989 | 148.383 |

## R2 (2026-09-30) — MUSIC verification incl. ReconUNet-C

_Appended 2026-09-30 21:45 UTC._

Basic scenario (K = 1, no multipath, mild imperfections), 1000 scenes per SNR. MUSIC uses the paper's 1° scan grid on [-60°, 60°]; 'refined' adds the three-point parabolic peak refinement; Root-MUSIC is gridless. A spurious peak is a scene whose selected maximum is more than 1°/3°/5° from the true angle.

| SNR (dB) | estimator | median abs err (°) | pooled RMSE (°) | mean abs err (°) | frac > 1° | frac > 3° | frac > 5° | n |
|---|---|---|---|---|---|---|---|---|
| 20 | MUSIC-grid | 0.269 | 0.371 | 0.305 | 0.0020 | 0.0000 | 0.0000 | 1000 |
| 20 | MUSIC-refined | 0.192 | 5.302 | 0.561 | 0.0060 | 0.0060 | 0.0060 | 1000 |
| 20 | Root-MUSIC | 0.147 | 0.229 | 0.182 | 0.0000 | 0.0000 | 0.0000 | 1000 |
| 20 | ReconUNet-C+MUSIC-grid | 0.270 | 0.382 | 0.312 | 0.0040 | 0.0000 | 0.0000 | 1000 |
| 20 | ReconUNet-C+MUSIC-refined | 0.224 | 0.985 | 0.288 | 0.0040 | 0.0010 | 0.0010 | 1000 |
| 20 | ReconUNet-C+Root-MUSIC | 0.163 | 0.249 | 0.197 | 0.0010 | 0.0000 | 0.0000 | 1000 |
| 0 | MUSIC-grid | 0.270 | 0.380 | 0.310 | 0.0060 | 0.0000 | 0.0000 | 1000 |
| 0 | MUSIC-refined | 0.188 | 3.972 | 0.450 | 0.0060 | 0.0050 | 0.0050 | 1000 |
| 0 | Root-MUSIC | 0.163 | 0.251 | 0.198 | 0.0000 | 0.0000 | 0.0000 | 1000 |
| 0 | ReconUNet-C+MUSIC-grid | 0.290 | 0.415 | 0.337 | 0.0120 | 0.0000 | 0.0000 | 1000 |
| 0 | ReconUNet-C+MUSIC-refined | 0.245 | 4.972 | 0.509 | 0.0080 | 0.0020 | 0.0020 | 1000 |
| 0 | ReconUNet-C+Root-MUSIC | 0.182 | 0.288 | 0.225 | 0.0030 | 0.0000 | 0.0000 | 1000 |

Reading: at 20 dB the refined grid-MUSIC median error is 0.192° against 0.147° for Root-MUSIC, i.e. the estimator itself is fine; the pooled RMSE gap (5.302° vs 0.229°) is driven by the 0.60 % of scenes whose selected peak is spurious (> 3°), which RMSE squares. Without refinement the grid alone contributes a 0.269° median quantisation error.

Data: `experiments/runs/sweeps_r2_20260930/music_verification.csv`.

## R2 (2026-09-30) — Source-bandwidth sweep, replica delays fixed at the training bandwidth 0.05 (primary)

_Appended 2026-09-30 21:45 UTC from `experiments/runs/sweeps_r2_20260930/bandwidth_sweep.csv`._

0 dB, mild imperfections, T = 512, 1000 scenes per scenario; identical scenes (seeds, angles, replica AoAs/gains/delays, imperfections, noise) at every bandwidth, only the source spectrum changes. moderate = paper Moderate manifest (2 direct + 1 replica), crowded = 4 + 3, moderate3 = seeded 3 + 1 draw. Models trained at bw 0.05 only.

| scenario | bw_frac | method | n_scenes | rmse_deg | rmse_ci_lo | rmse_ci_hi | median_rmspe_deg | median_ci_lo | median_ci_hi |
|---|---|---|---|---|---|---|---|---|---|
| moderate | 0.01 | Root-MUSIC | 1000 | 5.907 | 4.021 | 7.631 | 0.531 | 0.491 | 0.586 |
| moderate | 0.01 | ReconUNet | 1000 | 2.013 | 1.271 | 2.915 | 0.775 | 0.719 | 0.826 |
| moderate | 0.01 | ReconUNet-C | 1000 | 2.053 | 1.194 | 2.763 | 0.528 | 0.498 | 0.566 |
| moderate | 0.01 | SubspaceNet | 1000 | 5.333 | 4.293 | 6.337 | 1.421 | 1.342 | 1.521 |
| moderate | 0.01 | DA-MUSIC | 1000 | 6.719 | 6.038 | 7.359 | 3.555 | 3.381 | 3.800 |
| moderate | 0.01 | SubViT | 1000 | 10.075 | 8.270 | 11.808 | 0.415 | 0.400 | 0.443 |
| moderate | 0.02 | Root-MUSIC | 1000 | 5.930 | 3.857 | 7.675 | 0.531 | 0.487 | 0.577 |
| moderate | 0.02 | ReconUNet | 1000 | 1.851 | 1.085 | 2.799 | 0.666 | 0.629 | 0.698 |
| moderate | 0.02 | ReconUNet-C | 1000 | 0.817 | 0.744 | 0.901 | 0.424 | 0.395 | 0.452 |
| moderate | 0.02 | SubspaceNet | 1000 | 4.364 | 3.107 | 5.682 | 1.224 | 1.169 | 1.280 |
| moderate | 0.02 | DA-MUSIC | 1000 | 4.784 | 4.196 | 5.344 | 2.594 | 2.430 | 2.752 |
| moderate | 0.02 | SubViT | 1000 | 7.367 | 5.684 | 8.886 | 0.405 | 0.384 | 0.419 |
| moderate | 0.05 | Root-MUSIC | 1000 | 5.915 | 4.001 | 7.653 | 0.486 | 0.457 | 0.521 |
| moderate | 0.05 | ReconUNet | 1000 | 1.790 | 0.978 | 2.767 | 0.616 | 0.579 | 0.646 |
| moderate | 0.05 | ReconUNet-C | 1000 | 0.608 | 0.560 | 0.658 | 0.369 | 0.354 | 0.385 |
| moderate | 0.05 | SubspaceNet | 1000 | 2.918 | 2.300 | 3.516 | 1.035 | 0.984 | 1.091 |
| moderate | 0.05 | DA-MUSIC | 1000 | 3.600 | 3.167 | 4.101 | 2.143 | 2.030 | 2.262 |
| moderate | 0.05 | SubViT | 1000 | 8.724 | 7.073 | 10.273 | 0.380 | 0.365 | 0.404 |
| moderate | 0.1 | Root-MUSIC | 1000 | 5.665 | 3.730 | 7.302 | 0.419 | 0.393 | 0.450 |
| moderate | 0.1 | ReconUNet | 1000 | 1.939 | 0.990 | 3.034 | 0.630 | 0.607 | 0.668 |
| moderate | 0.1 | ReconUNet-C | 1000 | 1.648 | 0.768 | 2.606 | 0.475 | 0.454 | 0.500 |
| moderate | 0.1 | SubspaceNet | 1000 | 3.735 | 2.522 | 4.908 | 0.988 | 0.942 | 1.041 |
| moderate | 0.1 | DA-MUSIC | 1000 | 3.328 | 2.882 | 3.875 | 1.912 | 1.811 | 1.991 |
| moderate | 0.1 | SubViT | 1000 | 7.093 | 5.271 | 8.786 | 0.378 | 0.362 | 0.395 |
| moderate | 0.2 | Root-MUSIC | 1000 | 5.906 | 3.942 | 7.627 | 0.372 | 0.339 | 0.405 |
| moderate | 0.2 | ReconUNet | 1000 | 14.083 | 12.574 | 15.704 | 2.454 | 2.246 | 2.665 |
| moderate | 0.2 | ReconUNet-C | 1000 | 22.704 | 21.103 | 24.274 | 3.307 | 3.096 | 3.630 |
| moderate | 0.2 | SubspaceNet | 1000 | 5.869 | 4.173 | 7.224 | 1.238 | 1.185 | 1.285 |
| moderate | 0.2 | DA-MUSIC | 1000 | 3.117 | 2.697 | 3.662 | 1.748 | 1.682 | 1.843 |
| moderate | 0.2 | SubViT | 1000 | 5.347 | 3.570 | 6.861 | 0.371 | 0.357 | 0.386 |
| moderate | 0.4 | Root-MUSIC | 1000 | 5.626 | 3.710 | 7.335 | 0.348 | 0.325 | 0.374 |
| moderate | 0.4 | ReconUNet | 1000 | 24.654 | 23.750 | 25.614 | 18.866 | 17.271 | 20.320 |
| moderate | 0.4 | ReconUNet-C | 1000 | 26.419 | 25.096 | 27.766 | 7.323 | 6.593 | 8.204 |
| moderate | 0.4 | SubspaceNet | 1000 | 8.101 | 6.765 | 9.478 | 1.560 | 1.465 | 1.635 |
| moderate | 0.4 | DA-MUSIC | 1000 | 2.982 | 2.579 | 3.550 | 1.685 | 1.582 | 1.829 |
| moderate | 0.4 | SubViT | 1000 | 5.745 | 3.908 | 7.396 | 0.370 | 0.357 | 0.385 |
| moderate | white | Root-MUSIC | 1000 | 5.273 | 3.032 | 7.103 | 0.326 | 0.308 | 0.360 |
| moderate | white | ReconUNet | 1000 | 32.712 | 31.602 | 33.826 | 23.770 | 22.276 | 25.213 |
| moderate | white | ReconUNet-C | 1000 | 30.879 | 29.514 | 32.160 | 17.330 | 15.399 | 19.186 |
| moderate | white | SubspaceNet | 1000 | 11.640 | 10.336 | 12.924 | 2.896 | 2.733 | 3.076 |
| moderate | white | DA-MUSIC | 1000 | 2.838 | 2.576 | 3.146 | 1.681 | 1.622 | 1.759 |
| moderate | white | SubViT | 1000 | 6.909 | 5.112 | 8.630 | 0.384 | 0.368 | 0.399 |
| crowded | 0.01 | Root-MUSIC | 1000 | 9.513 | 8.649 | 10.345 | 1.550 | 1.427 | 1.659 |
| crowded | 0.01 | ReconUNet | 1000 | 9.035 | 8.271 | 9.739 | 2.463 | 2.305 | 2.563 |
| crowded | 0.01 | ReconUNet-C | 1000 | 5.932 | 5.158 | 6.700 | 1.775 | 1.700 | 1.840 |
| crowded | 0.01 | SubspaceNet | 1000 | 11.274 | 10.711 | 11.764 | 5.199 | 4.926 | 5.661 |
| crowded | 0.01 | DA-MUSIC | 1000 | 8.328 | 7.859 | 8.780 | 5.235 | 5.030 | 5.451 |
| crowded | 0.01 | SubViT | 1000 | 20.247 | 19.489 | 20.969 | 15.784 | 14.242 | 17.087 |
| crowded | 0.02 | Root-MUSIC | 1000 | 9.131 | 8.333 | 9.908 | 1.381 | 1.245 | 1.533 |
| crowded | 0.02 | ReconUNet | 1000 | 5.816 | 5.077 | 6.531 | 1.808 | 1.722 | 1.913 |
| crowded | 0.02 | ReconUNet-C | 1000 | 3.486 | 2.821 | 4.163 | 1.150 | 1.098 | 1.207 |
| crowded | 0.02 | SubspaceNet | 1000 | 9.203 | 8.577 | 9.865 | 3.431 | 3.268 | 3.609 |
| crowded | 0.02 | DA-MUSIC | 1000 | 7.405 | 6.968 | 7.861 | 4.562 | 4.344 | 4.717 |
| crowded | 0.02 | SubViT | 1000 | 17.945 | 17.210 | 18.690 | 10.047 | 8.717 | 11.607 |
| crowded | 0.05 | Root-MUSIC | 1000 | 8.871 | 8.068 | 9.665 | 1.291 | 1.162 | 1.396 |
| crowded | 0.05 | ReconUNet | 1000 | 4.804 | 4.111 | 5.548 | 1.430 | 1.366 | 1.499 |
| crowded | 0.05 | ReconUNet-C | 1000 | 2.446 | 1.751 | 3.128 | 0.903 | 0.872 | 0.955 |
| crowded | 0.05 | SubspaceNet | 1000 | 7.912 | 7.315 | 8.476 | 2.786 | 2.668 | 2.929 |
| crowded | 0.05 | DA-MUSIC | 1000 | 6.652 | 6.253 | 7.030 | 4.126 | 3.969 | 4.341 |
| crowded | 0.05 | SubViT | 1000 | 16.321 | 15.578 | 17.103 | 5.159 | 1.776 | 6.578 |
| crowded | 0.1 | Root-MUSIC | 1000 | 8.799 | 7.895 | 9.601 | 1.139 | 1.049 | 1.214 |
| crowded | 0.1 | ReconUNet | 1000 | 5.003 | 4.221 | 5.762 | 1.407 | 1.348 | 1.473 |
| crowded | 0.1 | ReconUNet-C | 1000 | 3.013 | 2.329 | 3.668 | 0.999 | 0.951 | 1.046 |
| crowded | 0.1 | SubspaceNet | 1000 | 8.585 | 7.884 | 9.219 | 2.515 | 2.360 | 2.644 |
| crowded | 0.1 | DA-MUSIC | 1000 | 6.140 | 5.751 | 6.552 | 3.759 | 3.571 | 3.886 |
| crowded | 0.1 | SubViT | 1000 | 14.394 | 13.543 | 15.214 | 0.886 | 0.794 | 1.059 |
| crowded | 0.2 | Root-MUSIC | 1000 | 8.777 | 7.947 | 9.606 | 1.092 | 1.033 | 1.214 |
| crowded | 0.2 | ReconUNet | 1000 | 14.517 | 13.830 | 15.192 | 4.899 | 4.180 | 5.533 |
| crowded | 0.2 | ReconUNet-C | 1000 | 16.674 | 15.981 | 17.312 | 10.865 | 8.853 | 12.270 |
| crowded | 0.2 | SubspaceNet | 1000 | 10.741 | 10.078 | 11.353 | 3.167 | 3.023 | 3.392 |
| crowded | 0.2 | DA-MUSIC | 1000 | 5.829 | 5.446 | 6.235 | 3.543 | 3.392 | 3.664 |
| crowded | 0.2 | SubViT | 1000 | 14.130 | 13.293 | 14.901 | 0.855 | 0.773 | 0.946 |
| crowded | 0.4 | Root-MUSIC | 1000 | 8.018 | 7.181 | 8.873 | 1.072 | 1.007 | 1.173 |
| crowded | 0.4 | ReconUNet | 1000 | 22.500 | 21.847 | 23.164 | 19.473 | 18.577 | 20.264 |
| crowded | 0.4 | ReconUNet-C | 1000 | 19.884 | 19.228 | 20.547 | 15.561 | 14.982 | 16.332 |
| crowded | 0.4 | SubspaceNet | 1000 | 11.613 | 10.940 | 12.326 | 3.958 | 3.699 | 4.292 |
| crowded | 0.4 | DA-MUSIC | 1000 | 5.414 | 5.062 | 5.796 | 3.388 | 3.229 | 3.567 |
| crowded | 0.4 | SubViT | 1000 | 14.238 | 13.337 | 15.173 | 0.854 | 0.789 | 0.935 |
| crowded | white | Root-MUSIC | 1000 | 7.565 | 6.685 | 8.277 | 1.093 | 1.000 | 1.202 |
| crowded | white | ReconUNet | 1000 | 22.281 | 21.584 | 22.971 | 18.222 | 17.509 | 18.841 |
| crowded | white | ReconUNet-C | 1000 | 20.520 | 19.936 | 21.140 | 17.353 | 16.667 | 18.161 |
| crowded | white | SubspaceNet | 1000 | 17.557 | 17.025 | 18.105 | 14.575 | 13.426 | 15.619 |
| crowded | white | DA-MUSIC | 1000 | 5.552 | 5.173 | 5.935 | 3.457 | 3.270 | 3.625 |
| crowded | white | SubViT | 1000 | 14.257 | 13.322 | 15.092 | 0.954 | 0.858 | 1.075 |
| moderate3 | 0.01 | Root-MUSIC | 1000 | 4.859 | 3.589 | 6.110 | 0.577 | 0.521 | 0.622 |
| moderate3 | 0.01 | ReconUNet | 1000 | 4.691 | 3.662 | 5.753 | 1.190 | 1.145 | 1.252 |
| moderate3 | 0.01 | ReconUNet-C | 1000 | 3.372 | 1.960 | 4.592 | 0.760 | 0.721 | 0.801 |
| moderate3 | 0.01 | SubspaceNet | 1000 | 7.680 | 6.847 | 8.467 | 2.240 | 2.161 | 2.334 |
| moderate3 | 0.01 | DA-MUSIC | 1000 | 6.289 | 5.828 | 6.804 | 3.978 | 3.837 | 4.183 |
| moderate3 | 0.01 | SubViT | 1000 | 15.211 | 14.003 | 16.357 | 0.583 | 0.538 | 0.616 |
| moderate3 | 0.02 | Root-MUSIC | 1000 | 4.535 | 3.193 | 5.812 | 0.528 | 0.494 | 0.571 |
| moderate3 | 0.02 | ReconUNet | 1000 | 2.819 | 1.524 | 3.848 | 0.864 | 0.823 | 0.911 |
| moderate3 | 0.02 | ReconUNet-C | 1000 | 2.045 | 0.798 | 3.341 | 0.556 | 0.534 | 0.574 |
| moderate3 | 0.02 | SubspaceNet | 1000 | 5.664 | 4.773 | 6.520 | 1.633 | 1.542 | 1.705 |
| moderate3 | 0.02 | DA-MUSIC | 1000 | 4.637 | 4.265 | 5.036 | 2.986 | 2.843 | 3.109 |
| moderate3 | 0.02 | SubViT | 1000 | 11.540 | 10.348 | 12.690 | 0.463 | 0.441 | 0.482 |
| moderate3 | 0.05 | Root-MUSIC | 1000 | 4.641 | 3.308 | 5.842 | 0.521 | 0.481 | 0.564 |
| moderate3 | 0.05 | ReconUNet | 1000 | 2.897 | 1.695 | 3.911 | 0.755 | 0.718 | 0.794 |
| moderate3 | 0.05 | ReconUNet-C | 1000 | 0.670 | 0.633 | 0.710 | 0.448 | 0.432 | 0.467 |
| moderate3 | 0.05 | SubspaceNet | 1000 | 5.459 | 4.449 | 6.468 | 1.344 | 1.285 | 1.417 |
| moderate3 | 0.05 | DA-MUSIC | 1000 | 3.785 | 3.531 | 4.062 | 2.577 | 2.440 | 2.685 |
| moderate3 | 0.05 | SubViT | 1000 | 10.962 | 9.662 | 12.265 | 0.429 | 0.409 | 0.448 |
| moderate3 | 0.1 | Root-MUSIC | 1000 | 4.461 | 3.125 | 5.613 | 0.447 | 0.416 | 0.478 |
| moderate3 | 0.1 | ReconUNet | 1000 | 3.131 | 1.822 | 4.373 | 0.762 | 0.737 | 0.805 |
| moderate3 | 0.1 | ReconUNet-C | 1000 | 1.626 | 0.787 | 2.572 | 0.583 | 0.557 | 0.603 |
| moderate3 | 0.1 | SubspaceNet | 1000 | 5.553 | 4.571 | 6.483 | 1.225 | 1.185 | 1.312 |
| moderate3 | 0.1 | DA-MUSIC | 1000 | 3.470 | 3.227 | 3.754 | 2.389 | 2.294 | 2.469 |
| moderate3 | 0.1 | SubViT | 1000 | 8.035 | 6.710 | 9.132 | 0.418 | 0.405 | 0.437 |
| moderate3 | 0.2 | Root-MUSIC | 1000 | 3.701 | 2.586 | 4.708 | 0.423 | 0.393 | 0.442 |
| moderate3 | 0.2 | ReconUNet | 1000 | 12.484 | 11.446 | 13.413 | 2.754 | 2.579 | 2.879 |
| moderate3 | 0.2 | ReconUNet-C | 1000 | 19.290 | 18.310 | 20.324 | 5.037 | 4.538 | 5.848 |
| moderate3 | 0.2 | SubspaceNet | 1000 | 8.937 | 7.801 | 9.978 | 1.579 | 1.497 | 1.648 |
| moderate3 | 0.2 | DA-MUSIC | 1000 | 3.366 | 3.099 | 3.635 | 2.215 | 2.137 | 2.286 |
| moderate3 | 0.2 | SubViT | 1000 | 7.945 | 6.541 | 9.217 | 0.399 | 0.384 | 0.418 |
| moderate3 | 0.4 | Root-MUSIC | 1000 | 3.750 | 2.410 | 4.986 | 0.395 | 0.378 | 0.430 |
| moderate3 | 0.4 | ReconUNet | 1000 | 24.715 | 23.972 | 25.421 | 21.609 | 20.639 | 22.601 |
| moderate3 | 0.4 | ReconUNet-C | 1000 | 21.975 | 21.058 | 22.891 | 14.384 | 12.617 | 15.922 |
| moderate3 | 0.4 | SubspaceNet | 1000 | 10.476 | 9.432 | 11.525 | 2.138 | 2.007 | 2.255 |
| moderate3 | 0.4 | DA-MUSIC | 1000 | 3.255 | 2.983 | 3.570 | 2.179 | 2.097 | 2.294 |
| moderate3 | 0.4 | SubViT | 1000 | 7.407 | 6.174 | 8.611 | 0.395 | 0.382 | 0.414 |
| moderate3 | white | Root-MUSIC | 1000 | 3.719 | 2.369 | 4.933 | 0.383 | 0.358 | 0.406 |
| moderate3 | white | ReconUNet | 1000 | 25.894 | 25.129 | 26.686 | 21.848 | 21.108 | 22.693 |
| moderate3 | white | ReconUNet-C | 1000 | 24.569 | 23.701 | 25.459 | 18.854 | 17.365 | 19.876 |
| moderate3 | white | SubspaceNet | 1000 | 14.102 | 13.300 | 14.948 | 4.128 | 3.966 | 4.335 |
| moderate3 | white | DA-MUSIC | 1000 | 3.320 | 3.013 | 3.676 | 2.172 | 2.101 | 2.320 |
| moderate3 | white | SubViT | 1000 | 8.030 | 6.744 | 9.223 | 0.413 | 0.392 | 0.432 |

## R2 (2026-09-30) — Source-bandwidth sweep, coupled (replica delays ∝ 1/bw as in the renderer; secondary)

_Appended 2026-09-30 21:45 UTC from `experiments/runs/sweeps_r2_20260930/bandwidth_sweep_coupled.csv`._

White sources use the renderer's legacy 0.05 delay range in both modes, so the white rows are identical.

| scenario | bw_frac | method | n_scenes | rmse_deg | rmse_ci_lo | rmse_ci_hi | median_rmspe_deg | median_ci_lo | median_ci_hi |
|---|---|---|---|---|---|---|---|---|---|
| moderate | 0.01 | Root-MUSIC | 1000 | 5.640 | 3.766 | 7.275 | 0.574 | 0.534 | 0.614 |
| moderate | 0.01 | ReconUNet | 1000 | 2.426 | 1.303 | 3.448 | 0.748 | 0.708 | 0.800 |
| moderate | 0.01 | ReconUNet-C | 1000 | 1.224 | 0.942 | 1.536 | 0.480 | 0.460 | 0.510 |
| moderate | 0.01 | SubspaceNet | 1000 | 5.848 | 4.644 | 7.054 | 1.397 | 1.280 | 1.470 |
| moderate | 0.01 | DA-MUSIC | 1000 | 6.478 | 5.927 | 7.026 | 3.534 | 3.315 | 3.770 |
| moderate | 0.01 | SubViT | 1000 | 10.147 | 8.095 | 11.995 | 0.414 | 0.398 | 0.444 |
| moderate | 0.02 | Root-MUSIC | 1000 | 6.909 | 4.894 | 8.679 | 0.518 | 0.487 | 0.556 |
| moderate | 0.02 | ReconUNet | 1000 | 2.119 | 1.033 | 3.350 | 0.654 | 0.607 | 0.703 |
| moderate | 0.02 | ReconUNet-C | 1000 | 2.094 | 0.662 | 3.452 | 0.406 | 0.383 | 0.433 |
| moderate | 0.02 | SubspaceNet | 1000 | 4.131 | 2.777 | 5.436 | 1.153 | 1.085 | 1.232 |
| moderate | 0.02 | DA-MUSIC | 1000 | 4.808 | 4.248 | 5.432 | 2.514 | 2.365 | 2.660 |
| moderate | 0.02 | SubViT | 1000 | 7.929 | 6.135 | 9.459 | 0.398 | 0.382 | 0.412 |
| moderate | 0.05 | Root-MUSIC | 1000 | 5.915 | 4.001 | 7.653 | 0.486 | 0.457 | 0.521 |
| moderate | 0.05 | ReconUNet | 1000 | 1.790 | 0.978 | 2.767 | 0.616 | 0.579 | 0.646 |
| moderate | 0.05 | ReconUNet-C | 1000 | 0.608 | 0.560 | 0.658 | 0.369 | 0.354 | 0.385 |
| moderate | 0.05 | SubspaceNet | 1000 | 2.918 | 2.300 | 3.516 | 1.035 | 0.984 | 1.091 |
| moderate | 0.05 | DA-MUSIC | 1000 | 3.600 | 3.167 | 4.101 | 2.143 | 2.030 | 2.262 |
| moderate | 0.05 | SubViT | 1000 | 8.724 | 7.073 | 10.273 | 0.380 | 0.365 | 0.404 |
| moderate | 0.1 | Root-MUSIC | 1000 | 5.804 | 3.455 | 7.599 | 0.502 | 0.465 | 0.546 |
| moderate | 0.1 | ReconUNet | 1000 | 1.337 | 1.030 | 1.778 | 0.650 | 0.626 | 0.698 |
| moderate | 0.1 | ReconUNet-C | 1000 | 0.896 | 0.751 | 1.094 | 0.486 | 0.462 | 0.508 |
| moderate | 0.1 | SubspaceNet | 1000 | 3.577 | 2.512 | 4.770 | 1.066 | 0.996 | 1.113 |
| moderate | 0.1 | DA-MUSIC | 1000 | 3.020 | 2.788 | 3.270 | 1.895 | 1.795 | 2.017 |
| moderate | 0.1 | SubViT | 1000 | 8.471 | 6.492 | 10.078 | 0.381 | 0.364 | 0.401 |
| moderate | 0.2 | Root-MUSIC | 1000 | 7.292 | 5.160 | 9.157 | 0.502 | 0.471 | 0.556 |
| moderate | 0.2 | ReconUNet | 1000 | 13.759 | 12.125 | 15.251 | 2.246 | 2.088 | 2.425 |
| moderate | 0.2 | ReconUNet-C | 1000 | 24.950 | 23.418 | 26.374 | 3.889 | 3.617 | 4.159 |
| moderate | 0.2 | SubspaceNet | 1000 | 7.196 | 5.668 | 8.589 | 1.340 | 1.291 | 1.413 |
| moderate | 0.2 | DA-MUSIC | 1000 | 3.180 | 2.765 | 3.758 | 1.853 | 1.753 | 1.949 |
| moderate | 0.2 | SubViT | 1000 | 8.217 | 6.377 | 9.910 | 0.384 | 0.364 | 0.400 |
| moderate | 0.4 | Root-MUSIC | 1000 | 6.815 | 4.644 | 8.808 | 0.498 | 0.453 | 0.535 |
| moderate | 0.4 | ReconUNet | 1000 | 25.610 | 24.561 | 26.556 | 20.764 | 19.052 | 21.913 |
| moderate | 0.4 | ReconUNet-C | 1000 | 25.339 | 23.936 | 26.853 | 6.386 | 5.935 | 6.870 |
| moderate | 0.4 | SubspaceNet | 1000 | 8.608 | 7.159 | 9.993 | 1.652 | 1.562 | 1.726 |
| moderate | 0.4 | DA-MUSIC | 1000 | 3.040 | 2.661 | 3.550 | 1.748 | 1.656 | 1.814 |
| moderate | 0.4 | SubViT | 1000 | 7.910 | 6.003 | 9.709 | 0.385 | 0.364 | 0.400 |
| moderate | white | Root-MUSIC | 1000 | 5.273 | 3.032 | 7.103 | 0.326 | 0.308 | 0.360 |
| moderate | white | ReconUNet | 1000 | 32.712 | 31.602 | 33.826 | 23.770 | 22.276 | 25.213 |
| moderate | white | ReconUNet-C | 1000 | 30.879 | 29.514 | 32.160 | 17.330 | 15.399 | 19.186 |
| moderate | white | SubspaceNet | 1000 | 11.640 | 10.336 | 12.924 | 2.896 | 2.733 | 3.076 |
| moderate | white | DA-MUSIC | 1000 | 2.838 | 2.576 | 3.146 | 1.681 | 1.622 | 1.759 |
| moderate | white | SubViT | 1000 | 6.909 | 5.112 | 8.630 | 0.384 | 0.368 | 0.399 |
| crowded | 0.01 | Root-MUSIC | 1000 | 10.250 | 9.500 | 11.074 | 1.792 | 1.641 | 1.931 |
| crowded | 0.01 | ReconUNet | 1000 | 9.012 | 8.252 | 9.711 | 2.489 | 2.375 | 2.607 |
| crowded | 0.01 | ReconUNet-C | 1000 | 6.407 | 5.622 | 7.196 | 1.755 | 1.656 | 1.842 |
| crowded | 0.01 | SubspaceNet | 1000 | 11.446 | 10.873 | 11.938 | 5.292 | 4.930 | 5.786 |
| crowded | 0.01 | DA-MUSIC | 1000 | 8.237 | 7.771 | 8.678 | 5.246 | 4.970 | 5.434 |
| crowded | 0.01 | SubViT | 1000 | 19.709 | 18.977 | 20.418 | 14.408 | 13.214 | 15.954 |
| crowded | 0.02 | Root-MUSIC | 1000 | 9.432 | 8.664 | 10.222 | 1.401 | 1.310 | 1.514 |
| crowded | 0.02 | ReconUNet | 1000 | 5.793 | 5.037 | 6.554 | 1.708 | 1.656 | 1.795 |
| crowded | 0.02 | ReconUNet-C | 1000 | 3.103 | 2.484 | 3.690 | 1.149 | 1.099 | 1.204 |
| crowded | 0.02 | SubspaceNet | 1000 | 9.137 | 8.530 | 9.760 | 3.352 | 3.180 | 3.553 |
| crowded | 0.02 | DA-MUSIC | 1000 | 7.091 | 6.673 | 7.515 | 4.441 | 4.184 | 4.679 |
| crowded | 0.02 | SubViT | 1000 | 17.084 | 16.227 | 17.956 | 7.874 | 6.574 | 9.241 |
| crowded | 0.05 | Root-MUSIC | 1000 | 8.871 | 8.068 | 9.665 | 1.291 | 1.162 | 1.396 |
| crowded | 0.05 | ReconUNet | 1000 | 4.804 | 4.111 | 5.548 | 1.430 | 1.366 | 1.499 |
| crowded | 0.05 | ReconUNet-C | 1000 | 2.446 | 1.751 | 3.128 | 0.903 | 0.872 | 0.955 |
| crowded | 0.05 | SubspaceNet | 1000 | 7.912 | 7.315 | 8.476 | 2.786 | 2.668 | 2.929 |
| crowded | 0.05 | DA-MUSIC | 1000 | 6.652 | 6.253 | 7.030 | 4.126 | 3.969 | 4.341 |
| crowded | 0.05 | SubViT | 1000 | 16.321 | 15.578 | 17.103 | 5.159 | 1.776 | 6.578 |
| crowded | 0.1 | Root-MUSIC | 1000 | 8.766 | 7.917 | 9.610 | 1.236 | 1.175 | 1.325 |
| crowded | 0.1 | ReconUNet | 1000 | 5.450 | 4.666 | 6.150 | 1.608 | 1.526 | 1.668 |
| crowded | 0.1 | ReconUNet-C | 1000 | 3.094 | 2.373 | 3.795 | 1.051 | 1.002 | 1.101 |
| crowded | 0.1 | SubspaceNet | 1000 | 7.990 | 7.420 | 8.558 | 2.572 | 2.450 | 2.703 |
| crowded | 0.1 | DA-MUSIC | 1000 | 5.987 | 5.670 | 6.342 | 3.865 | 3.732 | 4.030 |
| crowded | 0.1 | SubViT | 1000 | 16.312 | 15.503 | 17.107 | 6.167 | 2.487 | 7.457 |
| crowded | 0.2 | Root-MUSIC | 1000 | 9.996 | 9.125 | 10.874 | 1.231 | 1.103 | 1.384 |
| crowded | 0.2 | ReconUNet | 1000 | 14.195 | 13.542 | 14.908 | 5.103 | 4.512 | 5.931 |
| crowded | 0.2 | ReconUNet-C | 1000 | 17.606 | 16.936 | 18.284 | 12.132 | 10.782 | 13.418 |
| crowded | 0.2 | SubspaceNet | 1000 | 11.200 | 10.587 | 11.821 | 3.608 | 3.368 | 3.888 |
| crowded | 0.2 | DA-MUSIC | 1000 | 6.371 | 5.980 | 6.842 | 3.935 | 3.760 | 4.140 |
| crowded | 0.2 | SubViT | 1000 | 16.493 | 15.707 | 17.285 | 5.158 | 1.383 | 6.996 |
| crowded | 0.4 | Root-MUSIC | 1000 | 9.895 | 9.047 | 10.757 | 1.375 | 1.255 | 1.514 |
| crowded | 0.4 | ReconUNet | 1000 | 22.433 | 21.746 | 23.109 | 19.196 | 18.538 | 19.759 |
| crowded | 0.4 | ReconUNet-C | 1000 | 19.681 | 19.012 | 20.353 | 15.459 | 14.635 | 16.497 |
| crowded | 0.4 | SubspaceNet | 1000 | 11.683 | 11.071 | 12.249 | 4.620 | 4.320 | 4.952 |
| crowded | 0.4 | DA-MUSIC | 1000 | 6.532 | 6.036 | 7.008 | 3.813 | 3.637 | 4.036 |
| crowded | 0.4 | SubViT | 1000 | 16.680 | 15.896 | 17.481 | 5.940 | 3.025 | 7.522 |
| crowded | white | Root-MUSIC | 1000 | 7.565 | 6.685 | 8.277 | 1.093 | 1.000 | 1.202 |
| crowded | white | ReconUNet | 1000 | 22.281 | 21.584 | 22.971 | 18.222 | 17.509 | 18.841 |
| crowded | white | ReconUNet-C | 1000 | 20.520 | 19.936 | 21.140 | 17.353 | 16.667 | 18.161 |
| crowded | white | SubspaceNet | 1000 | 17.557 | 17.025 | 18.105 | 14.575 | 13.426 | 15.619 |
| crowded | white | DA-MUSIC | 1000 | 5.552 | 5.173 | 5.935 | 3.457 | 3.270 | 3.625 |
| crowded | white | SubViT | 1000 | 14.257 | 13.322 | 15.092 | 0.954 | 0.858 | 1.075 |
| moderate3 | 0.01 | Root-MUSIC | 1000 | 5.281 | 3.899 | 6.545 | 0.633 | 0.602 | 0.671 |
| moderate3 | 0.01 | ReconUNet | 1000 | 5.883 | 4.629 | 7.058 | 1.217 | 1.145 | 1.272 |
| moderate3 | 0.01 | ReconUNet-C | 1000 | 3.910 | 2.750 | 4.938 | 0.776 | 0.737 | 0.814 |
| moderate3 | 0.01 | SubspaceNet | 1000 | 7.719 | 6.895 | 8.565 | 2.145 | 2.052 | 2.283 |
| moderate3 | 0.01 | DA-MUSIC | 1000 | 6.618 | 6.050 | 7.250 | 4.010 | 3.819 | 4.153 |
| moderate3 | 0.01 | SubViT | 1000 | 14.858 | 13.701 | 15.999 | 0.589 | 0.556 | 0.617 |
| moderate3 | 0.02 | Root-MUSIC | 1000 | 4.506 | 3.153 | 5.776 | 0.541 | 0.502 | 0.574 |
| moderate3 | 0.02 | ReconUNet | 1000 | 2.918 | 1.707 | 4.069 | 0.865 | 0.820 | 0.912 |
| moderate3 | 0.02 | ReconUNet-C | 1000 | 1.656 | 0.785 | 2.631 | 0.538 | 0.516 | 0.559 |
| moderate3 | 0.02 | SubspaceNet | 1000 | 5.292 | 4.384 | 6.097 | 1.581 | 1.519 | 1.672 |
| moderate3 | 0.02 | DA-MUSIC | 1000 | 4.548 | 4.230 | 4.894 | 2.947 | 2.793 | 3.100 |
| moderate3 | 0.02 | SubViT | 1000 | 10.927 | 9.742 | 12.064 | 0.456 | 0.436 | 0.474 |
| moderate3 | 0.05 | Root-MUSIC | 1000 | 4.641 | 3.308 | 5.842 | 0.521 | 0.481 | 0.564 |
| moderate3 | 0.05 | ReconUNet | 1000 | 2.897 | 1.695 | 3.911 | 0.755 | 0.718 | 0.794 |
| moderate3 | 0.05 | ReconUNet-C | 1000 | 0.670 | 0.633 | 0.710 | 0.448 | 0.432 | 0.467 |
| moderate3 | 0.05 | SubspaceNet | 1000 | 5.459 | 4.449 | 6.468 | 1.344 | 1.285 | 1.417 |
| moderate3 | 0.05 | DA-MUSIC | 1000 | 3.785 | 3.531 | 4.062 | 2.577 | 2.440 | 2.685 |
| moderate3 | 0.05 | SubViT | 1000 | 10.962 | 9.662 | 12.265 | 0.429 | 0.409 | 0.448 |
| moderate3 | 0.1 | Root-MUSIC | 1000 | 4.342 | 3.154 | 5.464 | 0.511 | 0.475 | 0.547 |
| moderate3 | 0.1 | ReconUNet | 1000 | 3.132 | 1.886 | 4.285 | 0.814 | 0.785 | 0.864 |
| moderate3 | 0.1 | ReconUNet-C | 1000 | 1.873 | 0.864 | 2.952 | 0.569 | 0.553 | 0.598 |
| moderate3 | 0.1 | SubspaceNet | 1000 | 5.256 | 4.140 | 6.223 | 1.262 | 1.210 | 1.313 |
| moderate3 | 0.1 | DA-MUSIC | 1000 | 3.567 | 3.289 | 3.856 | 2.394 | 2.293 | 2.492 |
| moderate3 | 0.1 | SubViT | 1000 | 9.628 | 8.371 | 10.782 | 0.415 | 0.402 | 0.434 |
| moderate3 | 0.2 | Root-MUSIC | 1000 | 4.854 | 3.496 | 6.029 | 0.492 | 0.468 | 0.529 |
| moderate3 | 0.2 | ReconUNet | 1000 | 12.989 | 11.988 | 13.914 | 2.744 | 2.576 | 2.934 |
| moderate3 | 0.2 | ReconUNet-C | 1000 | 20.055 | 19.119 | 20.980 | 5.806 | 5.175 | 6.836 |
| moderate3 | 0.2 | SubspaceNet | 1000 | 9.327 | 8.328 | 10.260 | 1.769 | 1.673 | 1.839 |
| moderate3 | 0.2 | DA-MUSIC | 1000 | 3.542 | 3.265 | 3.834 | 2.324 | 2.224 | 2.437 |
| moderate3 | 0.2 | SubViT | 1000 | 9.791 | 8.411 | 10.902 | 0.417 | 0.398 | 0.433 |
| moderate3 | 0.4 | Root-MUSIC | 1000 | 4.936 | 3.521 | 6.136 | 0.500 | 0.479 | 0.546 |
| moderate3 | 0.4 | ReconUNet | 1000 | 24.604 | 23.839 | 25.348 | 21.723 | 20.648 | 22.491 |
| moderate3 | 0.4 | ReconUNet-C | 1000 | 21.860 | 20.937 | 22.745 | 12.682 | 11.464 | 14.666 |
| moderate3 | 0.4 | SubspaceNet | 1000 | 9.903 | 8.964 | 10.785 | 2.259 | 2.164 | 2.399 |
| moderate3 | 0.4 | DA-MUSIC | 1000 | 3.532 | 3.261 | 3.821 | 2.305 | 2.179 | 2.407 |
| moderate3 | 0.4 | SubViT | 1000 | 9.681 | 8.372 | 10.802 | 0.405 | 0.389 | 0.423 |
| moderate3 | white | Root-MUSIC | 1000 | 3.719 | 2.369 | 4.933 | 0.383 | 0.358 | 0.406 |
| moderate3 | white | ReconUNet | 1000 | 25.894 | 25.129 | 26.686 | 21.848 | 21.108 | 22.693 |
| moderate3 | white | ReconUNet-C | 1000 | 24.569 | 23.701 | 25.459 | 18.854 | 17.365 | 19.876 |
| moderate3 | white | SubspaceNet | 1000 | 14.102 | 13.300 | 14.948 | 4.128 | 3.966 | 4.335 |
| moderate3 | white | DA-MUSIC | 1000 | 3.320 | 3.013 | 3.676 | 2.172 | 2.101 | 2.320 |
| moderate3 | white | SubViT | 1000 | 8.030 | 6.744 | 9.223 | 0.413 | 0.392 | 0.432 |

## R2 (2026-09-30) — Measured source autocorrelation |r(l)| (l = 0..7) and direct/replica |γ| per bandwidth (decoupled delays)

_Appended 2026-09-30 21:45 UTC from `experiments/runs/sweeps_r2_20260930/bandwidth_autocorr.csv`._

| scenario | K | replicas | delay_mode | bw_frac | n_scenes | r0 | r1 | r2 | r3 | r4 | r5 | r6 | r7 | gamma_mean | gamma_median | gamma_min | delay_mean_samples | delay_max_samples |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| moderate | 2 | 1 | decoupled | 0.01 | 1000 | 1.000 | 1.000 | 1.000 | 0.999 | 0.998 | 0.997 | 0.996 | 0.994 | 0.996 | 0.997 | 0.977 | 4.880 | 9.900 |
| moderate | 2 | 1 | decoupled | 0.02 | 1000 | 1.000 | 0.999 | 0.997 | 0.994 | 0.989 | 0.983 | 0.975 | 0.967 | 0.978 | 0.984 | 0.901 | 4.880 | 9.900 |
| moderate | 2 | 1 | decoupled | 0.05 | 1000 | 1.000 | 0.996 | 0.985 | 0.967 | 0.941 | 0.909 | 0.871 | 0.827 | 0.887 | 0.916 | 0.514 | 4.880 | 9.900 |
| moderate | 2 | 1 | decoupled | 0.1 | 1000 | 1.000 | 0.984 | 0.937 | 0.862 | 0.763 | 0.647 | 0.519 | 0.388 | 0.618 | 0.659 | 0.007 | 4.880 | 9.900 |
| moderate | 2 | 1 | decoupled | 0.2 | 1000 | 1.000 | 0.935 | 0.756 | 0.505 | 0.239 | 0.086 | 0.175 | 0.229 | 0.394 | 0.240 | 0.010 | 4.880 | 9.900 |
| moderate | 2 | 1 | decoupled | 0.4 | 1000 | 1.000 | 0.758 | 0.239 | 0.165 | 0.197 | 0.061 | 0.135 | 0.087 | 0.251 | 0.134 | 0.002 | 4.880 | 9.900 |
| moderate | 2 | 1 | decoupled | white | 1000 | 1.000 | 0.039 | 0.039 | 0.040 | 0.039 | 0.038 | 0.039 | 0.039 | 0.129 | 0.063 | 0.004 | 4.880 | 9.900 |
| crowded | 4 | 3 | decoupled | 0.01 | 1000 | 1.000 | 1.000 | 1.000 | 0.999 | 0.998 | 0.997 | 0.996 | 0.994 | 0.996 | 0.997 | 0.976 | 4.893 | 9.900 |
| crowded | 4 | 3 | decoupled | 0.02 | 1000 | 1.000 | 0.999 | 0.997 | 0.994 | 0.989 | 0.983 | 0.975 | 0.966 | 0.978 | 0.984 | 0.896 | 4.893 | 9.900 |
| crowded | 4 | 3 | decoupled | 0.05 | 1000 | 1.000 | 0.996 | 0.985 | 0.966 | 0.941 | 0.908 | 0.870 | 0.825 | 0.887 | 0.913 | 0.530 | 4.893 | 9.900 |
| crowded | 4 | 3 | decoupled | 0.1 | 1000 | 1.000 | 0.984 | 0.937 | 0.862 | 0.763 | 0.647 | 0.519 | 0.388 | 0.620 | 0.655 | 0.016 | 4.893 | 9.900 |
| crowded | 4 | 3 | decoupled | 0.2 | 1000 | 1.000 | 0.936 | 0.757 | 0.506 | 0.242 | 0.089 | 0.174 | 0.227 | 0.390 | 0.234 | 0.001 | 4.893 | 9.900 |
| crowded | 4 | 3 | decoupled | 0.4 | 1000 | 1.000 | 0.758 | 0.239 | 0.164 | 0.195 | 0.063 | 0.137 | 0.087 | 0.241 | 0.129 | 0.001 | 4.893 | 9.900 |
| crowded | 4 | 3 | decoupled | white | 1000 | 1.000 | 0.039 | 0.039 | 0.040 | 0.039 | 0.039 | 0.040 | 0.039 | 0.126 | 0.062 | 0.002 | 4.893 | 9.900 |
| moderate3 | 3 | 1 | decoupled | 0.01 | 1000 | 1.000 | 1.000 | 1.000 | 0.999 | 0.998 | 0.997 | 0.996 | 0.994 | 0.996 | 0.997 | 0.978 | 4.993 | 9.900 |
| moderate3 | 3 | 1 | decoupled | 0.02 | 1000 | 1.000 | 0.999 | 0.997 | 0.994 | 0.989 | 0.983 | 0.975 | 0.966 | 0.977 | 0.983 | 0.898 | 4.993 | 9.900 |
| moderate3 | 3 | 1 | decoupled | 0.05 | 1000 | 1.000 | 0.996 | 0.985 | 0.967 | 0.941 | 0.909 | 0.870 | 0.826 | 0.884 | 0.908 | 0.557 | 4.993 | 9.900 |
| moderate3 | 3 | 1 | decoupled | 0.1 | 1000 | 1.000 | 0.984 | 0.937 | 0.862 | 0.763 | 0.645 | 0.517 | 0.385 | 0.607 | 0.639 | 0.014 | 4.993 | 9.900 |
| moderate3 | 3 | 1 | decoupled | 0.2 | 1000 | 1.000 | 0.935 | 0.756 | 0.505 | 0.239 | 0.086 | 0.175 | 0.229 | 0.378 | 0.236 | 0.006 | 4.993 | 9.900 |
| moderate3 | 3 | 1 | decoupled | 0.4 | 1000 | 1.000 | 0.758 | 0.241 | 0.162 | 0.195 | 0.061 | 0.134 | 0.086 | 0.236 | 0.123 | 0.005 | 4.993 | 9.900 |
| moderate3 | 3 | 1 | decoupled | white | 1000 | 1.000 | 0.039 | 0.039 | 0.039 | 0.039 | 0.039 | 0.040 | 0.039 | 0.120 | 0.062 | 0.001 | 4.993 | 9.900 |

## R2 (2026-09-30) — Measured |r(l)| and |γ| per bandwidth (coupled delays)

_Appended 2026-09-30 21:45 UTC from `experiments/runs/sweeps_r2_20260930/bandwidth_autocorr_coupled.csv`._

| scenario | K | replicas | delay_mode | bw_frac | n_scenes | r0 | r1 | r2 | r3 | r4 | r5 | r6 | r7 | gamma_mean | gamma_median | gamma_min | delay_mean_samples | delay_max_samples |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| moderate | 2 | 1 | coupled | 0.01 | 1000 | 1.000 | 1.000 | 1.000 | 0.999 | 0.998 | 0.997 | 0.996 | 0.994 | 0.906 | 0.936 | 0.489 | 24.596 | 49.900 |
| moderate | 2 | 1 | coupled | 0.02 | 1000 | 1.000 | 0.999 | 0.997 | 0.994 | 0.989 | 0.983 | 0.975 | 0.967 | 0.871 | 0.905 | 0.445 | 12.273 | 24.900 |
| moderate | 2 | 1 | coupled | 0.05 | 1000 | 1.000 | 0.996 | 0.985 | 0.967 | 0.941 | 0.909 | 0.871 | 0.827 | 0.887 | 0.916 | 0.514 | 4.880 | 9.900 |
| moderate | 2 | 1 | coupled | 0.1 | 1000 | 1.000 | 0.984 | 0.937 | 0.862 | 0.763 | 0.647 | 0.519 | 0.388 | 0.881 | 0.908 | 0.578 | 2.417 | 4.900 |
| moderate | 2 | 1 | coupled | 0.2 | 1000 | 1.000 | 0.935 | 0.756 | 0.505 | 0.239 | 0.086 | 0.175 | 0.229 | 0.882 | 0.909 | 0.605 | 1.187 | 2.400 |
| moderate | 2 | 1 | coupled | 0.4 | 1000 | 1.000 | 0.758 | 0.239 | 0.165 | 0.197 | 0.061 | 0.135 | 0.087 | 0.888 | 0.913 | 0.623 | 0.577 | 1.200 |
| moderate | 2 | 1 | coupled | white | 1000 | 1.000 | 0.039 | 0.039 | 0.040 | 0.039 | 0.038 | 0.039 | 0.039 | 0.129 | 0.063 | 0.004 | 4.880 | 9.900 |
| crowded | 4 | 3 | coupled | 0.01 | 1000 | 1.000 | 1.000 | 1.000 | 0.999 | 0.998 | 0.997 | 0.996 | 0.994 | 0.905 | 0.932 | 0.467 | 24.659 | 49.900 |
| crowded | 4 | 3 | coupled | 0.02 | 1000 | 1.000 | 0.999 | 0.997 | 0.994 | 0.989 | 0.983 | 0.975 | 0.966 | 0.873 | 0.902 | 0.429 | 12.305 | 24.900 |
| crowded | 4 | 3 | coupled | 0.05 | 1000 | 1.000 | 0.996 | 0.985 | 0.966 | 0.941 | 0.908 | 0.870 | 0.825 | 0.887 | 0.913 | 0.530 | 4.893 | 9.900 |
| crowded | 4 | 3 | coupled | 0.1 | 1000 | 1.000 | 0.984 | 0.937 | 0.862 | 0.763 | 0.647 | 0.519 | 0.388 | 0.882 | 0.908 | 0.560 | 2.423 | 4.900 |
| crowded | 4 | 3 | coupled | 0.2 | 1000 | 1.000 | 0.936 | 0.757 | 0.506 | 0.242 | 0.089 | 0.174 | 0.227 | 0.883 | 0.909 | 0.602 | 1.189 | 2.400 |
| crowded | 4 | 3 | coupled | 0.4 | 1000 | 1.000 | 0.758 | 0.239 | 0.164 | 0.195 | 0.063 | 0.137 | 0.087 | 0.888 | 0.912 | 0.604 | 0.577 | 1.200 |
| crowded | 4 | 3 | coupled | white | 1000 | 1.000 | 0.039 | 0.039 | 0.040 | 0.039 | 0.039 | 0.040 | 0.039 | 0.126 | 0.062 | 0.002 | 4.893 | 9.900 |
| moderate3 | 3 | 1 | coupled | 0.01 | 1000 | 1.000 | 1.000 | 1.000 | 0.999 | 0.998 | 0.997 | 0.996 | 0.994 | 0.905 | 0.931 | 0.509 | 25.166 | 49.900 |
| moderate3 | 3 | 1 | coupled | 0.02 | 1000 | 1.000 | 0.999 | 0.997 | 0.994 | 0.989 | 0.983 | 0.975 | 0.966 | 0.867 | 0.894 | 0.436 | 12.558 | 24.900 |
| moderate3 | 3 | 1 | coupled | 0.05 | 1000 | 1.000 | 0.996 | 0.985 | 0.967 | 0.941 | 0.909 | 0.870 | 0.826 | 0.884 | 0.908 | 0.557 | 4.993 | 9.900 |
| moderate3 | 3 | 1 | coupled | 0.1 | 1000 | 1.000 | 0.984 | 0.937 | 0.862 | 0.763 | 0.645 | 0.517 | 0.385 | 0.878 | 0.901 | 0.543 | 2.471 | 4.900 |
| moderate3 | 3 | 1 | coupled | 0.2 | 1000 | 1.000 | 0.935 | 0.756 | 0.505 | 0.239 | 0.086 | 0.175 | 0.229 | 0.878 | 0.906 | 0.593 | 1.213 | 2.400 |
| moderate3 | 3 | 1 | coupled | 0.4 | 1000 | 1.000 | 0.758 | 0.241 | 0.162 | 0.195 | 0.061 | 0.134 | 0.086 | 0.886 | 0.910 | 0.621 | 0.588 | 1.200 |
| moderate3 | 3 | 1 | coupled | white | 1000 | 1.000 | 0.039 | 0.039 | 0.039 | 0.039 | 0.039 | 0.040 | 0.039 | 0.120 | 0.062 | 0.001 | 4.993 | 9.900 |

## R2b (2026-10-01) — Headline (ReconUNet-CB, randomised source bandwidth)

_Appended 2026-10-01 17:51 UTC._

# R2b headline numbers (2026-10-01) — ReconUNet-CB (randomised source bandwidth)

## Plain statement

**Wideband (bw ≥ 0.2 and white sources, 9 cells: 3 scenarios × 3 bandwidths, decoupled delays, 0 dB).** Randomised-bandwidth training removed the wideband failure: the mean pooled RMSE over these cells is 1.14° for ReconUNet-CB vs 22.55° for ReconUNet-C and 5.81° for raw Root-MUSIC; ReconUNet-CB is at or below Root-MUSIC in 9/9 cells (its CI lies entirely above Root-MUSIC in 0/9). **At the training bandwidth 0.05** the sweep RMSE is moderate 0.78° vs 0.61°, crowded 2.57° vs 2.45°, moderate3 0.77° vs 0.67° (ReconUNet-CB vs ReconUNet-C); on the paper test split (rendered at bw 0.05) pooled RMSE 4.96° [4.51, 5.38] vs 4.23° [3.78, 4.67], median RMSPE 0.63° vs 0.58°. _(Generated by `r2b_report.py` from the CSVs: 'removed' = ReconUNet-CB ≤ Root-MUSIC in every wideband cell; 'substantially reduced' = mean wideband RMSE below half of ReconUNet-C's.)_

## Paper test split (rendered at bw 0.05), pooled over K = 1..4

| method | pooled RMSE (°) | 95 % CI | median RMSPE (°) | 95 % CI | scenes |
|---|---|---|---|---|---|
| ReconUNet-C | 4.23 | [3.78, 4.67] | 0.58 | [0.57, 0.59] | 12000 |
| ReconUNet-CB | 4.96 | [4.51, 5.38] | 0.63 | [0.61, 0.64] | 12000 |
| ReconUNet | 5.79 | [5.34, 6.22] | 0.94 | [0.93, 0.96] | 12000 |
| DA-MUSIC | 7.91 | [7.58, 8.22] | 3.17 | [3.12, 3.22] | 12000 |
| SubspaceNet | 8.37 | [8.03, 8.69] | 1.70 | [1.67, 1.74] | 12000 |
| ESPRIT | 11.00 | [10.55, 11.47] | 2.15 | [2.07, 2.23] | 12000 |
| R-MUSIC | 12.43 | [11.93, 12.91] | 0.80 | [0.78, 0.83] | 12000 |
| SubViT | 14.95 | [14.58, 15.31] | 0.58 | [0.57, 0.59] | 12000 |

## Matched wideband: paper test split re-rendered at each scene's drawn bandwidth, pooled over K

| method | pooled RMSE (°) | 95 % CI | median RMSPE (°) | 95 % CI | scenes |
|---|---|---|---|---|---|
| ReconUNet-CB | 4.80 | [4.36, 5.23] | 0.58 | [0.57, 0.59] | 12000 |
| DA-MUSIC | 7.65 | [7.34, 7.95] | 3.10 | [3.04, 3.16] | 12000 |
| ESPRIT | 10.59 | [10.10, 11.08] | 1.57 | [1.51, 1.63] | 12000 |
| SubspaceNet | 11.26 | [10.94, 11.61] | 2.13 | [2.09, 2.17] | 12000 |
| R-MUSIC | 11.60 | [11.12, 12.04] | 0.71 | [0.69, 0.73] | 12000 |
| SubViT | 14.57 | [14.21, 14.95] | 0.57 | [0.56, 0.58] | 12000 |
| ReconUNet | 15.20 | [14.83, 15.58] | 1.69 | [1.64, 1.75] | 12000 |
| ReconUNet-C | 15.45 | [15.09, 15.88] | 1.32 | [1.28, 1.38] | 12000 |

Pooled RMSE (°) by K:

| K | n | R-MUSIC | ESPRIT | ReconUNet | ReconUNet-C | ReconUNet-CB | SubspaceNet | DA-MUSIC | SubViT |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 3000 | 23.55 | 22.87 | 23.14 | 23.85 | 12.10 | 16.30 | 15.47 | 14.50 |
| 2 | 3000 | 13.69 | 11.49 | 16.88 | 18.81 | 4.77 | 11.17 | 7.97 | 15.13 |
| 3 | 3000 | 9.25 | 8.20 | 14.39 | 14.33 | 2.24 | 10.58 | 6.27 | 15.06 |
| 4 | 3000 | 6.34 | 5.75 | 12.07 | 11.13 | 2.40 | 10.21 | 5.02 | 13.93 |
| all | 12000 | 11.60 | 10.59 | 15.20 | 15.45 | 4.80 | 11.26 | 7.65 | 14.57 |


## Training

| model | epochs_run | best_epoch | best_val_loss | val_rmspe_at_best_deg | min_val_rmspe_deg | last_lr | early_stopped | wall_clock_min | min_per_epoch |
|---|---|---|---|---|---|---|---|---|---|
| ReconUNet-CB (R2b, L_rec only, randomised bw) | 204 | 179 | 0.08128 | 1.458 | 1.442 | 1e-06 | True | 529 | 2.59 |
| ReconUNet-C (R2, L_rec only, bw 0.05) | 207 | 182 | 0.07001 | 1.33 | 1.323 | 1e-06 | True | 531 | 2.57 |
| ReconUNet (R1 released, composite loss, bw 0.05) | 109 | 84 | 0.42371 | 2.174 | 2.018 | 8.2e-06 | True | 287 | 2.63 |


## Bandwidth sweep, pooled RMSE (°) — 0 dB, replica delays fixed at the training bandwidth 0.05 (decoupled, primary)

| scenario | bw_frac | Root-MUSIC | ReconUNet | ReconUNet-C | ReconUNet-CB | SubspaceNet | DA-MUSIC | SubViT |
|---|---|---|---|---|---|---|---|---|
| moderate | 0.01 | 5.91 | 2.01 | 2.05 | 0.96 | 5.33 | 6.72 | 10.07 |
| moderate | 0.02 | 5.93 | 1.85 | 0.82 | 0.81 | 4.36 | 4.78 | 7.37 |
| moderate | 0.05 | 5.91 | 1.79 | 0.61 | 0.78 | 2.92 | 3.60 | 8.72 |
| moderate | 0.1 | 5.67 | 1.94 | 1.65 | 0.67 | 3.73 | 3.33 | 7.09 |
| moderate | 0.2 | 5.91 | 14.08 | 22.70 | 0.53 | 5.87 | 3.12 | 5.35 |
| moderate | 0.4 | 5.63 | 24.65 | 26.42 | 0.53 | 8.10 | 2.98 | 5.74 |
| moderate | white | 5.27 | 32.71 | 30.88 | 0.53 | 11.64 | 2.84 | 6.91 |
| crowded | 0.01 | 9.51 | 9.04 | 5.93 | 5.57 | 11.27 | 8.33 | 20.25 |
| crowded | 0.02 | 9.13 | 5.82 | 3.49 | 3.39 | 9.20 | 7.41 | 17.95 |
| crowded | 0.05 | 8.87 | 4.80 | 2.45 | 2.57 | 7.91 | 6.65 | 16.32 |
| crowded | 0.1 | 8.80 | 5.00 | 3.01 | 2.24 | 8.58 | 6.14 | 14.39 |
| crowded | 0.2 | 8.78 | 14.52 | 16.67 | 1.75 | 10.74 | 5.83 | 14.13 |
| crowded | 0.4 | 8.02 | 22.50 | 19.88 | 2.27 | 11.61 | 5.41 | 14.24 |
| crowded | white | 7.56 | 22.28 | 20.52 | 2.81 | 17.56 | 5.55 | 14.26 |
| moderate3 | 0.01 | 4.86 | 4.69 | 3.37 | 3.61 | 7.68 | 6.29 | 15.21 |
| moderate3 | 0.02 | 4.54 | 2.82 | 2.04 | 1.95 | 5.66 | 4.64 | 11.54 |
| moderate3 | 0.05 | 4.64 | 2.90 | 0.67 | 0.77 | 5.46 | 3.79 | 10.96 |
| moderate3 | 0.1 | 4.46 | 3.13 | 1.63 | 0.89 | 5.55 | 3.47 | 8.04 |
| moderate3 | 0.2 | 3.70 | 12.48 | 19.29 | 0.62 | 8.94 | 3.37 | 7.95 |
| moderate3 | 0.4 | 3.75 | 24.72 | 21.97 | 0.58 | 10.48 | 3.25 | 7.41 |
| moderate3 | white | 3.72 | 25.89 | 24.57 | 0.60 | 14.10 | 3.32 | 8.03 |


## Bandwidth sweep, pooled RMSE (°) — 0 dB, replica delays ∝ 1/bw (coupled, secondary)

| scenario | bw_frac | Root-MUSIC | ReconUNet | ReconUNet-C | ReconUNet-CB | SubspaceNet | DA-MUSIC | SubViT |
|---|---|---|---|---|---|---|---|---|
| moderate | 0.01 | 5.64 | 2.43 | 1.22 | 0.93 | 5.85 | 6.48 | 10.15 |
| moderate | 0.02 | 6.91 | 2.12 | 2.09 | 0.78 | 4.13 | 4.81 | 7.93 |
| moderate | 0.05 | 5.91 | 1.79 | 0.61 | 0.78 | 2.92 | 3.60 | 8.72 |
| moderate | 0.1 | 5.80 | 1.34 | 0.90 | 0.70 | 3.58 | 3.02 | 8.47 |
| moderate | 0.2 | 7.29 | 13.76 | 24.95 | 0.91 | 7.20 | 3.18 | 8.22 |
| moderate | 0.4 | 6.81 | 25.61 | 25.34 | 1.35 | 8.61 | 3.04 | 7.91 |
| moderate | white | 5.27 | 32.71 | 30.88 | 0.53 | 11.64 | 2.84 | 6.91 |
| crowded | 0.01 | 10.25 | 9.01 | 6.41 | 5.50 | 11.45 | 8.24 | 19.71 |
| crowded | 0.02 | 9.43 | 5.79 | 3.10 | 3.49 | 9.14 | 7.09 | 17.08 |
| crowded | 0.05 | 8.87 | 4.80 | 2.45 | 2.57 | 7.91 | 6.65 | 16.32 |
| crowded | 0.1 | 8.77 | 5.45 | 3.09 | 2.96 | 7.99 | 5.99 | 16.31 |
| crowded | 0.2 | 10.00 | 14.19 | 17.61 | 2.77 | 11.20 | 6.37 | 16.49 |
| crowded | 0.4 | 9.90 | 22.43 | 19.68 | 3.31 | 11.68 | 6.53 | 16.68 |
| crowded | white | 7.56 | 22.28 | 20.52 | 2.81 | 17.56 | 5.55 | 14.26 |
| moderate3 | 0.01 | 5.28 | 5.88 | 3.91 | 2.45 | 7.72 | 6.62 | 14.86 |
| moderate3 | 0.02 | 4.51 | 2.92 | 1.66 | 1.20 | 5.29 | 4.55 | 10.93 |
| moderate3 | 0.05 | 4.64 | 2.90 | 0.67 | 0.77 | 5.46 | 3.79 | 10.96 |
| moderate3 | 0.1 | 4.34 | 3.13 | 1.87 | 0.70 | 5.26 | 3.57 | 9.63 |
| moderate3 | 0.2 | 4.85 | 12.99 | 20.06 | 0.67 | 9.33 | 3.54 | 9.79 |
| moderate3 | 0.4 | 4.94 | 24.60 | 21.86 | 0.98 | 9.90 | 3.53 | 9.68 |
| moderate3 | white | 3.72 | 25.89 | 24.57 | 0.60 | 14.10 | 3.32 | 8.03 |


## Bandwidth sweep, pooled RMSE (°) — −5 dB, decoupled

| scenario | bw_frac | Root-MUSIC | ReconUNet-C | ReconUNet-CB |
|---|---|---|---|---|
| moderate | 0.01 | 5.58 | 2.87 | 1.11 |
| moderate | 0.02 | 5.73 | 0.92 | 0.96 |
| moderate | 0.05 | 5.88 | 0.65 | 0.80 |
| moderate | 0.1 | 5.61 | 1.67 | 0.70 |
| moderate | 0.2 | 6.13 | 22.94 | 0.59 |
| moderate | 0.4 | 6.02 | 26.23 | 0.58 |
| moderate | white | 5.28 | 31.37 | 0.60 |
| crowded | 0.01 | 10.35 | 6.10 | 5.26 |
| crowded | 0.02 | 9.48 | 3.64 | 3.31 |
| crowded | 0.05 | 9.01 | 2.54 | 2.57 |
| crowded | 0.1 | 9.00 | 3.11 | 2.26 |
| crowded | 0.2 | 8.83 | 16.66 | 1.78 |
| crowded | 0.4 | 8.57 | 20.13 | 2.30 |
| crowded | white | 7.92 | 20.94 | 2.96 |
| moderate3 | 0.01 | 5.13 | 3.42 | 3.50 |
| moderate3 | 0.02 | 5.23 | 1.03 | 1.80 |
| moderate3 | 0.05 | 4.93 | 1.95 | 1.76 |
| moderate3 | 0.1 | 4.71 | 1.66 | 0.95 |
| moderate3 | 0.2 | 3.83 | 18.97 | 0.64 |
| moderate3 | 0.4 | 3.54 | 22.12 | 0.65 |
| moderate3 | white | 3.92 | 25.37 | 0.67 |


## Bandwidth sweep, pooled RMSE (°) — +10 dB, decoupled

| scenario | bw_frac | Root-MUSIC | ReconUNet-C | ReconUNet-CB |
|---|---|---|---|---|
| moderate | 0.01 | 6.01 | 1.77 | 0.95 |
| moderate | 0.02 | 6.01 | 0.81 | 0.81 |
| moderate | 0.05 | 6.00 | 0.60 | 0.80 |
| moderate | 0.1 | 5.89 | 1.65 | 0.55 |
| moderate | 0.2 | 5.90 | 22.59 | 0.52 |
| moderate | 0.4 | 5.62 | 25.98 | 0.52 |
| moderate | white | 5.28 | 30.36 | 0.54 |
| crowded | 0.01 | 9.05 | 5.97 | 5.54 |
| crowded | 0.02 | 8.99 | 3.37 | 3.51 |
| crowded | 0.05 | 8.81 | 2.68 | 2.57 |
| crowded | 0.1 | 8.85 | 3.12 | 2.24 |
| crowded | 0.2 | 8.64 | 16.85 | 1.74 |
| crowded | 0.4 | 8.29 | 19.93 | 2.31 |
| crowded | white | 7.53 | 20.69 | 2.54 |
| moderate3 | 0.01 | 4.19 | 3.40 | 3.36 |
| moderate3 | 0.02 | 4.19 | 1.30 | 1.96 |
| moderate3 | 0.05 | 4.53 | 0.64 | 0.81 |
| moderate3 | 0.1 | 4.34 | 1.61 | 0.67 |
| moderate3 | 0.2 | 3.70 | 19.12 | 0.63 |
| moderate3 | 0.4 | 3.86 | 21.68 | 0.56 |
| moderate3 | white | 3.86 | 24.12 | 0.59 |


## Cost

| model | class | trainable parameters | MMACs per scene | GPU fwd b1 ms/scene | GPU fwd+Root-MUSIC b1 ms/scene | CPU fwd b1 ms | GPU fwd b1024 ms/scene | GPU fwd+Root-MUSIC b1024 ms/scene | epochs | best_epoch | train_wallclock_min |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ReconUNet | EVDCovarianceReconstructionUNet | 337842 | 7.782 | 1.2 | 1.773 | 6.046 | 0.03229 | 0.06985 | 109 | 84 | 287 |
| ReconUNet-C | CovarianceOnlyReconstructionUNet | 323070 | 7.128 | 0.7837 | 1.357 | 0.4839 | 0.0136 | 0.05635 | 207 | 182 | 531 |
| ReconUNet-CB | CovarianceOnlyReconstructionUNet | 323070 | 7.128 | 0.7831 | 1.353 | 0.4667 | 0.01339 | 0.05637 | 204 | 179 | 529 |

## R2b (2026-10-01) — ReconUNet-CB training (randomised-bandwidth corpus, from scratch)

_Appended 2026-10-01 17:51 UTC from `experiments/runs/revision_r2b_20261001/reconunet_cb_training.csv`._

ReconUNet-CB = CovarianceOnlyReconstructionUNet + L_rec, identical to ReconUNet-C except the training corpus: the paper manifests rendered with a per-scene source bandwidth (white w.p. 0.15, else log-uniform on [0.01, 0.4], drawn from rng [scene.seed, 20261001]) and replica delays fixed to the bw-0.05 distribution (configs/data/paper_corpus_bwrand.yaml). Checkpoint selection on the randomised-bandwidth validation split, so its validation numbers are not comparable with the other rows. The L_rec target A_ideal A_ideal^H is bandwidth-independent (direct paths only, analytic).

| model | epochs_run | best_epoch | best_val_loss | val_rmspe_at_best_deg | min_val_rmspe_deg | last_lr | early_stopped | wall_clock_min | min_per_epoch |
|---|---|---|---|---|---|---|---|---|---|
| ReconUNet-CB (R2b, L_rec only, randomised bw) | 204 | 179 | 0.081 | 1.458 | 1.442 | 0.000 | True | 529 | 2.590 |
| ReconUNet-C (R2, L_rec only, bw 0.05) | 207 | 182 | 0.070 | 1.330 | 1.323 | 0.000 | True | 531 | 2.570 |
| ReconUNet (R1 released, composite loss, bw 0.05) | 109 | 84 | 0.424 | 2.174 | 2.018 | 0.000 | True | 287 | 2.630 |

## R2b (2026-10-01) — Source-bandwidth sweep, 0 dB, decoupled delays (primary), ReconUNet-CB added

_Appended 2026-10-01 17:51 UTC from `experiments/runs/sweeps_r2b_20261001/bandwidth_sweep.csv`._

| scenario | bw_frac | method | n_scenes | rmse_deg | rmse_ci_lo | rmse_ci_hi | median_rmspe_deg | median_ci_lo | median_ci_hi |
|---|---|---|---|---|---|---|---|---|---|
| moderate | 0.01 | Root-MUSIC | 1000 | 5.907 | 4.021 | 7.631 | 0.531 | 0.491 | 0.586 |
| moderate | 0.01 | ReconUNet | 1000 | 2.013 | 1.271 | 2.915 | 0.775 | 0.719 | 0.826 |
| moderate | 0.01 | ReconUNet-C | 1000 | 2.053 | 1.194 | 2.763 | 0.528 | 0.498 | 0.566 |
| moderate | 0.01 | ReconUNet-CB | 1000 | 0.961 | 0.885 | 1.035 | 0.469 | 0.446 | 0.501 |
| moderate | 0.01 | SubspaceNet | 1000 | 5.333 | 4.275 | 6.383 | 1.421 | 1.335 | 1.512 |
| moderate | 0.01 | DA-MUSIC | 1000 | 6.719 | 6.103 | 7.412 | 3.555 | 3.398 | 3.801 |
| moderate | 0.01 | SubViT | 1000 | 10.075 | 8.327 | 11.786 | 0.415 | 0.399 | 0.444 |
| moderate | 0.02 | Root-MUSIC | 1000 | 5.930 | 4.027 | 7.710 | 0.531 | 0.487 | 0.578 |
| moderate | 0.02 | ReconUNet | 1000 | 1.851 | 1.085 | 2.787 | 0.666 | 0.628 | 0.700 |
| moderate | 0.02 | ReconUNet-C | 1000 | 0.817 | 0.740 | 0.896 | 0.424 | 0.396 | 0.450 |
| moderate | 0.02 | ReconUNet-CB | 1000 | 0.807 | 0.741 | 0.871 | 0.428 | 0.403 | 0.443 |
| moderate | 0.02 | SubspaceNet | 1000 | 4.364 | 3.164 | 5.517 | 1.224 | 1.170 | 1.280 |
| moderate | 0.02 | DA-MUSIC | 1000 | 4.784 | 4.126 | 5.425 | 2.594 | 2.443 | 2.760 |
| moderate | 0.02 | SubViT | 1000 | 7.367 | 5.650 | 8.889 | 0.405 | 0.385 | 0.419 |
| moderate | 0.05 | Root-MUSIC | 1000 | 5.915 | 4.044 | 7.583 | 0.486 | 0.456 | 0.523 |
| moderate | 0.05 | ReconUNet | 1000 | 1.790 | 0.972 | 2.762 | 0.616 | 0.581 | 0.646 |
| moderate | 0.05 | ReconUNet-C | 1000 | 0.608 | 0.565 | 0.656 | 0.369 | 0.354 | 0.386 |
| moderate | 0.05 | ReconUNet-CB | 1000 | 0.780 | 0.657 | 0.947 | 0.381 | 0.359 | 0.400 |
| moderate | 0.05 | SubspaceNet | 1000 | 2.918 | 2.324 | 3.516 | 1.035 | 0.986 | 1.091 |
| moderate | 0.05 | DA-MUSIC | 1000 | 3.600 | 3.171 | 4.084 | 2.143 | 2.024 | 2.272 |
| moderate | 0.05 | SubViT | 1000 | 8.724 | 6.959 | 10.494 | 0.380 | 0.362 | 0.404 |
| moderate | 0.1 | Root-MUSIC | 1000 | 5.665 | 3.512 | 7.386 | 0.419 | 0.393 | 0.451 |
| moderate | 0.1 | ReconUNet | 1000 | 1.939 | 0.990 | 3.421 | 0.630 | 0.608 | 0.665 |
| moderate | 0.1 | ReconUNet-C | 1000 | 1.648 | 0.771 | 2.678 | 0.475 | 0.454 | 0.500 |
| moderate | 0.1 | ReconUNet-CB | 1000 | 0.668 | 0.526 | 0.862 | 0.346 | 0.330 | 0.364 |
| moderate | 0.1 | SubspaceNet | 1000 | 3.735 | 2.571 | 4.906 | 0.988 | 0.941 | 1.042 |
| moderate | 0.1 | DA-MUSIC | 1000 | 3.328 | 2.909 | 3.884 | 1.912 | 1.804 | 1.983 |
| moderate | 0.1 | SubViT | 1000 | 7.093 | 5.322 | 8.822 | 0.378 | 0.363 | 0.395 |
| moderate | 0.2 | Root-MUSIC | 1000 | 5.906 | 3.925 | 7.607 | 0.372 | 0.339 | 0.404 |
| moderate | 0.2 | ReconUNet | 1000 | 14.083 | 12.517 | 15.597 | 2.454 | 2.274 | 2.665 |
| moderate | 0.2 | ReconUNet-C | 1000 | 22.704 | 21.174 | 24.162 | 3.307 | 3.091 | 3.634 |
| moderate | 0.2 | ReconUNet-CB | 1000 | 0.533 | 0.498 | 0.571 | 0.337 | 0.319 | 0.353 |
| moderate | 0.2 | SubspaceNet | 1000 | 5.869 | 4.322 | 7.212 | 1.238 | 1.185 | 1.285 |
| moderate | 0.2 | DA-MUSIC | 1000 | 3.117 | 2.700 | 3.623 | 1.748 | 1.655 | 1.838 |
| moderate | 0.2 | SubViT | 1000 | 5.347 | 3.641 | 6.847 | 0.371 | 0.356 | 0.388 |
| moderate | 0.4 | Root-MUSIC | 1000 | 5.626 | 3.766 | 7.287 | 0.348 | 0.327 | 0.373 |
| moderate | 0.4 | ReconUNet | 1000 | 24.654 | 23.698 | 25.580 | 18.866 | 17.355 | 20.254 |
| moderate | 0.4 | ReconUNet-C | 1000 | 26.419 | 25.047 | 27.786 | 7.323 | 6.608 | 8.294 |
| moderate | 0.4 | ReconUNet-CB | 1000 | 0.525 | 0.496 | 0.553 | 0.340 | 0.326 | 0.353 |
| moderate | 0.4 | SubspaceNet | 1000 | 8.101 | 6.778 | 9.446 | 1.560 | 1.458 | 1.633 |
| moderate | 0.4 | DA-MUSIC | 1000 | 2.982 | 2.560 | 3.555 | 1.685 | 1.584 | 1.824 |
| moderate | 0.4 | SubViT | 1000 | 5.745 | 3.918 | 7.398 | 0.370 | 0.358 | 0.385 |
| moderate | white | Root-MUSIC | 1000 | 5.273 | 2.989 | 6.998 | 0.326 | 0.308 | 0.361 |
| moderate | white | ReconUNet | 1000 | 32.712 | 31.592 | 33.753 | 23.770 | 22.439 | 25.330 |
| moderate | white | ReconUNet-C | 1000 | 30.879 | 29.559 | 32.237 | 17.330 | 15.399 | 19.100 |
| moderate | white | ReconUNet-CB | 1000 | 0.530 | 0.494 | 0.569 | 0.331 | 0.311 | 0.352 |
| moderate | white | SubspaceNet | 1000 | 11.640 | 10.378 | 12.971 | 2.896 | 2.735 | 3.086 |
| moderate | white | DA-MUSIC | 1000 | 2.838 | 2.575 | 3.157 | 1.681 | 1.612 | 1.783 |
| moderate | white | SubViT | 1000 | 6.909 | 5.127 | 8.535 | 0.384 | 0.368 | 0.402 |
| crowded | 0.01 | Root-MUSIC | 1000 | 9.513 | 8.652 | 10.355 | 1.550 | 1.423 | 1.652 |
| crowded | 0.01 | ReconUNet | 1000 | 9.035 | 8.302 | 9.787 | 2.463 | 2.321 | 2.552 |
| crowded | 0.01 | ReconUNet-C | 1000 | 5.932 | 5.166 | 6.620 | 1.775 | 1.705 | 1.840 |
| crowded | 0.01 | ReconUNet-CB | 1000 | 5.572 | 4.776 | 6.275 | 1.564 | 1.482 | 1.641 |
| crowded | 0.01 | SubspaceNet | 1000 | 11.274 | 10.723 | 11.811 | 5.199 | 4.923 | 5.691 |
| crowded | 0.01 | DA-MUSIC | 1000 | 8.328 | 7.835 | 8.801 | 5.235 | 5.017 | 5.447 |
| crowded | 0.01 | SubViT | 1000 | 20.247 | 19.534 | 21.003 | 15.784 | 14.142 | 17.111 |
| crowded | 0.02 | Root-MUSIC | 1000 | 9.131 | 8.289 | 9.993 | 1.381 | 1.245 | 1.529 |
| crowded | 0.02 | ReconUNet | 1000 | 5.816 | 5.098 | 6.551 | 1.808 | 1.722 | 1.907 |
| crowded | 0.02 | ReconUNet-C | 1000 | 3.486 | 2.829 | 4.089 | 1.150 | 1.099 | 1.207 |
| crowded | 0.02 | ReconUNet-CB | 1000 | 3.388 | 2.700 | 3.970 | 1.193 | 1.134 | 1.254 |
| crowded | 0.02 | SubspaceNet | 1000 | 9.203 | 8.554 | 9.866 | 3.431 | 3.275 | 3.629 |
| crowded | 0.02 | DA-MUSIC | 1000 | 7.405 | 6.985 | 7.831 | 4.562 | 4.346 | 4.709 |
| crowded | 0.02 | SubViT | 1000 | 17.945 | 17.255 | 18.681 | 10.047 | 8.729 | 11.657 |
| crowded | 0.05 | Root-MUSIC | 1000 | 8.871 | 7.972 | 9.673 | 1.291 | 1.155 | 1.389 |
| crowded | 0.05 | ReconUNet | 1000 | 4.804 | 4.022 | 5.535 | 1.430 | 1.363 | 1.507 |
| crowded | 0.05 | ReconUNet-C | 1000 | 2.446 | 1.773 | 3.108 | 0.903 | 0.874 | 0.948 |
| crowded | 0.05 | ReconUNet-CB | 1000 | 2.575 | 1.957 | 3.182 | 0.980 | 0.925 | 1.016 |
| crowded | 0.05 | SubspaceNet | 1000 | 7.912 | 7.298 | 8.495 | 2.786 | 2.667 | 2.929 |
| crowded | 0.05 | DA-MUSIC | 1000 | 6.652 | 6.284 | 7.039 | 4.126 | 3.962 | 4.334 |
| crowded | 0.05 | SubViT | 1000 | 16.321 | 15.514 | 17.029 | 5.159 | 1.774 | 6.574 |
| crowded | 0.1 | Root-MUSIC | 1000 | 8.799 | 7.921 | 9.695 | 1.139 | 1.045 | 1.217 |
| crowded | 0.1 | ReconUNet | 1000 | 5.003 | 4.220 | 5.756 | 1.407 | 1.348 | 1.478 |
| crowded | 0.1 | ReconUNet-C | 1000 | 3.013 | 2.292 | 3.718 | 0.999 | 0.955 | 1.046 |
| crowded | 0.1 | ReconUNet-CB | 1000 | 2.241 | 1.574 | 2.898 | 0.819 | 0.788 | 0.858 |
| crowded | 0.1 | SubspaceNet | 1000 | 8.585 | 7.953 | 9.283 | 2.515 | 2.358 | 2.659 |
| crowded | 0.1 | DA-MUSIC | 1000 | 6.140 | 5.761 | 6.509 | 3.759 | 3.567 | 3.884 |
| crowded | 0.1 | SubViT | 1000 | 14.394 | 13.655 | 15.189 | 0.886 | 0.791 | 1.049 |
| crowded | 0.2 | Root-MUSIC | 1000 | 8.777 | 7.875 | 9.627 | 1.092 | 1.026 | 1.212 |
| crowded | 0.2 | ReconUNet | 1000 | 14.517 | 13.835 | 15.226 | 4.899 | 4.189 | 5.553 |
| crowded | 0.2 | ReconUNet-C | 1000 | 16.674 | 16.041 | 17.342 | 10.865 | 9.070 | 12.246 |
| crowded | 0.2 | ReconUNet-CB | 1000 | 1.754 | 1.226 | 2.319 | 0.743 | 0.707 | 0.793 |
| crowded | 0.2 | SubspaceNet | 1000 | 10.741 | 10.108 | 11.365 | 3.167 | 3.030 | 3.407 |
| crowded | 0.2 | DA-MUSIC | 1000 | 5.829 | 5.468 | 6.242 | 3.543 | 3.392 | 3.666 |
| crowded | 0.2 | SubViT | 1000 | 14.130 | 13.308 | 14.955 | 0.855 | 0.773 | 0.946 |
| crowded | 0.4 | Root-MUSIC | 1000 | 8.018 | 7.154 | 8.785 | 1.072 | 1.007 | 1.174 |
| crowded | 0.4 | ReconUNet | 1000 | 22.500 | 21.825 | 23.204 | 19.473 | 18.536 | 20.326 |
| crowded | 0.4 | ReconUNet-C | 1000 | 19.884 | 19.172 | 20.514 | 15.561 | 14.927 | 16.354 |
| crowded | 0.4 | ReconUNet-CB | 1000 | 2.270 | 1.470 | 3.003 | 0.763 | 0.730 | 0.794 |
| crowded | 0.4 | SubspaceNet | 1000 | 11.613 | 10.957 | 12.328 | 3.958 | 3.695 | 4.326 |
| crowded | 0.4 | DA-MUSIC | 1000 | 5.414 | 5.053 | 5.774 | 3.388 | 3.229 | 3.558 |
| crowded | 0.4 | SubViT | 1000 | 14.238 | 13.305 | 15.098 | 0.854 | 0.788 | 0.922 |
| crowded | white | Root-MUSIC | 1000 | 7.565 | 6.802 | 8.365 | 1.093 | 1.005 | 1.205 |
| crowded | white | ReconUNet | 1000 | 22.281 | 21.610 | 22.976 | 18.222 | 17.478 | 18.929 |
| crowded | white | ReconUNet-C | 1000 | 20.520 | 19.918 | 21.096 | 17.353 | 16.669 | 18.161 |
| crowded | white | ReconUNet-CB | 1000 | 2.813 | 2.018 | 3.612 | 0.785 | 0.756 | 0.823 |
| crowded | white | SubspaceNet | 1000 | 17.557 | 16.939 | 18.167 | 14.575 | 13.447 | 15.625 |
| crowded | white | DA-MUSIC | 1000 | 5.552 | 5.156 | 5.931 | 3.457 | 3.235 | 3.604 |
| crowded | white | SubViT | 1000 | 14.257 | 13.403 | 15.080 | 0.954 | 0.859 | 1.091 |
| moderate3 | 0.01 | Root-MUSIC | 1000 | 4.859 | 3.653 | 5.975 | 0.577 | 0.520 | 0.622 |
| moderate3 | 0.01 | ReconUNet | 1000 | 4.691 | 3.564 | 5.762 | 1.190 | 1.141 | 1.244 |
| moderate3 | 0.01 | ReconUNet-C | 1000 | 3.372 | 2.004 | 4.554 | 0.760 | 0.726 | 0.801 |
| moderate3 | 0.01 | ReconUNet-CB | 1000 | 3.605 | 2.341 | 4.770 | 0.688 | 0.660 | 0.718 |
| moderate3 | 0.01 | SubspaceNet | 1000 | 7.680 | 6.844 | 8.448 | 2.240 | 2.157 | 2.332 |
| moderate3 | 0.01 | DA-MUSIC | 1000 | 6.289 | 5.833 | 6.774 | 3.978 | 3.843 | 4.167 |
| moderate3 | 0.01 | SubViT | 1000 | 15.211 | 13.948 | 16.339 | 0.583 | 0.542 | 0.615 |
| moderate3 | 0.02 | Root-MUSIC | 1000 | 4.535 | 3.247 | 5.768 | 0.528 | 0.491 | 0.575 |
| moderate3 | 0.02 | ReconUNet | 1000 | 2.819 | 1.529 | 3.860 | 0.864 | 0.825 | 0.912 |
| moderate3 | 0.02 | ReconUNet-C | 1000 | 2.045 | 0.805 | 3.332 | 0.556 | 0.532 | 0.574 |
| moderate3 | 0.02 | ReconUNet-CB | 1000 | 1.954 | 0.876 | 3.131 | 0.560 | 0.535 | 0.588 |
| moderate3 | 0.02 | SubspaceNet | 1000 | 5.664 | 4.728 | 6.496 | 1.633 | 1.534 | 1.705 |
| moderate3 | 0.02 | DA-MUSIC | 1000 | 4.637 | 4.267 | 5.046 | 2.986 | 2.842 | 3.111 |
| moderate3 | 0.02 | SubViT | 1000 | 11.540 | 10.364 | 12.764 | 0.463 | 0.441 | 0.480 |
| moderate3 | 0.05 | Root-MUSIC | 1000 | 4.641 | 3.330 | 5.875 | 0.521 | 0.477 | 0.565 |
| moderate3 | 0.05 | ReconUNet | 1000 | 2.897 | 1.656 | 3.978 | 0.755 | 0.719 | 0.795 |
| moderate3 | 0.05 | ReconUNet-C | 1000 | 0.670 | 0.631 | 0.709 | 0.448 | 0.432 | 0.467 |
| moderate3 | 0.05 | ReconUNet-CB | 1000 | 0.773 | 0.703 | 0.859 | 0.479 | 0.460 | 0.503 |
| moderate3 | 0.05 | SubspaceNet | 1000 | 5.459 | 4.437 | 6.436 | 1.344 | 1.281 | 1.416 |
| moderate3 | 0.05 | DA-MUSIC | 1000 | 3.785 | 3.517 | 4.047 | 2.577 | 2.431 | 2.684 |
| moderate3 | 0.05 | SubViT | 1000 | 10.962 | 9.565 | 12.196 | 0.429 | 0.409 | 0.448 |
| moderate3 | 0.1 | Root-MUSIC | 1000 | 4.461 | 3.119 | 5.601 | 0.447 | 0.416 | 0.478 |
| moderate3 | 0.1 | ReconUNet | 1000 | 3.131 | 1.794 | 4.241 | 0.762 | 0.738 | 0.801 |
| moderate3 | 0.1 | ReconUNet-C | 1000 | 1.626 | 0.788 | 2.567 | 0.583 | 0.556 | 0.602 |
| moderate3 | 0.1 | ReconUNet-CB | 1000 | 0.891 | 0.583 | 1.263 | 0.430 | 0.415 | 0.447 |
| moderate3 | 0.1 | SubspaceNet | 1000 | 5.553 | 4.616 | 6.477 | 1.225 | 1.181 | 1.311 |
| moderate3 | 0.1 | DA-MUSIC | 1000 | 3.470 | 3.212 | 3.759 | 2.389 | 2.281 | 2.466 |
| moderate3 | 0.1 | SubViT | 1000 | 8.035 | 6.824 | 9.183 | 0.418 | 0.405 | 0.437 |
| moderate3 | 0.2 | Root-MUSIC | 1000 | 3.701 | 2.523 | 4.706 | 0.423 | 0.394 | 0.441 |
| moderate3 | 0.2 | ReconUNet | 1000 | 12.484 | 11.521 | 13.408 | 2.754 | 2.589 | 2.882 |
| moderate3 | 0.2 | ReconUNet-C | 1000 | 19.290 | 18.363 | 20.281 | 5.037 | 4.556 | 6.023 |
| moderate3 | 0.2 | ReconUNet-CB | 1000 | 0.622 | 0.543 | 0.737 | 0.391 | 0.372 | 0.412 |
| moderate3 | 0.2 | SubspaceNet | 1000 | 8.937 | 7.881 | 9.959 | 1.579 | 1.498 | 1.649 |
| moderate3 | 0.2 | DA-MUSIC | 1000 | 3.366 | 3.113 | 3.653 | 2.215 | 2.134 | 2.299 |
| moderate3 | 0.2 | SubViT | 1000 | 7.945 | 6.573 | 9.216 | 0.399 | 0.383 | 0.417 |
| moderate3 | 0.4 | Root-MUSIC | 1000 | 3.750 | 2.314 | 4.942 | 0.395 | 0.381 | 0.427 |
| moderate3 | 0.4 | ReconUNet | 1000 | 24.715 | 23.982 | 25.424 | 21.609 | 20.671 | 22.593 |
| moderate3 | 0.4 | ReconUNet-C | 1000 | 21.975 | 21.118 | 22.885 | 14.384 | 12.534 | 15.758 |
| moderate3 | 0.4 | ReconUNet-CB | 1000 | 0.580 | 0.542 | 0.621 | 0.400 | 0.382 | 0.418 |
| moderate3 | 0.4 | SubspaceNet | 1000 | 10.476 | 9.436 | 11.421 | 2.138 | 2.013 | 2.256 |
| moderate3 | 0.4 | DA-MUSIC | 1000 | 3.255 | 2.999 | 3.543 | 2.179 | 2.098 | 2.301 |
| moderate3 | 0.4 | SubViT | 1000 | 7.407 | 6.152 | 8.644 | 0.395 | 0.382 | 0.414 |
| moderate3 | white | Root-MUSIC | 1000 | 3.719 | 2.382 | 4.898 | 0.383 | 0.359 | 0.404 |
| moderate3 | white | ReconUNet | 1000 | 25.894 | 25.096 | 26.646 | 21.848 | 21.084 | 22.754 |
| moderate3 | white | ReconUNet-C | 1000 | 24.569 | 23.770 | 25.341 | 18.854 | 17.463 | 20.040 |
| moderate3 | white | ReconUNet-CB | 1000 | 0.600 | 0.566 | 0.635 | 0.396 | 0.380 | 0.417 |
| moderate3 | white | SubspaceNet | 1000 | 14.102 | 13.300 | 14.944 | 4.128 | 3.974 | 4.392 |
| moderate3 | white | DA-MUSIC | 1000 | 3.320 | 3.012 | 3.690 | 2.172 | 2.103 | 2.316 |
| moderate3 | white | SubViT | 1000 | 8.030 | 6.753 | 9.308 | 0.413 | 0.391 | 0.432 |

## R2b (2026-10-01) — Source-bandwidth sweep, 0 dB, coupled delays

_Appended 2026-10-01 17:51 UTC from `experiments/runs/sweeps_r2b_20261001/bandwidth_sweep_coupled.csv`._

| scenario | bw_frac | method | n_scenes | rmse_deg | rmse_ci_lo | rmse_ci_hi | median_rmspe_deg | median_ci_lo | median_ci_hi |
|---|---|---|---|---|---|---|---|---|---|
| moderate | 0.01 | Root-MUSIC | 1000 | 5.640 | 3.766 | 7.275 | 0.574 | 0.534 | 0.614 |
| moderate | 0.01 | ReconUNet | 1000 | 2.426 | 1.303 | 3.448 | 0.748 | 0.708 | 0.800 |
| moderate | 0.01 | ReconUNet-C | 1000 | 1.224 | 0.942 | 1.536 | 0.480 | 0.460 | 0.510 |
| moderate | 0.01 | ReconUNet-CB | 1000 | 0.933 | 0.855 | 1.011 | 0.482 | 0.459 | 0.510 |
| moderate | 0.01 | SubspaceNet | 1000 | 5.848 | 4.631 | 6.936 | 1.397 | 1.280 | 1.470 |
| moderate | 0.01 | DA-MUSIC | 1000 | 6.478 | 5.943 | 7.096 | 3.534 | 3.295 | 3.751 |
| moderate | 0.01 | SubViT | 1000 | 10.147 | 8.211 | 11.998 | 0.414 | 0.398 | 0.443 |
| moderate | 0.02 | Root-MUSIC | 1000 | 6.909 | 4.913 | 8.732 | 0.518 | 0.488 | 0.554 |
| moderate | 0.02 | ReconUNet | 1000 | 2.119 | 1.031 | 3.353 | 0.654 | 0.608 | 0.702 |
| moderate | 0.02 | ReconUNet-C | 1000 | 2.094 | 0.668 | 3.485 | 0.406 | 0.382 | 0.433 |
| moderate | 0.02 | ReconUNet-CB | 1000 | 0.778 | 0.700 | 0.872 | 0.419 | 0.405 | 0.439 |
| moderate | 0.02 | SubspaceNet | 1000 | 4.131 | 2.758 | 5.430 | 1.153 | 1.088 | 1.232 |
| moderate | 0.02 | DA-MUSIC | 1000 | 4.808 | 4.212 | 5.449 | 2.514 | 2.365 | 2.664 |
| moderate | 0.02 | SubViT | 1000 | 7.929 | 6.125 | 9.655 | 0.398 | 0.382 | 0.411 |
| moderate | 0.05 | Root-MUSIC | 1000 | 5.915 | 4.044 | 7.583 | 0.486 | 0.456 | 0.523 |
| moderate | 0.05 | ReconUNet | 1000 | 1.790 | 0.972 | 2.762 | 0.616 | 0.581 | 0.646 |
| moderate | 0.05 | ReconUNet-C | 1000 | 0.608 | 0.565 | 0.656 | 0.369 | 0.354 | 0.386 |
| moderate | 0.05 | ReconUNet-CB | 1000 | 0.780 | 0.657 | 0.947 | 0.381 | 0.359 | 0.400 |
| moderate | 0.05 | SubspaceNet | 1000 | 2.918 | 2.324 | 3.516 | 1.035 | 0.986 | 1.091 |
| moderate | 0.05 | DA-MUSIC | 1000 | 3.600 | 3.171 | 4.084 | 2.143 | 2.024 | 2.272 |
| moderate | 0.05 | SubViT | 1000 | 8.724 | 6.959 | 10.494 | 0.380 | 0.362 | 0.404 |
| moderate | 0.1 | Root-MUSIC | 1000 | 5.804 | 3.785 | 7.613 | 0.502 | 0.463 | 0.546 |
| moderate | 0.1 | ReconUNet | 1000 | 1.337 | 1.029 | 1.777 | 0.650 | 0.624 | 0.695 |
| moderate | 0.1 | ReconUNet-C | 1000 | 0.896 | 0.752 | 1.090 | 0.486 | 0.462 | 0.509 |
| moderate | 0.1 | ReconUNet-CB | 1000 | 0.704 | 0.572 | 0.888 | 0.372 | 0.354 | 0.387 |
| moderate | 0.1 | SubspaceNet | 1000 | 3.577 | 2.539 | 4.752 | 1.066 | 0.997 | 1.116 |
| moderate | 0.1 | DA-MUSIC | 1000 | 3.020 | 2.792 | 3.278 | 1.895 | 1.791 | 2.005 |
| moderate | 0.1 | SubViT | 1000 | 8.471 | 6.612 | 10.254 | 0.381 | 0.364 | 0.401 |
| moderate | 0.2 | Root-MUSIC | 1000 | 7.292 | 5.086 | 9.149 | 0.502 | 0.469 | 0.554 |
| moderate | 0.2 | ReconUNet | 1000 | 13.759 | 12.213 | 15.223 | 2.246 | 2.089 | 2.414 |
| moderate | 0.2 | ReconUNet-C | 1000 | 24.950 | 23.339 | 26.466 | 3.889 | 3.628 | 4.164 |
| moderate | 0.2 | ReconUNet-CB | 1000 | 0.905 | 0.624 | 1.263 | 0.389 | 0.372 | 0.402 |
| moderate | 0.2 | SubspaceNet | 1000 | 7.196 | 5.767 | 8.611 | 1.340 | 1.287 | 1.407 |
| moderate | 0.2 | DA-MUSIC | 1000 | 3.180 | 2.765 | 3.727 | 1.853 | 1.731 | 1.947 |
| moderate | 0.2 | SubViT | 1000 | 8.217 | 6.453 | 9.894 | 0.384 | 0.365 | 0.400 |
| moderate | 0.4 | Root-MUSIC | 1000 | 6.815 | 4.624 | 8.621 | 0.498 | 0.464 | 0.535 |
| moderate | 0.4 | ReconUNet | 1000 | 25.610 | 24.549 | 26.635 | 20.764 | 18.971 | 21.898 |
| moderate | 0.4 | ReconUNet-C | 1000 | 25.339 | 23.809 | 26.796 | 6.386 | 5.913 | 6.897 |
| moderate | 0.4 | ReconUNet-CB | 1000 | 1.348 | 0.664 | 2.119 | 0.418 | 0.395 | 0.439 |
| moderate | 0.4 | SubspaceNet | 1000 | 8.608 | 7.113 | 10.067 | 1.652 | 1.561 | 1.727 |
| moderate | 0.4 | DA-MUSIC | 1000 | 3.040 | 2.663 | 3.596 | 1.748 | 1.656 | 1.811 |
| moderate | 0.4 | SubViT | 1000 | 7.910 | 6.065 | 9.680 | 0.385 | 0.364 | 0.400 |
| moderate | white | Root-MUSIC | 1000 | 5.273 | 2.989 | 6.998 | 0.326 | 0.308 | 0.361 |
| moderate | white | ReconUNet | 1000 | 32.712 | 31.592 | 33.753 | 23.770 | 22.439 | 25.330 |
| moderate | white | ReconUNet-C | 1000 | 30.879 | 29.559 | 32.237 | 17.330 | 15.399 | 19.100 |
| moderate | white | ReconUNet-CB | 1000 | 0.530 | 0.494 | 0.569 | 0.331 | 0.311 | 0.352 |
| moderate | white | SubspaceNet | 1000 | 11.640 | 10.378 | 12.971 | 2.896 | 2.735 | 3.086 |
| moderate | white | DA-MUSIC | 1000 | 2.838 | 2.575 | 3.157 | 1.681 | 1.612 | 1.783 |
| moderate | white | SubViT | 1000 | 6.909 | 5.127 | 8.535 | 0.384 | 0.368 | 0.402 |
| crowded | 0.01 | Root-MUSIC | 1000 | 10.250 | 9.420 | 11.037 | 1.792 | 1.648 | 1.928 |
| crowded | 0.01 | ReconUNet | 1000 | 9.012 | 8.277 | 9.704 | 2.489 | 2.381 | 2.602 |
| crowded | 0.01 | ReconUNet-C | 1000 | 6.407 | 5.641 | 7.150 | 1.755 | 1.653 | 1.842 |
| crowded | 0.01 | ReconUNet-CB | 1000 | 5.504 | 4.759 | 6.198 | 1.576 | 1.503 | 1.667 |
| crowded | 0.01 | SubspaceNet | 1000 | 11.446 | 10.874 | 12.015 | 5.292 | 4.928 | 5.843 |
| crowded | 0.01 | DA-MUSIC | 1000 | 8.237 | 7.792 | 8.693 | 5.246 | 4.974 | 5.421 |
| crowded | 0.01 | SubViT | 1000 | 19.709 | 19.006 | 20.402 | 14.408 | 13.297 | 15.942 |
| crowded | 0.02 | Root-MUSIC | 1000 | 9.432 | 8.663 | 10.241 | 1.401 | 1.309 | 1.507 |
| crowded | 0.02 | ReconUNet | 1000 | 5.793 | 4.992 | 6.504 | 1.708 | 1.656 | 1.786 |
| crowded | 0.02 | ReconUNet-C | 1000 | 3.103 | 2.474 | 3.723 | 1.149 | 1.098 | 1.209 |
| crowded | 0.02 | ReconUNet-CB | 1000 | 3.494 | 2.825 | 4.166 | 1.170 | 1.113 | 1.213 |
| crowded | 0.02 | SubspaceNet | 1000 | 9.137 | 8.474 | 9.766 | 3.352 | 3.178 | 3.535 |
| crowded | 0.02 | DA-MUSIC | 1000 | 7.091 | 6.666 | 7.519 | 4.441 | 4.178 | 4.668 |
| crowded | 0.02 | SubViT | 1000 | 17.084 | 16.216 | 17.883 | 7.874 | 6.632 | 9.201 |
| crowded | 0.05 | Root-MUSIC | 1000 | 8.871 | 7.972 | 9.673 | 1.291 | 1.155 | 1.389 |
| crowded | 0.05 | ReconUNet | 1000 | 4.804 | 4.022 | 5.535 | 1.430 | 1.363 | 1.507 |
| crowded | 0.05 | ReconUNet-C | 1000 | 2.446 | 1.773 | 3.108 | 0.903 | 0.874 | 0.948 |
| crowded | 0.05 | ReconUNet-CB | 1000 | 2.575 | 1.957 | 3.182 | 0.980 | 0.925 | 1.016 |
| crowded | 0.05 | SubspaceNet | 1000 | 7.912 | 7.298 | 8.495 | 2.786 | 2.667 | 2.929 |
| crowded | 0.05 | DA-MUSIC | 1000 | 6.652 | 6.284 | 7.039 | 4.126 | 3.962 | 4.334 |
| crowded | 0.05 | SubViT | 1000 | 16.321 | 15.514 | 17.029 | 5.159 | 1.774 | 6.574 |
| crowded | 0.1 | Root-MUSIC | 1000 | 8.766 | 7.937 | 9.560 | 1.236 | 1.171 | 1.324 |
| crowded | 0.1 | ReconUNet | 1000 | 5.450 | 4.720 | 6.167 | 1.608 | 1.523 | 1.668 |
| crowded | 0.1 | ReconUNet-C | 1000 | 3.094 | 2.377 | 3.796 | 1.051 | 1.003 | 1.102 |
| crowded | 0.1 | ReconUNet-CB | 1000 | 2.957 | 2.188 | 3.652 | 0.913 | 0.872 | 0.967 |
| crowded | 0.1 | SubspaceNet | 1000 | 7.990 | 7.409 | 8.541 | 2.572 | 2.450 | 2.713 |
| crowded | 0.1 | DA-MUSIC | 1000 | 5.987 | 5.652 | 6.349 | 3.865 | 3.731 | 4.028 |
| crowded | 0.1 | SubViT | 1000 | 16.312 | 15.616 | 17.088 | 6.167 | 2.565 | 7.495 |
| crowded | 0.2 | Root-MUSIC | 1000 | 9.996 | 9.091 | 10.861 | 1.231 | 1.115 | 1.403 |
| crowded | 0.2 | ReconUNet | 1000 | 14.195 | 13.550 | 14.848 | 5.103 | 4.539 | 5.904 |
| crowded | 0.2 | ReconUNet-C | 1000 | 17.606 | 16.891 | 18.302 | 12.132 | 10.846 | 13.346 |
| crowded | 0.2 | ReconUNet-CB | 1000 | 2.771 | 2.073 | 3.434 | 0.920 | 0.878 | 0.974 |
| crowded | 0.2 | SubspaceNet | 1000 | 11.200 | 10.556 | 11.841 | 3.608 | 3.395 | 3.887 |
| crowded | 0.2 | DA-MUSIC | 1000 | 6.371 | 5.905 | 6.812 | 3.935 | 3.758 | 4.103 |
| crowded | 0.2 | SubViT | 1000 | 16.493 | 15.754 | 17.253 | 5.158 | 1.394 | 7.033 |
| crowded | 0.4 | Root-MUSIC | 1000 | 9.895 | 9.040 | 10.706 | 1.375 | 1.258 | 1.514 |
| crowded | 0.4 | ReconUNet | 1000 | 22.433 | 21.817 | 23.129 | 19.196 | 18.515 | 19.776 |
| crowded | 0.4 | ReconUNet-C | 1000 | 19.681 | 18.970 | 20.346 | 15.459 | 14.432 | 16.652 |
| crowded | 0.4 | ReconUNet-CB | 1000 | 3.313 | 2.517 | 4.007 | 0.974 | 0.933 | 1.021 |
| crowded | 0.4 | SubspaceNet | 1000 | 11.683 | 11.121 | 12.323 | 4.620 | 4.320 | 4.924 |
| crowded | 0.4 | DA-MUSIC | 1000 | 6.532 | 6.062 | 7.058 | 3.813 | 3.635 | 4.021 |
| crowded | 0.4 | SubViT | 1000 | 16.680 | 15.787 | 17.516 | 5.940 | 3.280 | 7.580 |
| crowded | white | Root-MUSIC | 1000 | 7.565 | 6.802 | 8.365 | 1.093 | 1.005 | 1.205 |
| crowded | white | ReconUNet | 1000 | 22.281 | 21.610 | 22.976 | 18.222 | 17.478 | 18.929 |
| crowded | white | ReconUNet-C | 1000 | 20.520 | 19.918 | 21.096 | 17.353 | 16.669 | 18.161 |
| crowded | white | ReconUNet-CB | 1000 | 2.813 | 2.018 | 3.612 | 0.785 | 0.756 | 0.823 |
| crowded | white | SubspaceNet | 1000 | 17.557 | 16.939 | 18.167 | 14.575 | 13.447 | 15.625 |
| crowded | white | DA-MUSIC | 1000 | 5.552 | 5.156 | 5.931 | 3.457 | 3.235 | 3.604 |
| crowded | white | SubViT | 1000 | 14.257 | 13.403 | 15.080 | 0.954 | 0.859 | 1.091 |
| moderate3 | 0.01 | Root-MUSIC | 1000 | 5.281 | 4.011 | 6.466 | 0.633 | 0.603 | 0.672 |
| moderate3 | 0.01 | ReconUNet | 1000 | 5.883 | 4.735 | 7.049 | 1.217 | 1.143 | 1.269 |
| moderate3 | 0.01 | ReconUNet-C | 1000 | 3.910 | 2.796 | 4.940 | 0.776 | 0.738 | 0.815 |
| moderate3 | 0.01 | ReconUNet-CB | 1000 | 2.449 | 1.136 | 3.653 | 0.694 | 0.659 | 0.733 |
| moderate3 | 0.01 | SubspaceNet | 1000 | 7.719 | 6.930 | 8.500 | 2.145 | 2.050 | 2.284 |
| moderate3 | 0.01 | DA-MUSIC | 1000 | 6.618 | 6.047 | 7.219 | 4.010 | 3.819 | 4.141 |
| moderate3 | 0.01 | SubViT | 1000 | 14.858 | 13.730 | 15.982 | 0.589 | 0.556 | 0.618 |
| moderate3 | 0.02 | Root-MUSIC | 1000 | 4.506 | 3.159 | 5.803 | 0.541 | 0.503 | 0.570 |
| moderate3 | 0.02 | ReconUNet | 1000 | 2.918 | 1.726 | 3.973 | 0.865 | 0.819 | 0.910 |
| moderate3 | 0.02 | ReconUNet-C | 1000 | 1.656 | 0.782 | 2.617 | 0.538 | 0.516 | 0.558 |
| moderate3 | 0.02 | ReconUNet-CB | 1000 | 1.202 | 0.840 | 1.663 | 0.563 | 0.535 | 0.594 |
| moderate3 | 0.02 | SubspaceNet | 1000 | 5.292 | 4.366 | 6.150 | 1.581 | 1.520 | 1.667 |
| moderate3 | 0.02 | DA-MUSIC | 1000 | 4.548 | 4.225 | 4.891 | 2.947 | 2.801 | 3.095 |
| moderate3 | 0.02 | SubViT | 1000 | 10.927 | 9.725 | 12.082 | 0.456 | 0.436 | 0.474 |
| moderate3 | 0.05 | Root-MUSIC | 1000 | 4.641 | 3.330 | 5.875 | 0.521 | 0.477 | 0.565 |
| moderate3 | 0.05 | ReconUNet | 1000 | 2.897 | 1.656 | 3.978 | 0.755 | 0.719 | 0.795 |
| moderate3 | 0.05 | ReconUNet-C | 1000 | 0.670 | 0.631 | 0.709 | 0.448 | 0.432 | 0.467 |
| moderate3 | 0.05 | ReconUNet-CB | 1000 | 0.773 | 0.703 | 0.859 | 0.479 | 0.460 | 0.503 |
| moderate3 | 0.05 | SubspaceNet | 1000 | 5.459 | 4.437 | 6.436 | 1.344 | 1.281 | 1.416 |
| moderate3 | 0.05 | DA-MUSIC | 1000 | 3.785 | 3.517 | 4.047 | 2.577 | 2.431 | 2.684 |
| moderate3 | 0.05 | SubViT | 1000 | 10.962 | 9.565 | 12.196 | 0.429 | 0.409 | 0.448 |
| moderate3 | 0.1 | Root-MUSIC | 1000 | 4.342 | 3.157 | 5.418 | 0.511 | 0.475 | 0.548 |
| moderate3 | 0.1 | ReconUNet | 1000 | 3.132 | 1.834 | 4.275 | 0.814 | 0.789 | 0.861 |
| moderate3 | 0.1 | ReconUNet-C | 1000 | 1.873 | 0.875 | 2.864 | 0.569 | 0.553 | 0.597 |
| moderate3 | 0.1 | ReconUNet-CB | 1000 | 0.703 | 0.626 | 0.785 | 0.438 | 0.417 | 0.459 |
| moderate3 | 0.1 | SubspaceNet | 1000 | 5.256 | 4.280 | 6.281 | 1.262 | 1.208 | 1.310 |
| moderate3 | 0.1 | DA-MUSIC | 1000 | 3.567 | 3.297 | 3.846 | 2.394 | 2.292 | 2.481 |
| moderate3 | 0.1 | SubViT | 1000 | 9.628 | 8.292 | 10.978 | 0.415 | 0.401 | 0.432 |
| moderate3 | 0.2 | Root-MUSIC | 1000 | 4.854 | 3.577 | 6.088 | 0.492 | 0.467 | 0.528 |
| moderate3 | 0.2 | ReconUNet | 1000 | 12.989 | 12.084 | 13.924 | 2.744 | 2.582 | 2.920 |
| moderate3 | 0.2 | ReconUNet-C | 1000 | 20.055 | 19.058 | 20.950 | 5.806 | 5.161 | 6.570 |
| moderate3 | 0.2 | ReconUNet-CB | 1000 | 0.669 | 0.612 | 0.746 | 0.442 | 0.421 | 0.466 |
| moderate3 | 0.2 | SubspaceNet | 1000 | 9.327 | 8.391 | 10.295 | 1.769 | 1.676 | 1.838 |
| moderate3 | 0.2 | DA-MUSIC | 1000 | 3.542 | 3.278 | 3.855 | 2.324 | 2.227 | 2.452 |
| moderate3 | 0.2 | SubViT | 1000 | 9.791 | 8.426 | 11.013 | 0.417 | 0.399 | 0.434 |
| moderate3 | 0.4 | Root-MUSIC | 1000 | 4.936 | 3.559 | 6.153 | 0.500 | 0.481 | 0.545 |
| moderate3 | 0.4 | ReconUNet | 1000 | 24.604 | 23.885 | 25.329 | 21.723 | 20.648 | 22.473 |
| moderate3 | 0.4 | ReconUNet-C | 1000 | 21.860 | 20.952 | 22.712 | 12.682 | 11.445 | 14.659 |
| moderate3 | 0.4 | ReconUNet-CB | 1000 | 0.981 | 0.683 | 1.384 | 0.467 | 0.434 | 0.495 |
| moderate3 | 0.4 | SubspaceNet | 1000 | 9.903 | 8.916 | 10.796 | 2.259 | 2.157 | 2.395 |
| moderate3 | 0.4 | DA-MUSIC | 1000 | 3.532 | 3.261 | 3.812 | 2.305 | 2.200 | 2.408 |
| moderate3 | 0.4 | SubViT | 1000 | 9.681 | 8.469 | 10.875 | 0.405 | 0.389 | 0.424 |
| moderate3 | white | Root-MUSIC | 1000 | 3.719 | 2.382 | 4.898 | 0.383 | 0.359 | 0.404 |
| moderate3 | white | ReconUNet | 1000 | 25.894 | 25.096 | 26.646 | 21.848 | 21.084 | 22.754 |
| moderate3 | white | ReconUNet-C | 1000 | 24.569 | 23.770 | 25.341 | 18.854 | 17.463 | 20.040 |
| moderate3 | white | ReconUNet-CB | 1000 | 0.600 | 0.566 | 0.635 | 0.396 | 0.380 | 0.417 |
| moderate3 | white | SubspaceNet | 1000 | 14.102 | 13.300 | 14.944 | 4.128 | 3.974 | 4.392 |
| moderate3 | white | DA-MUSIC | 1000 | 3.320 | 3.012 | 3.690 | 2.172 | 2.103 | 2.316 |
| moderate3 | white | SubViT | 1000 | 8.030 | 6.753 | 9.308 | 0.413 | 0.391 | 0.432 |

## R2b (2026-10-01) — Source-bandwidth sweep, −5 dB, decoupled

_Appended 2026-10-01 17:51 UTC from `experiments/runs/sweeps_r2b_20261001/bandwidth_sweep_m5dB.csv`._

| scenario | bw_frac | method | n_scenes | rmse_deg | rmse_ci_lo | rmse_ci_hi | median_rmspe_deg | median_ci_lo | median_ci_hi |
|---|---|---|---|---|---|---|---|---|---|
| moderate | 0.01 | Root-MUSIC | 1000 | 5.578 | 3.769 | 7.268 | 0.631 | 0.582 | 0.669 |
| moderate | 0.01 | ReconUNet-C | 1000 | 2.868 | 1.307 | 4.147 | 0.571 | 0.542 | 0.607 |
| moderate | 0.01 | ReconUNet-CB | 1000 | 1.106 | 0.952 | 1.326 | 0.535 | 0.507 | 0.569 |
| moderate | 0.02 | Root-MUSIC | 1000 | 5.728 | 3.912 | 7.284 | 0.589 | 0.550 | 0.626 |
| moderate | 0.02 | ReconUNet-C | 1000 | 0.923 | 0.792 | 1.130 | 0.467 | 0.443 | 0.486 |
| moderate | 0.02 | ReconUNet-CB | 1000 | 0.961 | 0.794 | 1.191 | 0.468 | 0.451 | 0.492 |
| moderate | 0.05 | Root-MUSIC | 1000 | 5.880 | 3.852 | 7.638 | 0.552 | 0.502 | 0.593 |
| moderate | 0.05 | ReconUNet-C | 1000 | 0.654 | 0.613 | 0.701 | 0.412 | 0.396 | 0.439 |
| moderate | 0.05 | ReconUNet-CB | 1000 | 0.796 | 0.703 | 0.908 | 0.435 | 0.401 | 0.458 |
| moderate | 0.1 | Root-MUSIC | 1000 | 5.614 | 3.705 | 7.376 | 0.478 | 0.439 | 0.511 |
| moderate | 0.1 | ReconUNet-C | 1000 | 1.670 | 0.802 | 2.614 | 0.536 | 0.515 | 0.558 |
| moderate | 0.1 | ReconUNet-CB | 1000 | 0.702 | 0.576 | 0.887 | 0.379 | 0.364 | 0.408 |
| moderate | 0.2 | Root-MUSIC | 1000 | 6.134 | 4.263 | 7.820 | 0.433 | 0.404 | 0.459 |
| moderate | 0.2 | ReconUNet-C | 1000 | 22.937 | 21.417 | 24.313 | 3.365 | 3.150 | 3.702 |
| moderate | 0.2 | ReconUNet-CB | 1000 | 0.588 | 0.552 | 0.621 | 0.391 | 0.375 | 0.407 |
| moderate | 0.4 | Root-MUSIC | 1000 | 6.023 | 4.127 | 7.695 | 0.412 | 0.387 | 0.439 |
| moderate | 0.4 | ReconUNet-C | 1000 | 26.231 | 24.842 | 27.624 | 6.978 | 6.294 | 7.766 |
| moderate | 0.4 | ReconUNet-CB | 1000 | 0.578 | 0.548 | 0.609 | 0.396 | 0.374 | 0.410 |
| moderate | white | Root-MUSIC | 1000 | 5.278 | 3.129 | 7.021 | 0.399 | 0.374 | 0.429 |
| moderate | white | ReconUNet-C | 1000 | 31.372 | 30.054 | 32.642 | 17.948 | 16.339 | 19.807 |
| moderate | white | ReconUNet-CB | 1000 | 0.601 | 0.563 | 0.641 | 0.406 | 0.384 | 0.428 |
| crowded | 0.01 | Root-MUSIC | 1000 | 10.354 | 9.569 | 11.110 | 2.072 | 1.899 | 2.227 |
| crowded | 0.01 | ReconUNet-C | 1000 | 6.101 | 5.395 | 6.826 | 1.790 | 1.708 | 1.879 |
| crowded | 0.01 | ReconUNet-CB | 1000 | 5.259 | 4.570 | 6.021 | 1.540 | 1.472 | 1.636 |
| crowded | 0.02 | Root-MUSIC | 1000 | 9.483 | 8.675 | 10.282 | 1.561 | 1.425 | 1.699 |
| crowded | 0.02 | ReconUNet-C | 1000 | 3.638 | 2.955 | 4.270 | 1.178 | 1.121 | 1.215 |
| crowded | 0.02 | ReconUNet-CB | 1000 | 3.314 | 2.645 | 3.964 | 1.213 | 1.164 | 1.259 |
| crowded | 0.05 | Root-MUSIC | 1000 | 9.006 | 8.184 | 9.827 | 1.357 | 1.269 | 1.488 |
| crowded | 0.05 | ReconUNet-C | 1000 | 2.543 | 1.868 | 3.231 | 0.946 | 0.899 | 0.978 |
| crowded | 0.05 | ReconUNet-CB | 1000 | 2.571 | 1.954 | 3.142 | 1.012 | 0.967 | 1.052 |
| crowded | 0.1 | Root-MUSIC | 1000 | 8.997 | 8.129 | 9.810 | 1.196 | 1.108 | 1.290 |
| crowded | 0.1 | ReconUNet-C | 1000 | 3.113 | 2.393 | 3.810 | 1.042 | 0.994 | 1.092 |
| crowded | 0.1 | ReconUNet-CB | 1000 | 2.262 | 1.588 | 2.885 | 0.863 | 0.826 | 0.901 |
| crowded | 0.2 | Root-MUSIC | 1000 | 8.834 | 7.926 | 9.690 | 1.158 | 1.095 | 1.256 |
| crowded | 0.2 | ReconUNet-C | 1000 | 16.662 | 16.013 | 17.319 | 10.196 | 8.542 | 11.796 |
| crowded | 0.2 | ReconUNet-CB | 1000 | 1.782 | 1.279 | 2.344 | 0.790 | 0.745 | 0.823 |
| crowded | 0.4 | Root-MUSIC | 1000 | 8.567 | 7.746 | 9.480 | 1.130 | 1.071 | 1.205 |
| crowded | 0.4 | ReconUNet-C | 1000 | 20.129 | 19.457 | 20.800 | 16.175 | 15.471 | 17.089 |
| crowded | 0.4 | ReconUNet-CB | 1000 | 2.296 | 1.528 | 2.959 | 0.794 | 0.767 | 0.829 |
| crowded | white | Root-MUSIC | 1000 | 7.921 | 7.065 | 8.673 | 1.155 | 1.072 | 1.267 |
| crowded | white | ReconUNet-C | 1000 | 20.940 | 20.339 | 21.561 | 17.776 | 16.996 | 18.656 |
| crowded | white | ReconUNet-CB | 1000 | 2.957 | 2.179 | 3.725 | 0.825 | 0.791 | 0.878 |
| moderate3 | 0.01 | Root-MUSIC | 1000 | 5.127 | 3.925 | 6.269 | 0.766 | 0.719 | 0.826 |
| moderate3 | 0.01 | ReconUNet-C | 1000 | 3.420 | 2.104 | 4.584 | 0.794 | 0.761 | 0.834 |
| moderate3 | 0.01 | ReconUNet-CB | 1000 | 3.495 | 2.163 | 4.657 | 0.727 | 0.694 | 0.759 |
| moderate3 | 0.02 | Root-MUSIC | 1000 | 5.226 | 3.826 | 6.560 | 0.642 | 0.599 | 0.702 |
| moderate3 | 0.02 | ReconUNet-C | 1000 | 1.034 | 0.837 | 1.273 | 0.597 | 0.573 | 0.617 |
| moderate3 | 0.02 | ReconUNet-CB | 1000 | 1.804 | 0.896 | 2.828 | 0.592 | 0.574 | 0.619 |
| moderate3 | 0.05 | Root-MUSIC | 1000 | 4.928 | 3.624 | 6.155 | 0.608 | 0.571 | 0.640 |
| moderate3 | 0.05 | ReconUNet-C | 1000 | 1.946 | 0.709 | 3.026 | 0.500 | 0.477 | 0.520 |
| moderate3 | 0.05 | ReconUNet-CB | 1000 | 1.757 | 0.768 | 2.939 | 0.528 | 0.497 | 0.557 |
| moderate3 | 0.1 | Root-MUSIC | 1000 | 4.713 | 3.411 | 5.888 | 0.521 | 0.505 | 0.551 |
| moderate3 | 0.1 | ReconUNet-C | 1000 | 1.661 | 0.845 | 2.613 | 0.617 | 0.589 | 0.656 |
| moderate3 | 0.1 | ReconUNet-CB | 1000 | 0.951 | 0.640 | 1.354 | 0.477 | 0.458 | 0.499 |
| moderate3 | 0.2 | Root-MUSIC | 1000 | 3.835 | 2.631 | 5.001 | 0.508 | 0.482 | 0.533 |
| moderate3 | 0.2 | ReconUNet-C | 1000 | 18.975 | 17.957 | 19.954 | 4.993 | 4.573 | 5.797 |
| moderate3 | 0.2 | ReconUNet-CB | 1000 | 0.635 | 0.597 | 0.681 | 0.454 | 0.435 | 0.473 |
| moderate3 | 0.4 | Root-MUSIC | 1000 | 3.540 | 2.284 | 4.735 | 0.484 | 0.455 | 0.516 |
| moderate3 | 0.4 | ReconUNet-C | 1000 | 22.122 | 21.257 | 23.011 | 14.637 | 13.009 | 16.162 |
| moderate3 | 0.4 | ReconUNet-CB | 1000 | 0.646 | 0.606 | 0.691 | 0.456 | 0.438 | 0.478 |
| moderate3 | white | Root-MUSIC | 1000 | 3.919 | 2.631 | 5.061 | 0.461 | 0.442 | 0.486 |
| moderate3 | white | ReconUNet-C | 1000 | 25.367 | 24.514 | 26.234 | 19.273 | 18.253 | 20.843 |
| moderate3 | white | ReconUNet-CB | 1000 | 0.668 | 0.635 | 0.704 | 0.472 | 0.457 | 0.492 |
