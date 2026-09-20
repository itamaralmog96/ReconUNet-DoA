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
