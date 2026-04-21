# EVD-Based UNet Architecture - Complete Specification

## Overview

The **EVDCovarianceReconstructionUNet** is a neural network that directly predicts eigenvalues and eigenvectors of covariance matrices from autocorrelation input tensors. It combines a UNet-based feature extractor with specialized prediction heads for eigenvalue decomposition.

## Architecture Diagram

```
Input: Rx_tau [B, τ, 2M, M]
           │
           ▼
    ┌─────────────────────────────────────────┐
    │     Base UNet (CovarianceReconstructionUNet)    │
    │  ┌────────────────────────────────────┐ │
    │  │         ENCODER                     │ │
    │  │  Block 1: Conv→Act→BN→Conv→Act→BN  │ │
    │  │         ↓ MaxPool                   │ │
    │  │  Block 2: Conv→Act→BN→Conv→Act→BN  │ │
    │  │         ↓ MaxPool                   │ │
    │  │  Block 3: Conv→Act→BN→Conv→Act→BN  │ │
    │  └────────────┬───────────────────────┘ │
    │               ▼                         │
    │  ┌────────────────────────────────────┐ │
    │  │       BOTTLENECK                    │ │
    │  │    Conv→Act→BN                      │ │
    │  └────────────┬───────────────────────┘ │
    │               ▼                         │
    │  ┌────────────────────────────────────┐ │
    │  │         DECODER                     │ │
    │  │  Block 3: UpConv + Skip → Conv×2   │ │
    │  │  Block 4: UpConv + Skip → Conv×2   │ │
    │  └────────────┬───────────────────────┘ │
    │               ▼                         │
    │  Final Conv → τ→1 Conv                  │
    └────────────────┬────────────────────────┘
                     │
        unet_features [B, 2M, M]
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
    unsqueeze(1)          unsqueeze(1)
   [B, 1, 2M, M]         [B, 1, 2M, M]
          │                     │
          ▼                     ▼
┌─────────────────┐   ┌─────────────────┐
│ EIGENVALUE HEAD │   │ EIGENVECTOR HEAD│
│  ┌───────────┐  │   │  ┌───────────┐  │
│  │ AvgPool   │  │   │  │ Conv3×3   │  │
│  │ (1,1)     │  │   │  │ 1→32      │  │
│  └─────┬─────┘  │   │  └─────┬─────┘  │
│        ▼        │   │        ▼        │
│  ┌───────────┐  │   │  ┌───────────┐  │
│  │ Flatten   │  │   │  │ BN+ReLU   │  │
│  └─────┬─────┘  │   │  │ +Dropout  │  │
│        ▼        │   │  └─────┬─────┘  │
│  ┌───────────┐  │   │        ▼        │
│  │ FC: 1→64  │  │   │  ┌───────────┐  │
│  │ ReLU+Drop │  │   │  │ Conv3×3   │  │
│  └─────┬─────┘  │   │  │ 32→16     │  │
│        ▼        │   │  └─────┬─────┘  │
│  ┌───────────┐  │   │        ▼        │
│  │ FC: 64→128│  │   │  ┌───────────┐  │
│  │ ReLU+Drop │  │   │  │ BN+ReLU   │  │
│  └─────┬─────┘  │   │  └─────┬─────┘  │
│        ▼        │   │        ▼        │
│  ┌───────────┐  │   │  ┌───────────┐  │
│  │ FC: 128→M │  │   │  │ Conv3×3   │  │
│  │ ReLU      │  │   │  │ 16→1      │  │
│  └─────┬─────┘  │   │  └─────┬─────┘  │
│        │        │   │        ▼        │
└────────┼────────┘   │  ┌───────────┐  │
         │            │  │   Tanh    │  │
         ▼            │  └─────┬─────┘  │
   eigenvals          └────────┼────────┘
   [B, M]                      │
         │                     ▼
         │            eigenvecs_raw
         │            [B, 1, 2M, M]
         │                     │
         │               squeeze(1)
         │                     │
         │                     ▼
         │            ┌────────────────┐
         │            │ ORTHOGONALIZATION │
         │            │ Split real/imag │
         │            │ QR Decomposition│
         │            │ Phase Reference │
         │            └───────┬────────┘
         │                    │
         │                    ▼
         │            eigenvecs [B, M, M]
         │                    │
         └────────┬───────────┘
                  │
                  ▼
         ┌────────────────┐
         │ RECONSTRUCTION │
         │ V @ Λ @ V^H    │
         └───────┬────────┘
                 │
                 ▼
         reconstructed_cov
         [B, M, M] complex

Outputs: (eigenvals, eigenvecs, reconstructed_cov)
```

---

## Detailed Layer-by-Layer Specification

### Configuration Parameters

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `tau` | 8 | Number of autocorrelation time lags |
| `M` | 8 | Number of array elements |
| `activation_type` | "anti_rectifier" | Activation function type |
| `use_dropout` | True | Enable dropout regularization |

---

## 1. Input Layer

| Property | Value |
|----------|-------|
| **Input Shape** | `[B, τ, 2M, M]` |
| **Example (M=8, τ=8)** | `[B, 8, 16, 8]` |
| **Data Type** | `float32` |
| **Description** | Autocorrelation tensor with τ time lags, stacked real/imaginary parts |

**Input Format Details:**
- Dimension 0: Batch size (B)
- Dimension 1: Time lags (τ = 8)
- Dimension 2: 2M = 16 (real part [0:M] + imaginary part [M:2M])
- Dimension 3: M = 8 (array elements)

---

## 2. Base UNet: CovarianceReconstructionUNet

The base UNet extracts features through an encoder-decoder architecture with skip connections.

### 2.1 Encoder Block 1

| Layer | Input Shape | Output Shape | Parameters |
|-------|-------------|--------------|------------|
| `enc_conv1` (Conv2d) | `[B, τ, 2M, M]` | `[B, 16, 2M, M]` | kernel=3×3, pad=1 |
| Anti-rectifier | `[B, 16, 2M, M]` | `[B, 32, 2M, M]` | ReLU(x) ∥ ReLU(-x) |
| `enc_bn1` (BatchNorm2d) | `[B, 32, 2M, M]` | `[B, 32, 2M, M]` | 32 features |
| `enc_conv1_2` (Conv2d) | `[B, 32, 2M, M]` | `[B, 16, 2M, M]` | kernel=3×3, pad=1 |
| Anti-rectifier | `[B, 16, 2M, M]` | `[B, 32, 2M, M]` | ReLU(x) ∥ ReLU(-x) |
| `enc_bn1_2` (BatchNorm2d) | `[B, 32, 2M, M]` | `[B, 32, 2M, M]` | 32 features |
| **Skip Connection (x1_skip)** | - | `[B, 32, 2M, M]` | Saved for decoder |
| `pool1` (MaxPool2d) | `[B, 32, 2M, M]` | `[B, 32, M, M/2]` | kernel=2, stride=2 |

### 2.2 Encoder Block 2

| Layer | Input Shape | Output Shape | Parameters |
|-------|-------------|--------------|------------|
| `enc_conv2` (Conv2d) | `[B, 32, M, M/2]` | `[B, 32, M, M/2]` | kernel=3×3, pad=1 |
| Anti-rectifier | `[B, 32, M, M/2]` | `[B, 64, M, M/2]` | ReLU(x) ∥ ReLU(-x) |
| `enc_bn2` (BatchNorm2d) | `[B, 64, M, M/2]` | `[B, 64, M, M/2]` | 64 features |
| `enc_conv2_2` (Conv2d) | `[B, 64, M, M/2]` | `[B, 32, M, M/2]` | kernel=3×3, pad=1 |
| Anti-rectifier | `[B, 32, M, M/2]` | `[B, 64, M, M/2]` | ReLU(x) ∥ ReLU(-x) |
| `enc_bn2_2` (BatchNorm2d) | `[B, 64, M, M/2]` | `[B, 64, M, M/2]` | 64 features |
| **Skip Connection (x2_skip)** | - | `[B, 64, M, M/2]` | Saved for decoder |
| `pool2` (MaxPool2d) | `[B, 64, M, M/2]` | `[B, 64, M/2, M/4]` | kernel=2, stride=2 |

### 2.3 Encoder Block 3

| Layer | Input Shape | Output Shape | Parameters |
|-------|-------------|--------------|------------|
| `enc_conv3` (Conv2d) | `[B, 64, M/2, M/4]` | `[B, 64, M/2, M/4]` | kernel=3×3, pad=1 |
| Anti-rectifier | `[B, 64, M/2, M/4]` | `[B, 128, M/2, M/4]` | ReLU(x) ∥ ReLU(-x) |
| `enc_bn3` (BatchNorm2d) | `[B, 128, M/2, M/4]` | `[B, 128, M/2, M/4]` | 128 features |
| `enc_conv3_2` (Conv2d) | `[B, 128, M/2, M/4]` | `[B, 64, M/2, M/4]` | kernel=3×3, pad=1 |
| Anti-rectifier | `[B, 64, M/2, M/4]` | `[B, 128, M/2, M/4]` | ReLU(x) ∥ ReLU(-x) |
| `enc_bn3_2` (BatchNorm2d) | `[B, 128, M/2, M/4]` | `[B, 128, M/2, M/4]` | 128 features |

### 2.4 Bottleneck

| Layer | Input Shape | Output Shape | Parameters |
|-------|-------------|--------------|------------|
| `bottleneck3` (Conv2d) | `[B, 128, M/2, M/4]` | `[B, 64, M/2, M/4]` | kernel=3×3, pad=1 |
| Anti-rectifier | `[B, 64, M/2, M/4]` | `[B, 128, M/2, M/4]` | ReLU(x) ∥ ReLU(-x) |
| `bn_bottleneck3` (BatchNorm2d) | `[B, 128, M/2, M/4]` | `[B, 128, M/2, M/4]` | 128 features |

### 2.5 Decoder Block 3

| Layer | Input Shape | Output Shape | Parameters |
|-------|-------------|--------------|------------|
| `upconv3` (ConvTranspose2d) | `[B, 128, M/2, M/4]` | `[B, 32, M, M/2]` | kernel=2, stride=2 |
| Concatenate with x2_skip | `[B, 32+64, M, M/2]` | `[B, 96, M, M/2]` | Skip connection |
| `dec_conv3` (Conv2d) | `[B, 96, M, M/2]` | `[B, 32, M, M/2]` | kernel=3×3, pad=1 |
| Anti-rectifier | `[B, 32, M, M/2]` | `[B, 64, M, M/2]` | ReLU(x) ∥ ReLU(-x) |
| `dec_bn3` (BatchNorm2d) | `[B, 64, M, M/2]` | `[B, 64, M, M/2]` | 64 features |
| `dec_conv3_2` (Conv2d) | `[B, 64, M, M/2]` | `[B, 32, M, M/2]` | kernel=3×3, pad=1 |
| Anti-rectifier | `[B, 32, M, M/2]` | `[B, 64, M, M/2]` | ReLU(x) ∥ ReLU(-x) |
| `dec_bn3_2` (BatchNorm2d) | `[B, 64, M, M/2]` | `[B, 64, M, M/2]` | 64 features |

### 2.6 Decoder Block 4

| Layer | Input Shape | Output Shape | Parameters |
|-------|-------------|--------------|------------|
| `upconv4` (ConvTranspose2d) | `[B, 64, M, M/2]` | `[B, 16, 2M, M]` | kernel=2, stride=2 |
| Concatenate with x1_skip | `[B, 16+32, 2M, M]` | `[B, 48, 2M, M]` | Skip connection |
| `dec_conv4` (Conv2d) | `[B, 48, 2M, M]` | `[B, 16, 2M, M]` | kernel=3×3, pad=1 |
| Anti-rectifier | `[B, 16, 2M, M]` | `[B, 32, 2M, M]` | ReLU(x) ∥ ReLU(-x) |
| `dec_bn4` (BatchNorm2d) | `[B, 32, 2M, M]` | `[B, 32, 2M, M]` | 32 features |
| `dec_conv4_2` (Conv2d) | `[B, 32, 2M, M]` | `[B, 16, 2M, M]` | kernel=3×3, pad=1 |
| Anti-rectifier | `[B, 16, 2M, M]` | `[B, 32, 2M, M]` | ReLU(x) ∥ ReLU(-x) |
| `dec_bn4_2` (BatchNorm2d) | `[B, 32, 2M, M]` | `[B, 32, 2M, M]` | 32 features |

### 2.7 Output Layers

| Layer | Input Shape | Output Shape | Parameters |
|-------|-------------|--------------|------------|
| `final_conv` (Conv2d) | `[B, 32, 2M, M]` | `[B, τ, 2M, M]` | kernel=1×1 |
| `dropout` (Dropout) | `[B, τ, 2M, M]` | `[B, τ, 2M, M]` | p=0.2 |
| `tau_to_single_conv` (Conv2d) | `[B, τ, 2M, M]` | `[B, 1, 2M, M]` | kernel=1×1 |
| Squeeze | `[B, 1, 2M, M]` | `[B, 2M, M]` | Remove channel dim |

**Base UNet Output:** `unet_features [B, 2M, M]`

---

## 3. Eigenvalue Prediction Head

Takes UNet features and predicts M eigenvalues (positive, sorted in descending order).

| Layer | Input Shape | Output Shape | Operation |
|-------|-------------|--------------|-----------|
| Unsqueeze | `[B, 2M, M]` | `[B, 1, 2M, M]` | Add channel dimension |
| `AdaptiveAvgPool2d((1,1))` | `[B, 1, 2M, M]` | `[B, 1, 1, 1]` | Global average pooling |
| `Flatten` | `[B, 1, 1, 1]` | `[B, 1]` | Flatten to vector |
| `Linear(1, 64)` | `[B, 1]` | `[B, 64]` | Fully connected |
| `ReLU` | `[B, 64]` | `[B, 64]` | Activation |
| `Dropout(0.1)` | `[B, 64]` | `[B, 64]` | Regularization |
| `Linear(64, 128)` | `[B, 64]` | `[B, 128]` | Fully connected |
| `ReLU` | `[B, 128]` | `[B, 128]` | Activation |
| `Dropout(0.1)` | `[B, 128]` | `[B, 128]` | Regularization |
| `Linear(128, M)` | `[B, 128]` | `[B, M]` | Output layer |
| `ReLU` | `[B, M]` | `[B, M]` | Ensure positive |
| `torch.sort(descending=True)` | `[B, M]` | `[B, M]` | Sort eigenvalues |

**Output:** `eigenvals [B, M]` - Real positive eigenvalues in descending order

---

## 4. Eigenvector Prediction Head

Takes UNet features and predicts M eigenvectors in real/imaginary stacked format.

| Layer | Input Shape | Output Shape | Operation |
|-------|-------------|--------------|-----------|
| Unsqueeze | `[B, 2M, M]` | `[B, 1, 2M, M]` | Add channel dimension |
| `Conv2d(1, 32, 3, pad=1)` | `[B, 1, 2M, M]` | `[B, 32, 2M, M]` | Convolution |
| `BatchNorm2d(32)` | `[B, 32, 2M, M]` | `[B, 32, 2M, M]` | Normalization |
| `ReLU` | `[B, 32, 2M, M]` | `[B, 32, 2M, M]` | Activation |
| `Dropout2d(0.1)` | `[B, 32, 2M, M]` | `[B, 32, 2M, M]` | Regularization |
| `Conv2d(32, 16, 3, pad=1)` | `[B, 32, 2M, M]` | `[B, 16, 2M, M]` | Convolution |
| `BatchNorm2d(16)` | `[B, 16, 2M, M]` | `[B, 16, 2M, M]` | Normalization |
| `ReLU` | `[B, 16, 2M, M]` | `[B, 16, 2M, M]` | Activation |
| `Conv2d(16, 1, 3, pad=1)` | `[B, 16, 2M, M]` | `[B, 1, 2M, M]` | Output convolution |
| `Tanh` | `[B, 1, 2M, M]` | `[B, 1, 2M, M]` | Bound to [-1, 1] |
| Squeeze | `[B, 1, 2M, M]` | `[B, 2M, M]` | Remove channel dim |

**Output:** `eigenvecs_raw [B, 2M, M]` - Stacked real/imaginary format

---

## 5. Orthogonalization Module

Converts raw predictions to proper orthonormal complex eigenvectors.

### 5.1 Complex Conversion

| Step | Input | Output | Operation |
|------|-------|--------|-----------|
| Split real | `[B, 2M, M]` | `[B, M, M]` | `eigenvecs_raw[:, :M, :]` |
| Split imag | `[B, 2M, M]` | `[B, M, M]` | `eigenvecs_raw[:, M:, :]` |
| Complex combine | 2 × `[B, M, M]` | `[B, M, M]` complex | `torch.complex(real, imag)` |

### 5.2 QR Orthogonalization (Primary)

| Step | Input | Output | Operation |
|------|-------|--------|-----------|
| QR decomposition | `[B, M, M]` complex | Q: `[B, M, M]`, R: `[B, M, M]` | `torch.linalg.qr()` |
| Extract R diagonal | R `[B, M, M]` | `[B, M]` | `torch.diagonal(R)` |
| Compute signs | `[B, M]` | `[B, M]` | `torch.sign(R_diag.real)` |
| Sign correction | Q, signs | `[B, M, M]` | `Q * signs.unsqueeze(-2)` |
| Phase reference | `[B, M, M]` | `[B, M]` | `torch.angle(Q[:, 0, :])` |
| Phase correction | `[B, M, M]`, `[B, M]` | `[B, M, M]` | `Q * exp(-j*phase)` |

### 5.3 Gram-Schmidt (Fallback)

Used when QR fails. Iterative orthogonalization of column vectors.

**Output:** `eigenvecs [B, M, M]` - Complex orthonormal eigenvectors

---

## 6. Eigenvector Sorting

Reorders eigenvectors to match sorted eigenvalues.

| Step | Input | Output | Operation |
|------|-------|--------|-----------|
| Sort eigenvalues | `[B, M]` | `[B, M]`, indices `[B, M]` | `torch.sort(descending=True)` |
| Create index grids | - | Batch, row, col indices | Broadcasting |
| Gather eigenvectors | `[B, M, M]`, indices | `[B, M, M]` | Advanced indexing |

---

## 7. Covariance Reconstruction

Reconstructs the covariance matrix from eigendecomposition.

| Step | Input | Output | Formula |
|------|-------|--------|---------|
| Convert to complex | `eigenvals [B, M]` | `[B, M]` complex | Type cast |
| Create Λ matrix | `[B, M]` | `[B, M, M]` | `torch.diag_embed()` |
| Reconstruct | V, Λ | `[B, M, M]` complex | `V @ Λ @ V^H` |

**Output:** `reconstructed_cov [B, M, M]` - Complex covariance matrix

---

## 8. Final Outputs

The model returns a tuple of three tensors:

| Output | Shape | Type | Description |
|--------|-------|------|-------------|
| `eigenvals` | `[B, M]` | `float32` | Positive eigenvalues (descending) |
| `eigenvecs` | `[B, M, M]` | `complex64` | Orthonormal eigenvectors |
| `reconstructed_cov` | `[B, M, M]` | `complex64` | Reconstructed covariance = V @ Λ @ V^H |

---

## Parameter Count Summary

For M=8, τ=8 with anti-rectifier activation:

| Component | Parameters | Percentage |
|-----------|------------|------------|
| Base UNet | ~320,000 | 95.2% |
| Eigenvalue Head | ~9,480 | 2.8% |
| Eigenvector Head | ~6,560 | 2.0% |
| **Total** | **336,003** | 100% |

---

## Anti-Rectifier Activation

The anti-rectifier activation doubles channels by concatenating positive and negative activations:

```
anti_rectifier(x) = [ReLU(x), ReLU(-x)]
```

**Properties:**
- Doubles number of channels
- Preserves both positive and negative information
- Improves gradient flow for complex-valued signal processing
- No learnable parameters

---

## Example Dimensions (M=8, τ=8, B=32)

| Stage | Shape |
|-------|-------|
| Input | `[32, 8, 16, 8]` |
| After Encoder 1 | `[32, 32, 16, 8]` |
| After Pool 1 | `[32, 32, 8, 4]` |
| After Encoder 2 | `[32, 64, 8, 4]` |
| After Pool 2 | `[32, 64, 4, 2]` |
| After Encoder 3 | `[32, 128, 4, 2]` |
| Bottleneck | `[32, 128, 4, 2]` |
| After Decoder 3 | `[32, 64, 8, 4]` |
| After Decoder 4 | `[32, 32, 16, 8]` |
| UNet Output | `[32, 16, 8]` |
| Eigenvalues | `[32, 8]` |
| Eigenvectors | `[32, 8, 8]` complex |
| Reconstructed | `[32, 8, 8]` complex |

---

## Key Design Features

1. **Direct EVD Prediction**: Avoids inverting matrix operations during training
2. **Orthogonality Enforcement**: QR decomposition ensures valid eigenvectors
3. **Phase Referencing**: Consistent eigenvector phases for stable training
4. **Positive Eigenvalues**: ReLU activation ensures physical validity
5. **Sorted Outputs**: Descending eigenvalue order for consistent interpretation
6. **Skip Connections**: Preserves spatial information through encoder-decoder
7. **Anti-Rectifier**: Handles complex-valued data characteristics

---

## Source Code Reference

- Implementation: `Tri4Net/src/models/deep_learning/EVDUNet.py`
- Base UNet: Lines 42-256
- EVD UNet: Lines 258-445


