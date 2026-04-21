# DOA Evaluation Process Documentation

## Overview
This document describes the complete evaluation process for Direction of Arrival (DOA) estimation algorithms, including the EVDUNet deep learning model and classic algorithms (MUSIC, MVDR, Beamformer, ESPRIT, Unitary ESPRIT, Root-MUSIC).

---

## Current Configuration

### Evaluation Mode
- **Mode 4: Random Angles Shared Across SNR**
  - Generates random source angles with minimum separation constraint
  - **Key Feature**: The SAME angle configurations are used across ALL SNR levels
  - This allows fair comparison across SNR by eliminating angle variation as a confounding variable
  - Each sample at different SNR levels uses the same underlying angle geometry
  - Example: If sample #1 has sources at [45°, 120°] at SNR=-20dB, the same angles [45°, 120°] are used at SNR=-15dB, -10dB, etc.

### Signal Configuration
```python
# Source Configuration
NUM_SOURCES = [1, 3]  # [main_sources, interference_sources]
# - 1 main source (the target we're trying to localize)
# - 3 interference sources (unwanted signals from other directions)
# - Total: 4 sources in the environment

# Signal Parameters
NUM_SNAPSHOTS = 512        # Number of time samples per observation
NUM_MULTIPATH = 3          # Number of multipath reflections per main source
SIR_DB = 0.0              # Signal-to-Interference Ratio (0 dB = equal power)

# SNR Configuration
SNR_DB = [-20, -15, -10, -5, 0, 5, 10, 15, 20]  # 9 SNR levels
SAMPLES_PER_SNR = 1000    # 1000 samples per SNR level
# Total dataset size: 9 SNR levels × 1000 samples = 9,000 samples
```

### Array Configuration
```python
ARRAY_TYPE = "linear"          # Uniform Linear Array (ULA)
NUM_ELEMENTS = 8               # 8 antenna elements
CARRIER_FREQ = 2.45e9          # 2.45 GHz (WiFi band)
SAMPLING_FREQ = 1e6            # 1 MHz sampling rate
ARRAY_IMPERFECTIONS = True     # Includes gain/phase errors, mutual coupling
```

### Angle Configuration
```python
ANGLE_RANGE = (30.0, 150.0)    # Valid angle range in degrees
# Mode 4 ensures:
# - Random angles are generated within this range
# - Minimum separation between sources (prevents overlapping)
# - Same angles used across all SNR levels for each sample index
```

---

## Dataset Generation Process (Mode 4)

### Dataset Size Overview
```
Total Samples: 9,000
- SNR Levels: 9 levels [-20, -15, -10, -5, 0, 5, 10, 15, 20 dB]
- Samples per SNR: 1,000 samples
- Calculation: 9 SNR levels × 1,000 samples/SNR = 9,000 total samples

Sample Distribution:
- Samples 0-999:     SNR = -20 dB (1,000 samples)
- Samples 1000-1999: SNR = -15 dB (1,000 samples)
- Samples 2000-2999: SNR = -10 dB (1,000 samples)
- Samples 3000-3999: SNR = -5 dB  (1,000 samples)
- Samples 4000-4999: SNR = 0 dB   (1,000 samples)
- Samples 5000-5999: SNR = 5 dB   (1,000 samples)
- Samples 6000-6999: SNR = 10 dB  (1,000 samples)
- Samples 7000-7999: SNR = 15 dB  (1,000 samples)
- Samples 8000-8999: SNR = 20 dB  (1,000 samples)
```

### Mode 4: How Dataset Creation Works

**Key Principle**: Generate 1,000 unique angle configurations, then create one sample at EACH SNR level for each configuration.

**Creation Process**:

1. **First Loop - Generate 1,000 Angle Configurations** (happens once)
   ```
   For i = 0 to 999:
       Generate random angles:
       - 1 main source angle
       - 3 interference source angles
       - 3 multipath angles
       Store configuration[i] = {main_angle, interference_angles, multipath_angles}
   ```

2. **Second Loop - Create Samples for All SNR Levels** (nested loop)
   ```
   For i = 0 to 999:  (each angle configuration)
       For snr in [-20, -15, -10, -5, 0, 5, 10, 15, 20]:  (each SNR level)
           Use configuration[i] angles
           Generate signals with current SNR
           Add noise according to SNR level
           Compute covariance matrices
           Store as sample[snr_index * 1000 + i]
   ```

**Result**: 
- Sample #0 at SNR=-20dB has same angles as Sample #1000 at SNR=-15dB
- Sample #0 at SNR=-20dB has same angles as Sample #2000 at SNR=-10dB
- ...and so on for all 9 SNR levels
- This allows us to see how the SAME angle configuration performs at different noise levels

**Example**:
```
Angle Configuration #0: {main: 87.3°, interference: [45.2°, 110.8°, 135.6°], multipath: [92.1°, 78.5°, 95.3°]}
- Sample #0    (index 0):    SNR = -20 dB, uses config #0
- Sample #1000 (index 1000): SNR = -15 dB, uses config #0 (SAME angles, different noise)
- Sample #2000 (index 2000): SNR = -10 dB, uses config #0 (SAME angles, different noise)
- ...
- Sample #8000 (index 8000): SNR = +20 dB, uses config #0 (SAME angles, different noise)

Angle Configuration #1: {main: 62.1°, interference: [38.7°, 95.4°, 142.3°], multipath: [67.8°, 58.2°, 71.5°]}
- Sample #1    (index 1):    SNR = -20 dB, uses config #1
- Sample #1001 (index 1001): SNR = -15 dB, uses config #1 (SAME angles, different noise)
- ...
```

### Step 1: Generate Shared Random Angles
For each of the 1,000 unique angle configurations:
1. **Generate random reference angle** for the main source
   - Uniformly sampled from ANGLE_RANGE = (30°, 150°)
   
2. **Generate interference source angles** (3 sources)
   - Each randomly sampled from ANGLE_RANGE
   - Must maintain minimum separation from all other sources
   - Typical minimum separation: ~10-15° to avoid ambiguity

3. **Generate multipath angles** (3 components)
   - Each randomly sampled from ANGLE_RANGE
   - Associated with the main source
   - Represents reflections from walls, objects, etc.

4. **Store angle configuration**
   - This SAME configuration is used for ALL 9 SNR levels
   - Example: Sample #0 might have:
     - Main source: 87.3°
     - Interference: [45.2°, 110.8°, 135.6°]
     - Multipath: [92.1°, 78.5°, 95.3°]

### Step 2: Generate Signals for Each SNR Level
For each sample and each SNR level (-20 dB to +20 dB):

1. **Generate Source Signals**
   ```
   - Main source: s_main(t) = complex narrowband signal, T=512 samples
   - Interference sources: s_int_1(t), s_int_2(t), s_int_3(t)
   - Power adjustment: SIR = 0 dB → all sources have equal power
   ```

2. **Generate Multipath Components**
   ```
   - For each multipath reflection:
     - Same waveform as main source (coherent)
     - Attenuated power (typically -6 dB to -12 dB)
     - Different angle of arrival (from angle configuration)
   ```

3. **Create Steering Vectors**
   ```
   For ULA with M=8 elements, element spacing d=λ/2:
   
   a(θ) = [1, e^(jπ cos(θ)), e^(j2π cos(θ)), ..., e^(j7π cos(θ))]^T / √8
   
   Where θ is the angle in degrees
   ```

4. **Apply Array Imperfections** (if enabled)
   ```
   - Gain errors: Each element has random gain ±0.5 dB
   - Phase errors: Each element has random phase ±3°
   - Mutual coupling: Cross-talk between adjacent elements
   ```

5. **Generate Received Signal**
   ```
   x(t) = Σ(sources) a(θ_k) * s_k(t) + Σ(multipath) a(θ_mp) * α_mp * s_main(t) + n(t)
   
   Where:
   - x(t): M×T received signal matrix (8 elements × 512 snapshots)
   - a(θ_k): Steering vector for source k
   - s_k(t): Source signal k
   - α_mp: Multipath attenuation factor
   - n(t): Additive white Gaussian noise (power determined by SNR)
   ```

6. **Compute Covariance Matrix**
   ```
   R = (1/T) * x * x^H
   
   Result: 8×8 complex Hermitian matrix
   - Contains spatial correlation information
   - Noisy due to finite snapshots and array imperfections
   ```

7. **Compute Clean Covariance Matrix** (ground truth)
   ```
   R_clean = Σ(sources) a(θ_k) * a(θ_k)^H + Σ(multipath) a(θ_mp) * |α_mp|^2 * a(θ_mp)^H
   
   - No noise component
   - Used as training target for UNet
   ```

8. **Compute Autocorrelation Tensor** (UNet input)
   ```
   For τ = 0, 1, ..., 7 lags:
   
   R(τ) = (1/(T-τ)) * Σ_t x(t) * x(t+τ)^H
   
   Stored as (8, 16, 8) real tensor:
   - 8 time lags
   - 16 = 2×M channels (real and imaginary parts)
   - 8 elements
   ```

### Step 3: Save Dataset
```
Dataset structure:
Data/datasets/controlled_eval_mode4_YYYYMMDD_HHMMSS/
├── data.npz (or .h5)
│   ├── received_signal: (9000, 8, 512) complex
│   ├── covariance_matrix: (9000, 8, 8) complex
│   ├── clean_covariance_matrix: (9000, 8, 8) complex
│   ├── autocorrelation_matrix: (9000, 8, 16, 8) float
│   └── steering_vectors: (9000, 4, 8) complex (4 total sources)
└── labels.npz (or .h5)
    ├── doas: (9000, 10) float  # [main(1), interference(3), multipath(3), padding(3)]
    ├── snr: (9000,) float
    ├── num_sources: (9000, 2) int  # [main, interference]
    ├── num_multipath: (9000,) int
    └── smr: (9000, 3) float  # Signal-to-Multipath Ratio for each path

Note: Sample indices 0-999 correspond to SNR=-20dB
      Sample indices 1000-1999 correspond to SNR=-15dB
      ...and so on
```

---

## Evaluation Process

### Step 1: Load Pre-trained UNet Model
```python
Model: EVDCovarianceReconstructionUNet
Architecture:
- Input: Autocorrelation tensor (8 lags, 16 channels, 8 elements)
- Encoder: 4 convolutional blocks with anti-rectifier activation
- Bottleneck: Eigenvalue decomposition branch
- Decoder: 4 transposed convolutional blocks
- Output: Reconstructed clean covariance matrix (8×8 complex)

Model Path: 'notebooks/evd_unet_denoising_model_20250929_015132.pth'
```

### Step 2: Evaluate Each Sample
For each of the 9,000 samples in the dataset:

#### A. Classic Algorithms (Direct on Noisy Covariance)
```python
Input: R (noisy covariance matrix from received signal)

1. MUSIC (Multiple Signal Classification)
   - Eigendecomposition: R = U_s Λ_s U_s^H + U_n Λ_n U_n^H
   - Noise subspace: U_n (last M-K eigenvectors)
   - Spectrum: P_MUSIC(θ) = 1 / (a(θ)^H U_n U_n^H a(θ))
   - Peaks: Locate 4 highest peaks → estimated DOAs

2. MVDR (Minimum Variance Distortionless Response / Capon)
   - Spectrum: P_MVDR(θ) = 1 / (a(θ)^H R^(-1) a(θ))
   - Peaks: Locate 4 highest peaks → estimated DOAs

3. Beamformer (Conventional / Bartlett)
   - Spectrum: P_BF(θ) = a(θ)^H R a(θ)
   - Peaks: Locate 4 highest peaks → estimated DOAs

4. ESPRIT (Estimation of Signal Parameters via Rotational Invariance)
   - Split array into two overlapping subarrays
   - Eigendecomposition to find signal subspace
   - Compute rotation matrix Φ
   - DOAs from eigenvalues: θ = arccos(angle(λ)/π)
   - Only for linear arrays

5. Unitary ESPRIT
   - Real-valued version of ESPRIT
   - More computationally efficient
   - Uses centro-Hermitian property
   - Only for linear arrays

6. Root-MUSIC
   - Polynomial rooting instead of spectrum search
   - Faster than MUSIC
   - More accurate peak finding
   - Only for linear arrays
```

#### B. UNet + Follow-up Algorithm
```python
Current Configuration: UNET_FOLLOWUP_ALGORITHM = 'ESPRIT'

Process:
1. Input: Autocorrelation tensor (8, 16, 8)
2. UNet Forward Pass:
   - Denoise and reconstruct: R_clean_estimated = UNet(autocorr)
3. Apply ESPRIT on clean covariance:
   - Input: R_clean_estimated
   - Output: 4 estimated DOAs
```

#### C. RMSE Calculation
```python
For each algorithm's estimates:
1. Match estimates to ground truth using optimal assignment
   - Handles case where algorithm estimates K sources
   - We evaluate on ALL 4 sources (1 main + 3 interference)
   - Hungarian algorithm finds best matching

2. Calculate RMSE:
   RMSE = sqrt(mean((true_angles - estimated_angles)^2))
   
   - Penalty: 240° for missed detections
   - Tracks: Total calls, penalties applied, missed sources

3. Store result for this sample and algorithm
```

### Step 3: Aggregate Results by SNR
```python
For each SNR level (-20 dB to +20 dB):
1. Collect all 1000 RMSE values per algorithm
2. Compute overall RMSE:
   
   For classic algorithms and UNet:
   RMSE_overall = sqrt(mean(all_rmse_values^2))
   
   For CRLB (if computed):
   CRLB_avg = mean(all_crlb_values)

3. Store in results_by_snr dictionary
```

---

## Algorithms Being Evaluated

### Classic Algorithms (6 total)
1. ✅ **MUSIC** - Spectrum search, subspace method
2. ✅ **MVDR** - Adaptive beamforming
3. ✅ **Beamformer** - Conventional delay-and-sum
4. ✅ **ESPRIT** - Rotational invariance (linear arrays only)
5. ✅ **Unitary ESPRIT** - Real-valued ESPRIT variant
6. ✅ **Root-MUSIC** - Polynomial rooting (linear arrays only)

### Deep Learning Method (1 total)
7. ✅ **ReconUNet_ESPRIT** - UNet denoising + ESPRIT on cleaned covariance

### Theoretical Bound (optional)
8. ⚠️ **CRLB** - Cramér-Rao Lower Bound (currently disabled due to multipath complexity)

---

## Output Files

### 1. Results JSON
```
Tri4Net/src/evaluation/controlled_angle_evaluation/
└── results_mode4_4src_mp3_imperfect.json

Structure:
{
  "-20": {
    "MUSIC": [rmse_sample_0, rmse_sample_1, ..., rmse_sample_999],
    "MVDR": [...],
    "Beamformer": [...],
    "ESPRIT": [...],
    "UnitaryESPRIT": [...],
    "RootMUSIC": [...],
    "ReconUNet_ESPRIT": [...]
  },
  "-15": { ... },
  ...
}
```

### 2. Performance Plot
```
Tri4Net/src/evaluation/controlled_angle_evaluation/
└── rmse_vs_snr_mode4_4src_mp3_imperfect.png

Plot features:
- X-axis: SNR (dB) from -20 to +20
- Y-axis: RMSE (degrees) in log scale
- Lines: One per algorithm with distinct colors/markers
- Legend: Algorithm names (fontsize=16)
- Title: Evaluation mode, source configuration, RMSE aggregation method
```

---

## Key Differences from Training

### Training Process (Not in This Script)
The UNet model was trained separately with:
- **Input**: Autocorrelation tensors from noisy received signals
- **Target**: Clean covariance matrices (ground truth without noise)
- **Loss**: Mean Squared Error between predicted and clean covariance
- **Objective**: Learn to denoise and reconstruct clean spatial covariance

### Evaluation Process (This Script)
This script performs:
- **No training**: Only inference/testing
- **Load pre-trained model**: From saved checkpoint
- **Compare multiple algorithms**: UNet + 6 classic methods
- **Metric**: RMSE between estimated and true DOAs
- **Objective**: Measure DOA estimation accuracy across SNR levels

---

## Performance Metrics

### Success Metrics
- **RMSE (Root Mean Square Error)**: Primary metric in degrees
- **Failure Rate**: Percentage of samples where algorithm failed to detect sources
- **Penalty Statistics**: Number of missed detections per algorithm

### Expected Performance Trends
1. **Low SNR (-20 to -10 dB)**:
   - High RMSE (10-100 degrees)
   - Classical algorithms struggle
   - UNet should show improvement

2. **Medium SNR (-5 to 5 dB)**:
   - Moderate RMSE (1-10 degrees)
   - Algorithms start to converge
   - Clear separation between methods

3. **High SNR (10 to 20 dB)**:
   - Low RMSE (<1 degree)
   - All algorithms perform well
   - UNet should match or exceed classical methods

---

## Summary

This evaluation script:
1. ✅ Generates 9,000 synthetic samples with controlled angles (Mode 4)
2. ✅ Uses SAME angles across all SNR levels for fair comparison
3. ✅ Simulates realistic conditions: 1 main + 3 interference + 3 multipath + array imperfections
4. ✅ Evaluates 7 algorithms (6 classic + 1 deep learning)
5. ✅ Computes RMSE on ALL 4 sources (main + interference)
6. ✅ Aggregates results by SNR level
7. ✅ Generates publication-ready plots
8. ✅ Tracks detailed failure statistics

**Total evaluation time**: ~10-30 minutes depending on hardware
**Total samples processed**: 9,000 samples × 7 algorithms = 63,000 evaluations
