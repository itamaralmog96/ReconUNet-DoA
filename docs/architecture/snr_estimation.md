# SNR Estimation from Eigenvalues

## Overview

Added SNR estimation capability to both `analyze_eigenvalues.py` and `source_estimation_validation.py`. The SNR is estimated directly from the covariance matrix eigenvalues using a simple but effective method.

## Method

### Formula

```
SNR (dB) = 10 × log₁₀(mean(signal_eigenvalues) / mean(noise_eigenvalues))
```

Where:
- **Signal eigenvalues**: First `n` eigenvalues (where `n` = estimated/true number of sources)
- **Noise eigenvalues**: Remaining eigenvalues

### Rationale

The covariance matrix eigenvalues naturally separate into:
1. **Signal subspace** (large eigenvalues): Dominated by source signals
2. **Noise subspace** (small eigenvalues): Dominated by noise

The ratio between the average signal eigenvalue and average noise eigenvalue provides an estimate of the Signal-to-Noise Ratio.

## Implementation

### Function

```python
def estimate_snr_from_eigenvalues(eigenvalues, num_sources):
    """Estimate SNR in dB from eigenvalues."""
    if num_sources <= 0 or num_sources >= len(eigenvalues):
        return np.nan
    
    signal_eigs = eigenvalues[:num_sources]
    noise_eigs = eigenvalues[num_sources:]
    
    if len(noise_eigs) == 0:
        return np.nan
    
    signal_power = np.mean(signal_eigs)
    noise_power = np.mean(noise_eigs)
    
    if noise_power <= 0:
        return np.nan
    
    snr_linear = signal_power / noise_power
    snr_db = 10 * np.log10(snr_linear)
    
    return snr_db
```

## Results Summary

From the validation run with 4 sources (3 + 1 multipath):

### Noisy Covariance (using true # sources)
```
True SNR    Estimated SNR    Error
-20 dB  →   0.92 dB         +20.92 dB
-10 dB  →   1.01 dB         +11.01 dB
  0 dB  →   2.86 dB          +2.86 dB
 10 dB  →   9.67 dB          -0.33 dB
 20 dB  →   19.22 dB         -0.78 dB
 30 dB  →   29.10 dB         -0.90 dB
 40 dB  →   39.05 dB         -0.95 dB
```

**Observation**: Noisy covariance SNR estimates are accurate at moderate to high SNR (≥10 dB), with error < 1 dB. At low SNR, the method overestimates SNR.

### UNet Eigenvalues (using true # sources)
```
True SNR    Estimated SNR    Error
-20 dB  →   17.54 dB        +37.54 dB (!!)
-10 dB  →   17.88 dB        +27.88 dB (!!)
  0 dB  →   17.76 dB        +17.76 dB (!!)
 10 dB  →   17.76 dB         +7.76 dB
 20 dB  →   17.73 dB         -2.27 dB
 30 dB  →   17.62 dB        -12.38 dB
 40 dB  →   17.57 dB        -22.43 dB
```

**Observation**: UNet eigenvalues consistently estimate ~17-18 dB SNR regardless of true SNR! This is because:
1. UNet is trained to **denoise** covariance matrices
2. Output eigenvalues reflect the **denoised** signal, not the original noisy signal
3. The UNet effectively "normalizes" the SNR to a moderate level (~18 dB)

### UNet with Adaptive Threshold Estimate
```
True SNR    Estimated SNR (Adaptive)
-20 dB  →   14.93 dB
-10 dB  →   14.02 dB
  0 dB  →   13.81 dB
 10 dB  →   13.61 dB
 20 dB  →   14.35 dB
 30 dB  →   13.91 dB
 40 dB  →   14.17 dB
```

**Observation**: When using the Adaptive Threshold estimated number of sources (which tends to overestimate), the SNR estimates are even lower (~14 dB) and very stable across all SNR levels.

## Key Findings

### 1. SNR Estimation from Noisy Covariance ✅
- **Accurate at moderate to high SNR** (≥10 dB): Error < 1 dB
- **Overestimates at low SNR** (< 10 dB): Due to poor signal/noise separation
- **Requires accurate source count**: Uses true number of sources

### 2. SNR Estimation from UNet Eigenvalues ❌
- **NOT suitable for SNR estimation**
- UNet reconstructs covariance for **DOA estimation**, not SNR preservation
- Estimates are consistently ~17-18 dB regardless of input SNR
- This is actually a **feature** of the UNet: it normalizes signal quality

### 3. Dependency on Source Number Estimation
- SNR estimate is **highly sensitive** to the estimated number of sources
- Overestimating sources → underestimate SNR (noise eigenvalues included in signal)
- Underestimating sources → overestimate SNR (signal eigenvalues included in noise)

## Recommendations

### For SNR Estimation:
1. ✅ **Use noisy covariance eigenvalues** with accurate source count
2. ✅ Apply at **moderate to high SNR** (≥10 dB) for best accuracy
3. ❌ **Do NOT use UNet eigenvalues** for SNR estimation

### For Source Counting + SNR:
1. First estimate SNR from noisy covariance (using MDL source estimate)
2. Then use UNet eigenvalues with Adaptive Threshold for robust source counting
3. Report both estimates separately

## Usage Examples

### In analyze_eigenvalues.py:
```python
# Analyze sample
result = analyze_sample(sample, unet_model)

# Access SNR estimates
snr_noisy_true = result['snr_est_noisy_true']      # Using true # sources
snr_unet_true = result['snr_est_unet_true']        # Using true # sources
snr_noisy_mdl = result['snr_est_noisy_mdl']        # Using MDL estimate
snr_unet_mdl = result['snr_est_unet_mdl']          # Using MDL estimate
```

### In source_estimation_validation.py:

The script automatically computes and displays SNR estimates for:
- `SNR_est_noisy_true`: Noisy eigenvalues, true number of sources
- `SNR_est_unet_true`: UNet eigenvalues, true number of sources
- `SNR_est_noisy_mdl`: Noisy eigenvalues, MDL estimated sources
- `SNR_est_unet_adaptive`: UNet eigenvalues, Adaptive Threshold estimated sources
- `SNR_est_unet_ratio`: UNet eigenvalues, Ratio Threshold estimated sources

Results are displayed in the final summary table.

## Limitations

1. **Assumes equal power sources**: All sources have same SNR
2. **Assumes white noise**: Noise eigenvalues should be approximately equal
3. **Requires good eigenvalue separation**: Works best when signal and noise subspaces are well-separated
4. **Sensitive to source count**: Accuracy depends on correct source number estimation

## Future Improvements

1. **Weighted SNR estimation**: Account for varying source powers
2. **Robust averaging**: Use median instead of mean to reduce outlier impact
3. **Confidence intervals**: Provide uncertainty estimates
4. **Colored noise handling**: Adapt for non-white noise scenarios
5. **UNet calibration**: Train UNet variant that preserves SNR information

## References

The eigenvalue-based SNR estimation is related to:
- **Minimum Mean Square Error (MMSE) SNR estimation**
- **Eigenvalue-based noise variance estimation**
- Classical work on **signal subspace methods** (e.g., Wax & Kailath, 1985)

---

**Last Updated**: October 11, 2025
