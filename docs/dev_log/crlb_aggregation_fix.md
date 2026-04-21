# CRLB Aggregation Fix Documentation

## Problem Identified

The original code was using **median** to aggregate CRLB values across samples, which is **incorrect** for theoretical bounds.

### Why This Was Wrong

1. **CRLB represents standard deviations** (theoretical lower bounds), not actual errors
2. **Algorithm RMSE values represent actual errors** from estimation
3. These two quantities have different statistical properties and should be aggregated differently

### Original (Incorrect) Code
```python
# Wrong: Using median for both algorithms and CRLB
for alg in results_by_snr[snr_vals[0]].keys():
    for snr in snr_vals:
        error_vals = [r for r in results_by_snr[snr][alg] if not np.isinf(r) and not np.isnan(r)]
        rmse = np.median(error_vals)  # ❌ Wrong for CRLB
```

## Solution

Use different aggregation methods based on the type of metric:

### Algorithm Errors (RMSE)
- **Method:** Median
- **Reason:** Robust to outliers and pathological angle configurations
- **Formula:** `median(error₁, error₂, ..., errorₙ)`

### CRLB (Standard Deviations)
- **Method:** RMS (Root Mean Square)
- **Reason:** Proper way to average variances/standard deviations
- **Formula:** `sqrt(mean(σ₁², σ₂², ..., σₙ²))`

## Mathematical Justification

### For CRLB Values

CRLB provides the variance bound for unbiased estimators:
```
Var(θ̂) ≥ CRLB(θ)
```

When we have multiple samples with different angle configurations, each has its own CRLB:
- Sample 1: CRLB₁ = σ₁ (standard deviation)
- Sample 2: CRLB₂ = σ₂
- Sample N: CRLBₙ = σₙ

To get the **average bound** across configurations:
```
CRLB_avg = sqrt(mean(σ₁², σ₂², ..., σₙ²))
         = sqrt((σ₁² + σ₂² + ... + σₙ²) / N)
```

This is the **RMS (Root Mean Square)** of the individual CRLBs.

### Why Not Median for CRLB?

Using median on standard deviations would give:
```
median(σ₁, σ₂, ..., σₙ)  ❌ Incorrect
```

This doesn't properly represent the average theoretical bound because:
1. Standard deviations should be combined through their **variances** (squares)
2. Median ignores the statistical distribution of variances
3. RMS properly accounts for the quadratic nature of variance

### Why Median for Algorithm Errors?

Algorithm RMSE values are actual errors that can have:
- Outliers from pathological angle configurations
- Non-normal distributions
- Occasional failures

Median is robust to these issues:
```
median(e₁, e₂, ..., eₙ)  ✓ Correct for errors
```

## Implementation

### Fixed Code

```python
for alg in results_by_snr[snr_vals[0]].keys():
    mean_rmse = []
    for snr in snr_vals:
        error_vals = [r for r in results_by_snr[snr][alg] if not np.isinf(r) and not np.isnan(r)]
        
        if alg == 'CRLB':
            # CRLB: Use RMS (Root Mean Square) to aggregate standard deviations
            # This gives the average theoretical bound across different angle configurations
            rmse = np.sqrt(np.mean(np.array(error_vals)**2)) if error_vals else np.nan
        else:
            # Algorithm errors: Use MEDIAN (robust to outliers/pathological cases)
            rmse = np.median(error_vals) if error_vals else np.nan
        
        mean_rmse.append(rmse)
```

### Updated in Two Places

1. **Plotting Function** (line ~877):
   - Aggregates values for creating plots
   - Distinguishes between CRLB and algorithm errors

2. **Summary Table** (line ~972):
   - Aggregates values for the printed summary table
   - Shows proper statistics for each metric type

## Impact

### Before Fix
- CRLB values were lower than they should be (median < RMS for positive values)
- Unfair comparison between algorithms and theoretical bounds
- Misleading performance assessment

### After Fix
- CRLB values properly represent the average theoretical bound
- Fair comparison between algorithm performance and theoretical limits
- Correct interpretation of how close algorithms are to optimal performance

## Example Comparison

Given CRLB values: [0.5°, 0.6°, 0.4°, 10.0°] (one outlier)

**Median (wrong):**
```
median([0.5, 0.6, 0.4, 10.0]) = 0.55°
```

**RMS (correct):**
```
RMS = sqrt((0.5² + 0.6² + 0.4² + 10.0²) / 4)
    = sqrt((0.25 + 0.36 + 0.16 + 100) / 4)
    = sqrt(101.77 / 4)
    = sqrt(25.44)
    = 5.04°
```

The RMS properly accounts for the high variance in the outlier case, which represents a genuinely difficult estimation scenario.

## Updated Output

### Summary Table Header
```
📊 Results Summary
   Algorithms: Median Error | CRLB: RMS (Root Mean Square)
```

### Plot Title
```
DOA Estimation Performance vs SNR
(Algorithms: Median Error | CRLB: RMS)
```

These clarify that different aggregation methods are used for different metric types.

## Verification

Run evaluation to verify:
```bash
python controlled_angle_evaluation.py --mode 4 --num_samples 100
```

Check that:
1. ✅ CRLB values are properly aggregated using RMS
2. ✅ Algorithm errors use median (robust to outliers)
3. ✅ Headers clearly indicate the aggregation method
4. ✅ Values are consistent and meaningful

## References

- Kay, S. M. (1993). *Fundamentals of Statistical Signal Processing: Estimation Theory*
  - Chapter 3: Cramér-Rao Lower Bound
  - Explains proper handling of variance bounds

- Van Trees, H. L. (2002). *Optimum Array Processing*
  - Chapter 3: Parameter Estimation Theory
  - Discusses CRLB for DOA estimation
