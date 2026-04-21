# Extract Plotted Values from Evaluation Results

This document describes the scripts for extracting and working with the aggregated values that are plotted from the raw evaluation results.

## Overview

The `controlled_angle_evaluation.py` script generates JSON files containing **raw per-sample error values** for each algorithm at each SNR level. When plotting, these raw values are aggregated using different methods depending on the algorithm type:

- **Classic Algorithms (MUSIC, MVDR, etc.)**: RMS aggregation
  ```
  RMSE = sqrt(mean(errors²))
  ```
  
- **CRLB (Theoretical Bound)**: Simple mean
  ```
  Mean CRLB = mean(crlb_values)
  ```

The scripts in this directory allow you to:
1. Extract these aggregated values into a clean JSON format
2. Verify the extraction by recreating the plots

## Files

### `extract_plotted_values.py`
Extracts aggregated values from raw results JSON files.

**Input Format:**
```json
{
  "-20": {
    "MUSIC": [6.94, 11.57, 11.11, ...],  // 1000 samples
    "MVDR": [0.39, ...],
    "CRLB": [...]
  },
  "-15": { ... },
  ...
}
```

**Output Format:**
```json
{
  "-20": {
    "MUSIC": 47.79,      // Single aggregated value
    "MVDR": 48.13,
    "CRLB": 3.12
  },
  "-15": { ... },
  ...
}
```

**Usage:**
```bash
# Basic usage (auto-generates output filename with _plotted_values suffix)
python extract_plotted_values.py results.json

# Specify output filename
python extract_plotted_values.py results.json output.json

# Example with actual file
python extract_plotted_values.py \
  src/evaluation/controlled_angle_evaluation/results_mode4_1src_mp0_perfect.json
```

**Safety Feature:**
The script **never overwrites** the input file. If you specify the same filename as output, it automatically adds the `_plotted_values` suffix to prevent data loss.

**Output:**
```
📖 Loading: src/evaluation/controlled_angle_evaluation/results_mode4_1src_mp0_perfect.json

📊 Processing 9 SNR levels...
   Algorithms found: MUSIC, MVDR, Beamformer, ESPRIT, UnitaryESPRIT, RootMUSIC, CRLB

📈 Summary:
   Total SNR levels: 9
   Samples per SNR: {'-20': 1000, '-15': 1000, ..., '20': 1000}
   
   Algorithm Statistics:
      MUSIC               : 9000 samples,    0 failures (0.0% failure rate)
      MVDR                : 9000 samples,    0 failures (0.0% failure rate)
      ...

✅ Plotted values saved: results_mode4_1src_mp0_perfect_plotted_values.json
```

### `plot_from_extracted_values.py`
Creates plots directly from extracted plotted values to verify correctness.

**Usage:**
```bash
# Plot from extracted values
python plot_from_extracted_values.py results_plotted_values.json

# Example with actual file
python plot_from_extracted_values.py \
  src/evaluation/controlled_angle_evaluation/results_mode4_1src_mp0_perfect_plotted_values.json
```

## Aggregation Methods Explained

### Why Different Aggregation Methods?

1. **Classic Algorithms - RMS Aggregation**
   - We're combining **errors** from different samples
   - Each sample has its own DOA estimation error
   - RMSE properly accounts for the squared nature of errors
   - Formula: `sqrt(mean(errors²))`
   - This matches how we define RMSE in estimation theory

2. **CRLB - Simple Mean**
   - CRLB values are **theoretical variance/std dev bounds**
   - We're averaging bounds across different angle geometries
   - Simple mean is appropriate for averaging variances
   - Formula: `mean(crlb_values)`
   - This gives us the average theoretical performance bound

### Example Calculation

Given 3 samples at SNR = -10 dB:

**Classic Algorithm (e.g., MUSIC):**
```
Raw errors: [5.2°, 3.1°, 4.8°]

RMS aggregation:
  RMSE = sqrt((5.2² + 3.1² + 4.8²) / 3)
       = sqrt((27.04 + 9.61 + 23.04) / 3)
       = sqrt(19.90)
       = 4.46°
```

**CRLB:**
```
Raw CRLB values: [0.8°, 0.9°, 0.85°]

Mean aggregation:
  Mean = (0.8 + 0.9 + 0.85) / 3
       = 0.85°
```

## Data Flow

```
controlled_angle_evaluation.py
    |
    | Generates raw per-sample results
    v
results_mode4_1src_mp0_perfect.json
    |  (9 SNR × 1000 samples × 7 algorithms = 63,000 values)
    |
    | extract_plotted_values.py
    v
results_mode4_1src_mp0_perfect_plotted_values.json
    |  (9 SNR × 7 algorithms = 63 aggregated values)
    |
    | plot_from_extracted_values.py
    v
results_mode4_1src_mp0_perfect_plotted_values_plot.png
```

## Verification

To verify the extraction is correct, compare:

1. **Original plot** from `controlled_angle_evaluation.py`:
   ```
   src/evaluation/controlled_angle_evaluation/rmse_vs_snr_mode4_1src_mp0_perfect.png
   ```

2. **Extracted values plot** from `plot_from_extracted_values.py`:
   ```
   src/evaluation/controlled_angle_evaluation/results_mode4_1src_mp0_perfect_plotted_values_plot.png
   ```

These should be **identical** because the extracted values represent exactly what gets plotted.

## Use Cases

### 1. Quick Data Access
Instead of loading and processing 1000s of samples, load the compact plotted values JSON:
```python
import json

# Load extracted values
with open('results_plotted_values.json', 'r') as f:
    data = json.load(f)

# Get MUSIC performance at SNR = 0 dB
music_error = data['0']['MUSIC']  # Single value, not 1000 samples
```

### 2. Comparative Analysis
Easily compare different evaluation runs:
```python
import json
import pandas as pd

# Load multiple runs
with open('results_mode4_perfect.json') as f:
    perfect = json.load(f)
with open('results_mode4_imperfect.json') as f:
    imperfect = json.load(f)

# Create comparison DataFrame
df = pd.DataFrame({
    'SNR': perfect.keys(),
    'MUSIC_perfect': [perfect[snr]['MUSIC'] for snr in perfect.keys()],
    'MUSIC_imperfect': [imperfect[snr]['MUSIC'] for snr in imperfect.keys()]
})
```

### 3. Custom Visualizations
Create custom plots with additional analysis:
```python
import json
import matplotlib.pyplot as plt

with open('results_plotted_values.json', 'r') as f:
    data = json.load(f)

snr_vals = sorted([float(k) for k in data.keys()])

# Plot with CRLB gap
music = [data[str(int(s))]['MUSIC'] for s in snr_vals]
crlb = [data[str(int(s))]['CRLB'] for s in snr_vals]
gap = [m - c for m, c in zip(music, crlb)]

plt.plot(snr_vals, gap, label='MUSIC - CRLB Gap')
plt.xlabel('SNR (dB)')
plt.ylabel('Performance Gap (degrees)')
plt.show()
```

## File Size Comparison

For Mode 4 evaluation (1 source, 1000 samples per SNR, 9 SNR levels):

- **Raw results**: `results_mode4_1src_mp0_perfect.json`
  - Size: ~20 MB
  - Contains: 9 × 1000 × 7 = 63,000 individual values
  
- **Plotted values**: `results_mode4_1src_mp0_perfect_plotted_values.json`
  - Size: ~2 KB
  - Contains: 9 × 7 = 63 aggregated values
  - **Reduction: 10,000×**

## Notes

- The extraction script handles `None`, `inf`, and `nan` values gracefully
- Failure statistics are reported during extraction
- The output JSON preserves the same SNR key format as the input
- Both scripts work with any evaluation mode (1, 2, 3, or 4)

## References

For more details on the evaluation pipeline, see:
- `controlled_angle_evaluation.py` - Main evaluation script
- `README_evaluation.md` - Evaluation methodology documentation
