# Angle Sweep Evaluation Script

## Overview

# Angle Sweep Evaluation Script

## Overview

This script evaluates DOA estimation algorithms by sweeping the angle of arrival across a range while keeping the source separation constant. **NOW USES EXACT DATASET GENERATOR SAMPLE CREATION PROCESS.**

## Features

- **Multiple Sources**: Configure any number of sources
- **Fixed Separation**: Keep constant angular separation between sources
- **Monte Carlo Simulation**: Generate multiple samples per angle for statistical analysis
- **Algorithm Selection**: Choose from multiple DOA estimation methods:
  - `music` - MUSIC algorithm (Noisy covariance)
  - `root_music` - Root-MUSIC algorithm (Noisy covariance)
  - `esprit` - ESPRIT algorithm (Noisy covariance)
  - `unet_music` - MUSIC with UNet-denoised covariance
  - `unet_root_music` - Root-MUSIC with UNet-denoised covariance
  - `unet_esprit` - ESPRIT with UNet-denoised covariance
- **Automated Plotting**: Generates two publication-quality plots:
  1. Estimated vs True angles (with zoomed inset)
  2. Angle errors vs sample index

## Configuration

Edit the configuration section at the top of `angle_sweep_evaluation.py`:

```python
# === SOURCE CONFIGURATION ===
NUM_SOURCES = 2  # Number of sources to track
SOURCE_SEPARATION = 10.0  # Degrees between sources
NUM_SNAPSHOTS = 512

# === ANGLE SWEEP CONFIGURATION ===
ANGLE_START = 30.0  # Starting angle (degrees)
ANGLE_STOP = 120.0  # Ending angle (degrees)
ANGLE_STEP = 1.0  # Step size (degrees)

# === SNR CONFIGURATION ===
SNR_DB = 10.0  # Signal-to-noise ratio in dB

# === MONTE CARLO SIMULATION ===
SAMPLES_PER_ANGLE = 100  # Number of samples to generate per angle

# === ALGORITHM SELECTION ===
ALGORITHM = 'music'  # Options: 'music', 'root_music', 'esprit', 'unet_music', etc.

# === ARRAY CONFIGURATION ===
ARRAY_TYPE = "linear"
NUM_ELEMENTS = 8
```

## How It Works

### 1. Angle Sweep

For each angle position from `ANGLE_START` to `ANGLE_STOP`:
- **First source**: Placed at the current angle position
- **Additional sources**: Placed at `angle + i * SEPARATION` for source i

Example with 2 sources, separation=10°:
- Position 1: θ₁=30°, θ₂=40°
- Position 2: θ₁=31°, θ₂=41°
- Position 3: θ₁=32°, θ₂=42°
- ...
- Position 91: θ₁=120°, θ₂=130°

### 2. Monte Carlo Simulation

At each angle position:
1. Generate `SAMPLES_PER_ANGLE` independent samples (default: 100)
2. Estimate DOAs for each sample using the selected algorithm
3. Compute statistics:
   - Mean estimated angles
   - Standard deviation of estimates
   - Mean errors (estimated - true)
   - Standard deviation of errors

### 3. Output Plots

#### Plot 1: Estimated vs True Angles
- **Main plot**: Shows entire sweep range
  - Solid lines: True angles for each source
  - Open circles: Estimated angles (averaged over all samples)
  - Different colors for each source
- **Zoomed inset**: Shows middle section with more detail
- **Arrow**: Points from main plot to zoomed region

#### Plot 2: Angle Errors
- Shows Δθᵢ = θᵢ - θ̂ᵢ for each source
- Horizontal line at zero for reference
- Points close to zero indicate accurate estimation

## Usage Examples

### Example 1: Basic 2-Source Sweep
```python
NUM_SOURCES = 2
SOURCE_SEPARATION = 10.0
SNR_DB = 10.0
ANGLE_START = 30.0
ANGLE_STOP = 120.0
ANGLE_STEP = 1.0
SAMPLES_PER_ANGLE = 100
ALGORITHM = 'music'
```

This will:
- Sweep from 30° to 120° in 1° steps (91 positions)
- Generate 100 samples per position (9,100 total samples)
- Run for approximately 5-10 minutes

### Example 2: High-Resolution 3-Source Sweep
```python
NUM_SOURCES = 3
SOURCE_SEPARATION = 5.0  # Closer spacing
SNR_DB = 15.0
ANGLE_START = 40.0
ANGLE_STOP = 100.0
ANGLE_STEP = 0.5  # Finer steps
SAMPLES_PER_ANGLE = 200  # More samples for better statistics
ALGORITHM = 'root_music'
```

### Example 3: UNet-Enhanced Evaluation
```python
NUM_SOURCES = 2
SOURCE_SEPARATION = 10.0
SNR_DB = 5.0  # Low SNR
SAMPLES_PER_ANGLE = 100
ALGORITHM = 'unet_music'  # Use UNet denoising
```

### Example 4: Quick Test Run
```python
NUM_SOURCES = 2
SOURCE_SEPARATION = 15.0
ANGLE_START = 50.0
ANGLE_STOP = 80.0  # Shorter range
ANGLE_STEP = 2.0  # Larger steps
SAMPLES_PER_ANGLE = 50  # Fewer samples
ALGORITHM = 'esprit'
```

This will run in ~2-3 minutes for quick testing.

## Running the Script

```bash
cd "/path/to/Tri4Net"
python angle_sweep_evaluation.py
```

## Output

### Console Output
```
🚀 Angle Sweep DOA Evaluation
======================================================================

📋 Configuration:
   Sources: 2
   Separation: 10.0°
   SNR: 10.0 dB
   Angle range: 30.0° to 120.0° (step: 1.0°)
   Samples per angle: 100
   Algorithm: music
   Array: 8 element linear

✅ Array model created

🎯 Angle Sweep Evaluation
======================================================================
...
Sweeping angles: 100%|████████████████| 91/91

📊 Creating plots...
✅ Plot saved: .../angle_sweep_music_2src_sep10.0deg_snr10.0dB.png
✅ Error plot saved: .../angle_errors_music_2src_sep10.0deg_snr10.0dB.png

======================================================================
📊 Results Summary
======================================================================
Algorithm: music
Number of angle positions: 91

Per-Source Statistics:

  Source 1:
    Mean Error: +0.0234°
    Std Error:  0.1567°
    RMSE:       0.1584°

  Source 2:
    Mean Error: -0.0189°
    Std Error:  0.1432°
    RMSE:       0.1445°
```

### Saved Files

Located in: `Tri4Net/src/evaluation/angle_sweep/`

1. **`angle_sweep_{algorithm}_{num_sources}src_sep{separation}deg_snr{snr}dB.png`**
   - Main plot with true vs estimated angles
   - Includes zoomed inset

2. **`angle_errors_{algorithm}_{num_sources}src_sep{separation}deg_snr{snr}dB.png`**
   - Error plot showing estimation accuracy

## Tips & Best Practices

### 1. Choosing SAMPLES_PER_ANGLE
- **50-100**: Quick testing, reasonable statistics
- **100-200**: Good balance for most evaluations
- **500-1000**: Publication-quality results, very stable statistics

### 2. Choosing ANGLE_STEP
- **0.5-1.0°**: Standard resolution, smooth plots
- **2.0-5.0°**: Quick evaluation, coarser plots
- **0.1-0.3°**: High resolution, slow but very detailed

### 3. Algorithm Selection
- **MUSIC**: Most common, works well for most scenarios
- **Root-MUSIC**: Faster than MUSIC, no grid search needed
- **ESPRIT**: Good for ULA, computationally efficient
- **UNet variants**: Use for low SNR scenarios (SNR < 5 dB)

### 4. Valid Angle Ranges
- Keep `ANGLE_START` ≥ 30°
- Keep `ANGLE_STOP` + `(NUM_SOURCES-1) * SEPARATION` ≤ 150°
- This ensures all sources stay within the observable range

### 5. Estimated Runtime
Approximate runtime formula:
```
Time ≈ NUM_POSITIONS × SAMPLES_PER_ANGLE × 0.1 seconds

where NUM_POSITIONS = (ANGLE_STOP - ANGLE_START) / ANGLE_STEP
```

Examples:
- 91 positions × 100 samples = ~15 minutes
- 121 positions × 200 samples = ~40 minutes
- 16 positions × 50 samples = ~2 minutes (quick test)

## Troubleshooting

### Script runs too slowly
- Reduce `SAMPLES_PER_ANGLE` (try 50)
- Increase `ANGLE_STEP` (try 2.0 or 5.0)
- Reduce angle range (smaller `ANGLE_STOP - ANGLE_START`)

### Errors about angles out of range
- Check that `ANGLE_STOP + (NUM_SOURCES-1) * SEPARATION ≤ 150`
- Adjust either `ANGLE_STOP` or `SEPARATION`

### UNet model not found
- Ensure `UNET_MODEL_PATH` points to valid model file
- Download model or adjust path

### Poor estimation accuracy
- Increase SNR_DB
- Increase SAMPLES_PER_ANGLE for better statistics
- Try different algorithms (e.g., Root-MUSIC instead of MUSIC)
- Use UNet-based methods for low SNR

## Customization

### Changing Colors
Edit the `colors` list in `plot_angle_sweep_results()`:
```python
colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']
```

### Adjusting Plot Style
Modify parameters in `plot_angle_sweep_results()`:
- `figsize=(12, 10)`: Change plot size
- `linewidth=2`: Change line thickness
- `markersize=4`: Change marker size
- `alpha=0.6`: Change transparency

### Adding More Algorithms
Add new cases in `estimate_doa_with_algorithm()` function.

## Related Scripts

- `source_estimation_validation.py` - Validates source number estimation
- `comprehensive_doa_evaluation.py` - Full DOA evaluation suite
- `controlled_angle_evaluation.py` - Fixed angle evaluation
- `analyze_eigenvalues.py` - Eigenvalue analysis

---

**Author**: GitHub Copilot
**Date**: October 11, 2025
**Version**: 1.0
