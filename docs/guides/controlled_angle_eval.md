# Controlled Angle DOA Evaluation

This document describes the new controlled angle evaluation system for DOA estimation algorithms.

## Overview

The system consists of two main components:

1. **`generate_test_dataset()` method** in `ControlledDatasetGenerator` (src/data/dataset_generator.py)
2. **`controlled_angle_evaluation.py`** - Evaluation script using the new method

## Purpose

Provides precise control over angle configurations for comprehensive performance analysis of:
- UNet + MUSIC denoising
- Classic DOA algorithms (MUSIC, MVDR, Beamformer, ESPRIT, Root-MUSIC)
- CRLB theoretical bounds

Primary focus: **RMSE vs SNR comparison** with controlled angle separation and source placement.

## Three Operational Modes

### Mode 1: Fixed Reference Angles with Fixed Delta
**Use case:** Test performance at specific challenging configurations

```python
EVALUATION_MODE = 1
REFERENCE_ANGLES = [45.0, 90.0, 135.0]  # Test at these specific angles
ANGLE_DELTA = 10.0  # Sources separated by exactly 10°
SAMPLES_PER_SNR = 100  # 100 identical samples per SNR for averaging
```

**Example:**
- Reference 90°, 3 sources, delta 10° → Sources at [90°, 100°, 110°]
- Reference 45°, 3 sources, delta 10° → Sources at [45°, 55°, 65°]
- All 100 samples per SNR have **identical** angle configuration

**Benefits:**
- Test specific critical angles (endfire, broadside, boundaries)
- Reproducible and deterministic
- Easy to interpret results

### Mode 2: Random Reference with Fixed Delta
**Use case:** Test performance across many scenarios while maintaining separation

```python
EVALUATION_MODE = 2
REFERENCE_ANGLES = 'random'  # Random reference for each sample
ANGLE_DELTA = 10.0  # Maintain 10° separation
ANGLE_RANGE = (30.0, 150.0)  # Sample references from this range
SAMPLES_PER_SNR = 100
```

**Example:**
- Sample 1: Reference 73.4° → Sources at [73.4°, 83.4°, 93.4°]
- Sample 2: Reference 112.8° → Sources at [112.8°, 122.8°, 132.8°]
- Sample 3: Reference 41.2° → Sources at [41.2°, 51.2°, 61.2°]
- Each sample has **different** reference but **same** delta

**Benefits:**
- Average performance across all possible reference angles
- Maintains controlled separation for fair comparison
- More realistic evaluation

### Mode 3: Random Angles with Minimum Separation
**Use case:** Test performance with fully random configurations (like current behavior)

```python
EVALUATION_MODE = 3
REFERENCE_ANGLES = None
ANGLE_DELTA = None  # Uses min_angle_separation_deg from config
ANGLE_RANGE = (30.0, 150.0)
SAMPLES_PER_SNR = 100
```

**Example:**
- Sample 1: Sources at [42.7°, 87.3°, 135.2°] (random with min separation)
- Sample 2: Sources at [68.1°, 95.4°, 129.8°] (random with min separation)
- Fully **random** angle placement with minimum separation constraint

**Benefits:**
- Most general evaluation
- Tests algorithm robustness across all scenarios
- Good for overall performance characterization

## Boundary Handling

The system automatically handles angle boundaries:

```python
ANGLE_RANGE = (30.0, 150.0)
reference = 140.0
delta = 10.0
total_sources = 3

# Naive: [140°, 150°, 160°] ❌ 160° exceeds boundary!
# Adjusted: [130°, 140°, 150°] ✅ All within range
```

**Algorithm:**
1. Try sequential placement forward from reference
2. If boundary exceeded, fill backwards from reference
3. If still can't fit, raise error (delta too large for range)

## Configuration Guide

### Basic Setup

```python
# === MODE SELECTION ===
EVALUATION_MODE = 2  # Choose 1, 2, or 3

# === ANGLE CONFIGURATION ===
REFERENCE_ANGLES = [90.0]  # For Mode 1
ANGLE_RANGE = (30.0, 150.0)  # For Mode 2 & 3
ANGLE_DELTA = 10.0  # For Mode 1 & 2 (None for Mode 3)

# === SNR CONFIGURATION ===
SNR_DB = list(range(-20, 21, 5))  # Test SNR range
SAMPLES_PER_SNR = 100  # Samples for each SNR level

# === SOURCE CONFIGURATION ===
NUM_SOURCES = [1, 2]  # [main_sources, interference_sources]
```

### Source Configuration Examples

```python
# Single source (no interference)
NUM_SOURCES = [1, 0]

# One main + 2 interference
NUM_SOURCES = [1, 2]
SIR_DB = 0.0  # Signal-to-Interference Ratio

# Two main sources (no interference)
NUM_SOURCES = [2, 0]

# Three main + 1 interference
NUM_SOURCES = [3, 1]
```

### Algorithm Selection

```python
# === EVALUATION ALGORITHMS ===
EVALUATE_UNET_MUSIC = True   # UNet denoising + MUSIC
EVALUATE_MUSIC = True         # Classic MUSIC
EVALUATE_MVDR = True          # MVDR (Capon)
EVALUATE_BEAMFORMER = True    # Conventional Beamformer
EVALUATE_ESPRIT = True        # ESPRIT
EVALUATE_ROOT_MUSIC = True    # Root-MUSIC (linear arrays only)
EVALUATE_CRLB = True          # CRLB theoretical bounds
```

### Advanced Options

```python
# === SIGNAL PARAMETERS ===
NUM_SNAPSHOTS = 512  # Time snapshots
NUM_MULTIPATH = 0    # Multipath components (0 = none)
ARRAY_IMPERFECTIONS = False  # Array errors (gain/phase/coupling)

# === ARRAY CONFIGURATION ===
ARRAY_TYPE = "linear"  # 'linear', 'triangular', 'circular', 'cross'
NUM_ELEMENTS = 8       # Number of array elements
CARRIER_FREQ = 2.45e9  # Hz
SAMPLING_FREQ = 1e6    # Hz

# === OUTPUT CONTROL ===
SAVE_AUTOCORRELATION_MATRIX = True  # Required for UNet
SAVE_CLEAN_COVARIANCE_MATRIX = True  # Required for UNet targets
```

## Usage Examples

### Example 1: Test UNet vs MUSIC at Close Separation

```python
EVALUATION_MODE = 1
REFERENCE_ANGLES = [90.0]  # Broadside
ANGLE_DELTA = 5.0  # Very close sources!
NUM_SOURCES = [2, 0]  # Two main sources
SAMPLES_PER_SNR = 200  # Good averaging
SNR_DB = list(range(-10, 21, 2))  # Fine SNR grid
```

**Question answered:** How does UNet perform when sources are only 5° apart?

### Example 2: Performance Degradation vs Separation

Run multiple times with different deltas:

```python
# Run 1: ANGLE_DELTA = 5.0
# Run 2: ANGLE_DELTA = 10.0
# Run 3: ANGLE_DELTA = 15.0
# Run 4: ANGLE_DELTA = 20.0

EVALUATION_MODE = 2  # Random reference
NUM_SOURCES = [1, 2]  # 1 main + 2 interference
SAMPLES_PER_SNR = 100
```

**Question answered:** How does performance improve as sources become more separated?

### Example 3: General Performance Characterization

```python
EVALUATION_MODE = 3  # Random angles
NUM_SOURCES = [1, 1]  # 1 main + 1 interference
SAMPLES_PER_SNR = 500  # Large sample size
SNR_DB = list(range(-20, 21, 5))
```

**Question answered:** What is the overall RMSE vs SNR curve across all scenarios?

### Example 4: Boundary Effects

```python
EVALUATION_MODE = 1
REFERENCE_ANGLES = [30.0, 60.0, 90.0, 120.0, 150.0]  # Include boundaries
ANGLE_DELTA = 10.0
ANGLE_RANGE = (30.0, 150.0)
NUM_SOURCES = [3, 0]
```

**Question answered:** Do algorithms perform differently near array boundaries?

## Output Files

The script generates:

1. **Dataset:** `Data/datasets/linear/controlled_eval_modeX/controlled_eval_modeX.h5`
2. **Results JSON:** `Tri4Net/src/evaluation/controlled_angle_evaluation/results_modeX.json`
3. **RMSE Plot:** `Tri4Net/src/evaluation/controlled_angle_evaluation/rmse_vs_snr_modeX.png`

## Running the Script

```bash
# 1. Configure parameters in controlled_angle_evaluation.py
# 2. Run the script
python controlled_angle_evaluation.py

# Output:
# 🚀 Controlled Angle DOA Evaluation
# ===================================================
# 📋 Configuration:
#    Mode: 2 - Random Reference Angles
#    Angle Delta: 10.0°
#    SNR Range: [-20, -15, ..., 15, 20]
#    Samples per SNR: 100
#    Sources: 1 main + 2 interference
# ...
# ✅ Evaluation completed! Results saved to ...
```

## Integration with Existing Scripts

The new `generate_test_dataset()` method is **fully compatible** with existing code:

```python
from data.dataset_generator import SimpleDatasetConfig, ControlledDatasetGenerator

# Create config (same as before)
config = SimpleDatasetConfig(...)

# Create generator
generator = ControlledDatasetGenerator(config, seed=42)

# Option 1: Use new test dataset generation
dataset_path = generator.generate_test_dataset(
    reference_angles='random',
    angle_delta=10.0,
    ...
)

# Option 2: Use original systematic generation (unchanged)
dataset_path = generator.generate_dataset(output_dir="Data/datasets")

# Both return standard HDF5 datasets compatible with DOADataset class
```

## Comparison with Existing Scripts

| Feature | `crlb_evaluation.py` | `comprehensive_doa_evaluation.py` | `controlled_angle_evaluation.py` |
|---------|---------------------|----------------------------------|--------------------------------|
| Purpose | CRLB computation | General comparison | Controlled angle analysis |
| Angle Control | Random/grid | Random/grid | **Precise control (3 modes)** |
| Samples per SNR | Fixed per combination | Variable | **Configurable per SNR** |
| Delta Control | No | Min separation only | **Exact delta or min separation** |
| CRLB | ✅ Yes | ❌ No | ✅ Yes |
| UNet | ❌ No | ✅ Yes | ✅ Yes |
| Classic Algorithms | ✅ MUSIC only | ✅ All | ✅ All |

## Tips and Best Practices

1. **Start with Mode 1** for quick verification at specific angles
2. **Use Mode 2** for statistically robust evaluation with controlled separation
3. **Use Mode 3** for general performance characterization
4. **Large SAMPLES_PER_SNR** (100-500) gives smooth RMSE curves
5. **Small ANGLE_DELTA** (3-5°) tests algorithm resolution limits
6. **Multiple reference angles** in Mode 1 helps identify angle-dependent behavior
7. **Enable CRLB** to verify theoretical bounds are not violated
8. **Save datasets** for later reuse with different algorithm parameters

## Troubleshooting

### "Cannot fit X sources with Y° separation"
- **Cause:** Delta too large for angle range
- **Solution:** Reduce ANGLE_DELTA or increase ANGLE_RANGE

### "Reference angle X° with Y sources cannot fit in range"
- **Cause:** Reference too close to boundary
- **Solution:** Choose reference further from boundaries or reduce sources/delta

### UNet model not found
- **Cause:** UNET_MODEL_PATH incorrect
- **Solution:** Update path or set EVALUATE_UNET_MUSIC = False

### Slow generation
- **Cause:** Large SAMPLES_PER_SNR or many SNR levels
- **Solution:** Reduce samples or use fewer SNR points initially

## Advanced: Programmatic Usage

```python
from data.dataset_generator import SimpleDatasetConfig, ControlledDatasetGenerator

# Create base config
config = SimpleDatasetConfig(...)
generator = ControlledDatasetGenerator(config, seed=42)

# Sweep over multiple deltas
deltas = [3, 5, 10, 15, 20, 30]
results = {}

for delta in deltas:
    print(f"Testing delta = {delta}°")
    
    # Generate dataset
    dataset_path = generator.generate_test_dataset(
        reference_angles='random',
        angle_delta=delta,
        snr_db=[-20, -10, 0, 10, 20],
        samples_per_snr=100,
        num_sources=[1, 2],
        ...
    )
    
    # Load and evaluate
    dataset = DOADataset(dataset_path)
    results[delta] = evaluate_dataset(dataset, ...)

# Plot delta vs RMSE
plot_delta_comparison(results)
```

## Questions?

For issues or questions about the controlled angle evaluation system:
1. Check configuration validation errors
2. Review mode descriptions above
3. Verify boundary constraints
4. Check dataset generation logs

Enjoy precise control over your DOA evaluation! 🎯
