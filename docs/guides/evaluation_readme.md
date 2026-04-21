# Comprehensive DOA Algorithm Evaluation

This directory contains a comprehensive evaluation suite for Direction of Arrival (DOA) estimation algorithms, including both classic algorithms and UNet-based denoising approaches.

## Overview

The evaluation suite tests the following algorithms:
- **Classic DOA Algorithms**: MUSIC, MVDR, Beamformer, ESPRIT, RootMUSIC, UnitaryESPRIT
- **UNet + MUSIC**: UNet denoising followed by MUSIC algorithm (as demonstrated in `testing_Unet_Denoising.ipynb`)

## Features

- **Systematic Dataset Generation**: Creates controlled datasets with varying SNR, number of sources, and multipath conditions
- **Comprehensive Algorithm Testing**: Evaluates all classic DOA algorithms on the same dataset
- **UNet Integration**: Applies UNet denoising to covariance matrices before MUSIC processing
- **Performance Metrics**: Calculates RMSE, success rates, and timing information
- **Flexible Configuration**: YAML-based configuration for easy parameter adjustment
- **Multiple Output Formats**: JSON, CSV, and visualization support

## Quick Start

### 1. Quick Test (Recommended for first run)
```bash
python run_evaluation.py --quick
```

This runs a quick evaluation with:
- 50 samples total
- SNR range: [-10, 0, 10] dB
- Sources: [1, 2]
- Multipath: [0, 1]

### 2. Full Evaluation
```bash
python run_evaluation.py --full
```

This runs the full evaluation using `evaluation_config.yaml`:
- 300 samples total
- SNR range: [-20, -10, 0, 10, 20] dB
- Sources: [1, 2, 3]
- Multipath: [0, 1, 2]

### 3. Custom Configuration
```bash
python run_evaluation.py --config my_config.yaml
```

### 4. With Trained UNet Model
```bash
python run_evaluation.py --unet-model path/to/trained_model.pth
```

## Direct Script Usage

You can also run the evaluation script directly:

```bash
# Quick test
python comprehensive_doa_evaluation.py --quick-test

# With configuration file
python comprehensive_doa_evaluation.py --config evaluation_config.yaml

# With UNet model
python comprehensive_doa_evaluation.py --config evaluation_config.yaml --unet-model model.pth
```

## Configuration

The evaluation is configured through YAML files. See `evaluation_config.yaml` for the full configuration options:

```yaml
# Array Configuration
array_type: "linear"
num_elements: 8
carrier_freq: 2.45e9

# Evaluation Parameters
snr_range: [-20, -10, 0, 10, 20]
num_sources_range: [1, 2, 3]
multipath_range: [0, 1, 2]

# Dataset Generation
num_samples: 300
min_angle_separation: 10.0
angle_range: [30, 150]
```

## Output

The evaluation generates:

1. **Detailed Results** (`detailed_results.json`): Per-sample results for all algorithms
2. **Summary** (`summary.json`): Statistical summary by SNR and algorithm
3. **Console Output**: Formatted table showing mean RMSE and success rates

### Example Output

```
📊 DOA Algorithm Performance Summary
========================================
SNR (dB)   MUSIC          MVDR           Beamformer     UNet+MUSIC    
-20.0      15.23°(85%)    18.45°(78%)    22.10°(65%)    12.80°(90%)   
-10.0      8.45°(95%)     11.20°(88%)    14.50°(82%)    6.90°(98%)    
0.0        3.20°(100%)    4.80°(95%)     7.20°(90%)     2.50°(100%)   
10.0       1.50°(100%)    2.20°(100%)    3.80°(98%)     1.20°(100%)   
20.0       0.80°(100%)    1.10°(100%)    2.10°(100%)    0.60°(100%)   
```

## Algorithm Details

### Classic Algorithms
- **MUSIC**: Multiple Signal Classification
- **MVDR**: Minimum Variance Distortionless Response
- **Beamformer**: Conventional beamforming
- **ESPRIT**: Estimation of Signal Parameters via Rotational Invariance Techniques
- **RootMUSIC**: Root-MUSIC algorithm
- **UnitaryESPRIT**: Unitary ESPRIT algorithm

### UNet Integration
The UNet denoising follows the approach from `testing_Unet_Denoising.ipynb`:
1. Convert received signal to autocorrelation tensor format
2. Apply UNet denoising to reconstruct clean covariance matrix
3. Use MUSIC algorithm on the denoised covariance matrix

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- SciPy
- Matplotlib
- PyYAML
- tqdm

## File Structure

```
Tri4Net/
├── comprehensive_doa_evaluation.py    # Main evaluation script
├── run_evaluation.py                  # Quick runner script
├── evaluation_config.yaml             # Default configuration
├── README_evaluation.md               # This file
├── src/
│   ├── models/classic/               # Classic DOA algorithms
│   ├── models/deep_learning/         # UNet and other DL models
│   ├── signalgen/                    # Signal generation
│   └── data/                         # Dataset utilities
└── evaluation_results/               # Output directory (created)
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you're running from the Tri4Net directory
2. **CUDA Errors**: The script automatically detects and uses CPU if CUDA is not available
3. **Memory Issues**: Reduce `num_samples` in the configuration for large evaluations
4. **Algorithm Failures**: Individual algorithm failures are caught and reported; evaluation continues

### Performance Tips

- Use `--quick-test` for initial testing
- Reduce `num_samples` for faster evaluation
- Limit `snr_range` and `num_sources_range` for focused testing
- Use trained UNet models for better denoising performance

## Extending the Evaluation

To add new algorithms:
1. Implement the algorithm following the `BaseDOAModel` interface
2. Add initialization in `DOAEvaluationSuite.initialize_algorithms()`
3. The evaluation framework will automatically include it

To modify dataset generation:
1. Edit the `DatasetGenerator.generate_sample()` method
2. Update configuration parameters in YAML files
3. Adjust the parameter ranges in `generate_dataset()`



