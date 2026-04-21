# Comprehensive DOA Algorithm Evaluation

This script provides a complete evaluation framework for DOA algorithms, following the structure from `testing_Unet_Denoising.ipynb`.

## Quick Start

1. **Open the script**: `comprehensive_doa_evaluation.py`
2. **Modify the configuration** at the top of the file (lines 30-80)
3. **Run**: `python comprehensive_doa_evaluation.py`

## Configuration

All configuration is done by editing the script directly (no command line arguments needed):

### Dataset Configuration

```python
DATASET_CONFIG = {
    'dataset_name': 'my_evaluation',
    'examples_per_combination': 20,
    
    # Array Configuration
    'array_type': 'linear',
    'num_elements': 8,
    'carrier_freq': 2.45e9,
    'element_spacing': 0.5,
    
    # Signal Configuration
    'sampling_freq': 1e6,
    'num_snapshots': 512,
    'bandwidth': 1e6,
    
    # Evaluation Parameters
    'snr_db': list(range(-20, 21, 5)),  # -20, -15, -10, ..., 20
    'num_sources': [[1, 0], [2, 0], [3, 0]],
    'num_multipath': [0, 1],
    'array_imperfections': [False],
    
    # ... more options
}
```

### Evaluation Configuration

```python
EVAL_CONFIG = {
    'max_samples_per_snr': 50,
    'scan_angles': np.arange(30, 151, 1),
    'output_dir': 'evaluation_results',
    'unet_model_path': None,  # or 'path/to/trained_model.pth'
}
```

### Use Existing Dataset

```python
USE_EXISTING_DATASET = True
EXISTING_DATASET_PATH = 'Data/datasets/linear/my_dataset/my_dataset.h5'
```

## Examples

### Example 1: Quick Test

```python
DATASET_CONFIG = {
    'dataset_name': 'quick_test',
    'examples_per_combination': 5,
    'array_type': 'linear',
    'num_elements': 8,
    'carrier_freq': 2.45e9,
    'sampling_freq': 1e6,
    'num_snapshots': 512,
    'snr_db': [-10, 0, 10],
    'num_sources': [[1, 0], [2, 0]],
    'num_multipath': [0],
    'array_imperfections': [False],
    'save_received_signal': True,
    'save_covariance_matrix': True,
    'save_autocorrelation_matrix': True,
    'autocorr_tau': 8,
}

EVAL_CONFIG = {
    'max_samples_per_snr': 10,
    'scan_angles': np.arange(30, 151, 1),
    'output_dir': 'quick_test_results',
    'unet_model_path': None,
}

USE_EXISTING_DATASET = False
```

### Example 2: Comprehensive Evaluation with Trained UNet

```python
DATASET_CONFIG = {
    'dataset_name': 'comprehensive_eval',
    'examples_per_combination': 50,
    'snr_db': list(range(-20, 21, 2)),  # Every 2 dB
    'num_sources': [[1, 0], [2, 0], [3, 0], [4, 0]],
    'num_multipath': [0, 1, 2],
    # ... other parameters
}

EVAL_CONFIG = {
    'max_samples_per_snr': 100,
    'unet_model_path': 'models/trained_unet.pth',  # Use trained model
    'output_dir': 'comprehensive_results',
}

USE_EXISTING_DATASET = False
```

### Example 3: Use Existing Dataset

```python
USE_EXISTING_DATASET = True
EXISTING_DATASET_PATH = 'Data/datasets/linear/UNet_Testing_dataset/UNet_Testing_dataset.h5'

EVAL_CONFIG = {
    'max_samples_per_snr': 50,
    'unet_model_path': 'models/my_trained_model.pth',
    'output_dir': 'results_with_existing_dataset',
}
```

## What It Does

1. **Generates/Loads Dataset**:
   - If `USE_EXISTING_DATASET = False`: generates new dataset with specified parameters
   - If `USE_EXISTING_DATASET = True`: loads existing dataset from `EXISTING_DATASET_PATH`

2. **Evaluates Each Sample**:
   - Applies all classic DOA algorithms:
     - MUSIC
     - MVDR
     - Beamformer
     - ESPRIT
     - RootMUSIC
     - UnitaryESPRIT
   - Applies UNet denoising + MUSIC (if UNet model available)

3. **Organizes Results by SNR**:
   - Groups results by SNR level
   - Calculates mean RMSE per algorithm per SNR
   - Tracks success rates

4. **Saves Results**:
   - `evaluation_results/results.json` - Detailed RMSE values
   - `evaluation_results/rmse_vs_snr.png` - Plot of RMSE vs SNR
   - Console summary table

## Output Format

### Console Output
```
📊 Results Summary
====================================================================================================
SNR (dB)  MUSIC          MVDR           Beamformer     ESPRIT         RootMUSIC      UnitaryESPRIT  UNet+MUSIC     
----------------------------------------------------------------------------------------------------
-20.0     15.23          18.45          22.10          16.80          15.50          17.20          12.80          
-10.0     8.45           11.20          14.50          9.10           8.20           9.50           6.90           
0.0       3.20           4.80           7.20           3.50           3.10           3.80           2.50           
10.0      1.50           2.20           3.80           1.70           1.45           1.90           1.20           
20.0      0.80           1.10           2.10           0.90           0.75           1.00           0.60           
```

### Plot
![RMSE vs SNR plot showing all algorithms' performance]

### JSON Results
```json
{
  "-20.0": {
    "MUSIC": [15.2, 16.1, 14.8, ...],
    "MVDR": [18.4, 19.2, 17.9, ...],
    "UNet_MUSIC": [12.5, 13.1, 12.2, ...]
  },
  ...
}
```

## Tips

1. **Start with quick test**: Use small `examples_per_combination` (5-10) and limited SNR range to verify everything works

2. **Adjust max_samples_per_snr**: Controls how many samples are evaluated per SNR level (balances speed vs accuracy)

3. **Use trained UNet model**: Set `unet_model_path` to see the benefit of trained denoising

4. **Save datasets**: Generated datasets are saved and can be reused by setting `USE_EXISTING_DATASET = True`

5. **Monitor progress**: The script shows progress bars and updates for each SNR level

## Troubleshooting

- **No samples generated**: Check `DATASET_CONFIG` parameters, especially `snr_db` and `num_sources`
- **UNet not available**: EVDUNet models not found - check import paths
- **Out of memory**: Reduce `examples_per_combination` or `max_samples_per_snr`
- **Slow evaluation**: Normal for large datasets; use smaller `max_samples_per_snr` for faster results



