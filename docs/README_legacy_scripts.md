# Model Analysis Script

## Overview

This script provides comprehensive analysis of the trained EVD UNet Denoising Model, including:
- Model architecture and parameters
- Training configuration and hyperparameters
- Model size and complexity metrics
- Layer-wise parameter breakdown
- Forward pass testing

## Setup

### Activate Environment

```bash
conda activate DOA_env
```

## Usage

### Analyze default model

```bash
cd "/Users/itamaralmog/Documents/Masters/Second Year/All_DOA_nets/Tri4Net"
export KMP_DUPLICATE_LIB_OK=TRUE  # Required for OpenMP compatibility
python scripts/analyze_model_performance.py
```

### Analyze specific model file

```bash
python scripts/analyze_model_performance.py --model your_model_file.pth
```

## Output

The script provides:

1. **Checkpoint Contents**: All keys available in the checkpoint file
2. **Training Metadata**: Epoch, losses, learning rate, configurations
3. **Model Parameters Summary**: Total, trainable, and non-trainable parameters
4. **Layer Breakdown**: Top 20 layers by parameter count
5. **Model Structure**: Distribution of different module types
6. **I/O Specifications**: Input and output shapes
7. **Forward Pass Test**: Verification that the model works correctly

## Example Output

```
================================================================================
EVD UNet Denoising Model - Comprehensive Analysis
================================================================================

📂 Model Path: .../evd_unet_denoising_model_20250929_015132.pth
📦 File Size: 3.81 MB

Model Parameters Summary:
  Total Parameters:        336,003
  Trainable Parameters:    336,003
  Model Size in Memory:    1.29 MB

Configuration: M=8, tau=8
Status: ✅ Ready for inference
```

## Model Details

- **Model Type**: EVDCovarianceReconstructionUNet
- **Input**: Autocorrelation tensor `[batch_size, tau=8, 2M=16, M=8]`
- **Output**: 
  - Eigenvalues: `[batch_size, 8]` (real)
  - Eigenvectors: `[batch_size, 8, 8]` (complex)
  - Reconstructed covariance: `[batch_size, 8, 8]` (complex)

## Notes

- The script requires PyTorch and other dependencies from `requirements.txt`
- Make sure you're in the correct conda environment (`DOA_env`)
- The model file should be in the `notebooks/` directory by default



