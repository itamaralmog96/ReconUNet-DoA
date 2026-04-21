# SubspaceNet ULA Evaluation Script

This script provides a simple evaluation framework for MUSIC and MVDR DOA estimation algorithms using the ULA (Uniform Linear Array) implementation from the SubspaceNet folder.

## Features

- Evaluates both MUSIC and MVDR algorithms
- Tests scenarios with 1, 2, and 3 sources
- Controllable angle separation parameter
- Monte Carlo evaluation with configurable parameters
- Automatic plotting and result saving

## Usage

### Basic Usage

```bash
cd Tri4Net/src/evaluation
python subspacenet_ula_evaluation.py
```

### Customizing Parameters

You can modify the evaluation parameters by editing the `main()` function in the script:

```python
eval_params = EvaluationParams(
    N=8,  # Number of sensors in the ULA
    T=200,  # Number of snapshots
    snr_range=[-10, -5, 0, 5, 10, 15],  # SNR values to test (dB)
    num_monte_carlo=50,  # Number of Monte Carlo runs per scenario
    angle_separations=[10, 15, 20, 30],  # Angle separations to test (degrees)
    scan_resolution=1.0  # Scan resolution in degrees
)
```

## Scenarios Tested

### Source Configurations
- **1 Source**: Single source at broadside (0°)
- **2 Sources**: Symmetric placement at ±(separation/2)°
- **3 Sources**: Placement at -separation°, 0°, +separation°

### Evaluation Metrics
- **RMSE**: Root Mean Square Error between true and estimated angles
- **Success Rate**: Percentage of trials with RMSE < 5°
- **Monte Carlo Statistics**: Mean and standard deviation over multiple runs

## Output

The script generates:

1. **Plots**: Comprehensive performance comparison plots saved as PNG
   - RMSE vs SNR for different number of sources
   - Success rate vs SNR
   - RMSE vs angle separation

2. **Results File**: JSON file with detailed numerical results
   - Individual trial results
   - Statistical summaries
   - Algorithm comparison data

## Requirements

- NumPy
- Matplotlib
- SubspaceNet implementation (should be in the correct relative path)

## Example Results

The evaluation will show performance differences between MUSIC and MVDR algorithms across:
- Different SNR conditions
- Various source configurations
- Multiple angle separations

## Troubleshooting

If you encounter import errors:
1. Ensure the SubspaceNet folder is in the correct location relative to this script
2. Check that all required SubspaceNet modules are present
3. Verify Python path configuration

## Customization

You can easily extend this script by:
- Adding new DOA algorithms
- Modifying test scenarios
- Changing evaluation metrics
- Adding new array configurations 