# Tri4Net ULA Evaluation Script

This script provides a comprehensive evaluation framework for MUSIC and MVDR DOA estimation algorithms using the Tri4Net ULA (Uniform Linear Array) implementation.

## Features

- Evaluates both MUSIC and MVDR algorithms using Tri4Net components
- Tests scenarios with 1, 2, and 3 sources
- Controllable angle separation parameter
- Monte Carlo evaluation with configurable parameters
- Automatic plotting and result saving
- Linear array (ULA) configuration with configurable parameters

## Key Differences from SubspaceNet Version

This script uses **Tri4Net** implementations instead of SubspaceNet:

### Signal Generation
- **Tri4Net**: Uses `SignalGenerator` and `SignalConfig` for proper signal generation
- **SubspaceNet**: Uses `Samples` class with built-in signal generation

### Array Processing
- **Tri4Net**: Uses `ArrayModel`, `ArrayConfig`, and `ReceivedSignal` with configurable hardware imperfections
- **SubspaceNet**: Uses `SystemModel` with basic array parameters

### DOA Algorithms
- **Tri4Net**: Object-oriented approach with `set_received_data()` and `estimate_doa()` methods, supports `num_sources` parameter for peak selection
- **SubspaceNet**: Functional approach with direct method calls

### Units and Conventions
- **Tri4Net**: Works directly in degrees
- **SubspaceNet**: Uses radians internally with degree conversion

## Usage

### Basic Usage

```bash
cd Tri4Net/src/evaluation
python tri4net_ula_evaluation.py
```

### Customizing Parameters

You can modify the evaluation parameters by editing the `main()` function in the script:

```python
eval_params = EvaluationParams(
    N=8,  # Number of sensors in the ULA
    T=1024,  # Number of snapshots
    snr_range=[-10, -5, 0, 5, 10, 15],  # SNR values to test (dB)
    num_monte_carlo=25,  # Number of Monte Carlo runs per scenario
    angle_separations=[3, 5, 10],  # Angle separations to test (degrees)
    scan_resolution=1.0,  # Scan resolution in degrees
    carrier_freq=2.45e9,  # Carrier frequency in Hz
    fs=1e6  # Sampling frequency in Hz
)
```

## Scenarios Tested

### Source Configurations
- **1 Source**: Single source at broadside (0°)
- **2 Sources**: Sources at 0° and +separation°
- **3 Sources**: Sources at -separation°, 0°, and +separation°

### Evaluation Metrics
- **RMSE**: Root Mean Square Error between true and estimated angles
- **Success Rate**: Percentage of trials with RMSE < 5°
- **Monte Carlo Statistics**: Mean and standard deviation over multiple runs

## Array Configuration

The script uses a linear array (ULA) with the following default settings:
- **Array Type**: Linear (ULA)
- **Element Spacing**: 0.5 wavelengths (half-wavelength spacing)
- **Hardware Imperfections**: Disabled for fair comparison
- **Mutual Coupling**: Disabled for fair comparison
- **Position Errors**: Disabled for fair comparison

## Enhanced DOA Algorithm Features

### Number of Sources Parameter

All DOA algorithms (MUSIC, MVDR, Beamformer) now support a `num_sources` parameter that controls peak selection:

- **MUSIC**: Uses `num_sources` for subspace estimation and selects the strongest peaks
- **MVDR**: Selects the `num_sources` strongest peaks from the spatial spectrum
- **Beamformer**: Selects the `num_sources` strongest peaks from the spatial spectrum

This ensures consistent behavior across all algorithms and improves estimation accuracy when the number of sources is known.

## Signal Parameters

- **Carrier Frequency**: 2.45 GHz (configurable)
- **Sampling Frequency**: 1 MHz (configurable)
- **Signal Type**: Complex narrowband signals generated using Tri4Net SignalGenerator
- **Bandwidth**: 10 kHz bandwidth per source (configurable)
- **Noise Type**: Gaussian broadband noise

## Output

The script generates:

1. **Plots**: Separate figures for each angle separation showing:
   - RMSE vs SNR for both algorithms
   - Different line styles for MUSIC (solid) and MVDR (dashed)
   - Different number of sources on the same plot

2. **Results File**: JSON file with detailed numerical results
   - Individual trial results
   - Statistical summaries
   - Algorithm comparison data

## Example Results

With default parameters, you'll get **3 separate figures** for angle separations [3°, 5°, 10°]:
- `tri4net_ula_evaluation_separation_3deg.png`
- `tri4net_ula_evaluation_separation_5deg.png`
- `tri4net_ula_evaluation_separation_10deg.png`

Each figure shows the performance comparison between MUSIC and MVDR algorithms across different SNR conditions and source configurations.

## Requirements

- NumPy
- Matplotlib
- Tri4Net signal generation and array processing modules
- Tri4Net classic DOA estimation models (MUSIC, MVDR)

## Comparison with SubspaceNet

Both evaluation scripts follow the **exact same structure** and produce **identical plot formats**, allowing for direct comparison between:
- SubspaceNet implementations
- Tri4Net implementations

The evaluation methodology, metrics, and plotting are identical - only the underlying DOA estimation implementations differ.

## Troubleshooting

If you encounter import errors:
1. Ensure you're running from the correct directory (`Tri4Net/src/evaluation/`)
2. Check that all required Tri4Net modules are present
3. Verify Python path configuration

## Customization

You can easily extend this script by:
- Adding new DOA algorithms from Tri4Net
- Modifying array configurations (circular, triangular, etc.)
- Adding hardware imperfections (gain/phase errors, mutual coupling)
- Changing signal parameters (bandwidth, frequency)
- Testing different array geometries 