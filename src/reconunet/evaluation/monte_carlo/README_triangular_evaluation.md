# Tri4Net Triangular Array Evaluation Script

This script provides a comprehensive evaluation framework for MUSIC and MVDR DOA estimation algorithms using the **Tri4Net Triangular Array** implementation.

## Features

- Evaluates both MUSIC and MVDR algorithms using Tri4Net triangular array geometry
- Tests scenarios with 1, 2, and 3 sources
- Controllable angle separation parameter
- Monte Carlo evaluation with configurable parameters
- Automatic plotting and result saving
- **Triangular array configuration** (4-element array)

## Key Differences from ULA Version

This script uses a **Triangular Array** instead of a Linear Array (ULA):

### Array Geometry
- **Triangular**: 4-element triangular array with equilateral triangle geometry
- **ULA**: 8-element uniform linear array

### Array Configuration
- **Array Type**: `'triangular'` (fixed geometry)
- **Number of Elements**: 4 (required for triangular arrays)
- **Element Spacing**: Controls the scale of the triangular geometry
- **Geometry**: Equilateral triangle with one central element

### Signal Parameters
- **Carrier Frequency**: 2.45 GHz (same as ULA)
- **Sampling Frequency**: 1 kHz (optimized for triangular array)
- **Signal Frequency**: 100 Hz (adjusted for lower sampling rate)

## Usage

### Basic Usage

```bash
cd Tri4Net/src/evaluation
python tri4net_triangular_evaluation.py
```

### Customizing Parameters

You can modify the evaluation parameters by editing the `main()` function in the script:

```python
eval_params = EvaluationParams(
    N=4,  # Number of sensors (fixed for triangular array)
    T=1024,  # Number of snapshots
    snr_range=np.arange(-10, 16, 1),  # SNR values to test (dB)
    num_monte_carlo=100,  # Number of Monte Carlo runs per scenario
    angle_separations=[3, 5, 10],  # Angle separations to test (degrees)
    scan_resolution=1.0,  # Scan resolution in degrees
    carrier_freq=2.45e9,  # Carrier frequency in Hz
    fs=1e3,  # Sampling frequency in Hz
    element_spacing=0.5  # Element spacing for triangular geometry (wavelengths)
)
```

## Scenarios Tested

### Source Configurations
- **1 Source**: Single source at broadside (90°)
- **2 Sources**: Sources at 90° and 90°+separation°
- **3 Sources**: Sources at 90°-separation°, 90°, and 90°+separation°

### Evaluation Metrics
- **RMSE**: Root Mean Square Error between true and estimated angles
- **Success Rate**: Percentage of trials with RMSE < 5°
- **Monte Carlo Statistics**: Mean and standard deviation over multiple runs

## Triangular Array Configuration

The script uses a 4-element triangular array with the following characteristics:
- **Array Type**: Triangular (equilateral triangle + center)
- **Number of Elements**: 4 (fixed)
- **Element Positions**: 
  - Center element at origin (0, 0)
  - Three outer elements forming equilateral triangle
- **Element Spacing**: Controls the radius of the triangular geometry
- **Hardware Imperfections**: Disabled for fair comparison

## Output

The script generates:

1. **Plots**: Separate figures for each angle separation showing:
   - RMSE vs SNR for both algorithms
   - Different line styles for MUSIC (solid) and MVDR (dashed)
   - Different number of sources on the same plot
   - Title indicates "Triangular Array" for clarity

2. **Results File**: JSON file with detailed numerical results
   - Individual trial results
   - Statistical summaries
   - Algorithm comparison data

## Example Results

With default parameters, you'll get **3 separate figures** for angle separations [3°, 5°, 10°]:
- `tri4net_triangular_evaluation_separation_3deg.png`
- `tri4net_triangular_evaluation_separation_5deg.png`
- `tri4net_triangular_evaluation_separation_10deg.png`

Each figure shows the performance comparison between MUSIC and MVDR algorithms on the triangular array.

## Advantages of Triangular Arrays

### Compared to Linear Arrays:
- **360° Coverage**: Can resolve sources from all directions
- **Compact Geometry**: Smaller physical footprint
- **Azimuth Resolution**: Better azimuth resolution capabilities
- **Unique Geometry**: Tri4Net-specific triangular configuration

### Potential Trade-offs:
- **Fewer Elements**: Only 4 elements vs 8 in ULA
- **Resolution Limits**: May have different resolution characteristics
- **Signal Processing**: Different array manifold and processing requirements

## Requirements

- NumPy
- Matplotlib
- Tri4Net signal generation and array processing modules
- Tri4Net classic DOA estimation models (MUSIC, MVDR)

## Comparison with ULA Evaluation

Both evaluation scripts follow the **exact same structure** and produce **identical plot formats**, allowing for direct comparison between:
- ULA (8-element linear array) performance
- Triangular Array (4-element triangular) performance

The evaluation methodology, metrics, and plotting are identical - only the underlying array geometry differs.

## Performance Expectations

### Expected Behavior:
- **1 Source**: Both algorithms should perform well
- **2-3 Sources**: Performance depends on angle separation and SNR
- **Close Sources**: More challenging for triangular array due to fewer elements
- **Well-Separated Sources**: Should perform comparably to ULA

### Key Factors:
- **Array Aperture**: Smaller than ULA, may affect resolution
- **Element Count**: Fewer elements may limit performance
- **Geometry**: Triangular shape provides different spatial sampling

## Troubleshooting

If you encounter issues:
1. Ensure you're running from the correct directory (`Tri4Net/src/evaluation/`)
2. Check that all required Tri4Net modules are present
3. Verify the triangular array configuration is correctly set up
4. Note that `N=4` is enforced for triangular arrays

## Customization

You can easily extend this script by:
- Modifying triangular array parameters (element_spacing)
- Adding other Tri4Net array geometries (circular, custom)
- Testing different signal parameters optimized for triangular arrays
- Adding hardware imperfections specific to triangular arrays
- Comparing with other 4-element array configurations 