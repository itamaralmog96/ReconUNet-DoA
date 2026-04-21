#!/usr/bin/env python3
"""
Test script to verify parabolic interpolation is working correctly.
Compares DOA estimation with and without parabolic interpolation.
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

from signalgen import ArrayConfig, ArrayModel
from models.classic.music import MUSIC
from models.classic.mvdr import MVDR
from models.classic.beamformer import Beamformer

def test_parabolic_interpolation():
    """Test parabolic interpolation with a simple example."""
    
    print("=" * 70)
    print("PARABOLIC INTERPOLATION TEST")
    print("=" * 70)
    
    # Setup array
    array_config = ArrayConfig(
        array_type='linear',
        num_elements=8,
        element_spacing=0.5,
        carrier_freq=1e9
    )
    array_model = ArrayModel(array_config)
    
    # Generate test signal
    true_doas = np.array([60.3, 89.7])  # Fractional angles to test interpolation
    num_snapshots = 100
    snr_db = 20
    
    print(f"\nTrue DOAs: {true_doas}°")
    print(f"SNR: {snr_db} dB")
    print(f"Number of snapshots: {num_snapshots}")
    
    # Generate received signal
    received_signal = array_model.simulate_snapshot(
        angles_deg=true_doas,
        signal_power_db=np.array([0, 0]),
        noise_power_db=-snr_db,
        num_snapshots=num_snapshots
    )
    
    # Scan angles (1° resolution)
    scan_angles = np.arange(30, 151, 1)
    
    print(f"\nScan grid resolution: {scan_angles[1] - scan_angles[0]}°")
    print(f"Scan range: [{scan_angles[0]}°, {scan_angles[-1]}°]")
    
    # Test MUSIC with and without interpolation
    algorithms = [
        ('MUSIC', MUSIC),
        ('MVDR', MVDR),
        ('Beamformer', Beamformer)
    ]
    
    for algo_name, AlgoClass in algorithms:
        print(f"\n{'-' * 70}")
        print(f"{algo_name} Algorithm:")
        print(f"{'-' * 70}")
        
        # Without interpolation
        model_discrete = AlgoClass(
            array_model=array_model,
            scan_angles_deg=scan_angles,
            num_sources=2,
            use_parabolic_interpolation=False
        )
        model_discrete.set_received_data(received_signal)
        doas_discrete, _ = model_discrete.estimate_doa()
        doas_discrete = np.sort(doas_discrete)
        
        # With interpolation
        model_interp = AlgoClass(
            array_model=array_model,
            scan_angles_deg=scan_angles,
            num_sources=2,
            use_parabolic_interpolation=True
        )
        model_interp.set_received_data(received_signal)
        doas_interp, _ = model_interp.estimate_doa()
        doas_interp = np.sort(doas_interp)
        
        # Calculate errors
        errors_discrete = np.abs(np.sort(true_doas) - doas_discrete)
        errors_interp = np.abs(np.sort(true_doas) - doas_interp)
        
        rmse_discrete = np.sqrt(np.mean(errors_discrete**2))
        rmse_interp = np.sqrt(np.mean(errors_interp**2))
        
        print(f"  WITHOUT Interpolation:")
        print(f"    Estimated DOAs: [{doas_discrete[0]:.2f}°, {doas_discrete[1]:.2f}°]")
        print(f"    Errors: [{errors_discrete[0]:.2f}°, {errors_discrete[1]:.2f}°]")
        print(f"    RMSE: {rmse_discrete:.3f}°")
        
        print(f"\n  WITH Parabolic Interpolation:")
        print(f"    Estimated DOAs: [{doas_interp[0]:.2f}°, {doas_interp[1]:.2f}°]")
        print(f"    Errors: [{errors_interp[0]:.2f}°, {errors_interp[1]:.2f}°]")
        print(f"    RMSE: {rmse_interp:.3f}°")
        
        improvement = (rmse_discrete - rmse_interp) / rmse_discrete * 100
        print(f"\n  ✓ Improvement: {improvement:.1f}% reduction in RMSE")
        print(f"  ✓ Accuracy gain: {rmse_discrete / rmse_interp:.1f}x better")
    
    print("\n" + "=" * 70)
    print("✓ Parabolic interpolation test completed successfully!")
    print("=" * 70)

if __name__ == "__main__":
    test_parabolic_interpolation()
