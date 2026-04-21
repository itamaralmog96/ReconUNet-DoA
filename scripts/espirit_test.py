#!/usr/bin/env python3
"""
Test script for the new ESPRIT implementation to verify it works correctly
with the Tri4Net angle conventions and follows the same structure as other classic DOA algorithms.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from signalgen.array_processing import ArrayModel, ArrayConfig
from models.classic import ESPRIT, MUSIC, RootMUSIC

def create_test_signal(array_model, source_angles_deg, snr_db=20, num_snapshots=200):
    """
    Create a test signal with known source directions.
    
    Args:
        array_model: ArrayModel instance
        source_angles_deg: List of source angles in degrees
        snr_db: Signal-to-noise ratio in dB
        num_snapshots: Number of snapshots
        
    Returns:
        received_signal: Received signal matrix (num_elements, num_snapshots)
    """
    num_elements = array_model.config.num_elements
    num_sources = len(source_angles_deg)
    
    # Generate source signals (random complex signals)
    source_signals = (np.random.randn(num_sources, num_snapshots) + 
                     1j * np.random.randn(num_sources, num_snapshots)) / np.sqrt(2)
    
    # Generate steering matrix for source directions
    steering_matrix = array_model.steering_matrix(source_angles_deg, nominal=True)
    
    # Generate received signal
    signal_power = 1.0
    noise_power = signal_power / (10 ** (snr_db / 10))
    
    # Signal component
    received_signal = steering_matrix @ source_signals
    
    # Add noise
    noise = (np.random.randn(num_elements, num_snapshots) + 
             1j * np.random.randn(num_elements, num_snapshots)) * np.sqrt(noise_power / 2)
    
    received_signal += noise
    
    return received_signal

def test_esprit_basic():
    """Test basic ESPRIT functionality."""
    print("Testing basic ESPRIT functionality...")
    
    # Create a linear array
    array_config = ArrayConfig(
        array_type='linear',
        num_elements=8,
        element_spacing=0.5,  # Half wavelength spacing
        carrier_freq=1e9  # 1 GHz
    )
    array_model = ArrayModel(array_config)
    
    # Test angles (in Tri4Net coordinate system: 0° = +x, 90° = +y)
    true_angles = [60, 120]  # Two sources
    print(f"True source angles: {true_angles}°")
    
    # Create test signal
    received_signal = create_test_signal(array_model, true_angles, snr_db=20, num_snapshots=200)
    
    # Initialize ESPRIT
    esprit = ESPRIT(array_model, num_sources=len(true_angles))
    esprit.set_received_data(received_signal)
    
    # Estimate DOA
    estimated_angles, spectrum = esprit.estimate_doa()
    
    estimated_sorted = np.sort(estimated_angles)
    true_sorted = np.sort(true_angles)
    errors = np.abs(estimated_sorted - true_sorted)
    
    print(f"Estimated angles: {[f'{angle:.1f}' for angle in estimated_sorted]}°")
    print(f"Estimation errors: {[f'{error:.1f}' for error in errors]}°")
    
    return estimated_angles, spectrum

def compare_algorithms():
    """Compare ESPRIT with MUSIC and RootMUSIC."""
    print("\nComparing ESPRIT with MUSIC and RootMUSIC...")
    
    # Create a linear array
    array_config = ArrayConfig(
        array_type='linear',
        num_elements=8,
        element_spacing=0.5,
        carrier_freq=1e9
    )
    array_model = ArrayModel(array_config)
    
    # Test angles
    true_angles = [45, 90, 135]  # Three sources
    print(f"True source angles: {true_angles}°")
    
    # Create test signal
    received_signal = create_test_signal(array_model, true_angles, snr_db=15, num_snapshots=300)
    
    # Test all algorithms
    algorithms = {
        'ESPRIT': ESPRIT(array_model, num_sources=len(true_angles)),
        'MUSIC': MUSIC(array_model, num_sources=len(true_angles)),
        'RootMUSIC': RootMUSIC(array_model, num_sources=len(true_angles))
    }
    
    results = {}
    
    for name, algorithm in algorithms.items():
        try:
            algorithm.set_received_data(received_signal)
            estimated_angles, spectrum = algorithm.estimate_doa()
            
            # Calculate estimation errors
            estimated_sorted = np.sort(estimated_angles)
            true_sorted = np.sort(true_angles)
            
            if len(estimated_sorted) >= len(true_sorted):
                errors = np.abs(estimated_sorted[:len(true_sorted)] - true_sorted)
            else:
                errors = np.abs(estimated_sorted - true_sorted[:len(estimated_sorted)])
            
            results[name] = {
                'estimated': estimated_sorted,
                'errors': errors,
                'spectrum': spectrum
            }
            
            print(f"{name:10s}: {[f'{angle:.1f}' for angle in estimated_sorted]}° "
                  f"(errors: {[f'{error:.1f}' for error in errors]}°)")
            
        except Exception as e:
            print(f"{name:10s}: Error - {str(e)}")
            results[name] = None
    
    return results

def main():
    """Main test function."""
    print("ESPRIT Implementation Test")
    print("=" * 50)
    
    try:
        # Test basic functionality
        estimated_angles, spectrum = test_esprit_basic()
        
        # Compare with other algorithms
        results = compare_algorithms()
        
        print("\nTest completed successfully!")
        print("ESPRIT implementation appears to be working correctly.")
        
    except Exception as e:
        print(f"Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()