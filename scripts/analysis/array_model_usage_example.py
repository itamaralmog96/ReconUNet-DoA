#!/usr/bin/env python3
"""
Usage example for the fixed array model functionality.

This example demonstrates the main ways to access and use array models
from datasets generated with save_array_metadata=True.
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from reconunet.data.dataset_generator import DOADataset

def demonstrate_array_model_usage(dataset_path):
    """Demonstrate various ways to use array models from a dataset."""
    
    print("=== Array Model Usage Example ===")
    print(f"Loading dataset: {dataset_path}")
    
    # Load dataset
    dataset = DOADataset(dataset_path)
    
    # Check if array models are available
    if not dataset.has_array_model_data():
        print("ERROR: This dataset does not contain array model data!")
        print("Generate a dataset with save_array_metadata=True to use this functionality.")
        return
    
    print(f"Dataset contains {len(dataset)} samples with array model data")
    
    # === Example 1: Basic Array Model Access ===
    print("\n=== Example 1: Basic Array Model Access ===")
    
    sample_idx = 0
    sample = dataset[sample_idx]
    
    print(f"Sample {sample_idx}:")
    print(f"  DOA: {sample['labels']['doas']}")
    print(f"  SNR: {sample['labels']['snr']} dB")
    print(f"  Array imperfections: {sample['labels']['array_imperfections']}")
    
    # Get array model
    array_model = dataset.get_array_model(sample_idx)
    print(f"  Array model type: {type(array_model).__name__}")
    
    # === Example 2: Array Model Information Without Full Reconstruction ===
    print("\n=== Example 2: Array Model Info (Lightweight) ===")
    
    array_info = dataset.get_array_model_info(sample_idx)
    config = array_info['array_config']
    
    print(f"Array configuration:")
    print(f"  Type: {config['array_type']}")
    print(f"  Elements: {config['num_elements']}")
    print(f"  Carrier frequency: {config['carrier_freq'] / 1e9:.2f} GHz")
    print(f"  Gain/phase errors: {config['enable_gain_phase_errors']}")
    print(f"  Mutual coupling: {config['enable_mutual_coupling']}")
    print(f"  Position error std: {config['position_error_std']}")
    print(f"  Reconstruction seed: {array_info['seed']}")
    
    # === Example 3: Using Array Models for DOA Estimation ===
    print("\n=== Example 3: DOA Estimation with Array Models ===")
    
    # Get a sample with multiple sources or specific conditions
    filtered_samples = dataset.filter(array_imperfections=True)
    if len(filtered_samples) > 0:
        sample = filtered_samples[0]
        array_model = filtered_samples.get_array_model(0)
        
        print(f"Using sample with imperfect array:")
        print(f"  DOAs: {sample['labels']['doas']}")
        
        # Get the received signal
        received_signal = sample['received_signal'].numpy()  # Shape: [num_elements, num_snapshots]
        print(f"  Received signal shape: {received_signal.shape}")
        
        # Demonstrate array model usage
        true_doas = sample['labels']['doas']
        if hasattr(true_doas, '__iter__'):
            doa_list = list(true_doas)
        else:
            doa_list = [float(true_doas)]
        
        # Get steering vectors for the true DOAs
        steering_matrix = array_model.steering_matrix(doa_list, nominal=False)  # With imperfections
        print(f"  Steering matrix shape: {steering_matrix.shape}")
        
        # Example: Simple beamforming
        # Note: This is just for demonstration - real DOA algorithms are more complex
        scan_angles = np.arange(0, 360, 1)  # 1-degree resolution
        beamformer_output = []
        
        for angle in scan_angles[:10]:  # Just first 10 angles for demo
            # Get steering vector for this angle
            a = array_model.steering_matrix([angle], nominal=False)
            
            # Simple conventional beamformer
            # P(θ) = a^H * R * a / (a^H * a)
            R = received_signal @ received_signal.conj().T / received_signal.shape[1]  # Sample covariance
            power = np.real(a.conj().T @ R @ a) / (a.conj().T @ a)
            beamformer_output.append(power[0, 0])
        
        print(f"  Beamformer output (first 10 angles): {beamformer_output[:5]}...")
        
    # === Example 4: Comparing Perfect vs Imperfect Arrays ===
    print("\n=== Example 4: Perfect vs Imperfect Array Comparison ===")
    
    # Find samples with same conditions but different array imperfections
    perfect_samples = dataset.filter(array_imperfections=False)
    imperfect_samples = dataset.filter(array_imperfections=True)
    
    if len(perfect_samples) > 0 and len(imperfect_samples) > 0:
        # Get array models
        perfect_model = perfect_samples.get_array_model(0)
        imperfect_model = imperfect_samples.get_array_model(0)
        
        # Compare steering vectors for same DOA
        test_doa = [90.0]  # Broadside
        
        perfect_steering = perfect_model.steering_matrix(test_doa, nominal=True)
        imperfect_steering = imperfect_model.steering_matrix(test_doa, nominal=False)
        
        print(f"Steering vector comparison for DOA = 90°:")
        print(f"  Perfect array:   {perfect_steering.flatten()}")
        print(f"  Imperfect array: {imperfect_steering.flatten()}")
        
        # Calculate difference
        steering_diff = np.abs(perfect_steering - imperfect_steering)
        max_diff = np.max(steering_diff)
        print(f"  Maximum difference: {max_diff:.4f}")
    
    # === Example 5: Batch Processing with Array Models ===
    print("\n=== Example 5: Batch Processing ===")
    
    print("Processing multiple samples...")
    
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        array_model = dataset.get_array_model(i)
        
        # Quick analysis
        doas = sample['labels']['doas']
        snr = sample['labels']['snr']
        imperfections = sample['labels']['array_imperfections']
        
        # Get array geometry information
        array_info = dataset.get_array_model_info(i)
        array_type = array_info['array_config']['array_type']
        num_elements = array_info['array_config']['num_elements']
        
        print(f"  Sample {i}: {array_type} array, {num_elements} elements, "
              f"DOA={doas}, SNR={snr}dB, imperfect={imperfections}")
    
    print("\n=== Usage Example Complete ===")


def example_custom_doa_algorithm(dataset_path, algorithm_name="MUSIC"):
    """Example of implementing a DOA algorithm using array models."""
    
    print(f"\n=== Custom DOA Algorithm Example: {algorithm_name} ===")
    
    dataset = DOADataset(dataset_path)
    
    if not dataset.has_array_model_data():
        print("Array models not available - skipping custom algorithm demo")
        return
    
    # Select a sample
    sample_idx = 0
    sample = dataset[sample_idx]
    array_model = dataset.get_array_model(sample_idx)
    
    # Get data
    received_signal = sample['received_signal'].numpy()
    true_doas = sample['labels']['doas']
    num_sources = len(true_doas) if hasattr(true_doas, '__len__') else 1
    
    print(f"Applying {algorithm_name} to sample {sample_idx}")
    print(f"  True DOAs: {true_doas}")
    print(f"  Number of sources: {num_sources}")
    print(f"  Signal shape: {received_signal.shape}")
    
    # Simple MUSIC-like algorithm demonstration
    # Note: This is simplified for demonstration
    
    # Compute sample covariance matrix
    R = received_signal @ received_signal.conj().T / received_signal.shape[1]
    
    # Eigendecomposition
    eigenvals, eigenvecs = np.linalg.eigh(R)
    
    # Sort in descending order
    idx = np.argsort(eigenvals)[::-1]
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]
    
    # Noise subspace (assuming we know number of sources)
    noise_subspace = eigenvecs[:, num_sources:]
    
    # Scan angles
    scan_angles = np.arange(0, 360, 2)  # 2-degree resolution
    music_spectrum = []
    
    for angle in scan_angles:
        # Get steering vector using the array model
        a = array_model.steering_matrix([angle], nominal=not sample['labels']['array_imperfections'])
        
        # MUSIC pseudo-spectrum
        numerator = a.conj().T @ a
        denominator = a.conj().T @ noise_subspace @ noise_subspace.conj().T @ a
        
        if np.abs(denominator) > 1e-10:
            spectrum_value = np.real(numerator / denominator)[0, 0]
        else:
            spectrum_value = np.inf
        
        music_spectrum.append(spectrum_value)
    
    # Find peaks
    music_spectrum = np.array(music_spectrum)
    peaks = []
    
    # Simple peak finding (you'd use a more sophisticated method in practice)
    for i in range(1, len(music_spectrum) - 1):
        if (music_spectrum[i] > music_spectrum[i-1] and 
            music_spectrum[i] > music_spectrum[i+1] and 
            music_spectrum[i] > np.max(music_spectrum) * 0.5):  # Threshold
            peaks.append(scan_angles[i])
    
    estimated_doas = sorted(peaks[:num_sources])  # Take top peaks
    
    print(f"  Estimated DOAs: {estimated_doas}")
    print(f"  MUSIC spectrum peak value: {np.max(music_spectrum):.2f}")
    
    # Calculate error
    if hasattr(true_doas, '__len__') and len(estimated_doas) == len(true_doas):
        errors = [abs(est - true) for est, true in zip(estimated_doas, sorted(true_doas))]
        print(f"  DOA errors: {[f'{e:.1f}°' for e in errors]}")
    
    return estimated_doas


if __name__ == "__main__":
    # This would typically be run with an existing dataset
    # For demonstration, you could first run test_array_model_fix.py to generate a test dataset
    
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    else:
        # Look for test dataset
        test_path = Path("test_output/triangular/array_model_test/array_model_test.h5")
        if test_path.exists():
            dataset_path = str(test_path)
        else:
            print("No dataset path provided and no test dataset found.")
            print("Usage: python array_model_usage_example.py <dataset_path>")
            print("Or first run: python test_array_model_fix.py")
            sys.exit(1)
    
    print(f"Using dataset: {dataset_path}")
    
    # Run demonstrations
    demonstrate_array_model_usage(dataset_path)
    example_custom_doa_algorithm(dataset_path)
    
    print("\n=== All Examples Complete ===") 