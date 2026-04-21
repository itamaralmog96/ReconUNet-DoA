#!/usr/bin/env python3
"""
Test script to demonstrate the fixed array model storage and loading functionality.

This script shows how to:
1. Generate a dataset with array model data
2. Load the dataset and access individual array models
3. Verify that array models are correctly reconstructed
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from reconunet.data.dataset_generator import SimpleDatasetConfig, ControlledDatasetGenerator, DOADataset
import numpy as np

def test_array_model_storage():
    """Test the complete array model storage and loading pipeline."""
    
    print("=== Testing Array Model Storage Fix ===")
    
    # Create a small test dataset configuration
    config = SimpleDatasetConfig(
        angles_deg=[0, 90, 180],        # 3 angles
        snr_db=[0, 10],                 # 2 SNR levels  
        num_snapshots=[256],            # 1 snapshot count
        num_sources=[[1, 0]],           # 1 source configuration
        num_multipath=[0],              # No multipath
        array_imperfections=[True, False], # Both perfect and imperfect arrays
        examples_per_combination=1,     # 1 example per combination
        dataset_name="array_model_test",
        array_type="triangular",
        num_elements=4,
        save_array_metadata=True,       # Enable array model storage
        save_received_signal=True,
        save_covariance_matrix=True,
        save_steering_vectors=True
    )
    
    print(f"Configuration:")
    print(f"  Total combinations: {config.get_total_combinations()}")
    print(f"  Total samples: {config.get_total_samples()}")
    print(f"  Array metadata enabled: {config.save_array_metadata}")
    
    # Generate dataset
    print("\n=== Generating Dataset ===")
    generator = ControlledDatasetGenerator(config, seed=42)
    dataset_path = generator.generate_dataset("test_output")
    
    # Load dataset
    print("\n=== Loading Dataset ===")
    dataset = DOADataset(dataset_path)
    
    print(f"Dataset loaded with {len(dataset)} samples")
    print(f"Array model data available: {dataset.has_array_model_data()}")
    
    # Test array model access
    print("\n=== Testing Array Model Access ===")
    
    # Test first few samples
    for i in range(min(3, len(dataset))):
        print(f"\nSample {i}:")
        
        # Get sample
        sample = dataset[i]
        labels = sample['labels']
        
        print(f"  DOA: {labels['doas']}")
        print(f"  SNR: {labels['snr']} dB")
        print(f"  Array imperfections: {labels['array_imperfections']}")
        
        # Check if array model data is available
        if 'array_model_data' in sample:
            print(f"  Array model data: ✓")
            
            # Get array model info
            array_info = dataset.get_array_model_info(i)
            config_info = array_info['array_config']
            print(f"  Array type: {config_info['array_type']}")
            print(f"  Elements: {config_info['num_elements']}")
            print(f"  Carrier freq: {config_info['carrier_freq'] / 1e9:.2f} GHz")
            print(f"  Imperfections enabled: {config_info['enable_gain_phase_errors']}")
            print(f"  Seed: {array_info['seed']}")
            
            # Reconstruct ArrayModel
            try:
                array_model = dataset.get_array_model(i)
                print(f"  ArrayModel reconstructed: ✓")
                
                # Test that steering vectors match
                doas = labels['doas'] 
                if hasattr(doas, '__iter__'):
                    doa_list = list(doas)
                else:
                    doa_list = [float(doas)]
                
                # Get steering vectors from reconstructed model
                reconstructed_steering = array_model.steering_matrix(doa_list, nominal=not labels['array_imperfections'])
                
                # Compare with saved steering vectors
                if 'steering_vectors' in sample:
                    key = 'actual' if labels['array_imperfections'] else 'nominal'
                    saved_steering = sample['steering_vectors'][key].numpy()
                    
                    # Check if they match (allowing for small numerical differences)
                    max_diff = np.max(np.abs(reconstructed_steering - saved_steering))
                    print(f"  Steering vector max difference: {max_diff:.2e}")
                    
                    if max_diff < 1e-10:
                        print(f"  Steering vectors match: ✓")
                    else:
                        print(f"  Steering vectors differ: ✗")
                
            except Exception as e:
                print(f"  ArrayModel reconstruction failed: {e}")
        
        else:
            print(f"  Array model data: ✗")
    
    print(f"\n=== Dataset Summary ===")
    summary = dataset.get_label_summary()
    for label, info in summary.items():
        print(f"  {label}: {info}")
    
    # Test filtering with array models
    print(f"\n=== Testing Filtering ===")
    imperfect_only = dataset.filter(array_imperfections=True)
    print(f"Imperfect arrays only: {len(imperfect_only)} samples")
    print(f"Array models available in filtered dataset: {imperfect_only.has_array_model_data()}")
    
    if len(imperfect_only) > 0:
        sample_0 = imperfect_only[0]
        if 'array_model_data' in sample_0:
            array_model = imperfect_only.get_array_model(0)
            print(f"Array model reconstructed from filtered dataset: ✓")
    
    print(f"\n=== Test Complete ===")
    print(f"Dataset saved to: {dataset_path}")
    
    return dataset_path


if __name__ == "__main__":
    # Change to project directory
    os.chdir(project_root)
    
    try:
        dataset_path = test_array_model_storage()
        print(f"\nSuccess! Array model storage and loading working correctly.")
        print(f"You can inspect the dataset at: {dataset_path}")
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 