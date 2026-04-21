#!/usr/bin/env python3
"""
Configuration-based Dataset Generation Script

Usage:
    python scripts/generate_dataset.py --config configs/dataset_configs/basic_dataset.yaml
    python scripts/generate_dataset.py --config configs/dataset_configs/comprehensive_dataset.yaml
"""

import argparse
import yaml
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from reconunet.data.controlled_dataset_generator import ControlledDatasetGenerator, ControlledDatasetConfig
from reconunet.signalgen import ArrayConfig


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_generator_from_config(config):
    """Create ControlledDatasetGenerator from configuration."""
    
    # Create ArrayConfig
    array_cfg = config['array_config']
    array_config = ArrayConfig(
        array_type=array_cfg['array_type'],
        num_elements=int(array_cfg['num_elements']),
        carrier_freq=float(array_cfg['carrier_freq']),
        enable_gain_phase_errors=array_cfg.get('enable_gain_phase_errors', True),
        enable_mutual_coupling=array_cfg.get('enable_mutual_coupling', True),
        position_error_std=float(array_cfg.get('position_error_std', 0.01))
    )
    
    # Process base_signal_config to ensure proper types
    base_signal_config = config['base_signal_config'].copy()
    base_signal_config['fs'] = float(base_signal_config['fs'])
    base_signal_config['T'] = int(base_signal_config['T'])
    base_signal_config['frequencies'] = [float(f) for f in base_signal_config['frequencies']]
    
    # Process sir_sweep to ensure proper types
    if config['sir_sweep']['enabled']:
        sir_sweep = config['sir_sweep'].copy()
        sir_sweep['interferer_freq_offsets'] = [float(f) for f in sir_sweep['interferer_freq_offsets']]
        config['sir_sweep'] = sir_sweep
    
    # Create ControlledDatasetConfig
    dataset_config = ControlledDatasetConfig(
        dataset_name=config['dataset_name'],
        create_splits=config['create_splits'],
        array_config=array_config,
        base_signal_config=base_signal_config,
        angle_sweep=config['angle_sweep'],
        snr_sweep=config['snr_sweep'],
        multipath_sweep=config['multipath_sweep'],
        sir_sweep=config['sir_sweep'],
        use_nominal_array=config['use_nominal_array'],
        regenerate_array_per_sample=config['regenerate_array_per_sample'],
        examples_per_combination=config['examples_per_combination'],
        train_ratio=config['train_ratio'],
        val_ratio=config['val_ratio'],
        test_ratio=config['test_ratio'],
        save_raw_signals=config['save_raw_signals'],
        save_covariance_matrices=config['save_covariance_matrices'],
        save_steering_vectors=config['save_steering_vectors'],
        save_metadata=config['save_metadata']
    )
    
    # Create generator
    generator = ControlledDatasetGenerator(dataset_config)
    
    return generator


def setup_parameter_sweeps(generator, config):
    """Parameter sweeps are now handled by the config object."""
    pass  # No longer needed - config handles everything


def main():
    parser = argparse.ArgumentParser(description='Generate controlled DOA dataset from configuration')
    parser.add_argument('--config', required=True, help='Path to YAML configuration file')
    parser.add_argument('--output-dir', default='Data/datasets', help='Output directory for datasets')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be generated without actually generating')
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Create generator
    print("Creating dataset generator...")
    generator = create_generator_from_config(config)
    
    # Setup parameter sweeps
    print("Setting up parameter sweeps...")
    setup_parameter_sweeps(generator, config)
    
    # Show generation plan
    dataset_size_info = generator.config.calculate_dataset_size()
    
    print(f"\n📊 Generation Plan:")
    print(f"   Dataset Name: {config['dataset_name']}")
    print(f"   Total Parameter Combinations: {dataset_size_info['total_combinations']:,}")
    print(f"   Examples per Combination: {config['examples_per_combination']}")
    print(f"   Total Samples: {dataset_size_info['total_samples']:,}")
    print(f"   Create Splits: {config['create_splits']}")
    
    if config['create_splits']:
        print(f"   Train/Val/Test: {dataset_size_info['train_samples']:,}/{dataset_size_info['val_samples']:,}/{dataset_size_info['test_samples']:,}")
    
    if args.dry_run:
        print("\n🔍 Dry run complete - no dataset generated")
        return
    
    # Generate dataset
    print(f"\n🚀 Starting dataset generation...")
    
    # Pass the base directory, not the full path (generator will add dataset_name)
    generator.generate_dataset(base_output_dir=args.output_dir)
    
    print(f"✅ Dataset generation complete!")
    print(f"📁 Output location: {Path(args.output_dir) / config['dataset_name']}")


if __name__ == "__main__":
    main() 