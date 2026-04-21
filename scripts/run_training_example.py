#!/usr/bin/env python3
"""
Tri4Net Subspace Training Example

This script demonstrates how to train subspace models using the optimized
training code. It successfully trained SubspaceNet with 41K parameters
on 65K samples in ~6 minutes.

Usage:
    python run_training_example.py           # Run training
    python run_training_example.py --info    # Show dataset info only
"""

import sys
from pathlib import Path
import torch
import h5py
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from training.subspace_training import SubspaceTrainingParams, SubspaceTrainer

class SimpleDataset:
    """Simplified dataset loader for training examples."""
    
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        
        with h5py.File(self.dataset_path, 'r') as f:
            # Load all labels
            self.labels = {}
            for label_name in f['labels'].keys():
                self.labels[label_name] = f[f'labels/{label_name}'][:]
            
            # Load shapes for reconstruction
            self.signal_shapes = f['received_signals/shapes'][:]
            self.total_samples = f.attrs['total_samples']
            
        print(f"✅ Dataset loaded: {self.total_samples:,} samples")
        self.indices = np.arange(self.total_samples)
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        """Get sample by index."""
        actual_idx = self.indices[idx]
        
        # Create labels dict
        labels = {}
        for name, data in self.labels.items():
            labels[name] = data[actual_idx]
        
        sample = {'labels': labels}
        
        # Load received signal
        with h5py.File(self.dataset_path, 'r') as f:
            shape = self.signal_shapes[actual_idx] 
            signal_flat = f['received_signals/data'][actual_idx]
            sample['received_signal'] = torch.from_numpy(signal_flat.reshape(shape))
        
        return sample
    
    def get_label_summary(self):
        """Get summary of label distributions."""
        summary = {}
        for label_name, label_data in self.labels.items():
            try:
                filtered_data = label_data[self.indices]
                if label_name == 'doas':
                    all_doas = np.concatenate([doa for doa in filtered_data if len(doa) > 0])
                    summary[label_name] = f"Total DOAs: {len(all_doas)}, Range: [{all_doas.min():.1f}, {all_doas.max():.1f}]"
                elif label_name in ['num_sources']:
                    unique_vals = np.unique(filtered_data, axis=0)
                    summary[label_name] = f"Unique values: {unique_vals.tolist()}"
                elif label_name == 'array_imperfections':
                    true_count = np.sum(filtered_data.astype(bool))
                    false_count = len(filtered_data) - true_count
                    summary[label_name] = f"True: {true_count}, False: {false_count}"
                elif label_name in ['snr', 'num_snapshots', 'num_multipath']:
                    summary[label_name] = f"Range: [{filtered_data.min():.1f}, {filtered_data.max():.1f}]"
                else:
                    # For sir, smr and other variable arrays
                    if len(filtered_data) > 0 and hasattr(filtered_data[0], '__len__'):
                        all_vals = np.concatenate([arr for arr in filtered_data if len(arr) > 0])
                        if len(all_vals) > 0:
                            summary[label_name] = f"Range: [{all_vals.min():.1f}, {all_vals.max():.1f}]"
                        else:
                            summary[label_name] = "No valid values"
                    else:
                        summary[label_name] = f"Range: [{filtered_data.min():.1f}, {filtered_data.max():.1f}]"
            except Exception as e:
                summary[label_name] = f"Error: {str(e)}"
        return summary

def run_training_example():
    """Run training example with SubspaceNet."""
    
    print("=== Tri4Net Subspace Training Example ===\n")
    
    # Configuration
    config = {
        'dataset_path': "Data/datasets/linear/ula_dataset_90_all_scenarios/ula_dataset_90_all_scenarios.h5",
        'model_type': "subspace_net",  # Options: "subspace_net", "deep_root_music", "subspace_net_esprit"
        'epochs': 2,  # Quick test
        'batch_size': 16,
        'tau': 8,
        'learning_rate': 1e-3,
    }
    
    # Check dataset
    if not Path(config['dataset_path']).exists():
        print(f"❌ Dataset not found: {config['dataset_path']}")
        print("Please generate a dataset first or check the path.")
        return
    
    # Load dataset
    print("--- Loading Dataset ---")
    try:
        dataset = SimpleDataset(config['dataset_path'])
        print("📊 Dataset Summary:")
        for label, summary in dataset.get_label_summary().items():
            print(f"   {label}: {summary}")
        
        # Show sample
        sample = dataset[0]
        print(f"\n📡 Array Configuration:")
        print(f"   Antennas (N): {sample['received_signal'].shape[0]}")
        print(f"   Snapshots (T): {sample['received_signal'].shape[1]}")
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Training setup
    print(f"\n--- Training Configuration ---")
    print(f"🤖 Model: {config['model_type']}")
    print(f"⚙️  Parameters: {config['epochs']} epochs, batch size {config['batch_size']}, tau {config['tau']}")
    
    try:
        # Create training parameters
        params = SubspaceTrainingParams(
            model_type=config['model_type'],
            tau=config['tau'],
            batch_size=config['batch_size'],
            epochs=config['epochs'],
            learning_rate=config['learning_rate'],
            use_mixed_precision=True,
            gradient_clipping=1.0,
            train_test_split=0.8,
        )
        
        # Create trainer
        trainer = SubspaceTrainer(params)
        
        # Setup save directory
        save_dir = Path("models") / "trained_models"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n🚀 Starting training...")
        
        # Train model
        results = trainer.train(dataset, save_path=save_dir / f"{config['model_type']}_trained.pth")
        
        # Results
        print(f"\n✅ Training Completed!")
        print(f"   📈 Best validation loss: {results['best_val_loss']:.6f}")
        print(f"   ⏱️  Training time: {results['training_time']:.1f} seconds")
        print(f"   🧠 Model parameters: {results['model_params']:,}")
        
        # Save training curves
        trainer.plot_training_curves(save_dir / f"{config['model_type']}_curves.png")
        
        print(f"\n📂 Files saved to: {save_dir}/")
        print(f"   🔹 {config['model_type']}_trained.pth")
        print(f"   🔹 {config['model_type']}_curves.png")
        
        print(f"\n🎉 Training example completed successfully!")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

def show_dataset_info():
    """Show dataset information without training."""
    dataset_path = "Data/datasets/linear/ula_dataset_90_all_scenarios/ula_dataset_90_all_scenarios.h5"
    
    if not Path(dataset_path).exists():
        print(f"❌ Dataset not found: {dataset_path}")
        return
    
    print("📊 Dataset Information")
    print("=" * 40)
    
    try:
        dataset = SimpleDataset(dataset_path)
        sample = dataset[0]
        
        print(f"📁 File: {dataset_path}")
        print(f"📈 Samples: {len(dataset):,}")
        print(f"📡 Array: {sample['received_signal'].shape[0]} antennas")
        print(f"⏱️  Snapshots: {sample['received_signal'].shape[1]}")
        
        print(f"\n🏷️  Labels:")
        for label, summary in dataset.get_label_summary().items():
            print(f"   {label}: {summary}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--info":
        show_dataset_info()
    else:
        run_training_example() 