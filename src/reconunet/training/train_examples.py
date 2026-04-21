"""Example Training Scripts for Tri4Net Subspace Models

This script provides examples of how to train different subspace models using the
subspace_training.py module. Each example shows the appropriate parameters and
configurations for different model types.

Run this script to train example models, or use the individual functions as templates
for your own training scripts.

Examples:
- train_subspace_net_example: Train SubspaceNet with Root-MUSIC
- train_subspace_net_esprit_example: Train SubspaceNet with ESPRIT
- train_deep_root_music_example: Train DeepRootMUSIC
- train_deep_augmented_music_example: Train DeepAugmentedMUSIC
- train_deep_cnn_example: Train DeepCNN baseline
"""

import argparse
from pathlib import Path
from .subspace_training import train_subspace_model


def train_subspace_net_example(dataset_path: str, save_dir: str = "models/subspace_net"):
    """Example: Train SubspaceNet with Root-MUSIC."""
    print("=== Training SubspaceNet (Root-MUSIC) ===")
    
    results = train_subspace_model(
        dataset_path=dataset_path,
        model_type="subspace_net",
        
        # Model parameters
        tau=5,  # Number of autocorrelation lags
        diff_method="root_music",  # Use Root-MUSIC
        
        # Training parameters
        epochs=100,
        batch_size=32,
        learning_rate=1e-3,
        weight_decay=1e-4,
        
        # Learning rate scheduler
        step_size=30,
        gamma=0.1,
        
        # Save configuration
        save_dir=save_dir
    )
    
    return results


def train_subspace_net_esprit_example(dataset_path: str, save_dir: str = "models/subspace_net_esprit"):
    """Example: Train SubspaceNet with ESPRIT."""
    print("=== Training SubspaceNet (ESPRIT) ===")
    
    results = train_subspace_model(
        dataset_path=dataset_path,
        model_type="subspace_net_esprit",
        
        # Model parameters
        tau=5,  # Number of autocorrelation lags
        
        # Training parameters
        epochs=100,
        batch_size=32,
        learning_rate=1e-3,
        weight_decay=1e-4,
        
        # Learning rate scheduler  
        step_size=30,
        gamma=0.1,
        
        # Save configuration
        save_dir=save_dir
    )
    
    return results


def train_deep_root_music_example(dataset_path: str, save_dir: str = "models/deep_root_music"):
    """Example: Train DeepRootMUSIC."""
    print("=== Training DeepRootMUSIC ===")
    
    results = train_subspace_model(
        dataset_path=dataset_path,
        model_type="deep_root_music",
        
        # Model parameters
        tau=5,  # Number of autocorrelation lags
        
        # Training parameters
        epochs=100,
        batch_size=32,
        learning_rate=1e-3,
        weight_decay=1e-4,
        
        # Learning rate scheduler
        step_size=30,
        gamma=0.1,
        
        # Save configuration
        save_dir=save_dir
    )
    
    return results


def train_deep_augmented_music_example(dataset_path: str, save_dir: str = "models/deep_augmented_music"):
    """Example: Train DeepAugmentedMUSIC."""
    print("=== Training DeepAugmentedMUSIC ===")
    
    results = train_subspace_model(
        dataset_path=dataset_path,
        model_type="deep_augmented_music",
        
        # Training parameters
        epochs=150,  # Usually needs more epochs
        batch_size=16,  # Smaller batch size due to GRU memory requirements
        learning_rate=5e-4,  # Lower learning rate
        weight_decay=1e-4,
        
        # Learning rate scheduler
        step_size=50,
        gamma=0.5,
        
        # Save configuration
        save_dir=save_dir
    )
    
    return results


def train_deep_cnn_example(dataset_path: str, save_dir: str = "models/deep_cnn"):
    """Example: Train DeepCNN baseline."""
    print("=== Training DeepCNN ===")
    
    results = train_subspace_model(
        dataset_path=dataset_path,
        model_type="deep_cnn_subspace",
        
        # Training parameters
        epochs=80,
        batch_size=64,  # Can use larger batch size
        learning_rate=1e-3,
        weight_decay=1e-3,  # Higher regularization
        
        # Learning rate scheduler
        step_size=25,
        gamma=0.1,
        
        # Save configuration
        save_dir=save_dir
    )
    
    return results


def train_quick_test_example(dataset_path: str, save_dir: str = "models/quick_test"):
    """Quick test with minimal epochs for debugging."""
    print("=== Quick Test Training ===")
    
    results = train_subspace_model(
        dataset_path=dataset_path,
        model_type="subspace_net",
        
        # Model parameters
        tau=3,  # Reduced for speed
        diff_method="root_music",
        
        # Minimal training for testing
        epochs=5,
        batch_size=8,
        learning_rate=1e-3,
        
        # Save configuration
        save_dir=save_dir
    )
    
    return results


def train_all_models_example(dataset_path: str, base_save_dir: str = "models"):
    """Train all model types with the same dataset."""
    print("=== Training All Subspace Models ===")
    
    models_to_train = [
        ("subspace_net", "SubspaceNet (Root-MUSIC)"),
        ("subspace_net_esprit", "SubspaceNet (ESPRIT)"), 
        ("deep_root_music", "DeepRootMUSIC"),
        ("deep_augmented_music", "DeepAugmentedMUSIC"),
        ("deep_cnn_subspace", "DeepCNN")
    ]
    
    results = {}
    
    for model_type, model_name in models_to_train:
        print(f"\n{'='*50}")
        print(f"Training {model_name}")
        print(f"{'='*50}")
        
        try:
            if model_type == "subspace_net":
                result = train_subspace_net_example(dataset_path, f"{base_save_dir}/{model_type}")
            elif model_type == "subspace_net_esprit":
                result = train_subspace_net_esprit_example(dataset_path, f"{base_save_dir}/{model_type}")
            elif model_type == "deep_root_music":
                result = train_deep_root_music_example(dataset_path, f"{base_save_dir}/{model_type}")
            elif model_type == "deep_augmented_music":
                result = train_deep_augmented_music_example(dataset_path, f"{base_save_dir}/{model_type}")
            elif model_type == "deep_cnn_subspace":
                result = train_deep_cnn_example(dataset_path, f"{base_save_dir}/{model_type}")
            
            results[model_type] = result
            print(f"\n✅ {model_name} training completed successfully!")
            print(f"Best validation loss: {result['best_val_loss']:.6f}")
            print(f"Training time: {result['training_time']:.2f} seconds")
            
        except Exception as e:
            print(f"\n❌ Error training {model_name}: {e}")
            results[model_type] = {"error": str(e)}
    
    # Print summary
    print(f"\n{'='*50}")
    print("TRAINING SUMMARY")
    print(f"{'='*50}")
    
    for model_type, model_name in models_to_train:
        if model_type in results:
            result = results[model_type]
            if "error" in result:
                print(f"{model_name}: ❌ FAILED - {result['error']}")
            else:
                print(f"{model_name}: ✅ SUCCESS - Val Loss: {result['best_val_loss']:.6f}")
    
    return results


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Train Tri4Net subspace models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--dataset", "-d", 
        required=True,
        help="Path to Tri4Net dataset (.h5 file)"
    )
    
    parser.add_argument(
        "--model", "-m",
        choices=["subspace_net", "subspace_net_esprit", "deep_root_music", 
                "deep_augmented_music", "deep_cnn", "quick_test", "all"],
        default="subspace_net",
        help="Model type to train"
    )
    
    parser.add_argument(
        "--save-dir", "-s",
        default="models",
        help="Directory to save trained models"
    )
    
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=None,
        help="Number of training epochs (overrides example defaults)"
    )
    
    parser.add_argument(
        "--batch-size", "-b",
        type=int, 
        default=None,
        help="Batch size (overrides example defaults)"
    )
    
    parser.add_argument(
        "--learning-rate", "-lr",
        type=float,
        default=None,
        help="Learning rate (overrides example defaults)"
    )
    
    args = parser.parse_args()
    
    # Check if dataset exists
    if not Path(args.dataset).exists():
        print(f"❌ Dataset not found: {args.dataset}")
        print("Please generate a dataset first using:")
        print("  python -m reconunet.data.dataset_generator --config <config_file>")
        return
    
    print(f"Using dataset: {args.dataset}")
    print(f"Model type: {args.model}")
    print(f"Save directory: {args.save_dir}")
    
    # Train selected model
    if args.model == "subspace_net":
        results = train_subspace_net_example(args.dataset, args.save_dir)
    elif args.model == "subspace_net_esprit":
        results = train_subspace_net_esprit_example(args.dataset, args.save_dir)
    elif args.model == "deep_root_music":
        results = train_deep_root_music_example(args.dataset, args.save_dir)
    elif args.model == "deep_augmented_music":
        results = train_deep_augmented_music_example(args.dataset, args.save_dir)
    elif args.model == "deep_cnn":
        results = train_deep_cnn_example(args.dataset, args.save_dir)
    elif args.model == "quick_test":
        results = train_quick_test_example(args.dataset, args.save_dir)
    elif args.model == "all":
        results = train_all_models_example(args.dataset, args.save_dir)
        return
    
    print(f"\n🎉 Training completed!")
    print(f"Results: {results}")


if __name__ == "__main__":
    main() 