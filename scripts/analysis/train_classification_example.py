#!/usr/bin/env python3
"""Example script for training scenario classification models.

This demonstrates how to use the classification training pipeline
with different models and datasets.
"""

from pathlib import Path
from reconunet.training.classification_training import ClassificationTrainer, ClassificationTrainingParams


def train_scenario_classification(
    dataset_path: str,
    model_type: str = "scenario_cnn",
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    save_dir: str = "models/classification"
):
    """Train a scenario classification model.
    
    Args:
        dataset_path: Path to the HDF5 dataset
        model_type: Model architecture ('scenario_cnn', 'scenario_resnet', 
                   'scenario_transformer', 'scenario_hybrid')
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate for optimization
        save_dir: Directory to save models and results
    """
    
    # Create training parameters
    params = ClassificationTrainingParams(
        dataset_path=dataset_path,
        model_type=model_type,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        save_dir=save_dir,
        
        # Optimization settings
        optimizer="adamw",
        scheduler="cosine",
        weight_decay=1e-4,
        
        # Regularization
        dropout_rate=0.5,
        label_smoothing=0.1,
        
        # Early stopping
        early_stopping_patience=15,
        
        # Logging
        use_tensorboard=True,
        log_interval=50
    )
    
    # Create trainer and run training
    trainer = ClassificationTrainer(params)
    results = trainer.train(dataset_path)
    
    return results


if __name__ == "__main__":
    """Example usage for different models."""
    
    # Path to your Tri4Net dataset
    dataset_path = "Data/datasets/linear/ula_dataset_90_all_scenarios/ula_dataset_90_all_scenarios.h5"
    
    if not Path(dataset_path).exists():
        print(f"❌ Dataset not found: {dataset_path}")
        print("Please update the dataset_path to point to your Tri4Net dataset.")
        exit(1)
    
    print("🚀 Starting scenario classification training examples...")
    
    # Example 1: Multi-Scale CNN (recommended for starting)
    print("\n" + "="*80)
    print("📊 Training Multi-Scale CNN (Fast & Effective)")
    print("="*80)
    
    results_cnn = train_scenario_classification(
        dataset_path=dataset_path,
        model_type="scenario_cnn",
        epochs=30,
        batch_size=64,
        learning_rate=2e-3,
        save_dir="models/classification/scenario_cnn"
    )
    
    print(f"✅ CNN Training completed!")
    print(f"   Best validation accuracy: {results_cnn['best_val_acc']:.2f}%")
    print(f"   Training time: {results_cnn['training_time']:.1f}s")
    
    # Example 2: Covariance ResNet (domain-specific)
    print("\n" + "="*80)
    print("🏗️ Training Covariance ResNet (Domain-Specific)")
    print("="*80)
    
    results_resnet = train_scenario_classification(
        dataset_path=dataset_path,
        model_type="scenario_resnet",
        epochs=40,
        batch_size=32,
        learning_rate=1e-3,
        save_dir="models/classification/scenario_resnet"
    )
    
    print(f"✅ ResNet Training completed!")
    print(f"   Best validation accuracy: {results_resnet['best_val_acc']:.2f}%")
    print(f"   Training time: {results_resnet['training_time']:.1f}s")
    
    # Example 3: Hybrid CNN-RNN (balanced approach)
    print("\n" + "="*80)
    print("⚡ Training Hybrid CNN-RNN (Balanced)")
    print("="*80)
    
    results_hybrid = train_scenario_classification(
        dataset_path=dataset_path,
        model_type="scenario_hybrid",
        epochs=35,
        batch_size=48,
        learning_rate=1.5e-3,
        save_dir="models/classification/scenario_hybrid"
    )
    
    print(f"✅ Hybrid Training completed!")
    print(f"   Best validation accuracy: {results_hybrid['best_val_acc']:.2f}%")
    print(f"   Training time: {results_hybrid['training_time']:.1f}s")
    
    # Summary comparison
    print("\n" + "="*80)
    print("📈 RESULTS SUMMARY")
    print("="*80)
    
    models_results = [
        ("Multi-Scale CNN", results_cnn),
        ("Covariance ResNet", results_resnet),
        ("Hybrid CNN-RNN", results_hybrid)
    ]
    
    print(f"{'Model':<20} {'Val Acc (%)':<12} {'Training Time (s)':<18} {'Parameters':<12}")
    print("-" * 70)
    
    for name, results in models_results:
        acc = results['best_val_acc']
        time_s = results['training_time']
        params = results['model_params']
        print(f"{name:<20} {acc:<12.2f} {time_s:<18.1f} {params:<12,}")
    
    print("\n🎯 Training completed for all models!")
    print("📁 Results saved in models/classification/ directory")
    print("📊 Check TensorBoard logs for detailed training curves")
    print("🔍 Confusion matrices saved as PNG files")
    
    # Optional: Train Transformer if you have time and resources
    train_transformer = input("\n🤖 Train Transformer model? (slower but potentially better) [y/N]: ")
    
    if train_transformer.lower() == 'y':
        print("\n" + "="*80)
        print("🤖 Training Signal Transformer (Advanced)")
        print("="*80)
        
        results_transformer = train_scenario_classification(
            dataset_path=dataset_path,
            model_type="scenario_transformer",
            epochs=50,
            batch_size=16,  # Smaller batch for transformer
            learning_rate=5e-4,  # Lower LR for transformer
            save_dir="models/classification/scenario_transformer"
        )
        
        print(f"✅ Transformer Training completed!")
        print(f"   Best validation accuracy: {results_transformer['best_val_acc']:.2f}%")
        print(f"   Training time: {results_transformer['training_time']:.1f}s") 