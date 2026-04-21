"""Fast training script for SubspaceNet models with performance optimizations.

This script provides significant speed improvements through:
- Larger batch sizes
- Mixed precision training (automatically disabled for complex operations)
- Optimized data loading
- Vectorized operations
- Simplified models

Note: Mixed precision (FP16) is automatically disabled for subspace models that use
complex number operations, as PyTorch doesn't support ComplexHalf on CUDA.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import time
from pathlib import Path
from typing import Dict, Any

try:
    from ..data.data_utils import load_tri4net_dataset
    from ..criterions import RMSPELoss
    from ..models.deep_learning.subspace_models_optimized import FastSubspaceNet, SimpleCNN
except ImportError:
    # Fallback imports
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    try:
        from data.data_utils import load_tri4net_dataset
        from criterions import RMSPELoss
        from models.deep_learning.subspace_models_optimized import FastSubspaceNet, SimpleCNN
    except ImportError:
        # Direct imports for testing
        from reconunet.data.data_utils import load_tri4net_dataset
        from reconunet.criterions import RMSPELoss
        from reconunet.models.deep_learning.subspace_models_optimized import FastSubspaceNet, SimpleCNN


def create_autocorrelation_tensor_fast(X: torch.Tensor, tau: int) -> torch.Tensor:
    """Fast autocorrelation tensor creation."""
    N, T = X.shape
    
    # Pre-allocate tensor
    Rx_tau = torch.zeros((tau, 2*N, N), dtype=torch.float32, device=X.device)
    
    # Vectorized autocorrelation computation
    for t in range(min(tau, T-1)):
        Rx = (X[:, t:] @ X[:, :-t if t > 0 else None].conj().T) / (T - t)
        Rx_tau[t, :N, :] = Rx.real
        Rx_tau[t, N:, :] = Rx.imag
    
    return Rx_tau


def fast_collate_fn(batch):
    """Optimized collate function."""
    signals, doas = zip(*batch)
    
    # Stack signals directly (all same size in dataset)
    signals_tensor = torch.stack(signals, dim=0)
    
    # Handle variable-length DoAs
    max_doas = max(len(doa) for doa in doas)
    doas_padded = []
    
    for doa in doas:
        padded = torch.full((max_doas,), -999.0)  # Padding value
        padded[:len(doa)] = doa
        doas_padded.append(padded)
    
    doas_tensor = torch.stack(doas_padded, dim=0)
    
    return signals_tensor, doas_tensor


class FastTrainer:
    """High-performance trainer with optimizations."""
    
    def __init__(
        self,
        model_type: str = "fast_subspace_net",
        batch_size: int = 128,  # Much larger batch size
        learning_rate: float = 1e-3,
        device: str = "auto"
    ):
        self.model_type = model_type
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        # Device setup
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        print(f"Using device: {self.device}")
        
        # Enable mixed precision if on CUDA (but disable for complex operations)
        # Complex number operations in subspace models don't support FP16
        self.use_mixed_precision = (
            self.device.type == "cuda" and 
            model_type not in ["fast_subspace_net", "subspace_net", "deep_root_music", "subspace_net_esprit"]
        )
        if self.use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
            print("Mixed precision training enabled")
        else:
            print(f"Mixed precision disabled for {model_type} (complex operations not compatible with FP16)")
    
    def create_model(self, N: int, M: int, tau: int = 5) -> nn.Module:
        """Create optimized model."""
        if self.model_type == "fast_subspace_net":
            model = FastSubspaceNet(tau=tau, M=M, N=N)
        elif self.model_type == "simple_cnn":
            model = SimpleCNN(tau=tau, M=M, N=N)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
            
        return model.to(self.device)
    
    def prepare_batch_data(self, batch):
        """Fast batch data preparation."""
        signals, doas = batch
        signals = signals.to(self.device, non_blocking=True)
        doas = doas.to(self.device, non_blocking=True)
        
        # Convert to radians
        doas_rad = doas * torch.pi / 180.0
        
        # Create input tensors efficiently
        if self.model_type in ["fast_subspace_net", "simple_cnn"]:
            batch_size = signals.shape[0]
            input_tensors = []
            
            for i in range(batch_size):
                X = signals[i]  # [N, T]
                Rx_tau = create_autocorrelation_tensor_fast(X, tau=5)
                input_tensors.append(Rx_tau)
            
            input_tensor = torch.stack(input_tensors, dim=0)  # [B, tau, 2N, N]
            return input_tensor, doas_rad
        else:
            return signals, doas_rad
    
    def train_fast(
        self,
        dataset_path: str,
        epochs: int = 50,  # Fewer epochs for testing
        validation_split: float = 0.2,
        num_workers: int = 0  # Default to 0 for compatibility
    ) -> Dict[str, Any]:
        """Fast training with all optimizations."""
        
        print("Loading dataset...")
        dataset = load_tri4net_dataset(dataset_path)
        
        # Get dataset info
        sample_data = dataset[0]
        N = sample_data[0].shape[0]  # Number of array elements
        M = len([x for x in sample_data[1] if x != -999.0])  # Number of sources
        
        print(f"Dataset: {len(dataset)} samples")
        print(f"Array elements (N): {N}")
        print(f"Sources (M): {M}")
        
        # Create optimized data loaders
        train_size = int(len(dataset) * (1 - validation_split))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        # Reduce workers for compatibility (especially on macOS)
        safe_num_workers = min(num_workers, 2) if num_workers > 0 else 0
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=safe_num_workers,
            pin_memory=True if self.device.type == "cuda" else False,
            collate_fn=fast_collate_fn,
            persistent_workers=safe_num_workers > 0
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=safe_num_workers,
            pin_memory=True if self.device.type == "cuda" else False,
            collate_fn=fast_collate_fn,
            persistent_workers=safe_num_workers > 0
        )
        
        # Create model and optimizer
        print("Creating model...")
        model = self.create_model(N=N, M=M)
        
        # Use AdamW with weight decay
        optimizer = AdamW(model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        criterion = RMSPELoss()
        
        # Training loop
        train_losses = []
        val_losses = []
        
        print(f"Starting training for {epochs} epochs...")
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Training
            model.train()
            train_loss = 0.0
            train_batches = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch in pbar:
                try:
                    # Prepare data
                    input_tensor, target_tensor = self.prepare_batch_data(batch)
                    
                    optimizer.zero_grad()
                    
                    # Forward pass with mixed precision
                    if self.use_mixed_precision:
                        with torch.cuda.amp.autocast():
                            output = model(input_tensor)
                            if isinstance(output, tuple):
                                doa_pred = output[0]
                            else:
                                doa_pred = output
                            
                            # Compute loss only for valid targets
                            valid_mask = target_tensor != -999.0 * torch.pi / 180.0
                            batch_losses = []
                            
                            for i in range(target_tensor.shape[0]):
                                valid_targets = target_tensor[i][valid_mask[i]]
                                if len(valid_targets) > 0:
                                    valid_preds = doa_pred[i][:len(valid_targets)]
                                    loss = criterion(valid_preds.unsqueeze(0), valid_targets.unsqueeze(0))
                                    batch_losses.append(loss)
                            
                            if batch_losses:
                                loss = torch.stack(batch_losses).mean()
                            else:
                                continue
                        
                        # Backward pass with scaling
                        self.scaler.scale(loss).backward()
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        # Regular forward pass
                        output = model(input_tensor)
                        if isinstance(output, tuple):
                            doa_pred = output[0]
                        else:
                            doa_pred = output
                        
                        loss = criterion(doa_pred, target_tensor)
                        loss.backward()
                        optimizer.step()
                    
                    train_loss += loss.item()
                    train_batches += 1
                    
                    pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                    
                except Exception as e:
                    print(f"Error in batch: {e}")
                    continue
            
            avg_train_loss = train_loss / max(train_batches, 1)
            train_losses.append(avg_train_loss)
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    try:
                        input_tensor, target_tensor = self.prepare_batch_data(batch)
                        
                        if self.use_mixed_precision:
                            with torch.cuda.amp.autocast():
                                output = model(input_tensor)
                                if isinstance(output, tuple):
                                    doa_pred = output[0]
                                else:
                                    doa_pred = output
                                loss = criterion(doa_pred, target_tensor)
                        else:
                            output = model(input_tensor)
                            if isinstance(output, tuple):
                                doa_pred = output[0]
                            else:
                                doa_pred = output
                            loss = criterion(doa_pred, target_tensor)
                        
                        val_loss += loss.item()
                        val_batches += 1
                        
                    except Exception as e:
                        continue
            
            avg_val_loss = val_loss / max(val_batches, 1)
            val_losses.append(avg_val_loss)
            
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}, time={epoch_time:.1f}s")
        
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.1f}s ({total_time/epochs:.1f}s per epoch)")
        
        return {
            'model': model,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'total_time': total_time,
            'time_per_epoch': total_time / epochs
        }


def train_fast_subspace_net(dataset_path: str, epochs: int = 20, batch_size: int = 128, learning_rate: float = 1e-3, **kwargs):
    """Quick function to train optimized SubspaceNet."""
    trainer = FastTrainer(model_type="fast_subspace_net", batch_size=batch_size, learning_rate=learning_rate)
    return trainer.train_fast(dataset_path, epochs=epochs, **kwargs)


def train_simple_cnn(dataset_path: str, epochs: int = 20, batch_size: int = 256, learning_rate: float = 1e-3, **kwargs):
    """Quick function to train simple CNN baseline."""
    trainer = FastTrainer(model_type="simple_cnn", batch_size=batch_size, learning_rate=learning_rate)
    return trainer.train_fast(dataset_path, epochs=epochs, **kwargs)


def main(
    dataset_path: str,
    model_type: str = "fast_subspace_net",
    batch_size: int = 128,
    epochs: int = 20,
    learning_rate: float = 1e-3,
    validation_split: float = 0.2,
    save_results: bool = True
):
    """Main API function for fast training.
    
    Args:
        dataset_path: Path to the dataset file
        model_type: Type of model ("fast_subspace_net" or "simple_cnn")
        batch_size: Training batch size
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        validation_split: Fraction of data to use for validation
        save_results: Whether to save training results
        
    Returns:
        Training results dictionary
    """
    print(f"🚀 Starting fast training...")
    print(f"Dataset: {dataset_path}")
    print(f"Model: {model_type}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {learning_rate}")
    
    # Create trainer
    trainer = FastTrainer(
        model_type=model_type,
        batch_size=batch_size,
        learning_rate=learning_rate
    )
    
    # Train model
    results = trainer.train_fast(
        dataset_path=dataset_path,
        epochs=epochs,
        validation_split=validation_split
    )
    
    # Print results
    print(f"\n✅ Training completed successfully!")
    print(f"⏱️  Time per epoch: {results['time_per_epoch']:.1f}s")
    print(f"📉 Final validation loss: {results['val_losses'][-1]:.4f} radians")
    print(f"🔄 Total training time: {results['total_time']:.1f}s")
    
    # Optionally save results
    if save_results:
        try:
            import json
            from datetime import datetime
            
            save_path = f"training_results_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # Prepare results for JSON serialization
            json_results = {
                'model_type': model_type,
                'batch_size': batch_size,
                'epochs': epochs,
                'learning_rate': learning_rate,
                'validation_split': validation_split,
                'total_time': results['total_time'],
                'time_per_epoch': results['time_per_epoch'],
                'train_losses': results['train_losses'],
                'val_losses': results['val_losses'],
                'final_train_loss': results['train_losses'][-1],
                'final_val_loss': results['val_losses'][-1],
                'timestamp': datetime.now().isoformat()
            }
            
            with open(save_path, 'w') as f:
                json.dump(json_results, f, indent=2)
            
            print(f"💾 Results saved to: {save_path}")
            
        except Exception as e:
            print(f"⚠️  Could not save results: {e}")
    
    return results


if __name__ == "__main__":
    import argparse
    import sys
    
    # Check if called with arguments (CLI mode) or without (API mode)
    if len(sys.argv) == 1:
        # API mode - prompt user for inputs
        print("🔧 Fast Training API Mode")
        print("=" * 40)
        
        # Get user inputs
        dataset_path = input("📁 Dataset path: ").strip()
        if not dataset_path:
            print("❌ Dataset path is required!")
            sys.exit(1)
            
        print("\n🤖 Available models:")
        print("1. fast_subspace_net (recommended)")
        print("2. simple_cnn (ultra-fast)")
        
        model_choice = input("Model choice (1 or 2, default=1): ").strip() or "1"
        model_type = "fast_subspace_net" if model_choice == "1" else "simple_cnn"
        
        batch_size = input("📦 Batch size (default=128): ").strip()
        batch_size = int(batch_size) if batch_size else 128
        
        epochs = input("🔄 Number of epochs (default=20): ").strip()
        epochs = int(epochs) if epochs else 20
        
        learning_rate = input("📈 Learning rate (default=1e-3): ").strip()
        learning_rate = float(learning_rate) if learning_rate else 1e-3
        
        print("\n" + "="*40)
        
        # Run training
        results = main(
            dataset_path=dataset_path,
            model_type=model_type,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate
        )
        
    else:
        # CLI mode - use argparse
        parser = argparse.ArgumentParser(description="Fast training for SubspaceNet models")
        parser.add_argument("--dataset", required=True, help="Path to dataset")
        parser.add_argument("--model", default="fast_subspace_net", 
                          choices=["fast_subspace_net", "simple_cnn"], help="Model type")
        parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
        parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
        parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
        parser.add_argument("--val-split", type=float, default=0.2, help="Validation split")
        parser.add_argument("--no-save", action="store_true", help="Don't save results")
        
        args = parser.parse_args()
        
        # Run training
        results = main(
            dataset_path=args.dataset,
            model_type=args.model,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.lr,
            validation_split=args.val_split,
            save_results=not args.no_save
        ) 