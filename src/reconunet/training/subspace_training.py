"""Tri4Net Subspace Models Training Script

This module provides training functionality for the subspace-based deep learning models
ported from SubspaceNet to Tri4Net. It handles data loading from Tri4Net datasets,
data preprocessing for different model types, and the training process.

Models supported:
- DeepRootMUSIC: CNN encoder + differentiable Root-MUSIC
- SubspaceNet: Generalized subspace method (Root-MUSIC/ESPRIT) 
- SubspaceNetEsprit: SubspaceNet hard-wired to ESPRIT
- SubspaceUNet: Lightweight U-Net encoder-decoder + differentiable subspace methods
- DeepAugmentedMUSIC: GRU-based MUSIC spectrum + MLP peak detector
- DeepCNN: Baseline CNN for DoA probability grid prediction

Classes:
- SubspaceTrainingParams: Training parameters for subspace models
- SubspaceTrainer: Main training class

Functions:
- create_autocorrelation_tensor: Create autocorrelation tensor for SubspaceNet models
- create_cov_tensor: Create covariance tensor for CNN models
- train_subspace_model: Main training function
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import time
import copy
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from typing import Union, Optional, Tuple, Dict, Any, List
from dataclasses import dataclass
import warnings

# Tri4Net imports
try:
    from ..data.dataset_generator import DOADataset
    from ..criterions import RMSPELoss, RMSPELoss_0_180, RMSPELoss_0_360
    from ..models.deep_learning.subspace_models import (
        DeepRootMUSIC, SubspaceNet, SubspaceNetEsprit, SubspaceUNet,
        DeepAugmentedMUSIC, DeepCNN
    )
except ImportError:
    # Fallback for when running as script
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from data.dataset_generator import DOADataset
    from criterions import RMSPELoss, RMSPELoss_0_180, RMSPELoss_0_360
    from models.deep_learning.subspace_models import (
        DeepRootMUSIC, SubspaceNet, SubspaceNetEsprit, SubspaceUNet,
        DeepAugmentedMUSIC, DeepCNN
    )

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# GPU optimizations
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
    torch.backends.cudnn.deterministic = False  # Allow non-deterministic algorithms for speed
    print("CUDA optimizations enabled")

# Note: Avoiding torch.set_default_device/dtype to prevent device mismatch issues
# Modern PyTorch handles device placement automatically


def custom_collate_fn(batch):
    """Custom collate function to handle variable-length DOA arrays - optimized."""
    batch_size = len(batch)
    
    # Pre-allocate for received signals (assuming consistent shape)
    first_sample = batch[0]
    signal_shape = first_sample['received_signal'].shape
    received_signals = torch.zeros((batch_size, *signal_shape), dtype=first_sample['received_signal'].dtype)
    
    # Pre-allocate label containers
    labels = {
        'doas': [],
        'snr': [],
        'num_snapshots': [],
        'num_sources': [],
        'sir': [],
        'num_multipath': [],
        'smr': [],
        'array_imperfections': []
    }
    
    # Batch process samples
    for i, sample in enumerate(batch):
        received_signals[i] = sample['received_signal']
        
        # Collect all labels as lists first to handle different data types
        for key in labels.keys():
            if key == 'doas':
                labels[key].append(sample['labels'][key])
            elif key in sample['labels']:
                labels[key].append(sample['labels'][key])
    
    # Convert lists to appropriate tensor formats
    for key in ['snr', 'num_snapshots', 'num_multipath']:
        if labels[key]:
            # Convert list to numpy array first, then to tensor for efficiency
            labels[key] = torch.tensor(np.array(labels[key]), dtype=torch.float32)
    
    for key in ['num_sources']:
        if labels[key]:
            # Convert list of numpy arrays to single numpy array first for efficiency
            labels[key] = torch.tensor(np.array(labels[key]), dtype=torch.long)
    
    for key in ['array_imperfections']:
        if labels[key]:
            # Convert numpy booleans to Python booleans first
            bool_values = [bool(val) for val in labels[key]]
            labels[key] = torch.tensor(np.array(bool_values), dtype=torch.bool)
    
    # Keep variable-length arrays as lists
    for key in ['sir', 'smr']:
        if not labels[key]:
            labels[key] = []
    
    return {
        'received_signal': received_signals,
        'labels': labels
    }


@dataclass
class SubspaceTrainingParams:
    """Training parameters for subspace models."""
    
    # Model configuration
    model_type: str  # 'deep_root_music', 'subspace_net', 'subspace_net_esprit', 'deep_augmented_music', 'deep_cnn_subspace'
    tau: int = 8  # Number of lags for autocorrelation (SubspaceNet models)
    diff_method: str = "root_music"  # 'root_music' or 'esprit' (SubspaceNet)
    
    # Training parameters
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    
    # Learning rate scheduler
    step_size: int = 30
    gamma: float = 0.1
    
    # Data parameters
    train_test_split: float = 0.9
    validation_split: float = 0.1
    
    # Performance optimizations
    use_mixed_precision: bool = True  # Use automatic mixed precision for speed
    gradient_clipping: float = 1.0  # Gradient clipping value (0 to disable)
    
    # Device
    device: str = "auto"  # 'auto', 'cpu', 'cuda'
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.device == "auto":
            self.device = str(device)
        
        # Disable mixed precision on CPU
        if self.device == "cpu":
            self.use_mixed_precision = False
            
        # Disable mixed precision for complex-based subspace models (CUDA doesn't support ComplexHalf)
        if self.model_type in ["subspace_net", "deep_root_music", "subspace_net_esprit", "subspace_unet"]:
            if self.use_mixed_precision:
                print("⚠️  Mixed precision disabled for complex-based subspace models (CUDA limitation)")
            self.use_mixed_precision = False


def create_autocorrelation_tensor(X: torch.Tensor, tau: int) -> torch.Tensor:
    """
    Create autocorrelation tensor for SubspaceNet models - vectorized.
    
    
    Args:
        X: Complex observation matrix [N, T]
        tau: Maximum lag for autocorrelation
        
    Returns:
        Autocorrelation tensor [tau, 2N, N]
    """
    N, T = X.shape
    device = X.device
    
    # Pre-allocate output tensor
    Rx_tau = torch.zeros(tau, 2*N, N, dtype=torch.float32, device=device)
    
    for lag in range(tau):
        if T - lag <= 0:
            continue
            
        # Vectorized autocorrelation computation
        X1 = X[:, :T-lag]  # [N, T-lag] - signals from t=0 to t=T-lag-1
        X2 = X[:, lag:T]   # [N, T-lag] - signals from t=lag to t=T-1
        
        # Compute autocorrelation matrix: E[X1 @ X2^H]
        Rx_lag = torch.matmul(X1, X2.conj().T) / (T - lag)  # [N, N]
        
        # Stack real and imaginary parts efficiently
        # Real part goes to indices [0:N, :], Imag part goes to indices [N:2N, :]
        Rx_tau[lag] = torch.cat([Rx_lag.real, Rx_lag.imag], dim=0)
    
    return Rx_tau





class SubspaceTrainer:
    """Main training class for subspace models."""
    
    def __init__(self, params: SubspaceTrainingParams):
        """Initialize trainer with parameters."""
        self.params = params
        self.device = torch.device(params.device)
        
        # Initialize model, optimizer, scheduler, criterion
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if params.use_mixed_precision else None
        
        # Training state
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_model_state = None
        
    def _create_model(self, N: int, M: int, T: int = None) -> nn.Module:
        """Create model based on configuration."""
        if self.params.model_type == "deep_root_music":
            model = DeepRootMUSIC(tau=self.params.tau, M=M)
        elif self.params.model_type == "subspace_net":
            model = SubspaceNet(tau=self.params.tau, M=M, diff_method=self.params.diff_method)
        elif self.params.model_type == "subspace_net_esprit":
            model = SubspaceNetEsprit(tau=self.params.tau, M=M)
        elif self.params.model_type == "subspace_unet":
            model = SubspaceUNet(tau=self.params.tau, M=M, diff_method=self.params.diff_method)
        elif self.params.model_type == "deep_augmented_music":
            if T is None:
                raise ValueError("T (number of snapshots) required for DeepAugmentedMUSIC")
            model = DeepAugmentedMUSIC(N=N, T=T, M=M)
        elif self.params.model_type == "deep_cnn_subspace":
            model = DeepCNN(N=N, grid_size=361)
        else:
            raise ValueError(f"Unknown model type: {self.params.model_type}")
            
        return model.to(self.device)
    
    def _create_optimizer_and_scheduler(self):
        """Create optimizer and learning rate scheduler."""
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.params.learning_rate,
            weight_decay=self.params.weight_decay
        )
        
        self.scheduler = lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.params.step_size,
            gamma=self.params.gamma
        )
    
    def _create_criterion(self):
        """Create loss criterion based on model type."""
        if self.params.model_type == "deep_cnn_subspace":
            self.criterion = nn.BCELoss()
        else:
            self.criterion = RMSPELoss_0_180()
    
    def _prepare_batch_data(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare batch data for specific model type - optimized.
        
        Args:
            batch: Batch from DOADataset
            
        Returns:
            Tuple of (input_tensor, target_tensor)
        """
        # Extract and move data to device once
        received_signal = batch['received_signal'].to(self.device, dtype=torch.complex64)  # [B, N, T]
        doas = batch['labels']['doas']  # Variable-length DoAs
        
        batch_size, N, T = received_signal.shape
        
        # Extract num_sources information to use only main sources (exclude multipath)
        num_sources_data = batch['labels']['num_sources']  # [B, 2] -> [main_sources, interference_sources]
        
        # Efficiently process DoAs to tensor with padding, using only main sources
        if isinstance(doas[0], np.ndarray):
            max_sources = max(len(d) for d in doas)
            doas_tensor = torch.full((batch_size, max_sources), fill_value=-999.0, 
                                   dtype=torch.float32, device=self.device)
            for i, doa_arr in enumerate(doas):
                # Use total sources (main + interference) but exclude multipath
                total_sources = int(num_sources_data[i].sum())  # main + interference sources
                source_doas = doa_arr[:total_sources]  # Take first total_sources DoAs (exclude multipath)
                doas_tensor[i, :len(source_doas)] = torch.from_numpy(source_doas.astype(np.float32))
        else:
            # Handle case where doas are already tensors or lists
            max_sources = max(len(d) if hasattr(d, '__len__') else 1 for d in doas)
            doas_tensor = torch.full((batch_size, max_sources), fill_value=-999.0,
                                   dtype=torch.float32, device=self.device)
            for i, d in enumerate(doas):
                # Use total sources (main + interference) but exclude multipath
                total_sources = int(num_sources_data[i].sum())  # main + interference sources
                if hasattr(d, '__len__'):
                    source_doas = d[:total_sources]  # Take first total_sources DoAs (exclude multipath)
                    if isinstance(source_doas, np.ndarray):
                        doas_tensor[i, :len(source_doas)] = torch.from_numpy(source_doas.astype(np.float32))
                    else:
                        doas_tensor[i, :len(source_doas)] = torch.tensor(source_doas, dtype=torch.float32)
                else:
                    if total_sources > 0:
                        doas_tensor[i, 0] = torch.tensor(d, dtype=torch.float32)
        
        # Prepare inputs based on model type
        if self.params.model_type in ["deep_root_music", "subspace_net", "subspace_net_esprit", "subspace_unet"]:
            # Vectorized autocorrelation computation for entire batch
            input_tensor = torch.zeros(batch_size, self.params.tau, 2*N, N, 
                                     dtype=torch.float32, device=self.device)
            
            # Process all samples in batch efficiently
            for i in range(batch_size):
                input_tensor[i] = create_autocorrelation_tensor(received_signal[i], self.params.tau)
            
            # Convert DoAs to radians once
            target_tensor = doas_tensor * (torch.pi / 180.0)
            
        elif self.params.model_type == "deep_augmented_music":
            # Use received signals directly (already on device)
            input_tensor = received_signal
            target_tensor = doas_tensor * (torch.pi / 180.0)
            
        elif self.params.model_type == "deep_cnn_subspace":
            # Vectorized covariance computation for entire batch
            input_tensor = torch.zeros(batch_size, N, N, 3, dtype=torch.float32, device=self.device)
            target_tensor = torch.zeros(batch_size, 361, dtype=torch.float32, device=self.device)
            
            # Create angle grid once (reuse for all samples)
            angles_grid = torch.linspace(-90, 90, 361, device=self.device)
            
            for i in range(batch_size):
                # Compute covariance efficiently
                X = received_signal[i]  # [N, T]
                Rx = torch.matmul(X, X.conj().T) / T  # [N, N]
                
                # Stack components efficiently
                input_tensor[i] = torch.stack([Rx.real, Rx.imag, torch.angle(Rx)], dim=2)
                
                # Create grid labels efficiently
                valid_doas = doas_tensor[i][doas_tensor[i] != -999.0]
                if len(valid_doas) > 0:
                    # Vectorized closest point finding
                    diffs = torch.abs(angles_grid.unsqueeze(0) - valid_doas.unsqueeze(1))  # [num_sources, 361]
                    closest_indices = torch.argmin(diffs, dim=1)
                    target_tensor[i, closest_indices] = 1.0
        else:
            raise ValueError(f"Unknown model type: {self.params.model_type}")
        
        return input_tensor, target_tensor
    
    def _compute_loss(self, model_output, target_tensor):
        """Compute loss based on model type - optimized."""
        if self.params.model_type in ["deep_root_music", "subspace_net", "subspace_unet"]:
            # These models return (doa_pred, doa_all, roots, Rz)
            if isinstance(model_output, (tuple, list)):
                doa_pred = model_output[0]
            else:
                doa_pred = model_output
        elif self.params.model_type == "subspace_net_esprit":
            # Returns (doa_pred, Rz)
            if isinstance(model_output, (tuple, list)):
                doa_pred = model_output[0]
            else:
                doa_pred = model_output
        else:
            # Other models return single tensor
            doa_pred = model_output
        
        # Convert model predictions from degrees to radians to match target_tensor
        # Models (RootMUSIC/ESPRIT) typically output angles in degrees
        doa_pred = doa_pred * (torch.pi / 180.0)
        
        # Handle different model types efficiently
        if self.params.model_type == "deep_cnn_subspace":
            # CNN uses standard BCE loss - no special handling needed
            return self.criterion(doa_pred, target_tensor)
        else:
            # Vectorized handling of variable-length sequences
            batch_size = target_tensor.shape[0]
            
            # Create mask for valid DoAs (not -999.0) once
            valid_mask = target_tensor != -999.0  # [B, max_sources]
            
            # Get number of valid targets per sample
            num_valid_per_sample = valid_mask.sum(dim=1)  # [B]
            
            # Skip samples with no valid targets
            valid_samples = num_valid_per_sample > 0
            if not valid_samples.any():
                return torch.tensor(0.0, device=self.device, requires_grad=True)
            
            # Process only valid samples
            valid_targets = target_tensor[valid_samples]  # [B_valid, max_sources]
            valid_preds = doa_pred[valid_samples]  # [B_valid, max_sources]
            valid_mask_filtered = valid_mask[valid_samples]  # [B_valid, max_sources]
            
            # Compute losses only for valid predictions/targets
            losses = []
            for i in range(valid_targets.shape[0]):
                sample_mask = valid_mask_filtered[i]
                n_valid = sample_mask.sum().item()
                
                if n_valid > 0:
                    sample_targets = valid_targets[i, sample_mask]  # [n_valid]
                    all_preds = valid_preds[i]  # [M] - all predictions
                    
                    # Try all combinations of n_valid predictions from all M predictions
                    from itertools import combinations
                    combination_losses = []
                    
                    for pred_indices in combinations(range(len(all_preds)), n_valid):
                        selected_preds = all_preds[list(pred_indices)]  # [n_valid]
                        
                        # Compute RMSPE loss for this combination
                        loss = self.criterion(selected_preds.unsqueeze(0), sample_targets.unsqueeze(0))
                        combination_losses.append(loss)
                    
                    # Take minimum loss while preserving gradients
                    min_loss = torch.stack(combination_losses).min()
                    losses.append(min_loss)
            
            # Return mean loss across valid samples
            if losses:
                return torch.stack(losses).mean()
            else:
                return torch.tensor(0.0, device=self.device, requires_grad=True)
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch - optimized."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
            # Prepare data
            input_tensor, target_tensor = self._prepare_batch_data(batch)
            
            # Forward pass with mixed precision (excluding complex operations)
            self.optimizer.zero_grad()
            
            if self.scaler and self.params.model_type not in ["subspace_net", "deep_root_music", "subspace_net_esprit", "subspace_unet"]:
                # Use mixed precision only for non-complex models
                with torch.cuda.amp.autocast():
                    model_output = self.model(input_tensor)
                    loss = self._compute_loss(model_output, target_tensor)
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping if enabled
                if self.params.gradient_clipping > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.gradient_clipping)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Full precision for complex-based subspace models or when no scaler
                model_output = self.model(input_tensor)
                loss = self._compute_loss(model_output, target_tensor)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping if enabled
                if self.params.gradient_clipping > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.gradient_clipping)
                
                self.optimizer.step()
            
            # Update statistics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
            
            # Memory cleanup for GPU
            if self.device.type == 'cuda':
                del input_tensor, target_tensor, model_output, loss
                torch.cuda.empty_cache()
        
        return total_loss / max(num_batches, 1)
    
    def validate_epoch(self, val_loader: DataLoader) -> float:
        """Validate for one epoch - optimized."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Prepare data
                input_tensor, target_tensor = self._prepare_batch_data(batch)
                
                # Forward pass with mixed precision (excluding complex operations)
                if self.scaler and self.params.model_type not in ["subspace_net", "deep_root_music", "subspace_net_esprit", "subspace_unet"]:
                    with torch.cuda.amp.autocast():
                        model_output = self.model(input_tensor)
                        loss = self._compute_loss(model_output, target_tensor)
                else:
                    # Full precision for complex-based subspace models
                    model_output = self.model(input_tensor)
                    loss = self._compute_loss(model_output, target_tensor)
                
                # Update statistics
                total_loss += loss.item()
                num_batches += 1
                
                # Memory cleanup for GPU
                if self.device.type == 'cuda':
                    del input_tensor, target_tensor, model_output, loss
        
        return total_loss / max(num_batches, 1)
    
    def train(self, dataset: DOADataset, save_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Main training function.
        
        Args:
            dataset: Tri4Net DOADataset
            save_path: Path to save trained model
            
        Returns:
            Training results dictionary
        """
        print(f"\n=== Training {self.params.model_type} ===")
        print(f"Dataset size: {len(dataset)}")
        print(f"Device: {self.device}")
        
        # Split dataset with explicit generator to avoid device mismatch
        train_size = int(self.params.train_test_split * len(dataset))
        val_size = len(dataset) - train_size
        
        # Use default generator to avoid device mismatch issues
        # PyTorch will automatically handle device placement
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Create optimized data loaders  
        # For demo purposes, use num_workers=0 to avoid multiprocessing issues
        num_workers = 0  # min(4, torch.get_num_threads()) if self.device.type == 'cuda' else 0
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.params.batch_size, 
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if self.device.type == 'cuda' else False,
            persistent_workers=False,  # Disabled for single-threaded
            collate_fn=custom_collate_fn
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.params.batch_size, 
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if self.device.type == 'cuda' else False,
            persistent_workers=False,  # Disabled for single-threaded
            collate_fn=custom_collate_fn
        )
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        
        # Get sample to determine model parameters
        sample = dataset[0]
        received_signal = sample['received_signal']
        N = received_signal.shape[0]  # Number of antennas
        T = received_signal.shape[1]  # Number of snapshots
        
        # Determine maximum number of sources across the entire dataset
        print("🔍 Analyzing dataset to find maximum number of sources...")
        
        # Try to get max sources efficiently from dataset metadata
        try:
            # If dataset has get_label_summary method, use it for efficiency
            if hasattr(dataset, 'get_label_summary'):
                label_summary = dataset.get_label_summary()
                print("📊 Using dataset label summary for source analysis")
                
                # Check num_sources field if available (format: [[main, interference], ...])
                if hasattr(dataset, 'labels') and 'num_sources' in dataset.labels:
                    num_sources_data = dataset.labels['num_sources']
                    if len(num_sources_data.shape) == 2:  # [[main, interference], ...]
                        max_total_sources = num_sources_data.sum(axis=1).max()
                    else:
                        max_total_sources = num_sources_data.max()
                    print(f"   From num_sources field: {max_total_sources}")
                else:
                    max_total_sources = 1
                
                # Double-check with actual DoA data from a few samples
                max_doa_length = 0
                sample_indices = range(0, len(dataset), max(1, len(dataset) // 50))  # Sample every 2%
                for idx in list(sample_indices)[:20]:  # Check max 20 samples
                    sample_doas = dataset[idx]['labels']['doas']
                    doa_length = len(sample_doas) if hasattr(sample_doas, '__len__') else 1
                    max_doa_length = max(max_doa_length, doa_length)
                
                M = max(max_total_sources, max_doa_length)
                print(f"   From DoA arrays: {max_doa_length}")
                
            else:
                # Fallback: sample the dataset
                print("📊 Sampling dataset to determine max sources")
                max_sources = 0
                sample_indices = range(0, len(dataset), max(1, len(dataset) // 100))
                
                for idx in sample_indices:
                    sample_doas = dataset[idx]['labels']['doas']
                    num_sources = len(sample_doas) if hasattr(sample_doas, '__len__') else 1
                    max_sources = max(max_sources, num_sources)
                
                M = max_sources
                
        except Exception as e:
            print(f"⚠️  Warning: Could not efficiently determine max sources: {e}")
            print("   Using fallback method...")
            # Safe fallback: assume reasonable maximum
            M = 5  # Conservative estimate
            
        # Ensure M is at least 1
        M = max(1, M)
        
        print(f"Array size (N): {N}")
        print(f"Snapshots (T): {T}")
        print(f"Maximum sources (M): {M} (determined from {len(list(sample_indices))} samples)")
        
        # Create model, optimizer, scheduler, criterion
        self.model = self._create_model(N, M, T)
        self._create_optimizer_and_scheduler()
        self._create_criterion()
        
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Training loop
        start_time = time.time()
        
        for epoch in range(self.params.epochs):
            print(f"\nEpoch {epoch+1}/{self.params.epochs}")
            
            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate_epoch(val_loader)
            self.val_losses.append(val_loss)
            
            # Update scheduler
            self.scheduler.step()
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                print(f"New best validation loss: {val_loss:.6f}")
            
            print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}")
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f} seconds")
        print(f"Best validation loss: {self.best_val_loss:.6f}")
        
        # Save model if path provided
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save model state dict
            torch.save(self.best_model_state, save_path)
            print(f"Model saved to: {save_path}")
            
            # Save training curves
            self.plot_training_curves(save_path.parent / f"{save_path.stem}_curves.png")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'training_time': training_time,
            'model_params': sum(p.numel() for p in self.model.parameters())
        }
    
    def plot_training_curves(self, save_path: Optional[Path] = None):
        """Plot training and validation loss curves."""
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.train_losses) + 1), self.train_losses, 'b-', label='Training Loss')
        plt.plot(range(1, len(self.val_losses) + 1), self.val_losses, 'r-', label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{self.params.model_type} Training Curves')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training curves saved to: {save_path}")
        else:
            plt.show()


def train_subspace_model(
    dataset_path: Union[str, Path],
    model_type: str,
    save_dir: Optional[Union[str, Path]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to train a subspace model.
    
    Args:
        dataset_path: Path to Tri4Net dataset (.h5 file)
        model_type: Type of model to train
        save_dir: Directory to save trained model and results
        **kwargs: Additional parameters for SubspaceTrainingParams
        
    Returns:
        Training results dictionary
    """
    # Load dataset
    print(f"Loading dataset from: {dataset_path}")
    dataset = DOADataset(dataset_path)
    
    print(f"Dataset loaded successfully!")
    print("Label summary:")
    for label, summary in dataset.get_label_summary().items():
        print(f"  {label}: {summary}")
    
    # Create training parameters
    params = SubspaceTrainingParams(model_type=model_type, **kwargs)
    
    # Create trainer
    trainer = SubspaceTrainer(params)
    
    # Determine save path
    if save_dir is not None:
        save_dir = Path(save_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = save_dir / f"{model_type}_{timestamp}.pth"
    else:
        save_path = None
    
    # Train model
    results = trainer.train(dataset, save_path)
    
    return results


if __name__ == "__main__":
    """Example usage."""
    
    # Example: Train SubspaceNet model (also try "subspace_unet" for U-Net variant)
    dataset_path = "/Users/itamaralmog/Documents/Masters/Second Year/All_DOA_nets/Tri4Net/Data/datasets/linear/ula_dataset_90_all_scenarios/ula_dataset_90_all_scenarios.h5"
    
    if Path(dataset_path).exists():
        results = train_subspace_model(
            dataset_path=dataset_path,
            model_type="subspace_unet",
            tau=5,
            diff_method="root_music",
            epochs=50,
            batch_size=16,
            learning_rate=1e-3,
            save_dir="models/trained"
        )
        
        print("\nTraining Results:")
        print(f"Best validation loss: {results['best_val_loss']:.6f}")
        print(f"Training time: {results['training_time']:.2f} seconds")
        print(f"Model parameters: {results['model_params']:,}")
    else:
        print(f"Dataset not found: {dataset_path}")
        print("Please generate a dataset first using the dataset_generator.py script") 