"""Training script for scenario classification models.

This script trains deep learning models to classify DoA scenarios into 36 classes
based on source/multipath combinations, array conditions, and SNR levels.

Usage:
    python classification_training.py --dataset path/to/dataset.h5 --model scenario_cnn --epochs 100
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Import models
try:
    from ..data.dataset_generator import DOADataset
    from ..models.deep_learning.classification import (
        ScenarioClassificationCNN, CovarianceResNet, 
        SignalTransformer, HybridCNNRNN
    )
except ImportError:
    # Fallback for when running as script
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from data.dataset_generator import DOADataset
    from models.deep_learning.classification import (
        ScenarioClassificationCNN, CovarianceResNet,
        SignalTransformer, HybridCNNRNN
    )


@dataclass
class ClassificationTrainingParams:
    """Training parameters for scenario classification."""
    
    # Data parameters
    dataset_path: str
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    
    # Model parameters
    model_type: str = "scenario_cnn"  # scenario_cnn, scenario_resnet, scenario_transformer, scenario_hybrid
    
    # Training parameters
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    
    # Optimization
    optimizer: str = "adamw"  # adamw, adam, sgd
    scheduler: str = "cosine"  # cosine, step, plateau
    warmup_epochs: int = 5
    
    # Regularization
    dropout_rate: float = 0.5
    label_smoothing: float = 0.1
    
    # Hardware
    device: str = "auto"  # auto, cuda, cpu
    num_workers: int = 4
    
    # Saving
    save_dir: str = "models/classification"
    save_best_only: bool = True
    early_stopping_patience: int = 15
    
    # Logging
    log_interval: int = 100
    use_tensorboard: bool = True


class ScenarioLabelEncoder:
    """Encodes dataset labels into scenario classes."""
    
    def __init__(self):
        self.scenario_map = self._create_scenario_mapping()
        self.num_classes = len(self.scenario_map)
        
    def _create_scenario_mapping(self) -> Dict[Tuple, int]:
        """Create mapping from (sources, multipath, array_perfect, snr_level) to class index."""
        scenario_map = {}
        class_idx = 0
        
        # 6 source/multipath combinations (examples - adjust based on your dataset)
        source_multipath_combos = [
            (1, 0, 0),  # 1 main source, 0 interference, 0 multipath
            (1, 0, 1),  # 1 main source, 0 interference, 1 multipath
            (1, 1, 0),  # 1 main source, 1 interference, 0 multipath
            (1, 1, 1),  # 1 main source, 1 interference, 1 multipath
            (1, 2, 0),  # 1 main source, 2 interference, 0 multipath
            (1, 2, 1),  # 1 main source, 2 interference, 1 multipath
        ]
        
        # 2 array conditions
        array_conditions = [True, False]  # Perfect, Imperfect
        
        # 3 SNR levels
        snr_levels = ["low", "mid", "high"]
        
        for sources_mp in source_multipath_combos:
            for array_perfect in array_conditions:
                for snr_level in snr_levels:
                    key = (*sources_mp, array_perfect, snr_level)
                    scenario_map[key] = class_idx
                    class_idx += 1
        
        return scenario_map
    
    def _categorize_snr(self, snr: float) -> str:
        """Categorize SNR into low/mid/high."""
        if snr < 0:
            return "low"
        elif snr < 10:
            return "mid"
        else:
            return "high"
    
    def encode(self, num_sources: np.ndarray, num_multipath: int, 
               array_imperfections: bool, snr: float) -> int:
        """Encode labels into scenario class."""
        main_sources = int(num_sources[0])
        interference_sources = int(num_sources[1]) if len(num_sources) > 1 else 0
        
        array_perfect = not array_imperfections
        snr_level = self._categorize_snr(snr)
        
        key = (main_sources, interference_sources, num_multipath, array_perfect, snr_level)
        
        if key not in self.scenario_map:
            # Handle unseen scenarios by mapping to closest scenario
            # For now, use a default scenario
            print(f"Warning: Unseen scenario {key}, using default class 0")
            return 0
        
        return self.scenario_map[key]
    
    def get_class_names(self) -> List[str]:
        """Get human-readable class names."""
        names = [""] * self.num_classes
        for key, idx in self.scenario_map.items():
            main_src, int_src, multipath, array_perfect, snr_level = key
            array_str = "perfect" if array_perfect else "imperfect"
            names[idx] = f"S{main_src}I{int_src}M{multipath}_{array_str}_{snr_level}"
        return names


class ScenarioDataset(Dataset):
    """Dataset wrapper for scenario classification."""
    
    def __init__(self, doa_dataset: DOADataset, label_encoder: ScenarioLabelEncoder):
        self.doa_dataset = doa_dataset
        self.label_encoder = label_encoder
        
    def __len__(self):
        return len(self.doa_dataset)
    
    def __getitem__(self, idx):
        sample = self.doa_dataset[idx]
        
        # Extract received signal
        received_signal = sample['received_signal']  # [N, T] complex
        
        # Extract labels for scenario encoding
        labels = sample['labels']
        num_sources = labels['num_sources']
        num_multipath = labels.get('num_multipath', 0)
        array_imperfections = labels.get('array_imperfections', False)
        snr = labels.get('snr', 10.0)  # Default SNR if not available
        
        # Encode scenario
        scenario_class = self.label_encoder.encode(
            num_sources, num_multipath, array_imperfections, snr
        )
        
        return {
            'signal': received_signal,
            'scenario': scenario_class
        }


class ClassificationTrainer:
    """Trainer for scenario classification models."""
    
    def __init__(self, params: ClassificationTrainingParams):
        self.params = params
        self.device = self._setup_device()
        self.label_encoder = ScenarioLabelEncoder()
        
        # Setup logging
        if params.use_tensorboard:
            self.writer = SummaryWriter(f"{params.save_dir}/logs")
        else:
            self.writer = None
        
        # Create save directory
        Path(params.save_dir).mkdir(parents=True, exist_ok=True)
        
    def _setup_device(self) -> torch.device:
        """Setup compute device."""
        if self.params.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.params.device)
        
        print(f"Using device: {device}")
        return device
    
    def _create_model(self, N: int, T: int) -> nn.Module:
        """Create model based on configuration."""
        num_classes = self.label_encoder.num_classes
        
        if self.params.model_type == "scenario_cnn":
            model = ScenarioClassificationCNN(
                N=N, T=T, num_classes=num_classes,
                dropout_rate=self.params.dropout_rate
            )
        elif self.params.model_type == "scenario_resnet":
            model = CovarianceResNet(
                N=N, num_classes=num_classes,
                pretrained=False, resnet_variant='resnet18'
            )
        elif self.params.model_type == "scenario_transformer":
            model = SignalTransformer(
                N=N, T=T, num_classes=num_classes
            )
        elif self.params.model_type == "scenario_hybrid":
            model = HybridCNNRNN(
                N=N, T=T, num_classes=num_classes
            )
        else:
            raise ValueError(f"Unknown model type: {self.params.model_type}")
        
        return model.to(self.device)
    
    def _create_dataloaders(self, dataset_path: str) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create train/val/test dataloaders."""
        print(f"Loading dataset from: {dataset_path}")
        
        # Load DOA dataset
        doa_dataset = DOADataset(dataset_path)
        print(f"Dataset loaded: {len(doa_dataset)} samples")
        
        # Wrap with scenario dataset
        scenario_dataset = ScenarioDataset(doa_dataset, self.label_encoder)
        
        # Split dataset
        total_size = len(scenario_dataset)
        train_size = int(self.params.train_split * total_size)
        val_size = int(self.params.val_split * total_size)
        test_size = total_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            scenario_dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        print(f"Split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, batch_size=self.params.batch_size, shuffle=True,
            num_workers=self.params.num_workers, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.params.batch_size, shuffle=False,
            num_workers=self.params.num_workers, pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.params.batch_size, shuffle=False,
            num_workers=self.params.num_workers, pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
    
    def _create_optimizer_and_scheduler(self, model: nn.Module) -> Tuple[optim.Optimizer, Any]:
        """Create optimizer and learning rate scheduler."""
        # Optimizer
        if self.params.optimizer == "adamw":
            optimizer = optim.AdamW(
                model.parameters(), lr=self.params.learning_rate,
                weight_decay=self.params.weight_decay
            )
        elif self.params.optimizer == "adam":
            optimizer = optim.Adam(
                model.parameters(), lr=self.params.learning_rate,
                weight_decay=self.params.weight_decay
            )
        elif self.params.optimizer == "sgd":
            optimizer = optim.SGD(
                model.parameters(), lr=self.params.learning_rate,
                momentum=0.9, weight_decay=self.params.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.params.optimizer}")
        
        # Scheduler
        if self.params.scheduler == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.params.epochs
            )
        elif self.params.scheduler == "step":
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=30, gamma=0.1
            )
        elif self.params.scheduler == "plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=5, factor=0.5
            )
        else:
            raise ValueError(f"Unknown scheduler: {self.params.scheduler}")
        
        return optimizer, scheduler
    
    def train_epoch(self, model: nn.Module, train_loader: DataLoader,
                   criterion: nn.Module, optimizer: optim.Optimizer) -> float:
        """Train for one epoch."""
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(train_loader):
            signals = batch['signal'].to(self.device, dtype=torch.complex64)
            targets = batch['scenario'].to(self.device)
            
            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if batch_idx % self.params.log_interval == 0:
                print(f'Batch {batch_idx}/{len(train_loader)}: '
                      f'Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
        
        return total_loss / len(train_loader)
    
    def validate_epoch(self, model: nn.Module, val_loader: DataLoader,
                      criterion: nn.Module) -> Tuple[float, float]:
        """Validate for one epoch."""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                signals = batch['signal'].to(self.device, dtype=torch.complex64)
                targets = batch['scenario'].to(self.device)
                
                outputs = model(signals)
                loss = criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def evaluate_model(self, model: nn.Module, test_loader: DataLoader) -> Dict[str, Any]:
        """Comprehensive model evaluation."""
        model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                signals = batch['signal'].to(self.device, dtype=torch.complex64)
                targets = batch['scenario'].to(self.device)
                
                outputs = model(signals)
                _, predicted = outputs.max(1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Calculate metrics
        accuracy = np.mean(np.array(all_predictions) == np.array(all_targets))
        
        # Classification report
        class_names = self.label_encoder.get_class_names()
        report = classification_report(
            all_targets, all_predictions,
            target_names=class_names, output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(all_targets, all_predictions)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': all_predictions,
            'targets': all_targets
        }
    
    def save_confusion_matrix(self, cm: np.ndarray, class_names: List[str], epoch: int):
        """Save confusion matrix plot."""
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - Epoch {epoch}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f"{self.params.save_dir}/confusion_matrix_epoch_{epoch}.png", dpi=300)
        plt.close()
    
    def train(self, dataset_path: str) -> Dict[str, Any]:
        """Main training loop."""
        print("🚀 Starting scenario classification training...")
        start_time = time.time()
        
        # Create dataloaders
        train_loader, val_loader, test_loader = self._create_dataloaders(dataset_path)
        
        # Get data dimensions from first batch
        sample_batch = next(iter(train_loader))
        N, T = sample_batch['signal'].shape[1], sample_batch['signal'].shape[2]
        print(f"Data dimensions: N={N}, T={T}")
        print(f"Number of scenario classes: {self.label_encoder.num_classes}")
        
        # Create model
        model = self._create_model(N, T)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {num_params:,}")
        
        # Create optimizer and scheduler
        optimizer, scheduler = self._create_optimizer_and_scheduler(model)
        
        # Loss function with label smoothing
        criterion = nn.CrossEntropyLoss(label_smoothing=self.params.label_smoothing)
        
        # Training variables
        best_val_loss = float('inf')
        best_val_acc = 0.0
        patience_counter = 0
        train_losses = []
        val_losses = []
        val_accuracies = []
        
        # Training loop
        for epoch in range(self.params.epochs):
            print(f"\nEpoch {epoch+1}/{self.params.epochs}")
            print("-" * 50)
            
            # Training
            train_loss = self.train_epoch(model, train_loader, criterion, optimizer)
            train_losses.append(train_loss)
            
            # Validation
            val_loss, val_acc = self.validate_epoch(model, val_loader, criterion)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Learning rate scheduling
            if self.params.scheduler == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()
            
            # Logging
            if self.writer:
                self.writer.add_scalar('Loss/Train', train_loss, epoch)
                self.writer.add_scalar('Loss/Val', val_loss, epoch)
                self.writer.add_scalar('Accuracy/Val', val_acc, epoch)
                self.writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_loss = val_loss
                patience_counter = 0
                
                if self.params.save_best_only:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_loss,
                        'val_acc': val_acc,
                        'params': self.params
                    }, f"{self.params.save_dir}/best_model.pth")
                    print(f"✅ New best model saved! Val Acc: {val_acc:.2f}%")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.params.early_stopping_patience:
                print(f"Early stopping after {epoch+1} epochs")
                break
        
        # Final evaluation
        print("\n🎯 Final evaluation on test set...")
        evaluation_results = self.evaluate_model(model, test_loader)
        
        # Save confusion matrix
        self.save_confusion_matrix(
            evaluation_results['confusion_matrix'],
            self.label_encoder.get_class_names(),
            epoch + 1
        )
        
        # Save training history
        training_history = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'best_val_acc': best_val_acc,
            'best_val_loss': best_val_loss,
            'training_time': time.time() - start_time,
            'model_params': num_params,
            'final_evaluation': evaluation_results
        }
        
        with open(f"{self.params.save_dir}/training_history.json", 'w') as f:
            json.dump(training_history, f, indent=2, default=str)
        
        if self.writer:
            self.writer.close()
        
        print(f"\n✅ Training completed!")
        print(f"Best validation accuracy: {best_val_acc:.2f}%")
        print(f"Test accuracy: {evaluation_results['accuracy']*100:.2f}%")
        print(f"Training time: {training_history['training_time']:.2f} seconds")
        
        return training_history


def load_config_file(config_path: str) -> Dict[str, Any]:
    """Load training configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def save_config_template(output_path: str):
    """Save a template configuration file with all parameters."""
    template = {
        "dataset_path": "path/to/your/dataset.h5",
        "model_type": "scenario_cnn",
        "epochs": 100,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "train_split": 0.8,
        "val_split": 0.1,
        "test_split": 0.1,
        "optimizer": "adamw",
        "scheduler": "cosine",
        "warmup_epochs": 5,
        "dropout_rate": 0.5,
        "label_smoothing": 0.1,
        "device": "auto",
        "num_workers": 4,
        "save_dir": "models/classification",
        "save_best_only": True,
        "early_stopping_patience": 15,
        "log_interval": 100,
        "use_tensorboard": True
    }
    
    with open(output_path, 'w') as f:
        json.dump(template, f, indent=2)
    print(f"📝 Configuration template saved to: {output_path}")
    print("✏️  Edit this file and use --config to load your custom parameters")


def manual_config() -> ClassificationTrainingParams:
    """Manual parameter configuration with full control."""
    print("\n🎛️  Manual Training Configuration")
    print("=" * 60)
    
    # Create default parameters
    params = ClassificationTrainingParams(
        dataset_path="Data/datasets/linear/ula_dataset_90_all_scenarios/ula_dataset_90_all_scenarios.h5",
        model_type="scenario_cnn",
        epochs=100,
        batch_size=32,
        learning_rate=1e-3
    )
    
    while True:
        # Display current configuration
        print("\n📋 Current Configuration:")
        print("-" * 40)
        print(f" 1. Dataset path: {params.dataset_path}")
        print(f" 2. Model type: {params.model_type}")
        print(f" 3. Epochs: {params.epochs}")
        print(f" 4. Batch size: {params.batch_size}")
        print(f" 5. Learning rate: {params.learning_rate}")
        print(f" 6. Weight decay: {params.weight_decay}")
        print(f" 7. Train split: {params.train_split}")
        print(f" 8. Validation split: {params.val_split}")
        print(f" 9. Test split: {params.test_split}")
        print(f"10. Optimizer: {params.optimizer}")
        print(f"11. Scheduler: {params.scheduler}")
        print(f"12. Dropout rate: {params.dropout_rate}")
        print(f"13. Label smoothing: {params.label_smoothing}")
        print(f"14. Device: {params.device}")
        print(f"15. Number of workers: {params.num_workers}")
        print(f"16. Save directory: {params.save_dir}")
        print(f"17. Early stopping patience: {params.early_stopping_patience}")
        print(f"18. Log interval: {params.log_interval}")
        print(f"19. Use TensorBoard: {params.use_tensorboard}")
        print("-" * 40)
        print(" 0. ✅ Start training with these parameters")
        print("99. ❌ Cancel")
        
        choice = input("\nSelect parameter to modify (0-19, 99): ").strip()
        
        if choice == "0":
            break
        elif choice == "99":
            print("❌ Configuration cancelled.")
            return None
        
        try:
            choice_num = int(choice)
            
            if choice_num == 1:
                new_path = input(f"Dataset path [{params.dataset_path}]: ").strip()
                if new_path:
                    params.dataset_path = new_path
                    
            elif choice_num == 2:
                print("Available models:")
                models = ['scenario_cnn', 'scenario_resnet', 'scenario_transformer', 'scenario_hybrid']
                for i, model in enumerate(models, 1):
                    print(f"  {i}. {model}")
                model_choice = input("Choose model (1-4): ").strip()
                if model_choice.isdigit() and 1 <= int(model_choice) <= 4:
                    params.model_type = models[int(model_choice) - 1]
                    
            elif choice_num == 3:
                new_epochs = input(f"Epochs [{params.epochs}]: ").strip()
                if new_epochs.isdigit():
                    params.epochs = int(new_epochs)
                    
            elif choice_num == 4:
                new_batch = input(f"Batch size [{params.batch_size}]: ").strip()
                if new_batch.isdigit():
                    params.batch_size = int(new_batch)
                    
            elif choice_num == 5:
                new_lr = input(f"Learning rate [{params.learning_rate}]: ").strip()
                if new_lr:
                    try:
                        params.learning_rate = float(new_lr)
                    except ValueError:
                        print("❌ Invalid learning rate")
                        
            elif choice_num == 6:
                new_wd = input(f"Weight decay [{params.weight_decay}]: ").strip()
                if new_wd:
                    try:
                        params.weight_decay = float(new_wd)
                    except ValueError:
                        print("❌ Invalid weight decay")
                        
            elif choice_num == 7:
                new_split = input(f"Train split [{params.train_split}]: ").strip()
                if new_split:
                    try:
                        params.train_split = float(new_split)
                    except ValueError:
                        print("❌ Invalid train split")
                        
            elif choice_num == 8:
                new_split = input(f"Validation split [{params.val_split}]: ").strip()
                if new_split:
                    try:
                        params.val_split = float(new_split)
                    except ValueError:
                        print("❌ Invalid validation split")
                        
            elif choice_num == 9:
                new_split = input(f"Test split [{params.test_split}]: ").strip()
                if new_split:
                    try:
                        params.test_split = float(new_split)
                    except ValueError:
                        print("❌ Invalid test split")
                        
            elif choice_num == 10:
                print("Available optimizers:")
                optimizers = ['adamw', 'adam', 'sgd']
                for i, opt in enumerate(optimizers, 1):
                    print(f"  {i}. {opt}")
                opt_choice = input("Choose optimizer (1-3): ").strip()
                if opt_choice.isdigit() and 1 <= int(opt_choice) <= 3:
                    params.optimizer = optimizers[int(opt_choice) - 1]
                    
            elif choice_num == 11:
                print("Available schedulers:")
                schedulers = ['cosine', 'step', 'plateau']
                for i, sched in enumerate(schedulers, 1):
                    print(f"  {i}. {sched}")
                sched_choice = input("Choose scheduler (1-3): ").strip()
                if sched_choice.isdigit() and 1 <= int(sched_choice) <= 3:
                    params.scheduler = schedulers[int(sched_choice) - 1]
                    
            elif choice_num == 12:
                new_dropout = input(f"Dropout rate [{params.dropout_rate}]: ").strip()
                if new_dropout:
                    try:
                        params.dropout_rate = float(new_dropout)
                    except ValueError:
                        print("❌ Invalid dropout rate")
                        
            elif choice_num == 13:
                new_label_smooth = input(f"Label smoothing [{params.label_smoothing}]: ").strip()
                if new_label_smooth:
                    try:
                        params.label_smoothing = float(new_label_smooth)
                    except ValueError:
                        print("❌ Invalid label smoothing")
                        
            elif choice_num == 14:
                print("Available devices:")
                devices = ['auto', 'cuda', 'cpu']
                for i, dev in enumerate(devices, 1):
                    print(f"  {i}. {dev}")
                dev_choice = input("Choose device (1-3): ").strip()
                if dev_choice.isdigit() and 1 <= int(dev_choice) <= 3:
                    params.device = devices[int(dev_choice) - 1]
                    
            elif choice_num == 15:
                new_workers = input(f"Number of workers [{params.num_workers}]: ").strip()
                if new_workers.isdigit():
                    params.num_workers = int(new_workers)
                    
            elif choice_num == 16:
                new_save_dir = input(f"Save directory [{params.save_dir}]: ").strip()
                if new_save_dir:
                    params.save_dir = new_save_dir
                    
            elif choice_num == 17:
                new_patience = input(f"Early stopping patience [{params.early_stopping_patience}]: ").strip()
                if new_patience.isdigit():
                    params.early_stopping_patience = int(new_patience)
                    
            elif choice_num == 18:
                new_log_interval = input(f"Log interval [{params.log_interval}]: ").strip()
                if new_log_interval.isdigit():
                    params.log_interval = int(new_log_interval)
                    
            elif choice_num == 19:
                tensorboard_choice = input(f"Use TensorBoard? (y/n) [{params.use_tensorboard}]: ").strip().lower()
                if tensorboard_choice in ['y', 'yes', 'true', '1']:
                    params.use_tensorboard = True
                elif tensorboard_choice in ['n', 'no', 'false', '0']:
                    params.use_tensorboard = False
                    
            else:
                print("❌ Invalid choice")
                
        except ValueError:
            print("❌ Invalid input")
        except KeyboardInterrupt:
            print("\n❌ Configuration cancelled.")
            return None
    
    print(f"\n✅ Configuration complete!")
    return params


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Train scenario classification models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python classification_training.py --dataset data.h5 --model scenario_cnn
  
  # Advanced configuration
  python classification_training.py --dataset data.h5 --model scenario_resnet \\
    --epochs 50 --batch_size 64 --lr 2e-3 --optimizer adamw --scheduler cosine
  
  # Load from config file
  python classification_training.py --config config.json
  
  # Manual parameter configuration (interactive menu)
  python classification_training.py --manual
  
  # Generate config template
  python classification_training.py --save_config template.json
        """
    )
    
    # Configuration options
    config_group = parser.add_mutually_exclusive_group()
    config_group.add_argument('--config', type=str,
                             help='Load configuration from JSON file')
    config_group.add_argument('--manual', action='store_true',
                             help='Manual parameter configuration (interactive menu)')
    config_group.add_argument('--save_config', type=str,
                             help='Save configuration template to file and exit')
    
    # Basic arguments for quick training
    parser.add_argument('--dataset', type=str,
                       help='Path to the dataset file (.h5)')
    parser.add_argument('--model', type=str, default='scenario_cnn',
                       choices=['scenario_cnn', 'scenario_resnet', 'scenario_transformer', 'scenario_hybrid'],
                       help='Model architecture to use')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use for training')
    parser.add_argument('--save_dir', type=str, default='models/classification',
                       help='Directory to save models and results')
    
    args = parser.parse_args()
    
    # Handle special modes
    if args.save_config:
        save_config_template(args.save_config)
        return
    
    if args.manual:
        params = manual_config()
        if params is None:  # User cancelled
            return
    elif args.config:
        print(f"📖 Loading configuration from: {args.config}")
        config_data = load_config_file(args.config)
        params = ClassificationTrainingParams(**config_data)
    else:
        # Validate required arguments
        if not args.dataset:
            parser.error("--dataset is required when not using --config or --manual")
        
        # Create training parameters from command line arguments
        params = ClassificationTrainingParams(
            dataset_path=args.dataset,
            model_type=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=args.device,
            save_dir=args.save_dir
        )
    
    # Display configuration summary
    print("\n🎯 Training Configuration Summary:")
    print("=" * 50)
    print(f"📁 Dataset: {params.dataset_path}")
    print(f"🤖 Model: {params.model_type}")
    print(f"📊 Data splits: Train={params.train_split}, Val={params.val_split}, Test={params.test_split}")
    print(f"⚙️  Training: {params.epochs} epochs, batch_size={params.batch_size}, lr={params.learning_rate}")
    print(f"🔧 Optimizer: {params.optimizer}, scheduler={params.scheduler}")
    print(f"🛡️  Regularization: dropout={params.dropout_rate}, label_smoothing={params.label_smoothing}")
    print(f"💻 Hardware: {params.device}, workers={params.num_workers}")
    print(f"💾 Save to: {params.save_dir}")
    print("=" * 50)
    
    # Confirm before starting training
    if args.manual:
        confirm = input("\n▶️  Start training with these parameters? [Y/n]: ").strip().lower()
        if confirm and confirm != 'y' and confirm != 'yes':
            print("❌ Training cancelled.")
            return
    
    # Create trainer and run training
    trainer = ClassificationTrainer(params)
    results = trainer.train(params.dataset_path)
    
    print(f"\n🎉 Training completed successfully!")
    print(f"📁 Results saved to: {params.save_dir}")
    print(f"🎯 Best validation accuracy: {results['best_val_acc']:.2f}%")
    print(f"⏱️  Training time: {results['training_time']:.1f}s")


if __name__ == "__main__":
    main() 