#!/usr/bin/env python3
"""
Model Training Script with Controlled DOA Dataset

Usage:
    python scripts/train_model.py --dataset Data/datasets/basic_controlled_dataset --model cnn
    python scripts/train_model.py --dataset Data/datasets/comprehensive_controlled_dataset --model cnn --epochs 50
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import sys
from pathlib import Path
import json
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from reconunet.data.controlled_dataset_dataloader import create_dataloaders, DataLoaderConfig


class SimpleCNN(nn.Module):
    """Simple CNN for DOA estimation."""
    
    def __init__(self, num_elements=4, sequence_length=50):
        super().__init__()
        
        # Input: [batch, num_elements, sequence_length, 2] (real/imag)
        self.conv1 = nn.Conv2d(num_elements, 32, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        
        # Output: angle prediction (regression)
        self.fc = nn.Linear(128, 1)
        
    def forward(self, x):
        # x shape: [batch, num_elements, sequence_length, 2]
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        
        x = self.pool(x)  # [batch, 128, 1, 1]
        x = x.view(x.size(0), -1)  # [batch, 128]
        x = self.dropout(x)
        x = self.fc(x)  # [batch, 1]
        
        return x.squeeze(-1)  # [batch]


def create_model(model_type, num_elements=4, sequence_length=50):
    """Create model based on type."""
    if model_type == 'cnn':
        return SimpleCNN(num_elements, sequence_length)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch in dataloader:
        # Get data
        signals = batch['raw_signals'].to(device)  # [batch, elements, time, 2]
        angles = batch['angles_deg'].to(device)    # [batch]
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(signals)
        loss = criterion(predictions, angles)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            signals = batch['raw_signals'].to(device)
            angles = batch['angles_deg'].to(device)
            
            predictions = model(signals)
            loss = criterion(predictions, angles)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def main():
    parser = argparse.ArgumentParser(description='Train DOA estimation model')
    parser.add_argument('--dataset', required=True, help='Path to dataset directory')
    parser.add_argument('--model', default='cnn', choices=['cnn'], help='Model type')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', default='auto', help='Device to use (auto/cpu/cuda)')
    parser.add_argument('--experiment-name', help='Experiment name for logging')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create experiment directory
    if args.experiment_name:
        exp_name = args.experiment_name
    else:
        exp_name = f"{args.model}_{Path(args.dataset).name}_{int(time.time())}"
    
    exp_dir = Path("Data/experiments") / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    writer = SummaryWriter(exp_dir / "tensorboard")
    
    # Create dataloaders
    print("Creating dataloaders...")
    config = DataLoaderConfig(
        batch_size=args.batch_size,
        shuffle_train=True,
        num_workers=4,
        data_format='real',  # Use real format for CNN
        normalize_data=True,
        include_raw_signals=True,
        include_covariance_matrices=False,
        include_steering_vectors=False,
        include_metadata=True
    )
    
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset_path=args.dataset,
        config=config
    )
    
    print(f"Dataset loaded:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Create model
    print("Creating model...")
    model = create_model(args.model)
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Setup training
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Save configuration
    config_dict = {
        'dataset': args.dataset,
        'model': args.model,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'device': str(device),
        'total_parameters': total_params,
        'trainable_parameters': trainable_params
    }
    
    with open(exp_dir / "config.json", 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    # Training loop
    print(f"\n🚀 Starting training for {args.epochs} epochs...")
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss = validate_epoch(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Log metrics
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss: {val_loss:.6f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, exp_dir / "best_model.pth")
            print(f"  ✅ New best model saved!")
    
    # Final test evaluation
    print(f"\n📊 Final evaluation on test set...")
    test_loss = validate_epoch(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.6f}")
    
    writer.add_scalar('Loss/Test', test_loss, args.epochs)
    writer.close()
    
    print(f"\n✅ Training complete!")
    print(f"📁 Experiment saved to: {exp_dir}")
    print(f"📈 Tensorboard logs: tensorboard --logdir {exp_dir}/tensorboard")


if __name__ == "__main__":
    main() 