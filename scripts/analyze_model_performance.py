#!/usr/bin/env python3
"""
Model Analysis Script for EVD UNet Denoising Model

This script loads a trained EVD UNet model and provides comprehensive details including:
- Model architecture and parameters
- Training configuration and hyperparameters
- Model size and complexity metrics
- Layer-wise parameter breakdown
- Training history and performance metrics

Usage:
    conda activate DOA_env
    python scripts/analyze_model_performance.py
"""

import torch
import numpy as np
from pathlib import Path
import sys
from collections import OrderedDict
import argparse
import time
from typing import Dict, List, Tuple

# Add parent directory to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from reconunet.models.deep_learning.EVDUNet import EVDCovarianceReconstructionUNet


def count_parameters(model):
    """Count trainable and non-trainable parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return trainable, non_trainable


def get_layer_parameters(model):
    """Get parameter count for each layer/module."""
    layer_params = OrderedDict()
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                layer_params[name] = params
    
    return layer_params


def get_model_size_mb(model):
    """Calculate model size in MB."""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / (1024**2)
    return size_mb


def analyze_checkpoint_metadata(checkpoint):
    """Extract and display metadata from checkpoint."""
    metadata = {}
    
    # Standard checkpoint keys
    if 'epoch' in checkpoint:
        metadata['Epoch'] = checkpoint['epoch']
    if 'train_loss' in checkpoint:
        metadata['Training Loss'] = f"{checkpoint['train_loss']:.6f}"
    if 'val_loss' in checkpoint:
        metadata['Validation Loss'] = f"{checkpoint['val_loss']:.6f}"
    if 'learning_rate' in checkpoint:
        metadata['Learning Rate'] = f"{checkpoint['learning_rate']:.2e}"
    
    # Model configuration
    if 'model_config' in checkpoint:
        metadata['Model Config'] = checkpoint['model_config']
    
    # Training configuration
    if 'training_config' in checkpoint:
        metadata['Training Config'] = checkpoint['training_config']
    
    # Optimizer state info
    if 'optimizer_state_dict' in checkpoint:
        metadata['Optimizer State'] = "Available"
    
    # Scheduler state info
    if 'scheduler_state_dict' in checkpoint:
        metadata['Scheduler State'] = "Available"
    
    return metadata


def display_layer_breakdown(layer_params, top_n=20):
    """Display top N layers by parameter count."""
    sorted_layers = sorted(layer_params.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n{'='*80}")
    print(f"Top {top_n} Layers by Parameter Count")
    print(f"{'='*80}")
    print(f"{'Layer Name':<50} {'Parameters':>15} {'Percentage':>10}")
    print(f"{'-'*80}")
    
    total_params = sum(layer_params.values())
    
    for i, (name, params) in enumerate(sorted_layers[:top_n], 1):
        percentage = (params / total_params) * 100
        print(f"{i:2}. {name:<47} {params:>15,} {percentage:>9.2f}%")


def analyze_model_structure(model):
    """Analyze and display model structure details."""
    print(f"\n{'='*80}")
    print("Model Structure Analysis")
    print(f"{'='*80}")
    
    # Count different module types
    module_types = {}
    for name, module in model.named_modules():
        module_type = type(module).__name__
        if module_type not in module_types:
            module_types[module_type] = 0
        module_types[module_type] += 1
    
    print("\nModule Type Distribution:")
    print(f"{'-'*40}")
    for mod_type, count in sorted(module_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  {mod_type:<30} {count:>5}")


def benchmark_single_sample_inference(
    model: torch.nn.Module,
    input_shape: Tuple[int, int, int, int],
    device: torch.device,
    num_iterations: int = 1000,
    num_warmup: int = 10
) -> Dict[str, float]:
    """
    Benchmark single sample inference time over multiple iterations.
    
    Args:
        model: The model to benchmark
        input_shape: Shape of input tensor (batch_size, tau, 2M, M)
        device: Device to run on (cpu or cuda)
        num_iterations: Number of iterations for timing
        num_warmup: Number of warmup iterations (not timed)
        
    Returns:
        Dictionary with timing statistics (mean, std, min, max, median, throughput)
    """
    model.eval()
    model = model.to(device)
    
    # Create dummy input
    dummy_input = torch.randn(*input_shape).to(device)
    
    # Warm-up runs
    print(f"  Running {num_warmup} warm-up iterations...")
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)
    
    # Timing runs
    print(f"  Running {num_iterations} timed iterations...")
    times = []
    
    with torch.no_grad():
        for i in range(num_iterations):
            # Synchronize for accurate GPU timing
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            _ = model(dummy_input)
            
            # Synchronize again for GPU
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # Convert to milliseconds
            
            # Progress indicator every 100 iterations
            if (i + 1) % 100 == 0:
                print(f"    Completed {i + 1}/{num_iterations} iterations")
    
    # Calculate statistics
    times_array = np.array(times)
    
    stats = {
        'mean_ms': np.mean(times_array),
        'std_ms': np.std(times_array),
        'min_ms': np.min(times_array),
        'max_ms': np.max(times_array),
        'median_ms': np.median(times_array),
        'p95_ms': np.percentile(times_array, 95),
        'p99_ms': np.percentile(times_array, 99),
        'throughput_samples_per_sec': 1000.0 / np.mean(times_array),  # 1000 ms / mean_ms
        'total_time_sec': np.sum(times_array) / 1000.0
    }
    
    return stats, times_array


def display_timing_statistics(stats: Dict[str, float]):
    """Display timing statistics in a formatted way."""
    print(f"\n{'='*80}")
    print("Inference Timing Statistics")
    print(f"{'='*80}")
    print(f"  Mean:        {stats['mean_ms']:>10.4f} ms")
    print(f"  Std Dev:     {stats['std_ms']:>10.4f} ms")
    print(f"  Min:         {stats['min_ms']:>10.4f} ms")
    print(f"  Max:         {stats['max_ms']:>10.4f} ms")
    print(f"  Median:      {stats['median_ms']:>10.4f} ms")
    print(f"  95th %ile:   {stats['p95_ms']:>10.4f} ms")
    print(f"  99th %ile:   {stats['p99_ms']:>10.4f} ms")
    print(f"\n  Throughput:  {stats['throughput_samples_per_sec']:>10.2f} samples/sec")
    print(f"  Total Time:  {stats['total_time_sec']:>10.2f} seconds")


def analyze_timing_distribution(times_array: np.ndarray):
    """Provide additional analysis of timing distribution."""
    print(f"\n{'='*80}")
    print("Timing Distribution Analysis")
    print(f"{'='*80}")
    
    # Calculate coefficient of variation
    cv = (np.std(times_array) / np.mean(times_array)) * 100
    print(f"  Coefficient of Variation: {cv:.2f}%")
    
    # Check for outliers (using IQR method)
    q1 = np.percentile(times_array, 25)
    q3 = np.percentile(times_array, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = times_array[(times_array < lower_bound) | (times_array > upper_bound)]
    
    print(f"  IQR: {iqr:.4f} ms")
    print(f"  Number of outliers: {len(outliers)} ({len(outliers)/len(times_array)*100:.2f}%)")
    
    # Distribution characteristics
    print(f"\n  Distribution Characteristics:")
    print(f"    Range: {np.max(times_array) - np.min(times_array):.4f} ms")
    print(f"    25th percentile: {q1:.4f} ms")
    print(f"    50th percentile: {np.percentile(times_array, 50):.4f} ms")
    print(f"    75th percentile: {q3:.4f} ms")


def main(model_file=None):
    print("="*80)
    print("EVD UNet Denoising Model - Comprehensive Analysis")
    print("="*80)
    
    # Model path
    if model_file is None:
        model_file = "evd_unet_denoising_model_20250929_015132.pth"
    
    model_path = project_root / "notebooks" / model_file
    
    if not model_path.exists():
        print(f"\n❌ Error: Model file not found at {model_path}")
        print(f"💡 Looking for model in: {project_root / 'notebooks'}")
        return None, None
    
    print(f"\n📂 Model Path: {model_path}")
    print(f"📦 File Size: {model_path.stat().st_size / (1024**2):.2f} MB")
    
    # Load checkpoint
    print("\n⏳ Loading checkpoint...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    
    print(f"✅ Checkpoint loaded successfully")
    print(f"🖥️  Device: {device}")
    
    # Display checkpoint keys
    print(f"\n{'='*80}")
    print("Checkpoint Contents")
    print(f"{'='*80}")
    print("Available keys in checkpoint:")
    for key in checkpoint.keys():
        if isinstance(checkpoint[key], dict):
            print(f"  - {key} (dict with {len(checkpoint[key])} items)")
        elif isinstance(checkpoint[key], (int, float, str)):
            print(f"  - {key}: {checkpoint[key]}")
        else:
            print(f"  - {key} ({type(checkpoint[key]).__name__})")
    
    # Extract and display metadata
    print(f"\n{'='*80}")
    print("Training Metadata")
    print(f"{'='*80}")
    metadata = analyze_checkpoint_metadata(checkpoint)
    
    if metadata:
        for key, value in metadata.items():
            if isinstance(value, dict):
                print(f"\n{key}:")
                for k, v in value.items():
                    print(f"  {k}: {v}")
            else:
                print(f"{key}: {value}")
    else:
        print("No metadata found in checkpoint")
    
    # Initialize model
    print(f"\n{'='*80}")
    print("Model Initialization")
    print(f"{'='*80}")
    
    # Try to get model config from checkpoint, otherwise use defaults
    if 'model_config' in checkpoint:
        config = checkpoint['model_config']
        M = config.get('M', 8)
        tau = config.get('tau', 8)
    else:
        M = 8  # Default antenna elements
        tau = 8  # Default time lags
        print("⚠️  Using default configuration: M=8, tau=8")
    
    print(f"Initializing EVDCovarianceReconstructionUNet with M={M}, tau={tau}")
    model = EVDCovarianceReconstructionUNet(M=M, tau=tau)
    
    # Load model state
    print("Loading model state dict...")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✅ Model loaded and set to evaluation mode")
    
    # Model parameters analysis
    print(f"\n{'='*80}")
    print("Model Parameters Summary")
    print(f"{'='*80}")
    
    trainable, non_trainable = count_parameters(model)
    total_params = trainable + non_trainable
    
    print(f"Total Parameters:        {total_params:>15,}")
    print(f"Trainable Parameters:    {trainable:>15,}")
    print(f"Non-trainable Parameters:{non_trainable:>15,}")
    print(f"\nModel Size in Memory:    {get_model_size_mb(model):>15.2f} MB")
    
    # Layer-wise parameter breakdown
    layer_params = get_layer_parameters(model)
    display_layer_breakdown(layer_params, top_n=20)
    
    # Model structure analysis
    analyze_model_structure(model)
    
    # Input/Output specifications
    print(f"\n{'='*80}")
    print("Model I/O Specifications")
    print(f"{'='*80}")
    print(f"Input Shape:  [batch_size, {tau}, {2*M}, {M}] (autocorrelation tensor)")
    print(f"              - tau: {tau} time lags")
    print(f"              - 2M: {2*M} (real and imaginary parts concatenated)")
    print(f"              - M: {M} array elements")
    print(f"\nOutput:")
    print(f"  - Eigenvalues:   [batch_size, {M}] (real)")
    print(f"  - Eigenvectors:  [batch_size, {M}, {M}] (complex)")
    print(f"  - Reconstructed: [batch_size, {M}, {M}] (complex covariance matrix)")
    
    # Test forward pass with dummy data
    print(f"\n{'='*80}")
    print("Forward Pass Test")
    print(f"{'='*80}")
    print("Testing forward pass with dummy input...")
    
    try:
        with torch.no_grad():
            dummy_input = torch.randn(1, tau, 2*M, M).to(device)
            model = model.to(device)
            
            eigenvals, eigenvecs, reconstructed_cov = model(dummy_input)
            
            print("✅ Forward pass successful!")
            print(f"\nOutput shapes:")
            print(f"  Eigenvalues:   {list(eigenvals.shape)}")
            print(f"  Eigenvectors:  {list(eigenvecs.shape)}")
            print(f"  Reconstructed: {list(reconstructed_cov.shape)}")
            
            # Basic sanity checks
            print(f"\nSanity Checks:")
            print(f"  ✓ Eigenvalues range: [{eigenvals.min().item():.4f}, {eigenvals.max().item():.4f}]")
            print(f"  ✓ Eigenvalues sorted: {torch.all(eigenvals[:, :-1] >= eigenvals[:, 1:]).item()}")
            print(f"  ✓ Eigenvectors are complex: {eigenvecs.is_complex()}")
            print(f"  ✓ Reconstructed is complex: {reconstructed_cov.is_complex()}")
            
            # Check if eigenvectors are approximately orthogonal
            evec_conj_t = torch.conj(eigenvecs.transpose(-2, -1))
            identity_check = torch.matmul(evec_conj_t, eigenvecs)
            identity_error = torch.abs(identity_check - torch.eye(M, device=device)).mean().item()
            print(f"  ✓ Eigenvector orthogonality error: {identity_error:.6f}")
            
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Timing and Complexity Analysis
    print(f"\n{'='*80}")
    print("⏱️  Performance Benchmarking")
    print(f"{'='*80}")
    print(f"\nBenchmarking single sample inference over 1000 iterations...")
    print(f"Input shape: [1, {tau}, {2*M}, {M}]")
    print(f"Device: {device}")
    
    try:
        stats, times_array = benchmark_single_sample_inference(
            model=model,
            input_shape=(1, tau, 2*M, M),
            device=device,
            num_iterations=1000,
            num_warmup=10
        )
        
        # Display statistics
        display_timing_statistics(stats)
        
        # Display distribution analysis
        analyze_timing_distribution(times_array)
        
    except Exception as e:
        print(f"❌ Timing benchmark failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Summary
    print(f"\n{'='*80}")
    print("Analysis Complete")
    print(f"{'='*80}")
    print(f"\n📊 Summary:")
    print(f"  Model: EVDCovarianceReconstructionUNet")
    print(f"  Parameters: {total_params:,} ({get_model_size_mb(model):.2f} MB)")
    print(f"  Configuration: M={M}, tau={tau}")
    if 'stats' in locals():
        print(f"  Avg Inference Time: {stats['mean_ms']:.4f} ms")
        print(f"  Throughput: {stats['throughput_samples_per_sec']:.2f} samples/sec")
    print(f"  Status: ✅ Ready for inference")
    
    return model, checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze EVD UNet Denoising Model')
    parser.add_argument('--model', type=str, default=None,
                        help='Model filename (default: evd_unet_denoising_model_20250929_015132.pth)')
    
    args = parser.parse_args()
    model, checkpoint = main(model_file=args.model)

