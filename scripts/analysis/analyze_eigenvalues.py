#!/usr/bin/env python3
"""
Eigenvalue Analysis Script

This script visualizes and compares eigenvalues from:
1. Noisy covariance matrix (from dataset)
2. UNet reconstructed covariance matrix

This helps understand why AIC/MDL perform differently on the two covariances.
"""

import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

from models.deep_learning.EVDUNet import EVDCovarianceReconstructionUNet
from data.dataset_generator import DOADataset

# Configuration
DATASET_PATH = "Tri4Net/src/evaluation/source_estimation_validation/linear/source_est_validation_mode3_test_mode3_random/source_est_validation_mode3_test_mode3_random.h5"
UNET_MODEL_PATH = 'notebooks/evd_unet_denoising_model_20250929_015132.pth'
SNR_TO_ANALYZE = [0, 10, 20, 40]  # SNR values to analyze
NUM_SAMPLES_PER_SNR = 3  # Number of samples to show per SNR
OUTPUT_DIR = 'Tri4Net/src/evaluation/source_estimation_validation'


def estimate_num_sources_mdl(eigenvalues: np.ndarray, num_snapshots: int) -> tuple:
    """
    Estimate number of sources using MDL and return all MDL values.
    
    Returns
    -------
    tuple: (estimated_sources, mdl_values)
    """
    M = len(eigenvalues)
    T = num_snapshots
    
    mdl = np.zeros(M)
    
    for k in range(M):
        if k == M:
            mdl[k] = np.inf
        else:
            noise_eigenvals = eigenvalues[k:]
            
            if len(noise_eigenvals) > 0 and np.all(noise_eigenvals > 0):
                geometric_mean = np.prod(noise_eigenvals) ** (1.0 / len(noise_eigenvals))
                arithmetic_mean = np.mean(noise_eigenvals)
                
                if geometric_mean > 0 and arithmetic_mean > 0:
                    term1 = -T * (M - k) * np.log(geometric_mean / arithmetic_mean)
                    term2 = 0.5 * k * (2 * M - k) * np.log(T)
                    mdl[k] = term1 + term2
                else:
                    mdl[k] = np.inf
            else:
                mdl[k] = np.inf
    
    return int(np.argmin(mdl)), mdl


def load_unet_model(model_path: str, device: str = 'cpu'):
    """Load UNet model from checkpoint."""
    full_path = Path(current_dir) / model_path
    
    checkpoint = torch.load(full_path, map_location=device)
    
    if 'model_config' in checkpoint:
        config = checkpoint['model_config']
        M = config.get('num_sensors', 8)
        tau = config.get('tau', 8)
    else:
        M = 8
        tau = 8
    
    model = EVDCovarianceReconstructionUNet(M=M, tau=tau)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model


def create_autocorrelation_input(sample, tau=8, device='cpu'):
    """Convert sample to UNet-compatible autocorrelation tensor."""
    if 'autocorrelation_matrix' in sample:
        autocorr_tensor = sample['autocorrelation_matrix']
        if not isinstance(autocorr_tensor, torch.Tensor):
            autocorr_tensor = torch.tensor(autocorr_tensor, dtype=torch.float32)
        return autocorr_tensor.to(device)
    
    received_signal = sample['received_signal']
    if not isinstance(received_signal, torch.Tensor):
        received_signal = torch.tensor(received_signal, dtype=torch.complex64)
    
    received_signal = received_signal.to(device)
    M, T = received_signal.shape
    
    autocorr_tensor = torch.zeros(tau, 2*M, M, dtype=torch.float32, device=device)
    max_lag = min(tau, T - 1)
    
    for lag in range(max_lag):
        samples_available = T - lag
        if samples_available > 0:
            x1 = received_signal[:, :samples_available]
            x2 = received_signal[:, lag:lag + samples_available]
            R_lag = torch.matmul(x1, x2.conj().T) / samples_available
            autocorr_tensor[lag, :M, :] = R_lag.real
            autocorr_tensor[lag, M:, :] = R_lag.imag
    
    return autocorr_tensor


def analyze_sample(sample, unet_model, device='cpu'):
    """Analyze a single sample and return eigenvalues."""
    # Get true number of sources
    labels = sample['labels']
    all_doas = labels['doas']
    num_sources_label = labels['num_sources']
    if isinstance(num_sources_label, (list, np.ndarray)):
        true_num_sources = int(np.sum(num_sources_label))
    else:
        true_num_sources = int(num_sources_label)
    
    # Get noisy covariance
    noisy_cov = sample['covariance_matrix']
    noisy_eigenvalues = np.linalg.eigvalsh(noisy_cov)
    noisy_eigenvalues = np.sort(noisy_eigenvalues)[::-1]
    
    # Run UNet
    autocorr_input = create_autocorrelation_input(sample, tau=8, device=device)
    autocorr_input = autocorr_input.unsqueeze(0)
    
    with torch.no_grad():
        eigenvals, eigenvecs, reconstructed_cov = unet_model(autocorr_input)
    
    # Use eigenvalues directly from UNet output (already sorted in descending order)
    unet_eigenvalues = eigenvals.squeeze().cpu().numpy()
    # Ensure they are real and sorted descending
    unet_eigenvalues = np.sort(np.real(unet_eigenvalues))[::-1]
    
    # Estimate sources using MDL
    mdl_est_noisy, mdl_vals_noisy = estimate_num_sources_mdl(noisy_eigenvalues, 512)
    mdl_est_unet, mdl_vals_unet = estimate_num_sources_mdl(unet_eigenvalues, 512)
    
    # Estimate SNR from eigenvalues
    # SNR = (mean of signal eigenvalues) / (mean of noise eigenvalues)
    def estimate_snr_from_eigenvalues(eigenvalues, num_sources):
        """Estimate SNR in dB from eigenvalues."""
        if num_sources == 0 or num_sources >= len(eigenvalues):
            return 0.0
        
        signal_eigs = eigenvalues[:num_sources]
        noise_eigs = eigenvalues[num_sources:]
        
        if len(noise_eigs) == 0:
            return 0.0
        
        signal_power = np.mean(signal_eigs)
        noise_power = np.mean(noise_eigs)
        
        if noise_power <= 0:
            return 0.0
        
        snr_linear = signal_power / noise_power
        snr_db = 10 * np.log10(snr_linear)
        
        return snr_db
    
    # Estimate SNR based on true number of sources
    snr_est_noisy_true = estimate_snr_from_eigenvalues(noisy_eigenvalues, true_num_sources)
    snr_est_unet_true = estimate_snr_from_eigenvalues(unet_eigenvalues, true_num_sources)
    
    # Estimate SNR based on MDL-estimated number of sources
    snr_est_noisy_mdl = estimate_snr_from_eigenvalues(noisy_eigenvalues, mdl_est_noisy)
    snr_est_unet_mdl = estimate_snr_from_eigenvalues(unet_eigenvalues, mdl_est_unet)
    
    return {
        'true_sources': true_num_sources,
        'noisy_eigs': noisy_eigenvalues,
        'unet_eigs': unet_eigenvalues,
        'mdl_est_noisy': mdl_est_noisy,
        'mdl_est_unet': mdl_est_unet,
        'mdl_vals_noisy': mdl_vals_noisy,
        'mdl_vals_unet': mdl_vals_unet,
        'snr_est_noisy_true': snr_est_noisy_true,
        'snr_est_unet_true': snr_est_unet_true,
        'snr_est_noisy_mdl': snr_est_noisy_mdl,
        'snr_est_unet_mdl': snr_est_unet_mdl
    }


def plot_eigenvalue_comparison(results_by_snr, output_dir):
    """Create comprehensive eigenvalue comparison plots."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    n_snrs = len(results_by_snr)
    n_samples = len(results_by_snr[list(results_by_snr.keys())[0]])
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 4 * n_snrs))
    
    plot_idx = 1
    for snr_idx, (snr, samples) in enumerate(results_by_snr.items()):
        for sample_idx, result in enumerate(samples):
            ax1 = plt.subplot(n_snrs, n_samples * 2, plot_idx)
            plot_idx += 1
            
            # Plot eigenvalues
            indices = np.arange(len(result['noisy_eigs']))
            
            ax1.semilogy(indices, result['noisy_eigs'], 'b-o', linewidth=2, 
                        markersize=8, label='Noisy Cov')
            ax1.semilogy(indices, result['unet_eigs'], 'r--s', linewidth=2, 
                        markersize=8, label='UNet Cov')
            
            # Mark true number of sources
            true_src = result['true_sources']
            ax1.axvline(x=true_src - 0.5, color='green', linestyle=':', 
                       linewidth=2, label=f'True={true_src}')
            
            # Mark MDL estimates
            ax1.axvline(x=result['mdl_est_noisy'] - 0.5, color='blue', 
                       linestyle='--', linewidth=1.5, alpha=0.7, 
                       label=f'MDL(noisy)={result["mdl_est_noisy"]}')
            ax1.axvline(x=result['mdl_est_unet'] - 0.5, color='red', 
                       linestyle='--', linewidth=1.5, alpha=0.7,
                       label=f'MDL(unet)={result["mdl_est_unet"]}')
            
            ax1.set_xlabel('Eigenvalue Index', fontsize=10)
            ax1.set_ylabel('Eigenvalue (log scale)', fontsize=10)
            ax1.set_title(f'SNR={snr}dB, Sample {sample_idx+1}\nEigenvalues', 
                         fontsize=11, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend(fontsize=8)
            
            # Plot MDL criterion values
            ax2 = plt.subplot(n_snrs, n_samples * 2, plot_idx)
            plot_idx += 1
            
            k_values = np.arange(len(result['mdl_vals_noisy']))
            ax2.plot(k_values, result['mdl_vals_noisy'], 'b-o', linewidth=2,
                    markersize=6, label='Noisy Cov')
            ax2.plot(k_values, result['mdl_vals_unet'], 'r--s', linewidth=2,
                    markersize=6, label='UNet Cov')
            
            # Mark true number of sources
            ax2.axvline(x=true_src, color='green', linestyle=':', 
                       linewidth=2, label=f'True={true_src}')
            
            # Mark minima
            ax2.scatter([result['mdl_est_noisy']], [result['mdl_vals_noisy'][result['mdl_est_noisy']]],
                       s=200, c='blue', marker='*', edgecolors='black', linewidth=1.5,
                       label=f'Min(noisy)={result["mdl_est_noisy"]}', zorder=5)
            ax2.scatter([result['mdl_est_unet']], [result['mdl_vals_unet'][result['mdl_est_unet']]],
                       s=200, c='red', marker='*', edgecolors='black', linewidth=1.5,
                       label=f'Min(unet)={result["mdl_est_unet"]}', zorder=5)
            
            ax2.set_xlabel('Number of Sources (k)', fontsize=10)
            ax2.set_ylabel('MDL Criterion', fontsize=10)
            ax2.set_title(f'SNR={snr}dB, Sample {sample_idx+1}\nMDL Criterion', 
                         fontsize=11, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend(fontsize=8, loc='upper right')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_path / 'eigenvalue_comparison_detailed.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Detailed plot saved: {plot_path}")
    plt.close()


def plot_eigenvalue_ratios(results_by_snr, output_dir):
    """Plot eigenvalue ratios to show the 'gap' between signal and noise."""
    output_path = Path(output_dir)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for snr_idx, (snr, samples) in enumerate(results_by_snr.items()):
        ax = axes[snr_idx]
        
        for sample_idx, result in enumerate(samples):
            noisy_eigs = result['noisy_eigs']
            unet_eigs = result['unet_eigs']
            
            # Calculate ratios λ_i / λ_{i+1}
            noisy_ratios = [noisy_eigs[i] / noisy_eigs[i+1] for i in range(len(noisy_eigs)-1)]
            unet_ratios = [unet_eigs[i] / unet_eigs[i+1] for i in range(len(unet_eigs)-1)]
            
            ratio_indices = np.arange(len(noisy_ratios))
            
            ax.plot(ratio_indices, noisy_ratios, 'b-o', linewidth=2, alpha=0.7,
                   label=f'Noisy (Sample {sample_idx+1})' if sample_idx == 0 else '')
            ax.plot(ratio_indices, unet_ratios, 'r--s', linewidth=2, alpha=0.7,
                   label=f'UNet (Sample {sample_idx+1})' if sample_idx == 0 else '')
            
            # Mark true transition point
            true_src = result['true_sources']
            ax.axvline(x=true_src - 1, color='green', linestyle=':', 
                      linewidth=2, alpha=0.5)
        
        ax.set_xlabel('Ratio Index (λ_i / λ_{i+1})', fontsize=12)
        ax.set_ylabel('Eigenvalue Ratio', fontsize=12)
        ax.set_title(f'SNR = {snr} dB\nEigenvalue Ratios (larger = bigger gap)', 
                    fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        # Add legend for first plot only
        if snr_idx == 0:
            ax.legend(fontsize=10)
    
    plt.tight_layout()
    
    plot_path = output_path / 'eigenvalue_ratios.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Ratio plot saved: {plot_path}")
    plt.close()


def print_detailed_analysis(results_by_snr):
    """Print detailed numerical analysis."""
    print("\n" + "="*90)
    print("DETAILED EIGENVALUE ANALYSIS")
    print("="*90)
    
    for snr, samples in results_by_snr.items():
        print(f"\n{'='*90}")
        print(f"SNR = {snr} dB")
        print(f"{'='*90}")
        
        for sample_idx, result in enumerate(samples):
            print(f"\nSample {sample_idx + 1}:")
            print(f"  True # sources: {result['true_sources']}")
            print(f"  MDL estimate (noisy): {result['mdl_est_noisy']}")
            print(f"  MDL estimate (UNet):  {result['mdl_est_unet']}")
            
            # Print SNR estimates
            print(f"\n  SNR Estimation (using true # sources = {result['true_sources']}):")
            print(f"    Noisy eigenvalues → SNR: {result['snr_est_noisy_true']:.2f} dB")
            print(f"    UNet eigenvalues  → SNR: {result['snr_est_unet_true']:.2f} dB")
            
            print(f"\n  SNR Estimation (using MDL estimates):")
            print(f"    Noisy (MDL={result['mdl_est_noisy']}) → SNR: {result['snr_est_noisy_mdl']:.2f} dB")
            print(f"    UNet (MDL={result['mdl_est_unet']})  → SNR: {result['snr_est_unet_mdl']:.2f} dB")
            
            print(f"\n  Noisy eigenvalues:")
            for i, eig in enumerate(result['noisy_eigs']):
                marker = " ← Signal/Noise boundary" if i == result['true_sources'] else ""
                print(f"    λ_{i}: {eig:.6f}{marker}")
            
            print(f"\n  UNet eigenvalues:")
            for i, eig in enumerate(result['unet_eigs']):
                marker = " ← Signal/Noise boundary" if i == result['true_sources'] else ""
                print(f"    λ_{i}: {eig:.6f}{marker}")
            
            print(f"\n  Eigenvalue ratios (noisy):")
            noisy_ratios = [result['noisy_eigs'][i] / result['noisy_eigs'][i+1] 
                           for i in range(len(result['noisy_eigs'])-1)]
            for i, ratio in enumerate(noisy_ratios):
                marker = " ← Should be large gap here" if i == result['true_sources'] - 1 else ""
                print(f"    λ_{i}/λ_{i+1}: {ratio:.3f}{marker}")
            
            print(f"\n  Eigenvalue ratios (UNet):")
            unet_ratios = [result['unet_eigs'][i] / result['unet_eigs'][i+1] 
                          for i in range(len(result['unet_eigs'])-1)]
            for i, ratio in enumerate(unet_ratios):
                marker = " ← Should be large gap here" if i == result['true_sources'] - 1 else ""
                print(f"    λ_{i}/λ_{i+1}: {ratio:.3f}{marker}")


def main():
    """Main execution function."""
    print("🔍 Eigenvalue Analysis")
    print("="*60)
    
    # Load dataset
    print(f"\n📂 Loading dataset...")
    dataset_path = Path(current_dir) / DATASET_PATH
    dataset = DOADataset(str(dataset_path))
    print(f"✅ Loaded {len(dataset)} samples")
    
    # Load UNet model
    print(f"\n🤖 Loading UNet model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   Using device: {device}")
    unet_model = load_unet_model(UNET_MODEL_PATH, device)
    print(f"✅ Model loaded")
    
    # Analyze samples at different SNRs
    print(f"\n🎯 Analyzing samples at SNRs: {SNR_TO_ANALYZE}")
    
    results_by_snr = {}
    
    for snr in SNR_TO_ANALYZE:
        print(f"\n  Analyzing SNR = {snr} dB...")
        filtered_dataset = dataset.filter(snr=float(snr))
        
        samples_results = []
        for idx in range(min(NUM_SAMPLES_PER_SNR, len(filtered_dataset))):
            sample = filtered_dataset[idx]
            result = analyze_sample(sample, unet_model, device)
            samples_results.append(result)
        
        results_by_snr[snr] = samples_results
        print(f"    Processed {len(samples_results)} samples")
    
    # Print detailed analysis
    print_detailed_analysis(results_by_snr)
    
    # Create plots
    print(f"\n📊 Creating visualizations...")
    output_dir = Path(current_dir) / OUTPUT_DIR
    plot_eigenvalue_comparison(results_by_snr, output_dir)
    plot_eigenvalue_ratios(results_by_snr, output_dir)
    
    print(f"\n✅ Analysis complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
