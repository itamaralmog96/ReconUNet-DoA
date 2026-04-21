#!/usr/bin/env python3
"""
Source Number Estimation Validation Script

This script validates different methods for estimating the number of sources:
1. AIC (Akaike Information Criterion) - applied to covariance matrix eigenvalues
2. MDL (Minimum Description Length) - applied to covariance matrix eigenvalues
3. UNet EVD - estimate from UNet eigenvalue decomposition output

The script generates a test dataset with controlled source configurations and
evaluates t        # Evaluate samples
        debug_eigenvalues = {'noisy': [], 'unet': []}
        for idx in tqdm(range(num_samples), desc=f"SNR {snr_db}dB"):
            sample = filtered_dataset[idx]
            result = evaluate_sample(sample, array_model, unet_model, 
                                    NUM_SNAPSHOTS, device)
            
            # Collect debug eigenvalues from first 10 samples
            if idx < 3:  # Just collect first 3 for debugging
                if '_debug_noisy_eig' in result:
                    debug_eigenvalues['noisy'].append(result['_debug_noisy_eig'])
                    debug_eigenvalues['unet'].append(result['_debug_unet_eig'])
            
            for alg in results_storage.keys():
                if alg in result:
                    results_storage[alg].append(result[alg])
        
        # Print debug info for 0 dB SNR
        if snr_db == 0:
            print(f"\n{'='*80}")
            print(f"DEBUG: Eigenvalue Analysis at SNR={snr_db}dB")
            print(f"Debug eigenvalues collected: {len(debug_eigenvalues['noisy'])} samples")
            print(f"{'='*80}")
            if debug_eigenvalues['noisy']:
                for i in range(min(3, len(debug_eigenvalues['noisy']))):
                    print(f"\nSample {i+1}:")
                    print(f"  True # sources: {results_storage['True'][i]}")
                    noisy_eigs = debug_eigenvalues['noisy'][i]
                    unet_eigs = debug_eigenvalues['unet'][i]
                    print(f"  Noisy eigenvalues: [{', '.join([f'{e:.4f}' for e in noisy_eigs])}]")
                    print(f"  UNet eigenvalues:  [{', '.join([f'{e:.4f}' for e in unet_eigs])}]")
                    print(f"  Eigenvalue ratios (noisy): [{', '.join([f'{noisy_eigs[j]/noisy_eigs[j+1]:.2f}' for j in range(len(noisy_eigs)-1)])}]")
                    print(f"  Eigenvalue ratios (unet):  [{', '.join([f'{unet_eigs[j]/unet_eigs[j+1]:.2f}' for j in range(len(unet_eigs)-1)])}]")
                    print(f"  AIC_noisy: {results_storage['AIC_noisy'][i]}, MDL_noisy: {results_storage['MDL_noisy'][i]}")
                    print(f"  AIC_unet:  {results_storage['AIC_unet'][i]}, MDL_unet:  {results_storage['MDL_unet'][i]}")
            else:
                print("  No debug eigenvalues collected!")
            print(f"{'='*80}\n")
        
        results_by_snr[snr_db] = results_storage of each method across different SNR levels.

Usage:
    1. Configure parameters below (similar to controlled_angle_evaluation.py)
    2. Run: python source_estimation_validation.py
"""

import os
import sys

# Fix OpenMP library conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
from tqdm import tqdm
import warnings
import time
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION SECTION
# ============================================================================

# === MODE SELECTION ===
EVALUATION_MODE = 3  # 1-4 (same as controlled_angle_evaluation.py)

# === ANGLE CONFIGURATION ===
REFERENCE_ANGLES = [90.0]  # For Mode 1
ANGLE_RANGE = (30.0, 150.0)  # For Modes 2, 3, 4
ANGLE_DELTA = 10.0  # For Modes 1, 2

# === SNR CONFIGURATION ===
SNR_DB = list(range(-20, 10, 5))  # SNR levels to test
SAMPLES_PER_SNR = 100  # Number of samples per SNR level

# === SOURCE CONFIGURATION ===
NUM_SOURCES = [1, 0]  # [main_sources, interference_sources] = 3 total sources
NUM_SNAPSHOTS = 4096
NUM_MULTIPATH = 0  # Number of multipath components

# === ARRAY CONFIGURATION ===
ARRAY_TYPE = "linear"
NUM_ELEMENTS = 8
CARRIER_FREQ = 2.45e9
SAMPLING_FREQ = 1e6
ARRAY_IMPERFECTIONS = False

# === UNET MODEL ===
UNET_MODEL_PATH = 'notebooks/evd_unet_denoising_model_20250929_015132.pth'
AUTOCORR_TAU = 8

# === OUTPUT ===
OUTPUT_DIR = 'Tri4Net/src/evaluation/source_estimation_validation'
DATASET_NAME = f"source_est_validation_mode{EVALUATION_MODE}"

# === DATASET GENERATION ===
DATASET_GENERATION_SEED = int(time.time())
MAX_SAMPLES_PER_SNR = 1000

# ============================================================================
# END OF CONFIGURATION
# ============================================================================

# Add src to path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

# Import dataset infrastructure
from data.dataset_generator import SimpleDatasetConfig, ControlledDatasetGenerator, DOADataset
from signalgen import ArrayConfig, ArrayModel
from models.deep_learning.EVDUNet import EVDCovarianceReconstructionUNet


def estimate_num_sources_mdl(eigenvalues: np.ndarray, num_snapshots: int) -> int:
    """
    Estimate number of sources using MDL (Minimum Description Length) criterion.
    
    MDL formula:
    MDL(k) = -T(M-k) * log[prod(λ_i)^(1/(M-k)) / mean(λ_i)] + 0.5*k*(2M-k)*log(T)
    
    where:
    - T = number of snapshots
    - M = number of array elements
    - k = number of sources (hypothesis)
    - λ_i = eigenvalues for i >= k (noise eigenvalues)
    
    Parameters
    ----------
    eigenvalues : np.ndarray
        Eigenvalues sorted in descending order
    num_snapshots : int
        Number of time snapshots used to compute covariance
        
    Returns
    -------
    int
        Estimated number of sources
    """
    M = len(eigenvalues)
    T = num_snapshots
    
    mdl = np.zeros(M)
    
    for k in range(M):
        if k == M:  # All eigenvalues are noise
            mdl[k] = np.inf
        else:
            # Get noise eigenvalues (from k to end)
            noise_eigenvals = eigenvalues[k:]
            
            if len(noise_eigenvals) > 0 and np.all(noise_eigenvals > 0):
                # Geometric mean
                geometric_mean = np.prod(noise_eigenvals) ** (1.0 / len(noise_eigenvals))
                # Arithmetic mean
                arithmetic_mean = np.mean(noise_eigenvals)
                
                if geometric_mean > 0 and arithmetic_mean > 0:
                    # MDL formula
                    term1 = -T * (M - k) * np.log(geometric_mean / arithmetic_mean)
                    term2 = 0.5 * k * (2 * M - k) * np.log(T)
                    mdl[k] = term1 + term2
                else:
                    mdl[k] = np.inf
            else:
                mdl[k] = np.inf
    
    # Return k that minimizes MDL
    return int(np.argmin(mdl))


def estimate_num_sources_aic(eigenvalues: np.ndarray, num_snapshots: int) -> int:
    """
    Estimate number of sources using AIC (Akaike Information Criterion).
    
    AIC formula:
    AIC(k) = -2*T*(M-k) * log[prod(λ_i)^(1/(M-k)) / mean(λ_i)] + 2*k*(2M-k)
    
    Parameters
    ----------
    eigenvalues : np.ndarray
        Eigenvalues sorted in descending order
    num_snapshots : int
        Number of time snapshots used to compute covariance
        
    Returns
    -------
    int
        Estimated number of sources
    """
    M = len(eigenvalues)
    T = num_snapshots
    
    aic = np.zeros(M)
    
    for k in range(M):
        if k == M:
            aic[k] = np.inf
        else:
            noise_eigenvals = eigenvalues[k:]
            
            if len(noise_eigenvals) > 0 and np.all(noise_eigenvals > 0):
                geometric_mean = np.prod(noise_eigenvals) ** (1.0 / len(noise_eigenvals))
                arithmetic_mean = np.mean(noise_eigenvals)
                
                if geometric_mean > 0 and arithmetic_mean > 0:
                    # AIC formula
                    term1 = -2 * T * (M - k) * np.log(geometric_mean / arithmetic_mean)
                    term2 = 2 * k * (2 * M - k)
                    aic[k] = term1 + term2
                else:
                    aic[k] = np.inf
            else:
                aic[k] = np.inf
    
    # Return k that minimizes AIC
    return int(np.argmin(aic))


def estimate_num_sources_from_unet_evd(eigenvalues: np.ndarray, threshold_ratio: float = 0.1) -> int:
    """
    Estimate number of sources from UNet EVD output eigenvalues.
    
    Strategy: Find the "elbow" in eigenvalue spectrum where signal subspace ends.
    Uses a threshold based on the ratio between consecutive eigenvalues.
    
    Parameters
    ----------
    eigenvalues : np.ndarray
        Eigenvalues from UNet EVD output (sorted descending)
    threshold_ratio : float
        Threshold for eigenvalue drop ratio to detect signal/noise boundary
        
    Returns
    -------
    int
        Estimated number of sources
    """
    M = len(eigenvalues)
    
    if M <= 1:
        return 0
    
    # Ensure eigenvalues are sorted descending
    eigenvalues = np.sort(eigenvalues)[::-1]
    
    # Method 1: Look for largest eigenvalue drop
    ratios = eigenvalues[:-1] / (eigenvalues[1:] + 1e-10)  # Avoid division by zero
    
    # Find the largest ratio (biggest drop)
    max_ratio_idx = np.argmax(ratios)
    
    # Check if this drop is significant
    if ratios[max_ratio_idx] > (1.0 + threshold_ratio):
        num_sources_ratio = max_ratio_idx + 1
    else:
        num_sources_ratio = 0
    
    # Method 2: Threshold based on eigenvalue magnitude
    # Assume noise floor is approximately the mean of the smallest eigenvalues
    noise_floor = np.mean(eigenvalues[-M//2:]) if M > 2 else eigenvalues[-1]
    threshold = noise_floor * (1.0 + threshold_ratio)
    num_sources_threshold = np.sum(eigenvalues > threshold)
    
    # Combine both methods (prefer ratio method if it finds something)
    if num_sources_ratio > 0:
        return min(num_sources_ratio, M - 1)
    else:
        return min(num_sources_threshold, M - 1)


def estimate_num_sources_ratio_threshold(eigenvalues: np.ndarray, 
                                        threshold: float = 2.5) -> int:
    """
    Estimate number of sources using eigenvalue ratio thresholding.
    
    Designed for UNet eigenvalues which have gradually decaying noise floor.
    Finds the index where λ_i / λ_{i+1} > threshold.
    
    Parameters
    ----------
    eigenvalues : np.ndarray
        Eigenvalues sorted in descending order
    threshold : float, optional
        Ratio threshold (default: 2.5)
        - For low SNR: use 2.0-2.5
        - For high SNR: use 3.0-5.0
        
    Returns
    -------
    int
        Estimated number of sources
    """
    if len(eigenvalues) < 2:
        return 0
    
    # Compute ratios λ_i / λ_{i+1}
    ratios = eigenvalues[:-1] / (eigenvalues[1:] + 1e-10)
    
    # Find first ratio exceeding threshold
    exceeds = np.where(ratios >= threshold)[0]
    
    if len(exceeds) > 0:
        # Return index after the largest drop
        return int(exceeds[0] + 1)  # +1 because ratio at index i is λ_i/λ_{i+1}
    else:
        return 0  # No significant drop found


def estimate_num_sources_adaptive_threshold(eigenvalues: np.ndarray) -> int:
    """
    Adaptively select threshold based on eigenvalue statistics.
    
    Designed for UNet eigenvalues. Uses the median ratio as a baseline 
    and looks for ratios significantly above this baseline.
    
    Parameters
    ----------
    eigenvalues : np.ndarray
        Eigenvalues sorted in descending order
        
    Returns
    -------
    int
        Estimated number of sources
    """
    if len(eigenvalues) < 2:
        return 0
    
    # Compute ratios
    ratios = eigenvalues[:-1] / (eigenvalues[1:] + 1e-10)
    
    # Compute adaptive threshold as median + k*MAD (median absolute deviation)
    median_ratio = np.median(ratios)
    mad = np.median(np.abs(ratios - median_ratio))
    adaptive_threshold = median_ratio + 2.0 * mad  # 2.0 is tunable
    
    # Find first ratio exceeding adaptive threshold
    exceeds = np.where(ratios >= adaptive_threshold)[0]
    
    if len(exceeds) > 0:
        return int(exceeds[0] + 1)
    else:
        return 0


def create_autocorrelation_input_for_unet(sample, tau=8, device='cpu'):
    """Convert sample to UNet-compatible autocorrelation tensor."""
    if 'autocorrelation_matrix' in sample:
        autocorr_tensor = sample['autocorrelation_matrix']
        if not isinstance(autocorr_tensor, torch.Tensor):
            autocorr_tensor = torch.tensor(autocorr_tensor, dtype=torch.float32)
        return autocorr_tensor.to(device)
    
    # Compute from received signal if not pre-computed
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


def load_unet_model(model_path: str, device: str = 'cpu') -> Optional[EVDCovarianceReconstructionUNet]:
    """Load UNet model from checkpoint."""
    full_path = Path(current_dir) / model_path
    
    if not full_path.exists():
        print(f"❌ UNet model not found at: {full_path}")
        return None
    
    try:
        checkpoint = torch.load(full_path, map_location=device)
        
        # Extract model configuration
        if 'model_config' in checkpoint:
            config = checkpoint['model_config']
            M = config.get('num_sensors', 8)
            tau = config.get('tau', 8)
        else:
            M = 8
            tau = 8
        
        # Initialize model
        model = EVDCovarianceReconstructionUNet(M=M, tau=tau)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        print(f"✅ Loaded UNet model: M={M}, tau={tau}")
        return model
    
    except Exception as e:
        print(f"❌ Failed to load UNet model: {e}")
        return None


def evaluate_sample(sample: Dict, array_model: ArrayModel, unet_model, 
                   num_snapshots: int, device: str = 'cpu') -> Dict[str, int]:
    """
    Evaluate a single sample with all source estimation methods.
    
    Returns
    -------
    dict
        Dictionary with keys: 'AIC_noisy', 'MDL_noisy', 'AIC_unet', 'MDL_unet', 'True'
    """
    results = {}
    
    # Get true number of sources from labels
    labels = sample['labels']
    all_doas = labels['doas']
    
    # Determine total number of sources (including multipath if present)
    if 'num_sources' in labels:
        num_sources_label = labels['num_sources']
        if isinstance(num_sources_label, (list, np.ndarray)):
            # num_sources_label = [main_sources, interference_sources]
            num_total_sources = int(np.sum(num_sources_label))  # Total = main + interference
        else:
            num_total_sources = int(num_sources_label)
    else:
        num_total_sources = len(all_doas[~np.isnan(all_doas)])
    
    # Add multipath components to the count
    num_multipath = int(labels.get('num_multipath', 0))
    true_num_signals = num_total_sources # Total signals in the array (sources + multipath)
    
    results['True'] = true_num_signals
    
    # Get NOISY covariance matrix from dataset
    if 'covariance_matrix' in sample:
        noisy_cov_matrix = sample['covariance_matrix']
    else:
        # Compute from received signal
        received_signal = sample['received_signal']
        noisy_cov_matrix = np.dot(received_signal, received_signal.conj().T) / received_signal.shape[1]
    
    # Compute eigenvalues of NOISY covariance
    noisy_eigenvalues = np.linalg.eigvalsh(noisy_cov_matrix)
    noisy_eigenvalues = np.sort(noisy_eigenvalues)[::-1]  # Sort descending
    
    # Method 1: AIC on noisy covariance
    try:
        results['AIC_noisy'] = estimate_num_sources_aic(noisy_eigenvalues, num_snapshots)
    except Exception as e:
        results['AIC_noisy'] = -1  # Error flag
    
    # Method 2: MDL on noisy covariance
    try:
        results['MDL_noisy'] = estimate_num_sources_mdl(noisy_eigenvalues, num_snapshots)
    except Exception as e:
        results['MDL_noisy'] = -1  # Error flag
    
    # Method 3 & 4: AIC/MDL on UNet reconstructed covariance
    if unet_model is not None:
        try:
            # Prepare input
            autocorr_input = create_autocorrelation_input_for_unet(sample, tau=AUTOCORR_TAU, device=device)
            autocorr_input = autocorr_input.unsqueeze(0)  # Add batch dimension
            
            # Run UNet to get reconstructed covariance
            with torch.no_grad():
                unet_output = unet_model(autocorr_input)
            
            # Extract eigenvalues directly from UNet output
            # UNet returns tuple: (eigenvals, eigenvecs, reconstructed_cov)
            if isinstance(unet_output, tuple):
                eigenvals_tensor, eigenvecs, reconstructed_cov_tensor = unet_output
                # Use eigenvalues directly from UNet (already sorted in descending order)
                unet_eigenvalues = eigenvals_tensor.squeeze().cpu().numpy()
                # Ensure they are real and sorted descending
                unet_eigenvalues = np.sort(np.real(unet_eigenvalues))[::-1]
            else:
                # Fallback: compute from reconstructed covariance
                reconstructed_cov = unet_output.squeeze().cpu().numpy()
                # If output needs complex conversion from real/imag stacked format
                if reconstructed_cov.dtype != np.complex64 and reconstructed_cov.dtype != np.complex128:
                    # Check if it's stacked real/imag
                    if reconstructed_cov.ndim == 3 and reconstructed_cov.shape[0] == 2:
                        # Shape: (2, M, M) where [0] is real, [1] is imaginary
                        reconstructed_cov = reconstructed_cov[0] + 1j * reconstructed_cov[1]
                
                # Ensure Hermitian symmetry
                reconstructed_cov = (reconstructed_cov + reconstructed_cov.conj().T) / 2
                
                # Compute eigenvalues of RECONSTRUCTED covariance
                unet_eigenvalues = np.linalg.eigvalsh(reconstructed_cov)
                unet_eigenvalues = np.sort(np.real(unet_eigenvalues))[::-1]  # Sort descending, take real part
            
            # Store eigenvalues for debugging
            results['_debug_noisy_eig'] = noisy_eigenvalues
            results['_debug_unet_eig'] = unet_eigenvalues
            
            # Apply AIC on UNet reconstructed covariance (will fail, but keep for comparison)
            try:
                results['AIC_unet'] = estimate_num_sources_aic(unet_eigenvalues, num_snapshots)
            except Exception as e:
                results['AIC_unet'] = -1
            
            # Apply MDL on UNet reconstructed covariance (will fail, but keep for comparison)
            try:
                results['MDL_unet'] = estimate_num_sources_mdl(unet_eigenvalues, num_snapshots)
            except Exception as e:
                results['MDL_unet'] = -1
            
            # NEW METHODS ADAPTED FOR UNET EIGENVALUES:
            
            # Method 5: Ratio Threshold (works well with UNet)
            try:
                results['Ratio_unet'] = estimate_num_sources_ratio_threshold(unet_eigenvalues, threshold=2.5)
            except Exception as e:
                results['Ratio_unet'] = -1
            
            # Method 6: Adaptive Threshold (works well with UNet)
            try:
                results['Adaptive_unet'] = estimate_num_sources_adaptive_threshold(unet_eigenvalues)
            except Exception as e:
                results['Adaptive_unet'] = -1
            
            # SNR ESTIMATION from eigenvalues
            def estimate_snr_from_eigenvalues(eigenvalues, num_sources):
                """Estimate SNR in dB from eigenvalues."""
                if num_sources <= 0 or num_sources >= len(eigenvalues):
                    return np.nan
                
                signal_eigs = eigenvalues[:num_sources]
                noise_eigs = eigenvalues[num_sources:]
                
                if len(noise_eigs) == 0:
                    return np.nan
                
                signal_power = np.mean(signal_eigs)
                noise_power = np.mean(noise_eigs)
                
                if noise_power <= 0:
                    return np.nan
                
                snr_linear = signal_power / noise_power
                snr_db = 10 * np.log10(snr_linear)
                
                return snr_db
            
            # Estimate SNR using different methods
            # Based on true number of sources
            results['SNR_est_noisy_true'] = estimate_snr_from_eigenvalues(noisy_eigenvalues, true_num_signals)
            results['SNR_est_unet_true'] = estimate_snr_from_eigenvalues(unet_eigenvalues, true_num_signals)
            
            # Based on Adaptive threshold estimate (best performing method)
            if results['Adaptive_unet'] > 0:
                results['SNR_est_unet_adaptive'] = estimate_snr_from_eigenvalues(unet_eigenvalues, results['Adaptive_unet'])
            else:
                results['SNR_est_unet_adaptive'] = np.nan
            
            # Based on Ratio threshold estimate
            if results['Ratio_unet'] > 0:
                results['SNR_est_unet_ratio'] = estimate_snr_from_eigenvalues(unet_eigenvalues, results['Ratio_unet'])
            else:
                results['SNR_est_unet_ratio'] = np.nan
            
            # Based on MDL estimate (noisy)
            if results['MDL_noisy'] > 0:
                results['SNR_est_noisy_mdl'] = estimate_snr_from_eigenvalues(noisy_eigenvalues, results['MDL_noisy'])
            else:
                results['SNR_est_noisy_mdl'] = np.nan
            
        except Exception as e:
            print(f"⚠️ UNet evaluation failed: {e}")
            results['AIC_unet'] = -1
            results['MDL_unet'] = -1
            results['Ratio_unet'] = -1
            results['Adaptive_unet'] = -1
    else:
        results['AIC_unet'] = -1
        results['MDL_unet'] = -1
        results['Ratio_unet'] = -1
        results['Adaptive_unet'] = -1
    
    return results


def evaluate_dataset(dataset: DOADataset, array_model: ArrayModel, 
                     unet_model, snr_levels: List[float]) -> Dict:
    """
    Evaluate all samples in the dataset.
    
    Returns
    -------
    dict
        Nested dictionary: {snr: {method: [estimates]}}
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    results_by_snr = {}
    
    for snr_db in snr_levels:
        print(f"\n🎯 Evaluating SNR = {snr_db} dB")
        
        # Filter dataset by SNR
        filtered_dataset = dataset.filter(snr=snr_db)
        num_samples = min(len(filtered_dataset), MAX_SAMPLES_PER_SNR)
        
        if num_samples == 0:
            continue
        
        print(f"   Processing {num_samples} samples...")
        
        # Initialize storage
        results_storage = {
            'AIC_noisy': [],
            'MDL_noisy': [],
            'AIC_unet': [],
            'MDL_unet': [],
            'Ratio_unet': [],
            'Adaptive_unet': [],
            'True': [],
            'SNR_est_noisy_true': [],
            'SNR_est_unet_true': [],
            'SNR_est_unet_adaptive': [],
            'SNR_est_unet_ratio': [],
            'SNR_est_noisy_mdl': []
        }
        
        # Evaluate samples
        for idx in tqdm(range(num_samples), desc=f"SNR {snr_db}dB"):
            sample = filtered_dataset[idx]
            result = evaluate_sample(sample, array_model, unet_model, 
                                    NUM_SNAPSHOTS, device)
            
            for method in results_storage.keys():
                if method in result:
                    results_storage[method].append(result[method])
        
        results_by_snr[snr_db] = results_storage
    
    return results_by_snr


def calculate_accuracy(results_by_snr: Dict) -> Dict:
    """
    Calculate accuracy for each method at each SNR level.
    
    Returns
    -------
    dict
        {snr: {method: accuracy}}
    """
    accuracy_by_snr = {}
    
    for snr_db, results in results_by_snr.items():
        true_values = np.array(results['True'])
        accuracies = {}
        
        for method in ['AIC_noisy', 'MDL_noisy', 'AIC_unet', 'MDL_unet', 'Ratio_unet', 'Adaptive_unet']:
            if method in results:
                estimates = np.array(results[method])
                # Filter out error cases (-1)
                valid_mask = estimates >= 0
                if np.sum(valid_mask) > 0:
                    correct = np.sum(estimates[valid_mask] == true_values[valid_mask])
                    total = np.sum(valid_mask)
                    accuracies[method] = (correct / total) * 100.0
                else:
                    accuracies[method] = 0.0
        
        accuracy_by_snr[snr_db] = accuracies
    
    return accuracy_by_snr


def plot_results(accuracy_by_snr: Dict, output_dir: str):
    """Plot accuracy vs SNR for all methods."""
    print("\n📈 Creating Plots...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    snr_vals = sorted(accuracy_by_snr.keys())
    
    plt.figure(figsize=(14, 8))
    
    # Plot each method
    methods = {
        'AIC_noisy': {'color': 'blue', 'marker': 'o', 'linestyle': '-', 'label': 'AIC (Noisy Cov)'},
        'MDL_noisy': {'color': 'green', 'marker': 's', 'linestyle': '-', 'label': 'MDL (Noisy Cov)'},
        'AIC_unet': {'color': 'red', 'marker': '^', 'linestyle': '--', 'label': 'AIC (UNet Cov) [fails]'},
        'MDL_unet': {'color': 'orange', 'marker': 'd', 'linestyle': '--', 'label': 'MDL (UNet Cov) [fails]'},
        'Ratio_unet': {'color': 'purple', 'marker': 'v', 'linestyle': '-', 'label': 'Ratio Threshold (UNet)'},
        'Adaptive_unet': {'color': 'cyan', 'marker': 'p', 'linestyle': '-', 'label': 'Adaptive Threshold (UNet)'}
    }
    
    for method, style in methods.items():
        accuracies = []
        for snr in snr_vals:
            if method in accuracy_by_snr[snr]:
                accuracies.append(accuracy_by_snr[snr][method])
            else:
                accuracies.append(0.0)
        
        plt.plot(snr_vals, accuracies, 
                marker=style['marker'], 
                color=style['color'],
                linestyle=style['linestyle'],
                linewidth=2,
                markersize=8,
                label=style['label'])
    
    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Source Number Estimation Accuracy vs SNR', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.ylim([0, 105])
    
    # Save plot
    plot_path = output_path / f"source_estimation_accuracy_{DATASET_NAME}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved: {plot_path}")
    
    plt.close()


def print_results_summary(accuracy_by_snr: Dict, results_by_snr: Dict):
    """Print detailed results summary."""
    print("\n" + "="*80)
    print("📊 Source Estimation Accuracy Summary")
    print("="*80)
    
    snr_vals = sorted(accuracy_by_snr.keys())
    
    # Print header
    print(f"{'SNR (dB)':<10} {'True #':<8} {'AIC_noisy':<12} {'MDL_noisy':<12} {'AIC_unet':<12} {'MDL_unet':<12} {'Ratio_unet':<12} {'Adaptive_unet':<12}")
    print("-"*120)
    
    for snr in snr_vals:
        true_num = int(np.mean(results_by_snr[snr]['True']))
        aic_noisy = accuracy_by_snr[snr].get('AIC_noisy', 0.0)
        mdl_noisy = accuracy_by_snr[snr].get('MDL_noisy', 0.0)
        aic_unet = accuracy_by_snr[snr].get('AIC_unet', 0.0)
        mdl_unet = accuracy_by_snr[snr].get('MDL_unet', 0.0)
        ratio_unet = accuracy_by_snr[snr].get('Ratio_unet', 0.0)
        adaptive_unet = accuracy_by_snr[snr].get('Adaptive_unet', 0.0)
        
        print(f"{snr:<10.1f} {true_num:<8} {aic_noisy:<11.1f}% {mdl_noisy:<11.1f}% {aic_unet:<11.1f}% {mdl_unet:<11.1f}% {ratio_unet:<11.1f}% {adaptive_unet:<11.1f}%")
    
    print("-"*120)
    
    # Overall statistics
    print("\n" + "="*100)
    print("📈 Overall Statistics")
    print("="*100)
    
    all_aic_noisy = [accuracy_by_snr[snr].get('AIC_noisy', 0.0) for snr in snr_vals]
    all_mdl_noisy = [accuracy_by_snr[snr].get('MDL_noisy', 0.0) for snr in snr_vals]
    all_aic_unet = [accuracy_by_snr[snr].get('AIC_unet', 0.0) for snr in snr_vals]
    all_mdl_unet = [accuracy_by_snr[snr].get('MDL_unet', 0.0) for snr in snr_vals]
    all_ratio_unet = [accuracy_by_snr[snr].get('Ratio_unet', 0.0) for snr in snr_vals]
    all_adaptive_unet = [accuracy_by_snr[snr].get('Adaptive_unet', 0.0) for snr in snr_vals]
    
    print(f"AIC (Noisy Cov)     - Mean: {np.mean(all_aic_noisy):.1f}%, Std: {np.std(all_aic_noisy):.1f}%")
    print(f"MDL (Noisy Cov)     - Mean: {np.mean(all_mdl_noisy):.1f}%, Std: {np.std(all_mdl_noisy):.1f}%")
    print(f"AIC (UNet Cov)      - Mean: {np.mean(all_aic_unet):.1f}%, Std: {np.std(all_aic_unet):.1f}%")
    print(f"MDL (UNet Cov)      - Mean: {np.mean(all_mdl_unet):.1f}%, Std: {np.std(all_mdl_unet):.1f}%")
    print(f"Ratio (UNet) ✓      - Mean: {np.mean(all_ratio_unet):.1f}%, Std: {np.std(all_ratio_unet):.1f}%")
    print(f"Adaptive (UNet) ✓   - Mean: {np.mean(all_adaptive_unet):.1f}%, Std: {np.std(all_adaptive_unet):.1f}%")
    
    # SNR Estimation Summary
    print("\n" + "="*100)
    print("📡 SNR Estimation Summary (using eigenvalue-based method)")
    print("="*100)
    print(f"{'True SNR':<12} {'Noisy(true#)':<15} {'UNet(true#)':<15} {'Noisy(MDL)':<15} {'UNet(Adaptive)':<15} {'UNet(Ratio)':<15}")
    print("-"*100)
    
    for snr in snr_vals:
        # Compute mean SNR estimates for this SNR level
        snr_est_noisy_true = np.array(results_by_snr[snr]['SNR_est_noisy_true'])
        snr_est_unet_true = np.array(results_by_snr[snr]['SNR_est_unet_true'])
        snr_est_noisy_mdl = np.array(results_by_snr[snr]['SNR_est_noisy_mdl'])
        snr_est_unet_adaptive = np.array(results_by_snr[snr]['SNR_est_unet_adaptive'])
        snr_est_unet_ratio = np.array(results_by_snr[snr]['SNR_est_unet_ratio'])
        
        # Remove NaN values and compute mean
        mean_noisy_true = np.nanmean(snr_est_noisy_true) if len(snr_est_noisy_true) > 0 else np.nan
        mean_unet_true = np.nanmean(snr_est_unet_true) if len(snr_est_unet_true) > 0 else np.nan
        mean_noisy_mdl = np.nanmean(snr_est_noisy_mdl) if len(snr_est_noisy_mdl) > 0 else np.nan
        mean_unet_adaptive = np.nanmean(snr_est_unet_adaptive) if len(snr_est_unet_adaptive) > 0 else np.nan
        mean_unet_ratio = np.nanmean(snr_est_unet_ratio) if len(snr_est_unet_ratio) > 0 else np.nan
        
        print(f"{snr:<12.1f} {mean_noisy_true:<15.2f} {mean_unet_true:<15.2f} {mean_noisy_mdl:<15.2f} {mean_unet_adaptive:<15.2f} {mean_unet_ratio:<15.2f}")
    
    print("-"*100)
    print("\nNote: SNR estimation formula: 10*log10(mean(signal_eigenvalues) / mean(noise_eigenvalues))")
    print("      'true#' uses true number of sources, 'MDL'/'Adaptive'/'Ratio' use estimated number")


def main():
    """Main execution function."""
    print("🚀 Source Number Estimation Validation")
    print("="*60)
    
    # Print configuration
    total_sources = sum(NUM_SOURCES)
    print(f"\n📋 Configuration:")
    print(f"   Mode: {EVALUATION_MODE}")
    print(f"   SNR Range: {SNR_DB}")
    print(f"   Samples per SNR: {SAMPLES_PER_SNR}")
    print(f"   Sources: {NUM_SOURCES[0]} main + {NUM_SOURCES[1]} interference = {total_sources} total")
    print(f"   Multipath: {NUM_MULTIPATH} component(s)")
    print(f"   Array: {NUM_ELEMENTS} element {ARRAY_TYPE}")
    
    # Create array model
    array_config = ArrayConfig(
        array_type=ARRAY_TYPE,
        num_elements=NUM_ELEMENTS,
        element_spacing=0.5,
        carrier_freq=CARRIER_FREQ
    )
    
    # Disable imperfections if requested
    if not ARRAY_IMPERFECTIONS:
        array_config.enable_gain_phase_errors = False
        array_config.enable_mutual_coupling = False
        array_config.position_error_std = 0.0
    
    array_model = ArrayModel(array_config)
    
    # Generate dataset
    print("\n🔧 Generating test dataset...")
    dataset_config = SimpleDatasetConfig(
        angles_deg=[90.0],  # Placeholder, will be overridden by generate_test_dataset
        snr_db=SNR_DB,
        num_snapshots=[NUM_SNAPSHOTS],
        num_sources=[NUM_SOURCES],
        sir_db=[0.0],
        num_multipath=[NUM_MULTIPATH],
        array_imperfections=[ARRAY_IMPERFECTIONS],
        random_sampling_mode=False,
        examples_per_combination=1,
        dataset_name=DATASET_NAME,
        array_type=ARRAY_TYPE,
        num_elements=NUM_ELEMENTS,
        carrier_freq=CARRIER_FREQ,
        sampling_freq=SAMPLING_FREQ,
        save_received_signal=True,
        save_covariance_matrix=True,
        save_clean_covariance_matrix=False,
        save_autocorrelation_matrix=True,
        save_music_estimates=False,
        save_steering_vectors=True,
        save_array_metadata=False,
        autocorr_tau=AUTOCORR_TAU
    )
    
    # Create generator with correct parameter order
    generator = ControlledDatasetGenerator(dataset_config, seed=DATASET_GENERATION_SEED)
    
    output_dir_path = Path(current_dir) / OUTPUT_DIR
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Generate dataset based on mode
    if EVALUATION_MODE == 1:
        dataset_path = generator.generate_test_dataset(
            reference_angles=REFERENCE_ANGLES,
            angle_delta=ANGLE_DELTA,
            angle_range=ANGLE_RANGE,
            snr_db=SNR_DB,
            samples_per_snr=SAMPLES_PER_SNR,
            num_sources=NUM_SOURCES,
            num_snapshots=NUM_SNAPSHOTS,
            num_multipath=NUM_MULTIPATH,
            array_imperfections=ARRAY_IMPERFECTIONS,
            sir_db=0.0,
            output_dir=str(output_dir_path)
        )
    elif EVALUATION_MODE == 2:
        dataset_path = generator.generate_test_dataset(
            reference_angles='random',
            angle_delta=ANGLE_DELTA,
            angle_range=ANGLE_RANGE,
            snr_db=SNR_DB,
            samples_per_snr=SAMPLES_PER_SNR,
            num_sources=NUM_SOURCES,
            num_snapshots=NUM_SNAPSHOTS,
            num_multipath=NUM_MULTIPATH,
            array_imperfections=ARRAY_IMPERFECTIONS,
            sir_db=0.0,
            output_dir=str(output_dir_path)
        )
    elif EVALUATION_MODE == 3:
        dataset_path = generator.generate_test_dataset(
            reference_angles=None,
            angle_delta=None,
            angle_range=ANGLE_RANGE,
            snr_db=SNR_DB,
            samples_per_snr=SAMPLES_PER_SNR,
            num_sources=NUM_SOURCES,
            num_snapshots=NUM_SNAPSHOTS,
            num_multipath=NUM_MULTIPATH,
            array_imperfections=ARRAY_IMPERFECTIONS,
            sir_db=0.0,
            output_dir=str(output_dir_path)
        )
    elif EVALUATION_MODE == 4:
        dataset_path = generator.generate_test_dataset(
            reference_angles=None,
            angle_delta=None,
            angle_range=ANGLE_RANGE,
            snr_db=SNR_DB,
            samples_per_snr=SAMPLES_PER_SNR,
            num_sources=NUM_SOURCES,
            num_snapshots=NUM_SNAPSHOTS,
            num_multipath=NUM_MULTIPATH,
            array_imperfections=ARRAY_IMPERFECTIONS,
            sir_db=0.0,
            output_dir=str(output_dir_path),
            shared_angles_across_snr=True
        )
    
    # Load dataset
    print("\n📂 Loading dataset...")
    dataset = DOADataset(str(dataset_path))
    print(f"✅ Loaded {len(dataset)} samples")
    
    # Load UNet model
    print("\n🤖 Loading UNet model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   Using device: {device}")
    unet_model = load_unet_model(UNET_MODEL_PATH, device)
    
    # Evaluate dataset
    print("\n🎯 Evaluating Dataset...")
    print("="*60)
    results_by_snr = evaluate_dataset(dataset, array_model, unet_model, SNR_DB)
    
    # Calculate accuracy
    accuracy_by_snr = calculate_accuracy(results_by_snr)
    
    # Print results
    print_results_summary(accuracy_by_snr, results_by_snr)
    
    # Convert all numpy types to Python types for JSON serialization
    def convert_to_json_serializable(obj):
        """Recursively convert numpy types to Python types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        else:
            return obj
    
    # Save results
    results_json = {
        'configuration': {
            'mode': EVALUATION_MODE,
            'snr_db': SNR_DB,
            'samples_per_snr': SAMPLES_PER_SNR,
            'num_sources': NUM_SOURCES,
            'num_multipath': NUM_MULTIPATH,
            'num_elements': NUM_ELEMENTS,
            'num_snapshots': NUM_SNAPSHOTS
        },
        'accuracy_by_snr': convert_to_json_serializable(accuracy_by_snr),
        'results_by_snr': convert_to_json_serializable({str(k): v for k, v in results_by_snr.items()})
    }
    
    json_path = output_dir_path / f"results_{DATASET_NAME}.json"
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\n✅ Results saved: {json_path}")
    
    # Plot results
    plot_results(accuracy_by_snr, str(output_dir_path))
    
    print(f"\n✅ Evaluation completed! Results saved to {output_dir_path}")


if __name__ == "__main__":
    main()
