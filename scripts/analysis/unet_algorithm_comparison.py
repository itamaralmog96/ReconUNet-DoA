#!/usr/bin/env python3
"""
UNet + DOA Algorithm Comparison Script

This script compares different DOA algorithms (MUSIC, MVDR, RootMUSIC, ESPRIT) 
when applied to UNet-denoised covariance matrices to determine the best follow-up algorithm.

The script:
1. Generates new dataset OR uses existing dataset with autocorrelation matrices
2. Applies UNet denoising to reconstruct clean covariance matrices
3. Applies different DOA algorithms to the UNet output
4. Compares performance to determine optimal follow-up algorithm
5. Saves results and creates comparison plots

Usage:
    1. Modify configuration parameters below for your needs
    2. Set USE_EXISTING_DATASET = False to generate new dataset, or True to use existing
    3. Set unet_model_path to your trained model
    4. Run: python unet_algorithm_comparison.py
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION - MODIFY THESE PARAMETERS AS NEEDED
# ============================================================================

# === PARAMETER RANGES ===
angles_deg = list(range(30, 151, 1))
snr_db = list(range(-20, 21, 5))  # -20, -15, -10, -5, 0, 5, 10, 15, 20
num_snapshots = [512]
num_sources = [
    [1, 3],  # 1 main + 0 interference
]
sir_db = [0]
num_multipath = [0]  # Multipath sources
array_imperfections = [True]

# === GENERATION CONTROL ===
examples_per_combination = 10  # Number of samples per parameter combination
dataset_name = "comprehensive_eval_dataset"

# === ARRAY CONFIGURATION ===
array_type = "linear"
num_elements = 8
carrier_freq = 2.45e9
sampling_freq = 1e6

# === OUTPUT CONTROL ===
save_received_signal = False
save_covariance_matrix = True
save_clean_covariance_matrix = True  # CRITICAL for UNet targets
save_autocorrelation_matrix = True  # CRITICAL for UNet inputs
save_music_estimates = False
save_steering_vectors = True
save_array_metadata = False

# === AUTOCORRELATION CONTROL ===
autocorr_tau = 8  # Must match UNet training

# === ANGLE SEPARATION CONTROL ===
min_angle_separation_deg = 10.0

# === EVALUATION CONFIGURATION ===
max_samples_per_snr = 1000  # Maximum samples to evaluate per SNR level
scan_angles = np.arange(30, 151, 1)  # Angle grid for MUSIC spectrum
output_dir = 'Tri4Net/src/evaluation/dataset_evaluation'
unet_model_path = 'Tri4Net/notebooks/evd_unet_denoising_model_20250929_015132.pth'

# === DATASET PATH CONFIGURATION ===
USE_EXISTING_DATASET = False  # Set to True to use existing dataset, False to generate new one
EXISTING_DATASET_PATH = 'Tri4Net/Data/datasets/linear/comprehensive_eval_dataset/comprehensive_eval_dataset.h5'

# === DATASET GENERATION SEED ===
# Set to None for random generation (different each time)
# Set to a number (e.g., 42) for reproducible generation (same each time)
DATASET_GENERATION_SEED = None  # None = random, 42 = reproducible

# === ALGORITHMS TO COMPARE ===
# All algorithms will be applied to UNet-denoised covariance matrices
ALGORITHMS_TO_COMPARE = ['MUSIC', 'MVDR', 'RootMUSIC', 'ESPRIT']

# ============================================================================
# END OF CONFIGURATION SECTION
# ============================================================================

# Add src to path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

# Import dataset infrastructure
from data.dataset_generator import SimpleDatasetConfig, ControlledDatasetGenerator, DOADataset

# Import signal generation
from signalgen import ArrayConfig, ArrayModel

# Import classic DOA algorithms
from models.classic.music import MUSIC
from models.classic.mvdr import MVDR
from models.classic.beamformer import Beamformer
from models.classic.esprit import ESPRIT
from models.classic.rootmusic import RootMUSIC

# Import UNet models
try:
    from models.deep_learning.EVDUNet import CovarianceReconstructionUNet, EVDCovarianceReconstructionUNet
    EVDUNET_AVAILABLE = True
except ImportError:
    EVDUNET_AVAILABLE = False
    print("❌ EVDUNet models not available - cannot run comparison")
    sys.exit(1)


def create_autocorrelation_input_for_unet(sample, tau=8, device='cpu'):
    """
    Convert a sample to UNet-compatible autocorrelation tensor.
    Uses pre-computed autocorrelation if available, otherwise computes from received signal.
    """
    # Check if autocorrelation matrix is already in the sample
    if 'autocorrelation_matrix' in sample:
        autocorr_tensor = sample['autocorrelation_matrix']
        # Convert to torch tensor if needed
        if not isinstance(autocorr_tensor, torch.Tensor):
            autocorr_tensor = torch.tensor(autocorr_tensor, dtype=torch.float32)
        return autocorr_tensor.to(device)
    
    # Otherwise compute from received signal
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


def calculate_doa_rmse(true_angles, estimated_angles):
    """Calculate RMSE between true and estimated DOAs."""
    if len(estimated_angles) == 0:
        return np.inf
    
    true_angles = np.array(true_angles)
    estimated_angles = np.array(estimated_angles)
    min_sources = min(len(true_angles), len(estimated_angles))
    
    if min_sources == 0:
        return np.inf
    
    true_sorted = np.sort(true_angles[:min_sources])
    est_sorted = np.sort(estimated_angles[:min_sources])
    errors = np.abs(true_sorted - est_sorted)
    
    return np.sqrt(np.mean(errors**2))


def generate_evaluation_dataset() -> str:
    """Generate dataset using the existing infrastructure (exactly like notebook)."""
    print("🚀 Generating Evaluation Dataset...")
    print("=" * 60)
    
    # Create dataset configuration (same as notebook)
    dataset_config = SimpleDatasetConfig(
        # Parameter ranges
        angles_deg=angles_deg,
        snr_db=snr_db,
        num_snapshots=num_snapshots,
        num_sources=num_sources,
        sir_db=sir_db,
        num_multipath=num_multipath,
        array_imperfections=array_imperfections,
        
        # Random sampling control
        random_sampling_mode=False,
        random_sample_params=None,
        
        # Generation control
        examples_per_combination=examples_per_combination,
        dataset_name=dataset_name,
        
        # Array configuration
        array_type=array_type,
        num_elements=num_elements,
        carrier_freq=carrier_freq,
        sampling_freq=sampling_freq,
        
        # Output control
        save_received_signal=save_received_signal,
        save_covariance_matrix=save_covariance_matrix,
        save_clean_covariance_matrix=save_clean_covariance_matrix,
        save_autocorrelation_matrix=save_autocorrelation_matrix,
        save_music_estimates=save_music_estimates,
        save_steering_vectors=save_steering_vectors,
        save_array_metadata=save_array_metadata,
        
        # Autocorrelation control
        autocorr_tau=autocorr_tau,
        
        # Angle separation control
        min_angle_separation_deg=min_angle_separation_deg
    )
    
    # Create generator with configured seed
    generator = ControlledDatasetGenerator(dataset_config, seed=DATASET_GENERATION_SEED)
    
    # Generate dataset
    output_dir = "Data/datasets"
    dataset_path = generator.generate_dataset(output_dir)
    
    print(f"✅ Dataset generated: {dataset_path}")
    return dataset_path


def evaluate_sample_with_unet_algorithms(sample: Dict, array_model: ArrayModel, 
                                         scan_angles: np.ndarray, unet_model,
                                         algorithms_to_test: List[str],
                                         debug: bool = False, sample_idx: int = 0) -> Dict:
    """
    Evaluate a single sample with UNet + different DOA algorithms.
    """
    labels = sample['labels']
    all_doas = labels['doas']
    
    # Get number of main sources
    if 'num_sources' in labels:
        num_sources_label = labels['num_sources']
        if isinstance(num_sources_label, (list, np.ndarray)):
            num_main_sources = int(num_sources_label.sum()) if len(num_sources_label) > 0 else len(all_doas[~np.isnan(all_doas)])
        else:
            num_main_sources = int(num_sources_label)
    else:
        num_main_sources = len(all_doas[~np.isnan(all_doas)])
    
    # Extract only main source angles (not multipath)
    true_doas_clean = all_doas[~np.isnan(all_doas)][:num_main_sources] if isinstance(all_doas, np.ndarray) else np.array([d for d in all_doas if not np.isnan(d)])[:num_main_sources]
    num_sources = len(true_doas_clean)
    
    if debug:
        print(f"\n🔍 DEBUG Sample Info:")
        print(f"   SNR: {sample['labels']['snr']} dB")
        print(f"   True DOAs (clean): {true_doas_clean}")
        print(f"   Num main sources: {num_main_sources}")
    
    results = {
        'snr': sample['labels']['snr'],
        'true_doas': true_doas_clean
    }
    
    # Apply UNet denoising
    try:
        autocorr = create_autocorrelation_input_for_unet(sample, tau=8)
        autocorr_batch = autocorr.unsqueeze(0)
        
        unet_model.eval()
        with torch.no_grad():
            if isinstance(unet_model, EVDCovarianceReconstructionUNet):
                _, _, recon_cov = unet_model(autocorr_batch)
            else:
                _, recon_cov, _ = unet_model(autocorr_batch)
            recon_cov_np = recon_cov.cpu().numpy().squeeze()
        
        if debug:
            print(f"   ✅ UNet denoising successful")
            print(f"   Reconstructed covariance shape: {recon_cov_np.shape}")
        
    except Exception as e:
        if debug:
            print(f"   ❌ UNet denoising failed: {e}")
        # Return inf for all algorithms if UNet fails
        for alg_name in algorithms_to_test:
            results[f'UNet+{alg_name}'] = np.inf
        return results
    
    # Algorithm mapping
    algorithm_classes = {
        'MUSIC': MUSIC,
        'MVDR': MVDR,
        'RootMUSIC': RootMUSIC,
        'ESPRIT': ESPRIT,
    }
    
    # Apply each DOA algorithm to the UNet-denoised covariance
    for alg_name in algorithms_to_test:
        if alg_name not in algorithm_classes:
            continue
            
        # Skip RootMUSIC for non-linear arrays
        if alg_name == 'RootMUSIC' and array_model.config.array_type != 'linear':
            continue
        
        AlgClass = algorithm_classes[alg_name]
        
        try:
            alg = AlgClass(array_model, array_manifold=None, scan_angles_deg=scan_angles, num_sources=num_main_sources)
            alg.set_received_covariance(recon_cov_np)
            estimated_doas, _ = alg.estimate_doa()
            rmse = calculate_doa_rmse(true_doas_clean, estimated_doas)
            results[f'UNet+{alg_name}'] = rmse
            
            if debug:
                print(f"   UNet+{alg_name}: estimated={estimated_doas}, RMSE={rmse:.2f}")
        except Exception as e:
            results[f'UNet+{alg_name}'] = np.inf
            if debug:
                import traceback
                print(f"   UNet+{alg_name}: ERROR - {e}")
                if sample_idx < 1:
                    traceback.print_exc()
    
    return results


def evaluate_dataset(dataset: DOADataset, array_model: ArrayModel, unet_model,
                     algorithms_to_test: List[str],
                     max_samples_per_snr: int = 50, scan_angles: np.ndarray = None,
                     snr_levels: List = None) -> Dict:
    """
    Evaluate dataset with UNet + different algorithms.
    """
    print("\n🎯 Evaluating UNet + Algorithm Combinations...")
    print("=" * 60)
    print(f"📋 Testing algorithms: {', '.join(algorithms_to_test)}")
    
    if scan_angles is None:
        scan_angles = np.arange(30, 151, 1)
    
    # Use provided SNR levels (from configuration)
    if snr_levels is None:
        all_snr_values = []
        for i in range(len(dataset)):
            sample = dataset[i]
            snr_value = float(sample['labels']['snr'])
            if snr_value not in all_snr_values:
                all_snr_values.append(snr_value)
        all_snr_values = sorted(all_snr_values)
    else:
        all_snr_values = sorted(snr_levels)
    
    print(f"📊 Evaluating SNR levels: {all_snr_values}")
    
    # Evaluate each SNR level
    results_by_snr = {}
    
    for snr_db in all_snr_values:
        print(f"\n🎯 Evaluating SNR = {snr_db} dB")
        
        # Filter dataset for this SNR
        filtered_dataset = dataset.filter(snr=snr_db)
        num_samples = min(len(filtered_dataset), max_samples_per_snr)
        print(f"   Processing {num_samples} samples...")
        
        if num_samples == 0:
            continue
        
        # Initialize results storage
        results_storage = {}
        for alg in algorithms_to_test:
            if alg == 'RootMUSIC' and array_model.config.array_type != 'linear':
                continue
            results_storage[f'UNet+{alg}'] = []
        
        results_by_snr[snr_db] = results_storage
        
        # Evaluate samples
        for idx in tqdm(range(num_samples), desc=f"SNR {snr_db}dB"):
            sample = filtered_dataset[idx]
            debug = False  # Set to True for debugging
            result = evaluate_sample_with_unet_algorithms(
                sample, array_model, scan_angles, unet_model, 
                algorithms_to_test, debug=debug, sample_idx=idx
            )
            
            for alg_key in results_by_snr[snr_db].keys():
                if alg_key in result:
                    results_by_snr[snr_db][alg_key].append(result[alg_key])
    
    print("\n✅ Evaluation complete")
    return results_by_snr


def plot_comparison(results_by_snr: Dict, output_dir: str, algorithms_tested: List[str]):
    """Plot comparison of UNet + different algorithms."""
    print("\n📈 Creating Comparison Plots...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    snr_vals = sorted(results_by_snr.keys())
    
    plt.figure(figsize=(14, 8))
    
    # Define colors and markers for each algorithm
    colors = {
        'MUSIC': 'blue',
        'MVDR': 'orange',
        'RootMUSIC': 'green',
        'ESPRIT': 'red',
    }
    
    markers = {
        'MUSIC': 'o',
        'MVDR': 's',
        'RootMUSIC': '^',
        'ESPRIT': 'd',
    }
    
    # Plot UNet + each algorithm
    for alg in algorithms_tested:
        alg_key = f'UNet+{alg}'
        if alg_key not in results_by_snr[snr_vals[0]]:
            continue
            
        mean_rmse = []
        for snr in snr_vals:
            if alg_key in results_by_snr[snr]:
                rmse_vals = [r for r in results_by_snr[snr][alg_key] if not np.isinf(r)]
                mean_rmse.append(np.mean(rmse_vals) if rmse_vals else np.nan)
            else:
                mean_rmse.append(np.nan)
        
        plt.plot(snr_vals, mean_rmse, 
                marker=markers.get(alg, 'o'), 
                linewidth=2.5, 
                markersize=8,
                label=f'UNet + {alg}',
                color=colors.get(alg, None))
    
    # Generate title with configuration details
    title_parts = ['UNet Denoising + DOA Algorithm Comparison']
    
    # Extract configuration details for title
    if isinstance(num_sources, list):
        main_sources = int(np.array(num_sources).sum()) if len(num_sources) > 0 else 1
    else:
        main_sources = int(num_sources)
    
    if isinstance(num_multipath, list):
        mp = int(num_multipath[0]) if len(num_multipath) > 0 else 0
    else:
        mp = int(num_multipath)
    
    has_imperfections = array_imperfections is not None and len(array_imperfections) > 0
    
    # Build subtitle with configuration
    config_parts = [f'{main_sources} Source{"s" if main_sources > 1 else ""}']
    if mp > 0:
        config_parts.append(f'{mp} Multipath')
    config_parts.append('With Imperfections' if has_imperfections else 'No Imperfections')
    
    title_parts.append('\n' + ' | '.join(config_parts))
    
    plt.xlabel('SNR (dB)', fontsize=14, fontweight='bold')
    plt.ylabel('RMSE (degrees)', fontsize=14, fontweight='bold')
    plt.title(''.join(title_parts), fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    # Generate filename with configuration details
    plot_filename = generate_result_filename("unet_algorithm_comparison", ".png")
    plot_file = output_path / plot_filename
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"✅ Plot saved: {plot_file}")
    plt.close()


def generate_result_filename(base_name: str, extension: str) -> str:
    """
    Generate filename with dataset configuration details.
    """
    # Extract number of sources
    if isinstance(num_sources, list):
        main_sources = int(np.array(num_sources).sum()) if len(num_sources) > 0 else 1
    else:
        main_sources = int(num_sources)
    
    # Extract number of multipath sources
    if isinstance(num_multipath, list):
        mp = int(num_multipath[0]) if len(num_multipath) > 0 else 0
    else:
        mp = int(num_multipath)
    
    # Check for array imperfections
    has_imperfections = array_imperfections is not None and len(array_imperfections) > 0
    
    # Build filename
    filename_parts = [base_name]
    filename_parts.append(f"sources_{main_sources}")
    
    if mp > 0:
        filename_parts.append(f"mp_{mp}")
    
    if has_imperfections:
        filename_parts.append("with_imperfections")
    else:
        filename_parts.append("no_imperfections")
    
    filename = "_".join(filename_parts) + extension
    return filename


def save_results(results_by_snr: Dict, output_dir: str):
    """Save results to JSON."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    json_results = {}
    for snr, alg_results in results_by_snr.items():
        json_results[str(snr)] = {
            alg: [float(r) if not np.isinf(r) else None for r in vals]
            for alg, vals in alg_results.items()
        }
    
    # Generate filename with configuration details
    results_filename = generate_result_filename("unet_comparison_results", ".json")
    results_file = output_path / results_filename
    
    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"✅ Results saved: {results_file}")


def print_summary(results_by_snr: Dict, algorithms_tested: List[str]):
    """Print summary table."""
    print("\n📊 UNet + Algorithm Comparison Summary")
    print("=" * 120)
    
    # Get all algorithm combinations present in results
    all_alg_combos = set()
    for snr_results in results_by_snr.values():
        all_alg_combos.update(snr_results.keys())
    
    # Create ordered list based on algorithms tested
    algorithms_to_show = []
    for alg in algorithms_tested:
        alg_key = f'UNet+{alg}'
        if alg_key in all_alg_combos:
            algorithms_to_show.append(alg_key)
    
    # Create header
    header = f"{'SNR (dB)':<10}"
    for alg in algorithms_to_show:
        header += f"{alg:<20}"
    print(header)
    print("-" * 120)
    
    for snr in sorted(results_by_snr.keys()):
        row = f"{snr:<10.1f}"
        for alg in algorithms_to_show:
            if alg in results_by_snr[snr]:
                rmse_vals = [r for r in results_by_snr[snr][alg] if not np.isinf(r)]
                row += f"{np.mean(rmse_vals) if rmse_vals else float('inf'):<20.2f}"
            else:
                row += f"{'N/A':<20}"
        print(row)
    
    print("-" * 120)
    
    # Print best algorithm per SNR
    print("\n🏆 Best Algorithm per SNR Level:")
    print("-" * 60)
    for snr in sorted(results_by_snr.keys()):
        best_alg = None
        best_rmse = float('inf')
        for alg in algorithms_to_show:
            if alg in results_by_snr[snr]:
                rmse_vals = [r for r in results_by_snr[snr][alg] if not np.isinf(r)]
                mean_rmse = np.mean(rmse_vals) if rmse_vals else float('inf')
                if mean_rmse < best_rmse:
                    best_rmse = mean_rmse
                    best_alg = alg
        print(f"   SNR {snr:>5.1f} dB: {best_alg:<20} (RMSE: {best_rmse:.2f}°)")


def main():
    print("🚀 UNet + DOA Algorithm Comparison")
    print("=" * 60)
    
    # Display configuration
    print("\n📋 Configuration:")
    print(f"   Dataset Name: {dataset_name}")
    print(f"   Array: {num_elements} element {array_type}")
    print(f"   SNR Range: {snr_db}")
    print(f"   Sources: {num_sources}")
    print(f"   Samples per condition: {examples_per_combination}")
    print(f"   Algorithms to test: {', '.join(ALGORITHMS_TO_COMPARE)}")
    
    # Get or generate dataset
    if USE_EXISTING_DATASET:
        dataset_path = EXISTING_DATASET_PATH
        print(f"\n📂 Using existing dataset: {dataset_path}")
        if not os.path.exists(dataset_path):
            print(f"❌ Dataset not found: {dataset_path}")
            print("   Set USE_EXISTING_DATASET = False to generate a new dataset")
            return
    else:
        print(f"\n🔧 Generating new dataset...")
        dataset_path = generate_evaluation_dataset()
    
    # Load dataset
    print(f"\n📂 Loading dataset...")
    dataset = DOADataset(dataset_path)
    print(f"✅ Loaded {len(dataset)} samples")
    
    # Display dataset info
    print(f"\n📊 Dataset Information:")
    print(f"   Total samples: {len(dataset)}")
    print(f"   Has autocorrelation: {dataset.has_autocorrelation}")
    
    if not dataset.has_autocorrelation:
        print("\n❌ Dataset does not have autocorrelation matrices!")
        print("   UNet requires autocorrelation matrices as input.")
        return
    
    # Create array model
    array_config = ArrayConfig(
        array_type=array_type,
        num_elements=num_elements,
        carrier_freq=carrier_freq,
        element_spacing=0.5
    )
    array_model = ArrayModel(array_config)
    print(f"   Created array model from configuration")
    
    # Initialize UNet
    if not EVDUNET_AVAILABLE:
        print("\n❌ UNet models not available - cannot run comparison")
        return
    
    model_params = {
        'tau': 8,
        'M': 8,
        'activation_type': 'anti_rectifier',
        'use_dropout': True
    }
    print("\n🤖 Initializing UNet...")
    unet_model = EVDCovarianceReconstructionUNet(**model_params)
    
    if unet_model_path and os.path.exists(unet_model_path):
        checkpoint = torch.load(unet_model_path, map_location='cpu')
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            unet_model.load_state_dict(checkpoint['model_state_dict'])
            if 'epoch' in checkpoint:
                print(f"   Model trained for {checkpoint['epoch']} epochs")
        else:
            unet_model.load_state_dict(checkpoint)
        
        print(f"✅ Loaded trained model from {unet_model_path}")
    else:
        print(f"❌ UNet model not found: {unet_model_path}")
        print("   Please provide a valid trained model path")
        return
    
    # Evaluate
    results_by_snr = evaluate_dataset(
        dataset, 
        array_model, 
        unet_model,
        ALGORITHMS_TO_COMPARE,
        max_samples_per_snr,
        scan_angles,
        snr_db
    )
    
    # Display and save results
    print_summary(results_by_snr, ALGORITHMS_TO_COMPARE)
    save_results(results_by_snr, output_dir)
    plot_comparison(results_by_snr, output_dir, ALGORITHMS_TO_COMPARE)
    
    print(f"\n✅ Comparison completed! Results saved to {output_dir}")


if __name__ == "__main__":
    main()

