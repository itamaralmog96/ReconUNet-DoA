#!/usr/bin/env python3
"""
Compare MUSIC Spectrum: Noisy vs UNet-Reconstructed Covariance

This script generates a single signal sample and compares the MUSIC spectrum
computed from:
1. Noisy covariance matrix (classical MUSIC)
2. UNet-reconstructed covariance matrix (SubNet+MUSIC)

The plot shows both spectra with vertical lines marking true DOA angles
and dashed lines for multipath components.

Usage:
    python compare_music_spectrum.py
"""

import os
import sys

# Fix OpenMP library conflict (needed for torch on some systems)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add src to path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

# Import dataset infrastructure
from data.dataset_generator import SimpleDatasetConfig, ControlledDatasetGenerator

# Import signal generation
from signalgen import ArrayConfig, ArrayModel

# Import classic DOA algorithms
from models.classic.music import MUSIC

# Import UNet models
try:
    from models.deep_learning.EVDUNet import EVDCovarianceReconstructionUNet
    EVDUNET_AVAILABLE = True
except ImportError:
    EVDUNET_AVAILABLE = False
    print("⚠️ EVDUNet models not available")

# ============================================================================
# CONFIGURATION SECTION - MODIFY THESE PARAMETERS
# ============================================================================

# === SOURCE CONFIGURATION ===
NUM_SOURCES = 2  # Number of sources
SOURCE_ANGLES = [80.1, 90.6]  # DOA angles in degrees (set to None for random)
# SOURCE_ANGLES = None
SNR_DB = 0.0  # Signal-to-noise ratio in dB
NUM_SNAPSHOTS = 512  # Number of time snapshots

# === MULTIPATH CONFIGURATION ===
NUM_MULTIPATH = 2  # Number of multipath components (0 = no multipath, 1 = one multipath per source, etc.)

# === ARRAY CONFIGURATION ===
ARRAY_TYPE = "linear"  # Array geometry
NUM_ELEMENTS = 8  # Number of array elements
CARRIER_FREQ = 2.45e9  # Carrier frequency in Hz
SAMPLING_FREQ = 1e6  # Sampling frequency in Hz
ARRAY_IMPERFECTIONS = False  # Enable array imperfections

# === MUSIC CONFIGURATION ===
MUSIC_SCAN_START = 30.0  # Start angle for MUSIC scan
MUSIC_SCAN_STOP = 150.0   # End angle for MUSIC scan
MUSIC_SCAN_STEP = 0.1     # Angle step for MUSIC scan (degrees)

# === UNET MODEL CONFIGURATION ===
UNET_MODEL_PATH = 'notebooks/evd_unet_denoising_model_20250929_015132.pth'
AUTOCORR_TAU = 8  # Number of time lags (must match UNet training)

# === OUTPUT CONFIGURATION ===
OUTPUT_DIR = 'Tri4Net/src/evaluation/spectrum_comparison'
PLOT_DPI = 150

# ============================================================================
# END OF CONFIGURATION SECTION
# ============================================================================


def create_autocorrelation_input_for_unet(sample: Dict, tau: int, device='cpu') -> torch.Tensor:
    """
    Create autocorrelation input tensor for UNet from a sample.
    Matches the exact format from controlled_angle_evaluation.py
    
    Parameters
    ----------
    sample : dict
        Sample containing received signal or autocorrelation_matrix
    tau : int
        Number of time lags
    device : str
        'cpu' or 'cuda'
        
    Returns
    -------
    torch.Tensor
        Autocorrelation tensor of shape (tau, 2*M, M) in real format
    """
    # Check if pre-computed autocorrelation is available
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
    
    # Create tensor with shape (tau, 2*M, M) - real format
    autocorr_tensor = torch.zeros(tau, 2*M, M, dtype=torch.float32, device=device)
    max_lag = min(tau, T - 1)
    
    for lag in range(max_lag):
        samples_available = T - lag
        if samples_available > 0:
            x1 = received_signal[:, :samples_available]
            x2 = received_signal[:, lag:lag + samples_available]
            R_lag = torch.matmul(x1, x2.conj().T) / samples_available
            # Store real part in first M rows, imaginary part in next M rows
            autocorr_tensor[lag, :M, :] = R_lag.real
            autocorr_tensor[lag, M:, :] = R_lag.imag
    
    return autocorr_tensor


def generate_single_sample(angles: List[float], 
                          snr_db: float,
                          num_snapshots: int,
                          num_multipath: int = 0) -> Dict:
    """
    Generate a single signal sample using the exact same method as controlled_angle_evaluation.
    
    Parameters
    ----------
    angles : list
        True DOA angles in degrees
    snr_db : float
        Signal-to-noise ratio in dB
    num_snapshots : int
        Number of time snapshots
    num_multipath : int
        Number of multipath components (0 means no multipath)
        
    Returns
    -------
    dict
        Sample containing received signal, covariance, and metadata
    """
    num_sources = len(angles)
    
    # Create dataset configuration matching controlled_angle_evaluation
    dataset_config = SimpleDatasetConfig(
        angles_deg=angles,
        snr_db=[snr_db],
        num_snapshots=[num_snapshots],
        num_sources=[num_sources],
        sir_db=None,
        num_multipath=[num_multipath],
        array_imperfections=[ARRAY_IMPERFECTIONS],
        random_sampling_mode=False,
        examples_per_combination=1,
        dataset_name="temp_spectrum_comparison",
        array_type=ARRAY_TYPE,
        num_elements=NUM_ELEMENTS,
        carrier_freq=CARRIER_FREQ,
        sampling_freq=SAMPLING_FREQ,
        save_received_signal=True,
        save_covariance_matrix=True,
        save_clean_covariance_matrix=False,
        save_autocorrelation_matrix=False,
        save_steering_vectors=False,
        save_array_metadata=False,
        autocorr_tau=AUTOCORR_TAU
    )
    
    # Create generator with random seed (None means random)
    generator = ControlledDatasetGenerator(dataset_config, seed=None)
    
    # Create parameter combination
    param_combo = {
        'angle_deg': angles[0],  # Base angle (will be used with separation) 
        'snr_db': snr_db,
        'num_snapshots': num_snapshots,
        'num_sources': [num_sources, 0],  # [main sources, interference sources]
        'sir_db': None,
        'num_multipath': num_multipath,
        'array_imperfections': ARRAY_IMPERFECTIONS
    }
    
    # Generate single sample using the internal method with random example_idx
    import random
    example_idx = random.randint(0, 999999)
    sample = generator._generate_sample(param_combo, example_idx=example_idx)
    
    return sample


def calculate_doa_rmse(true_angles, estimated_angles, penalty_for_missed=240.0):
    """
    Calculate RMSE between true and estimated DOAs using optimal matching.
    
    This function finds the best assignment between true and estimated angles
    to minimize the total squared error.
    
    Parameters
    ----------
    true_angles : array-like
        True DOA angles in degrees
    estimated_angles : array-like
        Estimated DOA angles in degrees
    penalty_for_missed : float
        Penalty value for missed sources
        
    Returns
    -------
    float
        RMSE in degrees
    """
    if len(estimated_angles) == 0:
        return penalty_for_missed
    
    true_angles = np.array(true_angles)
    estimated_angles = np.array(estimated_angles)
    
    n_true = len(true_angles)
    n_est = len(estimated_angles)
    
    if n_est < n_true:
        # Not enough estimates - penalize missed sources
        matched_errors = []
        used_est = set()
        
        for true_angle in true_angles:
            best_error = penalty_for_missed
            best_idx = None
            for i, est_angle in enumerate(estimated_angles):
                if i not in used_est:
                    error = abs(true_angle - est_angle)
                    if error < best_error:
                        best_error = error
                        best_idx = i
            if best_idx is not None:
                matched_errors.append(best_error)
                used_est.add(best_idx)
            else:
                matched_errors.append(penalty_for_missed)
        
        return np.sqrt(np.mean(np.array(matched_errors)**2))
    
    # For each true angle, find the closest estimated angle
    matched_errors = []
    used_est = set()
    
    for true_angle in true_angles:
        best_error = penalty_for_missed
        best_idx = None
        for i, est_angle in enumerate(estimated_angles):
            if i not in used_est:
                error = abs(true_angle - est_angle)
                if error < best_error:
                    best_error = error
                    best_idx = i
        if best_idx is not None:
            matched_errors.append(best_error)
            used_est.add(best_idx)
        else:
            matched_errors.append(penalty_for_missed)
    
    return np.sqrt(np.mean(np.array(matched_errors)**2))


def compute_music_spectrum(cov_matrix: np.ndarray,
                           array_model: ArrayModel,
                           num_sources: int,
                           scan_angles: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute MUSIC spectrum and DOA estimates from covariance matrix.
    
    Parameters
    ----------
    cov_matrix : np.ndarray
        Covariance matrix
    array_model : ArrayModel
        Array model
    num_sources : int
        Number of sources
    scan_angles : np.ndarray
        Scan angles in degrees
        
    Returns
    -------
    tuple
        (scan_angles, spectrum, estimated_doas)
    """
    music = MUSIC(
        array_model=array_model,
        scan_angles_deg=scan_angles,
        num_sources=num_sources
    )
    music.set_received_covariance(cov_matrix)
    estimated_doas, spectrum = music.estimate_doa()
    
    return scan_angles, spectrum, estimated_doas


def plot_spectrum_comparison(scan_angles: np.ndarray,
                            noisy_spectrum: np.ndarray,
                            reconstructed_spectrum: np.ndarray,
                            true_angles: List[float],
                            num_multipath: int,
                            sample_metadata: Dict,
                            output_dir: Path,
                            snr_db: float):
    """
    Plot comparison of MUSIC spectra.
    
    Parameters
    ----------
    scan_angles : np.ndarray
        Scan angles in degrees
    noisy_spectrum : np.ndarray
        MUSIC spectrum from noisy covariance
    reconstructed_spectrum : np.ndarray
        MUSIC spectrum from UNet-reconstructed covariance
    true_angles : list
        True DOA angles
    num_multipath : int
        Number of multipath components
    sample_metadata : dict
        Sample metadata containing multipath information
    output_dir : Path
        Output directory
    snr_db : float
        SNR in dB
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Convert spectra to dB scale
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    noisy_db = 10 * np.log10(noisy_spectrum + epsilon)
    reconstructed_db = 10 * np.log10(reconstructed_spectrum + epsilon)
    
    # Normalize to 0 dB max for better visualization
    # noisy_db = noisy_db - np.max(noisy_db)
    # reconstructed_db = reconstructed_db - np.max(reconstructed_db)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot spectra in dB
    ax.plot(scan_angles, noisy_db, 
            color='coral', linewidth=2, label='MUSIC', alpha=0.8)
    ax.plot(scan_angles, reconstructed_db, 
            color='mediumpurple', linewidth=2, label='UNet+MUSIC', alpha=0.8)
    
    # Mark true angles with solid green vertical lines
    true_doa_plotted = False
    for i, angle in enumerate(true_angles):
        if i == 0 and not true_doa_plotted:
            ax.axvline(x=angle, color='green', linestyle='-', linewidth=1.5, alpha=0.7, label='True DOA')
            true_doa_plotted = True
        else:
            ax.axvline(x=angle, color='green', linestyle='-', linewidth=1.5, alpha=0.7)
    
    # Mark multipath angles with dashed gray vertical lines if present
    multipath_plotted = False
    if num_multipath > 0 and 'multipath' in sample_metadata:
        multipath_info = sample_metadata['multipath']
        if 'angles' in multipath_info and multipath_info['angles'] is not None:
            multipath_angles = multipath_info['angles']
            # Handle different multipath angle formats
            if isinstance(multipath_angles, np.ndarray):
                if multipath_angles.ndim == 1:
                    # Single list of angles
                    for angle in multipath_angles:
                        if not np.isnan(angle):
                            if not multipath_plotted:
                                ax.axvline(x=angle, color='gray', linestyle='--', linewidth=1.5, alpha=0.6, label='Multipath')
                                multipath_plotted = True
                            else:
                                ax.axvline(x=angle, color='gray', linestyle='--', linewidth=1.5, alpha=0.6)
                else:
                    # 2D array: multipath per source
                    for mp_angles in multipath_angles:
                        for angle in mp_angles:
                            if not np.isnan(angle):
                                if not multipath_plotted:
                                    ax.axvline(x=angle, color='gray', linestyle='--', linewidth=1.5, alpha=0.6, label='Multipath')
                                    multipath_plotted = True
                                else:
                                    ax.axvline(x=angle, color='gray', linestyle='--', linewidth=1.5, alpha=0.6)
    
    # Formatting
    ax.set_xlabel('Angles [deg]', fontsize=14, fontweight='normal')
    ax.set_ylabel('Amplitude [dB]', fontsize=14, fontweight='normal')
    ax.set_xlim(MUSIC_SCAN_START, MUSIC_SCAN_STOP)
    # Auto-scale y-axis based on actual spectrum values
    # ax.set_ylim(-60, 5)  # Use auto-scale for non-normalized spectrum
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12, loc='upper right')
    
    # Set tick parameters
    ax.tick_params(axis='both', which='major', labelsize=11)
    
    plt.tight_layout()
    
    # Save figure
    filename = f'spectrum_comparison_{NUM_SOURCES}src_snr{snr_db}dB'
    if num_multipath > 0:
        filename += f'_mp{num_multipath}'
    
    output_file = output_path / f'{filename}.png'
    plt.savefig(output_file, dpi=PLOT_DPI, bbox_inches='tight')
    print(f"✅ Plot saved: {output_file}")
    
    plt.show()


def main():
    """Main execution function."""
    print("🎵 MUSIC Spectrum Comparison: Noisy vs UNet-Reconstructed")
    print("="*70)
    
    # Print configuration
    print(f"\n📋 Configuration:")
    print(f"   Sources: {NUM_SOURCES}")
    print(f"   Angles: {SOURCE_ANGLES if SOURCE_ANGLES else 'Random'}")
    print(f"   SNR: {SNR_DB} dB")
    print(f"   Multipath components: {NUM_MULTIPATH}")
    print(f"   Array: {NUM_ELEMENTS} element {ARRAY_TYPE}")
    print(f"   MUSIC scan: {MUSIC_SCAN_START}° to {MUSIC_SCAN_STOP}° (step: {MUSIC_SCAN_STEP}°)")
    
    # Create array configuration
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
    print(f"\n✅ Array model created")
    
    # Determine angles
    if SOURCE_ANGLES is None:
        # Generate random angles
        angles = np.random.uniform(30, 150, NUM_SOURCES).tolist()
        angles.sort()
        print(f"\n🎲 Generated random angles: {[f'{a:.1f}' for a in angles]}")
    else:
        angles = SOURCE_ANGLES
        print(f"\n🎯 Using specified angles: {angles}")
    
    # Initialize UNet
    unet_model = None
    if EVDUNET_AVAILABLE:
        print(f"\n🤖 Initializing UNet...")
        model_params = {
            'tau': AUTOCORR_TAU,
            'M': NUM_ELEMENTS,
            'activation_type': 'anti_rectifier',
            'use_dropout': True
        }
        unet_model = EVDCovarianceReconstructionUNet(**model_params)
        
        # Handle relative paths
        model_path = Path(UNET_MODEL_PATH)
        if not model_path.is_absolute():
            model_path = Path(__file__).parent / model_path
        
        if model_path.exists():
            try:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                print(f"   Model path: {model_path.name}")
                print(f"   Device: {device}")
                
                checkpoint = torch.load(str(model_path), map_location=device)
                
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    unet_model.load_state_dict(checkpoint['model_state_dict'])
                    epoch = checkpoint.get('epoch', 'unknown')
                    print(f"✅ Loaded UNet model from checkpoint (epoch {epoch})")
                else:
                    unet_model.load_state_dict(checkpoint)
                    print(f"✅ Loaded UNet model from {model_path.name}")
                
                unet_model = unet_model.to(device)
                unet_model.eval()
            except Exception as e:
                print(f"❌ Error loading UNet model: {e}")
                unet_model = None
        else:
            print(f"⚠️ UNet model not found at {model_path}")
            unet_model = None
    
    if unet_model is None:
        print("❌ Cannot proceed without UNet model")
        return
    
    # Generate single sample
    print(f"\n📡 Generating signal sample...")
    sample = generate_single_sample(
        angles=angles,
        snr_db=SNR_DB,
        num_snapshots=NUM_SNAPSHOTS,
        num_multipath=NUM_MULTIPATH
    )
    print(f"✅ Sample generated")
    
    # Extract multipath angles from the sample
    # The DOAs in labels include ALL angles (sources + multipath)
    # First num_sources angles are the true DOAs, rest are multipath
    all_doas = sample['labels']['doas']
    num_sources_total = len(angles)
    num_multipath_actual = sample['labels']['num_multipath']
    
    # Extract multipath angles (comes after the main source angles)
    multipath_angles_extracted = []
    if num_multipath_actual > 0 and len(all_doas) > num_sources_total:
        # Multipath angles are appended after source angles
        multipath_angles_extracted = all_doas[num_sources_total:].tolist()
        print(f"   Multipath angles: {[f'{a:.1f}°' for a in multipath_angles_extracted]}")
    
    # Store multipath info in a format compatible with our plot function
    sample['metadata'] = {
        'multipath': {
            'angles': np.array(multipath_angles_extracted) if multipath_angles_extracted else None
        }
    }
    
    # Define scan angles for MUSIC
    scan_angles = np.arange(MUSIC_SCAN_START, MUSIC_SCAN_STOP + MUSIC_SCAN_STEP, MUSIC_SCAN_STEP)
    
    # Compute MUSIC spectrum with noisy covariance
    print(f"\n🎵 Computing MUSIC spectrum (noisy covariance)...")
    noisy_cov = sample['covariance_matrix']
    _, noisy_spectrum, noisy_doa_estimates = compute_music_spectrum(
        noisy_cov, array_model, NUM_SOURCES, scan_angles
    )
    print(f"✅ Noisy spectrum computed")
    print(f"   MUSIC estimated DOAs: {[f'{a:.1f}°' for a in noisy_doa_estimates]}")
    
    # Compute MUSIC spectrum with UNet-reconstructed covariance
    print(f"\n🧠 Computing MUSIC spectrum (UNet-reconstructed covariance)...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    autocorr_input = create_autocorrelation_input_for_unet(sample, tau=AUTOCORR_TAU, device=device)
    autocorr_input = autocorr_input.unsqueeze(0)  # Add batch dimension
    
    with torch.no_grad():
        eigenvals, eigenvecs, reconstructed_cov = unet_model(autocorr_input)
    
    reconstructed_cov = reconstructed_cov.squeeze().cpu().numpy()
    # Ensure Hermitian
    # reconstructed_cov = (reconstructed_cov + reconstructed_cov.conj().T) / 2
    
    _, reconstructed_spectrum, reconstructed_doa_estimates = compute_music_spectrum(
        reconstructed_cov, array_model, NUM_SOURCES, scan_angles
    )
    print(f"✅ Reconstructed spectrum computed")
    print(f"   UNet+MUSIC estimated DOAs: {[f'{a:.1f}°' for a in reconstructed_doa_estimates]}")
    
    # Calculate RMSE (loss) for both methods
    print(f"\n📊 DOA Estimation Results & RMSE (Loss)")
    print(f"=" * 70)
    print(f"   True DOAs:              {[f'{a:.1f}°' for a in angles]}")
    print(f"   MUSIC Estimates:        {[f'{a:.1f}°' for a in noisy_doa_estimates]}")
    print(f"   UNet+MUSIC Estimates:   {[f'{a:.1f}°' for a in reconstructed_doa_estimates]}")
    print(f"-" * 70)
    
    noisy_rmse = calculate_doa_rmse(angles, noisy_doa_estimates)
    reconstructed_rmse = calculate_doa_rmse(angles, reconstructed_doa_estimates)
    
    print(f"   MUSIC RMSE:             {noisy_rmse:.3f}°")
    print(f"   UNet+MUSIC RMSE:        {reconstructed_rmse:.3f}°")
    print(f"-" * 70)
    
    if reconstructed_rmse < noisy_rmse:
        improvement = ((noisy_rmse - reconstructed_rmse) / noisy_rmse) * 100
        print(f"   ✅ UNet improves accuracy by {improvement:.1f}%")
    elif reconstructed_rmse > noisy_rmse:
        degradation = ((reconstructed_rmse - noisy_rmse) / noisy_rmse) * 100
        print(f"   ⚠️ UNet degrades accuracy by {degradation:.1f}%")
    else:
        print(f"   ➡️ UNet has same accuracy as MUSIC")
    print(f"=" * 70)
    
    # Plot comparison
    print(f"\n📊 Creating comparison plot...")
    plot_spectrum_comparison(
        scan_angles=scan_angles,
        noisy_spectrum=noisy_spectrum,
        reconstructed_spectrum=reconstructed_spectrum,
        true_angles=angles,
        num_multipath=NUM_MULTIPATH,
        sample_metadata=sample.get('metadata', {}),
        output_dir=OUTPUT_DIR,
        snr_db=SNR_DB
    )
    
    print(f"\n✅ Comparison completed!")
    print(f"   Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
