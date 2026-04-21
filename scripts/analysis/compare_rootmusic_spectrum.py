#!/usr/bin/env python3
"""
Compare Root-MUSIC Spectrum: Noisy vs UNet-Reconstructed Covariance

This script generates a single signal sample and compares the Root-MUSIC spectrum
computed from:
1. Noisy covariance matrix (classical Root-MUSIC)
2. UNet-reconstructed covariance matrix (UNet+Root-MUSIC)

The plot shows both spectra in polar format with markers for true DOA angles.
Root-MUSIC works by finding roots of the MUSIC polynomial on the unit circle.

Usage:
    python compare_rootmusic_spectrum.py
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
from models.classic.rootmusic import RootMUSIC

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
NUM_MULTIPATH = 2  # Number of multipath components (0 = no multipath)

# === ARRAY CONFIGURATION ===
ARRAY_TYPE = "linear"  # Array geometry (Root-MUSIC only works with linear arrays)
NUM_ELEMENTS = 8  # Number of array elements
CARRIER_FREQ = 2.45e9  # Carrier frequency in Hz
SAMPLING_FREQ = 1e6  # Sampling frequency in Hz
ARRAY_IMPERFECTIONS = True  # Enable array imperfections

# === ROOT-MUSIC CONFIGURATION ===
ANGLE_RANGE = (0.0, 180.0)  # Angle range for visualization (0-180 degrees)
ANGLE_STEP = 0.5  # Angle step for spectrum visualization (degrees)

# === UNET MODEL CONFIGURATION ===
UNET_MODEL_PATH = 'notebooks/evd_unet_denoising_model_20250929_015132.pth'
AUTOCORR_TAU = 8  # Number of time lags (must match UNet training)

# === OUTPUT CONFIGURATION ===
OUTPUT_DIR = 'Tri4Net/src/evaluation/rootmusic_comparison'
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
        dataset_name="temp_rootmusic_comparison",
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


def compute_rootmusic_roots(cov_matrix: np.ndarray,
                            array_model: ArrayModel,
                            num_sources: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Root-MUSIC polynomial roots and DOA estimates.
    
    Root-MUSIC finds roots of the polynomial on the unit circle.
    
    Parameters
    ----------
    cov_matrix : np.ndarray
        Covariance matrix
    array_model : ArrayModel
        Array model
    num_sources : int
        Number of sources
        
    Returns
    -------
    tuple
        (all_roots, selected_roots, estimated_doas)
        - all_roots: All polynomial roots (complex numbers)
        - selected_roots: Selected roots closest to unit circle
        - estimated_doas: DOA estimates in degrees
    """
    # Run Root-MUSIC to get DOA estimates and all roots
    rootmusic = RootMUSIC(
        array_model=array_model,
        num_sources=num_sources
    )
    rootmusic.set_received_covariance(cov_matrix)
    
    # Get DOA estimates and all roots from RootMUSIC
    estimated_doas, all_roots = rootmusic.estimate_doa()
    
    # Get the selected roots that correspond to the DOA estimates
    selected_roots = rootmusic.selected_roots
    
    return all_roots, selected_roots, estimated_doas


def plot_polar_roots_comparison(noisy_all_roots: np.ndarray,
                               noisy_selected_roots: np.ndarray,
                               reconstructed_all_roots: np.ndarray,
                               reconstructed_selected_roots: np.ndarray,
                               noisy_doas: np.ndarray,
                               reconstructed_doas: np.ndarray,
                               true_angles: List[float],
                               output_dir: Path,
                               snr_db: float):
    """
    Plot comparison of Root-MUSIC roots on unit circle.
    
    Parameters
    ----------
    noisy_all_roots : np.ndarray
        All polynomial roots from noisy covariance
    noisy_selected_roots : np.ndarray
        Selected roots (DOA estimates) from noisy covariance
    reconstructed_all_roots : np.ndarray
        All polynomial roots from reconstructed covariance
    reconstructed_selected_roots : np.ndarray
        Selected roots (DOA estimates) from reconstructed covariance
    noisy_doas : np.ndarray
        DOA estimates from noisy covariance
    reconstructed_doas : np.ndarray
        DOA estimates from reconstructed covariance
    true_angles : list
        True DOA angles
    output_dir : Path
        Output directory
    snr_db : float
        SNR in dB
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Convert true angles to radians for polar plot
    true_theta_rad = np.deg2rad(true_angles)
    
    # Create figure with two subplots (side by side)
    fig = plt.figure(figsize=(16, 7))
    
    # Left subplot: Noisy Root-MUSIC
    ax1 = fig.add_subplot(121, projection='polar')
    
    # For Root-MUSIC, roots are plotted on the complex plane
    # z = exp(j * k_d * cos(theta)) where theta is the DOA angle
    k_d = 2 * np.pi * 0.5  # element_spacing = 0.5 wavelengths
    num_elements = 8
    # Calculate array center offset (matching RootMUSIC algorithm)
    if num_elements % 2 == 0:
        c = num_elements / 2
    else:
        c = (num_elements - 1) / 2
    
    # Apply the SAME transformation that RootMUSIC uses to extract DOA from roots
    # This will plot roots at their corresponding DOA angle positions
    
    # Calculate max radius individually for each set of roots
    max_radius_noisy = max(np.abs(noisy_all_roots)) * 1.05  # Add 5% margin
    max_radius_reconstructed = max(np.abs(reconstructed_all_roots)) * 1.05  # Add 5% margin
    
    # Plot ALL roots as gray dots
    plotted_count = 0
    skipped_count = 0
    for root in noisy_all_roots:
        # Apply RootMUSIC transformation: roots * exp(j*k_d*c)
        root_transformed = root * np.exp(1j * k_d * c)
        roots_angle = np.angle(root_transformed) / k_d
        
        # Extract DOA angle: arccos(-roots_angle)
        # Only plot if the value is valid for arccos (in [-1, 1])
        if -1 <= roots_angle <= 1:
            doa_rad = np.arccos(-roots_angle)
            r = np.abs(root)
            ax1.plot(doa_rad, r, 'o', color='gray', markersize=6, alpha=0.6)
            plotted_count += 1
        else:
            skipped_count += 1
    
    if skipped_count > 0:
        print(f"   Note: {skipped_count} roots skipped (roots_angle outside [-1,1])")
    
    # Plot selected roots (DOA estimates) as red dots
    for root in noisy_selected_roots:
        # Apply the same transformation
        root_transformed = root * np.exp(1j * k_d * c)
        roots_angle = np.angle(root_transformed) / k_d
        
        if -1 <= roots_angle <= 1:
            doa_rad = np.arccos(-roots_angle)
            r = np.abs(root)
            ax1.plot(doa_rad, r, 'o', color='red', markersize=12, 
                    markeredgecolor='darkred', markeredgewidth=2)
    
    # Mark true DOAs with green lines at the actual DOA angles
    for angle_rad in true_theta_rad:
        ax1.plot([angle_rad, angle_rad], [0, max_radius_noisy * 0.95], color='green', linestyle='-', 
                linewidth=3, alpha=0.8)
    
    # Draw semicircle at r=1 (since arccos gives 0-180°)
    theta_circle = np.linspace(0, np.pi, 100)
    ax1.plot(theta_circle, np.ones_like(theta_circle), 'k--', linewidth=2, alpha=0.5)
    
    # Set orientation: 0° at right (East), 90° at top, counter-clockwise
    ax1.set_theta_zero_location('E')  # 0° at East (positive x-axis, right)
    ax1.set_theta_direction(1)        # Counter-clockwise: 90° at top (positive y-axis)
    ax1.set_thetamin(0)
    ax1.set_thetamax(180)
    ax1.set_ylim(0, max_radius_noisy)  # Dynamic limit based on noisy roots magnitude
    ax1.set_title(f'Root-MUSIC (Noisy)\nSNR = {snr_db} dB\n({plotted_count}/{len(noisy_all_roots)} roots plotted)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3)
    
    # Right subplot: UNet+Root-MUSIC
    ax2 = fig.add_subplot(122, projection='polar')
    
    # Apply the SAME transformation for reconstructed roots
    
    # Plot ALL roots as gray dots
    for root in reconstructed_all_roots:
        # Apply RootMUSIC transformation
        root_transformed = root * np.exp(1j * k_d * c)
        roots_angle = np.angle(root_transformed) / k_d
        
        # Extract DOA angle
        if -1 <= roots_angle <= 1:
            doa_rad = np.arccos(-roots_angle)
            r = np.abs(root)
            ax2.plot(doa_rad, r, 'o', color='gray', markersize=6, alpha=0.6)
    
    # Plot selected roots (DOA estimates) as red dots
    for root in reconstructed_selected_roots:
        # Apply the same transformation
        root_transformed = root * np.exp(1j * k_d * c)
        roots_angle = np.angle(root_transformed) / k_d
        
        if -1 <= roots_angle <= 1:
            doa_rad = np.arccos(-roots_angle)
            r = np.abs(root)
            ax2.plot(doa_rad, r, 'o', color='red', markersize=12,
                    markeredgecolor='darkred', markeredgewidth=2)
    
    # Mark true DOAs with green lines at the actual DOA angles
    for angle_rad in true_theta_rad:
        ax2.plot([angle_rad, angle_rad], [0, max_radius_reconstructed * 0.95], color='green', linestyle='-', 
                linewidth=3, alpha=0.8)
    
    # Draw semicircle at r=1 (since arccos gives 0-180°)
    theta_circle = np.linspace(0, np.pi, 100)
    ax2.plot(theta_circle, np.ones_like(theta_circle), 'k--', linewidth=2, alpha=0.5)
    
    # Set orientation: 0° at right (East), 90° at top, counter-clockwise
    ax2.set_theta_zero_location('E')  # 0° at East (positive x-axis, right)
    ax2.set_theta_direction(1)        # Counter-clockwise: 90° at top (positive y-axis)
    ax2.set_thetamin(0)
    ax2.set_thetamax(180)
    ax2.set_ylim(0, max_radius_reconstructed)  # Dynamic limit based on reconstructed roots magnitude
    ax2.set_title(f'UNet+Root-MUSIC (Reconstructed)\nSNR = {snr_db} dB\n({plotted_count}/{len(reconstructed_all_roots)} roots plotted)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    filename = f'rootmusic_comparison_{NUM_SOURCES}src_snr{snr_db}dB'
    if NUM_MULTIPATH > 0:
        filename += f'_mp{NUM_MULTIPATH}'
    
    output_file = output_path / f'{filename}.png'
    plt.savefig(output_file, dpi=PLOT_DPI, bbox_inches='tight')
    print(f"✅ Plot saved: {output_file}")
    
    plt.show()


def main():
    """Main execution function."""
    print("🎯 Root-MUSIC Spectrum Comparison: Noisy vs UNet-Reconstructed")
    print("="*70)
    
    # Verify linear array
    if ARRAY_TYPE != "linear":
        print("❌ Error: Root-MUSIC only works with linear arrays!")
        print(f"   Current array type: {ARRAY_TYPE}")
        print("   Please set ARRAY_TYPE = 'linear' in the configuration.")
        return
    
    # Print configuration
    print(f"\n📋 Configuration:")
    print(f"   Sources: {NUM_SOURCES}")
    print(f"   Angles: {SOURCE_ANGLES if SOURCE_ANGLES else 'Random'}")
    print(f"   SNR: {SNR_DB} dB")
    print(f"   Multipath components: {NUM_MULTIPATH}")
    print(f"   Array: {NUM_ELEMENTS} element {ARRAY_TYPE}")
    print(f"   Angle range: {ANGLE_RANGE[0]}° to {ANGLE_RANGE[1]}° (step: {ANGLE_STEP}°)")
    
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
        # Generate random angles in the specified range
        angles = np.random.uniform(ANGLE_RANGE[0], ANGLE_RANGE[1], NUM_SOURCES).tolist()
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
    
    # Store multipath info in a format compatible with plotting if needed
    sample['metadata'] = {
        'multipath': {
            'angles': np.array(multipath_angles_extracted) if multipath_angles_extracted else None
        }
    }
    
    # Compute Root-MUSIC roots with noisy covariance
    print(f"\n🎯 Computing Root-MUSIC roots (noisy covariance)...")
    noisy_cov = sample['covariance_matrix']
    noisy_all_roots, noisy_selected_roots, noisy_doa_estimates = compute_rootmusic_roots(
        noisy_cov, array_model, NUM_SOURCES
    )
    print(f"✅ Noisy roots computed")
    print(f"   Total roots: {len(noisy_all_roots)}")
    print(f"   Selected roots: {len(noisy_selected_roots)}")
    print(f"   Root-MUSIC estimated DOAs: {[f'{a:.1f}°' for a in noisy_doa_estimates]}")
    
    # Compute Root-MUSIC roots with UNet-reconstructed covariance
    print(f"\n🧠 Computing Root-MUSIC roots (UNet-reconstructed covariance)...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    autocorr_input = create_autocorrelation_input_for_unet(sample, tau=AUTOCORR_TAU, device=device)
    autocorr_input = autocorr_input.unsqueeze(0)  # Add batch dimension
    
    with torch.no_grad():
        eigenvals, eigenvecs, reconstructed_cov = unet_model(autocorr_input)
    
    reconstructed_cov = reconstructed_cov.squeeze().cpu().numpy()
    
    reconstructed_all_roots, reconstructed_selected_roots, reconstructed_doa_estimates = compute_rootmusic_roots(
        reconstructed_cov, array_model, NUM_SOURCES
    )
    print(f"✅ Reconstructed roots computed")
    print(f"   Total roots: {len(reconstructed_all_roots)}")
    print(f"   Selected roots: {len(reconstructed_selected_roots)}")
    print(f"   UNet+Root-MUSIC estimated DOAs: {[f'{a:.1f}°' for a in reconstructed_doa_estimates]}")
    
    # Calculate RMSE (loss) for both methods
    print(f"\n📊 DOA Estimation Results & RMSE (Loss)")
    print(f"=" * 70)
    print(f"   True DOAs:                  {[f'{a:.1f}°' for a in angles]}")
    print(f"   Root-MUSIC Estimates:       {[f'{a:.1f}°' for a in noisy_doa_estimates]}")
    print(f"   UNet+Root-MUSIC Estimates:  {[f'{a:.1f}°' for a in reconstructed_doa_estimates]}")
    print(f"-" * 70)
    
    noisy_rmse = calculate_doa_rmse(angles, noisy_doa_estimates)
    reconstructed_rmse = calculate_doa_rmse(angles, reconstructed_doa_estimates)
    
    print(f"   Root-MUSIC RMSE:            {noisy_rmse:.3f}°")
    print(f"   UNet+Root-MUSIC RMSE:       {reconstructed_rmse:.3f}°")
    print(f"-" * 70)
    
    if reconstructed_rmse < noisy_rmse:
        improvement = ((noisy_rmse - reconstructed_rmse) / noisy_rmse) * 100
        print(f"   ✅ UNet improves accuracy by {improvement:.1f}%")
    elif reconstructed_rmse > noisy_rmse:
        degradation = ((reconstructed_rmse - noisy_rmse) / noisy_rmse) * 100
        print(f"   ⚠️ UNet degrades accuracy by {degradation:.1f}%")
    else:
        print(f"   ➡️ UNet has same accuracy as Root-MUSIC")
    print(f"=" * 70)
    
    # Plot comparison
    print(f"\n📊 Creating polar roots plot...")
    plot_polar_roots_comparison(
        noisy_all_roots=noisy_all_roots,
        noisy_selected_roots=noisy_selected_roots,
        reconstructed_all_roots=reconstructed_all_roots,
        reconstructed_selected_roots=reconstructed_selected_roots,
        noisy_doas=noisy_doa_estimates,
        reconstructed_doas=reconstructed_doa_estimates,
        true_angles=angles,
        output_dir=OUTPUT_DIR,
        snr_db=SNR_DB
    )
    
    print(f"\n✅ Comparison completed!")
    print(f"   Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
