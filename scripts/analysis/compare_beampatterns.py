#!/usr/bin/env python3
"""
Compare 2D Beampatterns: Noisy vs UNet-Reconstructed Covariance

This script generates a single signal sample and compares the 2D beampatterns
computed from:
1. Noisy covariance matrix (classical beamforming)
2. UNet-reconstructed covariance matrix (denoised beamforming)

The plot shows both 2D beampatterns side by side.

Usage:
    python compare_beampatterns.py
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

# Import UNet models
try:
    from models.deep_learning.EVDUNet import EVDCovarianceReconstructionUNet
    EVDUNET_AVAILABLE = True
except ImportError:
    EVDUNET_AVAILABLE = False
    print("⚠️ EVDUNet models not available")

# Import visualization tools
from reconunet.visualization.array_plotting import plot_2d_beampattern

# ============================================================================
# CONFIGURATION SECTION - MODIFY THESE PARAMETERS
# ============================================================================

# === SOURCE CONFIGURATION ===
NUM_SOURCES = 1  # Number of sources
SOURCE_ANGLES = [90]  # DOA angles in degrees (set to None for random)
SNR_DB = -10.0  # Signal-to-noise ratio in dB
NUM_SNAPSHOTS = 512  # Number of time snapshots

# === MULTIPATH CONFIGURATION ===
NUM_MULTIPATH = 3  # Number of multipath components (0 = no multipath)

# === ARRAY CONFIGURATION ===
ARRAY_TYPE = "linear"  # Array geometry
NUM_ELEMENTS = 8  # Number of array elements
CARRIER_FREQ = 2.45e9  # Carrier frequency in Hz
SAMPLING_FREQ = 1e6  # Sampling frequency in Hz
ARRAY_IMPERFECTIONS = False  # Enable array imperfections

# === BEAMPATTERN CONFIGURATION ===
BEAMPATTERN_SCAN_START = 30.0  # Start angle for beampattern scan
BEAMPATTERN_SCAN_STOP = 150.0   # End angle for beampattern scan
BEAMPATTERN_SCAN_STEP = 1.0     # Angle step for beampattern scan (degrees)

# === UNET MODEL CONFIGURATION ===
UNET_MODEL_PATH = 'notebooks/evd_unet_denoising_model_20250929_015132.pth'
AUTOCORR_TAU = 8  # Number of time lags (must match UNet training)

# === OUTPUT CONFIGURATION ===
OUTPUT_DIR = 'Tri4Net/src/evaluation/beampattern_comparison'
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
        dataset_name="temp_beampattern_comparison",
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


def compute_2d_beampattern(cov_matrix: np.ndarray,
                          array_model: ArrayModel,
                          scan_angles: np.ndarray) -> np.ndarray:
    """
    Compute 2D beampattern from covariance matrix.
    
    Parameters
    ----------
    cov_matrix : np.ndarray
        Covariance matrix of shape (M, M)
    array_model : ArrayModel
        Array model
    scan_angles : np.ndarray
        Scan angles in degrees
        
    Returns
    -------
    np.ndarray
        2D beampattern of shape (len(scan_angles), len(scan_angles))
    """
    # Get steering matrix
    A = array_model.steering_matrix(scan_angles, nominal=False)
    
    # Compute received signal steering vectors (same as signal steering matrix)
    X = A  # For this visualization, we use the same steering matrix
    
    # Get conjugate transpose of steering vectors
    AH = A.conj().T
    M = A.shape[0]  # number of elements
    
    # Calculate denominator term (normalization factor)
    denom = np.diag(AH @ A)  # shape: (K,)
    
    # Calculate 2D beampattern
    BP = np.zeros((len(scan_angles), len(scan_angles)), dtype=complex)
    
    for i in range(len(scan_angles)):
        # Get steering vector for this direction
        a = A[:, i:i+1]  # shape: (M, 1)
        aH = a.conj().T  # shape: (1, M)
        
        # Calculate beamformer output for all possible signal directions
        # Response = aH @ R @ a
        BP[i, :] = np.diag(AH @ cov_matrix @ A) / (denom[i] * np.trace(cov_matrix))
    
    return BP


def plot_beampattern_comparison(scan_angles: np.ndarray,
                                noisy_bp: np.ndarray,
                                reconstructed_bp: np.ndarray,
                                true_angles: List[float],
                                output_dir: Path,
                                snr_db: float):
    """
    Plot comparison of 2D beampatterns side by side.
    
    Parameters
    ----------
    scan_angles : np.ndarray
        Scan angles in degrees
    noisy_bp : np.ndarray
        2D beampattern from noisy covariance
    reconstructed_bp : np.ndarray
        2D beampattern from UNet-reconstructed covariance
    true_angles : list
        True DOA angles
    output_dir : Path
        Output directory
    snr_db : float
        SNR in dB
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Convert to dB scale
    epsilon = 1e-10
    noisy_bp_real = np.real(noisy_bp)
    reconstructed_bp_real = np.real(reconstructed_bp)
    
    noisy_bp_real[noisy_bp_real <= 0] = epsilon
    reconstructed_bp_real[reconstructed_bp_real <= 0] = epsilon
    
    noisy_db = 10 * np.log10(noisy_bp_real)
    reconstructed_db = 10 * np.log10(reconstructed_bp_real)
    
    # Clip to minimum dB
    min_db = -15
    noisy_db[noisy_db < min_db] = min_db
    reconstructed_db[reconstructed_db < min_db] = min_db
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot noisy beampattern
    im1 = axes[0].contourf(scan_angles, scan_angles, noisy_db, 20, cmap='jet')
    axes[0].set_xlabel('Signal Direction [degrees]', fontsize=12)
    axes[0].set_ylabel('Steering Direction [degrees]', fontsize=12)
    axes[0].set_title(f'Noisy Covariance 2D Beampattern\n(SNR = {snr_db} dB)', fontsize=13, fontweight='bold')
    
    # Add true angles as white lines
    for angle in true_angles:
        axes[0].axvline(x=angle, color='white', linestyle='--', linewidth=1.5, alpha=0.8)
        axes[0].axhline(y=angle, color='white', linestyle='--', linewidth=1.5, alpha=0.8)
    
    cbar1 = plt.colorbar(im1, ax=axes[0])
    cbar1.set_label('Magnitude [dB]', fontsize=11)
    
    # Plot reconstructed beampattern
    im2 = axes[1].contourf(scan_angles, scan_angles, reconstructed_db, 20, cmap='jet')
    axes[1].set_xlabel('Signal Direction [degrees]', fontsize=12)
    axes[1].set_ylabel('Steering Direction [degrees]', fontsize=12)
    axes[1].set_title(f'UNet-Reconstructed Covariance 2D Beampattern\n(SNR = {snr_db} dB)', fontsize=13, fontweight='bold')
    
    # Add true angles as white lines
    for angle in true_angles:
        axes[1].axvline(x=angle, color='white', linestyle='--', linewidth=1.5, alpha=0.8)
        axes[1].axhline(y=angle, color='white', linestyle='--', linewidth=1.5, alpha=0.8)
    
    cbar2 = plt.colorbar(im2, ax=axes[1])
    cbar2.set_label('Magnitude [dB]', fontsize=11)
    
    plt.tight_layout()
    
    # Save figure
    filename = f'beampattern_comparison_{NUM_SOURCES}src_snr{snr_db}dB'
    if NUM_MULTIPATH > 0:
        filename += f'_mp{NUM_MULTIPATH}'
    
    output_file = output_path / f'{filename}.png'
    plt.savefig(output_file, dpi=PLOT_DPI, bbox_inches='tight')
    print(f"✅ Plot saved: {output_file}")
    
    plt.show()


def main():
    """Main execution function."""
    print("📡 2D Beampattern Comparison: Noisy vs UNet-Reconstructed")
    print("="*70)
    
    # Print configuration
    print(f"\n📋 Configuration:")
    print(f"   Sources: {NUM_SOURCES}")
    print(f"   Angles: {SOURCE_ANGLES if SOURCE_ANGLES else 'Random'}")
    print(f"   SNR: {SNR_DB} dB")
    print(f"   Multipath components: {NUM_MULTIPATH}")
    print(f"   Array: {NUM_ELEMENTS} element {ARRAY_TYPE}")
    print(f"   Beampattern scan: {BEAMPATTERN_SCAN_START}° to {BEAMPATTERN_SCAN_STOP}° (step: {BEAMPATTERN_SCAN_STEP}°)")
    
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
    
    # Define scan angles for beampattern
    scan_angles = np.arange(BEAMPATTERN_SCAN_START, BEAMPATTERN_SCAN_STOP + BEAMPATTERN_SCAN_STEP, BEAMPATTERN_SCAN_STEP)
    
    # Compute 2D beampattern with noisy covariance
    print(f"\n📊 Computing 2D beampattern (noisy covariance)...")
    noisy_cov = sample['covariance_matrix']
    noisy_bp = compute_2d_beampattern(noisy_cov, array_model, scan_angles)
    print(f"✅ Noisy beampattern computed")
    
    # Compute 2D beampattern with UNet-reconstructed covariance
    print(f"\n🧠 Computing 2D beampattern (UNet-reconstructed covariance)...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    autocorr_input = create_autocorrelation_input_for_unet(sample, tau=AUTOCORR_TAU, device=device)
    autocorr_input = autocorr_input.unsqueeze(0)  # Add batch dimension
    
    with torch.no_grad():
        eigenvals, eigenvecs, reconstructed_cov = unet_model(autocorr_input)
    
    reconstructed_cov = reconstructed_cov.squeeze().cpu().numpy()
    
    reconstructed_bp = compute_2d_beampattern(reconstructed_cov, array_model, scan_angles)
    print(f"✅ Reconstructed beampattern computed")
    
    # Plot comparison
    print(f"\n📊 Creating comparison plot...")
    plot_beampattern_comparison(
        scan_angles=scan_angles,
        noisy_bp=noisy_bp,
        reconstructed_bp=reconstructed_bp,
        true_angles=angles,
        output_dir=OUTPUT_DIR,
        snr_db=SNR_DB
    )
    
    print(f"\n✅ Comparison completed!")
    print(f"   Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
