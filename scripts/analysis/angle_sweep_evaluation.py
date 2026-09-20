#!/usr/bin/env python3
"""
Angle Sweep Evaluation Script

This script evaluates DOA estimation algorithms by sweeping the angle of arrival
across a range while keeping the source separation constant.

For each angle position:
- Generate multiple samples (Monte Carlo simulation)
- Estimate DOAs using the selected algorithm
- Average the estimates and compute errors
- Plot estimated vs true angles

Usage:
    python angle_sweep_evaluation.py
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
from tqdm import tqdm
import json
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
from models.classic.esprit import ESPRIT
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
NUM_SOURCES = 3  # Number of sources to track
SOURCE_SEPARATION = 10.0  # Degrees between sources
NUM_SNAPSHOTS = 512  # Number of time snapshots

# === ANGLE SWEEP CONFIGURATION ===
ANGLE_START = 30.0  # Starting angle (degrees)
ANGLE_STOP = 150.0  # Ending angle (degrees)
ANGLE_STEP = 1.0  # Step size (degrees)

# === SNR CONFIGURATION ===
SNR_DB = 0.0  # Signal-to-noise ratio in dB

# === MONTE CARLO SIMULATION ===
SAMPLES_PER_ANGLE = 100  # Number of samples to generate per angle

# === ALGORITHM SELECTION ===
# Options: 'music', 'root_music', 'esprit', 'unet_music', 'unet_root_music', 'unet_esprit'
ALGORITHM = 'unet_esprit'

# === ARRAY CONFIGURATION ===
ARRAY_TYPE = "linear"  # Array geometry: 'linear', 'triangular', 'circular', 'cross'
NUM_ELEMENTS = 8  # Number of array elements
CARRIER_FREQ = 2.45e9  # Carrier frequency in Hz
SAMPLING_FREQ = 1e6  # Sampling frequency in Hz
ARRAY_IMPERFECTIONS = True  # Enable array imperfections (gain/phase errors, coupling)

# === UNET MODEL CONFIGURATION (if using UNet-based methods) ===
UNET_MODEL_PATH = 'notebooks/evd_unet_denoising_model_20250929_015132.pth'  # Relative to Tri4Net directory
UNET_MODEL_PATH = 'experiments/runs/reconunet/checkpoints/best.pt'  # relative to the repo root
AUTOCORR_TAU = 8  # Number of time lags (must match UNet training)

# === MUSIC GRID CONFIGURATION ===
MUSIC_SCAN_STEP = 1.0  # Scan angle step for MUSIC (degrees). Use 0.5 for fine, 2.0 for fast

# === OUTPUT CONFIGURATION ===
OUTPUT_DIR = 'src/evaluation/angle_sweep'  # Relative to Tri4Net directory
PLOT_DPI = 150  # Plot resolution (DPI)

# ============================================================================
# END OF CONFIGURATION SECTION
# ============================================================================


def create_autocorrelation_input_for_unet(sample, tau=8, device='cpu'):
    """Convert sample to UNet-compatible autocorrelation tensor."""
    if 'autocorrelation_matrix' in sample:
        autocorr_tensor = sample['autocorrelation_matrix']
        if not isinstance(autocorr_tensor, torch.Tensor):
            autocorr_tensor = torch.tensor(autocorr_tensor, dtype=torch.float32)
        return autocorr_tensor.to(device)
    
    # Compute from received signal
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


def estimate_doa_with_algorithm(sample: Dict, array_model: ArrayModel, 
                                algorithm: str, unet_model=None, 
                                device='cpu') -> np.ndarray:
    """
    Estimate DOA using the specified algorithm.
    
    Parameters
    ----------
    sample : dict
        Sample containing received signal and covariance matrix
    array_model : ArrayModel
        Array model for DOA estimation
    algorithm : str
        Algorithm name: 'music', 'root_music', 'esprit', 'unet_music', etc.
    unet_model : optional
        UNet model for denoising (if using UNet-based methods)
    device : str
        'cpu' or 'cuda'
        
    Returns
    -------
    np.ndarray
        Estimated DOA angles in degrees, sorted
    """
    num_sources = len([x for x in sample['labels']['doas'] if not np.isnan(x)])
    
    # Check if using UNet-based method
    use_unet = algorithm.startswith('unet_')
    if use_unet:
        base_algorithm = algorithm.replace('unet_', '')
    else:
        base_algorithm = algorithm
    
    # Get covariance matrix
    if use_unet and unet_model is not None:
        # Use UNet to reconstruct covariance
        autocorr_input = create_autocorrelation_input_for_unet(sample, tau=AUTOCORR_TAU, device=device)
        autocorr_input = autocorr_input.unsqueeze(0)
        
        with torch.no_grad():
            eigenvals, eigenvecs, reconstructed_cov = unet_model(autocorr_input)
        
        cov_matrix = reconstructed_cov.squeeze().cpu().numpy()
        # Ensure Hermitian
        cov_matrix = (cov_matrix + cov_matrix.conj().T) / 2
    else:
        # Use noisy covariance
        cov_matrix = sample['covariance_matrix']
    
    # Apply DOA estimation algorithm
    try:
        if base_algorithm == 'music':
            # Use MUSIC algorithm
            scan_angles = np.arange(30, 151, MUSIC_SCAN_STEP)
            music = MUSIC(
                array_model=array_model,
                scan_angles_deg=scan_angles,
                num_sources=num_sources
            )
            music.set_received_covariance(cov_matrix)
            doa_estimates, _ = music.estimate_doa()
            
        elif base_algorithm == 'root_music':
            # Use Root-MUSIC algorithm
            root_music = RootMUSIC(
                array_model=array_model,
                num_sources=num_sources
            )
            root_music.set_received_covariance(cov_matrix)
            doa_estimates, _ = root_music.estimate_doa()  # Root-MUSIC returns (angles, spectrum)
            
        elif base_algorithm == 'esprit':
            # Use ESPRIT algorithm
            esprit = ESPRIT(
                array_model=array_model,
                num_sources=num_sources
            )
            esprit.set_received_covariance(cov_matrix)
            doa_estimates, _ = esprit.estimate_doa()  # ESPRIT returns (angles, spectrum)
            
        else:
            raise ValueError(f"Unknown algorithm: {base_algorithm}")
        
        # Convert to numpy array and handle different return types
        if isinstance(doa_estimates, tuple):
            doa_estimates = np.array(doa_estimates)
        elif isinstance(doa_estimates, list):
            doa_estimates = np.array(doa_estimates)
        elif not isinstance(doa_estimates, np.ndarray):
            doa_estimates = np.array([doa_estimates])
        
        # Flatten if needed (ESPRIT sometimes returns nested arrays)
        if doa_estimates.ndim > 1:
            doa_estimates = doa_estimates.flatten()
        
        doa_estimates = np.sort(doa_estimates)
        
        return doa_estimates
        
    except Exception as e:
        print(f"⚠️ DOA estimation failed: {e}")
        import traceback
        traceback.print_exc()
        return np.array([np.nan] * num_sources)


def generate_samples_for_angle(base_angle: float, separation: float, 
                               num_sources: int, num_samples: int,
                               snr_db: float, array_config: ArrayConfig) -> List[Dict]:
    """
    Generate multiple samples for a specific angle configuration.
    This function replicates EXACTLY how dataset_generator.py creates samples.
    
    Parameters
    ----------
    base_angle : float
        Angle of first source (degrees)
    separation : float
        Angular separation between sources (degrees)
    num_sources : int
        Number of sources
    num_samples : int
        Number of Monte Carlo samples
    snr_db : float
        SNR in dB
    array_config : ArrayConfig
        Array configuration
        
    Returns
    -------
    List[Dict]
        List of generated samples
    """
    # Create angle configuration
    angles = [base_angle + i * separation for i in range(num_sources)]
    
    # Ensure angles are within valid range
    if max(angles) > 150.0 or min(angles) < 30.0:
        return []
    
    from signalgen.signal_generator import SignalConfig, SignalGenerator
    from signalgen.array_processing import ReceivedSignal
    import hashlib
    import time
    
    # Use base frequency (same as dataset_generator)
    base_freq = 100e3
    frequencies = [base_freq] * num_sources
    
    # Equal power for all sources (SIR = 0 dB) - same as dataset_generator
    sir_db_list = [0.0] * (num_sources - 1) if num_sources > 1 else []
    
    samples = []
    for sample_idx in range(num_samples):
        # Create a unique deterministic seed for each sample (similar to dataset_generator)
        seed_str = f"{base_angle}_{separation}_{num_sources}_{snr_db}_{sample_idx}_{time.time()}"
        base_seed = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
        
        # Create array model with seed (for reproducibility)
        array_model = ArrayModel(array_config, seed=base_seed)
        
        # Create signal configuration with EXACT parameters from dataset_generator
        signal_config = SignalConfig(
            fs=SAMPLING_FREQ,
            T=NUM_SNAPSHOTS,
            num_sources=num_sources,
            angles=angles,
            frequencies=frequencies,
            powers=None,  # Let it calculate from SIR - dataset_generator approach
            main_source_power=0.0,  # As in dataset_generator
            sir_db=sir_db_list,  # Equal power sources
            snr_db=snr_db,
            bandwidth=None,  # Will default to 10% of Nyquist - as in dataset_generator
            use_full_bandwidth=True,  # KEY PARAMETER from dataset_generator
            enable_multipath=False,  # Disable multipath for clean evaluation
            noise_type='gaussian'
        )
        
        # Generate signals with deterministic seed
        signal_generator = SignalGenerator(signal_config, seed=base_seed)
        t, signals = signal_generator.generate_signals()
        
        # Get steering matrix (nominal=True for no imperfections)
        # This is EXACTLY how dataset_generator does it
        steering_matrix = array_model.steering_matrix(signal_config.angles, nominal=True)
        
        # Create received signal using steering matrix directly (as in dataset_generator)
        received_signal = ReceivedSignal(steering_matrix, signals, signal_config.angles, signal_config)
        
        # Add frequency-selective noise - EXACT process from dataset_generator
        received_signal_with_noise = received_signal.add_noise(
            snr_db=snr_db,
            source_frequencies=signal_config.frequencies,
            signal_bandwidth=signal_config.bandwidth if hasattr(signal_config, 'bandwidth') else signal_config.fs * 0.1,
            sampling_frequency=signal_config.fs,
            frequency_selective=True
        )
        
        # Compute sample covariance matrix - EXACT computation from dataset_generator
        signals_matrix = received_signal_with_noise.array_signals
        cov_matrix = (signals_matrix @ signals_matrix.conj().T) / signals_matrix.shape[1]
        
        # Create sample dictionary - format consistent with dataset_generator
        sample = {
            'labels': {
                'doas': np.array(angles, dtype=np.float32),
                'snr': np.float32(snr_db),
                'num_snapshots': np.int32(NUM_SNAPSHOTS),
                'num_sources': np.array([num_sources, 0], dtype=np.int32),  # [main, interference]
                'num_multipath': np.int32(0)
            },
            'received_signal': signals_matrix.astype(np.complex64),
            'covariance_matrix': cov_matrix.astype(np.complex64)
        }
        
        samples.append(sample)
    
    return samples


def angle_sweep_evaluation(angle_range: Tuple[float, float, float],
                          num_sources: int,
                          separation: float,
                          snr_db: float,
                          samples_per_angle: int,
                          algorithm: str,
                          array_model: ArrayModel,
                          array_config: ArrayConfig,
                          unet_model=None) -> Dict:
    """
    Perform angle sweep evaluation.
    
    Returns
    -------
    Dict
        Results containing true angles, estimated angles, and errors
    """
    angle_start, angle_stop, angle_step = angle_range
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Generate angle positions
    angle_positions = np.arange(angle_start, angle_stop + angle_step, angle_step)
    
    results = {
        'angle_positions': [],
        'true_angles': [],  # For each position, list of true angles for each source
        'estimated_angles_mean': [],  # For each position, mean estimated angles
        'estimated_angles_std': [],  # Standard deviation of estimates
        'angle_errors_mean': [],  # Mean error for each source
        'angle_errors_std': []  # Std of error for each source
    }
    
    print(f"\n🎯 Angle Sweep Evaluation")
    print(f"{'='*70}")
    print(f"Algorithm: {algorithm}")
    print(f"Sources: {num_sources}, Separation: {separation}°")
    print(f"SNR: {snr_db} dB")
    print(f"Angle range: {angle_start}° to {angle_stop}° (step: {angle_step}°)")
    print(f"Samples per angle: {samples_per_angle}")
    print(f"{'='*70}\n")
    
    for base_angle in tqdm(angle_positions, desc="Sweeping angles"):
        # Add small random offset to make angles off-grid (not exactly on 1° grid)
        offset = np.random.uniform(-0.3, 0.3)  # Random offset between -0.3° and +0.3°
        base_angle_offset = base_angle + offset
        
        # Generate true angles with offset
        true_angles = np.array([base_angle_offset + i * separation for i in range(num_sources)])
        
        # Check if angles are valid
        if max(true_angles) > 150.0 or min(true_angles) < 30.0:
            continue
        
        # Generate samples using the offset base angle
        samples = generate_samples_for_angle(
            base_angle_offset, separation, num_sources, 
            samples_per_angle, snr_db, array_config
        )
        
        if len(samples) == 0:
            continue
        
        # Estimate DOAs for all samples
        all_estimates = []
        for sample in samples:
            estimates = estimate_doa_with_algorithm(
                sample, array_model, algorithm, unet_model, device
            )
            
            # Only keep valid estimates
            if not np.any(np.isnan(estimates)) and len(estimates) == num_sources:
                all_estimates.append(estimates)
        
        if len(all_estimates) == 0:
            continue
        
        all_estimates = np.array(all_estimates)  # Shape: (num_samples, num_sources)
        
        # Compute statistics
        mean_estimates = np.mean(all_estimates, axis=0)
        std_estimates = np.std(all_estimates, axis=0)
        
        # Compute errors (estimated - true)
        errors = all_estimates - true_angles  # Broadcasting
        mean_errors = np.mean(errors, axis=0)
        std_errors = np.std(errors, axis=0)
        
        # Store results
        results['angle_positions'].append(base_angle)
        results['true_angles'].append(true_angles)
        results['estimated_angles_mean'].append(mean_estimates)
        results['estimated_angles_std'].append(std_estimates)
        results['angle_errors_mean'].append(mean_errors)
        results['angle_errors_std'].append(std_errors)
        
        # Clean up memory
        del samples
        del all_estimates
        import gc
        gc.collect()
    
    # Convert to arrays
    for key in ['angle_positions', 'true_angles', 'estimated_angles_mean', 
                'estimated_angles_std', 'angle_errors_mean', 'angle_errors_std']:
        results[key] = np.array(results[key])
    
    return results


def plot_angle_sweep_results(results: Dict, output_dir: Path, 
                            algorithm: str, num_sources: int,
                            separation: float, snr_db: float):
    """
    Create plots similar to the provided images.
    
    Creates:
    1. Estimated vs True angles (main plot with zoomed inset)
    2. Angle errors vs sample index
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    angle_positions = results['angle_positions']
    true_angles = results['true_angles']
    estimated_angles = results['estimated_angles_mean']
    angle_errors = results['angle_errors_mean']
    
    # Color scheme
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']
    
    # ========================================================================
    # PLOT 1: Estimated vs True Angles with Zoomed Inset
    # ========================================================================
    
    fig = plt.figure(figsize=(12, 10))
    
    # Main plot
    ax_main = plt.subplot(1, 1, 1)
    
    for source_idx in range(num_sources):
        true_vals = true_angles[:, source_idx]
        est_vals = estimated_angles[:, source_idx]
        
        # Plot true angles (lines)
        ax_main.plot(angle_positions, true_vals, 
                    color=colors[source_idx % len(colors)],
                    linewidth=2, linestyle='-',
                    label=f'$\\theta_{{{source_idx+1}}}$')
        
        # Plot estimated angles (markers)
        ax_main.plot(angle_positions, est_vals,
                    color=colors[source_idx % len(colors)],
                    marker='o', linestyle='', markersize=4, alpha=0.6,
                    markerfacecolor='none',
                    label=f'$\\hat{{\\theta}}_{{{source_idx+1}}}$')
    
    ax_main.set_xlabel('Experiment Index', fontsize=14)
    ax_main.set_ylabel('Angle (degrees)', fontsize=14)
    ax_main.set_title(f'DOA Estimation: {algorithm.upper()}\n'
                     f'{num_sources} sources, {separation}° separation, SNR={snr_db} dB',
                     fontsize=16, fontweight='bold')
    ax_main.grid(True, alpha=0.3)
    ax_main.legend(loc='upper left', fontsize=12, ncol=2)
    
    # Create zoomed inset (showing a portion of the data)
    # Position: [left, bottom, width, height] in figure coordinates
    ax_inset = fig.add_axes([0.52, 0.15, 0.35, 0.25])
    
    # Choose middle section for zoom
    zoom_start = len(angle_positions) // 3
    zoom_end = 2 * len(angle_positions) // 3
    zoom_indices = slice(zoom_start, zoom_end)
    
    for source_idx in range(num_sources):
        true_vals = true_angles[zoom_indices, source_idx]
        est_vals = estimated_angles[zoom_indices, source_idx]
        zoom_positions = angle_positions[zoom_indices]
        
        ax_inset.plot(zoom_positions, true_vals,
                     color=colors[source_idx % len(colors)],
                     linewidth=2, linestyle='-')
        ax_inset.plot(zoom_positions, est_vals,
                     color=colors[source_idx % len(colors)],
                     marker='o', linestyle='', markersize=4, alpha=0.6,
                     markerfacecolor='none')
    
    ax_inset.grid(True, alpha=0.3)
    ax_inset.set_xlabel('Experiment Index', fontsize=10)
    ax_inset.set_ylabel('Angle (degrees)', fontsize=10)
    ax_inset.set_title('Zoomed View', fontsize=9)
    
    plt.tight_layout()
    
    plot_path = output_path / f'angle_sweep_{algorithm}_{num_sources}src_sep{separation}deg_snr{snr_db}dB.png'
    plt.savefig(plot_path, dpi=PLOT_DPI, bbox_inches='tight')
    print(f"✅ Plot saved: {plot_path}")
    plt.close(fig)
    import gc
    gc.collect()
    
    # ========================================================================
    # PLOT 2: Angle Errors
    # ========================================================================
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for source_idx in range(num_sources):
        errors = angle_errors[:, source_idx]
        
        ax.plot(angle_positions, errors,
               color=colors[source_idx % len(colors)],
               marker='o', linestyle='', markersize=4, alpha=0.6,
               markerfacecolor='none',
               label=f'Δθ_{source_idx+1} = θ_{source_idx+1} - θ̂_{source_idx+1}')
    
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Experiment Index', fontsize=14)
    ax.set_ylabel('Angle Error (degrees)', fontsize=14)
    ax.set_title(f'DOA Estimation Errors: {algorithm.upper()}\n'
                f'{num_sources} sources, {separation}° separation, SNR={snr_db} dB',
                fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    
    error_path = output_path / f'angle_errors_{algorithm}_{num_sources}src_sep{separation}deg_snr{snr_db}dB.png'
    plt.savefig(error_path, dpi=PLOT_DPI, bbox_inches='tight')
    print(f"✅ Error plot saved: {error_path}")
    plt.close(fig)
    import gc
    gc.collect()
    plt.close()
    
    # ========================================================================
    # Print Statistics
    # ========================================================================
    
    print(f"\n{'='*70}")
    print(f"📊 Results Summary")
    print(f"{'='*70}")
    print(f"Algorithm: {algorithm}")
    print(f"Number of angle positions: {len(angle_positions)}")
    print(f"\nPer-Source Statistics:")
    for source_idx in range(num_sources):
        mean_error = np.mean(angle_errors[:, source_idx])
        std_error = np.std(angle_errors[:, source_idx])
        rmse = np.sqrt(np.mean(angle_errors[:, source_idx]**2))
        
        print(f"\n  Source {source_idx + 1}:")
        print(f"    Mean Error: {mean_error:+.4f}°")
        print(f"    Std Error:  {std_error:.4f}°")
        print(f"    RMSE:       {rmse:.4f}°")


def save_results_json(results: Dict, output_dir: Path, 
                     algorithm: str, num_sources: int,
                     separation: float, snr_db: float,
                     array_imperfections: bool):
    """Save results summary to JSON file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    angle_positions = results['angle_positions']
    angle_errors = results['angle_errors_mean']
    
    # Compute statistics for each source
    per_source_stats = []
    for source_idx in range(num_sources):
        mean_error = float(np.mean(angle_errors[:, source_idx]))
        std_error = float(np.std(angle_errors[:, source_idx]))
        rmse = float(np.sqrt(np.mean(angle_errors[:, source_idx]**2)))
        
        per_source_stats.append({
            'source': source_idx + 1,
            'mean_error_deg': mean_error,
            'std_error_deg': std_error,
            'rmse_deg': rmse
        })
    
    # Build JSON structure
    array_type = "imperfect" if array_imperfections else "perfect"
    
    json_data = {
        'configuration': {
            'algorithm': algorithm,
            'num_sources': num_sources,
            'separation_deg': separation,
            'snr_db': snr_db,
            'num_angle_positions': len(angle_positions),
            'angle_range': {
                'start': float(angle_positions[0]),
                'stop': float(angle_positions[-1]),
                'step': float(angle_positions[1] - angle_positions[0]) if len(angle_positions) > 1 else 0.0
            },
            'array_imperfections': array_imperfections,
            'array_type': array_type
        },
        'results': {
            'per_source_statistics': per_source_stats,
            'overall': {
                'average_rmse_deg': float(np.mean([s['rmse_deg'] for s in per_source_stats]))
            }
        },
        'raw_data': {
            'angle_positions': angle_positions.tolist(),
            'mean_errors_per_angle': angle_errors.tolist()
        }
    }
    
    # Build descriptive filename
    # Format: results_{algorithm}_{num_sources}src_sep{separation}deg_snr{snr}dB_{array_type}.json
    results_file = output_path / f'results_{algorithm}_{num_sources}src_sep{separation}deg_snr{snr_db}dB_{array_type}.json'
    
    with open(results_file, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"✅ Results JSON saved: {results_file}")


def main():
    """Main execution function."""
    print("🚀 Angle Sweep DOA Evaluation")
    print("="*70)
    
    # Print working directory for debugging
    print(f"\n🔍 Working Directory: {os.getcwd()}")
    print(f"   Script Location: {Path(__file__).parent}")
    
    # Check numpy configuration
    print(f"\n🔬 NumPy Configuration:")
    print(f"   NumPy version: {np.__version__}")
    try:
        config = np.show_config()
        print(f"   BLAS/LAPACK info: {type(config)}")
    except:
        print(f"   BLAS/LAPACK info: Not available")
    
    # Print configuration
    print(f"\n📋 Configuration:")
    print(f"   Sources: {NUM_SOURCES}")
    print(f"   Separation: {SOURCE_SEPARATION}°")
    print(f"   SNR: {SNR_DB} dB")
    print(f"   Angle range: {ANGLE_START}° to {ANGLE_STOP}° (step: {ANGLE_STEP}°)")
    print(f"   Samples per angle: {SAMPLES_PER_ANGLE}")
    print(f"   Algorithm: {ALGORITHM}")
    print(f"   Array: {NUM_ELEMENTS} element {ARRAY_TYPE}")
    
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
    
    # Initialize UNet
    unet_model = None
    if ALGORITHM.startswith('unet_') and EVDUNET_AVAILABLE:
        print(f"\n🤖 Initializing UNet...")
        model_params = {
            'tau': AUTOCORR_TAU,
            'M': NUM_ELEMENTS,
            'activation_type': 'anti_rectifier',
            'use_dropout': True
        }
        unet_model = EVDCovarianceReconstructionUNet(**model_params)
        
        # Handle relative paths (relative to Tri4Net directory)
        model_path = Path(UNET_MODEL_PATH)
        if not model_path.is_absolute():
            model_path = Path(__file__).parent / model_path
        
        if model_path.exists():
            try:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                print(f"   Model path: {model_path.name}")
                print(f"   Device: {device}")
                
                # Load checkpoint
                checkpoint = torch.load(str(model_path), map_location=device)
                
                # Check if it's a full checkpoint or just state_dict
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    # Full checkpoint format
                    unet_model.load_state_dict(checkpoint['model_state_dict'])
                    epoch = checkpoint.get('epoch', 'unknown')
                    print(f"✅ Loaded UNet model from checkpoint (epoch {epoch})")
                    if 'model_config' in checkpoint:
                        print(f"   Model config: {checkpoint['model_config']}")
                else:
                    # Direct state_dict format
                    unet_model.load_state_dict(checkpoint)
                    print(f"✅ Loaded UNet model from {model_path.name}")
                
                unet_model = unet_model.to(device)
                unet_model.eval()
            except Exception as e:
                print(f"❌ Error loading UNet model: {e}")
                print(f"   Disabling UNet evaluation")
                unet_model = None
        else:
            print(f"⚠️ UNet model not found at {model_path}")
            print(f"   Disabling UNet evaluation")
            unet_model = None
    
    # Run angle sweep evaluation
    results = angle_sweep_evaluation(
        angle_range=(ANGLE_START, ANGLE_STOP, ANGLE_STEP),
        num_sources=NUM_SOURCES,
        separation=SOURCE_SEPARATION,
        snr_db=SNR_DB,
        samples_per_angle=SAMPLES_PER_ANGLE,
        algorithm=ALGORITHM,
        array_model=array_model,
        array_config=array_config,
        unet_model=unet_model
    )
    
    # Create plots
    print(f"\n📊 Creating plots...")
    plot_angle_sweep_results(
        results,
        OUTPUT_DIR,
        ALGORITHM,
        NUM_SOURCES,
        SOURCE_SEPARATION,
        SNR_DB
    )
    
    # Save results to JSON
    print(f"\n💾 Saving results to JSON...")
    save_results_json(
        results,
        OUTPUT_DIR,
        ALGORITHM,
        NUM_SOURCES,
        SOURCE_SEPARATION,
        SNR_DB,
        ARRAY_IMPERFECTIONS
    )
    
    print(f"\n✅ Evaluation completed!")
    print(f"   Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
