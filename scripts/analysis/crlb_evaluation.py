#!/usr/bin/env python3
"""
CRLB Computation and Visualization Script

This script:
1. Uses existing dataset generation infrastructure (SimpleDatasetConfig, ControlledDatasetGenerator)
2. Generates dataset with specified parameters
3. Computes CRLB for each sample using the Slepian-Bangs formula
4. Optionally evaluates MUSIC algorithm for comparison
5. Plots CRLB vs SNR (averaged over angles)

CRLB Formula (deterministic signal model, unit-norm steering vectors):
- Steering vector: a(θ) = (1/√M) * exp(+jπ m cos(θ))
- Derivative: da/dθ = -jπ m sin(θ) a_m(θ)  [negative sign from d/dθ(cos(θ)) = -sin(θ)]
- Covariance: R(θ) = p·aa^H + σ²I, where p=1 (signal power)
- Derivative of R: R' = p·(da·a^H + a·da^H)
- Fisher Information: J(θ) = T * Tr[(R^{-1} R')²]
- CRLB: 1/J(θ) (radians²)

NOTE: MUSIC may appear to "beat" the CRLB at high SNR because:
1. CRLB is for UNBIASED estimators; MUSIC uses parabolic interpolation (biased but efficient)
2. Biased estimators can have MSE < CRLB when bias²+variance < unbiased variance
3. MUSIC's subspace method + interpolation is nearly optimal at high SNR
4. This is expected behavior and doesn't invalidate the CRLB!

MULTI-SOURCE SUPPORT:
The script automatically detects the number of sources from num_sources configuration:
- Single source (K=1): Uses three methods (stochastic, conditional, closed-form)
- Multiple sources (K>1): Uses conditional/deterministic multi-source CRLB:
  
  Fisher Information Matrix: J = (2T/σ²) * Re[(D^H Π⊥ D) ⊙ P^T]
  where:
  - D = [d₁, ..., dₖ] derivative matrix
  - Π⊥ = I - A(A^H A)^{-1}A^H (projector orthogonal to ALL sources)
  - P = diag(P₁, ..., Pₖ) source covariance (diagonal for uncorrelated)
  
  Returns average CRLB over all sources. Close sources degrade significantly!

Usage:
    python crlb_evaluation.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
import json
from tqdm import tqdm
import time

# Try to import torch for tensor handling
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ============================================================================
# DATASET CONFIGURATION - MODIFY THESE PARAMETERS AS NEEDED
# ============================================================================

# === PARAMETER RANGES ===
# Choose ONE of the following angle configurations:

# Option 1: Integer grid (on-grid angles)
angles_deg = list(range(30, 151, 1))  # Will sample uniformly from this range

# Option 2: Single angle for testing
# angles_deg = [90.1]

# Option 3: Random float angles (off-grid angles)
# Generate random float angles between 30 and 150 degrees
np.random.seed(int(time.time()))  # For reproducibility, set to None for different angles each run
num_random_angles = 100  # Number of random angles to generate
# angles_deg = list(np.random.uniform(30.0, 150.0, num_random_angles))

snr_db = list(range(-20, 21, 5))  # -20, -15, -10, -5, 0, 5, 10, 15, 20
num_snapshots = [512]  # Fixed snapshot length
num_sources = [[1, 0]]  # Number of sources: [main_sources, interference]. Examples: [[1,0]] single, [[2,0]] two sources
sir_db = [0]
num_multipath = [0]  # No multipath for clean CRLB
array_imperfections = [False]  # Perfect array for theoretical CRLB

# === GENERATION CONTROL ===
examples_per_combination = 10  # 1000 samples per SNR to average CRLB
dataset_name = "crlb_evaluation_dataset"

# === ARRAY CONFIGURATION ===
array_type = "linear"
num_elements = 8
carrier_freq = 2.45e9
sampling_freq = 1e6

# === OUTPUT CONTROL ===
save_received_signal = True
save_covariance_matrix = True  # Required for MUSIC evaluation
save_clean_covariance_matrix = False
save_autocorrelation_matrix = False
save_music_estimates = False
save_steering_vectors = True
save_array_metadata = False

# === ANGLE SEPARATION CONTROL ===
min_angle_separation_deg = 10.0  # Not used for single source

# === CRLB COMPUTATION PARAMETERS ===
T_snapshots = num_snapshots[0]  # Number of snapshots for CRLB calculation

# === ALGORITHM EVALUATION CONTROL ===
EVALUATE_MUSIC = True  # Set to True to evaluate MUSIC algorithm alongside CRLB
EVALUATE_ESPRIT = True  # Set to True to evaluate ESPRIT algorithm alongside CRLB
EVALUATE_MVDR = True  # Set to True to evaluate MVDR (Capon) algorithm alongside CRLB
EVALUATE_BEAMFORMER = True  # Set to True to evaluate conventional Beamformer alongside CRLB
EVALUATE_ROOT_MUSIC = True  # Set to True to evaluate Root-MUSIC algorithm alongside CRLB
EVALUATE_UNITARY_ESPRIT = True  # Set to True to evaluate Unitary ESPRIT algorithm alongside CRLB
scan_angles = list(range(30, 151, 1))  # Angle grid for MUSIC/MVDR/Beamformer spectrum

# === OUTPUT CONFIGURATION ===
output_dir = 'Tri4Net/src/evaluation/crlb_evaluation'

# === DATASET PATH CONFIGURATION ===
USE_EXISTING_DATASET = False
EXISTING_DATASET_PATH = 'Tri4Net/Data/datasets/linear/crlb_evaluation_dataset/crlb_evaluation_dataset.h5'

# === DATASET GENERATION SEED ===
DATASET_GENERATION_SEED = int(time.time())  # None = random, 42 = reproducible

# ============================================================================
# END OF CONFIGURATION SECTION
# ============================================================================

# Add src to path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

# Import dataset infrastructure
from data.dataset_generator import SimpleDatasetConfig, ControlledDatasetGenerator, DOADataset

# Import signal generation for array model
from signalgen import ArrayConfig, ArrayModel

# Import DOA algorithms
from models.classic.music import MUSIC
from models.classic.esprit import ESPRIT
from models.classic.mvdr import MVDR
from models.classic.beamformer import Beamformer
from models.classic.rootmusic import RootMUSIC
from models.classic.unitary_esprit import UnitaryESPRIT


def calculate_doa_rmse(true_angles, estimated_angles, penalty_for_missed=60.0):
    """
    Calculate RMSE between true and estimated DOAs.
    
    Args:
        true_angles: Array of true DOA angles
        estimated_angles: Array of estimated DOA angles  
        penalty_for_missed: Penalty (in degrees) for missed detections
        
    Returns:
        RMSE in degrees
    """
    true_angles = np.array(true_angles)
    estimated_angles = np.array(estimated_angles)
    
    # If MUSIC found no sources, this is a complete miss - use penalty
    if len(estimated_angles) == 0:
        return penalty_for_missed
    
    # If number of estimates doesn't match, use penalty for the missed ones
    if len(estimated_angles) < len(true_angles):
        # MUSIC found fewer sources than true
        true_sorted = np.sort(true_angles)
        est_sorted = np.sort(estimated_angles)
        
        # Match the estimated ones
        errors = []
        for i in range(len(estimated_angles)):
            errors.append(abs(true_sorted[i] - est_sorted[i]))
        
        # Add penalty for missed sources
        for i in range(len(estimated_angles), len(true_angles)):
            errors.append(penalty_for_missed)
            
        return np.sqrt(np.mean(np.array(errors)**2))
    
    # Normal case: same number or MUSIC found more (just use first N)
    min_sources = min(len(true_angles), len(estimated_angles))
    true_sorted = np.sort(true_angles[:min_sources])
    est_sorted = np.sort(estimated_angles[:min_sources])
    errors = np.abs(true_sorted - est_sorted)
    
    return np.sqrt(np.mean(errors**2))


def unit_norm_steering_vector(theta_rad, M):
    """
    Compute UNIT-NORM steering vector for ULA at angle theta (in radians).
    Using COSINE convention to MATCH DATASET: θ ∈ [0, π]
    
    This matches ArrayModel convention:
    a_m(θ) = (1/√M) * exp(+jπ m cos(θ))
    
    Note: Positive sign in exponent matches the dataset's nominal steering vectors.
    
    Returns:
        Complex array of shape (M,) with ||a|| = 1
    """
    m = np.arange(M)
    # Positive sign to match dataset convention
    a = (1.0 / np.sqrt(M)) * np.exp(1j * np.pi * m * np.cos(theta_rad))
    return a


def unit_norm_steering_vector_derivative(a, theta_rad, M):
    """
    Compute derivative of UNIT-NORM steering vector w.r.t. theta.
    
    For ArrayModel convention: a[m] = exp(+jπm·cos(θ))
    The derivative is: da/dθ = -jπ m sin(θ) * a_m(θ)
    
    Note: The negative sign comes from d/dθ[cos(θ)] = -sin(θ)
    
    Returns:
        Complex array of shape (M,)
    """
    m = np.arange(M)
    # Derivative with NEGATIVE sign to match dataset convention exp(+jπm·cos(θ))
    d = -1j * np.pi * m * np.sin(theta_rad) * a
    return d


def compute_crlb_multiple_sources(theta_rad_array, M, T, SNR_dB, steering_vectors=None, 
                                 source_powers=None, method='conditional', verbose=False):
    """
    Compute CRLB for MULTIPLE sources using the conditional/deterministic formulation.
    
    For K sources with angles θ = [θ₁, ..., θₖ]:
    - A = [a₁, ..., aₖ] with aₖ = a(θₖ) (unit-norm columns)
    - D = [d₁, ..., dₖ] with dₖ = ∂a(θₖ)/∂θ
    - Π⊥ = I - A(A^H A)^{-1}A^H (projector orthogonal to ALL steering vectors)
    - P = diag(P₁, ..., Pₖ) source covariance (diagonal for uncorrelated sources)
    
    Fisher Information Matrix (K×K):
    J_cond(θ) = (2T/σ²) * Re[(D^H Π⊥ D) ⊙ P^T]
    
    CRLB(θ) = J_cond^{-1}, per-angle variance bounds are diagonal elements
    
    Args:
        theta_rad_array: Array of K angles in radians
        M: Number of array elements
        T: Number of snapshots
        SNR_dB: Signal-to-Noise Ratio in dB (applies to all sources)
        steering_vectors: Optional pre-computed steering vectors (shape: [M, K])
        source_powers: Optional array of source powers [P₁, ..., Pₖ] (default: all 1.0)
        method: 'conditional' or 'stochastic' (currently only conditional implemented)
    
    Returns:
        Array of CRLB values for each source in radians² (diagonal of J^{-1})
    """
    K = len(theta_rad_array)
    
    # Default: equal power for all sources
    if source_powers is None:
        source_powers = np.ones(K)
    
    # Noise power from SNR (assuming reference power P=1)
    sigma2 = 10**(-SNR_dB / 10)
    
    # Build steering matrix A (M×K)
    A = np.zeros((M, K), dtype=complex)
    using_dataset_vectors = steering_vectors is not None
    for k in range(K):
        if steering_vectors is not None:
            # Use provided steering vector from dataset (normalized to unit norm)
            A[:, k] = steering_vectors[:, k] / np.linalg.norm(steering_vectors[:, k])
        else:
            # Compute unit-norm steering vector
            A[:, k] = unit_norm_steering_vector(theta_rad_array[k], M)
    
    if using_dataset_vectors and verbose:
        print(f"  Multi-source CRLB: Using steering vectors from dataset (K={K} sources)")
    
    # Build derivative matrix D (M×K)
    D = np.zeros((M, K), dtype=complex)
    for k in range(K):
        D[:, k] = unit_norm_steering_vector_derivative(A[:, k], theta_rad_array[k], M)
    
    # Build projector Π⊥ = I - A(A^H A)^{-1}A^H
    I = np.eye(M)
    AH_A = A.conj().T @ A
    try:
        AH_A_inv = np.linalg.inv(AH_A)
    except np.linalg.LinAlgError:
        # Sources too close, matrix is singular
        return np.full(K, np.inf)
    
    Pi_perp = I - A @ AH_A_inv @ A.conj().T
    
    # Compute D^H Π⊥ D (K×K matrix)
    DH_Pi_D = D.conj().T @ Pi_perp @ D
    
    # Build source covariance matrix P (K×K, diagonal for uncorrelated)
    P = np.diag(source_powers)
    
    # Fisher Information Matrix: J = (2T/σ²) * Re[(D^H Π⊥ D) ⊙ P^T]
    # Hadamard product
    J_matrix = (2.0 * T / sigma2) * np.real(DH_Pi_D * P.T)
    
    # Invert to get CRLB matrix
    try:
        CRLB_matrix = np.linalg.inv(J_matrix)
    except np.linalg.LinAlgError:
        # Matrix is singular
        return np.full(K, np.inf)
    
    # Extract diagonal elements (per-angle variance bounds)
    crlb_values = np.diag(CRLB_matrix)
    
    # Return as array
    return crlb_values


def compute_crlb_single_source(theta_rad, M, T, SNR_dB, steering_vector=None, method='all'):
    """
    Compute CRLB for a single source using THREE different formulations.
    
    Three methods:
    1. Unconditional/Stochastic Slepian-Bangs: R = a·a^H + σ²I (NO signal power in R)
    2. Conditional/Deterministic: J = 2T·(P/σ²)·d^H·(I - a·a^H)·d
    3. Closed-form ULA (from literature): CRLB = 6(1+Mρ) / [NM²ρ²(M²-1)[2π(d/λ)sin(θ)]²]
    
    Signal model:
    - Unit-norm steering vector: ||a(θ)|| = 1
    - Signal power: P = 1
    - Noise power: σ² = 10^(-SNR_dB/10)
    - SNR: ρ = P/σ² = 10^(SNR_dB/10)
    
    Args:
        theta_rad: Angle in radians
        M: Number of array elements
        T: Number of snapshots (N in literature)
        SNR_dB: Signal-to-Noise Ratio in dB
        steering_vector: Optional pre-computed steering vector from dataset (shape: [M,])
        method: 'stochastic', 'conditional', 'closedform', or 'all' (returns dict)
    
    Returns:
        If method='all': dict with {'stochastic': float, 'conditional': float, 'closedform': float}
        Otherwise: CRLB in radians² (variance)
    """
    # Signal power (fixed)
    P = 1.0
    
    # Noise power from SNR
    rho = 10**(SNR_dB / 10)  # SNR = P/σ²
    sigma2 = P / rho
    
    # Use provided steering vector or compute it
    if steering_vector is not None:
        # Normalize the provided steering vector to ensure unit norm
        a = steering_vector / np.linalg.norm(steering_vector)
    else:
        # Compute steering vector (unit-norm)
        a = unit_norm_steering_vector(theta_rad, M)
    
    # Always compute derivative from theta (not available in dataset)
    d = unit_norm_steering_vector_derivative(a, theta_rad, M)
    
    # Covariance matrix: R = aa^H + σ²I
    # R^{-1} = ρ(I - ρ/(1+ρ) aa^H)
    # Using ρ = 1/σ² for numerical stability
    I = np.eye(M)
    aaH = np.outer(a, a.conj())
    
    # === METHOD 1: UNCONDITIONAL/STOCHASTIC (Slepian-Bangs) ===
    # Uses: R = a·a^H + σ²I (NO signal power P in the covariance!)
    # This is the "stochastic signal" model where signal itself is random
    # R^{-1} = (1/σ²)(I - (1/(σ²+1))·a·a^H)  [Woodbury identity]
    R_inv_stochastic = (1.0 / sigma2) * (I - (1.0 / (sigma2 + 1.0)) * aaH)
    
    # R' = da·a^H + a·da^H (NO signal power factor!)
    R_prime_stochastic = np.outer(d, a.conj()) + np.outer(a, d.conj())
    
    # Fisher Information: J = T * Tr[(R^{-1} R')²]
    temp_stochastic = R_inv_stochastic @ R_prime_stochastic
    J_stochastic = T * np.real(np.trace(temp_stochastic @ temp_stochastic))
    
    crlb_stochastic = np.inf if J_stochastic <= 0 else (1.0 / J_stochastic)
    
    # === METHOD 2: CONDITIONAL/DETERMINISTIC ===
    # Eliminates unknown signal waveform, assumes deterministic signal with known power
    # J = 2T·(P/σ²)·d^H·(I - a·a^H)·d
    # where P=1 is the signal power, (I - a·a^H) is projection orthogonal to a
    P_perp = I - aaH  # Projection onto subspace orthogonal to a
    quad = np.real(np.conj(d) @ (P_perp @ d))
    J_conditional = 2.0 * T * rho * quad  # rho = P/σ² = SNR
    
    crlb_conditional = np.inf if J_conditional <= 0 else (1.0 / J_conditional)
    
    # === METHOD 3: CLOSED-FORM ULA FORMULA (from literature) ===
    # CRLB = 6(1 + Mρ) / [NM²ρ²(M²-1)[2π(d/λ)sin(θ)]²]
    # where d/λ = 0.5 for half-wavelength spacing
    # ||d||² = (M²-1)(1/12)[2π(d/λ)sin(θ)]² for ULA
    #
    # General form: CRLB = (1 + Mρ) / [2NM²ρ²||d||²]
    
    # Compute ||d||² directly from the derivative vector
    d_norm_sq = np.real(np.dot(d.conj(), d))
    
    # Closed-form formula
    numerator = 1.0 + M * rho
    denominator = 2.0 * T * M**2 * rho**2 * d_norm_sq
    crlb_closedform = numerator / denominator if denominator > 0 else np.inf
    
    # Return based on requested method
    if method == 'all':
        return {
            'stochastic': crlb_stochastic, 
            'conditional': crlb_conditional,
            'closedform': crlb_closedform
        }
    elif method == 'both':
        return {'stochastic': crlb_stochastic, 'conditional': crlb_conditional}
    elif method == 'stochastic':
        return crlb_stochastic
    elif method == 'conditional':
        return crlb_conditional
    elif method == 'closedform':
        return crlb_closedform
    else:
        # Default: return all three
        return {
            'stochastic': crlb_stochastic, 
            'conditional': crlb_conditional,
            'closedform': crlb_closedform
        }


def generate_crlb_dataset() -> str:
    """Generate dataset for CRLB evaluation."""
    print("🚀 Generating CRLB Evaluation Dataset...")
    print("=" * 60)
    
    # Create dataset configuration
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
        
        # Output control (minimal - we only need angles and SNR)
        save_received_signal=save_received_signal,
        save_covariance_matrix=save_covariance_matrix,
        save_clean_covariance_matrix=save_clean_covariance_matrix,
        save_autocorrelation_matrix=save_autocorrelation_matrix,
        save_music_estimates=save_music_estimates,
        save_steering_vectors=save_steering_vectors,
        save_array_metadata=save_array_metadata,
        
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


def compute_crlb_for_dataset(dataset: DOADataset, M: int, T: int, array_model: ArrayModel = None, 
                            evaluate_music: bool = False, evaluate_esprit: bool = False,
                            evaluate_mvdr: bool = False, evaluate_beamformer: bool = False,
                            evaluate_root_music: bool = False, evaluate_unitary_esprit: bool = False) -> Dict:
    """
    Compute CRLB for all samples in the dataset.
    Optionally also evaluate DOA algorithms.
    
    Args:
        dataset: DOADataset instance
        M: Number of array elements
        T: Number of snapshots
        array_model: ArrayModel instance (required if evaluating any algorithm)
        evaluate_music: Whether to evaluate MUSIC algorithm alongside CRLB
        evaluate_esprit: Whether to evaluate ESPRIT algorithm alongside CRLB
        evaluate_mvdr: Whether to evaluate MVDR algorithm alongside CRLB
        evaluate_beamformer: Whether to evaluate conventional Beamformer alongside CRLB
        evaluate_root_music: Whether to evaluate Root-MUSIC algorithm alongside CRLB
        evaluate_unitary_esprit: Whether to evaluate Unitary ESPRIT algorithm alongside CRLB
    
    Returns:
        Dictionary mapping SNR_dB -> {'crlb': list, 'music_rmse': list (optional), 
                                       'esprit_rmse': list (optional), 'mvdr_rmse': list (optional),
                                       'beamformer_rmse': list (optional), 'root_music_rmse': list (optional),
                                       'unitary_esprit_rmse': list (optional)}
    """
    print("\n🎯 Computing CRLB for Dataset...")
    
    # Verify number of sources
    num_sources_total = np.array(num_sources[0]).sum()
    print(f"📍 Number of sources per sample: {num_sources_total}")
    if num_sources_total == 1:
        print(f"   Using single-source CRLB (three methods: stochastic, conditional, closed-form)")
    else:
        print(f"   Using multi-source conditional CRLB (averaged over {num_sources_total} sources)")
    
    algorithms_to_eval = []
    if evaluate_music:
        algorithms_to_eval.append("MUSIC")
    if evaluate_esprit:
        algorithms_to_eval.append("ESPRIT")
    if evaluate_mvdr:
        algorithms_to_eval.append("MVDR")
    if evaluate_beamformer:
        algorithms_to_eval.append("Beamformer")
    if evaluate_root_music:
        algorithms_to_eval.append("Root-MUSIC")
    if evaluate_unitary_esprit:
        algorithms_to_eval.append("Unitary-ESPRIT")
    
    if algorithms_to_eval:
        print(f"📊 Also evaluating: {', '.join(algorithms_to_eval)}")
    print("=" * 60)
    
    # Use SNR values from dataset configuration
    all_snr_values = sorted(snr_db)
    
    print(f"📊 Computing CRLB for SNR levels: {all_snr_values}")
    print(f"   Array elements: M = {M}")
    print(f"   Snapshots: T = {T}")
    if evaluate_music or evaluate_mvdr:
        print(f"   Scan angles: [{scan_angles[0]}, {scan_angles[-1]}] ({len(scan_angles)} points)")
    
    crlb_by_snr = {}
    
    for snr_value in all_snr_values:
        print(f"\n🎯 Processing SNR = {snr_value} dB")
        
        # Filter dataset for this SNR
        filtered_dataset = dataset.filter(snr=snr_value)
        num_samples = len(filtered_dataset)
        print(f"   Processing {num_samples} samples...")
        
        crlb_values = []
        music_rmse_values = []
        esprit_rmse_values = []
        mvdr_rmse_values = []
        beamformer_rmse_values = []
        root_music_rmse_values = []
        unitary_esprit_rmse_values = []
        
        for idx in tqdm(range(num_samples), desc=f"SNR {snr_value}dB"):
            sample = filtered_dataset[idx]
            
            # Extract true angle (in degrees, convert to radians)
            all_doas = sample['labels']['doas']
            num_sources_sample = np.array(num_sources[0]).sum()
            true_doas_clean = all_doas[~np.isnan(all_doas)][:num_sources_sample]
            
            if len(true_doas_clean) == 0:
                continue
            
            # Extract steering vectors from dataset if available
            steering_vectors_all = None
            if 'steering_vectors' in sample:
                # steering_vectors shape: [M, num_sources] (complex) - stored as columns!
                steering_vectors_all = sample['steering_vectors']['nominal']
                if TORCH_AVAILABLE and isinstance(steering_vectors_all, torch.Tensor):
                    steering_vectors_all = steering_vectors_all.cpu().numpy()
                
                # Extract only the columns for the actual number of sources we're using
                if steering_vectors_all.ndim == 2 and steering_vectors_all.shape[1] >= num_sources_sample:
                    steering_vectors_all = steering_vectors_all[:, :num_sources_sample]
            
            # Compute CRLB based on number of sources
            if num_sources_sample == 1:
                # Single source case - use original three-method approach
                theta_deg = true_doas_clean[0]
                theta_rad = np.radians(theta_deg)
                
                steering_vector = None
                if steering_vectors_all is not None and steering_vectors_all.ndim == 2:
                    steering_vector = steering_vectors_all[:, 0]  # First column
                
                crlb_dict = compute_crlb_single_source(theta_rad, M, T, snr_value, 
                                                        steering_vector=steering_vector, 
                                                        method='all')
                crlb_values.append(crlb_dict)
            else:
                # Multiple sources case - use multi-source conditional CRLB
                theta_rad_array = np.radians(true_doas_clean)
                
                # Use multi-source CRLB with steering vectors from dataset (returns array of per-source bounds)
                crlb_array = compute_crlb_multiple_sources(
                    theta_rad_array, M, T, snr_value,
                    steering_vectors=steering_vectors_all,  # [M, K] matrix from dataset
                    source_powers=None,  # Equal power
                    method='conditional',
                    verbose=(idx == 0)  # Print confirmation for first sample only
                )
                
                # For multi-source, store as conditional only (average over sources)
                crlb_dict = {
                    'stochastic': np.mean(crlb_array),  # Use conditional for all three
                    'conditional': np.mean(crlb_array),
                    'closedform': np.mean(crlb_array)
                }
                crlb_values.append(crlb_dict)
            
            # Get covariance matrix (needed for all algorithms)
            received_cov = None
            if (evaluate_music or evaluate_esprit or evaluate_mvdr or evaluate_beamformer or 
                evaluate_root_music or evaluate_unitary_esprit) and array_model is not None:
                if 'covariance_matrix' in sample:
                    received_cov = sample['covariance_matrix']
                    if TORCH_AVAILABLE and isinstance(received_cov, torch.Tensor):
                        received_cov = received_cov.cpu().numpy()
            
            # Evaluate MUSIC if requested
            if evaluate_music and array_model is not None and received_cov is not None:
                try:
                    music = MUSIC(array_model, array_manifold=None, 
                                scan_angles_deg=np.array(scan_angles), 
                                num_sources=num_sources_sample)
                    music.set_received_covariance(received_cov)
                    estimated_doas, _ = music.estimate_doa()
                    
                    rmse = calculate_doa_rmse(true_doas_clean, estimated_doas)
                    music_rmse_values.append(rmse)
                except Exception as e:
                    music_rmse_values.append(np.inf)
            
            # Evaluate ESPRIT if requested
            if evaluate_esprit and array_model is not None and received_cov is not None:
                try:
                    esprit = ESPRIT(array_model, num_sources=num_sources_sample)
                    esprit.set_received_covariance(received_cov)
                    estimated_doas, _ = esprit.estimate_doa()  # ESPRIT returns (doas, spectrum)
                    
                    rmse = calculate_doa_rmse(true_doas_clean, estimated_doas)
                    esprit_rmse_values.append(rmse)
                except Exception as e:
                    esprit_rmse_values.append(np.inf)
            
            # Evaluate MVDR if requested
            if evaluate_mvdr and array_model is not None and received_cov is not None:
                try:
                    mvdr = MVDR(array_model, array_manifold=None,
                               scan_angles_deg=np.array(scan_angles),
                               num_sources=num_sources_sample)
                    mvdr.set_received_covariance(received_cov)
                    estimated_doas, _ = mvdr.estimate_doa()
                    
                    rmse = calculate_doa_rmse(true_doas_clean, estimated_doas)
                    mvdr_rmse_values.append(rmse)
                except Exception as e:
                    mvdr_rmse_values.append(np.inf)
            
            # Evaluate Beamformer if requested
            if evaluate_beamformer and array_model is not None and received_cov is not None:
                try:
                    beamformer = Beamformer(array_model, array_manifold=None,
                                           scan_angles_deg=np.array(scan_angles),
                                           num_sources=num_sources_sample)
                    beamformer.set_received_covariance(received_cov)
                    estimated_doas, _ = beamformer.estimate_doa()
                    
                    rmse = calculate_doa_rmse(true_doas_clean, estimated_doas)
                    beamformer_rmse_values.append(rmse)
                except Exception as e:
                    beamformer_rmse_values.append(np.inf)
            
            # Evaluate Root-MUSIC if requested
            if evaluate_root_music and array_model is not None and received_cov is not None:
                try:
                    root_music = RootMUSIC(array_model, num_sources=num_sources_sample)
                    root_music.set_received_covariance(received_cov)
                    estimated_doas, _ = root_music.estimate_doa()
                    
                    rmse = calculate_doa_rmse(true_doas_clean, estimated_doas)
                    root_music_rmse_values.append(rmse)
                except Exception as e:
                    root_music_rmse_values.append(np.inf)
            
            # Evaluate Unitary ESPRIT if requested
            if evaluate_unitary_esprit and array_model is not None and received_cov is not None:
                try:
                    unitary_esprit = UnitaryESPRIT(array_model, num_sources=num_sources_sample)
                    unitary_esprit.set_received_covariance(received_cov)
                    estimated_doas, _ = unitary_esprit.estimate_doa()
                    
                    rmse = calculate_doa_rmse(true_doas_clean, estimated_doas)
                    unitary_esprit_rmse_values.append(rmse)
                except Exception as e:
                    unitary_esprit_rmse_values.append(np.inf)
        
        # Store results
        crlb_by_snr[snr_value] = {
            'crlb': crlb_values,
            'music_rmse': music_rmse_values if evaluate_music else [],
            'esprit_rmse': esprit_rmse_values if evaluate_esprit else [],
            'mvdr_rmse': mvdr_rmse_values if evaluate_mvdr else [],
            'beamformer_rmse': beamformer_rmse_values if evaluate_beamformer else [],
            'root_music_rmse': root_music_rmse_values if evaluate_root_music else [],
            'unitary_esprit_rmse': unitary_esprit_rmse_values if evaluate_unitary_esprit else []
        }
        
        # Print statistics for all three CRLB methods
        stochastic_vals = [c['stochastic'] for c in crlb_values if not np.isinf(c['stochastic'])]
        conditional_vals = [c['conditional'] for c in crlb_values if not np.isinf(c['conditional'])]
        closedform_vals = [c['closedform'] for c in crlb_values if not np.isinf(c['closedform'])]
        
        if stochastic_vals:
            mean_stoch_deg2 = np.mean(stochastic_vals) * (180/np.pi)**2
            rmse_stoch_deg = np.sqrt(np.mean(stochastic_vals)) * (180/np.pi)
            print(f"   CRLB Stochastic (SB):  Mean = {mean_stoch_deg2:.4f} deg², RMSE LB = {rmse_stoch_deg:.4f} deg")
        
        if conditional_vals:
            mean_cond_deg2 = np.mean(conditional_vals) * (180/np.pi)**2
            rmse_cond_deg = np.sqrt(np.mean(conditional_vals)) * (180/np.pi)
            print(f"   CRLB Conditional:      Mean = {mean_cond_deg2:.4f} deg², RMSE LB = {rmse_cond_deg:.4f} deg")
        
        if closedform_vals:
            mean_closed_deg2 = np.mean(closedform_vals) * (180/np.pi)**2
            rmse_closed_deg = np.sqrt(np.mean(closedform_vals)) * (180/np.pi)
            print(f"   CRLB Closed-form ULA:  Mean = {mean_closed_deg2:.4f} deg², RMSE LB = {rmse_closed_deg:.4f} deg")
        
        if not stochastic_vals and not conditional_vals and not closedform_vals:
            print(f"   No valid CRLB values computed")
        
        # Print algorithm statistics
        if evaluate_music and music_rmse_values:
            valid_music = [r for r in music_rmse_values if not np.isinf(r) and not np.isnan(r)]
            if valid_music:
                mean_music_rmse = np.mean(valid_music)
                print(f"   MUSIC:  Mean RMSE = {mean_music_rmse:.4f} deg")
            else:
                print(f"   MUSIC:  No valid estimates")
        
        if evaluate_esprit and esprit_rmse_values:
            valid_esprit = [r for r in esprit_rmse_values if not np.isinf(r) and not np.isnan(r)]
            if valid_esprit:
                mean_esprit_rmse = np.mean(valid_esprit)
                print(f"   ESPRIT: Mean RMSE = {mean_esprit_rmse:.4f} deg")
            else:
                print(f"   ESPRIT: No valid estimates")
        
        if evaluate_mvdr and mvdr_rmse_values:
            valid_mvdr = [r for r in mvdr_rmse_values if not np.isinf(r) and not np.isnan(r)]
            if valid_mvdr:
                mean_mvdr_rmse = np.mean(valid_mvdr)
                print(f"   MVDR:   Mean RMSE = {mean_mvdr_rmse:.4f} deg")
            else:
                print(f"   MVDR:   No valid estimates")
        
        if evaluate_beamformer and beamformer_rmse_values:
            valid_beamformer = [r for r in beamformer_rmse_values if not np.isinf(r) and not np.isnan(r)]
            if valid_beamformer:
                mean_beamformer_rmse = np.mean(valid_beamformer)
                print(f"   Beamformer: Mean RMSE = {mean_beamformer_rmse:.4f} deg")
            else:
                print(f"   Beamformer: No valid estimates")
        
        if evaluate_root_music and root_music_rmse_values:
            valid_root_music = [r for r in root_music_rmse_values if not np.isinf(r) and not np.isnan(r)]
            if valid_root_music:
                mean_root_music_rmse = np.mean(valid_root_music)
                print(f"   Root-MUSIC: Mean RMSE = {mean_root_music_rmse:.4f} deg")
            else:
                print(f"   Root-MUSIC: No valid estimates")
        
        if evaluate_unitary_esprit and unitary_esprit_rmse_values:
            valid_unitary_esprit = [r for r in unitary_esprit_rmse_values if not np.isinf(r) and not np.isnan(r)]
            if valid_unitary_esprit:
                mean_unitary_esprit_rmse = np.mean(valid_unitary_esprit)
                print(f"   Unitary-ESPRIT: Mean RMSE = {mean_unitary_esprit_rmse:.4f} deg")
            else:
                print(f"   Unitary-ESPRIT: No valid estimates")
    
    print("\n✅ CRLB computation complete")
    return crlb_by_snr


def plot_crlb_results(crlb_by_snr: Dict, output_dir: str):
    """
    Plot CRLB vs SNR with optional MUSIC comparison.
    
    Shows:
    - RMSE Lower Bound vs SNR (averaged over all angles)
    - MUSIC RMSE vs SNR (if available)
    """
    print("\n📈 Creating CRLB Plots...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    snr_vals = sorted(crlb_by_snr.keys())
    
    # Check if algorithm data is available
    first_snr_data = crlb_by_snr[snr_vals[0]]
    has_music = isinstance(first_snr_data, dict) and 'music_rmse' in first_snr_data and len(first_snr_data['music_rmse']) > 0
    has_esprit = isinstance(first_snr_data, dict) and 'esprit_rmse' in first_snr_data and len(first_snr_data['esprit_rmse']) > 0
    has_mvdr = isinstance(first_snr_data, dict) and 'mvdr_rmse' in first_snr_data and len(first_snr_data['mvdr_rmse']) > 0
    has_beamformer = isinstance(first_snr_data, dict) and 'beamformer_rmse' in first_snr_data and len(first_snr_data['beamformer_rmse']) > 0
    has_root_music = isinstance(first_snr_data, dict) and 'root_music_rmse' in first_snr_data and len(first_snr_data['root_music_rmse']) > 0
    has_unitary_esprit = isinstance(first_snr_data, dict) and 'unitary_esprit_rmse' in first_snr_data and len(first_snr_data['unitary_esprit_rmse']) > 0
    
    # Compute mean CRLB (variance) for each SNR - separate for all three methods
    mean_crlb_stochastic_rad2 = []
    mean_crlb_conditional_rad2 = []
    mean_crlb_closedform_rad2 = []
    mean_music_rmse = []
    mean_esprit_rmse = []
    mean_mvdr_rmse = []
    mean_beamformer_rmse = []
    mean_root_music_rmse = []
    mean_unitary_esprit_rmse = []
    
    for snr in snr_vals:
        # Extract CRLB values (now dict with 'stochastic', 'conditional', and 'closedform')
        if isinstance(crlb_by_snr[snr], dict):
            crlb_values = crlb_by_snr[snr]['crlb']
        else:
            crlb_values = crlb_by_snr[snr]
        
        # Extract stochastic CRLB values
        stoch_vals = [c['stochastic'] for c in crlb_values if not np.isinf(c['stochastic'])]
        if stoch_vals:
            mean_crlb_stochastic_rad2.append(np.mean(stoch_vals))
        else:
            mean_crlb_stochastic_rad2.append(np.nan)
        
        # Extract conditional CRLB values
        cond_vals = [c['conditional'] for c in crlb_values if not np.isinf(c['conditional'])]
        if cond_vals:
            mean_crlb_conditional_rad2.append(np.mean(cond_vals))
        else:
            mean_crlb_conditional_rad2.append(np.nan)
        
        # Extract closed-form CRLB values
        closed_vals = [c['closedform'] for c in crlb_values if not np.isinf(c['closedform'])]
        if closed_vals:
            mean_crlb_closedform_rad2.append(np.mean(closed_vals))
        else:
            mean_crlb_closedform_rad2.append(np.nan)
        
        # Extract algorithm RMSE values
        if has_music:
            music_values = crlb_by_snr[snr]['music_rmse']
            valid_music = [r for r in music_values if not np.isinf(r)]
            if valid_music:
                mean_music_rmse.append(np.mean(valid_music))
            else:
                mean_music_rmse.append(np.nan)
        
        if has_esprit:
            esprit_values = crlb_by_snr[snr]['esprit_rmse']
            valid_esprit = [r for r in esprit_values if not np.isinf(r)]
            if valid_esprit:
                mean_esprit_rmse.append(np.mean(valid_esprit))
            else:
                mean_esprit_rmse.append(np.nan)
        
        if has_mvdr:
            mvdr_values = crlb_by_snr[snr]['mvdr_rmse']
            valid_mvdr = [r for r in mvdr_values if not np.isinf(r)]
            if valid_mvdr:
                mean_mvdr_rmse.append(np.mean(valid_mvdr))
            else:
                mean_mvdr_rmse.append(np.nan)
        
        if has_beamformer:
            beamformer_values = crlb_by_snr[snr]['beamformer_rmse']
            valid_beamformer = [r for r in beamformer_values if not np.isinf(r)]
            if valid_beamformer:
                mean_beamformer_rmse.append(np.mean(valid_beamformer))
            else:
                mean_beamformer_rmse.append(np.nan)
        
        if has_root_music:
            root_music_values = crlb_by_snr[snr]['root_music_rmse']
            valid_root_music = [r for r in root_music_values if not np.isinf(r)]
            if valid_root_music:
                mean_root_music_rmse.append(np.mean(valid_root_music))
            else:
                mean_root_music_rmse.append(np.nan)
        
        if has_unitary_esprit:
            unitary_esprit_values = crlb_by_snr[snr]['unitary_esprit_rmse']
            valid_unitary_esprit = [r for r in unitary_esprit_values if not np.isinf(r)]
            if valid_unitary_esprit:
                mean_unitary_esprit_rmse.append(np.mean(valid_unitary_esprit))
            else:
                mean_unitary_esprit_rmse.append(np.nan)
    
    # Convert to RMSE in degrees for all three methods
    rmse_stochastic_deg = np.array([np.sqrt(c) * (180/np.pi) if not np.isnan(c) else np.nan 
                                     for c in mean_crlb_stochastic_rad2])
    rmse_conditional_deg = np.array([np.sqrt(c) * (180/np.pi) if not np.isnan(c) else np.nan 
                                      for c in mean_crlb_conditional_rad2])
    rmse_closedform_deg = np.array([np.sqrt(c) * (180/np.pi) if not np.isnan(c) else np.nan 
                                     for c in mean_crlb_closedform_rad2])
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # === Plot 1: RMSE Lower Bound vs SNR (log scale) - THREE METHODS ===
    ax1 = axes[0]
    ax1.semilogy(snr_vals, rmse_stochastic_deg, 'o-', linewidth=2, markersize=8, 
                 color='red', label='CRLB Stochastic (Slepian-Bangs)')
    ax1.semilogy(snr_vals, rmse_conditional_deg, '^-', linewidth=2, markersize=8, 
                 color='darkorange', label='CRLB Conditional')
    ax1.semilogy(snr_vals, rmse_closedform_deg, 'd-', linewidth=2, markersize=7,
                 color='green', label='CRLB Closed-form ULA')
    
    # Add algorithm performance if available
    if has_music and mean_music_rmse:
        ax1.semilogy(snr_vals, mean_music_rmse, 's-', linewidth=2, markersize=6,
                    color='blue', label='MUSIC')
    
    if has_esprit and mean_esprit_rmse:
        ax1.semilogy(snr_vals, mean_esprit_rmse, 'v-', linewidth=2, markersize=6,
                    color='purple', label='ESPRIT')
    
    if has_mvdr and mean_mvdr_rmse:
        ax1.semilogy(snr_vals, mean_mvdr_rmse, 'D-', linewidth=2, markersize=6,
                    color='cyan', label='MVDR (Capon)')
    
    if has_beamformer and mean_beamformer_rmse:
        ax1.semilogy(snr_vals, mean_beamformer_rmse, 'p-', linewidth=2, markersize=6,
                    color='magenta', label='Beamformer')
    
    if has_root_music and mean_root_music_rmse:
        ax1.semilogy(snr_vals, mean_root_music_rmse, 'h-', linewidth=2, markersize=6,
                    color='brown', label='Root-MUSIC')
    
    if has_unitary_esprit and mean_unitary_esprit_rmse:
        ax1.semilogy(snr_vals, mean_unitary_esprit_rmse, '<-', linewidth=2, markersize=6,
                    color='olive', label='Unitary-ESPRIT')
    
    ax1.set_xlabel('SNR (dB)', fontsize=14)
    ax1.set_ylabel('RMSE (degrees)', fontsize=14)
    algorithms_str = []
    if has_music:
        algorithms_str.append('MUSIC')
    if has_esprit:
        algorithms_str.append('ESPRIT')
    if has_mvdr:
        algorithms_str.append('MVDR')
    if has_beamformer:
        algorithms_str.append('Beamformer')
    if has_root_music:
        algorithms_str.append('Root-MUSIC')
    if has_unitary_esprit:
        algorithms_str.append('Unitary-ESPRIT')
    
    title = f'Cramér-Rao Lower Bound vs SNR (Three Methods)\n({num_elements} elements, {T_snapshots} snapshots'
    if algorithms_str:
        title += f', with {", ".join(algorithms_str)}'
    title += ')'
    ax1.set_title(title, fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, which='major')
    ax1.grid(True, alpha=0.2, which='minor', linestyle='--')
    ax1.minorticks_on()
    ax1.legend(fontsize=10, loc='best')
    
    # === Plot 2: CRLB (Variance) vs SNR (log-log scale) - THREE METHODS ===
    ax2 = axes[1]
    crlb_stochastic_deg2 = np.array(mean_crlb_stochastic_rad2) * (180/np.pi)**2
    crlb_conditional_deg2 = np.array(mean_crlb_conditional_rad2) * (180/np.pi)**2
    crlb_closedform_deg2 = np.array(mean_crlb_closedform_rad2) * (180/np.pi)**2
    
    ax2.loglog(snr_vals, crlb_stochastic_deg2, 's-', linewidth=2, markersize=8, 
               color='red', label='CRLB Stochastic (Variance)')
    ax2.loglog(snr_vals, crlb_conditional_deg2, '^-', linewidth=2, markersize=8,
               color='darkorange', label='CRLB Conditional (Variance)')
    ax2.loglog(snr_vals, crlb_closedform_deg2, 'd-', linewidth=2, markersize=7,
               color='green', label='CRLB Closed-form ULA (Variance)')
    
    ax2.set_xlabel('SNR (dB)', fontsize=14)
    ax2.set_ylabel('CRLB (degrees²)', fontsize=14)
    ax2.set_title(f'CRLB (Variance) vs SNR - Three Methods\n({num_elements} elements, {T_snapshots} snapshots)', 
                  fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, which='major')
    ax2.grid(True, alpha=0.2, which='minor', linestyle='--')
    ax2.minorticks_on()
    ax2.legend(fontsize=10, loc='best')
    
    plt.tight_layout()
    
    # Save plot
    plot_file = output_path / 'crlb_vs_snr.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"✅ Plot saved: {plot_file}")
    plt.close()


def save_crlb_results(crlb_by_snr: Dict, output_dir: str):
    """Save CRLB and algorithm results to JSON."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Check if algorithm data is available
    first_snr = list(crlb_by_snr.keys())[0]
    first_snr_data = crlb_by_snr[first_snr]
    has_music = isinstance(first_snr_data, dict) and 'music_rmse' in first_snr_data and len(first_snr_data['music_rmse']) > 0
    has_esprit = isinstance(first_snr_data, dict) and 'esprit_rmse' in first_snr_data and len(first_snr_data['esprit_rmse']) > 0
    has_mvdr = isinstance(first_snr_data, dict) and 'mvdr_rmse' in first_snr_data and len(first_snr_data['mvdr_rmse']) > 0
    has_beamformer = isinstance(first_snr_data, dict) and 'beamformer_rmse' in first_snr_data and len(first_snr_data['beamformer_rmse']) > 0
    has_root_music = isinstance(first_snr_data, dict) and 'root_music_rmse' in first_snr_data and len(first_snr_data['root_music_rmse']) > 0
    has_unitary_esprit = isinstance(first_snr_data, dict) and 'unitary_esprit_rmse' in first_snr_data and len(first_snr_data['unitary_esprit_rmse']) > 0
    
    # Convert to serializable format
    json_results = {}
    for snr, data in crlb_by_snr.items():
        # Extract CRLB values
        if isinstance(data, dict):
            crlb_vals = data['crlb']
            music_vals = data.get('music_rmse', [])
            esprit_vals = data.get('esprit_rmse', [])
            mvdr_vals = data.get('mvdr_rmse', [])
            beamformer_vals = data.get('beamformer_rmse', [])
            root_music_vals = data.get('root_music_rmse', [])
            unitary_esprit_vals = data.get('unitary_esprit_rmse', [])
        else:
            crlb_vals = data
            music_vals = []
            esprit_vals = []
            mvdr_vals = []
            beamformer_vals = []
            root_music_vals = []
            unitary_esprit_vals = []
        
        # Separate stochastic, conditional, and closed-form values
        stoch_vals = [float(c['stochastic']) for c in crlb_vals if not np.isinf(c['stochastic'])]
        cond_vals = [float(c['conditional']) for c in crlb_vals if not np.isinf(c['conditional'])]
        closed_vals = [float(c['closedform']) for c in crlb_vals if not np.isinf(c['closedform'])]
        
        result_entry = {
            'stochastic': {
                'crlb_rad2': stoch_vals,
                'mean_crlb_rad2': float(np.mean(stoch_vals)) if stoch_vals else None,
                'mean_crlb_deg2': float(np.mean(stoch_vals) * (180/np.pi)**2) if stoch_vals else None,
                'rmse_lb_deg': float(np.sqrt(np.mean(stoch_vals)) * (180/np.pi)) if stoch_vals else None,
                'num_samples': len(stoch_vals)
            },
            'conditional': {
                'crlb_rad2': cond_vals,
                'mean_crlb_rad2': float(np.mean(cond_vals)) if cond_vals else None,
                'mean_crlb_deg2': float(np.mean(cond_vals) * (180/np.pi)**2) if cond_vals else None,
                'rmse_lb_deg': float(np.sqrt(np.mean(cond_vals)) * (180/np.pi)) if cond_vals else None,
                'num_samples': len(cond_vals)
            },
            'closedform': {
                'crlb_rad2': closed_vals,
                'mean_crlb_rad2': float(np.mean(closed_vals)) if closed_vals else None,
                'mean_crlb_deg2': float(np.mean(closed_vals) * (180/np.pi)**2) if closed_vals else None,
                'rmse_lb_deg': float(np.sqrt(np.mean(closed_vals)) * (180/np.pi)) if closed_vals else None,
                'num_samples': len(closed_vals)
            }
        }
        
        # Add algorithm data if available
        if has_music and music_vals:
            valid_music = [float(r) for r in music_vals if not np.isinf(r)]
            result_entry['music'] = {
                'rmse_values': valid_music,
                'mean_rmse_deg': float(np.mean(valid_music)) if valid_music else None,
                'num_samples': len(valid_music)
            }
        
        if has_esprit and esprit_vals:
            valid_esprit = [float(r) for r in esprit_vals if not np.isinf(r)]
            result_entry['esprit'] = {
                'rmse_values': valid_esprit,
                'mean_rmse_deg': float(np.mean(valid_esprit)) if valid_esprit else None,
                'num_samples': len(valid_esprit)
            }
        
        if has_mvdr and mvdr_vals:
            valid_mvdr = [float(r) for r in mvdr_vals if not np.isinf(r)]
            result_entry['mvdr'] = {
                'rmse_values': valid_mvdr,
                'mean_rmse_deg': float(np.mean(valid_mvdr)) if valid_mvdr else None,
                'num_samples': len(valid_mvdr)
            }
        
        if has_beamformer and beamformer_vals:
            valid_beamformer = [float(r) for r in beamformer_vals if not np.isinf(r)]
            result_entry['beamformer'] = {
                'rmse_values': valid_beamformer,
                'mean_rmse_deg': float(np.mean(valid_beamformer)) if valid_beamformer else None,
                'num_samples': len(valid_beamformer)
            }
        
        if has_root_music and root_music_vals:
            valid_root_music = [float(r) for r in root_music_vals if not np.isinf(r)]
            result_entry['root_music'] = {
                'rmse_values': valid_root_music,
                'mean_rmse_deg': float(np.mean(valid_root_music)) if valid_root_music else None,
                'num_samples': len(valid_root_music)
            }
        
        if has_unitary_esprit and unitary_esprit_vals:
            valid_unitary_esprit = [float(r) for r in unitary_esprit_vals if not np.isinf(r)]
            result_entry['unitary_esprit'] = {
                'rmse_values': valid_unitary_esprit,
                'mean_rmse_deg': float(np.mean(valid_unitary_esprit)) if valid_unitary_esprit else None,
                'num_samples': len(valid_unitary_esprit)
            }
        
        json_results[str(snr)] = result_entry
    
    results_file = output_path / 'crlb_results.json'
    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"✅ Results saved: {results_file}")


def print_crlb_summary(crlb_by_snr: Dict):
    """Print CRLB summary table with optional algorithm comparisons."""
    # Check if algorithm data is available
    first_snr = list(crlb_by_snr.keys())[0]
    has_music = isinstance(crlb_by_snr[first_snr], dict) and 'music_rmse' in crlb_by_snr[first_snr]
    has_esprit = isinstance(crlb_by_snr[first_snr], dict) and 'esprit_rmse' in crlb_by_snr[first_snr]
    has_mvdr = isinstance(crlb_by_snr[first_snr], dict) and 'mvdr_rmse' in crlb_by_snr[first_snr]
    has_beamformer = isinstance(crlb_by_snr[first_snr], dict) and 'beamformer_rmse' in crlb_by_snr[first_snr]
    has_root_music = isinstance(crlb_by_snr[first_snr], dict) and 'root_music_rmse' in crlb_by_snr[first_snr]
    has_unitary_esprit = isinstance(crlb_by_snr[first_snr], dict) and 'unitary_esprit_rmse' in crlb_by_snr[first_snr]
    
    print("\n📊 Results Summary - Comparison of CRLB Methods and DOA Algorithms")
    print("=" * 200)
    
    # Build header dynamically based on available algorithms
    header1 = f"{'SNR':<6}{'Stochastic':<14}{'Stoch²':<12}{'Conditional':<14}{'Cond²':<12}{'Closed-form':<14}{'Closed²':<12}"
    header2 = f"{'(dB)':<6}{'RMSE':<14}{'(deg²)':<12}{'RMSE':<14}{'(deg²)':<12}{'RMSE':<14}{'(deg²)':<12}"
    
    if has_music:
        header1 += f"{'MUSIC':<12}"
        header2 += f"{'RMSE':<12}"
    if has_esprit:
        header1 += f"{'ESPRIT':<12}"
        header2 += f"{'RMSE':<12}"
    if has_mvdr:
        header1 += f"{'MVDR':<12}"
        header2 += f"{'RMSE':<12}"
    if has_beamformer:
        header1 += f"{'Beamformer':<12}"
        header2 += f"{'RMSE':<12}"
    if has_root_music:
        header1 += f"{'Root-MUSIC':<12}"
        header2 += f"{'RMSE':<12}"
    if has_unitary_esprit:
        header1 += f"{'Unitary-ESP':<12}"
        header2 += f"{'RMSE':<12}"
    
    header1 += f"{'N':<6}"
    header2 += f"{'':<6}"
    
    print(header1)
    print(header2)
    print("-" * 200)
    
    for snr in sorted(crlb_by_snr.keys()):
        # Extract CRLB values and algorithm results
        if isinstance(crlb_by_snr[snr], dict):
            crlb_vals = crlb_by_snr[snr]['crlb']
            music_vals = crlb_by_snr[snr].get('music_rmse', [])
            esprit_vals = crlb_by_snr[snr].get('esprit_rmse', [])
            mvdr_vals = crlb_by_snr[snr].get('mvdr_rmse', [])
            beamformer_vals = crlb_by_snr[snr].get('beamformer_rmse', [])
            root_music_vals = crlb_by_snr[snr].get('root_music_rmse', [])
            unitary_esprit_vals = crlb_by_snr[snr].get('unitary_esprit_rmse', [])
        else:
            crlb_vals = crlb_by_snr[snr]
            music_vals = []
            esprit_vals = []
            mvdr_vals = []
            beamformer_vals = []
            root_music_vals = []
            unitary_esprit_vals = []
        
        # Extract all three CRLB method values
        stoch_vals = [c['stochastic'] for c in crlb_vals if not np.isinf(c['stochastic'])]
        cond_vals = [c['conditional'] for c in crlb_vals if not np.isinf(c['conditional'])]
        closed_vals = [c['closedform'] for c in crlb_vals if not np.isinf(c['closedform'])]
        
        if stoch_vals and cond_vals and closed_vals:
            # Stochastic CRLB
            mean_stoch_rad2 = np.mean(stoch_vals)
            rmse_stoch_deg = np.sqrt(mean_stoch_rad2) * (180/np.pi)
            stoch_deg2 = mean_stoch_rad2 * (180/np.pi)**2
            
            # Conditional CRLB
            mean_cond_rad2 = np.mean(cond_vals)
            rmse_cond_deg = np.sqrt(mean_cond_rad2) * (180/np.pi)
            cond_deg2 = mean_cond_rad2 * (180/np.pi)**2
            
            # Closed-form CRLB
            mean_closed_rad2 = np.mean(closed_vals)
            rmse_closed_deg = np.sqrt(mean_closed_rad2) * (180/np.pi)
            closed_deg2 = mean_closed_rad2 * (180/np.pi)**2
            
            # Build row string
            row = f"{snr:<6.1f}{rmse_stoch_deg:<14.4f}{stoch_deg2:<12.4f}{rmse_cond_deg:<14.4f}{cond_deg2:<12.4f}{rmse_closed_deg:<14.4f}{closed_deg2:<12.4f}"
            
            # Add algorithm results if available
            if has_music:
                valid_music = [r for r in music_vals if not np.isinf(r)]
                music_rmse = np.mean(valid_music) if valid_music else np.nan
                row += f"{music_rmse:<12.4f}"
            
            if has_esprit:
                valid_esprit = [r for r in esprit_vals if not np.isinf(r)]
                esprit_rmse = np.mean(valid_esprit) if valid_esprit else np.nan
                row += f"{esprit_rmse:<12.4f}"
            
            if has_mvdr:
                valid_mvdr = [r for r in mvdr_vals if not np.isinf(r)]
                mvdr_rmse = np.mean(valid_mvdr) if valid_mvdr else np.nan
                row += f"{mvdr_rmse:<12.4f}"
            
            if has_beamformer:
                valid_beamformer = [r for r in beamformer_vals if not np.isinf(r)]
                beamformer_rmse = np.mean(valid_beamformer) if valid_beamformer else np.nan
                row += f"{beamformer_rmse:<12.4f}"
            
            if has_root_music:
                valid_root_music = [r for r in root_music_vals if not np.isinf(r)]
                root_music_rmse = np.mean(valid_root_music) if valid_root_music else np.nan
                row += f"{root_music_rmse:<12.4f}"
            
            if has_unitary_esprit:
                valid_unitary_esprit = [r for r in unitary_esprit_vals if not np.isinf(r)]
                unitary_esprit_rmse = np.mean(valid_unitary_esprit) if valid_unitary_esprit else np.nan
                row += f"{unitary_esprit_rmse:<12.4f}"
            
            row += f"{len(stoch_vals):<6}"
            print(row)
        else:
            # Build N/A row
            row = f"{snr:<6.1f}{'N/A':<14}{'N/A':<12}{'N/A':<14}{'N/A':<12}{'N/A':<14}{'N/A':<12}"
            if has_music:
                row += f"{'N/A':<12}"
            if has_esprit:
                row += f"{'N/A':<12}"
            if has_mvdr:
                row += f"{'N/A':<12}"
            if has_beamformer:
                row += f"{'N/A':<12}"
            if has_root_music:
                row += f"{'N/A':<12}"
            if has_unitary_esprit:
                row += f"{'N/A':<12}"
            row += f"{'0':<6}"
            print(row)
    
    print("-" * 200)


def main():
    print("🚀 CRLB Computation and Evaluation")
    print("=" * 60)
    
    # Display configuration
    print("\n📋 Configuration:")
    print(f"   Dataset Name: {dataset_name}")
    print(f"   Array: {num_elements} element {array_type}")
    print(f"   SNR Range: {snr_db}")
    print(f"   Snapshots: {T_snapshots}")
    print(f"   Samples per SNR: {examples_per_combination}")
    print(f"   Signal Power: p = 1 (fixed)")
    print(f"   Steering Vector: Unit-norm, cosine convention")
    
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
        dataset_path = generate_crlb_dataset()
    
    # Load dataset
    print(f"\n📂 Loading dataset...")
    dataset = DOADataset(dataset_path)
    print(f"✅ Loaded {len(dataset)} samples")
    
    # Create array model for DOA algorithm evaluation
    array_model = None
    if any([EVALUATE_MUSIC, EVALUATE_ESPRIT, EVALUATE_MVDR, EVALUATE_BEAMFORMER, 
            EVALUATE_ROOT_MUSIC, EVALUATE_UNITARY_ESPRIT]):
        print(f"\n🎵 Creating array model for DOA algorithm evaluation...")
        array_config = ArrayConfig(
            array_type=array_type,
            num_elements=num_elements,
            carrier_freq=carrier_freq,
            element_spacing=0.5  # Half wavelength
        )
        array_model = ArrayModel(array_config)
        print(f"✅ Array model created")
    
    # Compute CRLB (and optionally evaluate DOA algorithms)
    crlb_by_snr = compute_crlb_for_dataset(
        dataset, 
        num_elements, 
        T_snapshots,
        array_model=array_model,
        evaluate_music=EVALUATE_MUSIC,
        evaluate_esprit=EVALUATE_ESPRIT,
        evaluate_mvdr=EVALUATE_MVDR,
        evaluate_beamformer=EVALUATE_BEAMFORMER,
        evaluate_root_music=EVALUATE_ROOT_MUSIC,
        evaluate_unitary_esprit=EVALUATE_UNITARY_ESPRIT
    )
    
    # Display and save results
    print_crlb_summary(crlb_by_snr)
    save_crlb_results(crlb_by_snr, output_dir)
    plot_crlb_results(crlb_by_snr, output_dir)
    
    print(f"\n✅ CRLB evaluation completed! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
