#!/usr/bin/env python3
"""
Controlled Angle DOA Evaluation Script

This script provides comprehensive evaluation of DOA algorithms with precise control over:
- Angle configurations (fixed/random reference with controlled separation)
- Number of samples per SNR level
- Multiple sources with controlled interference
- UNet denoising + classic algorithms + CRLB comparison

Three operational modes:
1. Fixed reference angles with fixed delta separation
2. Random reference angles with fixed delta separation  
3. Random angles with minimum separation constraint

Primary use case: RMSE vs SNR comparison for:
- UNet + configurable follow-up algorithm (MUSIC, MVDR, Beamformer, Root-MUSIC)
- Classic algorithms (MUSIC, MVDR, Beamformer, ESPRIT, Root-MUSIC)
- CRLB theoretical bounds

Usage:
    1. Configure parameters in the section below
       - Set UNET_FOLLOWUP_ALGORITHM to choose which algorithm to use after UNet denoising
    2. Run: python controlled_angle_evaluation.py
"""

import os
import sys

# Fix OpenMP library conflict (needed for torch on some systems)
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
# CONFIGURATION SECTION - MODIFY THESE PARAMETERS
# ============================================================================

# === MODE SELECTION ===
# Choose operational mode:
# 1: Fixed reference angles with fixed delta
# 2: Random reference angles with fixed delta
# 3: Random angles with minimum separation
# 4: Random angles with minimum separation, SAME angles across all SNR levels

EVALUATION_MODE = 4  # Set to 1, 2, 3, or 4

# === ANGLE CONFIGURATION ===
# Mode 1: Fixed reference angles
REFERENCE_ANGLES = [90.5]  # List of reference angles for Mode 1

# Mode 2 & 3: Angle range for random generation
ANGLE_RANGE = (30.0, 150.0)  # (min, max) in degrees

# Delta separation (for Modes 1 & 2)
ANGLE_DELTA = 10.0  # Angular separation in degrees (None for Mode 3)

# === SNR CONFIGURATION ===
SNR_DB = list(range(-20, 21, 5))  # [-20, -15, -10, -5, 0, 5, 10, 15, 20]
SAMPLES_PER_SNR = 1000  # Number of samples for EACH SNR level

# === SOURCE CONFIGURATION ===
NUM_SOURCES = [1, 3]  # [main_sources, interference_sources]
# Examples:
#   [1, 0] = Single source (no interference)
#   [1, 2] = One main source + 2 interference sources
#   [2, 0] = Two main sources (no interference)

# === SIGNAL PARAMETERS ===
NUM_SNAPSHOTS = 512  # Number of time snapshots
NUM_MULTIPATH = 3   # Number of multipath components (0 = no multipath)
SIR_DB = 0.0         # Signal-to-Interference Ratio in dB (only used if interference > 0)

# === ARRAY CONFIGURATION ===
ARRAY_TYPE = "linear"  # Array geometry: 'linear', 'triangular', 'circular', 'cross'
NUM_ELEMENTS = 8       # Number of array elements
CARRIER_FREQ = 2.45e9  # Carrier frequency in Hz
SAMPLING_FREQ = 1e6    # Sampling frequency in Hz
ARRAY_IMPERFECTIONS = True  # Enable array imperfections (gain/phase errors, coupling)

# === OUTPUT CONTROL ===
SAVE_RECEIVED_SIGNAL = True
SAVE_COVARIANCE_MATRIX = True
SAVE_CLEAN_COVARIANCE_MATRIX = True  # Required for UNet training targets
SAVE_AUTOCORRELATION_MATRIX = True   # Required for UNet inputs
SAVE_MUSIC_ESTIMATES = False
SAVE_STEERING_VECTORS = True
SAVE_ARRAY_METADATA = False

# === AUTOCORRELATION PARAMETERS ===
AUTOCORR_TAU = 8  # Number of time lags (must match UNet training)

# === EVALUATION ALGORITHMS ===
EVALUATE_UNET = True          # Evaluate UNet + follow-up algorithm
UNET_FOLLOWUP_ALGORITHM = 'ESPRIT'  # Choose: 'MUSIC', 'MVDR', 'BEAMFORMER', 'ROOT_MUSIC', 'ESPRIT', 'UNITARY_ESPRIT'
EVALUATE_MUSIC = True         # Evaluate classic MUSIC
EVALUATE_MVDR = True          # Evaluate MVDR (Capon)
EVALUATE_BEAMFORMER = True    # Evaluate conventional Beamformer
EVALUATE_ESPRIT = True        # Evaluate ESPRIT
EVALUATE_UNITARY_ESPRIT = True # Evaluate Unitary ESPRIT (real-valued, computationally efficient)
EVALUATE_ROOT_MUSIC = True    # Evaluate Root-MUSIC (linear arrays only)
EVALUATE_CRLB = False          # Compute CRLB bounds

# === UNET MODEL CONFIGURATION ===
UNET_MODEL_PATH = 'notebooks/evd_unet_denoising_model_20250929_015132.pth'  # Relative to Tri4Net directory

# === ALGORITHM PARAMETERS ===
SCAN_ANGLES = np.arange(30, 151, 1)  # Angle grid for spectrum-based algorithms (0.1° for high precision)
MAX_SAMPLES_PER_SNR = 1000  # Maximum samples to evaluate per SNR (if dataset larger)

# === OUTPUT CONFIGURATION ===
OUTPUT_DIR = 'Tri4Net/src/evaluation/controlled_angle_evaluation'
DATASET_NAME = f"controlled_eval_mode{EVALUATION_MODE}"

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

# Import signal generation
from signalgen import ArrayConfig, ArrayModel

# Import classic DOA algorithms
from models.classic.music import MUSIC
from models.classic.mvdr import MVDR
from models.classic.beamformer import Beamformer
from models.classic.esprit import ESPRIT
from models.classic.rootmusic import RootMUSIC
from models.classic.unitary_esprit import UnitaryESPRIT

# Import UNet models
try:
    from models.deep_learning.EVDUNet import EVDCovarianceReconstructionUNet
    EVDUNET_AVAILABLE = True
except ImportError:
    EVDUNET_AVAILABLE = False
    print("⚠️ EVDUNet models not available")


def validate_configuration():
    """Validate configuration parameters."""
    if EVALUATION_MODE not in [1, 2, 3, 4]:
        raise ValueError("EVALUATION_MODE must be 1, 2, 3, or 4")
    
    if EVALUATION_MODE == 1 and (not REFERENCE_ANGLES or ANGLE_DELTA is None):
        raise ValueError("Mode 1 requires REFERENCE_ANGLES and ANGLE_DELTA")
    
    if EVALUATION_MODE == 2 and ANGLE_DELTA is None:
        raise ValueError("Mode 2 requires ANGLE_DELTA")
    
    if EVALUATION_MODE in [3, 4] and ANGLE_DELTA is not None:
        print(f"⚠️ Warning: Mode {EVALUATION_MODE} ignores ANGLE_DELTA (uses minimum separation constraint)")
    
    # Validate UNet follow-up algorithm
    valid_followup_algorithms = ['MUSIC', 'MVDR', 'BEAMFORMER', 'ROOT_MUSIC', 'ESPRIT', 'UNITARY_ESPRIT']
    if EVALUATE_UNET and UNET_FOLLOWUP_ALGORITHM not in valid_followup_algorithms:
        raise ValueError(
            f"UNET_FOLLOWUP_ALGORITHM must be one of {valid_followup_algorithms}, "
            f"got '{UNET_FOLLOWUP_ALGORITHM}'"
        )
    
    total_sources = sum(NUM_SOURCES)
    if ANGLE_DELTA is not None:
        angle_span = (total_sources - 1) * ANGLE_DELTA
        if EVALUATION_MODE == 1:
            for ref in REFERENCE_ANGLES:
                if ref + angle_span > ANGLE_RANGE[1] and ref - angle_span < ANGLE_RANGE[0]:
                    raise ValueError(
                        f"Reference angle {ref}° with {total_sources} sources and {ANGLE_DELTA}° delta "
                        f"cannot fit in range {ANGLE_RANGE}"
                    )


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


def calculate_doa_rmse(true_angles, estimated_angles, penalty_for_missed=240.0, track_stats=None):
    """
    Calculate RMSE between true and estimated DOAs using optimal matching.
    
    This function finds the best assignment between true and estimated angles
    to minimize the total squared error. This is important when algorithms
    estimate more sources than we're evaluating (e.g., main + interference).
    
    Parameters
    ----------
    track_stats : dict, optional
        If provided, will track statistics: 'penalties_applied', 'total_calls', 'missed_sources'
    """
    if len(estimated_angles) == 0:
        if track_stats is not None:
            track_stats['penalties_applied'] += len(true_angles)
            track_stats['total_calls'] += 1
            track_stats['missed_sources'] += len(true_angles)
        return penalty_for_missed
    
    true_angles = np.array(true_angles)
    estimated_angles = np.array(estimated_angles)
    
    n_true = len(true_angles)
    n_est = len(estimated_angles)
    
    if n_est < n_true:
        # Not enough estimates - penalize missed sources
        num_missed = n_true - n_est
        if track_stats is not None:
            track_stats['missed_sources'] += num_missed
            track_stats['total_calls'] += 1
        # Match available estimates to closest true angles
        min_dist = min(len(true_angles), len(estimated_angles))
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
                if track_stats is not None:
                    track_stats['penalties_applied'] += 1
        
        return np.sqrt(np.mean(np.array(matched_errors)**2))
    
    # For each true angle, find the closest estimated angle
    # This handles the case where we estimate more sources than we evaluate
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
            if track_stats is not None:
                track_stats['penalties_applied'] += 1
    
    if track_stats is not None:
        track_stats['total_calls'] += 1
    
    return np.sqrt(np.mean(np.array(matched_errors)**2))


def compute_crlb_single_source(theta_rad, M, T, SNR_dB, steering_vector=None):
    """Compute CRLB for single source (conditional/deterministic formulation)."""
    P = 1.0
    rho = 10**(SNR_dB / 10)
    sigma2 = P / rho
    
    if steering_vector is not None:
        a = steering_vector / np.linalg.norm(steering_vector)
    else:
        m = np.arange(M)
        a = (1.0 / np.sqrt(M)) * np.exp(1j * np.pi * m * np.cos(theta_rad))
    
    d = -1j * np.pi * np.arange(M) * np.sin(theta_rad) * a
    
    I = np.eye(M)
    aaH = np.outer(a, a.conj())
    P_perp = I - aaH
    
    quad = np.real(np.conj(d).T @ (P_perp @ d))
    J_conditional = 2.0 * T * rho * quad
    
    return np.inf if J_conditional <= 0 else (1.0 / J_conditional)


def compute_crlb_multiple_sources(theta_rad_array, M, T, SNR_dB, steering_vectors=None, P=None):
    """Compute CRLB for multiple sources (conditional/deterministic formulation).
    
    Based on Stoica & Nehorai (1989) "MUSIC, Maximum Likelihood, and Cramér-Rao Bound"
    
    Model: y(t) = A(θ) s(t) + e(t),  e ~ CN(0, σ² I)
    
    The dataset uses per-source SNR definition:
    - signal_power = np.mean(np.abs(source_signals[0, :])**2) → power P of one source
    - noise_power = signal_power / 10^(SNR_dB/10) → σ² = P / SNR_linear
    
    For normalized sources (P=1): σ² = 10^(-SNR_dB/10)
    
    CRITICAL FIX: For UNCORRELATED sources (P=I), the FIM is:
        J = (2T/σ²) * Re{(D^H Π_⊥ D) ⊙ P^T}
    When P=I, the Hadamard product ZEROS all off-diagonals, making J diagonal.
    This means sources don't couple in the CRLB for uncorrelated signals!
    
    Parameters
    ----------
    theta_rad_array : array-like
        Source angles in radians
    M : int
        Number of array elements
    T : int
        Number of snapshots
    SNR_dB : float
        Signal-to-noise ratio in dB (per-source)
    steering_vectors : np.ndarray, optional
        Pre-computed steering vectors (M x K)
    P : np.ndarray, optional
        Source covariance matrix (K x K). If None, assumes P=I (uncorrelated sources)
    
    Returns
    -------
    crlb_vars : np.ndarray
        CRLB variance for each source (rad²), shape (K,)
    """
    K = len(theta_rad_array)
    # Per-source SNR: SNR = P/σ², with P=1 → σ² = 1/SNR_linear = 10^(-SNR_dB/10)
    sigma2 = 10**(-SNR_dB / 10)
    
    # Build steering matrix A (unit-norm columns)
    A = np.zeros((M, K), dtype=complex)
    m = np.arange(M, dtype=float)
    
    if steering_vectors is not None:
        for k in range(K):
            a = steering_vectors[:, k].astype(complex).ravel()
            A[:, k] = a / (np.linalg.norm(a) + 1e-16)
    else:
        for k, th in enumerate(theta_rad_array):
            A[:, k] = np.exp(1j * np.pi * m * np.cos(th)) / np.sqrt(M)
    
    # Derivative matrix D: da/dθ = -jπm sin(θ) a
    D = np.zeros((M, K), dtype=complex)
    for k, th in enumerate(theta_rad_array):
        D[:, k] = (-1j * np.pi * m * np.sin(th)) * A[:, k]
    
    # Projector Π_⊥ = I - A(A^H A)^(-1) A^H
    I = np.eye(M, dtype=complex)
    AH_A = A.conj().T @ A
    
    # Use pinv for robustness when sources are close
    try:
        AH_A_inv = np.linalg.pinv(AH_A)
    except np.linalg.LinAlgError:
        return np.full(K, np.inf)
    
    Pi_perp = I - A @ AH_A_inv @ A.conj().T
    
    # Compute Gram matrix G = D^H Π_⊥ D
    G = D.conj().T @ Pi_perp @ D  # KxK complex Hermitian
    G = (G + G.conj().T) / 2  # Force Hermitian numerically
    G_real = np.real(G)
    
    # Fisher Information Matrix
    if P is None:
        # UNCORRELATED sources (P=I): Hadamard with I keeps only diagonal
        # J = (2T/σ²) * diag(diag(Re{G}))
        J_matrix = (2.0 * T / sigma2) * np.diag(np.diag(G_real))
    else:
        # CORRELATED sources: J = (2T/σ²) * Re{G ⊙ P^T}
        P = np.asarray(P)
        if P.shape != (K, K):
            raise ValueError(f"P must be {K}x{K}")
        J_matrix = (2.0 * T / sigma2) * np.real(G * P.T)
    
    # Invert FIM to get CRLB
    if np.allclose(J_matrix, np.diag(np.diag(J_matrix))):
        # Diagonal J - just invert diagonal elements
        crlb_vars = np.zeros(K)
        diag_J = np.diag(J_matrix)
        crlb_vars[diag_J > 0] = 1.0 / diag_J[diag_J > 0]
        crlb_vars[diag_J <= 0] = np.inf
    else:
        # Non-diagonal - need full matrix inverse
        try:
            CRLB_matrix = np.linalg.pinv(J_matrix)
            crlb_vars = np.diag(CRLB_matrix)
        except np.linalg.LinAlgError:
            return np.full(K, np.inf)
    
    return crlb_vars


def compute_crlb_coherent_multipath(theta0_rad, theta_multipath_rad, alpha_multipath, 
                                     M, T, SNR_dB, steering_vector_main=None, 
                                     steering_vectors_multipath=None):
    """
    Compute CRLB for coherent multipath (Case B).
    
    This is for the scenario where:
    - All paths share the same source waveform (coherent)
    - We want to estimate only theta_0 (direct path)
    - Other path angles and gains are nuisance parameters
    
    Parameters
    ----------
    theta0_rad : float
        Direct path angle in radians
    theta_multipath_rad : array-like
        Multipath angles in radians [theta_1, ..., theta_L]
    alpha_multipath : array-like
        Complex multipath gains [alpha_1, ..., alpha_L] relative to direct path
        (alpha_0 = 1.0 is implicit for the direct path)
    M : int
        Number of array elements
    T : int
        Number of snapshots
    SNR_dB : float
        Signal-to-noise ratio in dB
    steering_vector_main : array-like, optional
        Steering vector for direct path (if None, uses ULA model)
    steering_vectors_multipath : array-like, optional
        Steering vectors for multipath [M x L] (if None, uses ULA model)
    
    Returns
    -------
    crlb : float
        CRLB for estimating theta_0 in presence of coherent multipath
        
    Notes
    -----
    Uses the Schur complement formula to eliminate nuisance parameters:
    J_θ0 = (2T/σ²) * Re{g0^H * Π_perp * (I - G(G^H Π_perp G)^-1 G^H Π_perp) * Π_perp * g0}
    
    where:
    - v = alpha_0*a(theta_0) + sum(alpha_l*a(theta_l)) is the composite steering
    - Π_perp = I - vv^H/|v|² projects onto v's orthogonal complement
    - g0 = ∂v/∂theta_0 is the gradient w.r.t. the parameter of interest
    - G contains gradients w.r.t. nuisance parameters (Re/Im of alphas, other thetas)
    """
    # Signal power and noise variance
    P = 1.0
    rho = 10**(SNR_dB / 10)
    sigma2 = P / rho
    
    L = len(theta_multipath_rad)  # Number of multipath components
    alpha_multipath = np.array(alpha_multipath, dtype=complex)
    
    # Check for over-parameterization
    # Parameters: [theta_0, Re(α1), Im(α1), ..., Re(αL), Im(αL), θ1, ..., θL]
    num_params = 1 + 2*L + L  # = 1 + 3L
    
    # For identifiability, we need enough observations (array elements)
    # Rule of thumb: num_params should be much less than M
    # With coherent multipath, the effective rank is further reduced
    if num_params >= M:
        # Over-parameterized: Cannot reliably estimate all parameters
        # Return inf to indicate CRLB is not meaningful
        return np.inf
    
    # Build steering vectors
    # NOTE: Steering vectors should be normalized (|a| = 1) for CRLB calculation
    m = np.arange(M)
    
    # Direct path (alpha_0 = 1.0)
    if steering_vector_main is not None:
        a0 = steering_vector_main  # Should already be normalized by caller
    else:
        # ULA steering vector, normalized
        a0 = (1.0 / np.sqrt(M)) * np.exp(1j * np.pi * m * np.cos(theta0_rad))
    
    # Multipath steering vectors
    A_multipath = np.zeros((M, L), dtype=complex)
    for l in range(L):
        if steering_vectors_multipath is not None:
            A_multipath[:, l] = steering_vectors_multipath[:, l]  # Should already be normalized by caller
        else:
            # ULA steering vector, normalized
            A_multipath[:, l] = (1.0 / np.sqrt(M)) * np.exp(1j * np.pi * m * np.cos(theta_multipath_rad[l]))
    
    # Composite steering vector: v = alpha_0*a0 + sum(alpha_l*a_l)
    # Note: alpha_0 = 1.0 for the direct path (implicit)
    v = a0.copy()
    for l in range(L):
        v += alpha_multipath[l] * A_multipath[:, l]
    
    # Check for degenerate case
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-12:
        return np.inf
    
    # Normalize v for the projector
    # The composite steering vector v should be normalized for proper CRLB calculation
    v = v / v_norm
    
    # For conditional (deterministic signal) model with coherent paths,
    # build the full parameter gradient matrix including theta_0
    # Parameters: [theta_0, Re(α1), Im(α1), ..., Re(αL), Im(αL), θ1, ..., θL]
    num_params = 1 + 2*L + L
    D_full = np.zeros((M, num_params), dtype=complex)
    
    # Gradient w.r.t. theta_0 (direct path angle)
    d0 = -1j * np.pi * m * np.sin(theta0_rad) * a0
    D_full[:, 0] = d0
    
    # Derivatives w.r.t. Re and Im of alpha_l
    for l in range(L):
        # ∂v/∂Re(α_l) = a_l
        D_full[:, 1 + 2*l] = A_multipath[:, l]
        # ∂v/∂Im(α_l) = j * a_l  
        D_full[:, 1 + 2*l + 1] = 1j * A_multipath[:, l]
    
    # Derivatives w.r.t. theta_l
    for l in range(L):
        # ∂v/∂θ_l = α_l * ∂a_l/∂θ_l
        d_l = -1j * np.pi * m * np.sin(theta_multipath_rad[l]) * A_multipath[:, l]
        D_full[:, 1 + 2*L + l] = alpha_multipath[l] * d_l
    
    # Projector onto v's orthogonal complement
    # P_⊥ = I - vv^H (since v is normalized, |v|² = 1)
    vvH = np.outer(v, v.conj())
    Pi_perp = np.eye(M) - vvH
    
    # Compute Fisher Information Matrix for all parameters
    # J = (2T/σ²) * Re{D^H * Π_⊥ * D}
    DH_Pi_D = D_full.conj().T @ Pi_perp @ D_full
    J_full = (2.0 * T / sigma2) * np.real(DH_Pi_D)
    
    # Check condition number to detect ill-conditioning
    cond_num = np.linalg.cond(J_full)
    if cond_num > 1e15:
        # Matrix is too ill-conditioned - CRLB unreliable
        # Use 1e15 threshold to allow some conditioning issues while rejecting truly singular cases
        return np.inf
    
    # Extract the CRLB for theta_0 using matrix inversion
    # The (0,0) element of J_full^{-1} is the CRLB for theta_0
    try:
        J_full_inv = np.linalg.inv(J_full)
        crlb_var = J_full_inv[0, 0]
    except np.linalg.LinAlgError:
        # Matrix is singular - parameters not identifiable
        return np.inf
    
    if crlb_var <= 0:
        return np.inf
    
    return crlb_var


def evaluate_sample_with_algorithms(sample: Dict, array_model: ArrayModel,
                                    scan_angles: np.ndarray, unet_model=None,
                                    evaluate_crlb: bool = False,
                                    penalty_stats: Dict = None) -> Dict:
    """Evaluate a single sample with all configured algorithms.
    
    Parameters
    ----------
    penalty_stats : dict, optional
        Dictionary to track penalty statistics per algorithm
    """
    received_cov = sample['covariance_matrix']
    if torch.is_tensor(received_cov):
        received_cov = received_cov.cpu().numpy()
    
    labels = sample['labels']
    all_doas = labels['doas']
    
    # Determine total number of sources and number of main sources
    if 'num_sources' in labels:
        num_sources_label = labels['num_sources']
        if isinstance(num_sources_label, (list, np.ndarray)):
            # num_sources_label = [main_sources, interference_sources]
            num_main_sources = int(num_sources_label[0]) if len(num_sources_label) > 0 else len(all_doas[~np.isnan(all_doas)])
            num_total_sources = int(np.sum(num_sources_label))  # Total = main + interference
        else:
            num_main_sources = int(num_sources_label)
            num_total_sources = num_main_sources
    else:
        num_total_sources = len(all_doas[~np.isnan(all_doas)])
        num_main_sources = num_total_sources
    
    # For evaluation, we only care about main sources (not interference)
    # But algorithms need to know total number of sources to estimate
    true_doas_clean = all_doas[~np.isnan(all_doas)][:num_main_sources]
    
    # For CRLB: need ALL source DOAs (main + interference) present in the signal  
    all_source_doas = all_doas[~np.isnan(all_doas)][:num_total_sources]
    
    # Extract multipath angles (they come AFTER the source angles in the doas array)
    num_multipath = int(labels.get('num_multipath', 0))
    all_angles_including_mp = all_doas[~np.isnan(all_doas)]
    multipath_doas = all_angles_including_mp[num_total_sources:num_total_sources + num_multipath] if num_multipath > 0 else np.array([])
    has_multipath = num_multipath > 0 and len(multipath_doas) > 0
    
    results = {
        'snr': sample['labels']['snr'],
        'true_doas': true_doas_clean
    }
    
    # Classic algorithms
    algorithms = {}
    if EVALUATE_MUSIC:
        algorithms['MUSIC'] = MUSIC
    if EVALUATE_MVDR:
        algorithms['MVDR'] = MVDR
    if EVALUATE_BEAMFORMER:
        algorithms['Beamformer'] = Beamformer
    if EVALUATE_ESPRIT:
        algorithms['ESPRIT'] = ESPRIT
    if EVALUATE_UNITARY_ESPRIT:
        algorithms['UnitaryESPRIT'] = UnitaryESPRIT
    
    # Add Root-MUSIC only for linear arrays
    if EVALUATE_ROOT_MUSIC and array_model.config.array_type == 'linear':
        algorithms['RootMUSIC'] = RootMUSIC
    
    for alg_name, AlgClass in algorithms.items():
        try:
            # Algorithms need to estimate ALL sources (main + interference)
            alg = AlgClass(array_model, array_manifold=None, scan_angles_deg=scan_angles, 
                          num_sources=num_total_sources)
            alg.set_received_covariance(received_cov)
            estimated_doas, _ = alg.estimate_doa()
            # Evaluate RMSE on ALL sources (main + interference) that we're estimating
            # Track penalty statistics if provided
            alg_stats = None
            if penalty_stats is not None:
                if alg_name not in penalty_stats:
                    penalty_stats[alg_name] = {'penalties_applied': 0, 'total_calls': 0, 'missed_sources': 0}
                alg_stats = penalty_stats[alg_name]
            rmse = calculate_doa_rmse(all_source_doas, estimated_doas, track_stats=alg_stats)
            results[alg_name] = rmse
        except Exception as e:
            results[alg_name] = np.inf
    
    # UNet + Follow-up Algorithm
    if EVALUATE_UNET and unet_model is not None:
        try:
            autocorr = create_autocorrelation_input_for_unet(sample, tau=AUTOCORR_TAU)
            autocorr_batch = autocorr.unsqueeze(0)
            
            unet_model.eval()
            with torch.no_grad():
                # EVDCovarianceReconstructionUNet returns (embedding, eigenvalues, recon_cov)
                _, _, recon_cov = unet_model(autocorr_batch)
                recon_cov_np = recon_cov.cpu().numpy().squeeze()
            
            # recon_cov_np is already complex covariance matrix
            # Apply the chosen follow-up algorithm
            alg_name = f'ReconUNet_{UNET_FOLLOWUP_ALGORITHM}'
            
            if UNET_FOLLOWUP_ALGORITHM == 'MUSIC':
                algorithm = MUSIC(array_model, array_manifold=None, scan_angles_deg=scan_angles,
                                 num_sources=num_total_sources)
                algorithm.set_received_covariance(recon_cov_np)
                est_doas, _ = algorithm.estimate_doa()
            elif UNET_FOLLOWUP_ALGORITHM == 'MVDR':
                algorithm = MVDR(array_model, array_manifold=None, scan_angles_deg=scan_angles,
                                num_sources=num_total_sources)
                algorithm.set_received_covariance(recon_cov_np)
                est_doas, _ = algorithm.estimate_doa()
            elif UNET_FOLLOWUP_ALGORITHM == 'BEAMFORMER':
                algorithm = Beamformer(array_model, array_manifold=None, scan_angles_deg=scan_angles,
                                      num_sources=num_total_sources)
                algorithm.set_received_covariance(recon_cov_np)
                est_doas, _ = algorithm.estimate_doa()
            elif UNET_FOLLOWUP_ALGORITHM == 'ROOT_MUSIC':
                if array_model.config.array_type == 'linear':
                    algorithm = RootMUSIC(array_model, num_sources=num_total_sources)
                    algorithm.set_received_covariance(recon_cov_np)
                    est_doas, _ = algorithm.estimate_doa()
                else:
                    raise ValueError("Root-MUSIC only works with linear arrays")
            elif UNET_FOLLOWUP_ALGORITHM == 'ESPRIT':
                if array_model.config.array_type == 'linear':
                    algorithm = ESPRIT(array_model, num_sources=num_total_sources)
                    algorithm.set_received_covariance(recon_cov_np)
                    est_doas, _ = algorithm.estimate_doa()
                else:
                    raise ValueError("ESPRIT only works with linear arrays")
            elif UNET_FOLLOWUP_ALGORITHM == 'UNITARY_ESPRIT':
                if array_model.config.array_type == 'linear':
                    algorithm = UnitaryESPRIT(array_model, num_sources=num_total_sources)
                    algorithm.set_received_covariance(recon_cov_np)
                    est_doas, _ = algorithm.estimate_doa()
                else:
                    raise ValueError("Unitary ESPRIT only works with linear arrays")
            else:
                raise ValueError(f"Unknown follow-up algorithm: {UNET_FOLLOWUP_ALGORITHM}")
            
            # Evaluate RMSE on ALL sources (main + interference) that we're estimating
            rmse = calculate_doa_rmse(all_source_doas, est_doas)
            results[alg_name] = rmse
        except Exception as e:
            print(f"⚠️  {alg_name} error: {e}")
            results[alg_name] = np.inf
    
    # CRLB computation
    if evaluate_crlb:
        try:
            M = array_model.config.num_elements
            T = NUM_SNAPSHOTS
            snr_val = float(sample['labels']['snr'])
            d = array_model.config.element_spacing
            wavelength = array_model.wavelength  # ArrayModel has wavelength, not ArrayConfig
            
            # Extract Signal-to-Multipath Ratio (SMR) if available
            smr_values = sample['labels'].get('smr', None)
            if smr_values is not None and isinstance(smr_values, np.ndarray):
                smr_values = smr_values[~np.isnan(smr_values)]  # Filter out NaN values
            
            if len(all_source_doas) == 1 and not has_multipath:
                # Single source without multipath - use simple CRLB
                steering_vec = array_model.steering_vector(all_source_doas[0], nominal=True)
                # Normalize: steering_vector() gives |a| = sqrt(M), CRLB expects |a| = 1
                steering_vec = steering_vec / np.linalg.norm(steering_vec)
                theta_rad = np.deg2rad(all_source_doas[0])
                crlb_var = compute_crlb_single_source(theta_rad, M, T, snr_val, steering_vec)
                results['CRLB'] = np.sqrt(crlb_var) * (180/np.pi)
                
            elif len(all_source_doas) == 1 and has_multipath:
                # Single source WITH coherent multipath - use coherent multipath CRLB
                # Main source angle
                theta0_rad = np.deg2rad(all_source_doas[0])
                steering_vec_main = array_model.steering_vector(all_source_doas[0], nominal=True)
                # Normalize: steering_vector() gives |a| = sqrt(M), CRLB expects |a| = 1
                steering_vec_main = steering_vec_main / np.linalg.norm(steering_vec_main)
                
                # Multipath angles - steering_matrix() already returns normalized columns!
                theta_mp_rad = np.deg2rad(multipath_doas)
                steering_vecs_mp = array_model.steering_matrix(multipath_doas, nominal=True)
                
                # Estimate multipath gains from SMR
                # SMR in dB: SMR = 20*log10(|main|/|multipath|) → |multipath|/|main| = 10^(-SMR/20)
                if smr_values is not None and len(smr_values) == num_multipath:
                    # Complex gains with zero phase (worst case for DOA estimation)
                    alpha_mp = np.array([10**(-smr/20) * np.exp(1j * 0) for smr in smr_values])
                else:
                    # Fallback: assume -6dB multipath (common for reflections)
                    alpha_mp = np.array([10**(-6/20) * np.exp(1j * 0) for _ in range(num_multipath)])
                
                crlb_var = compute_crlb_coherent_multipath(
                    theta0_rad, theta_mp_rad, alpha_mp, M, T, snr_val,
                    steering_vec_main, steering_vecs_mp
                )
                results['CRLB'] = np.sqrt(crlb_var) * (180/np.pi)
                
            elif has_multipath and len(all_source_doas) >= 1:
                # Multiple sources WITH multipath - use exact multipath CRLB for EACH source
                # Each source has its own multipath components
                
                # Print info message once
                if not hasattr(evaluate_sample_with_algorithms, '_exact_multipath_crlb_shown'):
                    print("\n✓ Using EXACT multipath CRLB formula (not approximation)")
                    print(f"  → Multipath components: {num_multipath}")
                    print(f"  → Computing rigorous CRLB with coherent multipath structure\n")
                    evaluate_sample_with_algorithms._exact_multipath_crlb_shown = True
                
                # For now, assume all multipath is associated with the first (main) source
                # This is a common scenario: one dominant source with reflections
                main_source_theta_rad = np.deg2rad(all_source_doas[0])
                steering_vec_main = array_model.steering_vector(all_source_doas[0], nominal=True)
                steering_vec_main = steering_vec_main / np.linalg.norm(steering_vec_main)
                
                # Multipath angles
                theta_mp_rad = np.deg2rad(multipath_doas)
                steering_vecs_mp = array_model.steering_matrix(multipath_doas, nominal=True)
                
                # Estimate multipath gains from SMR
                if smr_values is not None and len(smr_values) == num_multipath:
                    alpha_mp = np.array([10**(-smr/20) * np.exp(1j * 0) for smr in smr_values])
                else:
                    # Fallback: assume -6dB multipath
                    alpha_mp = np.array([10**(-6/20) * np.exp(1j * 0) for _ in range(num_multipath)])
                
                # Compute multipath CRLB for the main source
                crlb_var_main = compute_crlb_coherent_multipath(
                    main_source_theta_rad, theta_mp_rad, alpha_mp, M, T, snr_val,
                    steering_vec_main, steering_vecs_mp
                )
                
                # For other sources (if any), use standard CRLB (no multipath)
                if len(all_source_doas) > 1:
                    other_doas = all_source_doas[1:]
                    theta_rad_array = np.deg2rad(other_doas)
                    steering_vecs = array_model.steering_matrix(other_doas, nominal=True)
                    crlb_vars_others = compute_crlb_multiple_sources(theta_rad_array, M, T, snr_val, steering_vecs)
                    
                    # Combine: average over ALL sources (same as algorithms evaluate)
                    all_crlb_vars = [crlb_var_main] + list(crlb_vars_others)
                    avg_crlb = np.mean([v for v in all_crlb_vars if not np.isinf(v)])
                else:
                    avg_crlb = crlb_var_main
                
                results['CRLB'] = np.sqrt(avg_crlb) * (180/np.pi)
                
            else:
                # Multiple sources WITHOUT multipath
                theta_rad_array = np.deg2rad(all_source_doas)
                # steering_matrix() already returns normalized columns!
                steering_vecs = array_model.steering_matrix(all_source_doas, nominal=True)
                crlb_vars = compute_crlb_multiple_sources(theta_rad_array, M, T, snr_val, steering_vecs)
                # Average over ALL sources (same as algorithms evaluate)
                avg_crlb = np.mean(crlb_vars)
                results['CRLB'] = np.sqrt(avg_crlb) * (180/np.pi)
        except Exception as e:
            print(f"⚠️  CRLB computation error: {e}")
            results['CRLB'] = np.nan
    
    return results


def evaluate_dataset(dataset: DOADataset, array_model: ArrayModel, unet_model=None,
                     snr_levels: List = None) -> Dict:
    """Evaluate dataset and return results organized by SNR."""
    print("\n🎯 Evaluating Dataset...")
    print("=" * 60)
    
    if snr_levels is None:
        snr_levels = sorted(set([float(dataset.labels['snr'][i]) for i in range(len(dataset))]))
    
    print(f"📊 Evaluating SNR levels: {snr_levels}")
    
    results_by_snr = {}
    
    for snr_db in snr_levels:
        print(f"\n🎯 Evaluating SNR = {snr_db} dB")
        
        filtered_dataset = dataset.filter(snr=snr_db)
        num_samples = min(len(filtered_dataset), MAX_SAMPLES_PER_SNR)
        print(f"   Processing {num_samples} samples...")
        
        if num_samples == 0:
            continue
        
        # Initialize results storage
        results_storage = {}
        if EVALUATE_MUSIC:
            results_storage['MUSIC'] = []
        if EVALUATE_MVDR:
            results_storage['MVDR'] = []
        if EVALUATE_BEAMFORMER:
            results_storage['Beamformer'] = []
        if EVALUATE_ESPRIT:
            results_storage['ESPRIT'] = []
        if EVALUATE_UNITARY_ESPRIT:
            results_storage['UnitaryESPRIT'] = []
        if EVALUATE_ROOT_MUSIC and array_model.config.array_type == 'linear':
            results_storage['RootMUSIC'] = []
        if EVALUATE_UNET:
            unet_alg_name = f'ReconUNet_{UNET_FOLLOWUP_ALGORITHM}'
            results_storage[unet_alg_name] = []
        if EVALUATE_CRLB:
            results_storage['CRLB'] = []
        
        results_by_snr[snr_db] = results_storage
        
        # Initialize penalty tracking for all algorithms
        penalty_stats = {}
        for alg_name in results_storage.keys():
            if alg_name != 'CRLB':  # CRLB doesn't use penalty tracking
                penalty_stats[alg_name] = {'penalties_applied': 0, 'total_calls': 0, 'missed_sources': 0}
        
        # Evaluate samples
        for idx in tqdm(range(num_samples), desc=f"SNR {snr_db}dB"):
            sample = filtered_dataset[idx]
            result = evaluate_sample_with_algorithms(sample, array_model, SCAN_ANGLES, 
                                                    unet_model, EVALUATE_CRLB, penalty_stats)
            
            for alg in results_by_snr[snr_db].keys():
                if alg in result:
                    results_by_snr[snr_db][alg].append(result[alg])
        
        # Debug: Display penalty statistics for this SNR
        if penalty_stats:
            print(f"\n{'='*70}")
            print(f"DEBUG: Penalty Statistics for SNR = {snr_db} dB")
            print(f"{'='*70}")
            print(f"{'Algorithm':<20} {'Total Calls':>12} {'Penalties':>12} {'Missed Sources':>15} {'Penalty Rate':>12}")
            print(f"{'-'*70}")
            for alg_name in sorted(penalty_stats.keys()):
                stats = penalty_stats[alg_name]
                total = stats['total_calls']
                penalties = stats['penalties_applied']
                missed = stats['missed_sources']
                rate = (penalties / total * 100) if total > 0 else 0.0
                print(f"{alg_name:<20} {total:>12} {penalties:>12} {missed:>15} {rate:>11.1f}%")
            print(f"{'='*70}\n")
    
    print("\n✅ Evaluation complete")
    return results_by_snr


def plot_results(results_by_snr: Dict, output_dir: str, mode: int, 
                num_sources: List[int], num_multipath: int, array_imperfections: bool):
    """Plot RMSE vs SNR comparison."""
    print("\n📈 Creating Plots...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    snr_vals = sorted(results_by_snr.keys())
    
    plt.figure(figsize=(14, 8))
    
    # Plot each algorithm
    algorithm_styles = {
        'MUSIC': {'marker': 'o', 'color': 'blue', 'linewidth': 2},
        'MVDR': {'marker': 's', 'color': 'green', 'linewidth': 2},
        'Beamformer': {'marker': '^', 'color': 'purple', 'linewidth': 2},
        'ESPRIT': {'marker': 'd', 'color': 'orange', 'linewidth': 2},
        'UnitaryESPRIT': {'marker': 'p', 'color': 'brown', 'linewidth': 2},
        'RootMUSIC': {'marker': 'v', 'color': 'cyan', 'linewidth': 2},
        # ReconUNet styles based on follow-up algorithm
        'ReconUNet_MUSIC': {'marker': '*', 'color': 'red', 'linewidth': 3, 'linestyle': '--', 'markersize': 12},
        'ReconUNet_MVDR': {'marker': '*', 'color': 'darkgreen', 'linewidth': 3, 'linestyle': '--', 'markersize': 12},
        'ReconUNet_BEAMFORMER': {'marker': '*', 'color': 'darkviolet', 'linewidth': 3, 'linestyle': '--', 'markersize': 12},
        'ReconUNet_ROOT_MUSIC': {'marker': '*', 'color': 'darkcyan', 'linewidth': 3, 'linestyle': '--', 'markersize': 12},
        'ReconUNet_ESPRIT': {'marker': '*', 'color': 'darkorange', 'linewidth': 3, 'linestyle': '--', 'markersize': 12},
        'ReconUNet_UNITARY_ESPRIT': {'marker': '*', 'color': 'darkred', 'linewidth': 3, 'linestyle': '--', 'markersize': 12},
        'CRLB': {'marker': 'x', 'color': 'black', 'linewidth': 2, 'linestyle': ':'}
    }
    
    # Debug: Check for failures
    print("\n" + "="*70)
    print("DEBUG: Algorithm Failure Analysis")
    print("="*70)
    
    # Overall failures
    any_failures = False
    for alg in results_by_snr[snr_vals[0]].keys():
        total_failures = 0
        total_samples = 0
        for snr in snr_vals:
            all_vals = results_by_snr[snr][alg]
            total_samples += len(all_vals)
            failures = sum(1 for r in all_vals if np.isinf(r) or np.isnan(r))
            total_failures += failures
        
        if total_failures > 0:
            any_failures = True
        success_rate = 100 * (total_samples - total_failures) / total_samples if total_samples > 0 else 0
        print(f"{alg:20s}: {total_failures:4d}/{total_samples:4d} failures ({100-success_rate:.1f}% failure rate)")
    
    if any_failures:
        print("\n⚠️  Detailed per-SNR failure breakdown:")
        for snr in snr_vals:
            snr_has_failures = False
            for alg in results_by_snr[snr_vals[0]].keys():
                all_vals = results_by_snr[snr][alg]
                failures = sum(1 for r in all_vals if np.isinf(r) or np.isnan(r))
                if failures > 0:
                    if not snr_has_failures:
                        print(f"\n  SNR = {snr} dB:")
                        snr_has_failures = True
                    print(f"    {alg:20s}: {failures}/{len(all_vals)} failures")
    else:
        print("\n✓ All algorithms succeeded on all samples!")
    
    print("="*70 + "\n")
    
    for alg in results_by_snr[snr_vals[0]].keys():
        mean_rmse = []
        for snr in snr_vals:
            error_vals = [r for r in results_by_snr[snr][alg] if not np.isinf(r) and not np.isnan(r)]
            
            # Different aggregation for CRLB vs algorithms:
            # - Algorithms: RMS aggregation because we're combining errors from different samples
            #   RMSE = sqrt(mean(errors²)) - this is the proper way to aggregate errors
            # - CRLB: Simple mean because CRLB is a theoretical bound (variance/std dev)
            #   For theoretical bounds, we average the bounds across different geometries
            if alg == 'CRLB':
                rmse = np.mean(error_vals) if error_vals else np.nan
            else:
                rmse = np.sqrt(np.mean(np.array(error_vals)**2)) if error_vals else np.nan
            
            mean_rmse.append(rmse)
        
        style = algorithm_styles.get(alg, {'marker': 'o', 'linewidth': 2})
        # Use special label for CRLB 
        label = 'CRLB (Theoretical Bound)' if alg == 'CRLB' else alg
        plt.plot(snr_vals, mean_rmse, label=label, **style)
    
    # Generate title
    mode_names = {
        1: 'Fixed Reference Angles',
        2: 'Random Reference Angles',
        3: 'Random Angles with Min Separation',
        4: 'Random Angles Shared Across SNR'
    }
    
    title = f'DOA Estimation Performance vs SNR\n'
    title += f'Mode {mode}: {mode_names[mode]}'
    if mode in [1, 2] and ANGLE_DELTA is not None:
        title += f' (Δ={ANGLE_DELTA}°)'
    title += f'\n{num_sources[0]} Main + {num_sources[1]} Interference Sources'
    title += f'\n(RMSE across all samples)'
    
    plt.xlabel('SNR (dB)', fontsize=14)
    plt.ylabel('Error (degrees)', fontsize=14)
    plt.yscale('log')
    plt.title(title, fontsize=16, fontweight='bold')
    plt.legend(fontsize=16, loc='best')
    plt.grid(True, alpha=0.3, which='major')
    plt.grid(True, alpha=0.2, which='minor', linestyle='--')
    plt.minorticks_on()
    plt.tight_layout()
    
    # Build descriptive filename
    total_sources = num_sources[0] + num_sources[1]
    array_type = "imperfect" if array_imperfections else "perfect"
    
    # Format: rmse_vs_snr_mode{mode}_{total_sources}src_mp{multipath}_{array_type}.png
    plot_file = output_path / f'rmse_vs_snr_mode{mode}_{total_sources}src_mp{num_multipath}_{array_type}.png'
    plt.savefig(plot_file, dpi=150)
    print(f"✅ Plot saved: {plot_file}")
    plt.close()


def save_results(results_by_snr: Dict, output_dir: str, mode: int,
                num_sources: List[int], num_multipath: int, array_imperfections: bool):
    """Save results to JSON."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    json_results = {}
    for snr, alg_results in results_by_snr.items():
        json_results[str(snr)] = {
            alg: [float(r) if not np.isinf(r) and not np.isnan(r) else None for r in vals]
            for alg, vals in alg_results.items()
        }
    
    # Build descriptive filename
    total_sources = num_sources[0] + num_sources[1]
    array_type = "imperfect" if array_imperfections else "perfect"
    
    # Format: results_mode{mode}_{total_sources}src_mp{multipath}_{array_type}.json
    results_file = output_path / f'results_mode{mode}_{total_sources}src_mp{num_multipath}_{array_type}.json'
    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"✅ Results saved: {results_file}")


def print_summary(results_by_snr: Dict):
    """Print summary table."""
    print("\n📊 Results Summary (RMSE across all samples)")
    print("=" * 100)
    
    algorithms = list(results_by_snr[list(results_by_snr.keys())[0]].keys())
    
    header = f"{'SNR (dB)':<10}"
    for alg in algorithms:
        header += f"{alg:<15}"
    print(header)
    print("-" * 100)
    
    crlb_unavailable = False
    for snr in sorted(results_by_snr.keys()):
        row = f"{snr:<10.1f}"
        for alg in algorithms:
            vals = [r for r in results_by_snr[snr][alg] if not np.isinf(r) and not np.isnan(r)]
            
            # Different aggregation for CRLB vs algorithms:
            # - Algorithms: RMS aggregation for errors
            # - CRLB: Simple mean for theoretical bounds
            if alg == 'CRLB':
                rmse_val = np.mean(vals) if vals else np.nan
            else:
                rmse_val = np.sqrt(np.mean(np.array(vals)**2)) if vals else np.nan
            
            if alg == 'CRLB' and np.isnan(rmse_val):
                crlb_unavailable = True
                row += f"{'N/A':<15}"
            else:
                row += f"{rmse_val:<15.3f}"
        print(row)
    
    print("-" * 100)
    
    if crlb_unavailable:
        print("\n⚠️  Note: CRLB marked as 'N/A' due to over-parameterization")
        print("   → Too many multipath components relative to array size")
        print("   → Fisher Information Matrix is ill-conditioned")
        print("   → CRLB is not meaningful in this scenario")


def main():
    print("🚀 Controlled Angle DOA Evaluation")
    print("=" * 60)
    
    # Validate configuration
    validate_configuration()
    
    # Display configuration
    mode_names = {
        1: 'Fixed Reference Angles',
        2: 'Random Reference Angles',
        3: 'Random Angles with Min Separation',
        4: 'Random Angles Shared Across SNR'
    }
    
    print(f"\n📋 Configuration:")
    print(f"   Mode: {EVALUATION_MODE} - {mode_names[EVALUATION_MODE]}")
    if EVALUATION_MODE == 1:
        print(f"   Reference Angles: {REFERENCE_ANGLES}")
    if EVALUATION_MODE in [1, 2]:
        print(f"   Angle Delta: {ANGLE_DELTA}°")
    print(f"   Angle Range: {ANGLE_RANGE}")
    print(f"   SNR Range: {SNR_DB}")
    print(f"   Samples per SNR: {SAMPLES_PER_SNR}")
    total_sources = sum(NUM_SOURCES)
    print(f"   Sources: {NUM_SOURCES[0]} main + {NUM_SOURCES[1]} interference = {total_sources} total")
    print(f"   Multipath: {NUM_MULTIPATH} component(s)")
    print(f"   → Algorithms will estimate {total_sources} source(s)")
    print(f"   → RMSE computed on ALL {total_sources} sources")
    print(f"   Array: {NUM_ELEMENTS} element {ARRAY_TYPE}")
    
    # Create dataset configuration
    dataset_config = SimpleDatasetConfig(
        angles_deg=[90.0],  # Placeholder, will be overridden by generate_test_dataset
        snr_db=SNR_DB,
        num_snapshots=[NUM_SNAPSHOTS],
        num_sources=[NUM_SOURCES],
        sir_db=[SIR_DB],
        num_multipath=[NUM_MULTIPATH],
        array_imperfections=[ARRAY_IMPERFECTIONS],
        random_sampling_mode=False,
        examples_per_combination=1,
        dataset_name=DATASET_NAME,
        array_type=ARRAY_TYPE,
        num_elements=NUM_ELEMENTS,
        carrier_freq=CARRIER_FREQ,
        sampling_freq=SAMPLING_FREQ,
        save_received_signal=SAVE_RECEIVED_SIGNAL,
        save_covariance_matrix=SAVE_COVARIANCE_MATRIX,
        save_clean_covariance_matrix=SAVE_CLEAN_COVARIANCE_MATRIX,
        save_autocorrelation_matrix=SAVE_AUTOCORRELATION_MATRIX,
        save_music_estimates=SAVE_MUSIC_ESTIMATES,
        save_steering_vectors=SAVE_STEERING_VECTORS,
        save_array_metadata=SAVE_ARRAY_METADATA,
        autocorr_tau=AUTOCORR_TAU
    )
    
    # Create generator
    generator = ControlledDatasetGenerator(dataset_config, seed=DATASET_GENERATION_SEED)
    
    # Generate test dataset based on mode
    print(f"\n🔧 Generating test dataset...")
    
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
            sir_db=SIR_DB,
            output_dir="Data/datasets"
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
            sir_db=SIR_DB,
            output_dir="Data/datasets"
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
            sir_db=SIR_DB,
            output_dir="Data/datasets"
        )
    else:  # Mode 4
        dataset_path = generator.generate_test_dataset(
            reference_angles='shared_random',  # NEW: Share angles across SNR levels
            angle_delta=None,
            angle_range=ANGLE_RANGE,
            snr_db=SNR_DB,
            samples_per_snr=SAMPLES_PER_SNR,
            num_sources=NUM_SOURCES,
            num_snapshots=NUM_SNAPSHOTS,
            num_multipath=NUM_MULTIPATH,
            array_imperfections=ARRAY_IMPERFECTIONS,
            sir_db=SIR_DB,
            output_dir="Data/datasets"
        )
    
    # Load dataset
    print(f"\n📂 Loading dataset...")
    dataset = DOADataset(dataset_path)
    print(f"✅ Loaded {len(dataset)} samples")
    
    # Create array model
    array_config = ArrayConfig(
        array_type=ARRAY_TYPE,
        num_elements=NUM_ELEMENTS,
        carrier_freq=CARRIER_FREQ,
        element_spacing=0.5,
        
    )
    array_model = ArrayModel(array_config)
    
    # Initialize UNet
    unet_model = None
    if EVALUATE_UNET and EVDUNET_AVAILABLE:
        print(f"\n🤖 Initializing UNet (follow-up: {UNET_FOLLOWUP_ALGORITHM})...")
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
                # Load checkpoint
                checkpoint = torch.load(str(model_path), map_location='cpu')
                
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
            except Exception as e:
                print(f"❌ Error loading UNet model: {e}")
                print(f"   Disabling UNet evaluation")
                unet_model = None
        else:
            print(f"⚠️ UNet model not found at {model_path}")
            print(f"   Disabling UNet evaluation")
            unet_model = None
    
    # Evaluate
    results_by_snr = evaluate_dataset(dataset, array_model, unet_model, SNR_DB)
    
    # Display and save results
    print_summary(results_by_snr)
    save_results(results_by_snr, OUTPUT_DIR, EVALUATION_MODE, NUM_SOURCES, NUM_MULTIPATH, ARRAY_IMPERFECTIONS)
    plot_results(results_by_snr, OUTPUT_DIR, EVALUATION_MODE, NUM_SOURCES, NUM_MULTIPATH, ARRAY_IMPERFECTIONS)
    
    print(f"\n✅ Evaluation completed! Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
