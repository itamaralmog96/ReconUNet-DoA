"""
Modified source estimation methods adapted for UNet eigenvalues.

The standard AIC/MDL methods assume a flat noise floor, but UNet-reconstructed
eigenvalues have gradually decaying noise components. This module provides
alternative formulations that work better with UNet outputs.
"""

import numpy as np
from typing import Tuple


# ============================================================================
# APPROACH 1: Eigenvalue Ratio Thresholding
# ============================================================================

def estimate_num_sources_ratio_threshold(eigenvalues: np.ndarray, 
                                        threshold: float = 2.0) -> int:
    """
    Estimate number of sources using eigenvalue ratio thresholding.
    
    Find the index where λ_i / λ_{i+1} > threshold, indicating the transition
    from signal to noise eigenvalues.
    
    This works well with UNet because it looks for the DROP in eigenvalues,
    not the flatness of the noise floor.
    
    Parameters
    ----------
    eigenvalues : np.ndarray
        Eigenvalues sorted in descending order
    threshold : float, optional
        Ratio threshold (default: 2.0)
        - For low SNR: use smaller threshold (1.5-2.0)
        - For high SNR: use larger threshold (3.0-5.0)
        
    Returns
    -------
    int
        Estimated number of sources
    """
    if len(eigenvalues) < 2:
        return 0
    
    # Compute ratios
    ratios = eigenvalues[:-1] / eigenvalues[1:]
    
    # Find first ratio exceeding threshold
    exceeds = np.where(ratios >= threshold)[0]
    
    if len(exceeds) > 0:
        # Return index after the largest drop
        return int(exceeds[0] + 1)  # +1 because ratio at index i is λ_i/λ_{i+1}
    else:
        return 0  # No significant drop found


# ============================================================================
# APPROACH 2: Modified MDL with Exponential Decay Model
# ============================================================================

def estimate_num_sources_mdl_exponential(eigenvalues: np.ndarray, 
                                         num_snapshots: int,
                                         decay_penalty: float = 0.1) -> int:
    """
    Modified MDL that accounts for exponentially decaying noise eigenvalues.
    
    Instead of assuming geometric_mean ≈ arithmetic_mean for noise,
    we add a penalty term for eigenvalue decay rate.
    
    MDL(k) = -T(M-k) * log(geom/arith) + 0.5*k*(2M-k)*log(T) + α*decay_rate(k:M)
    
    Parameters
    ----------
    eigenvalues : np.ndarray
        Eigenvalues sorted in descending order
    num_snapshots : int
        Number of time snapshots
    decay_penalty : float, optional
        Weight for decay penalty term (default: 0.1)
        
    Returns
    -------
    int
        Estimated number of sources
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
                    # Standard MDL terms
                    term1 = -T * (M - k) * np.log(geometric_mean / arithmetic_mean)
                    term2 = 0.5 * k * (2 * M - k) * np.log(T)
                    
                    # Additional penalty for non-flat noise floor
                    # Measure variance of log-eigenvalues (should be ~0 for flat floor)
                    log_eigenvals = np.log(noise_eigenvals)
                    decay_variance = np.var(log_eigenvals)
                    term3 = decay_penalty * T * (M - k) * decay_variance
                    
                    mdl[k] = term1 + term2 + term3
                else:
                    mdl[k] = np.inf
            else:
                mdl[k] = np.inf
    
    return int(np.argmin(mdl))


# ============================================================================
# APPROACH 3: Normalized Eigenvalue Gap Detection
# ============================================================================

def estimate_num_sources_gap_detection(eigenvalues: np.ndarray,
                                      gap_threshold: float = 0.5) -> int:
    """
    Detect number of sources by finding the normalized eigenvalue gap.
    
    Computes: gap_i = (λ_i - λ_{i+1}) / λ_i
    
    This normalizes the gap by the current eigenvalue magnitude, making it
    more robust to absolute eigenvalue scales.
    
    Parameters
    ----------
    eigenvalues : np.ndarray
        Eigenvalues sorted in descending order
    gap_threshold : float, optional
        Normalized gap threshold (default: 0.5)
        - Typical range: 0.3 - 0.7
        
    Returns
    -------
    int
        Estimated number of sources
    """
    if len(eigenvalues) < 2:
        return 0
    
    # Compute normalized gaps
    normalized_gaps = (eigenvalues[:-1] - eigenvalues[1:]) / eigenvalues[:-1]
    
    # Find first gap exceeding threshold
    exceeds = np.where(normalized_gaps >= gap_threshold)[0]
    
    if len(exceeds) > 0:
        return int(exceeds[0] + 1)
    else:
        return 0


# ============================================================================
# APPROACH 4: Two-Stage MDL (First find approximate, then refine)
# ============================================================================

def estimate_num_sources_two_stage_mdl(eigenvalues: np.ndarray,
                                       num_snapshots: int,
                                       ratio_threshold: float = 2.0) -> int:
    """
    Two-stage approach:
    1. Use ratio thresholding to find approximate boundary
    2. Apply MDL in a narrow window around that boundary
    
    This combines the robustness of ratio methods with the statistical
    rigor of MDL.
    
    Parameters
    ----------
    eigenvalues : np.ndarray
        Eigenvalues sorted in descending order
    num_snapshots : int
        Number of time snapshots
    ratio_threshold : float, optional
        Initial ratio threshold for stage 1
        
    Returns
    -------
    int
        Estimated number of sources
    """
    M = len(eigenvalues)
    
    # Stage 1: Find approximate boundary using ratios
    if M < 2:
        return 0
    
    ratios = eigenvalues[:-1] / eigenvalues[1:]
    exceeds = np.where(ratios >= ratio_threshold)[0]
    
    if len(exceeds) == 0:
        return 0
    
    k_approx = exceeds[0] + 1
    
    # Stage 2: Refine using MDL in a window
    window_size = 3
    k_min = max(0, k_approx - window_size)
    k_max = min(M, k_approx + window_size)
    
    T = num_snapshots
    mdl = np.zeros(k_max - k_min)
    
    for idx, k in enumerate(range(k_min, k_max)):
        if k == M:
            mdl[idx] = np.inf
        else:
            noise_eigenvals = eigenvalues[k:]
            
            if len(noise_eigenvals) > 0 and np.all(noise_eigenvals > 0):
                geometric_mean = np.prod(noise_eigenvals) ** (1.0 / len(noise_eigenvals))
                arithmetic_mean = np.mean(noise_eigenvals)
                
                if geometric_mean > 0 and arithmetic_mean > 0:
                    term1 = -T * (M - k) * np.log(geometric_mean / arithmetic_mean)
                    term2 = 0.5 * k * (2 * M - k) * np.log(T)
                    mdl[idx] = term1 + term2
                else:
                    mdl[idx] = np.inf
            else:
                mdl[idx] = np.inf
    
    return int(k_min + np.argmin(mdl))


# ============================================================================
# APPROACH 5: Weighted MDL (Down-weight later eigenvalues)
# ============================================================================

def estimate_num_sources_weighted_mdl(eigenvalues: np.ndarray,
                                     num_snapshots: int) -> int:
    """
    Weighted MDL that gives more importance to early noise eigenvalues.
    
    When computing geometric/arithmetic means, use exponentially decreasing
    weights to emphasize eigenvalues closer to the signal/noise boundary.
    
    This reduces the impact of far-away decaying eigenvalues on the MDL criterion.
    
    Parameters
    ----------
    eigenvalues : np.ndarray
        Eigenvalues sorted in descending order
    num_snapshots : int
        Number of time snapshots
        
    Returns
    -------
    int
        Estimated number of sources
    """
    M = len(eigenvalues)
    T = num_snapshots
    
    mdl = np.zeros(M)
    
    for k in range(M):
        if k == M:
            mdl[k] = np.inf
        else:
            noise_eigenvals = eigenvalues[k:]
            n_noise = len(noise_eigenvals)
            
            if n_noise > 0 and np.all(noise_eigenvals > 0):
                # Create exponentially decreasing weights
                # First noise eigenvalue gets weight 1.0, last gets weight ~0.1
                weights = np.exp(-0.5 * np.arange(n_noise))
                weights = weights / np.sum(weights)  # Normalize
                
                # Weighted geometric mean
                log_weighted_geom = np.sum(weights * np.log(noise_eigenvals))
                geometric_mean = np.exp(log_weighted_geom)
                
                # Weighted arithmetic mean
                arithmetic_mean = np.sum(weights * noise_eigenvals)
                
                if geometric_mean > 0 and arithmetic_mean > 0:
                    term1 = -T * (M - k) * np.log(geometric_mean / arithmetic_mean)
                    term2 = 0.5 * k * (2 * M - k) * np.log(T)
                    mdl[k] = term1 + term2
                else:
                    mdl[k] = np.inf
            else:
                mdl[k] = np.inf
    
    return int(np.argmin(mdl))


# ============================================================================
# APPROACH 6: Adaptive Threshold Selection
# ============================================================================

def estimate_num_sources_adaptive_threshold(eigenvalues: np.ndarray) -> int:
    """
    Adaptively select threshold based on eigenvalue statistics.
    
    Uses the median ratio as a baseline and looks for ratios significantly
    above this baseline.
    
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
    
    ratios = eigenvalues[:-1] / eigenvalues[1:]
    
    # Compute adaptive threshold as median + k*MAD (median absolute deviation)
    median_ratio = np.median(ratios)
    mad = np.median(np.abs(ratios - median_ratio))
    adaptive_threshold = median_ratio + 2.0 * mad  # 2.0 is a tunable parameter
    
    # Find first ratio exceeding adaptive threshold
    exceeds = np.where(ratios >= adaptive_threshold)[0]
    
    if len(exceeds) > 0:
        return int(exceeds[0] + 1)
    else:
        return 0


# ============================================================================
# Utility: Evaluate all methods and return best consensus
# ============================================================================

def estimate_num_sources_consensus(eigenvalues: np.ndarray,
                                  num_snapshots: int) -> Tuple[int, dict]:
    """
    Apply multiple methods and return consensus estimate.
    
    Parameters
    ----------
    eigenvalues : np.ndarray
        Eigenvalues sorted in descending order
    num_snapshots : int
        Number of time snapshots
        
    Returns
    -------
    consensus : int
        Consensus estimate (median or mode)
    estimates : dict
        Dictionary with estimates from each method
    """
    estimates = {
        'ratio_threshold': estimate_num_sources_ratio_threshold(eigenvalues, threshold=2.5),
        'gap_detection': estimate_num_sources_gap_detection(eigenvalues, gap_threshold=0.4),
        'two_stage_mdl': estimate_num_sources_two_stage_mdl(eigenvalues, num_snapshots),
        'weighted_mdl': estimate_num_sources_weighted_mdl(eigenvalues, num_snapshots),
        'mdl_exponential': estimate_num_sources_mdl_exponential(eigenvalues, num_snapshots),
        'adaptive_threshold': estimate_num_sources_adaptive_threshold(eigenvalues),
    }
    
    # Use median as consensus
    all_estimates = list(estimates.values())
    consensus = int(np.median(all_estimates))
    
    return consensus, estimates


# ============================================================================
# Testing function
# ============================================================================

if __name__ == "__main__":
    # Test with example UNet eigenvalues from your analysis
    print("=" * 80)
    print("TESTING MODIFIED SOURCE ESTIMATION METHODS")
    print("=" * 80)
    
    # Example from SNR=0dB (True sources: 3)
    unet_eigs = np.array([1.178359, 0.966139, 0.562865, 0.112017, 
                         0.027369, 0.013709, 0.009949, 0.009057])
    
    noisy_eigs = np.array([2.432283, 2.051009, 1.945385, 1.123815,
                          1.022072, 0.976914, 0.900575, 0.848246])
    
    T = 512  # num snapshots
    
    print("\n📊 UNet Eigenvalues (True: 3 sources)")
    print(f"   {unet_eigs}")
    
    print("\n🔍 Method Comparisons:")
    print("-" * 80)
    
    # Standard MDL (for reference)
    from source_estimation_validation import estimate_num_sources_mdl, estimate_num_sources_aic
    standard_mdl = estimate_num_sources_mdl(unet_eigs, T)
    standard_aic = estimate_num_sources_aic(unet_eigs, T)
    print(f"Standard MDL:              {standard_mdl} sources")
    print(f"Standard AIC:              {standard_aic} sources")
    print()
    
    # Try all new methods
    ratio_est = estimate_num_sources_ratio_threshold(unet_eigs, threshold=2.5)
    print(f"Ratio Threshold (2.5):     {ratio_est} sources")
    
    gap_est = estimate_num_sources_gap_detection(unet_eigs, gap_threshold=0.4)
    print(f"Gap Detection (0.4):       {gap_est} sources")
    
    two_stage = estimate_num_sources_two_stage_mdl(unet_eigs, T, ratio_threshold=2.5)
    print(f"Two-Stage MDL:             {two_stage} sources")
    
    weighted = estimate_num_sources_weighted_mdl(unet_eigs, T)
    print(f"Weighted MDL:              {weighted} sources")
    
    exp_mdl = estimate_num_sources_mdl_exponential(unet_eigs, T, decay_penalty=0.1)
    print(f"MDL + Exponential Penalty: {exp_mdl} sources")
    
    adaptive = estimate_num_sources_adaptive_threshold(unet_eigs)
    print(f"Adaptive Threshold:        {adaptive} sources")
    
    consensus, all_est = estimate_num_sources_consensus(unet_eigs, T)
    print(f"\n✅ Consensus Estimate:     {consensus} sources")
    
    print("\n" + "=" * 80)
    print("📊 Noisy Eigenvalues (True: 3 sources)")
    print(f"   {noisy_eigs}")
    print("\n🔍 Standard methods on noisy:")
    standard_mdl_noisy = estimate_num_sources_mdl(noisy_eigs, T)
    print(f"Standard MDL:              {standard_mdl_noisy} sources")
