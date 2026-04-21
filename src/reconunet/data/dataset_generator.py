"""
Simple Controlled Dataset Generator for DOA Estimation.

This module provides systematic parameter sweep generation with simple configuration
and easy filtering for DOA estimation research.
"""

import numpy as np
import h5py
import json
import yaml
import argparse
import os
import hashlib  # For deterministic per-combination seeding
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from pathlib import Path
from tqdm import tqdm
import itertools
import warnings

# Import existing signalgen classes
try:
    from signalgen import SignalConfig, SignalGenerator, ArrayConfig, ArrayModel, ReceivedSignal
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from signalgen import SignalConfig, SignalGenerator, ArrayConfig, ArrayModel, ReceivedSignal

# Import MUSIC algorithm
try:
    from ..models.classic.music import MUSIC
except ImportError:
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from models.classic.music import MUSIC
    except ImportError:
        print("Warning: MUSIC implementation not found. MUSIC estimates will be disabled.")
        MUSIC = None


@dataclass
class SimpleDatasetConfig:
    """Simple configuration for controlled dataset generation."""
    
    # === PARAMETER RANGES ===
    angles_deg: Optional[Union[List, range]] = None  # If None, uses random from [0, 360)
    snr_db: Optional[Union[List, range]] = None      # If None, uses random from [-20, 20]
    num_snapshots: Optional[List[int]] = None        # If None, uses random from [256, 512, 1024]
    num_sources: Optional[List[List[int]]] = None    # [[main, interference], ...] If None, uses [[1,0], [1,1], [1,2]]
    sir_db: Optional[Union[List, range]] = None      # If None, uses random from [-20, 20] when interference present
    num_multipath: Optional[List[int]] = None        # If None, uses [0, 1, 2] (0 means no multipath)
    array_imperfections: Optional[List[bool]] = None # If None, uses [True, False]
    
    # === RANDOM SAMPLING CONTROL ===
    # Set to True to randomly sample from parameter lists instead of systematic sweep
    random_sampling_mode: bool = False
    random_sample_params: Optional[List[str]] = None  # Parameters to sample randomly: ['angles_deg', 'snr_db', etc.]
    
    # === GENERATION CONTROL ===
    examples_per_combination: int = 100
    dataset_name: str = "simple_doa_dataset"
    
    # === ARRAY CONFIGURATION ===
    array_type: str = "triangular"  # Array type: 'triangular', 'linear', 'circular', 'cross', 'custom'
    num_elements: int = 4           # Number of array elements
    carrier_freq: float = 2.45e9
    sampling_freq: float = 1e6
    
    # === OUTPUT CONTROL ===
    save_received_signal: bool = True
    save_covariance_matrix: bool = True
    save_clean_covariance_matrix: bool = False  # NEW: Clean covariance without noise/multipath
    save_autocorrelation_matrix: bool = False   # NEW: Save autocorrelation matrices
    save_steering_vectors: bool = True
    save_array_metadata: bool = True  # Enable to save/load individual ArrayModel objects per sample
    
    # === AUTOCORRELATION CONTROL ===
    autocorr_tau: int = 8  # NEW: Number of time lags for autocorrelation computation
    
    # === MUSIC ALGORITHM CONTROL ===
    save_music_estimates: bool = False  # NEW: Save MUSIC angle estimates and spectrum
    music_angle_grid_deg: Optional[List[float]] = None  # NEW: Angle grid for MUSIC spectrum (if None, uses 0:1:359)
    music_num_sources: Optional[int] = None  # NEW: Number of sources for MUSIC (if None, uses true number)
    
    # === ANGLE SEPARATION CONTROL ===
    min_angle_separation_deg: float = 10.0  # Minimum angular separation between sources in degrees
    
    def __post_init__(self):
        """Set defaults for None parameters."""
        if self.angles_deg is None:
            self.angles_deg = list(range(0, 360, 30))  # Every 30 degrees
        elif isinstance(self.angles_deg, range):
            self.angles_deg = list(self.angles_deg)
            
        if self.snr_db is None:
            self.snr_db = list(range(-20, 21, 10))  # Every 10 dB
        elif isinstance(self.snr_db, range):
            self.snr_db = list(self.snr_db)
            
        if self.num_snapshots is None:
            self.num_snapshots = [256, 512, 1024]
            
        if self.num_sources is None:
            self.num_sources = [[1, 0], [1, 1], [1, 2]]  # [main, interference]
            
        if self.sir_db is None:
            self.sir_db = list(range(-20, 21, 10))
        elif isinstance(self.sir_db, range):
            self.sir_db = list(self.sir_db)
            
        if self.num_multipath is None:
            self.num_multipath = [0, 1, 2]  # 0 means no multipath
            
        if self.array_imperfections is None:
            self.array_imperfections = [True, False]
            
        # Set MUSIC defaults
        if self.music_angle_grid_deg is None:
            self.music_angle_grid_deg = list(range(0, 360, 1))  # 1-degree resolution
    
    def get_total_combinations(self) -> int:
        """Calculate total number of valid parameter combinations."""
        # Count valid combinations for only the constrained parameters
        valid_source_multipath_combinations = 0
        
        for sources, multipath in itertools.product(self.num_sources, self.num_multipath):
            # Apply multipath constraint: main + interference + multipath ≤ num_elements - 1
            total_signals = sources[0] + sources[1] + multipath
            if total_signals <= self.num_elements - 1:
                valid_source_multipath_combinations += 1
        
        # Multiply by all other unconstrained parameters
        other_combinations = (
            len(self.angles_deg) *
            len(self.snr_db) *
            len(self.num_snapshots) *
            len(self.sir_db) *
            len(self.array_imperfections)
        )
        
        return valid_source_multipath_combinations * other_combinations
    
    def get_total_samples(self) -> int:
        """Calculate total number of samples."""
        return self.get_total_combinations() * self.examples_per_combination


class ControlledDatasetGenerator:
    """Simple controlled dataset generator with systematic parameter sweeps."""
    
    def __init__(self, config: SimpleDatasetConfig, seed: Optional[int] = None):
        """Initialize generator."""
        self.config = config
        self.generator_seed = seed  # Store the generator seed
        self.rng = np.random.default_rng(seed)
        # Cache for MUSIC estimates to avoid recomputation for same scenarios
        self.music_cache = {} if config.save_music_estimates else None
        
    def _create_parameter_combinations(self) -> List[Dict]:
        """Create parameter combinations with constraints and optional random sampling."""
        if self.config.random_sampling_mode:
            return self._create_random_combinations()
        else:
            return self._create_systematic_combinations()
    
    def _create_systematic_combinations(self) -> List[Dict]:
        """Create all valid parameter combinations systematically."""
        combinations = []
        
        for angle, snr, snapshots, sources, sir, multipath, imperfections in itertools.product(
            self.config.angles_deg,
            self.config.snr_db,
            self.config.num_snapshots,
            self.config.num_sources,
            self.config.sir_db,
            self.config.num_multipath,
            self.config.array_imperfections
        ):
            # Apply multipath constraint: main + interference + multipath ≤ num_elements - 1
            total_signals = sources[0] + sources[1] + multipath
            if total_signals > self.config.num_elements - 1:
                continue  # Skip invalid combinations
            
            # Only use SIR if there are interference sources
            if sources[1] == 0:  # No interference
                actual_sir = None
            else:
                actual_sir = sir
                
            combinations.append({
                'angle_deg': angle,
                'snr_db': snr,
                'num_snapshots': snapshots,
                'num_sources': sources,  # [main, interference]
                'sir_db': actual_sir,
                'num_multipath': multipath,
                'array_imperfections': imperfections
            })
        
        return combinations
    
    def _create_random_combinations(self) -> List[Dict]:
        """Create parameter combinations with random sampling for specified parameters."""
        if not self.config.random_sample_params:
            return self._create_systematic_combinations()
        
        random_params = set(self.config.random_sample_params)
        
        # Determine which parameters are systematic (not random)
        systematic_params = {}
        if 'angles_deg' not in random_params:
            systematic_params['angles_deg'] = self.config.angles_deg
        if 'snr_db' not in random_params:
            systematic_params['snr_db'] = self.config.snr_db
        if 'sir_db' not in random_params:
            systematic_params['sir_db'] = self.config.sir_db
        
        # Always systematic (not supported for random sampling)
        systematic_params['num_snapshots'] = self.config.num_snapshots
        systematic_params['num_sources'] = self.config.num_sources
        systematic_params['num_multipath'] = self.config.num_multipath
        systematic_params['array_imperfections'] = self.config.array_imperfections
        
        # Create all combinations of systematic parameters
        import itertools
        param_names = list(systematic_params.keys())
        param_values = [systematic_params[name] for name in param_names]
        
        combinations = []
        for combination in itertools.product(*param_values):
            combo_dict = dict(zip(param_names, combination))
            
            # Apply multipath constraint
            sources = combo_dict['num_sources']
            multipath = combo_dict['num_multipath'] 
            total_signals = sources[0] + sources[1] + multipath
            if total_signals > self.config.num_elements - 1:
                continue  # Skip invalid combinations
            
            # Create the parameter combination
            param_combo = {
                'angle_deg': combo_dict.get('angles_deg', self.rng.choice(self.config.angles_deg)),
                'snr_db': combo_dict.get('snr_db', self.rng.choice(self.config.snr_db)),
                'num_snapshots': combo_dict['num_snapshots'],
                'num_sources': combo_dict['num_sources'],
                'sir_db': combo_dict.get('sir_db', self.rng.choice(self.config.sir_db) if combo_dict['num_sources'][1] > 0 else None),
                'num_multipath': combo_dict['num_multipath'],
                'array_imperfections': combo_dict['array_imperfections']
            }
            
            # Handle SIR logic
            if param_combo['num_sources'][1] == 0:  # No interference
                param_combo['sir_db'] = None
            # elif 'sir_db' in random_params:
            #     param_combo['sir_db'] = self.rng.choice(self.config.sir_db)
            
            combinations.append(param_combo)
        
        return combinations
    
    def _create_array_config(self, use_imperfections: bool) -> ArrayConfig:
        """Create array configuration based on imperfections flag."""
        return ArrayConfig(
            array_type=self.config.array_type,
            num_elements=self.config.num_elements,
            carrier_freq=self.config.carrier_freq,
            enable_gain_phase_errors=use_imperfections,
            enable_mutual_coupling=use_imperfections,
            position_error_std=0.001 if use_imperfections else 0.0
        )
    
    def _generate_separated_angles(self, existing_angles: List[float], num_new_angles: int, 
                                 angle_range: Tuple[float, float], min_separation: float,
                                 max_attempts: int = 100) -> List[float]:
        """
        Generate new angles with minimum separation constraint using vectorized approach.
        
        Args:
            existing_angles: List of already placed angles
            num_new_angles: Number of new angles to generate
            angle_range: (min_angle, max_angle) range for new angles
            min_separation: Minimum angular separation in degrees
            max_attempts: Maximum attempts to find valid angle set
            
        Returns:
            List of new angles that satisfy separation constraint
        """
        if num_new_angles == 0:
            return []
            
        angle_min, angle_max = angle_range
        is_circular = angle_max > 180  # Full circle vs half circle
        
        for attempt in range(max_attempts):
            # Generate all candidate angles at once
            candidates = self.rng.uniform(angle_min, angle_max, size=num_new_angles)
            # Generate all candidate angles at once (as integers)
            # candidates = self.rng.integers(int(angle_min), int(angle_max) + 1, size=num_new_angles).astype(np.float64)
            
            # Check if this set of angles satisfies separation constraints
            if self._check_angle_separation(existing_angles, candidates.tolist(), 
                                          min_separation, is_circular):
                return candidates.tolist()
        
        # If we can't find a valid random set, use deterministic placement
        return self._generate_deterministic_angles(existing_angles, num_new_angles, 
                                                 angle_range, min_separation)
    
    def _check_angle_separation(self, existing_angles: List[float], new_angles: List[float], 
                              min_separation: float, is_circular: bool) -> bool:
        """
        Check if all angles satisfy minimum separation constraint using efficient sorting approach.
        
        Args:
            existing_angles: Already placed angles
            new_angles: New candidate angles to check
            min_separation: Minimum separation in degrees
            is_circular: Whether to use circular distance calculation
            
        Returns:
            True if all separations are valid
        """
        if not new_angles:
            return True
            
        # Combine and sort all angles
        all_angles = np.array(existing_angles + new_angles)
        sorted_angles = np.sort(all_angles)
        
        # Check consecutive differences
        diffs = np.diff(sorted_angles)
        
        if is_circular:
            # For circular case, also check wrap-around distance
            wrap_around_diff = 360 - (sorted_angles[-1] - sorted_angles[0])
            all_diffs = np.append(diffs, wrap_around_diff)
        else:
            all_diffs = diffs
        
        # Check if all separations meet minimum requirement
        return np.all(all_diffs >= min_separation)
    
    def _generate_deterministic_angles(self, existing_angles: List[float], num_new_angles: int,
                                     angle_range: Tuple[float, float], min_separation: float) -> List[float]:
        """
        Generate angles deterministically when random placement fails.
        
        Args:
            existing_angles: Already placed angles
            num_new_angles: Number of angles to generate
            angle_range: (min_angle, max_angle) range
            min_separation: Minimum separation in degrees
            
        Returns:
            List of deterministically placed angles
        """
        angle_min, angle_max = angle_range
        new_angles = []
        
        # Start from a safe position
        if existing_angles:
            start_angle = max(existing_angles) + min_separation
            # Wrap around within the valid range
            if start_angle > angle_max:
                start_angle = angle_min + (start_angle - angle_max)
        else:
            start_angle = angle_min
        
        # Place angles with exact minimum separation
        for i in range(num_new_angles):
            candidate = start_angle + i * min_separation
            
            # Wrap around within the valid range if needed (with safety check)
            wrap_count = 0
            while candidate > angle_max and wrap_count < 10:  # Prevent infinite loop
                candidate = angle_min + (candidate - angle_max)
                wrap_count += 1
            
            # If we couldn't find a valid position, use modulo wrapping
            if candidate > angle_max or candidate < angle_min:
                angle_range = angle_max - angle_min
                if angle_range > 0:
                    candidate = angle_min + ((start_angle + i * min_separation - angle_min) % angle_range)
                else:
                    candidate = angle_min  # Fallback if no range available
            
            new_angles.append(candidate)
        
        return new_angles
    
    def get_music_cache_stats(self) -> Dict[str, int]:
        """Get statistics about MUSIC cache usage."""
        if self.music_cache is None:
            return {'enabled': False, 'cache_size': 0, 'cache_hits': 0, 'cache_misses': 0}
        
        return {
            'enabled': True,
            'cache_size': len(self.music_cache),
            'estimated_speedup': f"{len(self.music_cache)}x for repeated scenarios"
        }
    
    def _compute_autocorrelation_matrix(self, received_signal: np.ndarray, tau: int) -> np.ndarray:
        """
        Compute autocorrelation matrix from received signal using highly optimized methods.
        
        Args:
            received_signal: [M, T] complex received signal
            tau: Number of time lags
            
        Returns:
            autocorr_matrix: [tau, 2M, M] autocorrelation matrix (real/imag stacked)
        """
        M, T = received_signal.shape
        
        # Pre-allocate output array
        autocorr_matrix = np.zeros((tau, 2*M, M), dtype=np.float32)
        
        # Early exit for edge cases
        if T <= 1 or tau <= 0:
            return autocorr_matrix
        
        max_lag = min(tau, T - 1)  # Don't compute beyond available samples
        
        # Choose optimization method based on problem size
        if T > 1024 and tau > 16:
            # FFT-based method for large signals (much faster for large T and tau)
            return self._compute_autocorr_fft_optimized(received_signal, tau, autocorr_matrix)
        else:
            # Direct method for smaller signals (faster for small T and tau)
            return self._compute_autocorr_direct_optimized(received_signal, tau, autocorr_matrix, max_lag)
    
    def _compute_autocorr_direct_optimized(self, received_signal: np.ndarray, tau: int, 
                                         autocorr_matrix: np.ndarray, max_lag: int) -> np.ndarray:
        """Optimized direct computation method."""
        M, T = received_signal.shape
        
        # Vectorized computation using broadcasting and einsum
        for lag in range(max_lag):
            samples_available = T - lag
            
            # Use einsum for optimal performance (faster than matmul for this pattern)
            x1 = received_signal[:, :samples_available]
            x2 = received_signal[:, lag:lag + samples_available]
            
            # Compute Rx_lag = (1/N) * x1 @ x2^H using einsum
            Rx_lag = np.einsum('ij,kj->ik', x1, x2.conj(), optimize=True) / samples_available
            
            # Efficiently assign real and imaginary parts
            autocorr_matrix[lag, :M, :] = Rx_lag.real
            autocorr_matrix[lag, M:, :] = Rx_lag.imag
        
        return autocorr_matrix
    
    def _compute_autocorr_fft_optimized(self, received_signal: np.ndarray, tau: int, 
                                      autocorr_matrix: np.ndarray) -> np.ndarray:
        """FFT-based autocorrelation computation for large signals."""
        M, T = received_signal.shape
        
        # Pad signal for FFT-based correlation
        next_power_of_2 = int(2 ** np.ceil(np.log2(2 * T - 1)))
        
        # Compute autocorrelation for each pair of sensors using FFT
        for i in range(M):
            for j in range(M):
                # FFT-based cross-correlation
                x_i = received_signal[i, :]
                x_j = received_signal[j, :]
                
                # Zero-pad signals
                x_i_padded = np.zeros(next_power_of_2, dtype=x_i.dtype)
                x_j_padded = np.zeros(next_power_of_2, dtype=x_j.dtype)
                x_i_padded[:T] = x_i
                x_j_padded[:T] = x_j
                
                # FFT-based correlation: IFFT(FFT(x_i) * conj(FFT(x_j)))
                fft_i = np.fft.fft(x_i_padded)
                fft_j = np.fft.fft(x_j_padded)
                correlation = np.fft.ifft(fft_i * fft_j.conj())
                
                # Extract autocorrelation values for different lags
                for lag in range(min(tau, T)):
                    samples_available = T - lag
                    if samples_available > 0:
                        # Normalize by number of overlapping samples
                        Rij_lag = correlation[lag] / samples_available
                        
                        # Store real and imaginary parts
                        autocorr_matrix[lag, i, j] = Rij_lag.real
                        autocorr_matrix[lag, M + i, j] = Rij_lag.imag
        
        return autocorr_matrix
    
    def _get_music_cache_key(self, param_combo: Dict, example_idx: int) -> str:
        """
        Create a cache key for MUSIC estimates based on parameters that affect the clean covariance matrix.
        Excludes array_imperfections and SNR since MUSIC is computed from clean covariance.
        """
        # Only include parameters that affect the clean signal generation
        key_params = (
            param_combo['angle_deg'],
            # SNR is NOT included since it doesn't affect clean covariance matrix
            param_combo['num_snapshots'],
            param_combo['num_sources'][0],
            param_combo['num_sources'][1],
            -999 if param_combo['sir_db'] is None else param_combo['sir_db'],
            param_combo['num_multipath'],
            example_idx,  # Same example index should give same result
            # Note: array_imperfections and snr_db are NOT included since clean covariance is noise-free and perfect
        )
        return str(hash(key_params))
    
    def _compute_music_estimates(self, clean_covariance_matrix: np.ndarray, array_model: ArrayModel, 
                                signal_config: SignalConfig, param_combo: Dict, example_idx: int) -> Dict:
        """
        Compute MUSIC DOA estimates and spectrum from clean covariance matrix.
        Uses caching to avoid recomputation for the same underlying scenario.
        
        Args:
            clean_covariance_matrix: [M, M] complex clean covariance matrix
            array_model: ArrayModel instance for steering vector computation
            signal_config: SignalConfig containing true source information
            param_combo: Parameter combination dict containing source configuration
            example_idx: Example index for cache key generation
            
        Returns:
            dict: Dictionary containing MUSIC estimates and spectrum
        """
        if MUSIC is None:
            return {
                'estimated_angles': np.array([], dtype=np.float32),
                'spectrum': np.array([], dtype=np.float32),
                'angle_grid': np.array([], dtype=np.float32),
                'num_sources_used': 0
            }
        
        # Check cache first if caching is enabled
        if self.music_cache is not None:
            cache_key = self._get_music_cache_key(param_combo, example_idx)
            if cache_key in self.music_cache:
                cached_result = self.music_cache[cache_key].copy()
                # Add current true_angles (might be different due to array_imperfections affecting angle generation)
                num_main = param_combo['num_sources'][0]
                num_interference = param_combo['num_sources'][1]
                true_num_sources = num_main + num_interference
                true_source_angles = signal_config.angles[:true_num_sources]
                cached_result['true_angles'] = np.array(true_source_angles, dtype=np.float32)
                return cached_result
        
        try:
            # Determine number of sources to use
            # true_num_sources should be the total number of sources (main + interference), excluding multipath
            # param_combo['num_sources'] = [num_main, num_interference]
            num_main = param_combo['num_sources'][0]
            num_interference = param_combo['num_sources'][1]
            true_num_sources = num_main + num_interference  # Only main + interference, not multipath
            music_num_sources = self.config.music_num_sources if self.config.music_num_sources is not None else true_num_sources
            
            # Create angle grid for MUSIC spectrum computation
            angle_grid = np.array(self.config.music_angle_grid_deg, dtype=np.float32)
            
            # Create MUSIC instance
            music = MUSIC(
                array_model=array_model,
                scan_angles_deg=angle_grid,
                num_sources=music_num_sources
            )
            
            # Set the covariance matrix for spectrum computation
            music.set_received_covariance(clean_covariance_matrix)
            
            # Compute MUSIC estimates
            estimated_angles, spectrum = music.estimate_doa()
            
            # Get only the main and interference source angles (exclude multipath)
            # signal_config.angles contains all angles including multipath
            # We want only the first (num_main + num_interference) angles
            true_source_angles = signal_config.angles[:true_num_sources]
            
            result = {
                'estimated_angles': estimated_angles.astype(np.float32),
                'spectrum': spectrum.astype(np.float32),
                'angle_grid': angle_grid,
                'num_sources_used': music_num_sources,
                'true_angles': np.array(true_source_angles, dtype=np.float32)
            }
            
            # Cache the result (excluding true_angles since they might vary with array_imperfections)
            if self.music_cache is not None:
                cache_key = self._get_music_cache_key(param_combo, example_idx)
                cached_result = result.copy()
                # Don't cache true_angles since they might be different for different array_imperfections
                cached_result.pop('true_angles', None)
                self.music_cache[cache_key] = cached_result
            
            return result
            
        except Exception as e:
            # If MUSIC computation fails, return empty results
            print(f"Warning: MUSIC computation failed: {e}")
            # Still calculate the correct true_angles even in error case
            num_main = param_combo['num_sources'][0]
            num_interference = param_combo['num_sources'][1]
            true_num_sources = num_main + num_interference
            true_source_angles = signal_config.angles[:true_num_sources]
            
            return {
                'estimated_angles': np.array([], dtype=np.float32),
                'spectrum': np.array([], dtype=np.float32),
                'angle_grid': np.array(self.config.music_angle_grid_deg, dtype=np.float32),
                'num_sources_used': 0,
                'true_angles': np.array(true_source_angles, dtype=np.float32)
            }
    
    def _create_signal_config(self, param_combo: Dict) -> SignalConfig:
        """Create signal configuration for parameter combination."""
        num_main = param_combo['num_sources'][0]
        num_interference = param_combo['num_sources'][1]
        total_sources = num_main + num_interference
        
        # Generate angles for all sources
        main_angle = param_combo['angle_deg']
        source_angles = [main_angle]
        
        # Total number of additional sources needed
        total_additional = num_interference + (num_main - 1)
        
        if total_additional > 0:
            # Determine angle range based on the configured angle range (same as main signal)
            # This ensures interference sources are in the same angular sector as main sources
            angle_min = min(self.config.angles_deg)
            angle_max = max(self.config.angles_deg)
            
            # Generate all additional angles at once with separation constraint
            additional_angles = self._generate_separated_angles(
                existing_angles=source_angles,
                num_new_angles=total_additional,
                angle_range=(angle_min, angle_max),
                min_separation=self.config.min_angle_separation_deg,
                max_attempts=1000
            )
            
            source_angles.extend(additional_angles)
        
        # Create frequencies for each source
        base_freq = 100e3
        frequencies = [base_freq] * num_main
        
        # Add interference frequencies with offset
        if num_interference > 0:
            interference_freq = base_freq + 50e3  # 50 kHz offset
            frequencies.extend([interference_freq] * num_interference)
        
        # Create powers based on SIR
        powers = [0.0] * total_sources  # All sources start at 0 dB
        main_source_power = 0.0         # Main source power in dB
        
        # Set SIR for interference sources (SignalConfig expects SIR for all non-first sources)
        sir_db_list = None
        if total_sources > 1:
            sir_db_list = []
            
            # For additional main sources (sources 1 to num_main-1), use 0 dB SIR (equal power)
            for i in range(1, num_main):
                sir_db_list.append(0.0)  # Same power as first main source
            
            # For interference sources, use the specified SIR
            if num_interference > 0 and param_combo['sir_db'] is not None:
                for i in range(num_interference):
                    sir_db_list.append(param_combo['sir_db'])
            elif num_interference > 0:
                # Default interference power equal to main sources
                for i in range(num_interference):
                    sir_db_list.append(0.0)
            
            # The SignalConfig will calculate powers automatically from SIR values
            powers = None  # Let SignalConfig calculate from SIR
        
        # Multipath configuration
        multipath_enabled = param_combo['num_multipath'] > 0
        multipath_config = None
        if multipath_enabled:
            # Simple multipath model
            num_paths = param_combo['num_multipath']
            multipath_config = {
                'path_delays': [0.1e-6 * (i+1) for i in range(num_paths)],  # Progressive delays
                'path_gains': [0.5 ** (i+1) for i in range(num_paths)],     # Decreasing gains
                'path_phases': [0.0] * num_paths                            # No additional phase shifts
            }
        
        return SignalConfig(
            fs=self.config.sampling_freq,
            T=param_combo['num_snapshots'],
            num_sources=total_sources,
            angles=source_angles,
            frequencies=frequencies,
            powers=powers,
            main_source_power=main_source_power,
            sir_db=sir_db_list,
            enable_multipath=multipath_enabled,
            num_paths=param_combo['num_multipath'],
            path_delays=multipath_config['path_delays'] if multipath_config else None,
            path_phases=multipath_config.get('path_phases') if multipath_config else None,
            path_gains=multipath_config['path_gains'] if multipath_config else None,
            noise_type='gaussian',
            use_full_bandwidth=True
        )
    
    def _generate_sample(self, param_combo: Dict, example_idx: int) -> Dict:
        """Generate a single sample for given parameter combination."""
        # ------------------------------------------------------------------
        # 1)  Deterministic seed so that the *same* waveform/noise is created
        #     for both array_imperfections = False / True variants.
        # ------------------------------------------------------------------
        # Build a tuple that excludes the imperfection flag but includes all
        # other parameters that define the physical scenario + example index
        # IMPORTANT: Include generator_seed to ensure different runs produce different results
        seed_tuple = (
            self.generator_seed if self.generator_seed is not None else 0,  # Include generator seed
            param_combo['angle_deg'],
            param_combo['snr_db'],
            param_combo['num_snapshots'],
            param_combo['num_sources'][0],
            param_combo['num_sources'][1],
            -999 if param_combo['sir_db'] is None else param_combo['sir_db'],
            param_combo['num_multipath'],
            example_idx,
        )

        # Create a stable 32-bit seed from the tuple using MD5 (hash() is randomised per run)
        seed_str = "_".join(map(str, seed_tuple))
        base_seed = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)  # 32-bit

        # Reset both the generator-level RNG and NumPy global RNG
        self.rng = np.random.default_rng(base_seed)
        np.random.seed(base_seed)

        # ------------------------------------------------------------------
        # 2)  Array model and signal generator now use deterministic seeds
        # ------------------------------------------------------------------
        array_config = self._create_array_config(param_combo['array_imperfections'])
        array_model = ArrayModel(array_config, seed=base_seed)  # deterministic

        # Apply random sampling for this specific sample if enabled
        actual_param_combo = param_combo.copy()
        if self.config.random_sampling_mode and self.config.random_sample_params:
            random_params = set(self.config.random_sample_params)
            
            # Randomly sample parameters for this individual sample
            # if 'angles_deg' in random_params:
            #     actual_param_combo['angle_deg'] = self.rng.choice(self.config.angles_deg)
            # if 'snr_db' in random_params:
            #     actual_param_combo['snr_db'] = self.rng.choice(self.config.snr_db)
            # if 'sir_db' in random_params and actual_param_combo['num_sources'][1] > 0:
            #     actual_param_combo['sir_db'] = self.rng.choice(self.config.sir_db)

        # Create signal configuration
        signal_config = self._create_signal_config(actual_param_combo)
        
        # Generate signals
        signal_generator = SignalGenerator(signal_config, seed=base_seed)
        t, signals = signal_generator.generate_signals()
        
        # COMPUTE CLEAN COVARIANCE MATRIX (before multipath and noise)
        # This should be computed with perfect array response regardless of array_imperfections
        clean_covariance_matrix = None
        if self.config.save_clean_covariance_matrix or self.config.save_music_estimates:
            # Always use perfect array configuration for clean covariance computation
            # This ensures MUSIC estimates are consistent regardless of array_imperfections
            perfect_array_config = self._create_array_config(use_imperfections=False)
            perfect_array_model = ArrayModel(perfect_array_config, seed=base_seed)
            perfect_steering_matrix = perfect_array_model.steering_matrix(signal_config.angles, nominal=True)
            
            # Create clean received signal using perfect steering and clean signals
            clean_received_signals = perfect_steering_matrix @ signals  # [M, T]
            
            # Compute clean covariance matrix
            clean_covariance_matrix = (clean_received_signals @ clean_received_signals.conj().T) / clean_received_signals.shape[1]
        
        # Apply multipath if enabled and get SMR from signal config
        if signal_config.enable_multipath:
            # Apply multipath to get combined signals and shift samples
            multipath_result = signal_generator.add_multipath(
                signals, t, signal_config.fs, 
                array_config.carrier_freq, 
                signal_config.bandwidth if hasattr(signal_config, 'bandwidth') else signal_config.fs * 0.1,
                max_paths=self.config.num_elements - 1,
                num_multipath_components=actual_param_combo['num_multipath']
            )
            # Unpack the tuple return from add_multipath
            if isinstance(multipath_result, tuple):
                signals, shift_samples = multipath_result
            else:
                signals = multipath_result
        
        # Prepare SIR and SMR values as numpy arrays (variable-length)
        if signal_config.sir_db is not None:
            # Convert list of SIR values to float32 numpy array
            sir_values = np.array(signal_config.sir_db, dtype=np.float32)
        else:
            # Sentinel value when no interference sources are present
            sir_values = np.array([-999.0], dtype=np.float32)

        if signal_config.smr is not None:
            # SMR returned from add_multipath can be a list (per multipath component)
            smr_values = np.array(signal_config.smr, dtype=np.float32)
        else:
            # Sentinel value when multipath is disabled
            smr_values = np.array([999.0], dtype=np.float32)
        
        # Get steering vectors for all sources
        if param_combo['array_imperfections']:
            steering_matrix = array_model.steering_matrix(signal_config.angles, nominal=False)
        else:
            steering_matrix = array_model.steering_matrix(signal_config.angles, nominal=True)
        
        # Create received signal using steering matrix directly
        received_signal = ReceivedSignal(steering_matrix, signals, signal_config.angles, signal_config)
        
        # Add noise based on SNR with frequency-selective noise
        received_signal_with_noise = received_signal.add_noise(
            snr_db=actual_param_combo['snr_db'],
            source_frequencies=signal_config.frequencies,
            signal_bandwidth=signal_config.bandwidth if hasattr(signal_config, 'bandwidth') else signal_config.fs * 0.1,
            sampling_frequency=signal_config.fs,
            frequency_selective=True
        )
        
        # Prepare sample data
        sample = {
            'labels': {
                'doas': np.array(signal_config.angles, dtype=np.float32),
                'snr': np.float32(actual_param_combo['snr_db']),
                'num_snapshots': np.int32(actual_param_combo['num_snapshots']),
                'num_sources': np.array(actual_param_combo['num_sources'], dtype=np.int32),
                'sir': sir_values,
                'num_multipath': np.int32(actual_param_combo['num_multipath']),
                'smr': smr_values,  # Signal-to-Multipath Ratio in dB (variable-length)
                'array_imperfections': bool(actual_param_combo['array_imperfections'])
            }
        }
        
        # Add data components based on configuration
        if self.config.save_received_signal:
            sample['received_signal'] = received_signal_with_noise.array_signals.astype(np.complex64)
        
        if self.config.save_covariance_matrix:
            # Compute sample covariance matrix
            signals_matrix = received_signal_with_noise.array_signals
            cov_matrix = (signals_matrix @ signals_matrix.conj().T) / signals_matrix.shape[1]
            sample['covariance_matrix'] = cov_matrix.astype(np.complex64)
        
        if self.config.save_clean_covariance_matrix:
            # Save the clean covariance matrix (computed before multipath and noise)
            sample['clean_covariance_matrix'] = clean_covariance_matrix.astype(np.complex64)
        
        if self.config.save_autocorrelation_matrix:
            # Compute autocorrelation matrix from the received signal with noise
            autocorr_matrix = self._compute_autocorrelation_matrix(
                received_signal_with_noise.array_signals, 
                self.config.autocorr_tau
            )
            sample['autocorrelation_matrix'] = autocorr_matrix.astype(np.float32)
        
        if self.config.save_music_estimates and clean_covariance_matrix is not None:
            # Compute MUSIC estimates from clean covariance matrix using perfect array model
            # This ensures consistent MUSIC results regardless of array_imperfections
            perfect_array_config = self._create_array_config(use_imperfections=False)
            perfect_array_model = ArrayModel(perfect_array_config, seed=base_seed)
            music_estimates = self._compute_music_estimates(
                clean_covariance_matrix, 
                perfect_array_model, 
                signal_config,
                actual_param_combo,
                example_idx
            )
            sample['music_estimates'] = music_estimates
        
        if self.config.save_steering_vectors:
            nominal_steering = array_model.steering_matrix(signal_config.angles, nominal=True)
            sample['steering_vectors'] = {
                'nominal': nominal_steering.astype(np.complex64),
                'actual': steering_matrix.astype(np.complex64)
            }
        
        if self.config.save_array_metadata:
            # Serialize array model information for HDF5 storage
            sample['array_model'] = {
                'array_config': {
                    'array_type': array_config.array_type,
                    'num_elements': array_config.num_elements,
                    'carrier_freq': array_config.carrier_freq,
                    'enable_gain_phase_errors': array_config.enable_gain_phase_errors,
                    'enable_mutual_coupling': array_config.enable_mutual_coupling,
                    'position_error_std': array_config.position_error_std,
                    'element_spacing': getattr(array_config, 'element_spacing', 0.5),
                    'radius': getattr(array_config, 'radius', 0.5),
                    'gain_error_range': getattr(array_config, 'gain_error_range', (0.975, 1.025)),
                    'phase_error_range_deg': getattr(array_config, 'phase_error_range_deg', (-5.0, 5.0)),
                    'coupling_strength': getattr(array_config, 'coupling_strength', 0.3),
                    'coupling_phase_deg': getattr(array_config, 'coupling_phase_deg', -100.0),
                    'coupling_variation': getattr(array_config, 'coupling_variation', 0.9)
                },
                'seed': base_seed,
                'array_imperfections': param_combo['array_imperfections']
            }
        
        return sample
    
    def generate_dataset(self, output_dir: Union[str, Path] = "Data/datasets") -> str:
        """Generate the complete dataset."""
        output_dir = Path(output_dir)
        
        # If relative path, make it relative to project root, not current working directory
        if not output_dir.is_absolute():
            # Find project root (look for src directory as indicator)
            current_path = Path(__file__).resolve()
            project_root = current_path
            while project_root.parent != project_root:
                if (project_root / 'src').exists():
                    break
                project_root = project_root.parent
            output_dir = project_root / output_dir
        
        # Create array-type-specific folder structure
        array_type_dir = output_dir / self.config.array_type
        dataset_dir = array_type_dir / self.config.dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        dataset_file = dataset_dir / f"{self.config.dataset_name}.h5"
        
        # Get all parameter combinations
        param_combinations = self._create_parameter_combinations()
        total_samples = len(param_combinations) * self.config.examples_per_combination
        
        print(f"Generating dataset with {len(param_combinations)} parameter combinations")
        print(f"Total samples: {total_samples}")
        print(f"Output file: {dataset_file}")
        
        with h5py.File(dataset_file, 'w') as f:
            # Create groups for different data types
            labels_group = f.create_group('labels')
            
            if self.config.save_received_signal:
                f.create_group('received_signals')
            if self.config.save_covariance_matrix:
                f.create_group('covariance_matrices')
            if self.config.save_clean_covariance_matrix:
                f.create_group('clean_covariance_matrices')
            if self.config.save_autocorrelation_matrix:
                f.create_group('autocorrelation_matrices')
            if self.config.save_music_estimates:
                f.create_group('music_estimates')
            if self.config.save_steering_vectors:
                steering_group = f.create_group('steering_vectors')
                steering_group.create_group('nominal')
                steering_group.create_group('actual')
            if self.config.save_array_metadata:
                f.create_group('array_models')
            
            # Generate samples
            sample_idx = 0
            for combo_idx, param_combo in enumerate(tqdm(param_combinations, desc="Parameter combinations")):
                for example_idx in range(self.config.examples_per_combination):
                    sample = self._generate_sample(param_combo, example_idx)
                    
                    # Save labels
                    for label_name, label_value in sample['labels'].items():
                        if label_name not in labels_group:
                            # Create dataset on first sample
                            if isinstance(label_value, np.ndarray):
                                if label_name == 'doas' or label_name == 'sir' or label_name == 'smr':
                                    # Use variable-length dataset for DOAs since they can have different numbers of sources
                                    dt = h5py.special_dtype(vlen=np.float64)
                                    labels_group.create_dataset(label_name, (total_samples,), dtype=dt, maxshape=(None,))
                                else:
                                    maxshape = (None,) + label_value.shape
                                    labels_group.create_dataset(label_name, (total_samples,) + label_value.shape,
                                                              maxshape=maxshape, dtype=label_value.dtype)
                            else:
                                if label_name == 'doas' or label_name == 'sir' or label_name == 'smr':
                                    dt = h5py.special_dtype(vlen=np.float64)
                                    labels_group.create_dataset(label_name, (total_samples,), dtype=dt, maxshape=(None,))
                                else:
                                    labels_group.create_dataset(label_name, (total_samples,),
                                                          maxshape=(None,), dtype=type(label_value))
                        
                        try:
                            labels_group[label_name][sample_idx] = label_value
                        except:
                            print(f"Error saving label: {label_name} with value: {label_value}")
                            print(f"Sample index: {sample_idx}")
                            print(f"Sample: {sample}")
                            print(f"Param combo: {param_combo}")
                            print(f"Example index: {example_idx}")
                            raise Exception(f"Error saving label: {label_name} with value: {label_value}")
                    # Save data components
                    if self.config.save_received_signal:
                        signal_data = sample['received_signal']
                        if 'data' not in f['received_signals']:
                            # Use variable-length datasets for received signals since they have different numbers of snapshots
                            dt_complex = h5py.special_dtype(vlen=np.complex128)
                            f['received_signals'].create_dataset('data', (total_samples,), dtype=dt_complex, maxshape=(None,))
                        
                        # Flatten the signal data for storage
                        f['received_signals/data'][sample_idx] = signal_data.flatten()
                        
                        # Store shape information for reconstruction
                        if 'shapes' not in f['received_signals']:
                            f['received_signals'].create_dataset('shapes', (total_samples, 2), dtype=np.int32, maxshape=(None, 2))
                        f['received_signals/shapes'][sample_idx] = signal_data.shape
                    
                    if self.config.save_covariance_matrix:
                        cov_data = sample['covariance_matrix']
                        if 'data' not in f['covariance_matrices']:
                            maxshape = (None,) + cov_data.shape
                            f['covariance_matrices'].create_dataset('data', (total_samples,) + cov_data.shape,
                                                                  maxshape=maxshape, dtype=cov_data.dtype)
                        f['covariance_matrices/data'][sample_idx] = cov_data
                    
                    if self.config.save_clean_covariance_matrix:
                        clean_cov_data = sample['clean_covariance_matrix']
                        if 'data' not in f['clean_covariance_matrices']:
                            maxshape = (None,) + clean_cov_data.shape
                            f['clean_covariance_matrices'].create_dataset('data', (total_samples,) + clean_cov_data.shape,
                                                                        maxshape=maxshape, dtype=clean_cov_data.dtype)
                        f['clean_covariance_matrices/data'][sample_idx] = clean_cov_data
                    
                    if self.config.save_autocorrelation_matrix:
                        autocorr_data = sample['autocorrelation_matrix']
                        if 'data' not in f['autocorrelation_matrices']:
                            maxshape = (None,) + autocorr_data.shape
                            f['autocorrelation_matrices'].create_dataset('data', (total_samples,) + autocorr_data.shape,
                                                                       maxshape=maxshape, dtype=autocorr_data.dtype)
                        f['autocorrelation_matrices/data'][sample_idx] = autocorr_data
                    
                    if self.config.save_music_estimates and 'music_estimates' in sample:
                        music_data = sample['music_estimates']
                        
                        # Create datasets for MUSIC data on first sample
                        if 'estimated_angles' not in f['music_estimates']:
                            # Variable-length datasets for estimated angles (different numbers of sources)
                            dt_float = h5py.special_dtype(vlen=np.float32)
                            f['music_estimates'].create_dataset('estimated_angles', (total_samples,), dtype=dt_float, maxshape=(None,))
                            f['music_estimates'].create_dataset('true_angles', (total_samples,), dtype=dt_float, maxshape=(None,))
                            
                            # Fixed-length datasets for spectrum and angle grid
                            spectrum_shape = music_data['spectrum'].shape
                            angle_grid_shape = music_data['angle_grid'].shape
                            
                            f['music_estimates'].create_dataset('spectrum', (total_samples,) + spectrum_shape, 
                                                              maxshape=(None,) + spectrum_shape, dtype=np.float32)
                            f['music_estimates'].create_dataset('angle_grid', (total_samples,) + angle_grid_shape,
                                                              maxshape=(None,) + angle_grid_shape, dtype=np.float32)
                            f['music_estimates'].create_dataset('num_sources_used', (total_samples,), dtype=np.int32, maxshape=(None,))
                        
                        # Save MUSIC data
                        f['music_estimates/estimated_angles'][sample_idx] = music_data['estimated_angles']
                        f['music_estimates/true_angles'][sample_idx] = music_data['true_angles']
                        f['music_estimates/spectrum'][sample_idx] = music_data['spectrum']
                        f['music_estimates/angle_grid'][sample_idx] = music_data['angle_grid']
                        f['music_estimates/num_sources_used'][sample_idx] = music_data['num_sources_used']
                    
                    if self.config.save_steering_vectors:
                        nominal_steering = sample['steering_vectors']['nominal']
                        actual_steering = sample['steering_vectors']['actual']
                        
                        if 'data' not in f['steering_vectors/nominal']:
                            # Use variable-length datasets for steering vectors since they have different numbers of sources
                            dt_complex = h5py.special_dtype(vlen=np.complex128)
                            f['steering_vectors/nominal'].create_dataset('data', (total_samples,), dtype=dt_complex, maxshape=(None,))
                            f['steering_vectors/actual'].create_dataset('data', (total_samples,), dtype=dt_complex, maxshape=(None,))
                        
                        # Flatten the steering matrices for storage
                        f['steering_vectors/nominal/data'][sample_idx] = nominal_steering.flatten()
                        f['steering_vectors/actual/data'][sample_idx] = actual_steering.flatten()
                        
                        # Store shape information for reconstruction
                        if 'shapes' not in f['steering_vectors']:
                            f['steering_vectors'].create_dataset('shapes', (total_samples, 2), dtype=np.int32, maxshape=(None, 2))
                        f['steering_vectors/shapes'][sample_idx] = nominal_steering.shape
                    
                    if self.config.save_array_metadata and 'array_model' in sample:
                        array_model_data = sample['array_model']
                        
                        # Create datasets for array model metadata on first sample
                        if 'array_type' not in f['array_models']:
                            # Array configuration parameters
                            f['array_models'].create_dataset('array_type', (total_samples,), dtype=h5py.string_dtype(), maxshape=(None,))
                            f['array_models'].create_dataset('num_elements', (total_samples,), dtype=np.int32, maxshape=(None,))
                            f['array_models'].create_dataset('carrier_freq', (total_samples,), dtype=np.float64, maxshape=(None,))
                            f['array_models'].create_dataset('enable_gain_phase_errors', (total_samples,), dtype=bool, maxshape=(None,))
                            f['array_models'].create_dataset('enable_mutual_coupling', (total_samples,), dtype=bool, maxshape=(None,))
                            f['array_models'].create_dataset('position_error_std', (total_samples,), dtype=np.float64, maxshape=(None,))
                            f['array_models'].create_dataset('element_spacing', (total_samples,), dtype=np.float64, maxshape=(None,))
                            f['array_models'].create_dataset('radius', (total_samples,), dtype=np.float64, maxshape=(None,))
                            # Store gain_error_range as two separate values
                            f['array_models'].create_dataset('gain_error_range_min', (total_samples,), dtype=np.float64, maxshape=(None,))
                            f['array_models'].create_dataset('gain_error_range_max', (total_samples,), dtype=np.float64, maxshape=(None,))
                            # Store phase_error_range_deg as two separate values
                            f['array_models'].create_dataset('phase_error_range_deg_min', (total_samples,), dtype=np.float64, maxshape=(None,))
                            f['array_models'].create_dataset('phase_error_range_deg_max', (total_samples,), dtype=np.float64, maxshape=(None,))
                            # Mutual coupling parameters
                            f['array_models'].create_dataset('coupling_strength', (total_samples,), dtype=np.float64, maxshape=(None,))
                            f['array_models'].create_dataset('coupling_phase_deg', (total_samples,), dtype=np.float64, maxshape=(None,))
                            f['array_models'].create_dataset('coupling_variation', (total_samples,), dtype=np.float64, maxshape=(None,))
                            # Deterministic seed for recreation
                            f['array_models'].create_dataset('seed', (total_samples,), dtype=np.int64, maxshape=(None,))
                            f['array_models'].create_dataset('array_imperfections', (total_samples,), dtype=bool, maxshape=(None,))
                        
                        # Save array model data
                        array_config = array_model_data['array_config']
                        f['array_models/array_type'][sample_idx] = array_config['array_type']
                        f['array_models/num_elements'][sample_idx] = array_config['num_elements']
                        f['array_models/carrier_freq'][sample_idx] = array_config['carrier_freq']
                        f['array_models/enable_gain_phase_errors'][sample_idx] = array_config['enable_gain_phase_errors']
                        f['array_models/enable_mutual_coupling'][sample_idx] = array_config['enable_mutual_coupling']
                        f['array_models/position_error_std'][sample_idx] = array_config['position_error_std']
                        f['array_models/element_spacing'][sample_idx] = array_config['element_spacing']
                        f['array_models/radius'][sample_idx] = array_config['radius']
                        # Save gain_error_range as two values
                        f['array_models/gain_error_range_min'][sample_idx] = array_config['gain_error_range'][0]
                        f['array_models/gain_error_range_max'][sample_idx] = array_config['gain_error_range'][1]
                        # Save phase_error_range_deg as two values
                        f['array_models/phase_error_range_deg_min'][sample_idx] = array_config['phase_error_range_deg'][0]
                        f['array_models/phase_error_range_deg_max'][sample_idx] = array_config['phase_error_range_deg'][1]
                        # Save mutual coupling parameters
                        f['array_models/coupling_strength'][sample_idx] = array_config['coupling_strength']
                        f['array_models/coupling_phase_deg'][sample_idx] = array_config['coupling_phase_deg']
                        f['array_models/coupling_variation'][sample_idx] = array_config['coupling_variation']
                        f['array_models/seed'][sample_idx] = array_model_data['seed']
                        f['array_models/array_imperfections'][sample_idx] = array_model_data['array_imperfections']
                    
                    sample_idx += 1
            
            # Save configuration as metadata
            f.attrs['config'] = json.dumps(self.config.__dict__, default=str)
            f.attrs['total_samples'] = total_samples
            f.attrs['total_combinations'] = len(param_combinations)
            f.attrs['examples_per_combination'] = self.config.examples_per_combination
        
        # Save human-readable info
        info_file = dataset_dir / f"{self.config.dataset_name}_info.json"
        with open(info_file, 'w') as f:
            json.dump({
                'dataset_name': self.config.dataset_name,
                'array_type': self.config.array_type,
                'num_elements': self.config.num_elements,
                'total_samples': total_samples,
                'total_combinations': len(param_combinations),
                'examples_per_combination': self.config.examples_per_combination,
                'parameter_ranges': {
                    'angles_deg': self.config.angles_deg,
                    'snr_db': self.config.snr_db,
                    'num_snapshots': self.config.num_snapshots,
                    'num_sources': self.config.num_sources,
                    'sir_db': self.config.sir_db,
                    'num_multipath': self.config.num_multipath,
                    'array_imperfections': self.config.array_imperfections
                },
                'array_configuration': {
                    'array_type': self.config.array_type,
                    'num_elements': self.config.num_elements,
                    'carrier_freq': self.config.carrier_freq,
                    'sampling_freq': self.config.sampling_freq
                },
                'data_components': {
                    'received_signal': self.config.save_received_signal,
                    'covariance_matrix': self.config.save_covariance_matrix,
                    'clean_covariance_matrix': self.config.save_clean_covariance_matrix,
                    'autocorrelation_matrix': self.config.save_autocorrelation_matrix,
                    'music_estimates': self.config.save_music_estimates,
                    'steering_vectors': self.config.save_steering_vectors,
                    'array_metadata': self.config.save_array_metadata
                },
                'autocorrelation_config': {
                    'tau': self.config.autocorr_tau,
                    'description': 'Autocorrelation matrices computed from received signals with configurable time lags',
                    'format': '[tau, 2M, M] where real and imaginary parts are stacked'
                },
                'music_config': {
                    'enabled': self.config.save_music_estimates,
                    'angle_grid_deg': self.config.music_angle_grid_deg if self.config.save_music_estimates else None,
                    'num_sources': self.config.music_num_sources if self.config.save_music_estimates else None,
                    'description': 'MUSIC DOA estimates and spectrum computed from clean covariance matrices',
                    'components': {
                        'estimated_angles': 'DOA estimates in degrees (variable length per sample)',
                        'true_angles': 'Ground truth DOA angles in degrees (variable length per sample)',
                        'spectrum': 'MUSIC spatial spectrum values',
                        'angle_grid': 'Angle grid used for spectrum computation',
                        'num_sources_used': 'Number of sources used in MUSIC algorithm'
                    }
                },
                'computed_labels': {
                    'smr': 'Signal-to-Multipath Ratio in dB (999.0 when no multipath)',
                    'sir': 'Signal-to-Interference Ratio in dB (-999 when no interference)',
                    'doas': 'Direction of Arrival angles in degrees for all sources'
                },
                'clean_covariance_matrix_info': {
                    'description': 'Covariance matrix computed from clean signals (main + interference) using perfect array response',
                    'includes': ['Main sources', 'Interference sources', 'Perfect steering vectors (no array imperfections)'],
                    'excludes': ['Multipath reflections', 'Additive noise', 'Array gain/phase errors', 'Mutual coupling'],
                    'use_cases': ['Algorithm development', 'Theoretical performance bounds', 'Ground truth for training', 'Signal structure analysis']
                }
            }, f, indent=2)
        
        print(f"Dataset generated successfully!")
        print(f"Dataset file: {dataset_file}")
        print(f"Info file: {info_file}")
        
        # Show MUSIC cache statistics if enabled
        if self.config.save_music_estimates:
            cache_stats = self.get_music_cache_stats()
            print(f"\n🎵 MUSIC Cache Statistics:")
            print(f"   - Cache enabled: {cache_stats['enabled']}")
            if cache_stats['enabled']:
                print(f"   - Unique scenarios cached: {cache_stats['cache_size']}")
                print(f"   - Estimated speedup: {cache_stats['estimated_speedup']}")
        
        return str(dataset_file)
    
    def generate_test_dataset(self, 
                            reference_angles: Optional[Union[List[float], str]] = 'random',
                            angle_delta: Optional[float] = None,
                            angle_range: Tuple[float, float] = (30.0, 150.0),
                            snr_db: List[float] = None,
                            samples_per_snr: int = 100,
                            num_sources: List[int] = [1, 0],
                            num_snapshots: int = 512,
                            num_multipath: int = 0,
                            array_imperfections: bool = False,
                            sir_db: Optional[float] = 0.0,
                            output_dir: Union[str, Path] = "Data/datasets") -> str:
        """
        Generate test dataset with controlled angle configurations for evaluation.
        
        Three operational modes:
        1. Fixed reference + fixed delta: All samples have same angle configuration
        2. Random reference + fixed delta: Each sample has different reference, same delta
        3. Random angles + minimum separation: Like current behavior (when angle_delta=None)
        
        Args:
            reference_angles: 
                - 'random': Each sample has random reference angle (Mode 2)
                - List of floats: Use these reference angles (Mode 1)
                - None: Use random angles with min separation (Mode 3, requires angle_delta=None)
            angle_delta: 
                - Float: Minimum angular separation between sources (degrees)
                - None: Use random angles with min_angle_separation_deg from config (Mode 3)
            angle_range: (min_angle, max_angle) boundary for angle generation
            snr_db: List of SNR values in dB
            samples_per_snr: Number of samples to generate for EACH SNR level
            num_sources: [main_sources, interference_sources] (e.g., [1, 2] = 1 main + 2 interference)
            num_snapshots: Number of time snapshots
            num_multipath: Number of multipath components (0 = no multipath)
            array_imperfections: Enable array imperfections (gain/phase errors, coupling)
            sir_db: Signal-to-Interference Ratio in dB (only used if interference_sources > 0)
            output_dir: Output directory for dataset
            
        Returns:
            Path to generated dataset file
            
        Examples:
            # Mode 1: Fixed reference angles with fixed delta (all samples identical)
            generate_test_dataset(
                reference_angles=[90.0, 45.0],  # Two scenarios
                angle_delta=10.0,               # 10° separation
                snr_db=[-20, -10, 0, 10, 20],
                samples_per_snr=100,
                num_sources=[1, 2]              # 1 main + 2 interference
            )
            # Generates: For ref=90°: sources at [90°, 100°, 110°]
            #           For ref=45°: sources at [45°, 55°, 65°]
            
            # Mode 2: Random reference angles with fixed delta (different each sample)
            generate_test_dataset(
                reference_angles='random',       # Random reference for each sample
                angle_delta=10.0,                # 10° separation maintained
                snr_db=[-20, -10, 0, 10, 20],
                samples_per_snr=100,
                num_sources=[1, 2]               # 1 main + 2 interference
            )
            # Generates: 100 samples per SNR, each with different reference but same 10° delta
            
            # Mode 3: Random angles with minimum separation (current behavior)
            generate_test_dataset(
                reference_angles=None,           # Random angles
                angle_delta=None,                # Use config's min_angle_separation_deg
                snr_db=[-20, -10, 0, 10, 20],
                samples_per_snr=100,
                num_sources=[1, 2]
            )
            # Generates: Fully random angles with minimum separation constraint
        """
        print("🚀 Generating Test/Evaluation Dataset...")
        print("=" * 60)
        
        # Determine operational mode
        if angle_delta is None and reference_angles is None:
            mode = 3  # Random angles with minimum separation
            mode_name = "Random angles with minimum separation"
        elif reference_angles == 'shared_random' and angle_delta is None:
            mode = 4  # Random angles shared across SNR levels
            mode_name = "Random angles with minimum separation (shared across SNR)"
        elif reference_angles == 'random' and angle_delta is not None:
            mode = 2  # Random reference + fixed delta
            mode_name = f"Random reference with fixed {angle_delta}° separation"
        elif isinstance(reference_angles, (list, np.ndarray)) and angle_delta is not None:
            mode = 1  # Fixed reference + fixed delta
            mode_name = f"Fixed reference with fixed {angle_delta}° separation"
        else:
            raise ValueError(
                "Invalid parameter combination. Use one of:\n"
                "  Mode 1: reference_angles=[...], angle_delta=X\n"
                "  Mode 2: reference_angles='random', angle_delta=X\n"
                "  Mode 3: reference_angles=None, angle_delta=None\n"
                "  Mode 4: reference_angles='shared_random', angle_delta=None"
            )
        
        print(f"📍 Mode: {mode_name}")
        print(f"   SNR levels: {snr_db}")
        print(f"   Samples per SNR: {samples_per_snr}")
        print(f"   Total samples: {len(snr_db) * samples_per_snr}")
        print(f"   Sources: {num_sources[0]} main + {num_sources[1]} interference")
        print(f"   Angle range: {angle_range}")
        
        # Store original config parameters
        original_config = {
            'angles_deg': self.config.angles_deg,
            'snr_db': self.config.snr_db,
            'num_snapshots': self.config.num_snapshots,
            'num_sources': self.config.num_sources,
            'num_multipath': self.config.num_multipath,
            'array_imperfections': self.config.array_imperfections,
            'sir_db': self.config.sir_db,
            'examples_per_combination': self.config.examples_per_combination,
            'dataset_name': self.config.dataset_name
        }
        
        # Update config for test dataset generation
        self.config.snr_db = snr_db
        self.config.num_snapshots = [num_snapshots]
        self.config.num_sources = [num_sources]
        self.config.num_multipath = [num_multipath]
        self.config.array_imperfections = [array_imperfections]
        self.config.sir_db = [sir_db] if num_sources[1] > 0 else [0]
        self.config.examples_per_combination = 1  # We'll handle replication ourselves
        
        # Update dataset name to include mode information
        mode_suffix = f"_mode{mode}_delta{angle_delta}" if angle_delta else f"_mode{mode}_random"
        self.config.dataset_name = f"{original_config['dataset_name']}_test{mode_suffix}"
        
        # Build parameter combinations based on mode
        total_sources = num_sources[0] + num_sources[1]
        
        if mode == 1:
            # Mode 1: Fixed reference angles
            print(f"\n🎯 Generating with fixed reference angles: {reference_angles}")
            param_combinations = []
            
            for ref_angle in reference_angles:
                # Generate angle configuration with sequential placement and boundary checking
                angles = self._generate_sequential_angles_with_boundary(
                    ref_angle, total_sources, angle_delta, angle_range
                )
                print(f"   Reference {ref_angle}° → Sources at: {angles}")
                
                for snr_val in snr_db:
                    for sample_idx in range(samples_per_snr):
                        param_combinations.append({
                            'angle_deg': angles[0],  # Main source angle
                            'all_angles': angles,    # All source angles
                            'snr': snr_val,
                            'num_snapshots': num_snapshots,
                            'num_sources': num_sources,
                            'sir_db': sir_db if num_sources[1] > 0 else None,
                            'num_multipath': num_multipath,
                            'array_imperfections': array_imperfections,
                            'sample_idx': sample_idx,
                            'reference_angle': ref_angle,
                            'delta': angle_delta
                        })
        
        elif mode == 2:
            # Mode 2: Random reference + fixed delta
            print(f"\n🎯 Generating with random reference angles, fixed delta: {angle_delta}°")
            param_combinations = []
            
            for snr_val in snr_db:
                for sample_idx in range(samples_per_snr):
                    # Generate random reference angle (as integer)
                    ref_angle = float(self.rng.integers(int(angle_range[0]), int(angle_range[1]) + 1))
                    
                    # Generate angle configuration with sequential placement and boundary checking
                    angles = self._generate_sequential_angles_with_boundary(
                        ref_angle, total_sources, angle_delta, angle_range
                    )
                    
                    param_combinations.append({
                        'angle_deg': angles[0],  # Main source angle
                        'all_angles': angles,    # All source angles
                        'snr': snr_val,
                        'num_snapshots': num_snapshots,
                        'num_sources': num_sources,
                        'sir_db': sir_db if num_sources[1] > 0 else None,
                        'num_multipath': num_multipath,
                        'array_imperfections': array_imperfections,
                        'sample_idx': sample_idx,
                        'reference_angle': ref_angle,
                        'delta': angle_delta
                    })
        
        elif mode == 3:
            # Mode 3: Random angles with minimum separation (current behavior)
            print(f"\n🎯 Generating with random angles, minimum separation: {self.config.min_angle_separation_deg}°")
            param_combinations = []
            
            for snr_val in snr_db:
                for sample_idx in range(samples_per_snr):
                    # Generate random separated angles (as integer)
                    main_angle = float(self.rng.integers(int(angle_range[0]), int(angle_range[1]) + 1))
                    main_angle = float(self.rng.uniform(int(angle_range[0]), int(angle_range[1]) + 1))
                    additional_angles = []
                    
                    if total_sources > 1:
                        additional_angles = self._generate_separated_angles(
                            [main_angle],
                            total_sources - 1,
                            angle_range,
                            self.config.min_angle_separation_deg
                        )
                    
                    angles = [main_angle] + additional_angles
                    
                    param_combinations.append({
                        'angle_deg': angles[0],  # Main source angle
                        'all_angles': angles,    # All source angles
                        'snr': snr_val,
                        'num_snapshots': num_snapshots,
                        'num_sources': num_sources,
                        'sir_db': sir_db if num_sources[1] > 0 else None,
                        'num_multipath': num_multipath,
                        'array_imperfections': array_imperfections,
                        'sample_idx': sample_idx,
                        'reference_angle': angles[0],
                        'delta': None
                    })
        
        else:  # mode == 4
            # Mode 4: Random angles with minimum separation, SHARED across all SNR levels
            print(f"\n🎯 Generating with random angles (shared across SNR), minimum separation: {self.config.min_angle_separation_deg}°")
            print(f"   ℹ️  Same {samples_per_snr} angle configurations will be used for all {len(snr_db)} SNR levels")
            param_combinations = []
            
            # Pre-generate angle configurations (once for all SNR levels)
            # This includes BOTH source angles AND multipath angles
            angle_configs = []
            multipath_angle_configs = []  # Store multipath angles separately
            
            for sample_idx in range(samples_per_snr):
                # Generate random separated angles for sources
                main_angle = float(self.rng.uniform(angle_range[0], angle_range[1]))
                additional_angles = []
                
                if total_sources > 1:
                    additional_angles = self._generate_separated_angles(
                        [main_angle],
                        total_sources - 1,
                        angle_range,
                        self.config.min_angle_separation_deg
                    )
                
                angles = [main_angle] + additional_angles
                angle_configs.append(angles)
                
                # Pre-generate multipath angles (random, but consistent across SNRs)
                if num_multipath > 0:
                    multipath_angles = []
                    for _ in range(num_multipath):
                        # Random angle in valid range (0-360 degrees for full coverage)
                        mp_angle = float(self.rng.uniform(0, 360))
                        multipath_angles.append(mp_angle)
                    multipath_angle_configs.append(multipath_angles)
                else:
                    multipath_angle_configs.append([])
                
                if sample_idx < 5:  # Print first 5 configurations
                    if num_multipath > 0:
                        print(f"   Sample {sample_idx+1}: {[f'{a:.1f}°' for a in angles]} + MP: {[f'{m:.1f}°' for m in multipath_angles]}")
                    else:
                        print(f"   Sample {sample_idx+1}: {[f'{a:.1f}°' for a in angles]}")
            
            if samples_per_snr > 5:
                print(f"   ... ({samples_per_snr - 5} more configurations)")
            
            # Now replicate these angle configurations across all SNR levels
            for snr_val in snr_db:
                for sample_idx, (angles, mp_angles) in enumerate(zip(angle_configs, multipath_angle_configs)):
                    param_combinations.append({
                        'angle_deg': angles[0],  # Main source angle
                        'all_angles': angles,    # All source angles
                        'multipath_angles': mp_angles,  # Pre-generated multipath angles
                        'snr': snr_val,
                        'num_snapshots': num_snapshots,
                        'num_sources': num_sources,
                        'sir_db': sir_db if num_sources[1] > 0 else None,
                        'num_multipath': num_multipath,
                        'array_imperfections': array_imperfections,
                        'sample_idx': sample_idx,
                        'reference_angle': angles[0],
                        'delta': None
                    })
        
        # Generate dataset using custom parameter combinations
        dataset_path = self._generate_dataset_from_combinations(param_combinations, output_dir, mode)
        
        # Restore original config
        self.config.angles_deg = original_config['angles_deg']
        self.config.snr_db = original_config['snr_db']
        self.config.num_snapshots = original_config['num_snapshots']
        self.config.num_sources = original_config['num_sources']
        self.config.num_multipath = original_config['num_multipath']
        self.config.array_imperfections = original_config['array_imperfections']
        self.config.sir_db = original_config['sir_db']
        self.config.examples_per_combination = original_config['examples_per_combination']
        self.config.dataset_name = original_config['dataset_name']
        
        print(f"\n✅ Test dataset generated successfully!")
        return dataset_path
    
    def _generate_sequential_angles_with_boundary(self, reference_angle: float, 
                                                  total_sources: int, 
                                                  delta: float, 
                                                  angle_range: Tuple[float, float]) -> List[float]:
        """
        Generate angles sequentially from reference with boundary checking.
        
        Places sources at: reference, reference+delta, reference+2*delta, ...
        If boundaries exceeded, adjusts to keep within range.
        
        Args:
            reference_angle: Reference angle in degrees
            total_sources: Total number of sources to place
            delta: Angular separation in degrees
            angle_range: (min_angle, max_angle) boundary
            
        Returns:
            List of angles within boundaries
            
        Example:
            reference=140°, total_sources=3, delta=10°, range=(30, 150)
            Normal: [140, 150, 160] → 160 exceeds!
            Adjusted: [130, 140, 150] → All within range
        """
        angle_min, angle_max = angle_range
        angles = [reference_angle]
        
        # Try sequential placement forward
        current_angle = reference_angle
        for i in range(1, total_sources):
            next_angle = current_angle + delta
            if next_angle <= angle_max:
                angles.append(next_angle)
                current_angle = next_angle
            else:
                # Hit upper boundary, start filling backwards from reference
                break
        
        # If we don't have enough angles, add backwards from reference
        while len(angles) < total_sources:
            backward_angle = angles[0] - delta
            if backward_angle >= angle_min:
                angles.insert(0, backward_angle)
            else:
                # Can't fit all sources with this delta!
                raise ValueError(
                    f"Cannot fit {total_sources} sources with {delta}° separation "
                    f"starting at {reference_angle}° within range {angle_range}. "
                    f"Try smaller delta or different reference angle."
                )
        
        return sorted(angles)
    
    def _generate_dataset_from_combinations(self, param_combinations: List[Dict], 
                                           output_dir: Union[str, Path],
                                           mode: int) -> str:
        """Generate dataset from pre-built parameter combinations."""
        output_dir = Path(output_dir)
        
        # If relative path, make it relative to project root
        if not output_dir.is_absolute():
            current_path = Path(__file__).resolve()
            project_root = current_path
            while project_root.parent != project_root:
                if (project_root / 'src').exists():
                    break
                project_root = project_root.parent
            output_dir = project_root / output_dir
        
        # Create array-type-specific folder structure
        array_type_dir = output_dir / self.config.array_type
        dataset_dir = array_type_dir / self.config.dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        dataset_file = dataset_dir / f"{self.config.dataset_name}.h5"
        total_samples = len(param_combinations)
        
        print(f"\nGenerating {total_samples} samples...")
        print(f"Output file: {dataset_file}")
        
        with h5py.File(dataset_file, 'w') as f:
            # Create groups for different data types
            labels_group = f.create_group('labels')
            
            if self.config.save_received_signal:
                f.create_group('received_signals')
            if self.config.save_covariance_matrix:
                f.create_group('covariance_matrices')
            if self.config.save_clean_covariance_matrix:
                f.create_group('clean_covariance_matrices')
            if self.config.save_autocorrelation_matrix:
                f.create_group('autocorrelation_matrices')
            if self.config.save_music_estimates:
                f.create_group('music_estimates')
            if self.config.save_steering_vectors:
                steering_group = f.create_group('steering_vectors')
                steering_group.create_group('nominal')
                steering_group.create_group('actual')
            if self.config.save_array_metadata:
                f.create_group('array_models')
            
            # Generate samples with custom angle handling
            for sample_idx, param_combo in enumerate(tqdm(param_combinations, desc="Generating samples")):
                # Generate sample with custom angles
                sample = self._generate_sample_with_custom_angles(param_combo)
                
                # Save sample (same logic as generate_dataset)
                self._save_sample_to_hdf5(f, sample, sample_idx, total_samples)
            
            # Save configuration as metadata
            f.attrs['config'] = json.dumps(self.config.__dict__, default=str)
            f.attrs['total_samples'] = total_samples
            f.attrs['generation_mode'] = mode
            f.attrs['mode_description'] = {
                1: 'Fixed reference angles with fixed delta',
                2: 'Random reference angles with fixed delta',
                3: 'Random angles with minimum separation',
                4: 'Random angles with minimum separation (shared across SNR)'
            }[mode]
        
        # Save human-readable info
        info_file = dataset_dir / f"{self.config.dataset_name}_info.json"
        with open(info_file, 'w') as f:
            json.dump({
                'dataset_name': self.config.dataset_name,
                'generation_mode': mode,
                'mode_description': {
                    1: 'Fixed reference angles with fixed delta',
                    2: 'Random reference angles with fixed delta',
                    3: 'Random angles with minimum separation',
                    4: 'Random angles with minimum separation (shared across SNR)'
                }[mode],
                'total_samples': total_samples,
                'array_type': self.config.array_type,
                'num_elements': self.config.num_elements,
            }, f, indent=2)
        
        return str(dataset_file)
    
    def _generate_sample_with_custom_angles(self, param_combo: Dict) -> Dict:
        """Generate sample with custom angle configuration."""
        # Extract angles from param_combo
        all_angles = param_combo['all_angles']
        multipath_angles = param_combo.get('multipath_angles', None)  # Pre-generated multipath angles (Mode 4)
        
        # Create modified param_combo for _generate_sample
        modified_combo = {
            'angle_deg': all_angles[0],  # Main source angle (for compatibility)
            'snr': param_combo['snr'],
            'num_snapshots': param_combo['num_snapshots'],
            'num_sources': param_combo['num_sources'],
            'sir_db': param_combo['sir_db'],
            'num_multipath': param_combo['num_multipath'],
            'array_imperfections': param_combo['array_imperfections'],
            'custom_angles': all_angles,  # Custom angles for all sources
            'custom_multipath_angles': multipath_angles  # Pre-generated multipath angles (None if not Mode 4)
        }
        
        # Use existing _generate_sample but override angle generation
        return self._generate_sample_custom(modified_combo, param_combo['sample_idx'])
    
    def _generate_sample_custom(self, param_combo: Dict, example_idx: int) -> Dict:
        """Modified version of _generate_sample that uses custom angles."""
        # Use custom angles instead of generating them
        custom_angles = param_combo.pop('custom_angles')
        
        # Create signal config with custom angles
        signal_config = self._create_signal_config_custom(param_combo, custom_angles)
        
        # Create deterministic seed (same approach as original _generate_sample)
        seed_tuple = (
            self.generator_seed if self.generator_seed is not None else 0,
            tuple(custom_angles),  # Use all custom angles for seed
            param_combo['snr'],
            param_combo['num_snapshots'],
            param_combo['num_sources'][0],
            param_combo['num_sources'][1],
            -999 if param_combo['sir_db'] is None else param_combo['sir_db'],
            param_combo['num_multipath'],
            example_idx,
        )
        
        seed_str = "_".join(map(str, seed_tuple))
        base_seed = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
        
        # Reset RNGs
        self.rng = np.random.default_rng(base_seed)
        np.random.seed(base_seed)
        
        # Create array and signal generator
        array_config = self._create_array_config(param_combo['array_imperfections'])
        array_model = ArrayModel(array_config, seed=base_seed)
        signal_generator = SignalGenerator(signal_config, seed=base_seed)
        
        # Generate signals
        t, signals = signal_generator.generate_signals()
        
        # Compute clean covariance matrix (before multipath and noise)
        clean_covariance_matrix = None
        if self.config.save_clean_covariance_matrix or self.config.save_music_estimates:
            perfect_array_config = self._create_array_config(use_imperfections=False)
            perfect_array_model = ArrayModel(perfect_array_config, seed=base_seed)
            perfect_steering_matrix = perfect_array_model.steering_matrix(custom_angles, nominal=True)
            clean_received_signals = perfect_steering_matrix @ signals
            clean_covariance_matrix = (clean_received_signals @ clean_received_signals.conj().T) / clean_received_signals.shape[1]
        
        # Apply multipath if enabled
        if signal_config.enable_multipath:
            # Get custom multipath angles if provided (Mode 4)
            custom_mp_angles = param_combo.get('custom_multipath_angles', None)
            
            multipath_result = signal_generator.add_multipath(
                signals, t, signal_config.fs,
                array_config.carrier_freq,
                signal_config.bandwidth if hasattr(signal_config, 'bandwidth') else signal_config.fs * 0.1,
                max_paths=self.config.num_elements - 1,
                num_multipath_components=param_combo['num_multipath'],
                custom_multipath_angles=custom_mp_angles  # Pass pre-generated angles (Mode 4) or None
            )
            if isinstance(multipath_result, tuple):
                signals, shift_samples = multipath_result
            else:
                signals = multipath_result
        
        # Prepare SIR and SMR values
        if signal_config.sir_db is not None:
            sir_values = np.array(signal_config.sir_db, dtype=np.float32)
        else:
            sir_values = np.array([-999.0], dtype=np.float32)
        
        if signal_config.smr is not None:
            smr_values = np.array(signal_config.smr, dtype=np.float32)
        else:
            smr_values = np.array([999.0], dtype=np.float32)
        
        # Get steering vectors for all sources (including multipath angles added by add_multipath)
        # IMPORTANT: Use signal_config.angles (updated by add_multipath) instead of custom_angles
        if param_combo['array_imperfections']:
            steering_matrix = array_model.steering_matrix(signal_config.angles, nominal=False)
        else:
            steering_matrix = array_model.steering_matrix(signal_config.angles, nominal=True)
        
        # Create received signal
        received_signal = ReceivedSignal(steering_matrix, signals, signal_config.angles, signal_config)
        
        # Add noise
        received_signal_with_noise = received_signal.add_noise(
            snr_db=param_combo['snr'],
            source_frequencies=signal_config.frequencies,
            signal_bandwidth=signal_config.bandwidth if hasattr(signal_config, 'bandwidth') else signal_config.fs * 0.1,
            sampling_frequency=signal_config.fs,
            frequency_selective=True
        )
        
        # Create sample dict
        # IMPORTANT: Use signal_config.angles (includes multipath angles added by add_multipath)
        # instead of custom_angles (which only has main/interference source angles)
        sample = {
            'labels': {
                'doas': np.array(signal_config.angles, dtype=np.float32),
                'snr': np.float32(param_combo['snr']),
                'num_snapshots': np.int32(param_combo['num_snapshots']),
                'num_sources': np.array(param_combo['num_sources'], dtype=np.int32),
                'sir': sir_values,
                'num_multipath': np.int32(param_combo['num_multipath']),
                'smr': smr_values,
                'array_imperfections': bool(param_combo['array_imperfections'])
            }
        }
        
        # Add data components
        if self.config.save_received_signal:
            sample['received_signal'] = received_signal_with_noise.array_signals.astype(np.complex64)
        
        if self.config.save_covariance_matrix:
            signals_matrix = received_signal_with_noise.array_signals
            cov_matrix = (signals_matrix @ signals_matrix.conj().T) / signals_matrix.shape[1]
            sample['covariance_matrix'] = cov_matrix.astype(np.complex64)
        
        if self.config.save_clean_covariance_matrix:
            sample['clean_covariance_matrix'] = clean_covariance_matrix.astype(np.complex64)
        
        if self.config.save_autocorrelation_matrix:
            autocorr_matrix = self._compute_autocorrelation_matrix(
                received_signal_with_noise.array_signals,
                self.config.autocorr_tau
            )
            sample['autocorrelation_matrix'] = autocorr_matrix.astype(np.float32)
        
        if self.config.save_music_estimates and clean_covariance_matrix is not None:
            perfect_array_config = self._create_array_config(use_imperfections=False)
            perfect_array_model = ArrayModel(perfect_array_config, seed=base_seed)
            music_estimates = self._compute_music_estimates(
                clean_covariance_matrix,
                perfect_array_model,
                signal_config,
                param_combo,
                example_idx
            )
            sample['music_estimates'] = music_estimates
        
        if self.config.save_steering_vectors:
            # Save steering vectors for main/interference sources only (not multipath)
            nominal_steering = array_model.steering_matrix(custom_angles, nominal=True)
            if param_combo['array_imperfections']:
                actual_steering = array_model.steering_matrix(custom_angles, nominal=False)
            else:
                actual_steering = array_model.steering_matrix(custom_angles, nominal=True)
            sample['steering_vectors'] = {
                'nominal': nominal_steering.astype(np.complex64),
                'actual': actual_steering.astype(np.complex64)
            }
        
        if self.config.save_array_metadata:
            sample['array_model'] = {
                'array_config': {
                    'array_type': array_config.array_type,
                    'num_elements': array_config.num_elements,
                    'carrier_freq': array_config.carrier_freq,
                    'enable_gain_phase_errors': array_config.enable_gain_phase_errors,
                    'enable_mutual_coupling': array_config.enable_mutual_coupling,
                    'position_error_std': array_config.position_error_std,
                    'element_spacing': getattr(array_config, 'element_spacing', 0.5),
                    'radius': getattr(array_config, 'radius', 0.5),
                    'gain_error_range': getattr(array_config, 'gain_error_range', (0.975, 1.025)),
                    'phase_error_range_deg': getattr(array_config, 'phase_error_range_deg', (-5.0, 5.0)),
                    'coupling_strength': getattr(array_config, 'coupling_strength', 0.3),
                    'coupling_phase_deg': getattr(array_config, 'coupling_phase_deg', -100.0),
                    'coupling_variation': getattr(array_config, 'coupling_variation', 0.9)
                },
                'seed': base_seed,
                'array_imperfections': param_combo['array_imperfections']
            }
        
        return sample
    
    def _create_signal_config_custom(self, param_combo: Dict, custom_angles: List[float]) -> SignalConfig:
        """Create signal configuration with custom angles."""
        num_main = param_combo['num_sources'][0]
        num_interference = param_combo['num_sources'][1]
        total_sources = num_main + num_interference
        
        # Use custom angles directly
        source_angles = custom_angles[:total_sources]
        
        # Create frequencies
        base_freq = 100e3
        frequencies = [base_freq] * num_main
        
        if num_interference > 0:
            interference_freqs = [base_freq + (i+1) * 50e3 for i in range(num_interference)]
            frequencies.extend(interference_freqs)
        
        # Create powers based on SIR
        powers = [0.0] * total_sources
        main_source_power = 0.0
        
        sir_db_list = None
        if total_sources > 1:
            sir_value = param_combo['sir_db'] if param_combo['sir_db'] is not None else 0.0
            sir_db_list = [sir_value] * (total_sources - 1)
        
        # Multipath configuration
        multipath_enabled = param_combo['num_multipath'] > 0
        multipath_config = None
        if multipath_enabled:
            num_paths = param_combo['num_multipath']
            path_delays = [i * 1e-6 for i in range(1, num_paths + 1)]
            path_gains = [0.3 / (i + 1) for i in range(num_paths)]
            multipath_config = {
                'path_delays': path_delays,
                'path_gains': path_gains
            }
        
        return SignalConfig(
            fs=self.config.sampling_freq,
            T=param_combo['num_snapshots'],
            num_sources=total_sources,
            angles=source_angles,
            frequencies=frequencies,
            powers=powers,
            main_source_power=main_source_power,
            sir_db=sir_db_list,
            enable_multipath=multipath_enabled,
            num_paths=param_combo['num_multipath'],
            path_delays=multipath_config['path_delays'] if multipath_config else None,
            path_phases=multipath_config.get('path_phases') if multipath_config else None,
            path_gains=multipath_config['path_gains'] if multipath_config else None,
            noise_type='gaussian',
            use_full_bandwidth=True
        )
    
    def _save_sample_to_hdf5(self, f: h5py.File, sample: Dict, sample_idx: int, total_samples: int):
        """Save a single sample to HDF5 file (extracted from generate_dataset for reuse)."""
        labels_group = f['labels']
        
        # Save labels
        for label_name, label_value in sample['labels'].items():
            if label_name not in labels_group:
                # Create dataset on first sample
                if isinstance(label_value, np.ndarray):
                    if label_name == 'doas' or label_name == 'sir' or label_name == 'smr':
                        dt = h5py.special_dtype(vlen=np.float64)
                        labels_group.create_dataset(label_name, (total_samples,), dtype=dt, maxshape=(None,))
                    else:
                        maxshape = (None,) + label_value.shape
                        labels_group.create_dataset(label_name, (total_samples,) + label_value.shape,
                                                  maxshape=maxshape, dtype=label_value.dtype)
                else:
                    if label_name == 'doas' or label_name == 'sir' or label_name == 'smr':
                        dt = h5py.special_dtype(vlen=np.float64)
                        labels_group.create_dataset(label_name, (total_samples,), dtype=dt, maxshape=(None,))
                    else:
                        labels_group.create_dataset(label_name, (total_samples,),
                                              maxshape=(None,), dtype=type(label_value))
            
            labels_group[label_name][sample_idx] = label_value
        
        # Save data components (identical logic to generate_dataset)
        if self.config.save_received_signal and 'received_signal' in sample:
            signal_data = sample['received_signal']
            if 'data' not in f['received_signals']:
                dt_complex = h5py.special_dtype(vlen=np.complex128)
                f['received_signals'].create_dataset('data', (total_samples,), dtype=dt_complex, maxshape=(None,))
                f['received_signals'].create_dataset('shapes', (total_samples, 2), dtype=np.int32, maxshape=(None, 2))
            
            f['received_signals/data'][sample_idx] = signal_data.flatten()
            f['received_signals/shapes'][sample_idx] = signal_data.shape
        
        if self.config.save_covariance_matrix and 'covariance_matrix' in sample:
            cov_data = sample['covariance_matrix']
            if 'data' not in f['covariance_matrices']:
                maxshape = (None,) + cov_data.shape
                f['covariance_matrices'].create_dataset('data', (total_samples,) + cov_data.shape,
                                                      maxshape=maxshape, dtype=cov_data.dtype)
            f['covariance_matrices/data'][sample_idx] = cov_data
        
        if self.config.save_clean_covariance_matrix and 'clean_covariance_matrix' in sample:
            clean_cov_data = sample['clean_covariance_matrix']
            if 'data' not in f['clean_covariance_matrices']:
                maxshape = (None,) + clean_cov_data.shape
                f['clean_covariance_matrices'].create_dataset('data', (total_samples,) + clean_cov_data.shape,
                                                            maxshape=maxshape, dtype=clean_cov_data.dtype)
            f['clean_covariance_matrices/data'][sample_idx] = clean_cov_data
        
        if self.config.save_autocorrelation_matrix and 'autocorrelation_matrix' in sample:
            autocorr_data = sample['autocorrelation_matrix']
            if 'data' not in f['autocorrelation_matrices']:
                maxshape = (None,) + autocorr_data.shape
                f['autocorrelation_matrices'].create_dataset('data', (total_samples,) + autocorr_data.shape,
                                                           maxshape=maxshape, dtype=autocorr_data.dtype)
            f['autocorrelation_matrices/data'][sample_idx] = autocorr_data
        
        if self.config.save_steering_vectors and 'steering_vectors' in sample:
            nominal_steering = sample['steering_vectors']['nominal']
            actual_steering = sample['steering_vectors']['actual']
            
            if 'data' not in f['steering_vectors/nominal']:
                dt_complex = h5py.special_dtype(vlen=np.complex128)
                f['steering_vectors/nominal'].create_dataset('data', (total_samples,), dtype=dt_complex, maxshape=(None,))
                f['steering_vectors/actual'].create_dataset('data', (total_samples,), dtype=dt_complex, maxshape=(None,))
                f['steering_vectors'].create_dataset('shapes', (total_samples, 2), dtype=np.int32, maxshape=(None, 2))
            
            f['steering_vectors/nominal/data'][sample_idx] = nominal_steering.flatten()
            f['steering_vectors/actual/data'][sample_idx] = actual_steering.flatten()
            f['steering_vectors/shapes'][sample_idx] = nominal_steering.shape


# PyTorch Dataset class for easy loading and filtering
try:
    import torch
    from torch.utils.data import Dataset
    
    class DOADataset(Dataset):
        """PyTorch Dataset for DOA data with filtering capabilities."""
        
        def __init__(self, dataset_path: Union[str, Path]):
            """Load dataset from HDF5 file."""
            self.dataset_path = Path(dataset_path)
            
            with h5py.File(self.dataset_path, 'r') as f:
                # Load all labels
                self.labels = {}
                for label_name in f['labels'].keys():
                    self.labels[label_name] = f[f'labels/{label_name}'][:]
                
                # Store available data components
                self.has_received_signal = 'received_signals' in f
                self.has_covariance = 'covariance_matrices' in f
                self.has_clean_covariance = 'clean_covariance_matrices' in f
                self.has_autocorrelation = 'autocorrelation_matrices' in f
                self.has_music_estimates = 'music_estimates' in f
                self.has_steering = 'steering_vectors' in f
                self.has_array_models = 'array_models' in f
                self.has_metadata = 'array_metadata' in f  # Legacy naming
                
                self.total_samples = f.attrs['total_samples']
            
            # Initialize with all indices
            self.indices = np.arange(self.total_samples)
        
        def filter(self, **kwargs) -> 'DOADataset':
            """Filter dataset based on label criteria."""
            filtered_dataset = DOADataset.__new__(DOADataset)
            filtered_dataset.dataset_path = self.dataset_path
            filtered_dataset.labels = self.labels
            filtered_dataset.has_received_signal = self.has_received_signal
            filtered_dataset.has_covariance = self.has_covariance
            filtered_dataset.has_clean_covariance = self.has_clean_covariance
            filtered_dataset.has_autocorrelation = self.has_autocorrelation
            filtered_dataset.has_music_estimates = self.has_music_estimates
            filtered_dataset.has_steering = self.has_steering
            filtered_dataset.has_array_models = self.has_array_models
            filtered_dataset.has_metadata = self.has_metadata
            filtered_dataset.total_samples = self.total_samples
            
            # Apply filters
            mask = np.ones(len(self.indices), dtype=bool)
            
            for label_name, value in kwargs.items():
                if label_name not in self.labels:
                    raise ValueError(f"Unknown label: {label_name}")
                
                label_data = self.labels[label_name][self.indices]
                
                # Special handling for 2D arrays like num_sources
                if label_name == 'num_sources':
                    if isinstance(value, list) and len(value) == 2 and all(isinstance(v, (int, np.integer)) for v in value):
                        # Single source configuration [main, interference]
                        submask = np.all(label_data == np.array(value), axis=1)
                        mask &= submask
                    elif isinstance(value, (list, tuple)) and all(isinstance(v, (list, tuple)) for v in value):
                        # Multiple source configurations [[main1, interference1], [main2, interference2], ...]
                        submask = np.zeros(len(label_data), dtype=bool)
                        for v in value:
                            submask |= np.all(label_data == np.array(v), axis=1)
                        mask &= submask
                    else:
                        raise ValueError(f"Invalid format for num_sources filter: {value}. Expected [main, interference] or list of such pairs.")
                elif label_name == 'doas':
                    # Special handling for variable-length DOA arrays
                    if isinstance(value, (list, tuple)):
                        # Multiple DOA values to match
                        submask = np.zeros(len(label_data), dtype=bool)
                        for v in value:
                            for i, doa_array in enumerate(label_data):
                                if v in doa_array:
                                    submask[i] = True
                        mask &= submask
                    else:
                        # Single DOA value
                        submask = np.zeros(len(label_data), dtype=bool)
                        for i, doa_array in enumerate(label_data):
                            if value in doa_array:
                                submask[i] = True
                        mask &= submask
                else:
                    # Standard filtering for scalar values
                    if isinstance(value, (list, tuple)):
                        # Multiple values
                        submask = np.zeros(len(label_data), dtype=bool)
                        for v in value:
                            submask |= (label_data == v)
                        mask &= submask
                    else:
                        # Single value
                        mask &= (label_data == value)
            
            filtered_dataset.indices = self.indices[mask]
            return filtered_dataset
        
        def __len__(self):
            return len(self.indices)
        
        def __getitem__(self, idx):
            """Get sample by index."""
            actual_idx = self.indices[idx]
            
            sample = {
                'labels': {name: data[actual_idx] for name, data in self.labels.items()}
            }
            
            # Load data components as needed
            with h5py.File(self.dataset_path, 'r') as f:
                if self.has_received_signal:
                    # Reconstruct received signal from flattened data
                    shape = f['received_signals/shapes'][actual_idx]
                    signal_flat = f['received_signals/data'][actual_idx]
                    sample['received_signal'] = torch.from_numpy(signal_flat.reshape(shape))
                
                if self.has_covariance:
                    sample['covariance_matrix'] = torch.from_numpy(f['covariance_matrices/data'][actual_idx])
                
                if self.has_clean_covariance:
                    sample['clean_covariance_matrix'] = torch.from_numpy(f['clean_covariance_matrices/data'][actual_idx])
                
                if self.has_autocorrelation:
                    sample['autocorrelation_matrix'] = torch.from_numpy(f['autocorrelation_matrices/data'][actual_idx])
                
                if self.has_music_estimates:
                    # Load MUSIC estimates
                    sample['music_estimates'] = {
                        'estimated_angles': f['music_estimates/estimated_angles'][actual_idx],
                        'true_angles': f['music_estimates/true_angles'][actual_idx],
                        'spectrum': torch.from_numpy(f['music_estimates/spectrum'][actual_idx]),
                        'angle_grid': torch.from_numpy(f['music_estimates/angle_grid'][actual_idx]),
                        'num_sources_used': int(f['music_estimates/num_sources_used'][actual_idx])
                    }
                
                if self.has_steering:
                    # Reconstruct steering matrices from flattened data
                    shape = f['steering_vectors/shapes'][actual_idx]
                    nominal_flat = f['steering_vectors/nominal/data'][actual_idx]
                    actual_flat = f['steering_vectors/actual/data'][actual_idx]
                    
                    sample['steering_vectors'] = {
                        'nominal': torch.from_numpy(nominal_flat.reshape(shape)),
                        'actual': torch.from_numpy(actual_flat.reshape(shape))
                    }
                
                if self.has_array_models:
                    # Load array model metadata (with backward compatibility)
                    array_config_data = {}
                    
                    # Always present fields
                    array_config_data['array_type'] = f['array_models/array_type'][actual_idx].decode('utf-8')
                    array_config_data['num_elements'] = int(f['array_models/num_elements'][actual_idx])
                    array_config_data['carrier_freq'] = float(f['array_models/carrier_freq'][actual_idx])
                    array_config_data['enable_gain_phase_errors'] = bool(f['array_models/enable_gain_phase_errors'][actual_idx])
                    array_config_data['enable_mutual_coupling'] = bool(f['array_models/enable_mutual_coupling'][actual_idx])
                    array_config_data['position_error_std'] = float(f['array_models/position_error_std'][actual_idx])
                    
                    # Optional fields with defaults (for backward compatibility)
                    array_config_data['element_spacing'] = float(f['array_models/element_spacing'][actual_idx]) if 'element_spacing' in f['array_models'] else 0.5
                    array_config_data['radius'] = float(f['array_models/radius'][actual_idx]) if 'radius' in f['array_models'] else 0.5
                    
                    # Handle gain_error_range (tuple or default)
                    if 'gain_error_range_min' in f['array_models'] and 'gain_error_range_max' in f['array_models']:
                        array_config_data['gain_error_range'] = (
                            float(f['array_models/gain_error_range_min'][actual_idx]),
                            float(f['array_models/gain_error_range_max'][actual_idx])
                        )
                    else:
                        array_config_data['gain_error_range'] = (0.975, 1.025)
                    
                    # Handle phase_error_range_deg (tuple or default)
                    if 'phase_error_range_deg_min' in f['array_models'] and 'phase_error_range_deg_max' in f['array_models']:
                        array_config_data['phase_error_range_deg'] = (
                            float(f['array_models/phase_error_range_deg_min'][actual_idx]),
                            float(f['array_models/phase_error_range_deg_max'][actual_idx])
                        )
                    else:
                        array_config_data['phase_error_range_deg'] = (-5.0, 5.0)
                    
                    # Handle coupling parameters (with defaults)
                    array_config_data['coupling_strength'] = float(f['array_models/coupling_strength'][actual_idx]) if 'coupling_strength' in f['array_models'] else 0.3
                    array_config_data['coupling_phase_deg'] = float(f['array_models/coupling_phase_deg'][actual_idx]) if 'coupling_phase_deg' in f['array_models'] else -100.0
                    array_config_data['coupling_variation'] = float(f['array_models/coupling_variation'][actual_idx]) if 'coupling_variation' in f['array_models'] else 0.9
                    
                    # Seed and imperfections
                    seed = int(f['array_models/seed'][actual_idx]) if 'seed' in f['array_models'] else 42
                    array_imperfections = bool(f['array_models/array_imperfections'][actual_idx]) if 'array_imperfections' in f['array_models'] else True
                    
                    sample['array_model_data'] = {
                        'array_config': array_config_data,
                        'seed': seed,
                        'array_imperfections': array_imperfections
                    }
            
            return sample
        
        def get_array_model(self, idx):
            """
            Reconstruct the ArrayModel object for a specific sample.
            
            Args:
                idx: Sample index
                
            Returns:
                ArrayModel: Reconstructed array model with exact same imperfections as original
                
            Raises:
                ValueError: If array models are not available in this dataset
            """
            if not self.has_array_models:
                raise ValueError("Array models are not available in this dataset. "
                               "Dataset must be generated with save_array_metadata=True.")
            
            # Load sample to get array model data
            try:
                sample = self[idx]
            except Exception as e:
                raise ValueError(f"Error loading sample {idx}: {e}")
            
            if 'array_model_data' not in sample:
                raise ValueError(f"Array model data not found for sample {idx}. "
                               "This dataset may have been generated with an older version of the code.")
            
            # Import here to avoid circular imports
            try:
                from signalgen import ArrayConfig, ArrayModel
            except ImportError:
                import sys
                sys.path.insert(0, str(Path(__file__).parent.parent))
                from signalgen import ArrayConfig, ArrayModel
            
            # Extract array model data
            array_model_data = sample['array_model_data']
            config_data = array_model_data['array_config']
            seed = array_model_data['seed']
            
            # Create ArrayConfig with error handling
            try:
                array_config = ArrayConfig(
                    array_type=config_data['array_type'],
                    num_elements=config_data['num_elements'],
                    carrier_freq=config_data['carrier_freq'],
                    enable_gain_phase_errors=config_data['enable_gain_phase_errors'],
                    enable_mutual_coupling=config_data['enable_mutual_coupling'],
                    position_error_std=config_data['position_error_std'],
                    element_spacing=config_data['element_spacing'],
                    radius=config_data['radius'],
                    gain_error_range=config_data['gain_error_range'],
                    phase_error_range_deg=config_data['phase_error_range_deg'],
                    coupling_strength=config_data['coupling_strength'],
                    coupling_phase_deg=config_data['coupling_phase_deg'],
                    coupling_variation=config_data['coupling_variation']
                )
            except Exception as e:
                raise ValueError(f"Error creating ArrayConfig for sample {idx}: {e}. "
                               f"Config data: {config_data}")
            
            # Recreate ArrayModel with same seed for identical imperfections
            try:
                array_model = ArrayModel(array_config, seed=seed)
            except Exception as e:
                raise ValueError(f"Error creating ArrayModel for sample {idx}: {e}")
            
            return array_model
        
        def has_array_model_data(self):
            """Check if array model data is available in this dataset."""
            if not self.has_array_models:
                return False
            
            # Check if this is a complete array model dataset (with new fields)
            try:
                with h5py.File(self.dataset_path, 'r') as f:
                    has_basic_fields = all(field in f['array_models'] for field in 
                                         ['array_type', 'num_elements', 'carrier_freq', 'seed'])
                    
                    has_extended_fields = all(field in f['array_models'] for field in 
                                            ['element_spacing', 'radius', 'gain_error_range_min'])
                    
                    if has_basic_fields and not has_extended_fields:
                        print("Note: Loading array models from older dataset format. Using default values for missing parameters.")
                    
                    return has_basic_fields
            except:
                return False
        
        def get_array_model_info(self, idx):
            """
            Get array model information without reconstructing the full ArrayModel object.
            
            Args:
                idx: Sample index
                
            Returns:
                dict: Array model metadata including configuration and seed
            """
            if not self.has_array_models:
                return None
            
            sample = self[idx]
            return sample.get('array_model_data', None)
        
        def get_label_summary(self):
            """Get summary of label distributions."""
            summary = {}
            for label_name, label_data in self.labels.items():
                filtered_data = label_data[self.indices]
                if label_name == 'doas':
                    # Handle variable-length DOAs
                    all_doas = np.concatenate([doa for doa in filtered_data if len(doa) > 0])
                    summary[label_name] = f"Total DOAs: {len(all_doas)}, Range: [{all_doas.min():.1f}, {all_doas.max():.1f}]"
                elif label_name in ['num_sources']:
                    summary[label_name] = f"Unique values: {np.unique(filtered_data, axis=0).tolist()}"
                elif isinstance(filtered_data[0], (bool, np.bool_)):
                    summary[label_name] = f"True: {np.sum(filtered_data)}, False: {np.sum(~filtered_data)}"
                elif label_name in ['smr', 'sir', 'snr']:
                    """Handle SNR (scalar) and variable-length SMR/SIR arrays.
                    For SMR we use sentinel 999.0, for SIR sentinel -999.0.
                    We first flatten the variable-length arrays, remove sentinel values,
                    and then compute the range if any valid entries remain."""

                    if label_name in ['smr', 'sir']:
                        # Flatten lists/arrays of variable length into one 1-D array
                        flat_vals = []
                        for arr in filtered_data:
                            # Each entry is an ndarray (possibly length 1 sentinel)
                            if arr is None:
                                continue
                            # Ensure we can iterate (convert scalars to list)
                            flat_vals.extend(np.asarray(arr).flatten())

                        flat_vals = np.array(flat_vals, dtype=np.float32)
                        sentinel = 999.0 if label_name == 'smr' else -999.0
                        valid_data = flat_vals[flat_vals != sentinel]
                    else:  # snr (scalar per sample)
                        valid_data = filtered_data

                    if valid_data.size > 0:
                        summary[label_name] = (
                            f"Range: [{valid_data.min():.1f}, {valid_data.max():.1f}] dB, "
                            f"{label_name} exists: {len(valid_data)}")
                    else:
                        summary[label_name] = "All invalid values"
                else:
                    summary[label_name] = f"Unique values: {np.unique(filtered_data).tolist()}"
            return summary

except ImportError:
    print("PyTorch not available. DOADataset class will not be available.")


def _expand_range_specification(param_value):
    """Expand range specifications in parameter values."""
    if isinstance(param_value, dict):
        # Handle range specification: {start: 0, stop: 360, step: 30}
        if all(key in param_value for key in ['start', 'stop', 'step']):
            start = param_value['start']
            stop = param_value['stop']
            step = param_value['step']
            return list(range(start, stop, step))
        # Handle range specification: {start: 0, stop: 360, num: 12}
        elif all(key in param_value for key in ['start', 'stop', 'num']):
            start = param_value['start']
            stop = param_value['stop']
            num = param_value['num']
            return list(np.linspace(start, stop, num, endpoint=False, dtype=int))
    elif isinstance(param_value, str):
        # Handle string range specification: "0:360:30" (start:stop:step)
        if ':' in param_value:
            parts = param_value.split(':')
            if len(parts) == 3:
                start, stop, step = map(int, parts)
                return list(range(start, stop, step))
            elif len(parts) == 2:
                start, stop = map(int, parts)
                return list(range(start, stop))
    
    return param_value

def load_config_from_yaml(config_path: Union[str, Path]) -> SimpleDatasetConfig:
    """Load configuration from YAML file."""
    config_path = Path(config_path)
    
    # Add .yaml extension if not present
    if not config_path.suffix:
        config_path = config_path.with_suffix('.yaml')
    
    # If path is relative, look in configs/dataset_configs
    if not config_path.is_absolute():
        config_path = Path('configs/dataset_configs') / config_path
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        yaml_config = yaml.safe_load(f)
    
    # Expand range specifications for applicable parameters
    range_params = ['angles_deg', 'snr_db', 'sir_db']
    for param in range_params:
        if param in yaml_config:
            yaml_config[param] = _expand_range_specification(yaml_config[param])
    
    # Create SimpleDatasetConfig from YAML data
    config = SimpleDatasetConfig(
        angles_deg=yaml_config.get('angles_deg'),
        snr_db=yaml_config.get('snr_db'),
        num_snapshots=yaml_config.get('num_snapshots'),
        num_sources=yaml_config.get('num_sources'),
        sir_db=yaml_config.get('sir_db'),
        num_multipath=yaml_config.get('num_multipath'),
        array_imperfections=yaml_config.get('array_imperfections'),
        random_sampling_mode=yaml_config.get('random_sampling_mode', False),
        random_sample_params=yaml_config.get('random_sample_params'),
        examples_per_combination=yaml_config.get('examples_per_combination', 100),
        dataset_name=yaml_config.get('dataset_name', 'simple_doa_dataset'),
        array_type=yaml_config.get('array_type', 'triangular'),
        num_elements=yaml_config.get('num_elements', 4),
        carrier_freq=float(yaml_config.get('carrier_freq', 2.45e9)),
        sampling_freq=float(yaml_config.get('sampling_freq', 1e6)),
        save_received_signal=yaml_config.get('save_received_signal', True),
        save_covariance_matrix=yaml_config.get('save_covariance_matrix', True),
        save_clean_covariance_matrix=yaml_config.get('save_clean_covariance_matrix', False),
        save_autocorrelation_matrix=yaml_config.get('save_autocorrelation_matrix', False),
        save_music_estimates=yaml_config.get('save_music_estimates', False),
        save_steering_vectors=yaml_config.get('save_steering_vectors', True),
        save_array_metadata=yaml_config.get('save_array_metadata', True),
        autocorr_tau=yaml_config.get('autocorr_tau', 8),
        music_angle_grid_deg=yaml_config.get('music_angle_grid_deg'),
        music_num_sources=yaml_config.get('music_num_sources')
    )
    
    return config


def main():
    """Main function with command line argument support."""
    # Change to project root directory (two levels up from src/data/)
    script_dir = Path(__file__).parent  # src/data/
    project_root = script_dir.parent.parent  # Tri4Net/
    os.chdir(project_root)
    
    parser = argparse.ArgumentParser(description='Generate controlled DOA dataset')
    parser.add_argument('--config', '-c', type=str, 
                       help='Path to YAML config file (relative to Tri4Net/configs/dataset_configs/)')
    parser.add_argument('--output-dir', '-o', type=str, default='Data/datasets',
                       help='Output directory for dataset (default: Data/datasets)')
    parser.add_argument('--array-type', '-a', type=str, default='triangular',
                       help='Array type: triangular, linear, circular, cross, custom (default: triangular)')
    parser.add_argument('--num-elements', '-n', type=int, default=4,
                       help='Number of array elements (default: 4)')
    
    args = parser.parse_args()
    
    if args.config:
        # Load config from file
        print(f"Loading configuration from: configs/dataset_configs/{args.config}")
        config = load_config_from_yaml(args.config)
        # Override array parameters from command line if provided
        if args.array_type != 'triangular':
            config.array_type = args.array_type
        if args.num_elements != 4:
            config.num_elements = args.num_elements
    else:
        # Use default test configuration
        print("No config file specified, using default test configuration")
        config = SimpleDatasetConfig(
            angles_deg=range(0, 361, 1),      
            snr_db=[-10, 0, 10],              # Three SNR levels
            num_snapshots=[256],              # Fixed number of snapshots
            num_sources=[[1, 0], [1, 1], [1, 2]],     
            num_multipath=[0, 1, 2],                # No multipath for simplicity
            array_imperfections=[True, False], # Test both perfect and imperfect arrays
            examples_per_combination=1,       # 1 example per parameter combination
            dataset_name="simple_example",
            array_type=args.array_type,
            num_elements=args.num_elements,
            random_sampling_mode=False,
            random_sample_params=["sir_db", "num_sources", "num_multipath", "array_imperfections"]
        )
    
    print(f"Array type: {config.array_type}")
    print(f"Number of elements: {config.num_elements}")
    print(f"Dataset name: {config.dataset_name}")
    # print(f"Total parameter combinations: {config.get_total_combinations()}")
    # print(f"Total samples: {config.get_total_samples()}")
    
    # Generate dataset
    generator = ControlledDatasetGenerator(config, seed=42)
    dataset_file = generator.generate_dataset(args.output_dir)
    
    # Load and test filtering
    try:
        dataset = DOADataset(dataset_file)
        print(f"\nLoaded dataset with {len(dataset)} samples")
        
        print("\nLabel summary:")
        for label, summary in dataset.get_label_summary().items():
            print(f"  {label}: {summary}")
            
    except ImportError:
        print("PyTorch not available for testing dataset loading")


if __name__ == "__main__":
    main() 