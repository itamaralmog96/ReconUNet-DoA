"""
Array processing module for antenna arrays and received signals.

This module provides:
1. ArrayConfig: Configurable antenna array with steering vector calculation
2. ReceivedSignal: Container for signals received at antenna arrays with noise handling
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from functools import lru_cache
import json
from signalgen.signal_generator import SignalConfig




# Constants
C = 3e8  # Speed of light in m/s

@dataclass 
class ArrayConfig:
    """Configuration for antenna array parameters."""
    # Array geometry
    array_type: str = 'triangular'  # 'triangular', 'linear', 'circular', 'cross', 'custom'
    num_elements: int = 4  # Number of antenna elements
    carrier_freq: float = 2.45e9  # Carrier frequency in Hz
    
    # Array-specific parameters
    element_spacing: Optional[float] = None  # For linear arrays (in wavelengths)
    radius: Optional[float] = None  # For circular arrays (in wavelengths)  
    custom_positions: Optional[List[List[float]]] = None  # Custom [x,y] positions in meters
    
    # Hardware imperfections
    enable_gain_phase_errors: bool = True
    gain_error_range: Tuple[float, float] = (0.975, 1.025)
    phase_error_range_deg: Tuple[float, float] = (-5.0, 5.0)
    
    enable_mutual_coupling: bool = True
    coupling_strength: float = 0.3  # Coupling coefficient magnitude
    coupling_phase_deg: float = -100.0  # Coupling phase in degrees
    coupling_variation: float = 0.9  # ±90% variation in coupling
    
    # Position errors
    position_error_std: float = 0.001  # Standard deviation of position errors in meters
    
    def __post_init__(self):
        """Validate and set default parameters."""
        if self.array_type == 'triangular' and self.num_elements != 4:
            raise ValueError("Triangular array must have 4 elements")
        
        # Set default element spacing if not provided
        if self.element_spacing is None:
            self.element_spacing = 0.5  # Half wavelength
            
        if self.radius is None:
            self.radius = 0.5  # Half wavelength

class ArrayModel:
    """Antenna array model with steering vector calculation and hardware imperfections."""
    
    def __init__(self, config: ArrayConfig, seed: Optional[int] = None):
        """
        Initialize array model.
        
        Parameters
        ----------
        config : ArrayConfig
            Array configuration parameters
        seed : Optional[int]
            Random seed for reproducible hardware imperfections
        """
        self.config = config
        self.rng = np.random.default_rng(seed)
        
        # Calculate fundamental parameters
        self.wavelength = C / config.carrier_freq
        self.k = 2 * np.pi / self.wavelength
        
        # Generate array geometry
        self.nominal_positions = self._generate_element_positions()
        
        # Apply position errors if enabled
        if config.position_error_std > 0:
            self.element_positions = self._add_position_errors(self.nominal_positions)
        else:
            self.element_positions = self.nominal_positions.copy()
        
        # Generate hardware imperfection matrices
        self.gain_phase_matrix = self._generate_gain_phase_matrix()
        self.mutual_coupling_matrix = self._generate_mutual_coupling_matrix()
        
        # Combined imperfection matrix
        self.imperfection_matrix = self.gain_phase_matrix @ self.mutual_coupling_matrix
        
    def _generate_element_positions(self) -> np.ndarray:
        """Generate element positions based on array type."""
        if self.config.array_type == 'triangular':
            return self._generate_triangular_array()
        elif self.config.array_type == 'linear':
            return self._generate_linear_array()
        elif self.config.array_type == 'circular':
            return self._generate_circular_array()
        elif self.config.array_type == 'cross':
            return self._generate_cross_array()
        elif self.config.array_type == 'custom':
            if self.config.custom_positions is None:
                raise ValueError("Custom positions must be provided for custom array type")
            return np.array(self.config.custom_positions)
        else:
            raise ValueError(f"Unsupported array type: {self.config.array_type}")
    
    def _generate_triangular_array(self) -> np.ndarray:
        """Generate triangular array positions (Tri-4 configuration)."""
        r = self.wavelength * self.config.element_spacing
        positions = np.array([
            [0.0, 0.0],                           # Center element
            [0.0, r],                             # Top element
            [np.sqrt(3)/2 * r, -0.5 * r],        # Bottom right
            [-np.sqrt(3)/2 * r, -0.5 * r],       # Bottom left
        ])
        return positions
    
    def _generate_linear_array(self) -> np.ndarray:
        """Generate linear array positions."""
        spacing = self.wavelength * self.config.element_spacing
        N = self.config.num_elements
        
        # Center the array at origin
        x_positions = np.arange(N) * spacing - (N-1) * spacing / 2
        positions = np.column_stack([x_positions, np.zeros(N)])
        return positions
    
    def _generate_circular_array(self) -> np.ndarray:
        """Generate circular array positions."""
        radius = self.wavelength * self.config.radius
        N = self.config.num_elements
        
        angles = np.arange(N) * 2 * np.pi / N
        x_positions = radius * np.cos(angles)
        y_positions = radius * np.sin(angles)
        positions = np.column_stack([x_positions, y_positions])
        return positions
    
    def _generate_cross_array(self) -> np.ndarray:
        """
        Generate cross array positions (two orthogonal linear arrays).
        
        The num_elements parameter specifies the total number of elements.
        The algorithm automatically distributes them:
        
        For odd N: places one element at center (0,0) and (N-1)/4 elements on each 
        of the 4 arms (±x, ±y directions). Requires N = 4k+1 (e.g., 5, 9, 13, 17...)
        
        For even N: no center element, N/4 elements on each of the 4 arms.
        Requires N = 4k (e.g., 4, 8, 12, 16...)
        
        Examples:
        - N=9: center + 2 elements on each arm = 1 + 4*2 = 9 total
        - N=8: 2 elements on each arm = 4*2 = 8 total
        - N=13: center + 3 elements on each arm = 1 + 4*3 = 13 total
        - N=12: 3 elements on each arm = 4*3 = 12 total
        """
        spacing = self.wavelength * self.config.element_spacing
        N = self.config.num_elements
        
        # Validate that N can be arranged in cross pattern
        if N % 2 == 1:  # Odd number - must be 4k+1
            if (N - 1) % 4 != 0:
                raise ValueError(f"For odd number of elements, N must be of form 4k+1. "
                               f"Got N={N}, which cannot be arranged as 1 center + 4 equal arms. "
                               f"Try N={4*((N-1)//4)+1} or N={4*((N-1)//4)+5}")
            elements_per_arm = (N - 1) // 4
        else:  # Even number - must be 4k
            if N % 4 != 0:
                raise ValueError(f"For even number of elements, N must be divisible by 4. "
                               f"Got N={N}, which cannot be arranged as 4 equal arms. "
                               f"Try N={4*(N//4)} or N={4*(N//4+1)}")
            elements_per_arm = N // 4
        
        positions = []
        
        if N % 2 == 1:  # Odd number - includes center element
            # Center element at (0,0)
            positions.append([0.0, 0.0])
            
            # elements_per_arm elements on each arm
            for i in range(1, elements_per_arm + 1):
                positions.append([i * spacing, 0.0])    # Positive x
                positions.append([-i * spacing, 0.0])   # Negative x
                positions.append([0.0, i * spacing])    # Positive y
                positions.append([0.0, -i * spacing])   # Negative y
                
        else:  # Even number - no center element
            # elements_per_arm elements on each arm
            for i in range(1, elements_per_arm + 1):
                positions.append([i * spacing, 0.0])    # Positive x
                positions.append([-i * spacing, 0.0])   # Negative x
                positions.append([0.0, i * spacing])    # Positive y
                positions.append([0.0, -i * spacing])   # Negative y
        
        return np.array(positions)
    
    def _add_position_errors(self, positions: np.ndarray) -> np.ndarray:
        """Add random position errors to element positions."""
        N = positions.shape[0]
        errors = self.rng.normal(0, self.config.position_error_std, size=(N, 2))
        return positions + errors
    
    def _generate_gain_phase_matrix(self) -> np.ndarray:
        """Generate gain and phase error matrix."""
        # Use actual number of array elements, not config parameter
        actual_num_elements = len(self.element_positions)
        
        if not self.config.enable_gain_phase_errors:
            return np.eye(actual_num_elements, dtype=complex)
        
        gain_errors = self.rng.uniform(*self.config.gain_error_range, size=actual_num_elements)
        phase_errors = np.deg2rad(self.rng.uniform(*self.config.phase_error_range_deg, size=actual_num_elements))
        
        return np.diag(gain_errors * np.exp(1j * phase_errors))
    
    def _generate_mutual_coupling_matrix(self) -> np.ndarray:
        """
        Generate a Toeplitz mutual coupling matrix.

        Uses a geometric sequence per diagonal:
            c_k = (gamma)**k * v_k,  k = 1..M-1
        where gamma = coupling_strength * exp( j * coupling_phase_deg ),
        and v_k ~ Uniform(1 - variation/2, 1 + variation/2) is shared across each diagonal
        to preserve the Toeplitz structure. Returns I + E_mc.
        """
        # Actual number of elements from current array definition
        M = len(self.element_positions)

        if not getattr(self.config, "enable_mutual_coupling", True) or M <= 1:
            return np.eye(M, dtype=complex)

        strength = float(self.config.coupling_strength)            # e.g., 0.3
        phase_rad = np.deg2rad(float(self.config.coupling_phase_deg))  # e.g., -100°
        variation = float(self.config.coupling_variation)           # e.g., 0.9 (±90%)

        # Complex base coefficient gamma
        gamma = strength * np.exp(1j * phase_rad)

        # One random variation factor per diagonal order k (shared across that diagonal)
        rng = getattr(self, "rng", np.random.default_rng())
        v_low, v_high = 1.0 - variation / 2.0, 1.0 + variation / 2.0
        v = rng.uniform(v_low, v_high, size=M-1) if M > 1 else np.array([])

        # Diagonal coefficients c_k (k=1..M-1): geometric decay with shared variation per k
        c = np.empty(M-1, dtype=complex)
        for k in range(1, M):
            c[k-1] = (gamma ** k) * (v[k-1] if M > 1 else 1.0)

        # Build Toeplitz off-diagonal matrix E_mc with E[i,j] = c_|i-j| (and zero diagonal)
        E = np.zeros((M, M), dtype=complex)
        for i in range(M):
            for j in range(M):
                k = abs(i - j)
                if k > 0:
                    E[i, j] = c[k-1]

        # Return I + E_mc (reciprocal, Toeplitz; symmetric but not necessarily Hermitian)
        return np.eye(M, dtype=complex) + E

    # def _generate_mutual_coupling_matrix(self) -> np.ndarray:
    #     """Generate mutual coupling matrix."""
    #     # Use actual number of array elements, not config parameter
    #     actual_num_elements = len(self.element_positions)
        
    #     if not self.config.enable_mutual_coupling:
    #         return np.eye(actual_num_elements, dtype=complex)
        
    #     coupling_matrix = np.eye(actual_num_elements, dtype=complex)
        
    #     # Calculate coupling based on element distances
    #     base_coupling = self.config.coupling_strength * np.exp(1j * np.deg2rad(self.config.coupling_phase_deg))
        
    #     for i in range(actual_num_elements):
    #         for j in range(actual_num_elements):
    #             if i != j:
    #                 # Distance between elements
    #                 distance = np.linalg.norm(self.element_positions[i] - self.element_positions[j])
                    
    #                 # Coupling decreases with distance
    #                 coupling_factor = base_coupling / (1 + distance / self.wavelength)
                    
    #                 # Add random variation
    #                 variation = self.rng.uniform(1 - self.config.coupling_variation/2, 
    #                                            1 + self.config.coupling_variation/2)
    #                 coupling_matrix[i, j] = coupling_factor * variation
        
    #     return coupling_matrix
    
    @lru_cache(maxsize=720)  # Cache for efficiency (every 0.5 degrees)
    def steering_vector(self, angle_deg: float, nominal: bool = False) -> np.ndarray:
        """
        Calculate steering vector for a given angle.
        
        Parameters
        ----------
        angle_deg : float
            Angle of arrival in degrees (0° = +x axis, 90° = +y axis)
        nominal : bool
            If True, ignore all hardware imperfections including position errors
            
        Returns
        -------
        np.ndarray
            Steering vector of shape (num_elements,)
        """
        angle_rad = np.deg2rad(angle_deg)
        
        # Calculate phase delays based on element positions
        kx = self.k * np.cos(angle_rad)
        ky = self.k * np.sin(angle_rad)
        
        # Choose positions based on nominal flag
        positions = self.nominal_positions if nominal else self.element_positions
        phases = positions[:, 0] * kx + positions[:, 1] * ky
        ideal_steering_vector = np.exp(1j * phases)
        
        if nominal:
            return ideal_steering_vector
        else:
            return self.imperfection_matrix @ ideal_steering_vector
    
    def steering_matrix(self, angles_deg: Union[np.ndarray, List[float]], 
                       nominal: bool = False) -> np.ndarray:
        """
        Calculate steering matrix for multiple angles.
        
        Parameters
        ----------
        angles_deg : array-like
            Angles of arrival in degrees
        nominal : bool
            If True, ignore all hardware imperfections including position errors
            
        Returns
        -------
        np.ndarray
            Steering matrix of shape (num_elements, num_angles)
        """
        angles_deg = np.asarray(angles_deg)
        steering_vectors = []
        
        for angle in angles_deg.flatten():
            steering_vectors.append(self.steering_vector(angle, nominal))
            
        A = np.column_stack(steering_vectors)
        A = A / A[0, :]
        A = A / np.linalg.norm(A, axis=0)
        return A
    
    def to_dict(self) -> Dict:
        """Export array model parameters to dictionary."""
        return {
            'config': asdict(self.config),
            'wavelength': self.wavelength,
            'nominal_positions': self.nominal_positions.tolist(),
            'element_positions': self.element_positions.tolist(),
            'gain_phase_matrix': self.gain_phase_matrix.tolist(),
            'mutual_coupling_matrix': self.mutual_coupling_matrix.tolist(),
            'imperfection_matrix': self.imperfection_matrix.tolist()
        }
    
    def __repr__(self) -> str:
        return (f"ArrayModel({self.config.array_type}, N={self.config.num_elements}, "
                f"f={self.config.carrier_freq/1e9:.2f}GHz)")


class ReceivedSignal:
    """Container for signals received at antenna array with noise handling."""
    
    def __init__(self, steering_matrix: np.ndarray, 
                 source_signals: Optional[np.ndarray] = None,
                 source_angles: Optional[List[float]] = None,
                 signal_config: Optional[SignalConfig] = None):
        """
        Initialize received signal container.
        
        Parameters
        ----------
        steering_matrix : np.ndarray
            Steering vectors matrix of shape (num_elements, num_sources)
        source_signals : Optional[np.ndarray]
            Source signals of shape (num_sources, num_samples)
        source_angles : Optional[List[float]]
            Angles of arrival for sources in degrees
        signal_config: Optional[SignalConfig]
            Signal configuration
        """
        self.steering_matrix = steering_matrix
        self.source_signals = source_signals
        self.source_angles = source_angles
        self.signal_config = signal_config
        self.array_signals = None
        self.noise = None
        self.snr_db = None
        
        # For backward compatibility
        self.array_manifold = self.steering_matrix
        
        # Generate array signals if source signals are provided
        if source_signals is not None:
            self.array_signals = self._compute_array_signals()
    
    def _compute_array_signals(self) -> np.ndarray:
        """Compute signals at array elements from source signals."""
        if self.source_signals is None:
            raise ValueError("Source signals must be provided")
        
        # Use the provided steering matrix directly
        steering_vectors = self.steering_matrix
        
        # Apply steering vectors to source signals
        array_signals = steering_vectors @ self.source_signals
        
        return array_signals
    
    def add_noise(self, snr_db: float, noise_type: str = 'gaussian', 
                  reference_signal: Optional[int] = None,
                  source_frequencies: Optional[List[float]] = None,
                  signal_bandwidth: Optional[float] = None,
                  sampling_frequency: Optional[float] = None,
                  frequency_selective: bool = True) -> 'ReceivedSignal':
        """
        Add noise to array signals.
        
        Parameters
        ----------
        snr_db : float
            Signal-to-noise ratio in dB
        noise_type : str
            Type of noise ('gaussian', 'uniform')
        reference_signal : Optional[int]
            Index of source signal to use as reference for SNR calculation
            If None, uses the strongest signal
        source_frequencies : Optional[List[float]]
            Frequencies of each source signal in Hz (required for frequency_selective=True)
        signal_bandwidth : Optional[float]
            Bandwidth of signals in Hz (required for frequency_selective=True)
        sampling_frequency : Optional[float]
            Sampling frequency in Hz (required for frequency_selective=True)
        frequency_selective : bool
            If True, applies frequency-selective noise around each source frequency
            If False, uses broadband white noise (default behavior)
            
        Returns
        -------
        ReceivedSignal
            New ReceivedSignal object with noise added
        """
        if self.array_signals is None:
            raise ValueError("Array signals must be computed before adding noise")
        
        
        # Validate frequency-selective parameters
        if frequency_selective:
            if source_frequencies is None:
                raise ValueError("source_frequencies must be provided when frequency_selective=True")
            if signal_bandwidth is None:
                raise ValueError("signal_bandwidth must be provided when frequency_selective=True")
            if sampling_frequency is None:
                raise ValueError("sampling_frequency must be provided when frequency_selective=True")
            if self.source_signals is None:
                raise ValueError("Source signals must be available for frequency-selective noise")
        
        # Create a copy to avoid modifying original
        noisy_signal = ReceivedSignal(self.steering_matrix, self.source_signals, self.source_angles)
        noisy_signal.array_signals = self.array_signals.copy()
        noisy_signal.snr_db = snr_db
        
        use_full_bandwidth = self.signal_config.use_full_bandwidth
        if use_full_bandwidth:
            # Frequency-selective noise generation
            M, T = self.array_signals.shape  # M: num_elements, T: num_samples
            num_sources = len(source_frequencies)
            
            # Initialize total noise
            total_noise = np.zeros((M, T), dtype=complex)
            # Calculate power of current source signal
            signal_power = np.mean(np.abs(self.source_signals[0, :]) ** 2)
            
            # Calculate noise power based on signal power and SNR
            # Scale by fs/bandwidth ratio to account for filtering
            noise_power = (signal_power / (10 ** (snr_db / 10))) * (sampling_frequency / signal_bandwidth)
            noise_std = np.sqrt(noise_power / 2)  # Complex noise has power split between real/imag
            
            noise_freq = noise_std * (np.random.randn(*self.array_signals.shape) + 
                    1j * np.random.randn(*self.array_signals.shape))
            
            # Convert to time domain
            noise = np.fft.ifft(noise_freq, axis=1) * np.sqrt(T)
            
        elif frequency_selective and not use_full_bandwidth:
            # Frequency-selective noise generation
            M, T = self.array_signals.shape  # M: num_elements, T: num_samples
            num_sources = len(source_frequencies)
            
            # Create frequency axis for filtering
            freqs = np.fft.fftfreq(T, 1/sampling_frequency)
            
            # Initialize total noise
            total_noise = np.zeros((M, T), dtype=complex)
            # Calculate power of current source signal
            signal_power = np.mean(np.abs(self.source_signals[0, :]) ** 2)
            
            # Calculate noise power based on signal power and SNR
            # Scale by fs/bandwidth ratio to account for filtering
            noise_power = (signal_power / (10 ** (snr_db / 10))) * (sampling_frequency / signal_bandwidth)
            noise_std = np.sqrt(noise_power / 2)  # Complex noise has power split between real/imag
            
            # Process each source separately
            for source_idx in range(num_sources):

                # Generate white noise in time domain for each array element
                for element_idx in range(M):
                    # Generate white noise
                    if noise_type == 'gaussian':
                        noise_time = noise_std * (np.random.randn(T) + 1j * np.random.randn(T))
                    elif noise_type == 'uniform':
                        # Uniform noise with same power as Gaussian
                        noise_bound = np.sqrt(3) * noise_std
                        noise_time = noise_bound * ((2 * np.random.rand(T) - 1) + 1j * (2 * np.random.rand(T) - 1))
                    else:
                        raise ValueError(f"Unsupported noise type: {noise_type}")
                    
                    # Convert to frequency domain
                    noise_freq = np.fft.fft(noise_time)
                    
                    # Create filter response (rectangular filter centered on signal frequency)
                    filter_response = np.zeros_like(freqs, dtype=complex)
                    center_freq = source_frequencies[source_idx]
                    bw = signal_bandwidth
                    
                    # Apply rectangular filter where signal exists
                    # Handle both positive and negative frequencies for real signals
                    mask_pos = (freqs >= center_freq - bw/2) & (freqs <= center_freq + bw/2)
                    # mask_neg = (freqs >= -center_freq - bw/2) & (freqs <= -center_freq + bw/2)
                    filter_response[mask_pos] = 1.0
                    
                    # Apply filter to noise in frequency domain
                    filtered_noise_freq = noise_freq * filter_response
                    
                    # Convert back to time domain
                    filtered_noise = np.fft.ifft(filtered_noise_freq)
                    
                    # Add to total noise for this element
                    total_noise[element_idx, :] += filtered_noise
            
            noise = total_noise
        else:
            # Calculate reference signal power
            if reference_signal is not None:
                # Use specific source signal as reference
                if self.source_signals is not None:
                    ref_power = np.mean(np.abs(self.source_signals[reference_signal, :]) ** 2)
                else:
                    raise ValueError("Source signals not available for reference power calculation")
            else:
                # Use total signal power at array
                ref_power = np.mean(np.abs(self.array_signals) ** 2)
                # Original broadband noise generation
                # Calculate noise power
                noise_power = ref_power / (10 ** (snr_db / 10))
            
            # Generate noise
            if noise_type == 'gaussian':
                noise_std = np.sqrt(noise_power / 2)  # Complex noise has power split between real/imag
                noise = noise_std * (np.random.randn(*self.array_signals.shape) + 
                                   1j * np.random.randn(*self.array_signals.shape))
            elif noise_type == 'uniform':
                # Uniform noise with same power as Gaussian
                noise_bound = np.sqrt(3 * noise_power / 2)
                noise = noise_bound * (2 * np.random.rand(*self.array_signals.shape) - 1 + 
                                     1j * (2 * np.random.rand(*self.array_signals.shape) - 1))
            else:
                raise ValueError(f"Unsupported noise type: {noise_type}")
        
        noisy_signal.noise = noise
        noisy_signal.array_signals += noise
        
        return noisy_signal
    
    def get_signal_power(self) -> np.ndarray:
        """Get signal power at each array element."""
        if self.array_signals is None:
            raise ValueError("Array signals not available")
        return np.mean(np.abs(self.array_signals) ** 2, axis=1)
    
    def get_noise_power(self) -> Optional[np.ndarray]:
        """Get noise power at each array element."""
        if self.noise is None:
            return None
        return np.mean(np.abs(self.noise) ** 2, axis=1)
    
    def get_snr_per_element(self) -> Optional[np.ndarray]:
        """Get SNR at each array element in dB."""
        signal_power = self.get_signal_power()
        noise_power = self.get_noise_power()
        
        if noise_power is None:
            return None
            
        return 10 * np.log10(signal_power / (noise_power + 1e-12))
    
    def to_dict(self) -> Dict:
        """Export received signal data to dictionary."""
        data = {
            'array_manifold_shape': list(self.array_manifold.shape) if self.array_manifold is not None else None,
            'source_angles': self.source_angles,
            'snr_db': self.snr_db,
            'signal_power_per_element': self.get_signal_power().tolist() if self.array_signals is not None else None,
            'noise_power_per_element': self.get_noise_power().tolist() if self.noise is not None else None,
            'snr_per_element': self.get_snr_per_element().tolist() if self.get_snr_per_element() is not None else None
        }
        
        # Add signal data (but not the full arrays to keep size reasonable)
        if self.array_signals is not None:
            data['array_signals_shape'] = list(self.array_signals.shape)
            data['array_signals_dtype'] = str(self.array_signals.dtype)
        
        return data
    
    def save(self, filename: str):
        """Save received signal to file."""
        data = self.to_dict()
        
        # Add full signal arrays for saving
        if self.array_signals is not None:
            data['array_signals_real'] = np.real(self.array_signals).tolist()
            data['array_signals_imag'] = np.imag(self.array_signals).tolist()
        
        if self.source_signals is not None:
            data['source_signals_real'] = np.real(self.source_signals).tolist()
            data['source_signals_imag'] = np.imag(self.source_signals).tolist()
        
        # Save steering matrix directly
        if self.steering_matrix is not None:
            data['steering_matrix_real'] = np.real(self.steering_matrix).tolist()
            data['steering_matrix_imag'] = np.imag(self.steering_matrix).tolist()
        
        # Save signal configuration if available
        if self.signal_config is not None:
            data['signal_config'] = {
                'fs': self.signal_config.fs,
                'T': self.signal_config.T,
                'num_sources': self.signal_config.num_sources,
                'angles': self.signal_config.angles,
                'frequencies': self.signal_config.frequencies,
                'use_full_bandwidth': self.signal_config.use_full_bandwidth,
                'bandwidth': self.signal_config.bandwidth,
                'snr_db': self.signal_config.snr_db
            }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, filename: str) -> 'ReceivedSignal':
        """Load received signal from file."""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        # Reconstruct steering matrix
        steering_matrix = None
        if 'steering_matrix_real' in data and 'steering_matrix_imag' in data:
            real_part = np.array(data['steering_matrix_real'])
            imag_part = np.array(data['steering_matrix_imag'])
            steering_matrix = real_part + 1j * imag_part
        elif 'array_manifold_real' in data and 'array_manifold_imag' in data:
            # Backward compatibility with old format
            real_part = np.array(data['array_manifold_real'])
            imag_part = np.array(data['array_manifold_imag'])
            steering_matrix = real_part + 1j * imag_part
        
        if steering_matrix is None:
            raise ValueError("No steering matrix found in saved file")
        
        # Reconstruct source signals
        source_signals = None
        if 'source_signals_real' in data and 'source_signals_imag' in data:
            real_part = np.array(data['source_signals_real'])
            imag_part = np.array(data['source_signals_imag'])
            source_signals = real_part + 1j * imag_part
        
        # Reconstruct signal config if available
        signal_config = None
        if 'signal_config' in data:
            from .signal_generator import SignalConfig
            config_data = data['signal_config']
            signal_config = SignalConfig(**config_data)
        
        # Create ReceivedSignal instance
        source_angles = data.get('source_angles')
        received_signal = cls(steering_matrix, source_signals, source_angles, signal_config)
        received_signal.snr_db = data.get('snr_db')
        
        # Reconstruct array signals (override if saved explicitly)
        if 'array_signals_real' in data and 'array_signals_imag' in data:
            real_part = np.array(data['array_signals_real'])
            imag_part = np.array(data['array_signals_imag'])
            received_signal.array_signals = real_part + 1j * imag_part
        
        return received_signal
    
    def __repr__(self) -> str:
        shape_str = f"{self.array_signals.shape}" if self.array_signals is not None else "None"
        snr_str = f"{self.snr_db}dB" if self.snr_db is not None else "No noise"
        manifold_shape = f"{self.array_manifold.shape}"
        return f"ReceivedSignal(manifold_shape={manifold_shape}, signal_shape={shape_str}, SNR={snr_str})"

# Example usage
if __name__ == "__main__":
    # Test cross array configuration
    print("Testing cross array configurations:")
    
    # Test N=9 (odd, 4k+1) - 1 center + 2 per arm
    config_9 = ArrayConfig(
        array_type='cross',
        num_elements=9,
        carrier_freq=2.4e9,
        enable_gain_phase_errors=False,
        enable_mutual_coupling=False
    )
    
    array_9 = ArrayModel(config_9, seed=42)
    print(f"\nN=9 (odd, 4k+1): {array_9}")
    print(f"Total elements: {len(array_9.element_positions)}")
    print(f"Element positions:\n{array_9.element_positions}")
    
    # Test N=8 (even, 4k) - 2 per arm, no center
    config_8 = ArrayConfig(
        array_type='cross',
        num_elements=8,
        carrier_freq=2.4e9,
        enable_gain_phase_errors=False,
        enable_mutual_coupling=False
    )
    
    array_8 = ArrayModel(config_8, seed=42)
    print(f"\nN=8 (even, 4k): {array_8}")
    print(f"Total elements: {len(array_8.element_positions)}")
    print(f"Element positions:\n{array_8.element_positions}")
    
    # Test N=13 (odd, 4k+1) - 1 center + 3 per arm
    config_13 = ArrayConfig(
        array_type='cross',
        num_elements=13,
        carrier_freq=2.4e9,
        enable_gain_phase_errors=False,
        enable_mutual_coupling=False
    )
    
    array_13 = ArrayModel(config_13, seed=42)
    print(f"\nN=13 (odd, 4k+1): {array_13}")
    print(f"Total elements: {len(array_13.element_positions)}")
    print(f"Element positions:\n{array_13.element_positions}")
    
    # Test original triangular array for comparison
    config_triangular = ArrayConfig(
        array_type='triangular',
        num_elements=4,
        carrier_freq=2.4e9,
        enable_gain_phase_errors=True,
        enable_mutual_coupling=True
    )
    
    array_triangular = ArrayModel(config_triangular, seed=42)
    print(f"\nTriangular array (N=4): {array_triangular}")
    print(f"Element positions:\n{array_triangular.element_positions}")
    
    # Generate some test signals
    num_sources = 2
    num_samples = 512
    fs = 1e6
    
    t = np.arange(num_samples) / fs
    source_signals = np.array([
        np.exp(2j * np.pi * 100e3 * t),  # 100 kHz
        0.5 * np.exp(2j * np.pi * 150e3 * t)  # 150 kHz, weaker
    ])
    source_angles = [30.0, 120.0]  # degrees
    
    # Test with cross array
    received_cross = ReceivedSignal(array_9.steering_matrix(source_angles), source_signals, source_angles)
    print(f"\nReceived signal with cross array: {received_cross}")
    
    # Add noise
    noisy_received = received_cross.add_noise(snr_db=20)
    print(f"Noisy signal: {noisy_received}")
    print(f"SNR per element: {noisy_received.get_snr_per_element()}")
    
    # Original triangular array example
    received = ReceivedSignal(array_triangular.steering_matrix(source_angles), source_signals, source_angles)
    print(f"Received signal: {received}")
    
    # Add noise
    noisy_received = received.add_noise(snr_db=20)
    print(f"Noisy signal: {noisy_received}")
    print(f"SNR per element: {noisy_received.get_snr_per_element()}") 