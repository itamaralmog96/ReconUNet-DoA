"""
Signal generation module for Tri4Net.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import torch
from scipy.signal import hilbert
import scipy.signal

@dataclass
class SignalConfig:
    """Configuration for signal generation."""
    # Time parameters
    fs: float  # Sampling frequency in Hz
    T: int     # Number of snapshots
    
    # Source parameters
    num_sources: int  # Number of sources
    angles: List[float]  # Angles of arrival in degrees
    frequencies: List[float]  # Frequencies of sources in Hz
    powers: List[float] = None  # Powers of sources in dB
    main_source_power: float = 0.0  # power of main source when using SIR
    
    # Noise parameters
    snr_db: Optional[float] = None  # Signal-to-noise ratio in dB
    noise_type: str = 'gaussian'  # Type of noise ('gaussian', 'uniform', etc.)
    
    # Interference parameters
    sir_db: Optional[List[float]] = None  # Signal-to-Interference Ratio in dB (optional)
    
    # Signal bandwidth parameters
    bandwidth: Optional[float] = None  # Signal bandwidth in Hz, defaults to 10% of Nyquist frequency
    use_full_bandwidth: bool = False  # Whether to use the full available bandwidth instead of filtering around center frequency
    
    # Multipath parameters
    enable_multipath: bool = False  # Whether to enable multipath effects and delay samples
    num_paths: int = 1  # Number of paths per source
    path_delays: Optional[List[float]] = None  # Path delays in seconds
    path_phases: Optional[List[float]] = None  # Path phases in degrees
    path_gains: Optional[List[float]] = None  # Path gains
    smr: Optional[List[float]] = None  # Signal-to-Multipath Ratio in dB (saved from dB_loss)
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if len(self.angles) != self.num_sources:
            raise ValueError(f"Number of angles ({len(self.angles)}) must match num_sources ({self.num_sources})")
        if len(self.frequencies) != self.num_sources:
            raise ValueError(f"Number of frequencies ({len(self.frequencies)}) must match num_sources ({self.num_sources})")
        
            # Calculate amplitudes based on SIR
        # First source is the main source with specified amplitude
        calculated_powers = [float(self.main_source_power)]
        # Handle SIR-based amplitude calculation or validate manual amplitudes
        if self.sir_db is not None:
            if len(self.sir_db) != self.num_sources - 1:
                raise ValueError(f"Number of SIR values ({len(self.sir_db)}) must be num_sources - 1 ({self.num_sources - 1})")
            
            # Other sources have amplitudes calculated from SIR
            for sir_val in self.sir_db:
                # Convert SIR from dB to linear scale and calculate interference power
                interference_power = float(self.main_source_power) - float(sir_val)
                calculated_powers.append(interference_power)
            
            self.powers = calculated_powers
            # print(f"Calculated powers from SIR: {[f'{a:.3f}' for a in self.powers]}")
        else:
            self.powers = [float(self.main_source_power)] * self.num_sources
            
        # Calculate bandwidth if not provided (10% of Nyquist frequency)
        if self.bandwidth is None:
            self.bandwidth = (self.fs / 2) * 0.1

class SignalGenerator:
    """Signal generator for DOA estimation."""
    
    def __init__(self, config: SignalConfig, seed: Optional[int] = None):
        """
        Initialize signal generator.
        
        Parameters
        ----------
        config : SignalConfig
            Configuration for signal generation
        seed : Optional[int]
            Random seed for reproducibility (affects random phases)
        """
        self.config = config
        self.rng = np.random.default_rng(seed)
        self._validate_config()
        
        # Generate random phases for each source to ensure uncorrelated signals
        self.random_phases = self.rng.uniform(0, 2 * np.pi, size=config.num_sources)
        
    def _validate_config(self):
        """Validate configuration parameters."""
        if self.config.fs <= 0:
            raise ValueError(f"Sampling frequency must be positive, got {self.config.fs}")
        if self.config.T <= 0:
            raise ValueError(f"Number of snapshots must be positive, got {self.config.T}")
        if self.config.num_sources <= 0:
            raise ValueError(f"Number of sources must be positive, got {self.config.num_sources}")

    def generate_time_axis(self) -> np.ndarray:
        """Generate time axis based on sampling frequency and number of snapshots."""
        dt = 1 / self.config.fs
        if self.config.enable_multipath:
            return np.arange(0, self.config.T + 2) * dt
        else:
            return np.arange(0, self.config.T) * dt
        
    def generate_source_signals(self, t: np.ndarray) -> np.ndarray:
        """
        Generate source signals using white noise filtered with a rectangular filter in frequency domain.
        
        The filtering behavior depends on the use_full_bandwidth configuration:
        - If use_full_bandwidth=False (default): Applies bandpass filtering around each source's center frequency
        - If use_full_bandwidth=True: Uses the entire available bandwidth (0 to fs/2), ignoring center frequencies
        
        Parameters
        ----------
        t : np.ndarray
            Time axis in seconds
            
        Returns
        -------
        np.ndarray
            Source signals of shape (num_sources, T) or (num_sources, T + max_delay_samples) if enable_multipath is True
            
        Notes
        -----
        When use_full_bandwidth=True, all signals will occupy the full spectrum from 0 to the Nyquist frequency (fs/2).
        This is useful for broadband signal scenarios where you want signals to span the entire available bandwidth
        rather than being confined to narrow bands around specific center frequencies.
        """
        signals = []
        T = self.config.T
        fs = self.config.fs
        if self.config.use_full_bandwidth:
            self.config.bandwidth = fs * 0.99
        bw = self.config.bandwidth  # Use bandwidth from config

        # Calculate maximum delay in samples if needed
        if self.config.enable_multipath:
            max_delay_seconds = 1 / bw  # Using max_delay_factor of 10.0
            max_delay_samples = int(max_delay_seconds * fs)
            signal_length = T + max_delay_samples
        else:
            signal_length = T
        
        step = fs/signal_length
        for i in range(self.config.num_sources):
            # Generate white noise in frequency domain
            freq_noise = np.random.normal(0, 1, signal_length) + 1j * np.random.normal(0, 1, signal_length)
            # Create frequency axis that matches the freq_noise length
            freq_axis = np.linspace(-fs/2, fs/2, len(freq_noise))
            
            
            if self.config.use_full_bandwidth:
                # Use the entire available bandwidth (-fs/2 to fs/2)
                filter_response = np.ones_like(freq_noise, dtype=complex)
            else:
                # Apply rectangular filter around center frequency
                center_freq = self.config.frequencies[i]
                filter_response = np.zeros_like(freq_axis, dtype=complex)
                mask = (freq_axis >= center_freq - bw/2) & (freq_axis <= center_freq + bw/2)
                filter_response[mask] = 1.0
            
            # Apply filter to noise
            filtered_freq = freq_noise * filter_response
            
            # Convert back to time domain
            time_signal = np.fft.ifft(filtered_freq, n=signal_length)
            # Normalize and scale by amplitude
            # Calculate signal power
            signal_power = np.mean(np.abs(time_signal)**2)
            # Scale signal to match desired power
            time_signal = time_signal * np.sqrt(np.power(10, (self.config.powers[i] / 10)) / signal_power)
            signals.append(time_signal)
            
        return np.array(signals)
        
    def add_multipath(self, signals: np.ndarray, t: np.ndarray, fs: float, 
                     carrier_frequency: float, signal_bandwidth: float,
                     max_paths: int = 3, max_delay_factor: float = 10.0,
                     enable_carrier_phase: bool = True,
                     multipath_distribution: str = 'uniform',
                     num_multipath_components: Optional[int] = None,
                     custom_multipath_angles: Optional[List[float]] = None) -> np.ndarray:
        """
        Add multipath effects to signals with corresponding angles.
        
        Parameters
        ----------
        signals : np.ndarray
            Source signals of shape (num_sources, T)
        t : np.ndarray
            Time axis in seconds
        fs : float
            Sampling frequency in Hz
        carrier_frequency : float
            Carrier frequency in Hz
        signal_bandwidth : float
            Bandwidth of the baseband signal in Hz
        max_paths : int, optional
            Maximum total number of paths (including direct paths), default=3
        max_delay_factor : float, optional
            Maximum delay as a factor of 1/bandwidth, default=10.0
        enable_carrier_phase : bool, optional
            Whether to include carrier phase effects, default=True
        multipath_distribution : str, optional
            Distribution type for multipath parameters ('uniform' or 'exponential'), default='uniform'
        num_multipath_components : Optional[int], optional
            Number of multipath components to add. If None, chooses randomly up to max_additional_paths.
            If specified and exceeds max_additional_paths, chooses randomly within allowed range, default=None
        custom_multipath_angles : Optional[List[float]], optional
            Pre-generated multipath angles in degrees. If provided, uses these instead of generating random angles.
            This ensures consistent multipath angles across different samples (e.g., Mode 4 evaluation), default=None
            
        Returns
        -------
        np.ndarray
            Signals with multipath effects of shape (total_num_paths, T)
        """
        # Input validation
        if not isinstance(signals, np.ndarray) or signals.ndim != 2:
            raise ValueError("signals must be a 2D numpy array")
        if fs <= 0:
            raise ValueError("fs must be positive")
        if carrier_frequency <= 0:
            raise ValueError("carrier_frequency must be positive")
        if signal_bandwidth <= 0:
            raise ValueError("signal_bandwidth must be positive")
        if max_paths < 1:
            raise ValueError("max_paths must be at least 1")
        if max_delay_factor <= 0:
            raise ValueError("max_delay_factor must be positive")
        if num_multipath_components is not None and num_multipath_components < 0:
            raise ValueError(f"num_multipath_components must be non-negative, got {num_multipath_components}")
            
        num_sources = signals.shape[0]
        T_samples = signals.shape[1]
        
        # Calculate signal powers for amplitude scaling
        signal_powers = np.mean(np.abs(signals)**2, axis=1)
        
        # Calculate maximum delay in samples
        max_delay_seconds = 1 / signal_bandwidth
        max_delay_samples = int(max_delay_seconds * fs)
        
        if max_delay_samples >= T_samples:
            raise ValueError(f"Maximum delay ({max_delay_seconds:.3f}s) exceeds signal duration ({T_samples/fs:.3f}s)")
            
        # Initialize multipath configuration
        self.config.multipath = {
            'num_paths': 0,
            'path_delays': [],
            'path_gains': [],
            'path_phases': [],
            'path_angles': []
        }
        
        # Calculate number of additional paths
        max_additional_paths = max(0, max_paths - num_sources)
        
        # Determine number of multipath components to add
        if num_multipath_components is not None:
            # User specified number of multipath components
            if num_multipath_components > max_additional_paths:
                # If specified number exceeds maximum, choose randomly within allowed range
                print(f"Warning: Requested {num_multipath_components} multipath components exceeds maximum "
                      f"additional paths ({max_additional_paths}). Choosing randomly within allowed range.")
                num_additional_paths = np.random.randint(0, max_additional_paths + 1)
            else:
                # Use the specified number
                num_additional_paths = num_multipath_components
        else:
            # No specific number requested, choose randomly as before
            num_additional_paths = np.random.randint(0, max_additional_paths + 1)
        
        all_signals = []
        all_angles = []
        all_shift_samples = []  # Store shift values for each multipath component
        dB_loss_values = []  # Store dB_loss values for SMR calculation
        
        # Upsample the original signal using scipy's resample
        upsampled_signals = np.zeros((1, T_samples * int(max_delay_factor)), dtype=complex)
      
        # Use scipy's resample for upsampling
        upsampled_signals = scipy.signal.resample_poly(signals[0], max_delay_factor, 1)
        # Start from max_delay_samples position
        start_idx = int(max_delay_samples * max_delay_factor)
        source_idx = 0
        signals = signals[:, :-1*max_delay_samples]
        current_source_signal = upsampled_signals[start_idx:]
        current_power = signal_powers[source_idx]
        
        # Generate multipath components in a single loop
        for k in range(num_additional_paths):
            # Generate multipath parameters
            # Use custom angle if provided, otherwise generate random
            if custom_multipath_angles is not None and k < len(custom_multipath_angles):
                AOA_k_rad = np.deg2rad(custom_multipath_angles[k])
            else:
                AOA_k_rad = np.random.uniform(0, 2 * np.pi)
            
            if multipath_distribution == 'uniform':
                time_delay_k_seconds = np.random.uniform(0, max_delay_seconds)
                # Base dB loss is now correlated with delay
                base_dB_loss = 10 * (time_delay_k_seconds / (max_delay_seconds))  # 0-10 dB based on delay
                # Add some randomness to the loss
                dB_loss = base_dB_loss + np.random.uniform(1, 2)  # Additional 0-2 dB random loss
            else:  # exponential
                time_delay_k_seconds = np.random.exponential(max_delay_seconds/2)
                # Base dB loss is exponentially correlated with delay
                base_dB_loss = 10 * (1 - np.exp(-time_delay_k_seconds / max_delay_seconds))  # 0-10 dB based on delay
                # Add some randomness to the loss
                dB_loss = base_dB_loss + np.random.exponential(2.5)  # Additional random loss with mean 2.5 dB
                
            phase_offset_k_rad = np.random.normal(np.pi, np.pi/4)
            amplitude_factor_k = 10 ** (-dB_loss / 20.0)
            
            # Store dB_loss for SMR calculation
            dB_loss_values.append(dB_loss)
            
            # Scale amplitude based on original signal power
            amplitude_factor_k *= np.sqrt(current_power)
            
            # Calculate complex gain
            if enable_carrier_phase:
                complex_gain_k = amplitude_factor_k * np.exp(1j * (phase_offset_k_rad - 
                                                                  2 * np.pi * carrier_frequency * time_delay_k_seconds))
            else:
                complex_gain_k = amplitude_factor_k * np.exp(1j * phase_offset_k_rad)
                
            # Shift signal (now in upsampled domain)
            shift_samples_k = int(time_delay_k_seconds * fs * max_delay_factor/2)
            if shift_samples_k >= len(current_source_signal):
                continue
            if shift_samples_k <= 0:
                shift_samples_k = 1
                
            shifted_signal_k = upsampled_signals[start_idx - shift_samples_k:-1*shift_samples_k]
            
            # Add multipath signal and parameters
            all_signals.append(complex_gain_k * shifted_signal_k)
            all_angles.append(np.rad2deg(AOA_k_rad))
            all_shift_samples.append(shift_samples_k)  # Store the shift value for this component
            
            # Update config
            self.config.multipath['num_paths'] += 1
            self.config.multipath['path_delays'].append(time_delay_k_seconds)
            self.config.multipath['path_gains'].append(amplitude_factor_k)
            self.config.multipath['path_phases'].append(phase_offset_k_rad)
            self.config.multipath['path_angles'].append(AOA_k_rad)
        
        # Save SMR to config (average dB_loss from multipath components)
        if dB_loss_values:
            self.config.smr = dB_loss_values
        else:
            self.config.smr = [999.0]  # No multipath case
        
        # Decimate all signals back to original rate
        if all_signals:
            # Decimate original signal
            decimated_original = upsampled_signals[(start_idx)::int(max_delay_factor)]
            signals[0] = decimated_original
            try:
                # Decimate multipath signals using the correct shift value for each component
                decimated_multipath = np.vstack([signal[::int(max_delay_factor)]
                                                for signal, shift_val in zip(all_signals, all_shift_samples)])
            except:
                print(all_signals)
                print(all_shift_samples)
                print(max_delay_factor)
                print(start_idx)
                print(upsampled_signals.shape)
            
            # Update angles in config
            if hasattr(self.config, 'angles') and self.config.angles is not None:
                self.config.angles = np.concatenate([self.config.angles, np.array(all_angles)])
            else:
                self.config.angles = np.array(all_angles)
            try:
            # Combine with original signals
                return np.vstack([signals, decimated_multipath]), all_shift_samples
            except:
                print(all_signals)
                print(all_shift_samples)
                print(max_delay_factor)
                print(start_idx)
                print(upsampled_signals.shape)
        else:
            return signals, all_shift_samples
        
    def add_noise(self, signals: np.ndarray) -> np.ndarray:
        """
        Add noise to signals.
        
        Parameters
        ----------
        signals : np.ndarray
            Source signals of shape (num_sources, T)
            
        Returns
        -------
        np.ndarray
            Signals with noise
        """
        # Calculate signal power
        signal_power = np.mean(np.abs(signals) ** 2)
        
        # Calculate noise power based on SNR
        noise_power = signal_power / (10 ** (self.config.snr_db / 10))
        
        # Generate noise using the same random generator for reproducibility
        if self.config.noise_type == 'gaussian':
            noise = np.sqrt(noise_power/2) * (self.rng.standard_normal(signals.shape) + 
                                            1j * self.rng.standard_normal(signals.shape))
        else:
            raise ValueError(f"Unsupported noise type: {self.config.noise_type}")
            
        return signals + noise
        
    def generate_signals(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate signals for DOA estimation.
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (time_axis, signals)
            time_axis: Time axis in seconds
            signals: Generated signals of shape (num_sources, T)
        """
        # Generate time axis
        t = self.generate_time_axis()
        
        # Generate source signals
        signals = self.generate_source_signals(t)
        
        # Add multipath effects
        # Example 1: Add exactly 2 multipath components
        # signals = self.add_multipath(signals, t, self.config.fs, self.config.frequencies[0], self.config.bandwidth, num_multipath_components=2)
        # 
        # Example 2: Let the function choose randomly (default behavior)  
        # signals = self.add_multipath(signals, t, self.config.fs, self.config.frequencies[0], self.config.bandwidth)
        
        
        return t, signals
        
    def to_torch(self, signals: np.ndarray) -> torch.Tensor:
        """
        Convert signals to PyTorch tensor.
        
        Parameters
        ----------
        signals : np.ndarray
            Signals to convert
            
        Returns
        -------
        torch.Tensor
            Signals as PyTorch tensor
        """
        return torch.from_numpy(signals).to(torch.complex64)

# Example usage
if __name__ == "__main__":
    # Example 1: Traditional bandpass filtered signals around center frequencies
    print("Example 1: Traditional bandpass filtered signals")
    config1 = SignalConfig(
        fs=1000,  # 1 kHz
        T=1000,   # 1000 snapshots
        num_sources=2,
        angles=[30, 60],  # degrees
        frequencies=[100, 200],  # Hz (center frequencies for filtering)
        bandwidth=50,  # 50 Hz bandwidth around each center frequency
        use_full_bandwidth=False,  # Default: use bandpass filtering
        main_source_power=0.0,  # Main source power in dB
        snr_db=20,
        num_paths=2,
        path_delays=[0.1],  # 100ms delay
        path_gains=[0.5]    # -6dB gain
    )
    
    # Create signal generator
    generator1 = SignalGenerator(config1)
    
    # Generate signals
    t1, signals1 = generator1.generate_signals()
    
    print(f"Bandpass filtered - Time axis shape: {t1.shape}")
    print(f"Bandpass filtered - Signals shape: {signals1.shape}")
    print(f"Bandpass filtered - Signal power: {np.mean(np.abs(signals1)**2, axis=1)}")
    
    # Example 2: Full bandwidth signals (occupy entire spectrum)
    print("\nExample 2: Full bandwidth signals")
    config2 = SignalConfig(
        fs=1000,  # 1 kHz
        T=1000,   # 1000 snapshots
        num_sources=2,
        angles=[30, 60],  # degrees
        frequencies=[100, 200],  # Hz (not used for filtering when use_full_bandwidth=True)
        use_full_bandwidth=True,  # Enable full bandwidth signals
        main_source_power=0.0,  # Main source power in dB
        snr_db=20,
        num_paths=2,
        path_delays=[0.1],  # 100ms delay
        path_gains=[0.5]    # -6dB gain
    )
    
    # Create signal generator
    generator2 = SignalGenerator(config2)
    
    # Generate signals
    t2, signals2 = generator2.generate_signals()
    
    print(f"Full bandwidth - Time axis shape: {t2.shape}")
    print(f"Full bandwidth - Signals shape: {signals2.shape}")
    print(f"Full bandwidth - Signal power: {np.mean(np.abs(signals2)**2, axis=1)}")
    
    # Compare frequency content (basic analysis)
    from scipy.fft import fft, fftfreq
    
    # Frequency analysis for bandpass filtered signal
    freq_spectrum1 = np.abs(fft(signals1[0, :]))
    freqs1 = fftfreq(len(signals1[0, :]), 1/config1.fs)
    
    # Frequency analysis for full bandwidth signal
    freq_spectrum2 = np.abs(fft(signals2[0, :]))
    freqs2 = fftfreq(len(signals2[0, :]), 1/config2.fs)
    
    # Find peak frequencies
    peak_idx1 = np.argmax(freq_spectrum1[:len(freq_spectrum1)//2])
    peak_freq1 = freqs1[peak_idx1]
    
    peak_idx2 = np.argmax(freq_spectrum2[:len(freq_spectrum2)//2])
    peak_freq2 = freqs2[peak_idx2]
    
    print(f"\nFrequency Analysis:")
    print(f"Bandpass filtered - Peak frequency: {peak_freq1:.1f} Hz")
    print(f"Full bandwidth - Peak frequency: {peak_freq2:.1f} Hz")
    print(f"Full bandwidth signals occupy the entire spectrum from 0 to {config2.fs/2} Hz") 