import scipy.signal
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

# Add the src directory to the path for imports
script_dir = Path(__file__).parent  # src/visualization/
src_dir = script_dir.parent  # src/
sys.path.insert(0, str(src_dir))

from signalgen.signal_generator import SignalConfig, SignalGenerator
from signalgen.array_processing import ArrayModel, ArrayConfig, ReceivedSignal


def plot_frequency_domain_comparison(clean_signal, noise, signal_config):
    """
    Plot frequency domain comparison between clean signal and noise for multiple channels.
    
    Args:
        clean_signal: Array of clean signals for each channel
        noise: Array of noise signals for each channel
        signal_config: Signal configuration object containing sampling frequency
    """
    # Create figure with subplots for each channel
    fig, axes = plt.subplots(clean_signal.shape[0], 1, figsize=(8, 6))
    fig.suptitle('Clean Signal vs Noise Comparison (Frequency Domain)')

    # Plot each channel
    for i in range(clean_signal.shape[0]):
        # Compute FFT for clean signal and noise
        clean_fft = np.fft.fftshift(np.fft.fft(clean_signal[i]))
        noise_fft = np.fft.fftshift(np.fft.fft(noise[i]))
        
        # Create frequency axis
        freq = np.fft.fftshift(np.fft.fftfreq(len(clean_signal[i]), 1/signal_config.fs))
        
        # Plot magnitude spectrum in dB
        clean_magnitude_db = 20 * np.log10(np.abs(clean_fft) + 1e-12)
        noise_magnitude_db = 20 * np.log10(np.abs(noise_fft) + 1e-12)
        
        # Plot clean signal
        axes[i].plot(freq, clean_magnitude_db, label='Clean Signal', alpha=0.7, linewidth=1)
        # Plot noise
        axes[i].plot(freq, noise_magnitude_db, label='Noise', alpha=0.7, linewidth=1)
        axes[i].set_ylabel(f'Channel {i+1} [dB]')
        axes[i].legend()
        if i == clean_signal.shape[0]-1:  # Only show xlabel for bottom subplot
            axes[i].set_xlabel('Frequency [Hz]')

    plt.tight_layout()
    plt.show()


def plot_time_domain_comparison(clean_signal, noise, signal_config):
    """
    Plot time domain comparison between clean signal and noise for multiple channels.
    
    Args:
        clean_signal: Array of clean signals for each channel
        noise: Array of noise signals for each channel
        signal_config: Signal configuration object containing sampling frequency
    """
    # Create figure with subplots for each channel
    fig, axes = plt.subplots(clean_signal.shape[0], 1, figsize=(8, 6))
    fig.suptitle('Clean Signal vs Noise Comparison (Time Domain)')

    # Plot each channel
    for i in range(clean_signal.shape[0]):
        # Create time axis
        time = np.arange(len(clean_signal[i])) / signal_config.fs
        
        # Plot clean signal
        axes[i].plot(time, clean_signal[i], label='Clean Signal', alpha=0.7, linewidth=1)
        # Plot noise
        axes[i].plot(time, noise[i], label='Noise', alpha=0.7, linewidth=1)
        axes[i].set_ylabel(f'Channel {i+1} [Amplitude]')
        axes[i].legend()
        if i == clean_signal.shape[0]-1:  # Only show xlabel for bottom subplot
            axes[i].set_xlabel('Time [s]')

    plt.tight_layout()
    plt.show()
    
    
def plot_eigenvalue_analysis(noisy_signal, num_sources):
    """
    Plot eigenvalue analysis of covariance matrix for noisy signal.
    
    Args:
        noisy_signal: Array of noisy signals for each channel
        num_sources: Number of signal sources
    """
    # Calculate covariance matrix
    cov_matrix = np.cov(noisy_signal)
    print(cov_matrix.shape)

    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    # Normalize eigenvectors to unit length (orthonormal)
    # eigenvectors = eigenvectors / np.linalg.norm(eigenvectors, axis=0, keepdims=True)

    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Calculate main eigenvalue divided by sum of smaller eigenvalues and convert to dB
    main_eigenvalue = np.abs(eigenvalues[0])
    smaller_eigenvalues_sum = np.sum(np.abs(eigenvalues[num_sources:]))
    eigenvalue_ratio = main_eigenvalue / smaller_eigenvalues_sum
    eigenvalue_ratio_db = 10 * np.log10(eigenvalue_ratio)

    print(f"\nMain eigenvalue: {main_eigenvalue:.6f}")
    print(f"Sum of smaller eigenvalues: {smaller_eigenvalues_sum:.6f}")
    print(f"Ratio (main/smaller): {eigenvalue_ratio:.6f}")
    print(f"Ratio in dB: {eigenvalue_ratio_db:.2f} dB")

    # Plot eigenvalues
    plt.figure(figsize=(8, 6))
    plt.subplot(1, 2, 1)
    plt.plot(eigenvalues, 'bo-')
    plt.title(f'Eigenvalues of Covariance Matrix - Estimated SNR: {eigenvalue_ratio:.2f}')
    plt.xlabel('Index')
    plt.ylabel('Eigenvalue')
    plt.grid(True)

    # Plot eigenvectors
    plt.subplot(1, 2, 2)
    plt.imshow(np.abs(eigenvectors), cmap='viridis', aspect='auto')
    plt.title(f'Magnitude of Eigenvectors')
    plt.xlabel('Eigenvector Index')
    plt.ylabel('Channel Index')
    plt.colorbar(label='Magnitude')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Create time vector for all scenarios
    fs = 1e6  # 1 MHz sampling rate
    T = 1024  # 1000 samples
    t = np.arange(T) / fs

    # Common parameters
    freq_main = 100e3  # Main signal frequency
    freq_interference1 = 150e3  # First interference frequency
    freq_interference2 = 200e3  # Second interference frequency
    # ===== 1. Main Signal Only =====
    config_main_only = SignalConfig(
        fs=fs,
        T=T,
        num_sources=1,
        angles=[0.0],
        frequencies=[freq_main],
        main_source_power=0,
        use_full_bandwidth=True
    )

    generator_main_only = SignalGenerator(config_main_only)

    # ===== 2. Main + 1 Interference =====
    config_main_1_interference = SignalConfig(
        fs=fs,
        T=T,
        num_sources=2,
        angles=[0.0, 60.0],
        frequencies=[freq_main, freq_interference1],
        main_source_power=0,  # Main at 0dB, interference at -6dB
        sir_db=[0],
        use_full_bandwidth=True
    )

    generator_main_1_interference = SignalGenerator(config_main_1_interference)

    # ===== 3. Main + 2 Interferences =====
    config_main_2_interference = SignalConfig(
        fs=fs,
        T=T,
        num_sources=3,
        angles=[0.0, 60.0, 120.0],
        frequencies=[freq_main, freq_interference1, freq_interference2],
        main_source_power=0,  # Main at 0dB, interferences at -6dB and -10dB
        sir_db=[0, 0],
        use_full_bandwidth=True
    )

    generator_main_2_interference = SignalGenerator(config_main_2_interference)

    # ===== 4. Main + 1 Multipath =====
    config_main_1_mp = SignalConfig(
        fs=fs,
        T=T,
        num_sources=1,
        angles=[0.0],
        frequencies=[freq_main],
        main_source_power=0,
        use_full_bandwidth=True,
        enable_multipath=True
    )

    generator_main_1_mp = SignalGenerator(config_main_1_mp)

    # ===== 5. Main + 2 Multipath =====
    config_main_2_mp = SignalConfig(
        fs=fs,
        T=T,
        num_sources=1,
        angles=[0.0],
        frequencies=[freq_main],
        main_source_power=0,
        use_full_bandwidth=True,
        enable_multipath=True
    )

    generator_main_2_mp = SignalGenerator(config_main_2_mp)

    # ===== 6. Main + 1 Interference + 1 Multipath =====
    config_main_1_interference_1_mp = SignalConfig(
        fs=fs,
        T=T,
        num_sources=2,
        angles=[0.0, 60.0],
        frequencies=[freq_main, freq_interference1],
        main_source_power=0,
        sir_db=[0],
        use_full_bandwidth=True,
        enable_multipath=True
    )

    generator_main_1_interference_1_mp = SignalGenerator(config_main_1_interference_1_mp)

    print("\n🔧 All signal generators configured successfully!")
    print("📋 Configured scenarios:")
    print("  1. Main Signal Only")
    print("  2. Main + 1 Interference") 
    print("  3. Main + 2 Interferences")
    print("  4. Main + 1 Multipath")
    print("  5. Main + 2 Multipath")
    print("  6. Main + 1 Interference + 1 Multipath")
    t, signals_main_only = generator_main_only.generate_signals()
    t, signals_main_1_interference = generator_main_1_interference.generate_signals()
    t, signals_main_2_interference = generator_main_2_interference.generate_signals()
    t, signals_main_1_mp = generator_main_1_mp.generate_signals()
    t, signals_main_2_mp = generator_main_2_mp.generate_signals()
    t, signals_main_1_interference_1_mp = generator_main_1_interference_1_mp.generate_signals()

    print("📊 Generated source signals for all scenarios")
    print(f"  Main Only: {signals_main_only.shape}")
    print(f"  Main + 1 Interference: {signals_main_1_interference.shape}")
    print(f"  Main + 2 Interferences: {signals_main_2_interference.shape}")
    print(f"  Main + 1 Multipath: {signals_main_1_mp.shape}")
    print(f"  Main + 2 Multipath: {signals_main_2_mp.shape}")
    print(f"  Main + 1 Interference + 1 Multipath: {signals_main_1_interference_1_mp.shape}")

    # Add multipath effects to multipath scenarios
    print("\n🌊 Adding multipath effects...")

    # Note: For this demo, we'll simulate multipath by adding delayed and attenuated versions
    # In a real implementation, you'd use the generator's add_multipath method

    # Scenario 4: Main + 1 Multipath
    # Add one multipath component with 0.1 delay and 0.3 attenuation
    signals_main_1_mp, _ = generator_main_1_mp.add_multipath(signals_main_1_mp, t, config_main_1_mp.fs, 2.4e9, config_main_1_mp.bandwidth, num_multipath_components=1)

    # Scenario 5: Main + 2 Multipath
    # Add two multipath components
    signals_main_2_mp, _ = generator_main_2_mp.add_multipath(signals_main_2_mp, t, config_main_2_mp.fs, 2.4e9, config_main_2_mp.bandwidth, num_multipath_components=2)

    # Scenario 6: Main + 1 Interference + 1 Multipath
    # Add multipath to the main signal (first source)
    signals_main_1_interference_1_mp, _ = generator_main_1_interference_1_mp.add_multipath(signals_main_1_interference_1_mp, t, config_main_1_interference_1_mp.fs, 2.4e9, config_main_1_interference_1_mp.bandwidth, num_multipath_components=1)
    # Create different array configurations
    configs = {
        'triangular': ArrayConfig(
            array_type='triangular',
            num_elements=4,
            carrier_freq=2.4e9,
            enable_gain_phase_errors=True,
            enable_mutual_coupling=True
        ),
        'linear': ArrayConfig(
            array_type='linear',
            num_elements=8,
            carrier_freq=2.4e9,
            element_spacing=0.45,  # Half wavelength
            enable_gain_phase_errors=True,
            enable_mutual_coupling=False  # Less relevant for linear arrays
        ),
        'circular': ArrayConfig(
            array_type='circular',
            num_elements=6,
            carrier_freq=2.4e9,
            radius=0.8,  # 0.8 wavelengths
            enable_gain_phase_errors=True,
            enable_mutual_coupling=True
        ),
        'cross': ArrayConfig(
            array_type='cross',
            num_elements=17,
            carrier_freq=2.4e9,
            enable_gain_phase_errors=True,
            enable_mutual_coupling=True,
            element_spacing=1/3,  # Half wavelength
        )
    }

    # Create array models
    arrays = {}
    for name, config in configs.items():
        arrays[name] = ArrayModel(config, seed=42)
        print(f"✅ {name.capitalize()} array: {arrays[name]}")

    print(f"\n📏 Wavelength at 2.4 GHz: {arrays['triangular'].wavelength:.3f} m")
    
    ula_array = arrays['linear']
    triangular_array = arrays['triangular']
    cross_array = arrays['cross']
    
    signal = signals_main_1_interference
    signal_config = config_main_1_interference
    
    steering_matrix = ula_array.steering_matrix(signal_config.angles, nominal=True)
    x = ReceivedSignal(steering_matrix, signal, signal_config.angles, signal_config)
    clean_signal = x.array_signals.copy()
    noisy_object = x.add_noise(snr_db=0, source_frequencies=signal_config.frequencies, signal_bandwidth=signal_config.bandwidth, sampling_frequency=signal_config.fs)
    noise = noisy_object.noise.copy()
    noisy_signal = noisy_object.array_signals.copy()
    
    plot_frequency_domain_comparison(clean_signal, noise, signal_config)
    plot_time_domain_comparison(clean_signal, noise, signal_config)
    plot_eigenvalue_analysis(noisy_signal, signal_config.num_sources)