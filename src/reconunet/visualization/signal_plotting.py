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


from scipy.stats import pearsonr


def compute_statistics(signals, fs, title="Signal Analysis"):
    """
    Compute and display statistical summary of signals
    
    Args:
        signals: Signal array of shape (num_sources, num_samples)
        fs: Sampling frequency
        title: Title for the analysis
    """
    num_sources = signals.shape[0]
    
    print(f"\n📊 {title} - Statistical Summary")
    print("=" * 60)
    
    for i in range(num_sources):
        signal = signals[i, :]
        
        # Basic statistics
        power = np.mean(np.abs(signal)**2)
        rms = np.sqrt(power)
        peak = np.max(np.abs(signal))
        crest_factor = peak / rms
        
        # Frequency content
        f, psd = scipy.signal.periodogram(signal, fs=fs)
        peak_freq_idx = np.argmax(psd)
        peak_freq = f[peak_freq_idx]
        
        print(f"\n🎯 Source {i+1}:")
        print(f"  💪 Power: {power:.6f}")
        print(f"  📐 RMS: {rms:.6f}")
        print(f"  ⛰️  Peak: {peak:.6f}")
        print(f"  📊 Crest Factor: {crest_factor:.3f}")
        print(f"  🎵 Peak Frequency: {peak_freq/1e3:.1f} kHz")
        print(f"  🔄 Mean Phase: {np.mean(np.angle(signal)):.3f} rad")
        print(f"  📈 Phase Std: {np.std(np.angle(signal)):.3f} rad")
    
    # Cross-correlation summary
    if num_sources > 1:
        print(f"\n🔗 Cross-Correlation Summary:")
        for i in range(num_sources):
            for j in range(i+1, num_sources):
                corr_coef = np.corrcoef(np.real(signals[i, :]), np.real(signals[j, :]))[0, 1]
                print(f"  Source {i+1} ↔ Source {j+1}: {corr_coef:.4f}")
    
    print("=" * 60)




def plot_signal_analysis(signals, t, fs, title="Signal Analysis"):
    """
    Plot time domain and frequency domain analysis of signals.
    
    Args:
        signals: Array of signals to plot (can be 1D or 2D array)
        t: Time axis
        fs: Sampling frequency
        title: Plot title
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
    
    # Ensure signals is a list/array of signals
    if isinstance(signals, np.ndarray) and signals.ndim == 1:
        signals = [signals]
    elif isinstance(signals, np.ndarray) and signals.ndim == 2:
        signals = [signals[i] for i in range(signals.shape[0])]
    
    # Time domain plot
    for i, signal in enumerate(signals):
        # Ensure signal length matches time axis
        if len(signal) != len(t):
            # Truncate or pad signal to match time axis
            if len(signal) > len(t):
                signal = signal[:len(t)]
            else:
                # Pad with zeros if signal is shorter
                padded_signal = np.zeros(len(t), dtype=signal.dtype)
                padded_signal[:len(signal)] = signal
                signal = padded_signal
        
        label = f'Signal {i+1}' if len(signals) > 1 else 'Signal'
        ax1.plot(t, np.real(signal), label=label)
    
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title(f'{title} - Time Domain')
    ax1.legend()
    ax1.grid(True)
    
    # Frequency domain plot
    for i, signal in enumerate(signals):
        # Ensure signal length matches time axis
        if len(signal) != len(t):
            if len(signal) > len(t):
                signal = signal[:len(t)]
            else:
                padded_signal = np.zeros(len(t), dtype=signal.dtype)
                padded_signal[:len(signal)] = signal
                signal = padded_signal
        
        # Compute FFT
        fft = np.fft.fftshift(np.fft.fft(signal))
        freq = np.fft.fftshift(np.fft.fftfreq(len(signal), 1/fs))
        
        # Plot spectrum in dB
        label = f'Signal {i+1}' if len(signals) > 1 else 'Signal'
        ax2.plot(freq, 20 * np.log10(np.abs(fft)), label=label)
    
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Magnitude (dB)')
    ax2.set_title(f'{title} - Frequency Domain')
    ax2.legend()
    ax2.grid(True)
    
    # Set reasonable frequency limits
    max_freq = fs / 2  # Nyquist frequency
    # ax2.set_xlim(-max_freq/4, max_freq/4)  # Show central portion of spectrum
    
    plt.tight_layout()
    plt.show()

def analyze_correlation(signals, title="Signal Analysis"):
    """
    Analyze cross-correlation between signals
    
    Args:
        signals: Signal array of shape (num_sources, num_samples)
        title: Title for the analysis
    """
    num_sources = signals.shape[0]
    
    if num_sources < 2:
        print(f"⚠️  {title}: Only {num_sources} source(s) - skipping correlation analysis")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(8, 6))
    fig.suptitle(f'{title} - Correlation Analysis', fontsize=16, fontweight='bold')
    
    # Cross-correlation matrix
    corr_matrix = np.zeros((num_sources, num_sources))
    for i in range(num_sources):
        for j in range(num_sources):
            corr_matrix[i, j] = np.abs(np.corrcoef((signals[i, :]), (signals[j, :]))[0, 1])
    
    # Plot correlation matrix
    im = axes[0].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    axes[0].set_title('Cross-Correlation Matrix')
    axes[0].set_xlabel('Source Index')
    axes[0].set_ylabel('Source Index')
    
    # Add correlation values as text
    for i in range(num_sources):
        for j in range(num_sources):
            text = axes[0].text(j, i, f'{corr_matrix[i, j]:.3f}',
                              ha="center", va="center", color="black", fontweight='bold')
    
    plt.colorbar(im, ax=axes[0], label='Correlation Coefficient')
    
    # Constellation diagram
    axes[1].set_title('Constellation Diagram')
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    decimation = 10
    for i in range(num_sources):
        color = colors[i % len(colors)]
        axes[1].scatter(np.real(signals[i, ::decimation]), np.imag(signals[i, ::decimation]), 
                       alpha=0.6, s=20, c=color, label=f'Source {i+1}')
    
    axes[1].set_xlabel('Real Part')
    axes[1].set_ylabel('Imaginary Part')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].axis('equal')
    
    plt.tight_layout()
    plt.show()

def plot_autocorrelation(signals, fs, title="Signal Analysis"):
    """
    Plot autocorrelation function for each signal
    
    Args:
        signals: Signal array of shape (num_sources, num_samples)
        fs: Sampling frequency
        title: Title for the analysis
    """
    num_sources = signals.shape[0]
    
    fig, axes = plt.subplots(1, 2, figsize=(8, 6))
    fig.suptitle(f'{title} - Autocorrelation Analysis', fontsize=16, fontweight='bold')
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    # Plot autocorrelation for each signal
    for i in range(num_sources):
        signal = signals[i, :]
        color = colors[i % len(colors)]
        
        # Compute autocorrelation
        autocorr = np.correlate(signal, signal, mode='full')
        lags = np.arange(-len(signal) + 1, len(signal)) / fs
        
        # Normalize autocorrelation
        autocorr = autocorr / len(signal)
        
        # Plot full autocorrelation
        axes[0].plot(lags * 1e6, np.real(autocorr), color=color, 
                    label=f'Source {i+1}', alpha=0.8)
        
        # Plot zoomed autocorrelation around zero lag
        zero_lag_idx = len(signal) - 1
        zoom_range = int(0.1 * len(signal))  # Show 10% of signal length around zero
        start_idx = max(0, zero_lag_idx - zoom_range)
        end_idx = min(len(autocorr), zero_lag_idx + zoom_range)
        
        axes[1].plot(lags[start_idx:end_idx] * 1e6, np.real(autocorr[start_idx:end_idx]), 
                    color=color, label=f'Source {i+1}', alpha=0.8)
    
    axes[0].set_xlabel('Lag (μs)')
    axes[0].set_ylabel('Autocorrelation')
    axes[0].set_title('Full Autocorrelation')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Lag (μs)')
    axes[1].set_ylabel('Autocorrelation')
    axes[1].set_title('Zoomed Autocorrelation (around zero lag)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
def plot_crosscorrelation(signals, fs, title="Signal Analysis"):
    """
    Plot crosscorrelation function between all pairs of signals

    Args:
        signals: Signal array of shape (num_sources, num_samples)
        fs: Sampling frequency
        title: Title for the analysis
    """
    num_sources = signals.shape[0]
    
    # Check if there's only one signal
    if num_sources == 1:
        print("Only one signal provided. Crosscorrelation requires at least two signals.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(8, 6))
    fig.suptitle(f'{title} - Crosscorrelation Analysis', fontsize=16, fontweight='bold')

    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    color_idx = 0

    # Plot crosscorrelation for each pair of signals
    for i in range(num_sources):
        for j in range(i + 1, num_sources):
            signal1 = signals[i, :]
            signal2 = signals[j, :]
            color = colors[color_idx % len(colors)]
            color_idx += 1
            
            # Compute crosscorrelation
            crosscorr = np.correlate(signal1, signal2, mode='full')
            lags = np.arange(-len(signal1) + 1, len(signal1)) / fs
            
            # Normalize crosscorrelation
            crosscorr = crosscorr / len(signal1)
            
            # Plot full crosscorrelation
            axes[0].plot(lags * 1e6, np.real(crosscorr), color=color, 
                        label=f'Source {i+1} vs {j+1}', alpha=0.8)
            
            # Plot zoomed crosscorrelation around zero lag
            zero_lag_idx = len(signal1) - 1
            zoom_range = int(0.1 * len(signal1))  # Show 10% of signal length around zero
            start_idx = max(0, zero_lag_idx - zoom_range)
            end_idx = min(len(crosscorr), zero_lag_idx + zoom_range)
            
            axes[1].plot(lags[start_idx:end_idx] * 1e6, np.real(crosscorr[start_idx:end_idx]), 
                        color=color, label=f'Source {i+1} vs {j+1}', alpha=0.8)

    axes[0].set_xlabel('Lag (μs)')
    axes[0].set_ylabel('Crosscorrelation')
    axes[0].set_title('Full Crosscorrelation')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Lag (μs)')
    axes[1].set_ylabel('Crosscorrelation')
    axes[1].set_title('Zoomed Crosscorrelation (around zero lag)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    
    
    
def full_signal_analysis(signals, t, fs, title="Signal Analysis"):
    """
    Perform complete signal analysis combining all functions
    
    Args:
        signals: Signal array of shape (num_sources, num_samples)
        t: Time vector
        fs: Sampling frequency
        title: Title for the analysis
    """
    print(f"\n🚀 Starting Full Analysis: {title}")
    print(f"📊 Signal Shape: {signals.shape}")
    

    
    # Time and Frequency domain analysis
    plot_signal_analysis(signals, t, fs, title)
    
    # Correlation analysis
    analyze_correlation(signals, title)
    
    # Autocorrelation analysis
    plot_autocorrelation(signals, fs, title)
    
    # Crosscorrelation analysis
    plot_crosscorrelation(signals, fs, title)
    
    # Statistical summary
    compute_statistics(signals, fs, title)
    
    print(f"✅ Analysis complete for: {title}\n")
    
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

    analyze_correlation(signals_main_1_interference_1_mp)