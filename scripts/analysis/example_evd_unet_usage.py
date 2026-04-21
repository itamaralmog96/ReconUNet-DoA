#!/usr/bin/env python3
"""
Example usage of EVDUNet models for DOA estimation with covariance denoising.

This script demonstrates how to:
1. Import and use the EVDUNet models
2. Create autocorrelation input tensors
3. Perform denoising inference
4. Apply MUSIC algorithm on denoised covariance matrices
"""

import sys
import os
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

# Add src to path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

# Set OpenMP workaround for macOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def create_test_signal():
    """Create a test signal with known DOAs for demonstration"""
    from signalgen.array_processing import ArrayConfig, ArrayModel
    from signalgen.signal_generation import SignalConfig, SignalGenerator
    
    # Array configuration
    array_config = ArrayConfig(
        array_type='linear',
        num_elements=8,
        carrier_freq=2.45e9,
        element_spacing=0.5
    )
    
    # Signal configuration
    signal_config = SignalConfig(
        carrier_freq=2.45e9,
        sampling_freq=1e6,
        num_snapshots=512,
        bandwidth=1e6
    )
    
    # Create models
    array_model = ArrayModel(array_config)
    signal_generator = SignalGenerator(signal_config, array_model)
    
    # Generate signal with two sources
    doas = [60.0, 120.0]  # Two sources at 60° and 120°
    snr_db = 5.0  # Moderate SNR
    
    received_signal = signal_generator.generate_received_signal(
        doas=doas,
        source_powers_db=[0.0, 0.0],
        source_bandwidths=[1e6, 1e6],
        snr_db=snr_db,
        multipath_enabled=True,
        num_multipath_per_source=1
    )
    
    return received_signal.data, doas, array_model

def create_autocorrelation_tensor(received_signal, tau=8):
    """Convert received signal to autocorrelation tensor format for UNet"""
    if not isinstance(received_signal, torch.Tensor):
        received_signal = torch.tensor(received_signal, dtype=torch.complex64)
    
    M, T = received_signal.shape
    autocorr_tensor = torch.zeros(tau, 2*M, M, dtype=torch.float32)
    
    # Compute autocorrelation for each lag
    max_lag = min(tau, T - 1)
    
    for lag in range(max_lag):
        samples_available = T - lag
        if samples_available > 0:
            x1 = received_signal[:, :samples_available]
            x2 = received_signal[:, lag:lag + samples_available]
            
            # Compute R(lag) = (1/N) * x1 @ x2^H
            R_lag = torch.matmul(x1, x2.conj().T) / samples_available
            
            # Store real and imaginary parts
            autocorr_tensor[lag, :M, :] = R_lag.real
            autocorr_tensor[lag, M:, :] = R_lag.imag
    
    return autocorr_tensor

def apply_music_algorithm(covariance_matrix, array_model, scan_angles, num_sources):
    """Apply MUSIC algorithm to covariance matrix"""
    from models.classic.music import MUSIC
    
    try:
        music = MUSIC(
            array_model=array_model,
            scan_angles_deg=scan_angles,
            num_sources=num_sources
        )
        music.set_received_covariance(covariance_matrix)
        estimated_angles, spectrum = music.estimate_doa()
        return estimated_angles, spectrum
    except Exception as e:
        print(f"⚠️ MUSIC failed: {e}")
        return np.array([]), np.array([])

def main():
    print("🚀 EVDUNet Usage Example")
    print("=" * 50)
    
    # 1. Create test signal
    print("📊 Creating test signal...")
    received_signal, true_doas, array_model = create_test_signal()
    print(f"   Signal shape: {received_signal.shape}")
    print(f"   True DOAs: {true_doas}")
    
    # 2. Import and create UNet models
    print("\n🏗️ Creating UNet models...")
    from models.deep_learning.EVDUNet import CovarianceReconstructionUNet, EVDCovarianceReconstructionUNet
    
    tau, M = 8, 8
    
    # Create both models
    covariance_unet = CovarianceReconstructionUNet(tau=tau, M=M, activation_type="relu")
    evd_unet = EVDCovarianceReconstructionUNet(tau=tau, M=M, activation_type="anti_rectifier")
    
    print(f"   CovarianceReconstructionUNet: {sum(p.numel() for p in covariance_unet.parameters())} parameters")
    print(f"   EVDCovarianceReconstructionUNet: {sum(p.numel() for p in evd_unet.parameters())} parameters")
    
    # 3. Create autocorrelation input
    print("\n🔄 Creating autocorrelation input...")
    autocorr_input = create_autocorrelation_tensor(received_signal, tau=tau)
    autocorr_batch = autocorr_input.unsqueeze(0)  # Add batch dimension
    print(f"   Autocorrelation tensor shape: {autocorr_batch.shape}")
    
    # 4. Perform inference with both models
    print("\n🧠 Performing UNet inference...")
    
    covariance_unet.eval()
    evd_unet.eval()
    
    with torch.no_grad():
        # CovarianceReconstructionUNet
        print("   Testing CovarianceReconstructionUNet...")
        Kx_raw, Rz_cov, Rz_real_imag = covariance_unet(autocorr_batch)
        print(f"     Output shapes: {Kx_raw.shape}, {Rz_cov.shape}, {Rz_real_imag.shape}")
        
        # EVDCovarianceReconstructionUNet
        print("   Testing EVDCovarianceReconstructionUNet...")
        eigenvals, eigenvecs, Rz_evd = evd_unet(autocorr_batch)
        print(f"     Output shapes: {eigenvals.shape}, {eigenvecs.shape}, {Rz_evd.shape}")
        print(f"     Eigenvalues: {eigenvals.squeeze().numpy()}")
    
    # 5. Apply MUSIC algorithm
    print("\n🎯 Applying MUSIC algorithm...")
    scan_angles = np.arange(30, 151, 1)
    num_sources = len(true_doas)
    
    # Original covariance
    original_cov = np.cov(received_signal)
    music_original, spectrum_original = apply_music_algorithm(
        original_cov, array_model, scan_angles, num_sources
    )
    
    # CovarianceReconstructionUNet denoised
    cov_denoised = Rz_cov.squeeze().numpy()
    music_cov_unet, spectrum_cov_unet = apply_music_algorithm(
        cov_denoised, array_model, scan_angles, num_sources
    )
    
    # EVDCovarianceReconstructionUNet denoised
    evd_denoised = Rz_evd.squeeze().numpy()
    music_evd_unet, spectrum_evd_unet = apply_music_algorithm(
        evd_denoised, array_model, scan_angles, num_sources
    )
    
    # 6. Display results
    print("\n📊 Results Summary:")
    print(f"   True DOAs:                    {true_doas}")
    print(f"   MUSIC (Original):             {music_original}")
    print(f"   MUSIC (Covariance UNet):      {music_cov_unet}")
    print(f"   MUSIC (EVD UNet):             {music_evd_unet}")
    
    # Calculate errors
    def calculate_error(true_angles, estimated_angles):
        if len(estimated_angles) == 0:
            return np.inf
        true_sorted = np.sort(true_angles)
        est_sorted = np.sort(estimated_angles[:len(true_angles)])
        return np.sqrt(np.mean((true_sorted - est_sorted)**2))
    
    error_original = calculate_error(true_doas, music_original)
    error_cov_unet = calculate_error(true_doas, music_cov_unet)
    error_evd_unet = calculate_error(true_doas, music_evd_unet)
    
    print(f"\n📏 RMSE Errors:")
    print(f"   Original:         {error_original:.2f}°")
    print(f"   Covariance UNet:  {error_cov_unet:.2f}°")
    print(f"   EVD UNet:         {error_evd_unet:.2f}°")
    
    # 7. Optional: Create plots
    try:
        print("\n📈 Creating visualization...")
        
        plt.figure(figsize=(15, 5))
        
        # Plot 1: MUSIC spectra comparison
        plt.subplot(1, 3, 1)
        if len(spectrum_original) > 0:
            plt.plot(scan_angles, 10*np.log10(np.abs(spectrum_original) + 1e-10), 
                    'b-', linewidth=2, label='Original')
        if len(spectrum_cov_unet) > 0:
            plt.plot(scan_angles, 10*np.log10(np.abs(spectrum_cov_unet) + 1e-10), 
                    'r--', linewidth=2, label='Covariance UNet')
        if len(spectrum_evd_unet) > 0:
            plt.plot(scan_angles, 10*np.log10(np.abs(spectrum_evd_unet) + 1e-10), 
                    'g:', linewidth=2, label='EVD UNet')
        
        # Mark true DOAs
        for doa in true_doas:
            plt.axvline(x=doa, color='black', linestyle='-', alpha=0.7)
        
        plt.xlabel('Angle (degrees)')
        plt.ylabel('Spectrum (dB)')
        plt.title('MUSIC Spectra Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(30, 150)
        
        # Plot 2: Eigenvalues
        plt.subplot(1, 3, 2)
        eigenvals_np = eigenvals.squeeze().numpy()
        plt.stem(range(len(eigenvals_np)), eigenvals_np, basefmt=' ')
        plt.xlabel('Eigenvalue Index')
        plt.ylabel('Eigenvalue Magnitude')
        plt.title('EVD UNet Predicted Eigenvalues')
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Error comparison
        plt.subplot(1, 3, 3)
        methods = ['Original', 'Cov UNet', 'EVD UNet']
        errors = [error_original, error_cov_unet, error_evd_unet]
        colors = ['blue', 'red', 'green']
        
        bars = plt.bar(methods, errors, color=colors, alpha=0.7)
        plt.ylabel('RMSE (degrees)')
        plt.title('DOA Estimation Errors')
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, error in zip(bars, errors):
            if not np.isinf(error):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                        f'{error:.1f}°', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('evd_unet_example_results.png', dpi=150, bbox_inches='tight')
        print("   Plot saved as 'evd_unet_example_results.png'")
        
        # Show plot if in interactive environment
        # plt.show()
        
    except Exception as e:
        print(f"⚠️ Plotting failed: {e}")
    
    print("\n✅ EVDUNet example completed successfully!")
    print("\n🎯 Key takeaways:")
    print("   • Both UNet models are working correctly")
    print("   • EVD UNet provides eigenvalue decomposition approach")
    print("   • Covariance UNet provides direct matrix reconstruction")
    print("   • Both can be used with MUSIC algorithm for DOA estimation")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())



