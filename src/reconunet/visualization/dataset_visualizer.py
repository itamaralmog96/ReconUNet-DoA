import os
import sys
from pathlib import Path

# Add the src directory to the Python path
current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.append(str(src_dir))

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import torch
from scipy import signal
from scipy.signal import ShortTimeFFT
from scipy.linalg import toeplitz
from tqdm import tqdm
import scipy

from data.dataset_loader import load_dataset
from Tri4_array import SystemModel, SystemModelParams

class DatasetVisualizer:
    def __init__(self, data_dir: str = "Data/dataset"):
        """
        Initialize the dataset visualizer.
        
        Args:
            data_dir (str): Directory containing the dataset
        """
        print("Loading dataset...")
        self.data_dir = Path(data_dir)
        datasets = load_dataset(self.data_dir)
        self.train_data = datasets['train']
        self.val_data = datasets['val']
        self.test_data = datasets['test']
        print(f"Train signals shape: {self.train_data['signals'].shape}")
        print(f"Train metadata entries: {len(self.train_data['metadata'])}")
        self.system_model = SystemModel(SystemModelParams())
        
    def plot_time_domain(self, signal_idx: int, split: str = 'train') -> None:
        """
        Plot the time domain representation of a signal (real part).
        
        Args:
            signal_idx (int): Index of the signal to plot
            split (str): Dataset split ('train', 'val', or 'test')
        """
        data = getattr(self, f"{split}_data")
        signal_data = data['signals'][signal_idx].T
        angle = data['metadata'][signal_idx]['source_params']['angles'][0]
        snr = data['metadata'][signal_idx]['source_params']['snr_db']
        
        plt.figure(figsize=(12, 6))
        plt.plot(signal_data.real)
        plt.legend([f'Channel {i+1}' for i in range(signal_data.shape[1])])
        plt.title(f'Time Domain (Real Part)\nAngle: {angle}°, SNR: {snr}dB')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.show()
        
    def plot_frequency_domain(self, signal_idx: int, split: str = 'train') -> None:
        """
        Plot the frequency domain representation of a signal.
        
        Args:
            signal_idx (int): Index of the signal to plot
            split (str): Dataset split ('train', 'val', or 'test')
        """
        data = getattr(self, f"{split}_data")
        signal_data = data['signals'][signal_idx].T
        angle = data['metadata'][signal_idx]['source_params']['angles'][0]
        snr = data['metadata'][signal_idx]['source_params']['snr_db']
        
        # Compute FFT along the time axis for each channel
        fft_data = np.fft.fft(signal_data, axis=0)
        fft_data = np.fft.fftshift(fft_data, axes=0)
        freqs = np.fft.fftfreq(len(signal_data)) # Frequencies for the time axis
        # freqs = np.fft.fftshift(freqs)
        plt.figure(figsize=(12, 6))
        # Plot magnitude spectrum for each channel
        plt.plot(freqs, np.abs(fft_data))
        plt.legend([f'Channel {i+1}' for i in range(signal_data.shape[1])])
        plt.title(f'Frequency Domain\nAngle: {angle}°, SNR: {snr}dB')
        plt.xlabel('Normalized Frequency')
        plt.ylabel('Magnitude')
        plt.grid(True)
        plt.show()
        
    def plot_spatial_spectrum(self, signal_idx: int, split: str = 'train') -> None:
        """
        Plot the spatial spectrum of a signal.
        
        Args:
            signal_idx (int): Index of the signal to plot
            split (str): Dataset split ('train', 'val', or 'test')
        """
        data = getattr(self, f"{split}_data")
        signal_data = data['signals'][signal_idx].T  # Shape: (num_samples, num_channels)
        angle = data['metadata'][signal_idx]['source_params']['angles'][0]
        snr = data['metadata'][signal_idx]['source_params']['snr_db']
        
        # Ensure signal_sample is a time snapshot (shape = num_channels,)
        # signal_data has shape (num_samples, num_channels)
        signal_sample = signal_data[0, :] # Take the first sample across all channels
        
        print(f"Shape of signal_data: {signal_data.shape}")
        print(f"Shape of signal_sample: {signal_sample.shape}")
        
        angles = np.linspace(0, 360, 360)
        spectrum = np.zeros_like(angles, dtype=complex)
        
        print("Computing spatial spectrum...")
        for i, theta in enumerate(tqdm(angles, desc="Processing angles")):
            steering_vec = self.system_model.steering_vec(theta)
            spectrum[i] = np.abs(np.dot(steering_vec.conj(), signal_sample))
        
        plt.figure(figsize=(12, 6))
        plt.plot(angles, np.abs(spectrum))
        plt.title(f'Spatial Spectrum\nAngle: {angle}°, SNR: {snr}dB')
        plt.xlabel('Angle (degrees)')
        plt.ylabel('Magnitude')
        plt.grid(True)
        plt.show()
        
    def plot_covariance_matrix(self, signal_idx: int, split: str = 'train') -> None:
        """
        Plot the covariance matrix of a signal.
        
        Args:
            signal_idx (int): Index of the signal to plot
            split (str): Dataset split ('train', 'val', or 'test')
        """
        data = getattr(self, f"{split}_data")
        signal_data = data['signals'][signal_idx]
        angle = data['metadata'][signal_idx]['source_params']['angles'][0]
        snr = data['metadata'][signal_idx]['source_params']['snr_db']
        
        print("Computing covariance matrix...")
        R = np.cov(signal_data)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(np.abs(R), cmap='viridis')
        plt.colorbar(label='Magnitude')
        plt.title(f'Covariance Matrix\nAngle: {angle}°, SNR: {snr}dB')
        plt.show()
        
    def plot_time_frequency(self, signal_idx: int, split: str = 'train', channel: int = 0) -> None:
        """
        Plot the time-frequency representation of a signal channel using STFT.
        
        Args:
            signal_idx (int): Index of the signal to plot
            split (str): Dataset split ('train', 'val', or 'test')
            channel (int): Channel index to plot (default: 0)
        """
        data = getattr(self, f"{split}_data")
        signal_data = data['signals'][signal_idx].T  # Shape: (num_samples, num_channels)
        angle = data['metadata'][signal_idx]['source_params']['angles'][0]
        snr = data['metadata'][signal_idx]['source_params']['snr_db']
        
        # Get the signal for the specified channel
        signal = signal_data[:, channel]
        
        # Compute Short-Time FFT
        fs = 5e9  # Sampling frequency
        nperseg = max(64, min(256, len(signal) // 8))  # Window length
        noverlap = nperseg // 2  # Overlap between windows
        window = scipy.signal.windows.hann(nperseg)  # Create Hann window
        
        # Create ShortTimeFFT object
        stft = ShortTimeFFT(window, noverlap, fs=fs, fft_mode='centered')
        
        # Compute the transform
        Zxx = stft.stft(signal)
        
        # Get time and frequency values
        t = np.arange(Zxx.shape[1]) * (nperseg - noverlap) / fs
        f = np.fft.fftfreq(nperseg, 1/fs)
        f = np.fft.fftshift(f)
        
        # Plot time-frequency representation
        plt.figure(figsize=(12, 6))
        plt.imshow(np.abs(Zxx), 
                  aspect='auto',
                  origin='lower',
                  extent=[t[0], t[-1], f[0], f[-1]],
                  cmap='viridis')
        plt.colorbar(label='Magnitude')
        plt.title(f'Time-Frequency Analysis (Channel {channel+1})\nAngle: {angle}°, SNR: {snr}dB')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.show()
        
    def plot_array_positions(self) -> None:
        """
        Plot the nominal and actual array positions.
        """
        nominal_positions = self.system_model._coords_nominal
        actual_positions = self.system_model.coords
        
        plt.figure(figsize=(10, 8))
        plt.scatter(nominal_positions[:, 0], nominal_positions[:, 1], 
                   label='Nominal Positions', marker='o')
        plt.scatter(actual_positions[:, 0], actual_positions[:, 1], 
                   label='Actual Positions', marker='x')
        plt.title('Array Positions')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.show()

def main():
    """Example usage of the DatasetVisualizer"""
    # Get the absolute path to the dataset
    current_dir = Path(__file__).parent
    dataset_dir = current_dir.parent.parent / "Data" / "dataset"
    visualizer = DatasetVisualizer(data_dir=str(dataset_dir))
    
    # Select a random signal for visualization
    signal_idx = np.random.randint(0, len(visualizer.test_data['signals']))
    
    # Plot all visualizations for the selected signal
    print(f"Visualizing signal {signal_idx}...")
    visualizer.plot_time_domain(signal_idx)
    visualizer.plot_frequency_domain(signal_idx)
    visualizer.plot_spatial_spectrum(signal_idx)
    visualizer.plot_covariance_matrix(signal_idx)
    visualizer.plot_array_positions()
    
    # Plot spectrograms for all channels
    for channel in range(4):  # Tri4 array has 4 channels
        visualizer.plot_time_frequency(signal_idx, channel=channel)

if __name__ == "__main__":
    main() 