"""
CNN-based DOA estimation model.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List
from ..base_model import BaseModel
from ..model_registry import ModelRegistry

@ModelRegistry.register('cnn_doa')
class CNNDOA(BaseModel):
    """CNN-based DOA estimation model."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the CNN DOA model.
        
        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary containing:
            - num_sensors: int, number of sensors in the array
            - num_sources: int, number of sources to detect
            - num_angles: int, number of angles in the search grid
            - conv_channels: List[int], number of channels in each conv layer
            - kernel_sizes: List[int], kernel sizes for each conv layer
            - dropout_rate: float, dropout rate for regularization
        """
        super().__init__(config)
        
        self.num_sensors = config['num_sensors']
        self.num_sources = config['num_sources']
        self.num_angles = config['num_angles']
        
        # Build CNN layers
        layers = []
        in_channels = 2  # Real and imaginary parts
        
        for i, (out_channels, kernel_size) in enumerate(zip(
            config['conv_channels'], config['kernel_sizes'])):
            
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Dropout2d(config['dropout_rate'])
            ])
            in_channels = out_channels
        
        # Final layers
        layers.extend([
            nn.AdaptiveAvgPool2d((1, self.num_angles)),
            nn.Flatten(),
            nn.Linear(self.num_angles, self.num_angles),
            nn.Sigmoid()
        ])
        
        self.cnn = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CNN model.
        
        Parameters
        ----------
        x : torch.Tensor
            Input signal tensor of shape (batch_size, num_sensors, num_snapshots)
            
        Returns
        -------
        torch.Tensor
            Probability distribution over angles
        """
        # Reshape input to include real and imaginary parts
        x = torch.stack([x.real, x.imag], dim=1)
        
        # Add channel dimension for CNN
        x = x.unsqueeze(-1)
        
        # Forward pass through CNN
        return self.cnn(x)
    
    def estimate_doa(self, x: torch.Tensor) -> np.ndarray:
        """
        Estimate DOA using the CNN model.
        
        Parameters
        ----------
        x : torch.Tensor
            Input signal tensor of shape (batch_size, num_sensors, num_snapshots)
            
        Returns
        -------
        np.ndarray
            Estimated DOA angles
        """
        # Get probability distribution
        probs = self.forward(x)
        
        # Find peaks in the probability distribution
        doa_estimates = []
        for i in range(x.size(0)):
            peaks, _ = self._find_peaks(probs[i].cpu().numpy())
            doa_estimates.append(peaks[:self.num_sources])
            
        return np.array(doa_estimates)
    
    def _find_peaks(self, probs: np.ndarray, min_distance: int = 5) -> tuple:
        """
        Find peaks in the probability distribution.
        
        Parameters
        ----------
        probs : np.ndarray
            Probability distribution
        min_distance : int
            Minimum distance between peaks
            
        Returns
        -------
        tuple
            Peak indices and peak values
        """
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(probs, distance=min_distance)
        return peaks, properties['peak_heights'] 