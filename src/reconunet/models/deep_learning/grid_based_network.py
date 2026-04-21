"""
Grid-based neural network for DOA estimation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Tuple, Optional
from ..base_model import BaseModel
from ..model_registry import ModelRegistry

@ModelRegistry.register('grid_based_network')
class GridBasedNetwork(BaseModel):
    """Grid-based neural network for DOA estimation."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the grid-based network.
        
        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary containing:
            - num_sensors: int, number of sensors in the array
            - num_sources: int, number of sources to detect
            - start_angle: float, start angle of the search grid
            - end_angle: float, end angle of the search grid
            - step: float, step size of the search grid
            - threshold: float, threshold for peak detection
            - conv_channels: List[int], number of channels in each conv layer
            - kernel_sizes: List[int], kernel sizes for each conv layer
            - dropout_rate: float, dropout rate for regularization
        """
        super().__init__(config)
        
        self.num_sensors = config['num_sensors']
        self.num_sources = config['num_sources']
        self.threshold = config.get('threshold', 0)
        
        # Create search grid
        self.register_buffer('_grid', torch.arange(
            config['start_angle'],
            config['end_angle'] + 0.0001,
            step=config['step']
        ))
        
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
            nn.AdaptiveAvgPool2d((1, len(self._grid))),
            nn.Flatten(),
            nn.Linear(len(self._grid), len(self._grid)),
            nn.Sigmoid()
        ])
        
        self.cnn = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.
        
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
        Estimate DOA using the grid-based network.
        
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
            doa_estimates.append(self._grid[peaks[:self.num_sources]].cpu().numpy())
            
        return np.array(doa_estimates)
    
    def _find_peaks(self, probs: np.ndarray, min_distance: int = 5) -> Tuple[np.ndarray, np.ndarray]:
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
        Tuple[np.ndarray, np.ndarray]
            Peak indices and peak values
        """
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(probs, distance=min_distance)
        return peaks, properties['peak_heights']
    
    @property
    def grid(self) -> torch.Tensor:
        """
        Get the search grid.
        
        Returns
        -------
        torch.Tensor
            Search grid angles
        """
        return self._grid 