"""
Multi-Layer Perceptron (MLP) for DOA estimation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Tuple, List
from ..base_model import BaseModel
from ..model_registry import ModelRegistry

@ModelRegistry.register('mlp')
class MLP(BaseModel):
    """Multi-Layer Perceptron for DOA estimation."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the MLP model.
        
        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary containing:
            - num_sensors: int, number of sensors in the array
            - num_sources: int, number of sources to detect
            - start_angle: float, start angle of the search grid
            - end_angle: float, end angle of the search grid
            - step: float, step size of the search grid
            - model_size: List[int], list of layer sizes
            - dropout_ratio: float, dropout rate
            - sp_mode: bool, whether to use spatial spectrum mode
        """
        super().__init__(config)
        
        self.num_sensors = config['num_sensors']
        self.num_sources = config['num_sources']
        self.sp_mode = config.get('sp_mode', False)
        
        # Create search grid
        self.register_buffer('_grid', torch.arange(
            config['start_angle'],
            config['end_angle'] + 0.0001,
            step=config['step']
        ))
        
        # Build MLP layers
        model_size = config['model_size']
        dropout_ratio = config.get('dropout_ratio', 0.0)
        
        layers = []
        for input_size, output_size in zip(model_size[:-2], model_size[1:-1]):
            layers.extend([
                nn.Linear(input_size, output_size),
                nn.BatchNorm1d(output_size),
                nn.ReLU(inplace=True)
            ])
        
        layers.append(nn.Dropout(dropout_ratio))
        layers.append(nn.Linear(model_size[-2], model_size[-1]))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP.
        
        Parameters
        ----------
        x : torch.Tensor
            Input signal tensor of shape (batch_size, num_sensors, num_snapshots)
            
        Returns
        -------
        torch.Tensor
            Probability distribution over angles
        """
        # Reshape input for MLP
        batch_size = x.size(0)
        x = x.reshape(batch_size, -1)
        
        # Forward pass through MLP
        x = self.layers(x)
        
        return torch.sigmoid(x)
    
    def estimate_doa(self, x: torch.Tensor) -> np.ndarray:
        """
        Estimate DOA using the MLP.
        
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