import numpy as np
from .base_doa_model import BaseDOAModel
from typing import Optional, Tuple

# Import our new array processing components
try:
    # Try relative import first (when imported as package)
    from ...signalgen.array_processing import ArrayModel
except ImportError:
    try:
        # Fallback for notebook/direct execution context
        from signalgen.array_processing import ArrayModel
    except ImportError:
        # Final fallback with path manipulation
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
        from signalgen.array_processing import ArrayModel

class MVDR(BaseDOAModel):
    """
    Minimum Variance Distortionless Response (MVDR) DOA estimation model.
    This method minimizes the output power while maintaining a distortionless response
    in the look direction.
    """
    
    def __init__(self, 
                 array_model: ArrayModel,
                 array_manifold: Optional[np.ndarray] = None,
                 scan_angles_deg: Optional[np.ndarray] = None,
                 num_sources: Optional[int] = None):
        """
        Initialize the MVDR model.
        
        Args:
            array_model (ArrayModel): Array model instance containing array geometry and characteristics
            array_manifold (np.ndarray, optional): Pre-computed array manifold matrix (num_elements, num_angles).
                                                  If None, will be computed for scan_angles_deg.
            scan_angles_deg (np.ndarray, optional): Scan angles in degrees. If None, defaults to 0-359 degrees.
            num_sources (int, optional): Number of sources to detect. If None, will return all detected peaks.
        """
        super().__init__(array_model, array_manifold, scan_angles_deg, num_sources)
        self.R_inv = None  # Cache for inverse covariance matrix
        
    def _compute_spectrum_values(self, array_manifold: np.ndarray, covariance_matrix: np.ndarray) -> np.ndarray:
        """
        Compute the MVDR spectrum values for all steering vectors at once.
        
        Args:
            array_manifold (np.ndarray): Steering vectors for all angles (num_elements, num_angles)
            covariance_matrix (np.ndarray): Estimated covariance matrix
            
        Returns:
            np.ndarray: MVDR spectrum values for all angles
        """
        # Add small diagonal loading for numerical stability
        R = covariance_matrix + 1e-6 * np.eye(self.num_elements)
        
        # Compute inverse of covariance matrix (cached)
        if self.R_inv is None:
            self.R_inv = np.linalg.inv(R)
            
        # Compute MVDR spectrum for all angles at once
        # 1 / (A^H @ R_inv @ A) where A is (num_elements, num_angles)
        denominator = np.real(np.diag(array_manifold.conj().T @ self.R_inv @ array_manifold))
        return 1.0 / denominator
        
    def _compute_spectrum_value(self, steering_vector: np.ndarray, covariance_matrix: np.ndarray) -> float:
        """
        Compute the MVDR spectrum value for a single steering vector.
        
        Args:
            steering_vector (np.ndarray): Steering vector for the current angle
            covariance_matrix (np.ndarray): Estimated covariance matrix
            
        Returns:
            float: MVDR spectrum value
        """
        # Add small diagonal loading for numerical stability
        R = covariance_matrix + 1e-6 * np.eye(self.num_elements)
        
        # Compute inverse of covariance matrix (cached)
        if self.R_inv is None:
            self.R_inv = np.linalg.inv(R)
            
        # Compute MVDR spectrum
        denominator = steering_vector.conj().T @ self.R_inv @ steering_vector
        return 1.0 / np.real(denominator)
        
    def estimate_doa(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate DOA using the MVDR method.
        
        Returns:
            tuple: (estimated_angles, spectrum) where:
                - estimated_angles: Array of estimated DOA angles in degrees
                - spectrum: Spatial spectrum values
        """
        # Reset cached inverse covariance matrix
        self.R_inv = None
        
        # Compute spatial spectrum
        angles_deg, spectrum = self.compute_spatial_spectrum()
        
        # Find peaks in the spectrum (returns angles directly with interpolation)
        estimated_angles = self._find_peaks(spectrum)
        
        return estimated_angles, spectrum 