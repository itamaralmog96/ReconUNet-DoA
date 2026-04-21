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

class ESPRIT(BaseDOAModel):
    """
    Estimation of Signal Parameters via Rotational Invariance Techniques (ESPRIT) DOA estimation model.
    This method exploits the rotational invariance property of uniform linear arrays (ULA)
    to estimate DOA without requiring a spectral search.
    
    ESPRIT is applicable to uniform linear arrays and provides improved computational
    efficiency compared to spectral methods like MUSIC.
    """
    
    def __init__(self, 
                 array_model: ArrayModel,
                 array_manifold: Optional[np.ndarray] = None,
                 scan_angles_deg: Optional[np.ndarray] = None,
                 num_sources: Optional[int] = None,
                 spatial_smoothing: bool = False,
                 subarray_length: Optional[int] = None):
        """
        Initialize the ESPRIT model.
        
        Args:
            array_model (ArrayModel): Array model instance containing array geometry and characteristics
            array_manifold (np.ndarray, optional): Pre-computed array manifold matrix (num_elements, num_angles).
                                                  If None, will be computed for scan_angles_deg.
            scan_angles_deg (np.ndarray, optional): Scan angles in degrees. If None, defaults to 0-359 degrees.
            num_sources (int, optional): Number of sources to detect. If None, will be estimated.
            spatial_smoothing (bool): Whether to apply spatial smoothing for coherent sources
            subarray_length (int, optional): Length of subarrays for spatial smoothing. If None, uses M//2
        """
        super().__init__(array_model, array_manifold, scan_angles_deg, num_sources)
        self.signal_subspace = None  # Cache for signal subspace
        self.spatial_smoothing = spatial_smoothing
        self.subarray_length = subarray_length
        
    def _apply_spatial_smoothing(self, covariance_matrix: np.ndarray) -> np.ndarray:
        """
        Apply spatial smoothing to handle coherent sources.
        
        Args:
            covariance_matrix (np.ndarray): Original covariance matrix
            
        Returns:
            np.ndarray: Spatially smoothed covariance matrix
        """
        M = self.num_elements
        P = self.subarray_length if self.subarray_length is not None else M // 2
        L = M - P + 1  # Number of overlapping subarrays
        
        if P < 2:
            raise ValueError("Subarray length must be at least 2")
        if L < 1:
            raise ValueError("Invalid subarray configuration")
            
        # Initialize smoothed covariance
        R_smoothed = np.zeros((P, P), dtype=complex)
        
        # Average over all overlapping subarrays
        for i in range(L):
            # Extract P×P submatrix for subarray i
            R_sub = covariance_matrix[i:i+P, i:i+P]
            R_smoothed += R_sub
            
        R_smoothed /= L
        return R_smoothed
        
    def _extract_predictions_from_roots(self, roots: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts the DoA predictions from the eigenvalues (roots) of the Phi matrix.
        Uses the same angle conversion as Tri4Net RootMUSIC for consistency.
        
        Args:
            roots (np.ndarray): Eigenvalues of the Phi matrix.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (doa_predictions, roots_angles) where:
                - doa_predictions: Extracted DOAs in degrees
                - roots_angles: Phase components of the roots
        """
        # Calculate the phase component of the roots
        k_d = self.array_model.config.element_spacing * 2 * np.pi
        num_elements = self.array_model.config.num_elements
        if num_elements % 2 == 0:
            c = num_elements / 2
        else:
            c = (num_elements - 1) / 2
        # roots = np.array(roots) * np.exp(1j * k_d * c)
        roots_angles = np.angle(roots) / k_d
        
        # For ESPRIT with ULA, the relationship between eigenvalues and angles is:
        # The eigenvalues of Phi are related to the spatial frequencies
        # Using the same conversion as RootMUSIC for consistency with Tri4Net coordinate system
        
        # The Tri4Net coordinate system uses 0° as +x axis, 90° as +y axis
        # For a linear array along x-axis, broadside is at 90°
        # Convert from spatial frequency to angle in Tri4Net coordinates
        thetas = np.rad2deg(np.arccos(1 * roots_angles))
        doa_predictions = thetas
        
        return doa_predictions, roots_angles
        
    def _compute_spectrum_values(self, array_manifold: np.ndarray, covariance_matrix: np.ndarray) -> np.ndarray:
        """
        Compute the ESPRIT spectrum values for visualization purposes.
        Note: ESPRIT doesn't actually compute a spectrum, but we provide this for compatibility.
        
        Args:
            array_manifold (np.ndarray): Steering vectors for all angles (num_elements, num_angles)
            covariance_matrix (np.ndarray): Estimated covariance matrix
            
        Returns:
            np.ndarray: Pseudo-spectrum values for visualization
        """
        # For ESPRIT, we don't compute a traditional spectrum
        # Instead, we'll return a pseudo-spectrum based on the signal subspace
        if self.signal_subspace is None:
            self._compute_signal_subspace(covariance_matrix)
        
        # Adjust array manifold for spatial smoothing
        if self.spatial_smoothing:
            P = self.signal_subspace.shape[0]
            array_manifold_adjusted = array_manifold[:P, :]
        else:
            array_manifold_adjusted = array_manifold
            
        # Compute pseudo-spectrum similar to MUSIC for visualization
        Us = self.signal_subspace @ np.conj(self.signal_subspace).T
        spectrum = np.real(np.diag(array_manifold_adjusted.conj().T @ Us @ array_manifold_adjusted))
        return spectrum
        
    def _compute_spectrum_value(self, steering_vector: np.ndarray, covariance_matrix: np.ndarray) -> float:
        """
        Compute the spectrum value for a single steering vector.
        Note: This is for compatibility with the base class.
        
        Args:
            steering_vector (np.ndarray): Steering vector for the current angle
            covariance_matrix (np.ndarray): Estimated covariance matrix
            
        Returns:
            float: Spectrum value for the current angle
        """
        if self.signal_subspace is None:
            self._compute_signal_subspace(covariance_matrix)
            
        # Compute pseudo-spectrum value based on signal subspace
        spectrum_value = steering_vector.conj().T @ self.signal_subspace @ self.signal_subspace.conj().T @ steering_vector
        return np.real(spectrum_value)
    
    def _compute_signal_subspace(self, covariance_matrix: np.ndarray):
        """
        Compute the signal subspace from the covariance matrix.
        
        Args:
            covariance_matrix (np.ndarray): Estimated covariance matrix
        """
        # Apply spatial smoothing if enabled
        if self.spatial_smoothing:
            R_input = self._apply_spatial_smoothing(covariance_matrix)
        else:
            R_input = covariance_matrix
            
        # Perform eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(R_input)
        
        # Sort eigenvalues and eigenvectors in descending order
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Estimate number of sources if not provided
        if self.num_sources is None:
            self.num_sources = self._estimate_num_sources(eigenvalues)
        
        # Get signal subspace
        self.signal_subspace = eigenvectors[:, :self.num_sources]
    
    def _estimate_num_sources(self, eigenvalues: np.ndarray) -> int:
        """
        Estimate the number of sources using multiple criteria with fallbacks.
        
        Args:
            eigenvalues (np.ndarray): Eigenvalues of the covariance matrix (sorted descending)
            
        Returns:
            int: Estimated number of sources
        """
        n = len(eigenvalues)
        
        # Use a simple but effective method: look for significant drops in eigenvalue ratios
        num_sources_ratio = 0
        if n > 1:
            ratios = eigenvalues[:-1] / (eigenvalues[1:] + 1e-12)  # Add small epsilon to avoid division by zero
            # Find the largest ratio (biggest drop)
            max_ratio_idx = np.argmax(ratios)
            # If the ratio is significant (>3), consider it a signal/noise boundary
            if ratios[max_ratio_idx] > 3.0:
                num_sources_ratio = max_ratio_idx + 1
        
        # Fallback: simple threshold-based approach
        eigenvalue_threshold = np.mean(eigenvalues) + 0.5 * np.std(eigenvalues)
        num_sources_threshold = np.sum(eigenvalues > eigenvalue_threshold)
        
        # Choose the more conservative estimate
        final_estimate = min(num_sources_ratio, num_sources_threshold) if num_sources_ratio > 0 else num_sources_threshold
        
        # Additional safety check: ensure we don't estimate more sources than elements-1
        final_estimate = min(final_estimate, n-1)
        final_estimate = max(final_estimate, 1)  # At least 1 source
        
        return final_estimate
        
    def estimate_doa(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate DOA using the ESPRIT method.
        
        Returns:
            tuple: (estimated_angles, spectrum) where:
                - estimated_angles: Array of estimated DOA angles in degrees
                - spectrum: Pseudo-spectrum values for visualization
        """
        if self.received_data is None and self.received_covariance is None:
            raise ValueError("No received data or covariance matrix available. "
                           "Call set_received_data or set_received_covariance first.")
            
        # Check if array is suitable for ESPRIT (should be linear)
        if self.array_model.config.array_type != 'linear':
            raise ValueError("ESPRIT is only applicable to linear arrays (ULA). "
                           f"Current array type: {self.array_model.config.array_type}")
            
        # Reset cached signal subspace
        self.signal_subspace = None
        
        # Get covariance matrix (either provided directly or estimated from data)
        covariance_matrix = self._estimate_covariance_matrix()
        
        # Compute signal subspace
        self._compute_signal_subspace(covariance_matrix)
        
        # Separate the signal subspace into 2 overlapping subspaces
        # Us_upper: elements 0 to P-2 (first P-1 elements)
        # Us_lower: elements 1 to P-1 (last P-1 elements)
        # where P is the effective array size (original M or smoothed subarray length)
        Us_upper = self.signal_subspace[:-1, :]  # Remove last row
        Us_lower = self.signal_subspace[1:, :]   # Remove first row
        
        # Generate Phi matrix using pseudo-inverse
        phi = np.linalg.pinv(Us_upper) @ Us_lower
        
        # Find eigenvalues of Phi matrix
        phi_eigenvalues, _ = np.linalg.eig(phi)
        
        # Extract DOA predictions from eigenvalues
        doa_predictions, _ = self._extract_predictions_from_roots(phi_eigenvalues)
        
        # Limit to num_sources if specified
        if self.num_sources is not None and len(doa_predictions) > self.num_sources:
            doa_predictions = doa_predictions[:self.num_sources]
        
        # Compute pseudo-spectrum for visualization compatibility
        angles_deg, spectrum = self.compute_spatial_spectrum()
        
        return doa_predictions, spectrum
