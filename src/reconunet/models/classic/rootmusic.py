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

class RootMUSIC(BaseDOAModel):
    """
    Root Multiple Signal Classification (RootMUSIC) DOA estimation model.
    This method uses polynomial rooting to find DOA estimates from the noise subspace,
    avoiding the need for scanning through all possible angles like conventional MUSIC.
    
    RootMUSIC is applicable to uniform linear arrays (ULA) and provides improved
    computational efficiency compared to spectral MUSIC.
    """
    
    def __init__(self, 
                 array_model: ArrayModel,
                 array_manifold: Optional[np.ndarray] = None,
                 scan_angles_deg: Optional[np.ndarray] = None,
                 num_sources: Optional[int] = None,
                 spatial_smoothing: bool = False,
                 subarray_length: Optional[int] = None):
        """
        Initialize the RootMUSIC model.
        
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
        self.noise_subspace = None  # Cache for noise subspace
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
        
    def _sum_of_diag(self, matrix: np.ndarray) -> list:
        """
        Calculates the sum of diagonals in a square matrix.
        
        Args:
            matrix (np.ndarray): Square matrix for which diagonals need to be summed.
            
        Returns:
            list: A list containing the sums of all diagonals in the matrix, from left to right.
        """
        diag_sum = []
        diag_index = np.linspace(
            -matrix.shape[0] + 1,
            matrix.shape[0] + 1,
            2 * matrix.shape[0] - 1,
            endpoint=False,
            dtype=int,
        )
        for idx in diag_index:
            diag_sum.append(np.sum(matrix.diagonal(idx)))
        return diag_sum
    
    def _find_roots(self, coefficients: list) -> np.ndarray:
        """
        Finds the roots of a polynomial defined by its coefficients.
        
        Args:
            coefficients (list): List of polynomial coefficients in descending order of powers.
            
        Returns:
            np.ndarray: An array containing the roots of the polynomial.
        """
        coefficients = np.array(coefficients)
        A = np.diag(np.ones((len(coefficients) - 2,), coefficients.dtype), -1)
        if np.abs(coefficients[0]) == 0:
            A[0, :] = -coefficients[1:] / (coefficients[0] + 1e-9)
        else:
            A[0, :] = -coefficients[1:] / coefficients[0]
        roots = np.array(np.linalg.eigvals(A))
        return roots
    
    def _extract_predictions_from_roots(self, roots: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts the DoA predictions from the roots of the polynomial.
        
        Args:
            roots (np.ndarray): Roots of the polynomial.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (doa_predictions, roots_angles) where:
                - doa_predictions: Extracted DOAs in degrees
                - roots_angles: Phase components of the roots
        """
        # Calculate the phase component of the roots
        k_d = self.array_model.config.element_spacing*2*np.pi
        num_elements = self.array_model.config.num_elements
        if num_elements % 2 == 0:
            c = num_elements/2
        else:
            c = (num_elements-1)/2
        roots = np.array(roots)*np.exp(1j*k_d*c)
        roots_angles = np.angle(roots) / k_d
        
        # For RootMUSIC with ULA, the relationship between roots and angles is:
        # z = exp(j * pi * cos(theta)) where theta is the angle from broadside
        # Therefore: cos(theta) = angle(z) / pi
        # But we need to handle the coordinate system difference
        
        # The Tri4Net coordinate system uses 0° as +x axis, 90° as +y axis
        # For a linear array along x-axis, broadside is at 90°
        # So we need to convert from sin(theta) to the angle in Tri4Net coordinates
        thetas = np.rad2deg(np.arccos(-1*roots_angles))
        doa_predictions = thetas
        
        # Convert from "angle from broadside" to "angle in Tri4Net coordinates"
        # If array is along x-axis, broadside is at 90°
        # So angle in Tri4Net = 90° - angle_from_broadside
        # doa_predictions = theta_from_broadside
        
        return doa_predictions, roots_angles
    
    def _compute_spectrum_values(self, array_manifold: np.ndarray, covariance_matrix: np.ndarray) -> np.ndarray:
        """
        Compute the RootMUSIC spectrum values for visualization purposes.
        Note: RootMUSIC doesn't actually compute a spectrum, but we provide this for compatibility.
        
        Args:
            array_manifold (np.ndarray): Steering vectors for all angles (num_elements, num_angles)
            covariance_matrix (np.ndarray): Estimated covariance matrix
            
        Returns:
            np.ndarray: Pseudo-spectrum values for visualization
        """
        # For RootMUSIC, we don't compute a traditional spectrum
        # Instead, we'll return a pseudo-spectrum based on the noise subspace
        if self.noise_subspace is None:
            self._compute_noise_subspace(covariance_matrix)
        
        # Adjust array manifold for spatial smoothing
        if self.spatial_smoothing:
            P = self.noise_subspace.shape[0]
            array_manifold_adjusted = array_manifold[:P, :]
        else:
            array_manifold_adjusted = array_manifold
            
        # Compute pseudo-spectrum similar to MUSIC for visualization
        E_n = self.noise_subspace @ np.conj(self.noise_subspace).T
        denominator = np.real(np.diag(array_manifold_adjusted.conj().T @ E_n @ array_manifold_adjusted))
        return 1.0 / (denominator + 1e-12)  # Add small epsilon to avoid division by zero
        
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
        if self.noise_subspace is None:
            self._compute_noise_subspace(covariance_matrix)
            
        # Compute pseudo-spectrum value similar to MUSIC
        denominator = steering_vector.conj().T @ self.noise_subspace @ self.noise_subspace.conj().T @ steering_vector
        return 1.0 / (np.real(denominator) + 1e-12)
    
    def _compute_noise_subspace(self, covariance_matrix: np.ndarray):
        """
        Compute the noise subspace from the covariance matrix.
        
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
        
        # Get noise subspace
        self.noise_subspace = eigenvectors[:, self.num_sources:]
    
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
        Estimate DOA using the RootMUSIC method.
        
        Returns:
            tuple: (estimated_angles, spectrum) where:
                - estimated_angles: Array of estimated DOA angles in degrees
                - spectrum: Pseudo-spectrum values for visualization
        """
        if self.received_data is None and self.received_covariance is None:
            raise ValueError("No received data or covariance matrix available. "
                           "Call set_received_data or set_received_covariance first.")
            
        # Check if array is suitable for RootMUSIC (should be linear)
        if self.array_model.config.array_type != 'linear':
            raise ValueError("RootMUSIC is only applicable to linear arrays (ULA). "
                           f"Current array type: {self.array_model.config.array_type}")
            
        # Reset cached noise subspace
        self.noise_subspace = None
        
        M = self.num_sources
        
        # Get covariance matrix (either provided directly or estimated from data)
        covariance_matrix = self._estimate_covariance_matrix()
        
        # Compute noise subspace
        self._compute_noise_subspace(covariance_matrix)
        
        # Generate hermitian noise subspace matrix
        F = self.noise_subspace @ np.conj(self.noise_subspace).T
        
        # Calculate the sum of F matrix diagonals
        coefficients = self._sum_of_diag(F)
        
        # Calculate the roots of the polynomial defined by F matrix diagonals
        roots = list(self._find_roots(coefficients))
        
        # Sort roots by their distance from the unit circle
        roots.sort(key=lambda x: abs(abs(x) - 1))
        
        doa_predictions_all, roots_angels_all = self._extract_predictions_from_roots(
            roots
        )
        # Take only roots which inside the unit circle
        roots_inside = [root for root in roots if ((abs(root) - 1) < 0)][:M]
        # Calculate DoA out of the roots inside the unit circle
        doa_predictions, _ = self._extract_predictions_from_roots(roots_inside)
        
        # Compute pseudo-spectrum for visualization compatibility
        angles_deg, spectrum = self.compute_spatial_spectrum()
        
        # Store selected roots for visualization
        self.selected_roots = np.array(roots_inside)
        
        return doa_predictions, roots