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

class MUSIC(BaseDOAModel):
    """
    MUltiple SIgnal Classification (MUSIC) DOA estimation model.
    This method uses the orthogonality between signal and noise subspaces
    to achieve high-resolution DOA estimation.
    """
    
    def __init__(self, 
                 array_model: ArrayModel,
                 array_manifold: Optional[np.ndarray] = None,
                 scan_angles_deg: Optional[np.ndarray] = None,
                 num_sources: Optional[int] = None,
                 spatial_smoothing: bool = False,
                 subarray_length: Optional[int] = None):
        """
        Initialize the MUSIC model.
        
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
        
    def _compute_spectrum_values(self, array_manifold: np.ndarray, covariance_matrix: np.ndarray) -> np.ndarray:
        """
        Compute the MUSIC spectrum values for all steering vectors at once.
        
        Args:
            array_manifold (np.ndarray): Steering vectors for all angles (num_elements, num_angles)
            covariance_matrix (np.ndarray): Estimated covariance matrix
            
        Returns:
            np.ndarray: MUSIC spectrum values for all angles
        """
        # Apply spatial smoothing if enabled
        if self.spatial_smoothing:
            R_input = self._apply_spatial_smoothing(covariance_matrix)
            # Adjust array manifold for smoothed case
            P = R_input.shape[0]
            array_manifold_smoothed = array_manifold[:P, :]
        else:
            R_input = covariance_matrix
            array_manifold_smoothed = array_manifold
            
        # Compute noise subspace if not already computed
        if self.noise_subspace is None:
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
            
        # Compute MUSIC spectrum for all angles at once
        # 1 / (A^H @ E_n @ E_n^H @ A) where A is (num_elements, num_angles)
        E_n = self.noise_subspace @ np.conj(self.noise_subspace).T
        denominator = np.real(np.diag(array_manifold_smoothed.conj().T @ E_n @ array_manifold_smoothed))
        return 1.0 / denominator
        
    def _compute_spectrum_value(self, steering_vector: np.ndarray, covariance_matrix: np.ndarray) -> float:
        """
        Compute the MUSIC spectrum value for a single steering vector.
        
        Args:
            steering_vector (np.ndarray): Steering vector for the current angle
            covariance_matrix (np.ndarray): Estimated covariance matrix
            
        Returns:
            float: MUSIC spectrum value
        """
        # Apply spatial smoothing if enabled
        if self.spatial_smoothing:
            R_input = self._apply_spatial_smoothing(covariance_matrix)
            # Adjust steering vector for smoothed case
            P = R_input.shape[0]
            steering_vector_smoothed = steering_vector[:P]
        else:
            R_input = covariance_matrix
            steering_vector_smoothed = steering_vector
            
        # Compute noise subspace if not already computed
        if self.noise_subspace is None:
            # Perform eigenvalue decomposition
            eigenvalues, eigenvectors = np.linalg.eigh(R_input)
            # Normalize eigenvectors to unit length
            eigenvectors = eigenvectors / np.linalg.norm(eigenvectors, axis=0, keepdims=True)
            
            # Sort eigenvalues and eigenvectors in descending order
            idx = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # Estimate number of sources if not provided
            if self.num_sources is None:
                self.num_sources = self._estimate_num_sources(eigenvalues)
            
            # Get noise subspace
            self.noise_subspace = eigenvectors[:, self.num_sources:]
            
        # Compute MUSIC spectrum
        denominator = steering_vector_smoothed.conj().T @ self.noise_subspace @ self.noise_subspace.conj().T @ steering_vector_smoothed
        return 1.0 / np.real(denominator)
        
    def estimate_doa(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate DOA using the MUSIC method.
        
        Returns:
            tuple: (estimated_angles, spectrum) where:
                - estimated_angles: Array of estimated DOA angles in degrees
                - spectrum: Spatial spectrum values
        """
        # Reset cached noise subspace
        self.noise_subspace = None
        
        # Compute spatial spectrum
        angles_deg, spectrum = self.compute_spatial_spectrum()
        
        # Find peaks in the spectrum (returns angles directly with interpolation)
        estimated_angles = self._find_peaks(spectrum)
        
        return estimated_angles, spectrum
    
    def _estimate_num_sources(self, eigenvalues: np.ndarray) -> int:
        """
        Estimate the number of sources using multiple criteria with fallbacks.
        
        Args:
            eigenvalues (np.ndarray): Eigenvalues of the covariance matrix (sorted descending)
            
        Returns:
            int: Estimated number of sources
        """
        n = len(eigenvalues)
        
        # Method 1: Try MDL criterion
        try:
            mdl = np.zeros(n)
            for k in range(n):
                if k == n:  # All eigenvalues are noise
                    mdl[k] = np.inf
                else:
                    # Compute geometric mean of noise eigenvalues
                    noise_eigenvals = eigenvalues[k:]
                    if len(noise_eigenvals) > 0 and np.all(noise_eigenvals > 0):
                        geometric_mean = np.prod(noise_eigenvals) ** (1.0 / len(noise_eigenvals))
                        arithmetic_mean = np.mean(noise_eigenvals)
                        if geometric_mean > 0 and arithmetic_mean > 0:
                            mdl[k] = -np.log(geometric_mean / arithmetic_mean) * (n - k) + 0.5 * k * (2 * n - k) * np.log(n)
                        else:
                            mdl[k] = np.inf
                    else:
                        mdl[k] = np.inf
            
            num_sources_mdl = np.argmin(mdl)
        except:
            num_sources_mdl = 0
            
        # Method 2: Eigenvalue ratio test (simple but effective)
        # Look for significant drops in eigenvalue ratios
        num_sources_ratio = 0
        if n > 1:
            ratios = eigenvalues[:-1] / eigenvalues[1:]  # Ratio of consecutive eigenvalues
            # Find the largest ratio (biggest drop)
            max_ratio_idx = np.argmax(ratios)
            # If the ratio is significant (>2), consider it a signal/noise boundary
            if ratios[max_ratio_idx] > 2.0:
                num_sources_ratio = max_ratio_idx + 1
        
        # Method 3: Simple threshold-based approach
        # Assume sources have eigenvalues significantly larger than noise floor
        eigenvalue_threshold = np.mean(eigenvalues) + 0.5 * np.std(eigenvalues)
        num_sources_threshold = np.sum(eigenvalues > eigenvalue_threshold)
        
        # Method 4: Use AIC criterion as backup
        try:
            aic = np.zeros(n)
            for k in range(n):
                if k == n:
                    aic[k] = np.inf
                else:
                    noise_eigenvals = eigenvalues[k:]
                    if len(noise_eigenvals) > 0 and np.all(noise_eigenvals > 0):
                        geometric_mean = np.prod(noise_eigenvals) ** (1.0 / len(noise_eigenvals))
                        arithmetic_mean = np.mean(noise_eigenvals)
                        if geometric_mean > 0 and arithmetic_mean > 0:
                            aic[k] = -2 * np.log(geometric_mean / arithmetic_mean) * (n - k) + 2 * k
                        else:
                            aic[k] = np.inf
                    else:
                        aic[k] = np.inf
            num_sources_aic = np.argmin(aic)
        except:
            num_sources_aic = 0
        
        # Combine estimates with preference order
        candidates = [num_sources_mdl, num_sources_aic, num_sources_ratio, num_sources_threshold]
        
        # Remove invalid candidates (0 or >= n)
        valid_candidates = [c for c in candidates if 0 < c < n]
        
        if valid_candidates:
            # Use the most common estimate, or the smallest if tied
            from collections import Counter
            counts = Counter(valid_candidates)
            most_common = counts.most_common(1)[0][0]
            final_estimate = most_common
        else:
            # Fallback: assume at least 1 source, but no more than n-1
            final_estimate = min(1, n-1)
        
        # Additional safety check: ensure we don't estimate more sources than elements-1
        final_estimate = min(final_estimate, n-1)
        
        return final_estimate 