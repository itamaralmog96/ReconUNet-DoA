from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List

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

from scipy.signal import find_peaks

class BaseDOAModel(ABC):
    """
    Base class for all classic Direction of Arrival (DOA) estimation models.
    This class defines the common interface and functionality that all classic DOA models should implement.
    """
    
    def __init__(self, 
                 array_model: ArrayModel,
                 array_manifold: Optional[np.ndarray] = None,
                 scan_angles_deg: Optional[np.ndarray] = None,
                 num_sources: Optional[int] = None,
                 use_parabolic_interpolation: bool = True):
        """
        Initialize the base DOA model.
        
        Args:
            array_model (ArrayModel): Array model instance containing array geometry and characteristics
            array_manifold (np.ndarray, optional): Pre-computed array manifold matrix (num_elements, num_angles).
                                                  If None, will be computed for scan_angles_deg.
            scan_angles_deg (np.ndarray, optional): Scan angles in degrees. If None, defaults to 0-359 degrees.
            num_sources (int, optional): Number of sources to detect. If None, will return all detected peaks.
            use_parabolic_interpolation (bool, optional): If True, use parabolic interpolation for sub-grid peak accuracy.
                                                         Defaults to True.
        """
        # Store the array model
        self.array_model = array_model
        
        # Store number of sources
        self.num_sources = num_sources
        
        # Store parabolic interpolation setting
        self.use_parabolic_interpolation = use_parabolic_interpolation
        
        # Initialize scan angles (0 to 359 degrees by default)
        if scan_angles_deg is None:
            self.scan_angles_deg = np.arange(0, 360)
        else:
            self.scan_angles_deg = scan_angles_deg
        self.scan_angles_rad = np.deg2rad(self.scan_angles_deg)
        
        # Set or compute array manifold
        if array_manifold is not None:
            # Use provided array manifold
            if array_manifold.shape[1] != len(self.scan_angles_deg):
                raise ValueError(f"Array manifold shape {array_manifold.shape} does not match "
                               f"scan angles length {len(self.scan_angles_deg)}")
            self.A = array_manifold
        else:
            # Compute array manifold for scan angles
            self.A = self.array_model.steering_matrix(self.scan_angles_deg, nominal=True)
        
        # Store dimensions
        self.num_elements = self.A.shape[0]
        self.num_angles = self.A.shape[1]
        
        # Initialize received data and covariance matrix
        self.received_data = None
        self.received_covariance = None
        
    def set_received_data(self, data: np.ndarray):
        """
        Set the received data for DOA estimation.
        
        Args:
            data (np.ndarray): Received signal matrix of shape (num_elements, num_snapshots)
        """
        if data.shape[0] != self.num_elements:
            raise ValueError(f"Received data must have {self.num_elements} elements to match array model")
        self.received_data = data
        # Clear any previously set covariance matrix since we now have new data
        self.received_covariance = None
        
    def set_received_covariance(self, covariance_matrix: np.ndarray):
        """
        Set the received covariance matrix directly for DOA estimation.
        This bypasses the need to estimate the covariance from received data.
        
        Args:
            covariance_matrix (np.ndarray): Covariance matrix of shape (num_elements, num_elements)
        """
        if covariance_matrix.shape != (self.num_elements, self.num_elements):
            raise ValueError(f"Covariance matrix must have shape ({self.num_elements}, {self.num_elements}) "
                           f"to match array model, got {covariance_matrix.shape}")
        if not np.allclose(covariance_matrix, covariance_matrix.conj().T):
            raise ValueError("Covariance matrix must be Hermitian (or symmetric for real matrices)")
        self.received_covariance = covariance_matrix
        
    def _calculate_steering_vector(self, angle_rad: float | np.ndarray) -> np.ndarray:
        """
        Calculate the steering vector(s) for given angle(s).
        
        Args:
            angle_rad (float or np.ndarray): Angle(s) in radians
            
        Returns:
            np.ndarray: Steering vector(s) for the given angle(s)
        """
        # Convert to degrees for ArrayModel's steering_vector method
        angle_deg = np.rad2deg(angle_rad)
        # Use ArrayModel's steering_vector method which handles both single angles and arrays
        if np.isscalar(angle_deg):
            return self.array_model.steering_vector(angle_deg, nominal=True)
        else:
            return self.array_model.steering_matrix(angle_deg, nominal=True)
        
    def _estimate_covariance_matrix(self) -> np.ndarray:
        """
        Get the covariance matrix - either from directly set covariance or estimate from received data.
        
        Returns:
            np.ndarray: Covariance matrix
        """
        # If covariance matrix was directly provided, use it
        if self.received_covariance is not None:
            return self.received_covariance
            
        # Otherwise, estimate from received data
        if self.received_data is None:
            raise ValueError("No received data or covariance matrix available. "
                           "Call set_received_data or set_received_covariance first.")
            
        # Compute sample covariance matrix
        R = np.cov(self.received_data)
        return R
        
    def compute_spatial_spectrum(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the spatial spectrum for all scan angles.
            
        Returns:
            tuple: (angles_deg, spectrum) where:
                - angles_deg: Array of scan angles in degrees
                - spectrum: Spatial spectrum values for each angle
        """
        if self.received_data is None and self.received_covariance is None:
            raise ValueError("No received data or covariance matrix available. "
                           "Call set_received_data or set_received_covariance first.")
            
        # Get covariance matrix (either provided directly or estimated from data)
        R = self._estimate_covariance_matrix()
        
        # Compute spectrum values for all angles at once using pre-computed array manifold
        spectrum = self._compute_spectrum_values(self.A, R)
            
        return self.scan_angles_deg, spectrum
    
    def _compute_spectrum_values(self, steering_vectors: np.ndarray, covariance_matrix: np.ndarray) -> np.ndarray:
        """
        Compute the spectrum values for all steering vectors at once.
        This method should be implemented by derived classes.
        
        Args:
            steering_vectors (np.ndarray): Steering vectors for all angles (num_elements, num_angles)
            covariance_matrix (np.ndarray): Estimated covariance matrix
            
        Returns:
            np.ndarray: Spectrum values for all angles
        """
        raise NotImplementedError("Subclasses must implement _compute_spectrum_values()")
        
    def _compute_spectrum_value(self, steering_vector: np.ndarray, covariance_matrix: np.ndarray) -> float:
        """
        Compute the spectrum value for a single steering vector.
        This method should be implemented by derived classes.
        
        Args:
            steering_vector (np.ndarray): Steering vector for the current angle
            covariance_matrix (np.ndarray): Estimated covariance matrix
            
        Returns:
            float: Spectrum value for the current angle
        """
        raise NotImplementedError("Subclasses must implement _compute_spectrum_value()")
        
    @abstractmethod
    def estimate_doa(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate the Direction of Arrival (DOA).
        This method should be implemented by derived classes.
        
        Returns:
            tuple: (estimated_angles, spectrum) where:
                - estimated_angles: Array of estimated DOA angles in degrees
                - spectrum: Spatial spectrum values
        """
        pass
    
    def plot_spectrum(self, spectrum, title="DOA Spectrum", estimated_doas=None):
        """
        Plot the spatial spectrum.
        
        Args:
            spectrum (np.ndarray): Spatial spectrum values
            title (str): Plot title
            estimated_doas (np.ndarray, optional): Array of estimated DOA angles to mark on the plot
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.scan_angles_deg, spectrum)
        plt.xlabel('Angle (degrees)')
        plt.ylabel('Spatial Spectrum')
        plt.title(title)
        plt.grid(True)
        
        if estimated_doas is not None:
            for doa in estimated_doas:
                plt.axvline(x=doa, color='r', linestyle='--', alpha=0.5)
                
        plt.show()

    def _find_peaks(self, 
                   spectrum: np.ndarray, 
                   height: Optional[float] = None,
                   threshold: Optional[float] = None,
                   distance: Optional[int] = 2,
                   prominence: Optional[float] = None,
                   plateau_size: Optional[int] = 1,
                   width: Optional[int] = None) -> np.ndarray:
                
        """
        Find peaks in the spatial spectrum using scipy.signal.find_peaks.
        Handles circular/periodic nature of angles properly.
        
        Args:
            spectrum (np.ndarray): Spatial spectrum
            height (float, optional): Required height of peaks
            threshold (float, optional): Required threshold (relative to neighboring samples)
            distance (int, optional): Required minimal horizontal distance between peaks
            prominence (float, optional): Required prominence of peaks
            plateau_size (int, optional): Required size of plateau of peaks
            width (int, optional): Required width of peaks
            
        Returns:
            np.ndarray: Indices of detected peaks (limited to top 3)
        """
        # Return early if spectrum is empty to avoid reduction errors
        if spectrum.size == 0:
            return np.array([], dtype=int)

        # Normalize spectrum (guard against all-zero spectrum to avoid division by zero)
        spectrum_max = np.max(spectrum)
        if spectrum_max == 0:
            normalized_spectrum = spectrum.copy()
        else:
            normalized_spectrum = spectrum / spectrum_max
        spectrum = normalized_spectrum
        
        # Set default parameters if not provided
        # if height is None:
        #     height = 0.5  # Default to 50% of maximum
        # if distance is None:
        #     distance = 5  # Default to 5 degrees minimum separation
        # if prominence is None:
        #     prominence = 0.1  # Default to 10% prominence
            
        # Handle circular nature of angles by extending spectrum at both ends
        # This allows peaks at 0° and 360° to be detected properly
        extension_length = max(distance if distance else 5, 10)  # Extend by at least 10 samples
        
        # Create extended spectrum: [end_part, original_spectrum, start_part]
        extended_spectrum = np.concatenate([
            spectrum[-extension_length:],  # Last part at the beginning
            spectrum,                      # Original spectrum
            spectrum[:extension_length]    # First part at the end
        ])
        
        # Find peaks in extended spectrum
        extended_peaks, _ = find_peaks(
            extended_spectrum,
            height=height,
            threshold=threshold,
            distance=distance,
            prominence=prominence,
            width=width,
            plateau_size=plateau_size
        )
        
        # Map extended peaks back to original spectrum indices
        original_peaks = []
        for peak in extended_peaks:
            if peak < extension_length:
                # Peak is in the left extension (corresponds to end of original spectrum)
                original_idx = len(spectrum) - extension_length + peak
            elif peak >= extension_length + len(spectrum):
                # Peak is in the right extension (corresponds to start of original spectrum)
                original_idx = peak - extension_length - len(spectrum)
            else:
                # Peak is in the original spectrum
                original_idx = peak - extension_length
                
            # Ensure index is within bounds and avoid duplicates
            if 0 <= original_idx < len(spectrum):
                if original_idx not in original_peaks:
                    original_peaks.append(original_idx)
        
        # Convert to numpy array and sort
        peaks = np.array(original_peaks, dtype=int)
        
        # # Remove peaks that are too close due to circular wrapping
        # if len(peaks) > 1:
        #     peaks_to_keep = []
        #     for i, peak in enumerate(peaks):
        #         keep_peak = True
        #         for j, other_peak in enumerate(peaks):
        #             if i != j:
        #                 # Calculate circular distance
        #                 circular_distance = min(
        #                     abs(peak - other_peak),
        #                     len(spectrum) - abs(peak - other_peak)
        #                 )
        #                 if circular_distance < (distance if distance else 5):
        #                     # Keep the peak with higher spectrum value
        #                     if spectrum[peak] <= spectrum[other_peak] and i > j:
        #                         keep_peak = False
        #                         break
        #         if keep_peak:
        #             peaks_to_keep.append(peak)
        #     peaks = np.array(peaks_to_keep, dtype=int)
        
        # Sort peaks by spectrum value (descending) and limit by num_sources if specified
        if len(peaks) > 0:
            # Sort peaks by their spectrum values in descending order
            peak_values = spectrum[peaks]
            sorted_indices = np.argsort(peak_values)[::-1]
            peaks = peaks[sorted_indices]
            
            # Limit to num_sources if specified, otherwise return top 3 for backward compatibility
            max_peaks = self.num_sources if self.num_sources is not None else 3
            peaks = peaks[:max_peaks]
            
            # Apply parabolic interpolation to refine peak locations
            if self.use_parabolic_interpolation:
                return self._parabolic_interpolation(spectrum, peaks)
        
        return peaks
    
    def _parabolic_interpolation(self, spectrum: np.ndarray, peak_indices: np.ndarray) -> np.ndarray:
        """
        Apply parabolic interpolation to refine peak locations for sub-grid angular resolution.
        
        Uses 3-point parabolic fit around each peak to achieve sub-degree accuracy.
        For peaks at boundaries or where interpolation is numerically unstable,
        returns the discrete peak location.
        
        Mathematical basis:
        Given 3 points around a peak: (θ_{i-1}, P_{i-1}), (θ_i, P_i), (θ_{i+1}, P_{i+1})
        Fit parabola: P(θ) = a(θ - θ_i)² + b(θ - θ_i) + c
        Refined peak location: θ_refined = θ_i + δ * Δθ
        where δ = 0.5 * (P_{i-1} - P_{i+1}) / (P_{i-1} - 2*P_i + P_{i+1})
        
        Args:
            spectrum (np.ndarray): Spatial spectrum values
            peak_indices (np.ndarray): Indices of peaks in the spectrum (integers)
        
        Returns:
            np.ndarray: Refined peak angles in degrees (can have fractional values)
        """
        refined_angles = []
        
        for idx in peak_indices:
            # Handle boundary cases - use discrete peak location
            # Cannot interpolate at boundaries (need 3 points centered on peak)
            if idx == 0 or idx == len(spectrum) - 1:
                refined_angles.append(self.scan_angles_deg[idx])
                continue
            
            # Extract 3-point neighborhood around the peak
            y_prev = spectrum[idx - 1]  # Left neighbor
            y_peak = spectrum[idx]       # Peak itself
            y_next = spectrum[idx + 1]   # Right neighbor
            
            # Parabolic interpolation formula
            # For parabola y = a(x - x_peak)² + b(x - x_peak) + c
            # The peak offset is: delta = 0.5 * (y_prev - y_next) / (y_prev - 2*y_peak + y_next)
            denominator = y_prev - 2.0 * y_peak + y_next
            
            # Check for numerical stability
            # If denominator is too small, the peak is too flat (nearly horizontal)
            if abs(denominator) < 1e-10:
                # Peak is too flat - use discrete location
                refined_angles.append(self.scan_angles_deg[idx])
                continue
            
            # Compute fractional offset from peak index
            delta = 0.5 * (y_prev - y_next) / denominator
            
            # Clamp delta to prevent unrealistic extrapolations
            # For well-behaved peaks, delta should be in [-0.5, 0.5]
            # We allow up to [-1.0, 1.0] for robustness
            delta = np.clip(delta, -1.0, 1.0)
            
            # Compute refined angle using linear interpolation
            # Assumes uniform angle spacing (typical for scan grids)
            angle_spacing = self.scan_angles_deg[idx] - self.scan_angles_deg[idx - 1]
            refined_angle = self.scan_angles_deg[idx] + delta * angle_spacing
            
            refined_angles.append(refined_angle)
        
        return np.array(refined_angles) 