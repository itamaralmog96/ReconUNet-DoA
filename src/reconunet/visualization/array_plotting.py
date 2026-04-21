import scipy.signal
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

# Add the src directory to the path for imports
script_dir = Path(__file__).parent  # src/visualization/
src_dir = script_dir.parent  # src/
sys.path.insert(0, str(src_dir))

from signalgen.signal_generator import SignalConfig, SignalGenerator
from signalgen.array_processing import ArrayConfig, ArrayModel


def plot_array_geometries(arrays, figsize=(8, 12)):
    """
    Visualize array geometries for one or more array models
    
    Args:
        arrays: ArrayModel object, list of ArrayModel objects, or dict of {name: ArrayModel}
        figsize: Tuple of (width, height) for the figure
    """
    # Handle different input types
    if isinstance(arrays, ArrayModel):
        # Single array model
        arrays_dict = {'Array': arrays}
    elif isinstance(arrays, list):
        # List of array models
        arrays_dict = {f'Array_{i+1}': array for i, array in enumerate(arrays)}
    elif isinstance(arrays, dict):
        # Dictionary of {name: array_model}
        arrays_dict = arrays
    else:
        raise ValueError("Input must be ArrayModel, list of ArrayModel, or dict of {name: ArrayModel}")
    
    num_arrays = len(arrays_dict)
    fig, axes = plt.subplots(num_arrays, 1, figsize=(8, 3*num_arrays))
    
    # Handle single array case
    if num_arrays == 1:
        axes = [axes]
    
    for idx, (name, array) in enumerate(arrays_dict.items()):
        ax = axes[idx]
        positions = array.element_positions
        
        # Plot element positions
        ax.scatter(positions[:, 0], positions[:, 1], s=80, c='red', alpha=0.7, edgecolors='black')
        
        # Label elements
        for i, (x, y) in enumerate(positions):
            ax.annotate(f'{i}', (x, y), xytext=(3, 3), textcoords='offset points', fontsize=9)
        
        # Set equal aspect ratio and labels
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X position (m)')
        ax.set_ylabel('Y position (m)')
        ax.set_title(f'{name.capitalize()} Array\n({array.config.num_elements} elements)', fontsize=11)
        
        # Add wavelength reference circle
        circle = plt.Circle((0, 0), array.wavelength, fill=False, linestyle='--', alpha=0.5, color='blue')
        ax.add_patch(circle)
        ax.text(0.7*array.wavelength, 0.7*array.wavelength, '1λ', color='blue', alpha=0.7, fontsize=9)

    plt.tight_layout()
    plt.show()
    
def plot_steering_patterns(array, angles, title="Steering Patterns"):
    """
    Plot ideal vs realistic steering matrix patterns for an array
    
    Args:
        array: ArrayModel object
        angles: Array of angles in degrees
        title: Title for the plot
    """
    A_nominal = array.steering_matrix(angles, nominal=True)
    A = array.steering_matrix(angles, nominal=False)

    # Plot magnitude and phase patterns with smaller figure size
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    fig.suptitle(title, fontsize=12, fontweight='bold')

    # Magnitude patterns
    for elem in range(len(array.element_positions)):
        axes[0, 0].plot(angles, np.abs(A_nominal[elem, :]), '--', alpha=0.7, label=f'Ideal {elem}')
        axes[0, 1].plot(angles, np.abs(A[elem, :]), '-', alpha=0.8, label=f'Real {elem}')

    axes[0, 0].set_title('Ideal Magnitudes', fontsize=10)
    axes[0, 0].set_xlabel('Angle (deg)', fontsize=8)
    axes[0, 0].set_ylabel('Magnitude', fontsize=8)
    axes[0, 0].legend(fontsize=7)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_title('Realistic Magnitudes\n(with imperfections)', fontsize=10)
    axes[0, 1].set_xlabel('Angle (deg)', fontsize=8)
    axes[0, 1].set_ylabel('Magnitude', fontsize=8)
    axes[0, 1].legend(fontsize=7)
    axes[0, 1].grid(True, alpha=0.3)

    # Phase patterns
    for elem in range(len(array.element_positions)):
        axes[1, 0].plot(angles, np.angle(A_nominal[elem, :]), '--', alpha=0.7, label=f'Ideal {elem}')
        axes[1, 1].plot(angles, np.angle(A[elem, :]), '-', alpha=0.8, label=f'Real {elem}')

    axes[1, 0].set_title('Ideal Phases', fontsize=10)
    axes[1, 0].set_xlabel('Angle (deg)', fontsize=8)
    axes[1, 0].set_ylabel('Phase (rad)', fontsize=8)
    axes[1, 0].legend(fontsize=7)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_title('Realistic Phases\n(with imperfections)', fontsize=10)
    axes[1, 1].set_xlabel('Angle (deg)', fontsize=8)
    axes[1, 1].set_ylabel('Phase (rad)', fontsize=8)
    axes[1, 1].legend(fontsize=7)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    
def plot_2d_beampattern(A, X, angles, nominal=True, title=None):
    """
    Plot 2D beampattern for given steering matrix and angles
    
    Args:
        A: Steering matrix of shape (N, K) where N is number of elements, K is number of angles
        angles: Array of angles in degrees
        nominal: Boolean indicating if this is nominal (ideal) or realistic pattern
        title: Optional title for the plot
    """
    # Get conjugate transpose of steering vectors
    AH = A.conj().T

    # Optimized calculation of 2D beampattern
    # Instead of nested loops, we use matrix operations
    N = A.shape[0]  # number of elements

    # Calculate denominator term (normalization factor)
    # This is the same for all angles: aH @ a
    denom = np.diag(AH @ A)  # shape: (K,)

    # Calculate numerator term
    # For each steering direction, calculate the response to all possible signal directions
    BP = np.zeros((len(angles), len(angles)), dtype=complex)

    for i in range(len(angles)):
        # Get steering vector for this direction
        x = X[:, i:i+1]  # shape: (N, 1)
        xH = x.conj().T  # shape: (1, N)
        
        # Calculate covariance matrix for this steering direction
        R = x @ xH  # shape: (N, N)
        
        # Calculate response to all possible signal directions
        # This is equivalent to: aH[j] @ R @ a[j] for each j
        # We can do this for all j at once using matrix multiplication
        BP[i, :] = np.diag(AH @ R @ A) / (denom * R.trace())

    # Convert to dB and clip
    # Handle zero and negative values properly
    BP_real = np.real(BP)
    BP_real[BP_real <= 0] = 1e-10  # Replace zero/negative values with small positive number
    BP_db = 10 * np.log10(BP_real)
    min_db = -15
    BP_db[BP_db < min_db] = min_db

    # Create figure with smaller size to fit screen better
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)

    # Plot 2D beampattern
    im = ax.contourf(angles, angles, BP_db, 20, cmap='jet')
    plt.colorbar(im, label='Magnitude [dB]')

    # Set labels and title
    ax.set_xlabel('Signal Direction [degrees]')
    ax.set_ylabel('Steering Direction [degrees]')
    
    if title is None:
        title = 'Nominal 2D Beampattern' if nominal else '2D Beampattern with Mismatches'
    ax.set_title(title)

    # Adjust layout to prevent cutoff
    plt.tight_layout()
    plt.show()
    
    return BP_db


def plot_vertical_cut_and_calculate_3db_width(angles, BP_db, target_angle, title_suffix=""):
    """
    Plot vertical cut of beampattern at specified angle and calculate 3dB width.
    
    Parameters:
    angles: array of angles
    BP_db: 2D beampattern array
    target_angle: angle to plot vertical cut (degrees)
    title_suffix: additional text for title
    """
    # Find closest angle index
    angle_idx = np.argmin(np.abs(angles - target_angle))
    actual_angle = angles[angle_idx]
    
    # Extract vertical cut (steering direction vs magnitude)
    vertical_cut = BP_db[:, angle_idx]
    
    # Find peak
    peak_idx = np.argmax(vertical_cut)
    peak_angle = angles[peak_idx]
    peak_magnitude = vertical_cut[peak_idx]
    
    # Calculate 3dB down point
    threshold_3db = peak_magnitude - 3
    
    # Find points where magnitude crosses 3dB threshold
    # Look for crossings in both directions from peak
    left_3db_idx = None
    right_3db_idx = None
    
    # Search left from peak
    for i in range(peak_idx, -1, -1):
        if vertical_cut[i] <= threshold_3db:
            left_3db_idx = i
            break
    
    # Search right from peak
    for i in range(peak_idx, len(vertical_cut)):
        if vertical_cut[i] <= threshold_3db:
            right_3db_idx = i
            break
    
    # Calculate 3dB width
    if left_3db_idx is not None and right_3db_idx is not None:
        left_angle = angles[left_3db_idx]
        right_angle = angles[right_3db_idx]
        width_3db = right_angle - left_angle
    else:
        width_3db = None
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot vertical cut
    ax.plot(angles, vertical_cut, 'b-', linewidth=2, label=f'Vertical cut at {actual_angle:.1f}°')
    
    # Mark peak
    ax.plot(peak_angle, peak_magnitude, 'ro', markersize=8, label=f'Peak: {peak_angle:.1f}°')
    
    # Mark 3dB points if found
    if left_3db_idx is not None:
        ax.plot(angles[left_3db_idx], vertical_cut[left_3db_idx], 'gs', markersize=6, 
                label=f'Left 3dB: {angles[left_3db_idx]:.1f}°')
    if right_3db_idx is not None:
        ax.plot(angles[right_3db_idx], vertical_cut[right_3db_idx], 'gs', markersize=6, 
                label=f'Right 3dB: {angles[right_3db_idx]:.1f}°')
    
    # Plot 3dB threshold line
    if width_3db is not None:
        ax.axhline(y=threshold_3db, color='r', linestyle='--', alpha=0.7, 
                   label=f'3dB threshold: {threshold_3db:.1f} dB')
    
    # Add width annotation
    if width_3db is not None:
        ax.text(0.02, 0.98, f'3dB Width: {width_3db:.1f}°', 
                transform=ax.transAxes, fontsize=12, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_xlabel('Steering Direction [degrees]')
    ax.set_ylabel('Magnitude [dB]')
    ax.set_title(f'Vertical Cut of Beampattern at {actual_angle:.1f}°{title_suffix}')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print results
    print(f"Vertical cut at angle: {actual_angle:.1f}°")
    print(f"Peak found at: {peak_angle:.1f}° (magnitude: {peak_magnitude:.1f} dB)")
    if width_3db is not None:
        print(f"3dB width: {width_3db:.1f}°")
        print(f"3dB points: {angles[left_3db_idx]:.1f}° to {angles[right_3db_idx]:.1f}°")
    else:
        print("Could not determine 3dB width (threshold not crossed)")
    
    return width_3db, peak_angle, peak_magnitude

if __name__ == "__main__":
    array = ArrayModel(ArrayConfig(array_type="linear", num_elements=8))
    angles = np.arange(0, 180, 1)
    plot_steering_patterns(array, angles)
    BP = plot_2d_beampattern(array.steering_matrix(angles, nominal=True), array.steering_matrix(angles, nominal=True), angles)
    plot_vertical_cut_and_calculate_3db_width(angles, BP, 90)
    
