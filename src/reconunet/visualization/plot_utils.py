"""
Plotting utilities for Tri4Net array visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from Tri4_array import SystemModel, SystemModelParams

def plot_array_responses(sys: SystemModel, 
                        angles: Optional[np.ndarray] = None,
                        nominal: bool = False,
                        figsize: tuple[int, int] = (12, 8)) -> None:
    """
    Plot the phase and magnitude of all array elements in a single figure.
    
    Parameters
    ----------
    sys : SystemModel
        The system model instance
    angles : Optional[np.ndarray]
        Array of angles to plot. If None, uses full 0-359° sweep
    nominal : bool
        If True, plot nominal (ideal) responses without mismatches
    figsize : tuple[int, int]
        Figure size in inches (width, height)
    """
    # Get steering vectors
    A = sys.steering_vec(angles, nominal=nominal)
    
    # Convert to phases (in degrees) and magnitudes
    phases = np.angle(A, deg=True)
    magnitudes = np.abs(A)
    
    # Create angles array if not provided
    if angles is None:
        angles = np.arange(360)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    # Plot phases
    for i in range(A.shape[0]):
        ax1.plot(angles, phases[i], label=f'Element {i}')
    ax1.set_xlabel('Angle [degrees]')
    ax1.set_ylabel('Phase [degrees]')
    ax1.set_title('Array Element Phase Responses')
    ax1.grid(True)
    ax1.legend()
    
    # Plot magnitudes
    for i in range(A.shape[0]):
        ax2.plot(angles, magnitudes[i], label=f'Element {i}')
    ax2.set_xlabel('Angle [degrees]')
    ax2.set_ylabel('Magnitude')
    ax2.set_title('Array Element Magnitude Responses')
    ax2.grid(True)
    ax2.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Add overall title
    title = 'Nominal Array Responses' if nominal else 'Array Responses with Mismatches'
    fig.suptitle(title, y=1.02)
    
    return fig

def plot_beampattern(sys: SystemModel,
                     angles: Optional[np.ndarray] = None,
                     nominal: bool = False,
                     figsize: tuple[int, int] = (10, 8),
                     polar: bool = True) -> None:
    """
    Plot the array beampattern (array factor).
    
    Parameters
    ----------
    sys : SystemModel
        The system model instance
    angles : Optional[np.ndarray]
        Array of angles to plot. If None, uses full 0-359° sweep
    nominal : bool
        If True, plot nominal (ideal) beampattern without mismatches
    figsize : tuple[int, int]
        Figure size in inches (width, height)
    polar : bool
        If True, plot in polar coordinates, otherwise in Cartesian
    """
    # Get steering vectors
    A = sys.steering_vec(angles, nominal=nominal)
    
    # Create angles array if not provided
    if angles is None:
        angles = np.arange(360)
    
    # Convert angles to radians for polar plot
    angles_rad = np.deg2rad(angles)
    
    # Calculate array factor (beampattern)
    # For a uniform array, we can use the sum of steering vectors
    array_factor = np.abs(np.sum(A, axis=0))
    
    # Normalize to 0 dB
    array_factor_db = 20 * np.log10(array_factor / np.max(array_factor))
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    
    if polar:
        # Polar plot
        ax = fig.add_subplot(111, projection='polar')
        ax.plot(angles_rad, array_factor_db)
        ax.set_theta_zero_location('N')  # Set 0 degrees to North
        ax.set_theta_direction(-1)  # Set clockwise direction
        ax.set_rlabel_position(0)  # Move radial labels to 0 degrees
        ax.set_ylim([-40, 0])  # Set dB range
        ax.grid(True)
        ax.set_title('Array Beampattern (Polar)')
    else:
        # Cartesian plot
        ax = fig.add_subplot(111)
        ax.plot(angles, array_factor_db)
        ax.set_xlabel('Angle [degrees]')
        ax.set_ylabel('Magnitude [dB]')
        ax.set_ylim([-40, 0])  # Set dB range
        ax.grid(True)
        ax.set_title('Array Beampattern (Cartesian)')
    
    # Add overall title
    title = 'Nominal Array Beampattern' if nominal else 'Array Beampattern with Mismatches'
    fig.suptitle(title, y=1.02)
    
    return fig

def plot_2d_beampattern(sys: SystemModel,
                        angles: Optional[np.ndarray] = None,
                        nominal: bool = False,
                        figsize: tuple[int, int] = (10, 8),
                        min_db: float = -15) -> None:
    """
    Plot the 2D beampattern showing array response to signals from different directions.
    This shows how the array responds to a signal from one direction when steering in another direction.
    
    Parameters
    ----------
    sys : SystemModel
        The system model instance
    angles : Optional[np.ndarray]
        Array of angles to plot. If None, uses full 0-359° sweep
    nominal : bool
        If True, plot nominal (ideal) beampattern without mismatches
    figsize : tuple[int, int]
        Figure size in inches (width, height)
    min_db : float
        Minimum dB value to display (clips values below this)
    """
    # Get steering vectors
    A = sys.steering_vec(angles, nominal=nominal)
    
    # Create angles array if not provided
    if angles is None:
        angles = np.arange(360)
    
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
        x = A[:, i:i+1]  # shape: (N, 1)
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
    BP_db[BP_db < min_db] = min_db
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    
    # Plot 2D beampattern
    im = ax.contourf(angles, angles, BP_db, 20, cmap='jet')
    plt.colorbar(im, label='Magnitude [dB]')
    
    # Set labels and title
    ax.set_xlabel('Signal Direction [degrees]')
    ax.set_ylabel('Steering Direction [degrees]')
    title = 'Nominal 2D Beampattern' if nominal else '2D Beampattern with Mismatches'
    ax.set_title(title)
    
    return fig

if __name__ == '__main__':
    # Example usage
    p = SystemModelParams(gp_enable=True, mc_enable=True)
    sys = SystemModel(p, seed=0)
    
    # Plot responses
    fig = plot_array_responses(sys)
    plt.show()
    
    # Plot beampattern
    fig = plot_beampattern(sys)
    plt.show()
    
    # Plot 2D beampattern
    fig = plot_2d_beampattern(sys)
    plt.show() 