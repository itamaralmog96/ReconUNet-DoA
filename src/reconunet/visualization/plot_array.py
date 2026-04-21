"""
Script to visualize Tri4Net array responses.
"""

from Tri4_array import SystemModel, SystemModelParams
from .plot_utils import plot_array_responses, plot_beampattern, plot_2d_beampattern
import matplotlib.pyplot as plt

def main():
    # Create system model with mismatches enabled
    p = SystemModelParams(
        gp_enable=True,  # Enable gain/phase mismatches
        mc_enable=True,  # Enable mutual coupling
        max_r=0.01       # Add some position jitter
    )
    sys = SystemModel(p, seed=0)
    
    # Plot both nominal and actual responses
    print("Plotting array responses...")
    
    # Plot responses with mismatches
    fig_perturbed = plot_array_responses(sys)
    plt.savefig('array_responses_perturbed.png', dpi=300, bbox_inches='tight')
    
    # Plot nominal responses (without mismatches)
    fig_nominal = plot_array_responses(sys, nominal=True)
    plt.savefig('array_responses_nominal.png', dpi=300, bbox_inches='tight')
    
    # Plot beampatterns
    print("Plotting array beampatterns...")
    
    # Plot beampattern with mismatches (both polar and Cartesian)
    fig_bp_perturbed_polar = plot_beampattern(sys, polar=True)
    plt.savefig('beampattern_perturbed_polar.png', dpi=300, bbox_inches='tight')
    
    fig_bp_perturbed_cart = plot_beampattern(sys, polar=False)
    plt.savefig('beampattern_perturbed_cartesian.png', dpi=300, bbox_inches='tight')
    
    # Plot nominal beampattern (both polar and Cartesian)
    fig_bp_nominal_polar = plot_beampattern(sys, nominal=True, polar=True)
    plt.savefig('beampattern_nominal_polar.png', dpi=300, bbox_inches='tight')
    
    fig_bp_nominal_cart = plot_beampattern(sys, nominal=True, polar=False)
    plt.savefig('beampattern_nominal_cartesian.png', dpi=300, bbox_inches='tight')
    
    # Plot 2D beampatterns
    print("Plotting 2D beampatterns...")
    
    # Plot 2D beampattern with mismatches
    fig_2d_perturbed = plot_2d_beampattern(sys)
    plt.savefig('beampattern_2d_perturbed.png', dpi=300, bbox_inches='tight')
    
    # Plot nominal 2D beampattern
    fig_2d_nominal = plot_2d_beampattern(sys, nominal=True)
    plt.savefig('beampattern_2d_nominal.png', dpi=300, bbox_inches='tight')
    
    print("Plots saved as:")
    print("- array_responses_perturbed.png")
    print("- array_responses_nominal.png")
    print("- beampattern_perturbed_polar.png")
    print("- beampattern_perturbed_cartesian.png")
    print("- beampattern_nominal_polar.png")
    print("- beampattern_nominal_cartesian.png")
    print("- beampattern_2d_perturbed.png")
    print("- beampattern_2d_nominal.png")

if __name__ == '__main__':
    main() 