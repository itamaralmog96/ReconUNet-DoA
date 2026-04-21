#!/usr/bin/env python3
"""
Generate Array Model Block Diagram for DOA Estimation

This script creates a professional diagram showing:
- Linear antenna array with impinging waves
- Angle theta measurement (DOA)
- Array imperfections and effects
- Signal processing flow

Usage:
    python create_array_model_diagram.py
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Arc
import numpy as np

# Configuration
OUTPUT_DIR = "architecture_diagrams"
OUTPUT_NAME = "array_model_diagram"
DPI = 300

# Color scheme
COLORS = {
    'antenna': '#87CEEB',      # Sky blue
    'rf': '#98FB98',           # Pale green
    'processing': '#FFB6C1',   # Light pink
    'wave': '#000000',         # Black
    'angle': '#FF4500',        # Orange red
    'effect': '#4169E1',       # Royal blue
    'text': '#000000',         # Black
}

def create_array_model_diagram():
    """Create the array model diagram with impinging waves and DOA measurement."""
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(-2, 18)
    ax.set_ylim(-1, 13)
    ax.axis('off')
    
    # # Title
    # ax.text(8, 12.5, 'Antenna Array Model for DOA Estimation', 
    #         ha='center', fontsize=18, fontweight='bold')
    # ax.text(8, 11.8, 'Linear Array with Impinging Waves and Array Imperfections', 
    #         ha='center', fontsize=14, style='italic', color='#555555')
    
    # === ANTENNA ARRAY ===
    num_antennas = 8
    antenna_spacing = 0.5  # λ/2 spacing
    array_start_x = 1.0
    array_y = 8.0
    
    # Draw antennas (triangular shapes pointing down)
    antennas = []
    for i in range(num_antennas):
        x = array_start_x + i * antenna_spacing * 2
        # Triangle pointing down (base at top, tip at bottom)
        triangle = mpatches.Polygon([(x, array_y - 0.3), (x-0.15, array_y), (x+0.15, array_y)], 
                                   closed=True, facecolor=COLORS['antenna'], 
                                   edgecolor='black', linewidth=1.5)
        ax.add_patch(triangle)
        antennas.append(x)
    
    # Array baseline (dashed horizontal line)
    ax.plot([array_start_x - 0.2, array_start_x + (num_antennas-1) * antenna_spacing * 2 + 0.2], 
            [array_y, array_y], 'k--', linewidth=2, alpha=0.7)
    
    # Dots will be added after RF modules are defined
    
    # === IMPINGING WAVES ===
    wave_start_x = 0.5
    wave_end_x = array_start_x + (num_antennas-1) * antenna_spacing * 2 + 0.5
    wave_y_start = 10.5
    wave_y_end = array_y + 0.5
    
    # Draw fewer, more separated wave lines
    num_waves = 2
    for i in range(num_waves):
        y_offset = i * 0.8  # More separation
        ax.plot([wave_start_x, wave_end_x], 
                [wave_y_start - y_offset, wave_y_end - y_offset], 
                'k-', linewidth=2.5, alpha=0.9)
        # Removed arrowheads for cleaner look
    
    # === DOA ANGLE MEASUREMENT ===
    # Reference line (normal to array)
    ref_x = array_start_x + (num_antennas-1) * antenna_spacing * 2 / 2  # Center of array
    ax.plot([ref_x, ref_x], [array_y, array_y + 1.5], 'k--', linewidth=2, alpha=0.7)
    
    # Wave vector k (direction of wave propagation)
    # Calculate the angle of the wave fronts
    wave_dx = wave_end_x - wave_start_x
    wave_dy = wave_y_end - wave_y_start
    wave_angle_rad = np.arctan2(wave_dy, wave_dx)
    
    # k vector should be perpendicular to wave fronts, pointing from source toward array
    # Add π/2 to get perpendicular direction, then reverse for downward pointing
    k_angle_rad = wave_angle_rad + np.pi/2
    k_length = 1.3
    
    # Shift the vector up slightly from the array (partial shift toward wave front)
    vertical_shift = 0.18  # Small upward shift
    horizontal_shift = 0.05  # Small rightward shift
    
    # Start point is away from array (at wave front)
    k_start_x = ref_x + k_length * np.cos(k_angle_rad) + horizontal_shift
    k_start_y = array_y + k_length * np.sin(k_angle_rad) + vertical_shift
    # End point is slightly above array origin
    k_end_x = ref_x + horizontal_shift
    k_end_y = array_y + vertical_shift
    
    # Convert k angle to degrees for later use
    k_angle = np.degrees(k_angle_rad)
    
    # Draw wave vector k (from wave front to array)
    ax.plot([k_start_x, k_end_x], [k_start_y, k_end_y], 'b-', linewidth=3, alpha=0.8)
    
    # Add arrowhead to k vector (pointing toward array/origin)
    ax.annotate('', xy=(k_end_x, k_end_y), 
               xytext=(k_end_x - 0.2*np.cos(k_angle_rad), 
                      k_end_y - 0.2*np.sin(k_angle_rad)),
               arrowprops=dict(arrowstyle='->', lw=3, color='blue'))
    
    # k vector label
    k_label_x = ref_x + k_length/2 * np.cos(k_angle_rad) + 0.3
    k_label_y = array_y + k_length/2 * np.sin(k_angle_rad) + 0.2
    ax.text(k_label_x, k_label_y, r'$\vec{k}$', 
            fontsize=16, fontweight='bold', color='blue',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                     edgecolor='blue', linewidth=2, alpha=0.9))
    
    # Lambda removed as requested
    
    # Angle arc - measured from positive x-axis (horizontal) to k vector
    angle_radius = 0.5
    angle_start = 0  # degrees (start from positive x-axis / horizontal)
    angle_end = np.degrees(k_angle_rad)   # degrees (DOA angle from horizontal)
    
    # Create angle arc from horizontal to k vector
    arc_center_x = ref_x
    arc = Arc((arc_center_x, array_y), 2*angle_radius, 2*angle_radius, 
              angle=0, theta1=angle_start, theta2=angle_end, 
              color=COLORS['angle'], linewidth=3)
    ax.add_patch(arc)
    
    # Angle label positioned at the arc midpoint
    mid_angle_rad = k_angle_rad / 2
    angle_label_x = arc_center_x + angle_radius * np.cos(mid_angle_rad)
    angle_label_y = array_y + angle_radius * np.sin(mid_angle_rad)
    ax.text(angle_label_x + 0.3, angle_label_y + 0.1, r'$\theta$ (DOA)', 
            fontsize=14, fontweight='bold', color=COLORS['angle'],
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                     edgecolor=COLORS['angle'], linewidth=2, alpha=0.9))
    
    # === RF MODULES ===
    rf_y = 6.0
    rf_width = 0.3
    rf_height = 0.6
    
    for i, x in enumerate(antennas):
        rf_rect = Rectangle((x - rf_width/2, rf_y - rf_height/2), 
                           rf_width, rf_height, 
                           facecolor=COLORS['rf'], edgecolor='black', linewidth=1.5)
        ax.add_patch(rf_rect)
        
        # RF text inside the box
        ax.text(x, rf_y, 'RF', ha='center', va='center', fontsize=8, fontweight='bold', color='white')
        
        # RF label below the box with new numbering
        if i < 4:
            rf_label = f'{i}'
        else:
            rf_label = f'N-{7-i+1}'
        ax.text(x, rf_y - 1.0, rf_label, ha='center', fontsize=9, fontweight='bold')
        
        # Signal flow arrows (dashed and transparent)
        ax.annotate('', xy=(x, rf_y + rf_height/2), xytext=(x, array_y),
                   arrowprops=dict(arrowstyle='->', lw=2, color='black', 
                                 linestyle='--', alpha=0.6))
    
    # Add three dots between RF 4 and RF N-3 (between green boxes) - moved right with smaller spacing
    dot_x = array_start_x + 3.35 * antenna_spacing * 2  # Moved right from 3.2 to 3.4
    # Three dots with smaller spacing between them
    for i in range(3):
        dot_position = dot_x + i * (antenna_spacing * 2) / 6  # Reduced spacing from /3 to /6
        ax.plot(dot_position, rf_y, 'ko', markersize=3, alpha=0.7)
    
    # === BASEBAND PROCESSING ===
    processing_x = array_start_x + (num_antennas-1) * antenna_spacing * 2 / 2
    processing_y = 3.5
    processing_width = 4.0
    processing_height = 1.0
    
    # Use FancyBboxPatch for rounded corners
    from matplotlib.patches import FancyBboxPatch
    processing_rect = FancyBboxPatch((processing_x - processing_width/2, processing_y - processing_height/2),
                                   processing_width, processing_height,
                                   boxstyle="round,pad=0.1",
                                   facecolor=COLORS['processing'], edgecolor='black', linewidth=2)
    ax.add_patch(processing_rect)
    
    ax.text(processing_x, processing_y, 'Baseband Signal\nProcessing', 
            ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Arrows from RF to processing (all pointing to center, dashed and transparent)
    for i, x in enumerate(antennas):
        ax.annotate('', xy=(processing_x, processing_y + processing_height/2),
                   xytext=(x, rf_y - rf_height/2),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='black', 
                                 linestyle='--', alpha=0.6))
    
    # === OUTPUT ===
    output_y = 1.5
    ax.text(processing_x, output_y, 'Estimated DOA', 
            ha='center', fontsize=14, fontweight='bold', color=COLORS['text'])
    
    # Arrow from processing to output
    ax.annotate('', xy=(processing_x, output_y + 0.3), 
               xytext=(processing_x, processing_y - processing_height/2),
               arrowprops=dict(arrowstyle='->', lw=3, color='black'))
    
    # === ARRAY IMPERFECTIONS ===
    # (Removed position perturbation for cleaner diagram)
    
    # # === ARRAY PARAMETERS ANNOTATION ===
    # params_text = (
    #     f'Array Parameters:\n'
    #     f'• Number of elements: {num_antennas}\n'
    #     f'• Element spacing: λ/2\n'
    #     f'• Array type: Uniform Linear Array (ULA)\n'
    #     f'• DOA range: -90° to +90°\n'
    #     f'• Wavelength: λ = c/f'
    # )
    
    # ax.text(14, 9, params_text, ha='left', va='top', fontsize=10, 
    #         bbox=dict(boxstyle='round,pad=0.5', facecolor='#F8F9FA', 
    #                  edgecolor='#333333', linewidth=1.5, alpha=0.95))
    
    # # === SIGNAL FLOW ANNOTATION ===
    # flow_text = (
    #     'Signal Flow:\n'
    #     '1. Impinging waves arrive at array\n'
    #     '2. Each antenna receives signal\n'
    #     '3. RF modules process signals\n'
    #     '4. Baseband processing estimates DOA\n'
    #     '5. Output: Estimated angle θ'
    # )
    
    # ax.text(14, 5, flow_text, ha='left', va='top', fontsize=10,
    #         bbox=dict(boxstyle='round,pad=0.5', facecolor='#F0F8FF', 
    #                  edgecolor='#333333', linewidth=1.5, alpha=0.95))
    
    plt.tight_layout()
    return fig

def save_figure(fig, filename, formats=['pdf', 'png', 'svg', 'eps']):
    """Save figure in multiple formats."""
    from pathlib import Path
    
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    
    saved_files = []
    for fmt in formats:
        filepath = output_path / f"{filename}.{fmt}"
        
        if fmt == 'png':
            fig.savefig(filepath, dpi=DPI, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
        elif fmt == 'pdf':
            fig.savefig(filepath, format='pdf', bbox_inches='tight')
        elif fmt == 'svg':
            fig.savefig(filepath, format='svg', bbox_inches='tight')
        elif fmt == 'eps':
            fig.savefig(filepath, format='eps', bbox_inches='tight')
        
        saved_files.append(filepath)
        print(f"  ✅ Saved: {filepath}")
    
    return saved_files

def main():
    """Main execution function."""
    print("=" * 80)
    print("🎯 Generating Antenna Array Model Diagram")
    print("=" * 80)
    
    # Create the diagram
    print("\n📊 Creating array model diagram...")
    fig = create_array_model_diagram()
    files = save_figure(fig, OUTPUT_NAME)
    
    # Summary
    print("\n" + "=" * 80)
    print("✅ Array model diagram generated successfully!")
    print("\n📁 Output directory:", OUTPUT_DIR)
    print("\n📄 Generated files:")
    for f in files:
        print(f"    • {f.name}")
    
    print("\n💡 Diagram features:")
    print("  • 8-element linear antenna array")
    print("  • Impinging waves with DOA angle θ")
    print("  • RF processing modules")
    print("  • Baseband signal processing")
    print("  • Array imperfections (coupling, perturbations, etc.)")
    print("  • Professional color scheme")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
