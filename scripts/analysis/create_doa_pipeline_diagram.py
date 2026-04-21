#!/usr/bin/env python3
"""
Generate a Professional DOA Estimation Pipeline Block Diagram.

This script creates a high-quality, publication-ready diagram illustrating
the signal processing chain for Direction of Arrival (DOA) estimation.

The pipeline includes:
- Input Signal
- Preprocessing
- Multilag Autocorrelation
- Proposed Method (EVD Covariance Reconstruction UNet)
- Reconstructed Covariance
- A stacked representation of DOA Algorithms (MUSIC, ESPRIT, MVDR, Root-MUSIC)
- Final Estimated DOA Output

Usage:
    python create_doa_pipeline_diagram.py
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

# --- Configuration ---
OUTPUT_DIR = "architecture_diagrams"
OUTPUT_NAME = "doa_pipeline_diagram_professional"
DPI = 300
SAVE_FORMATS = ['pdf', 'png', 'svg'] # Formats suitable for publication

# --- Aesthetics ---
# A professional, cohesive color palette (shades of blue, teal, gray)
COLORS = {
    'input': '#E0F7FA',         # Light Cyan
    'preprocessing': '#B2EBF2', # Medium Cyan
    'autocorr': '#80DEEA',      # Deeper Cyan
    'proposed': '#4DD0E1',      # Strong Cyan/Teal
    'reconstructed': '#26C6DA', # Darker Teal
    'output': '#E0F2F1',        # Light Teal
    'arrow': '#424242',         # Dark Gray
    'text': '#212121',          # Near Black
    'shadow': '#BDBDBD',        # Medium Gray
    'bg': '#FFFFFF',            # White background
    # Colors for the stacked algorithm blocks
    'algo_stack': ['#FFCDD2', '#EF9A9A', '#E57373', '#EF5350'] # Shades of red
}

# Set Matplotlib font properties for a professional look
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 10
})


def add_block(ax, center_x, center_y, width, height, title, subtitle, color, fontsize=12):
    """
    Adds a styled block with a subtle shadow to the axes.
    The main title is at the top, and the subtitle (e.g., math symbol) is centered.
    """
    # Shadow
    shadow_offset = 0.05
    shadow_patch = FancyBboxPatch(
        (center_x - width / 2 + shadow_offset, center_y - height / 2 - shadow_offset),
        width, height,
        boxstyle="round,pad=0.1,rounding_size=0.1",
        facecolor=COLORS['shadow'],
        edgecolor='none',
        alpha=0.6
    )
    ax.add_patch(shadow_patch)

    # Main block
    block_patch = FancyBboxPatch(
        (center_x - width / 2, center_y - height / 2),
        width, height,
        boxstyle="round,pad=0.1,rounding_size=0.1",
        facecolor=color,
        edgecolor=COLORS['text'],
        linewidth=1.5
    )
    ax.add_patch(block_patch)

    # Text
    title_y_offset = 0.25 if subtitle else 0
    ax.text(center_x, center_y + title_y_offset, title, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', color=COLORS['text'])
    if subtitle:
        ax.text(center_x, center_y - 0.15, subtitle, ha='center', va='center',
                fontsize=fontsize + 2, color=COLORS['text'])


def add_stacked_blocks(ax, start_x, start_y, width, height, algorithms, colors):
    """
    Adds the stacked algorithm blocks with an offset, 3D-like effect.
    The list of algorithms should be from back to front.
    Draws all blocks first, then all text to prevent text from being hidden.
    """
    offset_x = 0.2
    offset_y = -0.2
    
    block_positions = []

    # First pass: Draw all blocks and store their center positions
    for i, color in enumerate(colors):
        center_x = start_x + i * offset_x
        center_y = start_y + i * offset_y
        block_positions.append((center_x, center_y))
        
        # Draw shadow
        shadow_offset = 0.05
        shadow_patch = FancyBboxPatch(
            (center_x - width / 2 + shadow_offset, center_y - height / 2 - shadow_offset),
            width, height,
            boxstyle="round,pad=0.1,rounding_size=0.2",
            facecolor=COLORS['shadow'], alpha=0.6, edgecolor='none'
        )
        ax.add_patch(shadow_patch)

        # Draw block
        block_patch = FancyBboxPatch(
            (center_x - width / 2, center_y - height / 2),
            width, height,
            boxstyle="round,pad=0.1,rounding_size=0.2",
            facecolor=color,
            edgecolor=COLORS['text'],
            linewidth=1.5
        )
        ax.add_patch(block_patch)

    # Second pass: Draw all text on top of the blocks
    for i, algo in enumerate(algorithms):
        center_x, center_y = block_positions[i]
        ax.text(center_x, center_y, algo, ha='center', va='center',
                fontsize=14, fontweight='bold', color=COLORS['text'])


def add_arrow(ax, start_pos, end_pos):
    """
    Adds a styled arrow between two points.
    """
    ax.annotate('', xy=end_pos, xytext=start_pos,
                arrowprops=dict(
                    arrowstyle='-|>',
                    lw=2.5,
                    color=COLORS['arrow'],
                    shrinkA=10, # Starts after the block
                    shrinkB=10  # Ends before the block
                ))


def create_doa_pipeline_diagram():
    """
    Creates and orchestrates the generation of the DOA pipeline diagram.
    """
    fig, ax = plt.subplots(figsize=(20, 6))
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 6)
    ax.axis('off')
    ax.set_facecolor(COLORS['bg'])
    fig.set_facecolor(COLORS['bg'])

    # --- Block Positions and Dimensions ---
    y_center = 3.0
    block_h = 1.2
    block_w = 2.0
    
    # Define horizontal positions for more space
    input_x = 2.0
    prep_x = 5.0
    autocorr_x = 8.0
    proposed_x = 11.0
    reconstructed_x = 14.0
    stack_x = 17.0

    # Input
    add_block(ax, input_x, y_center, block_w, block_h, 'Input Signal', r'$x(t)$', COLORS['input'])
    
    # Preprocessing
    add_block(ax, prep_x, y_center, block_w, block_h, 'Preprocessing', '', COLORS['preprocessing'])

    # Autocorrelation
    add_block(ax, autocorr_x, y_center, block_w, block_h, 'Multilag Autocorrelation', r'$R_x$', COLORS['autocorr'])
    
    # Proposed Method
    add_block(ax, proposed_x, y_center, block_w, block_h, 'Proposed Method', 'EVD UNet', COLORS['proposed'])
    
    # Reconstructed Covariance
    add_block(ax, reconstructed_x, y_center, block_w, block_h, 'Reconstructed Covariance', r'$\hat{R}_x$', COLORS['reconstructed'])

    # Algorithm Stack
    algo_list = ['MUSIC', 'ESPRIT', 'MVDR', 'Root-MUSIC']
    stack_start_pos = (stack_x, y_center + 0.3)
    add_stacked_blocks(ax, stack_start_pos[0], stack_start_pos[1], 2.0, 1.8, algo_list, COLORS['algo_stack'])

    # Output
    final_algo_offset_x = (len(algo_list) - 1) * 0.2
    output_x = stack_start_pos[0] + final_algo_offset_x + 3.0 # Moved further to the right
    add_block(ax, output_x, y_center, 1.8, block_h, 'Estimated DOA', r'$\hat{\theta}$', COLORS['output'], fontsize=14)

    # --- Arrows and Labels ---
    add_arrow(ax, (input_x + block_w/2, y_center), (prep_x - block_w/2, y_center))
    add_arrow(ax, (prep_x + block_w/2, y_center), (autocorr_x - block_w/2, y_center))
    add_arrow(ax, (autocorr_x + block_w/2, y_center), (proposed_x - block_w/2, y_center))
    add_arrow(ax, (proposed_x + block_w/2, y_center), (reconstructed_x - block_w/2, y_center))
    add_arrow(ax, (reconstructed_x + block_w/2, y_center), (stack_x - 1.0, y_center))

    # Arrow from Algorithm stack to Output (horizontal alignment)
    arrow_start_x = stack_start_pos[0] + final_algo_offset_x + 1.0
    add_arrow(ax, (arrow_start_x, y_center), (output_x - 0.9, y_center))

    plt.tight_layout(pad=0.5)
    return fig


def save_figure(fig, filename, formats):
    """
    Saves the figure in specified formats to the output directory.
    """
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n📄 Saving generated files:")
    for fmt in formats:
        filepath = output_path / f"{filename}.{fmt}"
        fig.savefig(filepath, dpi=DPI, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
        print(f"  ✅ Saved: {filepath}")

def main():
    """Main execution function."""
    print("=" * 60)
    print("🎨 Generating Professional DOA Pipeline Diagram 🎨")
    print("=" * 60)
    
    fig = create_doa_pipeline_diagram()
    save_figure(fig, OUTPUT_NAME, SAVE_FORMATS)
    
    print("\n✨ Diagram generation complete! ✨")
    print(f"📁 Files are located in the '{OUTPUT_DIR}' directory.")
    print("=" * 60)
    # To prevent the plot from showing in interactive environments automatically.
    plt.close(fig)

if __name__ == "__main__":
    main()

