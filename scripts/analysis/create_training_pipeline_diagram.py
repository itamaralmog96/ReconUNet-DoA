#!/usr/bin/env python3
"""
Generate Training Pipeline Block Diagram for EVD UNet

This script creates a professional diagram showing:
- Input data (Clean ↔ Corrupted Pairs)
- Feature Extraction (Multi-lag Autocorrelation)
- Base UNet
- Post-Processing (eigenvalue & eigenvector heads)
- Loss Computation (Subspace + EVD + MSE)
- Backpropagation loop

Usage:
    python create_training_pipeline_diagram.py
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path
import matplotlib.patches as mpatches

# --- Configuration ---
OUTPUT_DIR = "architecture_diagrams"
OUTPUT_NAME = "training_pipeline_diagram"
DPI = 300
SAVE_FORMATS = ['pdf', 'png', 'svg', 'eps']

# --- Color Palette ---
COLORS = {
    'input': '#FFE8CC',         # Light Orange/Beige
    'feature': '#FFB6C1',       # Light Pink
    'unet': '#98FB98',          # Pale Green
    'postprocess': '#87CEEB',   # Sky Blue
    'loss': '#FFFF99',          # Light Yellow
    'arrow': '#424242',         # Dark Gray
    'backprop': '#FF6B6B',      # Red for backpropagation
    'text': '#000000',          # Black
    'shadow': '#BDBDBD',        # Gray
    'bg': '#FFFFFF',            # White
}

# Set Matplotlib font properties
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 10
})


def add_block(ax, center_x, center_y, width, height, title, subtitle='', color='white', 
              fontsize=11, subtitle_fontsize=10, edgecolor='black', linewidth=1.5, title_subtitle_spacing=0.2):
    """
    Adds a styled block with shadow to the axes.
    """
    # Shadow
    shadow_offset = 0.08
    shadow_patch = FancyBboxPatch(
        (center_x - width / 2 + shadow_offset, center_y - height / 2 - shadow_offset),
        width, height,
        boxstyle="round,pad=0.08,rounding_size=0.15",
        facecolor=COLORS['shadow'],
        edgecolor='none',
        alpha=0.4
    )
    ax.add_patch(shadow_patch)

    # Main block
    block_patch = FancyBboxPatch(
        (center_x - width / 2, center_y - height / 2),
        width, height,
        boxstyle="round,pad=0.08,rounding_size=0.15",
        facecolor=color,
        edgecolor=edgecolor,
        linewidth=linewidth
    )
    ax.add_patch(block_patch)

    # Title text
    title_y_offset = title_subtitle_spacing if subtitle else 0
    ax.text(center_x, center_y + title_y_offset, title, 
            ha='center', va='center', fontsize=fontsize, 
            fontweight='bold', color=COLORS['text'])
    
    # Subtitle text
    if subtitle:
        ax.text(center_x, center_y - title_subtitle_spacing, subtitle, 
                ha='center', va='center', fontsize=subtitle_fontsize, 
                color=COLORS['text'], style='italic')


def add_arrow(ax, start_pos, end_pos, color=None, linestyle='-', linewidth=2.5, label='', label_offset=0.3):
    """
    Adds a styled arrow between two points.
    """
    if color is None:
        color = COLORS['arrow']
    
    arrow = FancyArrowPatch(
        start_pos, end_pos,
        arrowstyle='-|>',
        mutation_scale=20,
        linewidth=linewidth,
        color=color,
        linestyle=linestyle,
        shrinkA=5,
        shrinkB=5
    )
    ax.add_patch(arrow)
    
    # Add label if provided, centered on the arrow
    if label:
        mid_x = (start_pos[0] + end_pos[0]) / 2
        mid_y = (start_pos[1] + end_pos[1]) / 2
        
        # Adjust label position based on arrow direction
        if abs(start_pos[1] - end_pos[1]) < 0.5:  # Horizontal arrow
            va = 'bottom'
            offset_y = label_offset
        else:  # Diagonal arrow
            va = 'bottom'
            offset_y = label_offset
        
        ax.text(mid_x, mid_y + offset_y, label, 
                ha='center', va=va, fontsize=22, 
                color='black', style='italic', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor='none', alpha=0.8))


def create_training_pipeline_diagram():
    """
    Creates the training pipeline diagram.
    """
    fig, ax = plt.subplots(figsize=(20, 12))
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 12)
    ax.axis('off')
    ax.set_facecolor(COLORS['bg'])
    fig.set_facecolor(COLORS['bg'])

    # --- Define positions ---
    # Top row (forward pass)
    input_x = 3.0
    input_y = 9.0
    
    feature_x = 7.5
    feature_y = 9.0
    
    unet_x = 12.0
    unet_y = 9.0
    
    postprocess_x = 16.5
    postprocess_y = 9.0
    
    # Right side - Loss computation (aligned vertically with post-processing)
    loss_x = 16.5  # Same x as post-processing for vertical arrow
    loss_y = 4.5  # Moved up a bit
    
    # Block dimensions
    block_w = 2.8
    block_h = 1.4
    
    loss_w = 2.5
    loss_h = 2.0

    # === INPUT BLOCK ===
    add_block(ax, input_x, input_y, block_w, block_h,
              r'$X$, $R^*$, $U^*$, $\lambda^*$',
              '(Clean $\\leftrightarrow$ Corrupted\nPair)',
              COLORS['input'], fontsize=22, subtitle_fontsize=18)

    # === FEATURE EXTRACTION ===
    add_block(ax, feature_x, feature_y, block_w, block_h,
              'Feature Extraction',
              '(Multi-lag\nAutocorrelation)',
              COLORS['feature'], fontsize=22, subtitle_fontsize=18)

    # === PROPOSED METHOD BOX (Transparent container) ===
    # Calculate the box dimensions to encompass both UNet and Post-processing
    proposed_padding = 0.4
    proposed_left = unet_x - block_w/2 - proposed_padding
    proposed_right = postprocess_x + block_w/2 + proposed_padding
    proposed_top = unet_y + block_h/2 + proposed_padding + 0.4  # Extra space for label
    proposed_bottom = unet_y - block_h/2 - proposed_padding
    proposed_width = proposed_right - proposed_left
    proposed_height = proposed_top - proposed_bottom
    proposed_center_x = (proposed_left + proposed_right) / 2
    proposed_center_y = (proposed_top + proposed_bottom) / 2
    
    # Draw transparent box with teal/cyan color (between blue and green)
    teal_color = '#17A2B8'  # Teal/cyan color
    proposed_box = FancyBboxPatch(
        (proposed_left, proposed_bottom),
        proposed_width, proposed_height,
        boxstyle="round,pad=0.1,rounding_size=0.2",
        facecolor='lightgray',
        edgecolor=teal_color,
        linewidth=2.5,
        linestyle='--',
        alpha=0.15
    )
    ax.add_patch(proposed_box)
    
    # Add "Proposed Method" label at the top
    ax.text(proposed_center_x, proposed_top - 0.25, 'Proposed Method', 
            ha='center', va='center', fontsize=24, fontweight='bold', 
            color=teal_color,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                     edgecolor=teal_color, linewidth=2, alpha=0.9))
    
    # === BASE UNET ===
    add_block(ax, unet_x, unet_y, block_w, block_h,
              'Base UNet',
              '(Encoder-Decoder)',
              COLORS['unet'], fontsize=22, subtitle_fontsize=18)

    # === POST-PROCESSING ===
    add_block(ax, postprocess_x, postprocess_y, block_w, block_h,
              'Eigenvector &',
              'Eigenvalue Heads',
              COLORS['postprocess'], fontsize=22, subtitle_fontsize=18)

    # === LOSS COMPUTATION ===
    # Need a bigger block for the full formula
    loss_w = 3.5
    loss_h = 2.2
    add_block(ax, loss_x, loss_y, loss_w, loss_h,
              'Loss Computation',
              r'$\mathcal{L} = w_{\text{eig}} \mathcal{L}_{\text{eig}} + w_{\text{proj}} \mathcal{L}_{\text{proj}}$' + '\n' + 
              r'$+ w_{\text{dom}} \mathcal{L}_{\text{dom}} + w_{\text{rec}} \mathcal{L}_{\text{rec}}$',
              COLORS['loss'], fontsize=22, subtitle_fontsize=18, title_subtitle_spacing=0.4)

    # === FORWARD PASS ARROWS ===
    # Input to Feature Extraction
    add_arrow(ax, (input_x + block_w/2, input_y), 
              (feature_x - block_w/2, feature_y), 
              label=r'$X$')
    
    # Feature Extraction to Base UNet
    add_arrow(ax, (feature_x + block_w/2, feature_y), 
              (unet_x - block_w/2, unet_y),
              label=r'$R_{\tau}$')
    
    # Base UNet to Post-Processing
    add_arrow(ax, (unet_x + block_w/2, unet_y), 
              (postprocess_x - block_w/2, postprocess_y))
    
    # Post-Processing to Loss (vertical from center of bottom edge)
    add_arrow(ax, (postprocess_x, postprocess_y - block_h/2), 
              (loss_x, loss_y + loss_h/2),
              label=r'$\hat{R}$, $U$, $\lambda$')

    # === CLEAN TARGETS TO LOSS (dashed line) ===
    # From center bottom of input block to center left of loss block (horizontal entry)
    target_arrow_start = (input_x, input_y - block_h/2)
    target_arrow_mid1 = (input_x, loss_y)  # Go down to the level of loss block
    target_arrow_end = (loss_x - loss_w/2, loss_y)  # Horizontal entry to left side
    
    # Draw dashed line path (2 segments: vertical then horizontal)
    # Start slightly below the input block, stop before reaching the block edges
    ax.plot([target_arrow_start[0], target_arrow_mid1[0]], 
            [target_arrow_start[1] - 0.2, target_arrow_mid1[1]], 
            'k--', linewidth=2, alpha=0.6)
    # Horizontal line stops before the loss block edge
    ax.plot([target_arrow_mid1[0], loss_x - loss_w/2 - 0.2], 
            [target_arrow_mid1[1], target_arrow_end[1]], 
            'k--', linewidth=2, alpha=0.6)
    
    # Add arrowhead at the end (pointing horizontally to center left of loss block)
    ax.annotate('', xy=(loss_x - loss_w/2, loss_y),
                xytext=(loss_x - loss_w/2 - 0.5, loss_y),
                arrowprops=dict(arrowstyle='->', lw=2, color='black', 
                              linestyle='--', alpha=0.6))
    
    # Label for clean targets - centered on the horizontal path
    label_x = (input_x + loss_x - loss_w/2) / 2
    label_y = loss_y
    ax.text(label_x, label_y + 0.3, 
            r'Clean Targets $R^*$, $U^*$, $\lambda^*$', 
            fontsize=22, style='italic', color='black', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                     edgecolor='none', alpha=0.8))

    # === BACKPROPAGATION ARROW ===
    # Draw backpropagation path (red dashed line with arrow)
    # Goes from loss block into the right side of the proposed method box
    backprop_start = (loss_x, loss_y - loss_h/2)  # Bottom center of loss block
    backprop_mid1 = (loss_x, 2.0)  # Go down vertically first
    backprop_mid2 = (loss_x + 1.5, 2.0)  # Go right horizontally
    backprop_mid3 = (proposed_right + 0.8, 2.0)  # Continue right (with more spacing)
    backprop_mid4 = (proposed_right + 0.8, proposed_center_y)  # Go up to center height
    backprop_end = (proposed_right, proposed_center_y)  # Enter horizontally from the right
    
    # Draw the backpropagation path (5 segments now)
    # First segment: vertical down from loss block (start slightly away from block)
    ax.plot([backprop_start[0], backprop_mid1[0]], 
            [backprop_start[1] - 0.2, backprop_mid1[1]], 
            color=COLORS['backprop'], linestyle='--', linewidth=2.5, alpha=0.8)
    ax.plot([backprop_mid1[0], backprop_mid2[0]], 
            [backprop_mid1[1], backprop_mid2[1]], 
            color=COLORS['backprop'], linestyle='--', linewidth=2.5, alpha=0.8)
    ax.plot([backprop_mid2[0], backprop_mid3[0]], 
            [backprop_mid2[1], backprop_mid3[1]], 
            color=COLORS['backprop'], linestyle='--', linewidth=2.5, alpha=0.8)
    ax.plot([backprop_mid3[0], backprop_mid4[0]], 
            [backprop_mid3[1], backprop_mid4[1]], 
            color=COLORS['backprop'], linestyle='--', linewidth=2.5, alpha=0.8)
    # Last horizontal segment: stop before reaching the proposed method box
    ax.plot([backprop_mid4[0], proposed_right + 0.2], 
            [backprop_mid4[1], backprop_end[1]], 
            color=COLORS['backprop'], linestyle='--', linewidth=2.5, alpha=0.8)
    
    # Add arrowhead pointing horizontally into the right side of proposed method box
    ax.annotate('', xy=backprop_end,
                xytext=(proposed_right + 0.5, proposed_center_y),
                arrowprops=dict(arrowstyle='->', lw=2.5, 
                              color=COLORS['backprop'], linestyle='--'))
    
    # Backpropagation label with box - centered on the bottom horizontal part
    backprop_label_x = (backprop_mid2[0] + backprop_mid3[0]) / 2
    ax.text(backprop_label_x, 2.0 + 0.5, 'Backpropagation', 
            fontsize=24, fontweight='bold', color=COLORS['backprop'],
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                     edgecolor=COLORS['backprop'], linewidth=2, alpha=0.9))

    plt.tight_layout(pad=0.5)
    return fig


def save_figure(fig, filename, formats):
    """
    Saves the figure in specified formats to the output directory.
    """
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    
    saved_files = []
    print("\n📄 Saving generated files:")
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
    print("🎯 Generating Training Pipeline Block Diagram")
    print("=" * 80)
    
    # Create the diagram
    print("\n📊 Creating training pipeline diagram...")
    fig = create_training_pipeline_diagram()
    files = save_figure(fig, OUTPUT_NAME, SAVE_FORMATS)
    
    # Summary
    print("\n" + "=" * 80)
    print("✅ Training pipeline diagram generated successfully!")
    print("\n📁 Output directory:", OUTPUT_DIR)
    print("\n📄 Generated files:")
    for f in files:
        print(f"    • {f.name}")
    
    print("\n💡 Diagram features:")
    print("  • Input data (Clean ↔ Corrupted pairs) in LaTeX format")
    print("  • Feature extraction (Multi-lag Autocorrelation)")
    print("  • Base UNet")
    print("  • Post-processing (eigenvalue & eigenvector heads)")
    print("  • Loss computation (Subspace + EVD + MSE) in LaTeX format")
    print("  • Backpropagation loop")
    print("  • Clean targets path (dashed line)")
    print("  • All mathematical symbols in LaTeX format")
    print("  • Professional color scheme")
    
    print("\n" + "=" * 80)
    plt.close(fig)


if __name__ == "__main__":
    main()
