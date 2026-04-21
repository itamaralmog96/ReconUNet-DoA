#!/usr/bin/env python3
"""
Generate publication-quality block diagram for EVD UNet architecture.

This script creates a professional diagram suitable for academi        # Title
    ax.text(20.5, 19.5, 'EVD-Based Covariance Reconstruction UNet', 
            ha='center', fontsize=26, fontweight='bold')
    ax.text(20.5, 18.6, 'Dual-Head Architecture for DOA Estimation', 
            ha='center', fontsize=20, style='italic', color='#555555')
    
    # Define vertical levels (4 levels: 0 at top, 3 at bottom) - much larger spacing
    y_level_0 = 14.0   # Top level
    y_level_1 = 9.0    # One level down (5.0 spacing)
    y_level_2 = 4.0    # Two levels down (5.0 spacing)
    y_level_3 = -1.0   # Bottom level (bottleneck) (5.0 spacing)  ax.text(20.5, 13.2, 'EVD-Based Covariance Reconstruction UNet', 
            ha='center', fontsize=26, fontweight='bold')
    ax.text(20.5, 12.6, 'Dual-Head Architecture for DOA Estimation',x.text(17.5, 13.2, 'EVD-Based Covariance Reconstruction UNet', 
            ha='center', fontsize=26, fontweight='bold')
    ax.text(17.5, 12.6, 'Dual-Head Architecture for DOA Estimation',apers using matplotlib.
Outputs multiple formats: PDF (vector), PNG (raster), SVG (vector), and     # Decoder 3 (Level 0) - Two conv blocks
    x_pos += 1.0
    dec3_x = x_pos
    dec3_y = y_level_0
    add_block(ax, dec3_x, dec3_y, 1.3, 1.8, 'Decoder 3', 
              '2×Conv+BN
16 ch', color=COLORS['decoder'], fontsize=15, sublabel_fontsize=13)
    add_arrow(ax, up2_x, y_level_0 - 0.8, dec3_x - 0.65, dec3_y)
    
    # === GRAM DIAGONAL LOADING ===
    x_pos += 1.8
    gram_x = x_pos
    gram_y = y_level_0
    add_block(ax, gram_x, gram_y, 1.4, 1.8, 'Gram Diagonal
Loading', 
              'Ensure PSD', color=COLORS['decoder'], fontsize=13, sublabel_fontsize=12)
    add_arrow(ax, dec3_x + 0.65, dec3_y, gram_x - 0.7, gram_y)
    
    # === UNET OUTPUT ===
    x_pos += 1.9
    unet_out_x = x_pos
    unet_out_y = y_level_0
    add_block(ax, unet_out_x, unet_out_y, 1.2, 1.6, 'UNet Out', 
              'Features
[8, 16, 8]', color=COLORS['input'], fontsize=15, sublabel_fontsize=13)r LaTeX).

Usage:
    python create_evdunet_paper_diagram.py

Output:
    - evdunet_architecture_paper.pdf  (recommended for papers)
    - evdunet_architecture_paper.png  (for presentations/web)
    - evdunet_architecture_paper.svg  (editable vector)
    - evdunet_architecture_paper.eps  (for LaTeX papers)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from matplotlib.patheffects import withStroke
import numpy as np

# Configuration
OUTPUT_DIR = "architecture_diagrams"
OUTPUT_NAME = "evdunet_architecture_paper"
DPI = 300  # High resolution for papers

# Color scheme (professional, colorblind-friendly)
COLORS = {
    'input': '#E8F4F8',      # Light blue
    'encoder': '#B3D9E6',    # Medium blue
    'pool': '#7FB3D5',       # Darker blue (for pool blocks)
    'bottleneck': '#FB8072', # Red/coral (distinct from encoder/decoder)
    'decoder': '#C7E9C0',    # Light green
    'eigenval': '#FDB462',   # Orange
    'eigenvec': '#BEBADA',   # Purple
    'output': '#FCCDE5',     # Pink
    'arrow': '#333333',      # Dark gray
    'text': '#000000',       # Black
}

def add_block(ax, x, y, width, height, label, sublabel="", color='lightblue', 
              alpha=0.9, fontsize=14, sublabel_fontsize=12):
    """Add a rectangular block to the diagram."""
    # Create fancy box with rounded corners
    box = FancyBboxPatch(
        (x - width/2, y - height/2), width, height,
        boxstyle="round,pad=0.05", 
        linewidth=2.5, 
        edgecolor='#333333',
        facecolor=color, 
        alpha=alpha,
        zorder=2
    )
    ax.add_patch(box)
    
    # Add main label with more spacing from center
    ax.text(x, y + 0.25, label, ha='center', va='center', 
            fontsize=fontsize, fontweight='bold', color=COLORS['text'], zorder=3)
    
    # Add sublabel if provided with more spacing
    if sublabel:
        ax.text(x, y - 0.35, sublabel, ha='center', va='center', 
                fontsize=sublabel_fontsize, style='italic', color=COLORS['text'], zorder=3)
    
    return box

def add_arrow(ax, x1, y1, x2, y2, label="", style='->',  linewidth=3.0, 
              color=None, label_offset=0.1):
    """Add an arrow between blocks."""
    if color is None:
        color = COLORS['arrow']
    
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style,
        linewidth=linewidth,
        color=color,
        zorder=1,
        mutation_scale=25,
        alpha=0.8
    )
    ax.add_patch(arrow)
    
    # Add label if provided
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x, mid_y + label_offset, label, ha='center', va='bottom',
                fontsize=12, style='italic', color=color, zorder=3,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor='none', alpha=0.8))
    
    return arrow

def add_skip_connection(ax, x1, y1, x2, y2, height_offset=0.5):
    """Add curved skip connection."""
    # Create curved path
    path = mpatches.FancyBboxPatch(
        (x1, y1), x2-x1, y2-y1,
        boxstyle=f"round,pad={height_offset}",
        linewidth=2,
        edgecolor='#2E86AB',
        facecolor='none',
        linestyle='--',
        alpha=0.6,
        zorder=0
    )
    
    # Draw arc manually
    from matplotlib.path import Path
    import matplotlib.patches as patches
    
    verts = [
        (x1, y1),  # Start
        (x1, y1 + height_offset),  # Control point 1
        (x2, y2 + height_offset),  # Control point 2
        (x2, y2),  # End
    ]
    
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
    path = Path(verts, codes)
    patch = patches.PathPatch(path, facecolor='none', 
                              edgecolor='#2E86AB', linewidth=2, 
                              linestyle='--', alpha=0.6, zorder=0)
    ax.add_patch(patch)

def create_evdunet_diagram():
    """Create the main EVD UNet architecture diagram with proper U-shape and 3 encoder/decoder pairs."""
    
    # Create larger figure with more space
    fig, ax = plt.subplots(figsize=(52, 24))
    ax.set_xlim(-1, 42)
    ax.set_ylim(-1, 18)
    ax.axis('off')
    
    # Title
    ax.text(17.5, 13.2, 'EVD-Based Covariance Reconstruction UNet', 
            ha='center', fontsize=26, fontweight='bold')
    # ax.text(15.5, 12.6, 'Dual-Head Architecture for DOA Estimation', 
    #         ha='center', fontsize=20, style='italic', color='#555555')
    
    # Define vertical levels (4 levels: 0 at top, 3 at bottom)
    y_level_0 = 8.0   # Top level
    y_level_1 = 6.0   # One level down
    y_level_2 = 4.0   # Two levels down
    y_level_3 = 2.0   # Bottom level (bottleneck)
    
    # === INPUT ===
    x_pos = 1.2
    add_block(ax, x_pos, y_level_0, 1.2, 1.6, 'Input', 
              r'$R_x(\tau)$' + '\n[8, 16, 8]', 
              color=COLORS['input'], fontsize=16, sublabel_fontsize=14)
    
    # === ENCODER PATH - Going DOWN ===
    # Encoder 1 (Level 0) - Two conv blocks
    x_pos = 3.0
    enc1_x = x_pos
    enc1_y = y_level_0
    add_block(ax, enc1_x, enc1_y, 1.3, 1.8, 'Encoder 1', 
              '2×Conv+BN\n16 ch', color=COLORS['encoder'], fontsize=15, sublabel_fontsize=13)
    add_arrow(ax, 1.8, y_level_0, enc1_x - 0.65, enc1_y)
    
    # Pool 1 (at Level 0, same as Encoder 1)
    x_pos += 1.8
    pool1_x = x_pos
    add_block(ax, pool1_x, y_level_0, 0.8, 1.2, 'Pool', '2×2', 
              color=COLORS['pool'], fontsize=14, sublabel_fontsize=12)
    add_arrow(ax, enc1_x + 0.65, enc1_y, pool1_x - 0.4, y_level_0)
    
    # Single 90-degree arrow down to next level
    add_arrow(ax, pool1_x, y_level_0 - 0.6, pool1_x, y_level_1 + 0.9, 
              linewidth=2.5, color=COLORS['arrow'])
    
    # Encoder 2 (Level 1) - Two conv blocks
    x_pos += 1.0
    enc2_x = x_pos
    enc2_y = y_level_1
    add_block(ax, enc2_x, enc2_y, 1.3, 1.8, 'Encoder 2', 
              '2×Conv+BN\n32 ch', color=COLORS['encoder'], fontsize=15, sublabel_fontsize=13)
    add_arrow(ax, pool1_x, y_level_1 + 0.9, enc2_x - 0.65, enc2_y)
    
    # Pool 2 (at Level 1, same as Encoder 2)
    x_pos += 1.8
    pool2_x = x_pos
    add_block(ax, pool2_x, y_level_1, 0.8, 1.2, 'Pool', '2×2', 
              color=COLORS['pool'], fontsize=14, sublabel_fontsize=12)
    add_arrow(ax, enc2_x + 0.65, enc2_y, pool2_x - 0.4, y_level_1)
    
    # Single 90-degree arrow down to next level
    add_arrow(ax, pool2_x, y_level_1 - 0.6, pool2_x, y_level_2 + 0.9, 
              linewidth=2.5, color=COLORS['arrow'])
    
    # Encoder 3 (Level 2) - Two conv blocks
    x_pos += 1.0
    enc3_x = x_pos
    enc3_y = y_level_2
    add_block(ax, enc3_x, enc3_y, 1.3, 1.8, 'Encoder 3', 
              '2×Conv+BN\n64 ch', color=COLORS['encoder'], fontsize=15, sublabel_fontsize=13)
    add_arrow(ax, pool2_x, y_level_2 + 0.9, enc3_x - 0.65, enc3_y)
    
    # Pool 3 (at Level 2, same as Encoder 3)
    x_pos += 1.8
    pool3_x = x_pos
    add_block(ax, pool3_x, y_level_2, 0.8, 1.2, 'Pool', '2×2', 
              color=COLORS['pool'], fontsize=14, sublabel_fontsize=12)
    add_arrow(ax, enc3_x + 0.65, enc3_y, pool3_x - 0.4, y_level_2)
    
    # === BOTTLENECK (Level 3 - bottom) - Single conv layer ===
    # Position bottleneck with space from pool3
    x_pos += 2.2  # Add more space between pool3 and bottleneck
    bottleneck_x = x_pos
    bottleneck_y = y_level_3
    add_block(ax, bottleneck_x, bottleneck_y, 1.4, 2.0, 'Bottleneck', 
              'Conv+BN\n128 ch', color=COLORS['bottleneck'], fontsize=15, sublabel_fontsize=13)
    
    # Arrow from pool3 to bottleneck
    add_arrow(ax, pool3_x + 0.4, y_level_2 - 0, bottleneck_x - 0.7, bottleneck_y + 1.0, 
              linewidth=2.5, color=COLORS['arrow'])
    
    # === DECODER PATH - Going UP ===
    # Arrow from bottleneck to decoder1
    x_pos += 2.2  # Add more space between bottleneck and decoder1
    dec1_x = x_pos
    dec1_y = y_level_2
    add_block(ax, dec1_x, dec1_y, 1.3, 1.8, 'Decoder 1', 
              '2×Conv+BN\n64 ch', color=COLORS['decoder'], fontsize=15, sublabel_fontsize=13)
    add_arrow(ax, bottleneck_x + 0.7, bottleneck_y + 1.0, dec1_x - 0.65, dec1_y, 
              linewidth=2.5, color=COLORS['arrow'])
    
    # Up 1 (at Level 2, same as Decoder 1)
    x_pos += 1.8
    up1_x = x_pos
    add_block(ax, up1_x, y_level_2, 0.8, 1.2, 'Up', '2×2', 
              color=COLORS['pool'], fontsize=14, sublabel_fontsize=12)
    add_arrow(ax, dec1_x + 0.65, dec1_y, up1_x - 0.4, y_level_2)
    
    # Single 90-degree arrow up to next level
    add_arrow(ax, up1_x, y_level_2 + 0.6, up1_x, y_level_1 - 0.9, 
              linewidth=2.5, color=COLORS['arrow'])
    
    # Decoder 2 (Level 1) - Two conv blocks
    x_pos += 1.0
    dec2_x = x_pos
    dec2_y = y_level_1
    add_block(ax, dec2_x, dec2_y, 1.3, 1.8, 'Decoder 2', 
              '2×Conv+BN\n32 ch', color=COLORS['decoder'], fontsize=15, sublabel_fontsize=13)
    add_arrow(ax, up1_x, y_level_1 - 0.7, dec2_x - 0.65, dec2_y)
    
    # Up 2 (at Level 1, same as Decoder 2)
    x_pos += 1.8
    up2_x = x_pos
    add_block(ax, up2_x, y_level_1, 0.8, 1.2, 'Up', '2×2', 
              color=COLORS['pool'], fontsize=14, sublabel_fontsize=12)
    add_arrow(ax, dec2_x + 0.65, dec2_y, up2_x - 0.4, y_level_1)
    
    # Single 90-degree arrow up to top level
    add_arrow(ax, up2_x, y_level_1 + 0.6, up2_x, y_level_0 - 0.8, 
              linewidth=2.5, color=COLORS['arrow'])
    
    # Decoder 3 (Level 0) - Two conv blocks
    x_pos += 1.0
    dec3_x = x_pos
    dec3_y = y_level_0
    add_block(ax, dec3_x, dec3_y, 1.3, 1.8, 'Decoder 3', 
              '2×Conv+BN\n16 ch', color=COLORS['decoder'], fontsize=15, sublabel_fontsize=13)
    add_arrow(ax, up2_x, y_level_0 - 0.7, dec3_x - 0.65, dec3_y)
    
    # === GRAM DIAGONAL LOADING ===
    x_pos += 2.0
    gram_x = x_pos
    gram_y = y_level_0
    add_block(ax, gram_x, gram_y, 1.4, 1.8, 'Gram Diagonal\nLoading', 
              'Ensure PSD', color=COLORS['decoder'], fontsize=13, sublabel_fontsize=12)
    add_arrow(ax, dec3_x + 0.65, dec3_y, gram_x - 0.7, gram_y)
    
    # Add transparent box around Gram Diagonal Loading
    gram_box_padding_left = 0.2
    gram_box_padding_right = 0.5
    gram_box_padding_vertical = 0.5
    gram_box = FancyBboxPatch(
        (gram_x - 1.4/2 - gram_box_padding_left, gram_y - 1.8/2 - gram_box_padding_vertical), 
        1.4 + gram_box_padding_left + gram_box_padding_right, 1.8 + 2*gram_box_padding_vertical,
        boxstyle="round,pad=0.2", 
        linewidth=2.5, 
        edgecolor='#06B6D4',
        facecolor='#BFDBFE',
        linestyle='--',
        alpha=0.2,
        zorder=1
    )
    ax.add_patch(gram_box)
    
    # Add label for Gram Diagonal Loading box
    ax.text(gram_x, gram_y + 1.8/2 + gram_box_padding_vertical + 0.3, 'SCM Head', 
            ha='center', va='bottom', fontsize=14, fontweight='bold', 
            color='#06B6D4',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                     edgecolor='#06B6D4', linewidth=2.0, alpha=0.9))
    
    # === UNET OUTPUT ===
    x_pos += 2.3
    unet_out_x = x_pos
    unet_out_y = y_level_0
    add_block(ax, unet_out_x, unet_out_y, 1.2, 1.6, 'UNet Out', 
              'Features\n[8, 16, 8]', color=COLORS['input'], fontsize=15, sublabel_fontsize=13)
    add_arrow(ax, gram_x + 0.7, gram_y, unet_out_x - 0.6, unet_out_y)
    
    # === SKIP CONNECTIONS (Straight horizontal lines above pool level) ===
    skip_y_offset = 0.8  # Higher to avoid intersection with pool blocks
    
    # Skip from Enc1 to Dec3 (Level 0)
    ax.plot([enc1_x + 0.65, dec3_x - 0.8], [enc1_y + skip_y_offset, dec3_y + skip_y_offset], 
            color='#2E86AB', linewidth=2.5, linestyle='--', alpha=0.7, zorder=0)
    # Add arrowhead at the end
    ax.annotate('', xy=(dec3_x - 0.8, dec3_y + skip_y_offset), xytext=(dec3_x - 0.95, dec3_y + skip_y_offset),
                arrowprops=dict(arrowstyle='->', color='#2E86AB', linewidth=2.5, alpha=0.7))
    ax.text((enc1_x + dec3_x) / 2, enc1_y + skip_y_offset + 0.3, 'Skip', 
            ha='center', fontsize=14, style='italic', color='#2E86AB', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.8))
    
    # Skip from Enc2 to Dec2 (Level 1)
    ax.plot([enc2_x + 0.65, dec2_x - 0.8], [enc2_y + skip_y_offset, dec2_y + skip_y_offset], 
            color='#2E86AB', linewidth=2.5, linestyle='--', alpha=0.7, zorder=0)
    # Add arrowhead at the end
    ax.annotate('', xy=(dec2_x - 0.8, dec2_y + skip_y_offset), xytext=(dec2_x - 0.95, dec2_y + skip_y_offset),
                arrowprops=dict(arrowstyle='->', color='#2E86AB', linewidth=2.5, alpha=0.7))
    ax.text((enc2_x + dec2_x) / 2, enc2_y + skip_y_offset + 0.3, 'Skip', 
            ha='center', fontsize=14, style='italic', color='#2E86AB', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.8))
    
    # Skip from Enc3 to Dec1 (Level 2)
    ax.plot([enc3_x + 0.65, dec1_x - 0.8], [enc3_y + skip_y_offset, dec1_y + skip_y_offset], 
            color='#2E86AB', linewidth=2.5, linestyle='--', alpha=0.7, zorder=0)
    # Add arrowhead at the end
    ax.annotate('', xy=(dec1_x - 0.8, dec1_y + skip_y_offset), xytext=(dec1_x - 0.95, dec1_y + skip_y_offset),
                arrowprops=dict(arrowstyle='->', color='#2E86AB', linewidth=2.5, alpha=0.7))
    ax.text((enc3_x + dec1_x) / 2, enc3_y + skip_y_offset + 0.3, 'Skip', 
            ha='center', fontsize=14, style='italic', color='#2E86AB', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.8))
    
    # === EIGENVALUE HEAD (TOP BRANCH) - Up and to the right ===
    eigenval_y = y_level_0 + 2.2  # Above UNet output
    branch_start_x = unet_out_x + 1.5
    
    # Vertical then horizontal connection
    add_arrow(ax, unet_out_x + 0.6, unet_out_y, branch_start_x - 0.5, unet_out_y, 
              linewidth=2.5, color='#FF6B35', style='->')
    add_arrow(ax, branch_start_x - 0.5, unet_out_y, branch_start_x - 0.5, eigenval_y, 
              linewidth=2.5, color='#FF6B35', style='->')
    
    # Global pooling
    eig_x = branch_start_x + 1.0
    add_block(ax, eig_x, eigenval_y, 1.0, 1.4, 'Global\nPool', 
              '[1, 1, 1]', color=COLORS['eigenval'], fontsize=14, sublabel_fontsize=12)
    add_arrow(ax, branch_start_x - 0.1, eigenval_y, eig_x - 0.5, eigenval_y, 
              linewidth=2.5, color='#FF6B35')
    
    # FC layers
    fc_x = eig_x + 1.5
    add_block(ax, fc_x, eigenval_y, 1.2, 1.6, 'FC Layers', 
              '1→64→128→N', color=COLORS['eigenval'], fontsize=15, sublabel_fontsize=13)
    add_arrow(ax, eig_x + 0.5, eigenval_y, fc_x - 0.6, eigenval_y, 
              linewidth=2.5, color='#FF6B35')
    
    # Eigenvalues output
    eigenval_x = fc_x + 1.8
    add_block(ax, eigenval_x, eigenval_y, 1.4, 1.6, 'Eigenvalues', 
              r'$\mathbf{\Lambda}$ [N]', color=COLORS['eigenval'], fontsize=15, sublabel_fontsize=13)
    add_arrow(ax, fc_x + 0.6, eigenval_y, eigenval_x - 0.7, eigenval_y, 
              linewidth=2.5, color='#FF6B35')
    
    # === EIGENVECTOR HEAD (BOTTOM BRANCH) - Down and to the right ===
    eigenvec_y = y_level_0 - 2.5  # Below UNet output
    
    # Vertical then horizontal connection
    add_arrow(ax, unet_out_x + 0.6, unet_out_y, branch_start_x - 0.5, unet_out_y, 
              linewidth=2.5, color='#7209B7', style='->')
    add_arrow(ax, branch_start_x - 0.5, unet_out_y, branch_start_x - 0.5, eigenvec_y, 
              linewidth=2.5, color='#7209B7', style='->')
    
    # Conv layers
    conv_x = branch_start_x + 1.2
    add_block(ax, conv_x, eigenvec_y, 1.2, 1.6, 'Conv\nLayers', 
              '\n1→32→16→1', color=COLORS['eigenvec'], fontsize=15, sublabel_fontsize=13)
    add_arrow(ax, branch_start_x - 0.1, eigenvec_y, conv_x - 0.6, eigenvec_y, 
              linewidth=2.5, color='#7209B7')
    
    # QR decomposition
    qr_x = conv_x + 1.7
    add_block(ax, qr_x, eigenvec_y, 1.3, 1.6, 'QR Decomp', 
              'Orthogonalize', color=COLORS['eigenvec'], fontsize=15, sublabel_fontsize=13)
    add_arrow(ax, conv_x + 0.6, eigenvec_y, qr_x - 0.65, eigenvec_y, 
              linewidth=2.5, color='#7209B7')
    
    # Eigenvectors output
    eigenvec_x = qr_x + 1.8
    add_block(ax, eigenvec_x, eigenvec_y, 1.4, 1.6, 'Eigenvectors', 
              r'$\mathbf{U}$ [N, N]', color=COLORS['eigenvec'], fontsize=15, sublabel_fontsize=13)
    add_arrow(ax, qr_x + 0.65, eigenvec_y, eigenvec_x - 0.7, eigenvec_y, 
              linewidth=2.5, color='#7209B7')
    
    # === RECONSTRUCTION (On the right side) ===
    recon_x = 35.5  # Positioned to the left for better spacing
    recon_y = y_level_0  # Same level as UNet output
    add_block(ax, recon_x, recon_y, 1.8, 2.2, 'Reconstruction', 
              r'$\mathbf{R}_z = \mathbf{U}\mathbf{\Lambda}\mathbf{U}^H$' + '\n[N, N]', 
              color=COLORS['output'], fontsize=16, sublabel_fontsize=14)
    
    # Direct arrows from eigenvalues to reconstruction
    add_arrow(ax, eigenval_x + 0.7, eigenval_y, recon_x - 0.9, recon_y + 0.4, 
              linewidth=2.5, color='#FF6B35', style='->')
    
    # Direct arrows from eigenvectors to reconstruction  
    add_arrow(ax, eigenvec_x + 0.7, eigenvec_y, recon_x - 0.9, recon_y - 0.4, 
              linewidth=2.5, color='#7209B7', style='->')
    
    # === EVD HEADS BOX ===
    # Create a transparent dashed box around both eigenvalue and eigenvector branches
    evd_box_left = branch_start_x - 0.3
    evd_box_right = eigenvec_x + 1.0
    evd_box_top = eigenval_y + 1.8
    evd_box_bottom = eigenvec_y - 1.8
    evd_box_width = evd_box_right - evd_box_left
    evd_box_height = evd_box_top - evd_box_bottom
    
    evd_box = FancyBboxPatch(
        (evd_box_left, evd_box_bottom), evd_box_width, evd_box_height,
        boxstyle="round,pad=0.3", 
        linewidth=3.0, 
        edgecolor='#9D4EDD',
        facecolor='#E0AAFF',
        linestyle='--',
        alpha=0.15,
        zorder=1
    )
    ax.add_patch(evd_box)
    
    # Add EVD Heads label
    ax.text((evd_box_left + evd_box_right) / 2, evd_box_top + 0.4, 'EVD Heads', 
            ha='center', va='bottom', fontsize=20, fontweight='bold', 
            color='#9D4EDD',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                     edgecolor='#9D4EDD', linewidth=2.5, alpha=0.9))
    
    # === ANNOTATIONS ===
    # Add region labels
    ax.text(15.0, 0.5, 'Base UNet (Encoder-Decoder)', ha='center', 
            fontsize=18, fontweight='bold', color='#2E86AB',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                     edgecolor='#2E86AB', linewidth=2.5, alpha=0.9))
    
    ax.text(branch_start_x + 2.5, eigenval_y + 1.5, 'Eigenvalue Branch', ha='center', 
            fontsize=17, fontweight='bold', color='#FF6B35',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                     edgecolor='#FF6B35', linewidth=2.5, alpha=0.9))
    
    ax.text(branch_start_x + 3.0, eigenvec_y - 1.5, 'Eigenvector Branch', ha='center', 
            fontsize=17, fontweight='bold', color='#7209B7',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                     edgecolor='#7209B7', linewidth=2.5, alpha=0.9))
    
    # Key features box
    features_text = (
        'Key Features:\n'
        '• Dual-head architecture for eigendecomposition\n'
        '• Skip connections preserve spatial features\n'
        '• QR ensures orthonormal eigenvectors\n'
        '• Output: Clean covariance for DOA estimation'
    )
    ax.text(20.5, -0.3, features_text, ha='center', va='top', 
            fontsize=13, family='monospace',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='#F8F9FA', 
                     edgecolor='#333333', linewidth=2, alpha=0.95))
    
    plt.tight_layout()
    return fig

def create_detailed_architecture_table():
    """Create a detailed architecture specification table."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Title
    ax.text(0.5, 0.98, 'EVD UNet Architecture Specifications', 
            ha='center', va='top', transform=ax.transAxes,
            fontsize=16, fontweight='bold')
    
    # Table data
    table_data = [
        ['Component', 'Layer Type', 'Input Shape', 'Output Shape', 'Parameters'],
        ['', '', '', '', ''],
        ['Input', 'Autocorrelation', '[B, τ=8, 2M=16, M=8]', '[B, 8, 16, 8]', '—'],
        ['', '', '', '', ''],
        ['Encoder-1', 'Conv2d + BN + ReLU', '[B, 8, 16, 8]', '[B, 16, 16, 8]', 'K=3×3, P=1'],
        ['Pool-1', 'MaxPool2d', '[B, 16, 16, 8]', '[B, 16, 8, 4]', 'K=2×2, S=2'],
        ['Encoder-2', 'Conv2d + BN + ReLU', '[B, 16, 8, 4]', '[B, 32, 8, 4]', 'K=3×3, P=1'],
        ['Pool-2', 'MaxPool2d', '[B, 32, 8, 4]', '[B, 32, 4, 2]', 'K=2×2, S=2'],
        ['Bottleneck', 'Conv2d + BN + ReLU', '[B, 32, 4, 2]', '[B, 64, 4, 2]', 'K=3×3, P=1'],
        ['', '', '', '', ''],
        ['Decoder-1', 'ConvTrans + Conv + BN', '[B, 64, 4, 2]', '[B, 32, 8, 4]', 'K=2×2, S=2'],
        ['Decoder-2', 'ConvTrans + Conv + BN', '[B, 32, 8, 4]', '[B, 16, 16, 8]', 'K=2×2, S=2'],
        ['UNet Out', 'Conv2d(1×1)', '[B, 16, 16, 8]', '[B, 8, 16, 8]', 'K=1×1'],
        ['', '', '', '', ''],
        ['Eigenvalue Head:', '', '', '', ''],
        ['  Global Pool', 'AdaptiveAvgPool2d', '[B, 8, 16, 8]', '[B, 1, 1, 1]', '—'],
        ['  FC-1', 'Linear + ReLU', '[B, 1]', '[B, 64]', '—'],
        ['  FC-2', 'Linear + ReLU', '[B, 64]', '[B, 128]', '—'],
        ['  FC-3', 'Linear + ReLU', '[B, 128]', '[B, M=8]', '—'],
        ['', '', '', '', ''],
        ['Eigenvector Head:', '', '', '', ''],
        ['  Conv-1', 'Conv2d + BN + ReLU', '[B, 8, 16, 8]', '[B, 32, 16, 8]', 'K=3×3, P=1'],
        ['  Conv-2', 'Conv2d + BN + ReLU', '[B, 32, 16, 8]', '[B, 16, 16, 8]', 'K=3×3, P=1'],
        ['  Conv-3', 'Conv2d + Tanh', '[B, 16, 16, 8]', '[B, 1, 16, 8]', 'K=3×3, P=1'],
        ['  QR Decomp', 'Orthogonalization', '[B, 16, 8]', '[B, M=8, M=8]', '—'],
        ['', '', '', '', ''],
        ['Output', 'Reconstruction', 'U [B,M,M], Λ [B,M]', '[B, M=8, M=8]', 'Rz = UΛU^H'],
    ]
    
    # Create table
    table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                    colWidths=[0.2, 0.25, 0.2, 0.2, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.2)
    
    # Style header row
    for i in range(5):
        cell = table[(0, i)]
        cell.set_facecolor('#2E86AB')
        cell.set_text_props(weight='bold', color='white', fontsize=10)
    
    # Style section headers
    section_rows = [4, 10, 14, 20, 25]
    for row in section_rows:
        for col in range(5):
            cell = table[(row, col)]
            cell.set_facecolor('#E8F4F8')
    
    # Alternate row colors
    for row in range(1, len(table_data)):
        if row not in section_rows:
            for col in range(5):
                cell = table[(row, col)]
                if row % 2 == 0:
                    cell.set_facecolor('#F8F9FA')
    
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
    print("🎨 Generating Publication-Quality EVD UNet Diagrams")
    print("=" * 80)
    
    # Create main architecture diagram
    print("\n📊 Creating main architecture diagram...")
    fig1 = create_evdunet_diagram()
    files1 = save_figure(fig1, OUTPUT_NAME)
    
    # Create detailed specifications table
    print("\n📋 Creating architecture specifications table...")
    fig2 = create_detailed_architecture_table()
    files2 = save_figure(fig2, f"{OUTPUT_NAME}_specs")
    
    # Summary
    print("\n" + "=" * 80)
    print("✅ All diagrams generated successfully!")
    print("\n📁 Output directory:", OUTPUT_DIR)
    print("\n📄 Generated files:")
    print("\n  Main Architecture Diagram:")
    for f in files1:
        print(f"    • {f.name}")
    print("\n  Architecture Specifications:")
    for f in files2:
        print(f"    • {f.name}")
    
    print("\n💡 Recommendations for your paper:")
    print("  • Use PDF format for LaTeX papers (vector graphics, scales perfectly)")
    print("  • Use PNG format for Word/PowerPoint (high resolution)")
    print("  • Use EPS format if required by journal (legacy vector format)")
    print("  • Use SVG format if you need to edit in Inkscape/Illustrator")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()

