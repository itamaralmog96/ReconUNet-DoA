#!/usr/bin/env python3
"""
Generate PlotNeuralNet diagram for EVDCovarianceReconstructionUNet architecture.

This script creates a beautiful LaTeX-based visualization of the EVD UNet architecture
using PlotNeuralNet. It shows the encoder-decoder structure, eigenvalue head, and 
eigenvector head with proper dimensions and connections.

Requirements:
    - PlotNeuralNet: https://github.com/HarisIqbal88/PlotNeuralNet
    - LaTeX with TikZ packages installed

Usage:
    1. Clone PlotNeuralNet: git clone https://github.com/HarisIqbal88/PlotNeuralNet.git
    2. Place this script in the PlotNeuralNet directory or update PLOTNEURALNET_PATH
    3. Run: python visualize_evdunet_architecture.py
    4. Output: evdunet_architecture.tex and evdunet_architecture.pdf (if pdflatex available)
"""

import os
import sys
from pathlib import Path

# Configuration
PLOTNEURALNET_PATH = "./PlotNeuralNet"  # Path to PlotNeuralNet repository
OUTPUT_DIR = "architecture_diagrams"
OUTPUT_NAME = "evdunet_architecture"

# Architecture parameters (matching EVDUNet.py)
TAU = 8  # Number of autocorrelation lags
M = 8    # Number of array elements
CH1, CH2, CH3 = 16, 32, 64  # Channel numbers (simplified for visualization)


def generate_evdunet_diagram():
    """Generate PlotNeuralNet LaTeX code for EVD UNet architecture."""
    
    # Create output directory
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # LaTeX preamble
    latex_code = r"""\documentclass[border=8pt, multi, tikz]{standalone}
\usepackage{import}
\subimport{""" + PLOTNEURALNET_PATH + r"""/layers/}{init}
\usetikzlibrary{positioning}
\usetikzlibrary{3d}

\def\ConvColor{rgb:yellow,5;red,2.5;white,5}
\def\ConvReluColor{rgb:yellow,5;red,5;white,5}
\def\PoolColor{rgb:red,1;black,0.3}
\def\UnpoolColor{rgb:blue,2;green,1;black,0.3}
\def\FcColor{rgb:blue,5;red,2.5;white,5}
\def\FcReluColor{rgb:blue,5;red,5;white,4}
\def\SoftmaxColor{rgb:magenta,5;black,7}
\def\EigenColor{rgb:green,3;blue,3;white,3}

\newcommand{\copymidarrow}{\tikz \draw[-Stealth,line width=0.8mm,draw={rgb:blue,4;red,1;green,1;black,3}] (-0.3,0) -- ++(0.3,0);}

\begin{document}
\begin{tikzpicture}
\tikzstyle{connection}=[ultra thick,every node/.style={sloped,allow upside down},draw=\edgecolor,opacity=0.7]
\tikzstyle{copyconnection}=[ultra thick,every node/.style={sloped,allow upside down},draw={rgb:blue,4;red,1;green,1;black,3},opacity=0.7]

"""

    # Input layer
    latex_code += r"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Input: Autocorrelation Tensor [batch, tau=8, 2M=16, M=8]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\pic[shift={(0,0,0)}] at (0,0,0) 
    {Box={
        name=input,
        caption=Input $R_x(\tau)$,
        xlabel={{tau=8, }},
        zlabel=2M=16,
        fill=\ConvColor,
        height=32,
        width=2,
        depth=16
        }
    };

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Encoder Block 1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\pic[shift={(2,0,0)}] at (input-east) 
    {Box={
        name=enc1,
        caption=Enc-1,
        xlabel={{16, }},
        zlabel=2M=16,
        fill=\ConvReluColor,
        height=32,
        width=3,
        depth=16
        }
    };

\pic[shift={(0,0,0)}] at (enc1-east) 
    {Box={
        name=enc1_bn,
        caption=,
        fill=\ConvReluColor,
        height=32,
        width=0.5,
        depth=16
        }
    };

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Pool 1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\pic[shift={(1.5,0,0)}] at (enc1_bn-east) 
    {Box={
        name=pool1,
        caption=Pool,
        fill=\PoolColor,
        opacity=0.5,
        height=16,
        width=1,
        depth=8
        }
    };

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Encoder Block 2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\pic[shift={(2,0,0)}] at (pool1-east) 
    {Box={
        name=enc2,
        caption=Enc-2,
        xlabel={{32, }},
        zlabel=M=8,
        fill=\ConvReluColor,
        height=16,
        width=5,
        depth=8
        }
    };

\pic[shift={(0,0,0)}] at (enc2-east) 
    {Box={
        name=enc2_bn,
        caption=,
        fill=\ConvReluColor,
        height=16,
        width=0.5,
        depth=8
        }
    };

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Pool 2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\pic[shift={(1.5,0,0)}] at (enc2_bn-east) 
    {Box={
        name=pool2,
        caption=Pool,
        fill=\PoolColor,
        opacity=0.5,
        height=8,
        width=1,
        depth=4
        }
    };

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Encoder Block 3 / Bottleneck
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\pic[shift={(2,0,0)}] at (pool2-east) 
    {Box={
        name=bottleneck,
        caption=Bottleneck,
        xlabel={{64, }},
        fill=\FcReluColor,
        height=8,
        width=8,
        depth=4
        }
    };

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Decoder Block 3
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\pic[shift={(2.5,0,0)}] at (bottleneck-east) 
    {Box={
        name=upconv3,
        caption=Up-3,
        fill=\UnpoolColor,
        opacity=0.5,
        height=16,
        width=1,
        depth=8
        }
    };

\pic[shift={(2,0,0)}] at (upconv3-east) 
    {Box={
        name=dec3,
        caption=Dec-3,
        xlabel={{32, }},
        fill=\ConvReluColor,
        height=16,
        width=5,
        depth=8
        }
    };

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Decoder Block 4
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\pic[shift={(2.5,0,0)}] at (dec3-east) 
    {Box={
        name=upconv4,
        caption=Up-4,
        fill=\UnpoolColor,
        opacity=0.5,
        height=32,
        width=1,
        depth=16
        }
    };

\pic[shift={(2,0,0)}] at (upconv4-east) 
    {Box={
        name=dec4,
        caption=Dec-4,
        xlabel={{16, }},
        zlabel=2M=16,
        fill=\ConvReluColor,
        height=32,
        width=3,
        depth=16
        }
    };

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% UNet Output
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\pic[shift={(2,0,0)}] at (dec4-east) 
    {Box={
        name=unet_out,
        caption=UNet Out,
        xlabel={{tau=8, }},
        zlabel=2M=16,
        fill=\ConvColor,
        height=32,
        width=2,
        depth=16
        }
    };

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Eigenvalue Head (Top Branch)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\pic[shift={(0,6,0)}] at (unet_out-east) 
    {Box={
        name=eig_pool,
        caption=Global Pool,
        xlabel={{1, }},
        fill=\PoolColor,
        opacity=0.7,
        height=4,
        width=4,
        depth=4
        }
    };

\pic[shift={(3,0,0)}] at (eig_pool-east) 
    {RightBandedBox={
        name=eig_fc1,
        caption=FC-64,
        xlabel={{64, }},
        fill=\FcReluColor,
        bandfill=\FcReluColor,
        height=6,
        width=2,
        depth=6
        }
    };

\pic[shift={(2,0,0)}] at (eig_fc1-east) 
    {RightBandedBox={
        name=eig_fc2,
        caption=FC-128,
        xlabel={{128, }},
        fill=\FcReluColor,
        bandfill=\FcReluColor,
        height=8,
        width=2,
        depth=8
        }
    };

\pic[shift={(2,0,0)}] at (eig_fc2-east) 
    {RightBandedBox={
        name=eigenvals,
        caption=Eigenvalues,
        xlabel={{M=8, }},
        fill=\EigenColor,
        bandfill=\EigenColor,
        height=4,
        width=1,
        depth=4
        }
    };

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Eigenvector Head (Bottom Branch)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\pic[shift={(0,-6,0)}] at (unet_out-east) 
    {Box={
        name=eigvec_conv1,
        caption=Conv-32,
        xlabel={{32, }},
        zlabel=2M=16,
        fill=\ConvReluColor,
        height=32,
        width=5,
        depth=16
        }
    };

\pic[shift={(2,0,0)}] at (eigvec_conv1-east) 
    {Box={
        name=eigvec_conv2,
        caption=Conv-16,
        xlabel={{16, }},
        fill=\ConvReluColor,
        height=32,
        width=3,
        depth=16
        }
    };

\pic[shift={(2,0,0)}] at (eigvec_conv2-east) 
    {Box={
        name=eigvec_raw,
        caption=Conv-1,
        xlabel={{1, }},
        zlabel=2M=16,
        fill=\ConvColor,
        height=32,
        width=2,
        depth=16
        }
    };

\pic[shift={(2,0,0)}] at (eigvec_raw-east) 
    {Box={
        name=eigvecs,
        caption=Eigenvectors (QR),
        xlabel={{M, }},
        zlabel=M,
        fill=\EigenColor,
        height=16,
        width=2,
        depth=16
        }
    };

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Reconstruction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\pic[shift={(4,0,0)}] at (eigvecs-east) 
    {Box={
        name=recon_cov,
        caption=Reconstructed $R_z$,
        xlabel={{$U\Lambda U^H$, }},
        zlabel=M×M,
        fill=\SoftmaxColor,
        opacity=0.8,
        height=16,
        width=3,
        depth=16
        }
    };

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Connections
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Forward path
\draw [connection]  (input-east)    -- node {\midarrow} (enc1-west);
\draw [connection]  (enc1_bn-east)  -- node {\midarrow} (pool1-west);
\draw [connection]  (pool1-east)    -- node {\midarrow} (enc2-west);
\draw [connection]  (enc2_bn-east)  -- node {\midarrow} (pool2-west);
\draw [connection]  (pool2-east)    -- node {\midarrow} (bottleneck-west);
\draw [connection]  (bottleneck-east) -- node {\midarrow} (upconv3-west);
\draw [connection]  (upconv3-east)  -- node {\midarrow} (dec3-west);
\draw [connection]  (dec3-east)     -- node {\midarrow} (upconv4-west);
\draw [connection]  (upconv4-east)  -- node {\midarrow} (dec4-west);
\draw [connection]  (dec4-east)     -- node {\midarrow} (unet_out-west);

% Skip connections
\draw [copyconnection]  (enc2_bn-east)  
    to[out=0,in=180] node {\copymidarrow} (dec3-west);
\draw [copyconnection]  (enc1_bn-east)  
    to[out=0,in=180] node {\copymidarrow} (dec4-west);

% Eigenvalue head connections
\draw [connection]  (unet_out-east) 
    to[out=0,in=180] node {\midarrow} (eig_pool-west);
\draw [connection]  (eig_pool-east) -- node {\midarrow} (eig_fc1-west);
\draw [connection]  (eig_fc1-east)  -- node {\midarrow} (eig_fc2-west);
\draw [connection]  (eig_fc2-east)  -- node {\midarrow} (eigenvals-west);

% Eigenvector head connections
\draw [connection]  (unet_out-east) 
    to[out=0,in=180] node {\midarrow} (eigvec_conv1-west);
\draw [connection]  (eigvec_conv1-east) -- node {\midarrow} (eigvec_conv2-west);
\draw [connection]  (eigvec_conv2-east) -- node {\midarrow} (eigvec_raw-west);
\draw [connection]  (eigvec_raw-east)   -- node {\midarrow} (eigvecs-west);

% Reconstruction connections
\draw [connection]  (eigenvals-east) 
    to[out=0,in=90] node {\midarrow} (recon_cov-north);
\draw [connection]  (eigvecs-east) 
    to[out=0,in=-90] node {\midarrow} (recon_cov-south);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Labels and Annotations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Title
\node[text width=20cm, align=center] at (15, 12, 0) {
    \Huge\textbf{EVD-Based Covariance Reconstruction UNet}
};

\node[text width=20cm, align=center] at (15, 10.5, 0) {
    \Large Eigenvalue Decomposition for DOA Estimation
};

% Branch labels
\node[text width=8cm, align=center, fill=white, rounded corners, draw=black] at (20, 6, 0) {
    \textbf{Eigenvalue Branch}\\
    Predicts $\lambda_1, \ldots, \lambda_M$
};

\node[text width=8cm, align=center, fill=white, rounded corners, draw=black] at (20, -6, 0) {
    \textbf{Eigenvector Branch}\\
    Predicts $U$ (orthogonalized via QR)
};

% Architecture info
\node[text width=15cm, align=left, fill=white, rounded corners, draw=black] at (15, -11, 0) {
    \textbf{Architecture Details:}\\
    • Input: Autocorrelation tensor [batch, $\tau$=8, 2M=16, M=8]\\
    • Base UNet: Encoder-Decoder with skip connections\\
    • Eigenvalue Head: Global pooling + FC layers (1→64→128→M)\\
    • Eigenvector Head: Conv layers (1→32→16→1) + QR decomposition\\
    • Output: $R_z = U \Lambda U^H$ [batch, M, M]
};

\end{tikzpicture}
\end{document}
"""

    # Write to file
    tex_file = output_path / f"{OUTPUT_NAME}.tex"
    with open(tex_file, 'w') as f:
        f.write(latex_code)
    
    print(f"✅ LaTeX file generated: {tex_file}")
    
    return tex_file


def compile_to_pdf(tex_file):
    """Compile LaTeX to PDF using pdflatex."""
    import subprocess
    
    print("\n🔨 Compiling LaTeX to PDF...")
    
    try:
        # Run pdflatex twice for proper rendering
        for i in range(2):
            result = subprocess.run(
                ['pdflatex', '-interaction=nonstopmode', str(tex_file)],
                cwd=tex_file.parent,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                print(f"⚠️ pdflatex warning (pass {i+1}/2)")
                if i == 1:  # Only show error on second pass
                    print(result.stdout[-500:])  # Last 500 chars
        
        pdf_file = tex_file.with_suffix('.pdf')
        if pdf_file.exists():
            print(f"✅ PDF generated: {pdf_file}")
            
            # Clean up auxiliary files
            for ext in ['.aux', '.log', '.out']:
                aux_file = tex_file.with_suffix(ext)
                if aux_file.exists():
                    aux_file.unlink()
            
            return pdf_file
        else:
            print("❌ PDF compilation failed")
            return None
            
    except FileNotFoundError:
        print("⚠️ pdflatex not found. Install LaTeX to generate PDF.")
        print("   LaTeX file is ready for manual compilation.")
        return None


def generate_simplified_ascii_diagram():
    """Generate a simple ASCII representation of the architecture."""
    
    # Create output directory
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    
    ascii_diagram = """
╔══════════════════════════════════════════════════════════════════════════════════╗
║         EVD-Based Covariance Reconstruction UNet Architecture                    ║
╚══════════════════════════════════════════════════════════════════════════════════╝

Input: Autocorrelation Rx(τ) [batch, 8, 16, 8]
   │
   ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            Base UNet (Encoder-Decoder)                          │
│                                                                                 │
│  ┌──────┐  Pool  ┌──────┐  Pool  ┌────────────┐  Up  ┌──────┐  Up  ┌──────┐ │
│  │Enc-1 │───────▶│Enc-2 │───────▶│ Bottleneck │─────▶│Dec-3 │─────▶│Dec-4 │ │
│  │ 16ch │        │ 32ch │        │   64ch     │      │ 32ch │      │ 16ch │ │
│  └──┬───┘        └──┬───┘        └────────────┘      └──┬───┘      └──┬───┘ │
│     │               │              ▲                     │              │      │
│     └───────────────┼──────────────┘ Skip connections ──┼──────────────┘      │
│                     └──────────────────────────────────────┘                   │
└────────────────────────────────────────┬───────────────────────────────────────┘
                                         │
                         UNet Features [batch, 8, 16, 8]
                                         │
                    ┌────────────────────┴────────────────────┐
                    │                                         │
                    ▼                                         ▼
        ╔═══════════════════════╗              ╔═══════════════════════════╗
        ║  Eigenvalue Head      ║              ║  Eigenvector Head         ║
        ╠═══════════════════════╣              ╠═══════════════════════════╣
        ║ • Global Pool         ║              ║ • Conv-32 (3×3)          ║
        ║ • FC: 1 → 64          ║              ║ • Conv-16 (3×3)          ║
        ║ • FC: 64 → 128        ║              ║ • Conv-1 (3×3)           ║
        ║ • FC: 128 → M (8)     ║              ║ • QR Decomposition       ║
        ║ • ReLU (positive)     ║              ║ • Orthogonalization      ║
        ╚═══════════╦═══════════╝              ╚═══════════╦═══════════════╝
                    │                                      │
                    ▼                                      ▼
           Eigenvalues λ                        Eigenvectors U
            [batch, M]                        [batch, M, M] complex
                    │                                      │
                    └──────────────┬───────────────────────┘
                                   │
                                   ▼
                        ┌─────────────────────┐
                        │ Reconstruction:     │
                        │   Rz = U Λ U^H     │
                        │   [batch, M, M]    │
                        └─────────────────────┘
                                   │
                                   ▼
                    Clean Covariance Matrix for DOA Estimation

╔══════════════════════════════════════════════════════════════════════════════════╗
║ Key Features:                                                                    ║
║ • Dual-head architecture: separate eigenvalue and eigenvector prediction         ║
║ • QR decomposition ensures orthonormal eigenvectors                             ║
║ • Eigenvalues sorted in descending order                                        ║
║ • Output: Clean covariance matrix Rz = U Λ U^H                                 ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""
    
    ascii_file = Path(OUTPUT_DIR) / f"{OUTPUT_NAME}_ascii.txt"
    with open(ascii_file, 'w') as f:
        f.write(ascii_diagram)
    
    print(f"\n📝 ASCII diagram saved: {ascii_file}")
    print(ascii_diagram)
    
    return ascii_file


def main():
    """Main execution function."""
    print("🎨 EVD UNet Architecture Visualization")
    print("=" * 80)
    
    # Check if PlotNeuralNet exists
    plotnn_path = Path(PLOTNEURALNET_PATH)
    if not plotnn_path.exists():
        print(f"\n⚠️  PlotNeuralNet not found at: {PLOTNEURALNET_PATH}")
        print("\n📥 To install PlotNeuralNet:")
        print("   git clone https://github.com/HarisIqbal88/PlotNeuralNet.git")
        print("\nGenerating without LaTeX visualization...")
        print("\n" + "="*80)
        
        # Generate ASCII diagram as fallback
        generate_simplified_ascii_diagram()
        
        print("\n💡 Install PlotNeuralNet and LaTeX for beautiful diagrams!")
        return
    
    # Generate LaTeX diagram
    print(f"\n✅ PlotNeuralNet found at: {plotnn_path}")
    tex_file = generate_evdunet_diagram()
    
    # Try to compile to PDF
    pdf_file = compile_to_pdf(tex_file)
    
    # Also generate ASCII diagram
    ascii_file = generate_simplified_ascii_diagram()
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 Summary:")
    print(f"   LaTeX file: {tex_file}")
    if pdf_file:
        print(f"   PDF file: {pdf_file}")
    print(f"   ASCII diagram: {ascii_file}")
    print("\n✅ Architecture visualization complete!")
    print("\n💡 Tip: Open the PDF or ASCII file to view the architecture diagram")


if __name__ == "__main__":
    main()

