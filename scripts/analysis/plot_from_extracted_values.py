#!/usr/bin/env python3
"""
Plot from Extracted Plotted Values

This script creates a plot directly from the extracted plotted values JSON,
demonstrating that it contains exactly what would be plotted by the 
controlled_angle_evaluation.py script.

Usage:
    python plot_from_extracted_values.py <plotted_values_json>
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path


def plot_extracted_values(json_path, output_path=None):
    """
    Create plot from extracted plotted values.
    
    Parameters
    ----------
    json_path : str or Path
        Path to JSON with extracted plotted values
    output_path : str or Path, optional
        Path to save plot. If None, uses input filename.
    """
    # Load data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extract SNR levels and algorithms
    snr_vals = sorted([float(k) for k in data.keys()])
    algorithms = list(data[str(int(snr_vals[0]))].keys())
    
    print(f"📊 Plotting {len(algorithms)} algorithms across {len(snr_vals)} SNR levels")
    print(f"   SNR range: {snr_vals[0]} to {snr_vals[-1]} dB")
    print(f"   Algorithms: {', '.join(algorithms)}")
    
    # Create figure
    plt.figure(figsize=(14, 8))
    
    # Algorithm styles (matching controlled_angle_evaluation.py)
    algorithm_styles = {
        'MUSIC': {'marker': 'o', 'color': 'blue', 'linewidth': 2},
        'MVDR': {'marker': 's', 'color': 'green', 'linewidth': 2},
        'Beamformer': {'marker': '^', 'color': 'purple', 'linewidth': 2},
        'ESPRIT': {'marker': 'd', 'color': 'orange', 'linewidth': 2},
        'UnitaryESPRIT': {'marker': 'p', 'color': 'brown', 'linewidth': 2},
        'RootMUSIC': {'marker': 'v', 'color': 'cyan', 'linewidth': 2},
        'UNet_MUSIC': {'marker': '*', 'color': 'red', 'linewidth': 3, 'linestyle': '--', 'markersize': 12},
        'UNet_MVDR': {'marker': '*', 'color': 'darkgreen', 'linewidth': 3, 'linestyle': '--', 'markersize': 12},
        'UNet_BEAMFORMER': {'marker': '*', 'color': 'darkviolet', 'linewidth': 3, 'linestyle': '--', 'markersize': 12},
        'UNet_ROOT_MUSIC': {'marker': '*', 'color': 'darkcyan', 'linewidth': 3, 'linestyle': '--', 'markersize': 12},
        'UNet_ESPRIT': {'marker': '*', 'color': 'darkorange', 'linewidth': 3, 'linestyle': '--', 'markersize': 12},
        'UNet_UNITARY_ESPRIT': {'marker': '*', 'color': 'darkred', 'linewidth': 3, 'linestyle': '--', 'markersize': 12},
        'CRLB': {'marker': 'x', 'color': 'black', 'linewidth': 2, 'linestyle': ':'}
    }
    
    # Plot each algorithm
    for alg in algorithms:
        values = []
        for snr in snr_vals:
            snr_key = str(int(snr)) if float(snr).is_integer() else str(snr)
            val = data[snr_key].get(alg)
            values.append(val if val is not None else np.nan)
        
        style = algorithm_styles.get(alg, {'marker': 'o', 'linewidth': 2})
        label = 'CRLB (Theoretical Bound)' if alg == 'CRLB' else alg
        plt.plot(snr_vals, values, label=label, **style)
    
    # Formatting
    plt.xlabel('SNR (dB)', fontsize=14)
    plt.ylabel('Error (degrees)', fontsize=14)
    plt.yscale('log')
    plt.title('DOA Estimation Performance vs SNR\n(Plotted from Extracted Values)', 
              fontsize=16, fontweight='bold')
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3, which='major')
    plt.grid(True, alpha=0.2, which='minor', linestyle='--')
    plt.minorticks_on()
    plt.tight_layout()
    
    # Save plot
    if output_path is None:
        input_path = Path(json_path)
        output_path = input_path.parent / f"{input_path.stem}_plot.png"
    
    plt.savefig(output_path, dpi=150)
    print(f"\n✅ Plot saved: {output_path}")
    plt.close()


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print(__doc__)
        print("\n❌ Error: Missing input JSON file")
        print("\nUsage:")
        print("    python plot_from_extracted_values.py <plotted_values_json>")
        sys.exit(1)
    
    input_json = sys.argv[1]
    
    # Check if input file exists
    if not Path(input_json).exists():
        print(f"❌ Error: Input file not found: {input_json}")
        sys.exit(1)
    
    # Create plot
    plot_extracted_values(input_json)
    
    print("\n✨ Done!")


if __name__ == "__main__":
    main()
