#!/usr/bin/env python3
"""
Extract Plotted Values from Controlled Angle Evaluation Results

This script takes the raw JSON results from controlled_angle_evaluation.py
and creates a new JSON containing only the aggregated values that are actually
plotted (RMSE for algorithms, mean for CRLB).

Input JSON format:
{
  "SNR_level": {
    "Algorithm": [error1, error2, error3, ...],  # Raw per-sample errors
    ...
  },
  ...
}

Output JSON format:
{
  "SNR_level": {
    "Algorithm": aggregated_value,  # Single value per algorithm per SNR
    ...
  },
  ...
}

Usage:
    python extract_plotted_values.py <input_json> [output_json]
    
    If output_json is not specified, it will be named:
    <input_basename>_plotted_values.json
"""

import json
import numpy as np
import sys
from pathlib import Path


def compute_aggregated_value(values, is_crlb=False):
    """
    Compute aggregated value for plotting.
    
    Parameters
    ----------
    values : list
        List of error values (may contain None, inf, nan)
    is_crlb : bool
        If True, use mean aggregation; if False, use RMS aggregation
        
    Returns
    -------
    float or None
        Aggregated value, or None if no valid values
    """
    # Filter out None, inf, and nan values
    valid_vals = [v for v in values if v is not None and not np.isinf(v) and not np.isnan(v)]
    
    if len(valid_vals) == 0:
        return None
    
    valid_array = np.array(valid_vals)
    
    if is_crlb:
        # CRLB: Simple mean (theoretical bound averaging)
        return float(np.mean(valid_array))
    else:
        # Algorithms: RMS aggregation
        # RMSE = sqrt(mean(errors²))
        return float(np.sqrt(np.mean(valid_array**2)))


def extract_plotted_values(input_json_path, output_json_path=None):
    """
    Extract plotted values from raw results JSON.
    
    Parameters
    ----------
    input_json_path : str or Path
        Path to input JSON file with raw per-sample errors
    output_json_path : str or Path, optional
        Path to output JSON file. If None, auto-generates name.
        
    Returns
    -------
    dict
        Dictionary with aggregated values
    """
    input_path = Path(input_json_path)
    
    # Load input JSON
    print(f"📖 Loading: {input_path}")
    with open(input_path, 'r') as f:
        raw_results = json.load(f)
    
    # Process each SNR level
    plotted_values = {}
    # Keep original string keys but sort numerically
    snr_keys = list(raw_results.keys())
    snr_levels = sorted([float(k) for k in snr_keys])
    
    print(f"\n📊 Processing {len(snr_levels)} SNR levels...")
    
    # Get list of algorithms from first SNR level (use original key)
    first_snr_key = snr_keys[0]
    algorithms = list(raw_results[first_snr_key].keys())
    
    print(f"   Algorithms found: {', '.join(algorithms)}")
    
    # Compute statistics
    total_samples_per_snr = {}
    failure_counts = {alg: 0 for alg in algorithms}
    total_samples = {alg: 0 for alg in algorithms}
    
    for snr_key in snr_keys:
        snr = float(snr_key)
        
        if snr_key not in raw_results:
            print(f"⚠️  Warning: SNR {snr_key} not found in input JSON")
            continue
        
        plotted_values[snr_key] = {}
        
        for alg in algorithms:
            if alg not in raw_results[snr_key]:
                print(f"⚠️  Warning: Algorithm {alg} not found for SNR {snr}")
                continue
            
            raw_values = raw_results[snr_key][alg]
            
            # Count failures and valid samples
            num_samples = len(raw_values)
            valid_vals = [v for v in raw_values if v is not None and not np.isinf(v) and not np.isnan(v)]
            num_valid = len(valid_vals)
            num_failures = num_samples - num_valid
            
            total_samples[alg] += num_samples
            failure_counts[alg] += num_failures
            
            # Track samples per SNR
            if snr_key not in total_samples_per_snr:
                total_samples_per_snr[snr_key] = num_samples
            
            # Compute aggregated value
            is_crlb = (alg == 'CRLB')
            aggregated = compute_aggregated_value(raw_values, is_crlb)
            
            plotted_values[snr_key][alg] = aggregated
    
    # Print summary
    print(f"\n📈 Summary:")
    print(f"   Total SNR levels: {len(plotted_values)}")
    print(f"   Samples per SNR: {total_samples_per_snr}")
    
    print(f"\n   Algorithm Statistics:")
    for alg in algorithms:
        total = total_samples[alg]
        failures = failure_counts[alg]
        success_rate = 100 * (total - failures) / total if total > 0 else 0
        print(f"      {alg:20s}: {total:4d} samples, {failures:4d} failures ({100-success_rate:.1f}% failure rate)")
    
    # Save output
    if output_json_path is None:
        # Auto-generate output filename with _plotted_values suffix
        output_path = input_path.parent / f"{input_path.stem}_plotted_values.json"
    else:
        output_path = Path(output_json_path)
        # Ensure output is different from input (prevent overwriting)
        if output_path.resolve() == input_path.resolve():
            print(f"⚠️  Warning: Output path same as input, adding '_plotted_values' suffix")
            output_path = input_path.parent / f"{input_path.stem}_plotted_values.json"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(plotted_values, f, indent=2)
    
    print(f"\n✅ Plotted values saved: {output_path}")
    
    return plotted_values


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print(__doc__)
        print("\n❌ Error: Missing input JSON file")
        print("\nUsage:")
        print("    python extract_plotted_values.py <input_json> [output_json]")
        sys.exit(1)
    
    input_json = sys.argv[1]
    output_json = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Check if input file exists
    if not Path(input_json).exists():
        print(f"❌ Error: Input file not found: {input_json}")
        sys.exit(1)
    
    # Extract and save plotted values
    extract_plotted_values(input_json, output_json)
    
    print("\n✨ Done!")


if __name__ == "__main__":
    main()
