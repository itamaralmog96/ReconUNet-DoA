#!/usr/bin/env python3
"""
Quick runner script for comprehensive DOA evaluation

Usage:
    python run_evaluation.py                    # Quick test
    python run_evaluation.py --full            # Full evaluation
    python run_evaluation.py --config custom.yaml  # Custom configuration
"""

import argparse
import subprocess
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Run DOA Algorithm Evaluation')
    parser.add_argument('--quick', action='store_true', 
                       help='Run quick test (default)')
    parser.add_argument('--full', action='store_true',
                       help='Run full evaluation with default config')
    parser.add_argument('--config', type=str,
                       help='Path to custom configuration file')
    parser.add_argument('--unet-model', type=str,
                       help='Path to trained UNet model')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Build command
    cmd = [sys.executable, 'comprehensive_doa_evaluation.py']
    
    if args.config:
        cmd.extend(['--config', args.config])
    elif args.full:
        cmd.extend(['--config', 'evaluation_config.yaml'])
    else:
        # Default to quick test
        cmd.append('--quick-test')
    
    if args.unet_model:
        cmd.extend(['--unet-model', args.unet_model])
    
    cmd.extend(['--output-dir', args.output_dir])
    
    print(f"🚀 Running: {' '.join(cmd)}")
    print("=" * 60)
    
    # Run the evaluation
    try:
        result = subprocess.run(cmd, check=True)
        print("\n✅ Evaluation completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Evaluation failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\n⚠️ Evaluation interrupted by user")
        sys.exit(1)

if __name__ == "__main__":
    main()



