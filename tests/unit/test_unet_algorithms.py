#!/usr/bin/env python3
"""
Quick test script to verify all UNet follow-up algorithms work correctly.
Tests each algorithm with a small number of samples.
"""

import subprocess
import sys
from pathlib import Path

def update_algorithm_config(algorithm_name):
    """Update the UNET_FOLLOWUP_ALGORITHM in the config file."""
    config_file = Path("controlled_angle_evaluation.py")
    content = config_file.read_text()
    
    # Find and replace the line
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'UNET_FOLLOWUP_ALGORITHM' in line and '=' in line and not line.strip().startswith('#'):
            # Preserve indentation and comments
            indent = len(line) - len(line.lstrip())
            comment_part = ''
            if '#' in line:
                comment_part = ' ' + line.split('#', 1)[1]
            lines[i] = ' ' * indent + f"UNET_FOLLOWUP_ALGORITHM = '{algorithm_name}'" + comment_part
            break
    
    config_file.write_text('\n'.join(lines))
    print(f"✓ Updated configuration to use: {algorithm_name}")

def run_evaluation(algorithm_name):
    """Run evaluation with the specified algorithm."""
    print(f"\n{'='*70}")
    print(f"Testing UNet + {algorithm_name}")
    print(f"{'='*70}\n")
    
    # Update config
    update_algorithm_config(algorithm_name)
    
    # Run evaluation with minimal samples
    cmd = [
        sys.executable,
        "controlled_angle_evaluation.py",
        "--mode", "4",  # Mode 4: Controlled angle evaluation
        "--num_samples", "5"  # Just 5 samples for quick test
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            print(f"✅ SUCCESS: UNet + {algorithm_name} works correctly!")
            
            # Extract key results from output
            for line in result.stdout.split('\n'):
                if 'UNet_' in line and algorithm_name in line:
                    print(f"   {line.strip()}")
                elif 'Results Summary' in line:
                    # Print summary section
                    print(f"\n   {line.strip()}")
            
            return True
        else:
            print(f"❌ FAILED: UNet + {algorithm_name} returned error")
            print(f"   Error output:")
            for line in result.stderr.split('\n')[-10:]:  # Last 10 lines
                if line.strip():
                    print(f"   {line}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"❌ TIMEOUT: UNet + {algorithm_name} took too long")
        return False
    except Exception as e:
        print(f"❌ EXCEPTION: {e}")
        return False

def main():
    """Test all six UNet follow-up algorithms."""
    print("="*70)
    print("TESTING ALL UNET FOLLOW-UP ALGORITHMS")
    print("="*70)
    print("\nThis will test each algorithm with 5 samples to verify functionality.")
    print("Full evaluation should be done separately with more samples.\n")
    
    algorithms = ['MUSIC', 'MVDR', 'BEAMFORMER', 'ROOT_MUSIC', 'ESPRIT', 'UNITARY_ESPRIT']
    results = {}
    
    for algorithm in algorithms:
        success = run_evaluation(algorithm)
        results[algorithm] = success
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    all_passed = True
    for algorithm, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{algorithm:15} {status}")
        if not success:
            all_passed = False
    
    print("="*70)
    
    if all_passed:
        print("\n🎉 All UNet follow-up algorithms are working correctly!")
        return 0
    else:
        print("\n⚠️  Some algorithms failed. Check errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
