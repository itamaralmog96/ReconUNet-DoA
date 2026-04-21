#!/usr/bin/env python3
"""
Batch Extract Plotted Values

This script processes multiple result JSON files at once, extracting plotted
values from all of them. Useful when you have multiple evaluation runs.

Usage:
    python batch_extract_plotted_values.py [directory]
    
    If directory is not specified, uses current directory.
    Processes all files matching: results_*.json (excluding *_plotted_values.json)
"""

import json
import sys
from pathlib import Path
from extract_plotted_values import extract_plotted_values


def batch_extract(directory='.', pattern='results_*.json'):
    """
    Batch extract plotted values from multiple JSON files.
    
    Parameters
    ----------
    directory : str or Path
        Directory to search for result files
    pattern : str
        Glob pattern for matching result files
    """
    dir_path = Path(directory)
    
    if not dir_path.exists():
        print(f"❌ Error: Directory not found: {directory}")
        return
    
    # Find all matching files (exclude already-processed files)
    all_files = list(dir_path.glob(pattern))
    result_files = [f for f in all_files if '_plotted_values' not in f.stem]
    
    if not result_files:
        print(f"❌ No result files found matching pattern: {pattern}")
        print(f"   in directory: {dir_path.absolute()}")
        return
    
    print(f"🔍 Found {len(result_files)} result file(s) in {dir_path}")
    print("=" * 70)
    
    success_count = 0
    error_count = 0
    
    for i, result_file in enumerate(result_files, 1):
        print(f"\n[{i}/{len(result_files)}] Processing: {result_file.name}")
        print("-" * 70)
        
        try:
            extract_plotted_values(result_file)
            success_count += 1
        except Exception as e:
            print(f"❌ Error processing {result_file.name}: {e}")
            error_count += 1
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 BATCH PROCESSING SUMMARY")
    print("=" * 70)
    print(f"   Total files processed: {len(result_files)}")
    print(f"   ✅ Successful: {success_count}")
    print(f"   ❌ Errors: {error_count}")
    print("=" * 70)


def main():
    """Main entry point."""
    if len(sys.argv) > 2:
        print(__doc__)
        print("\n❌ Error: Too many arguments")
        print("\nUsage:")
        print("    python batch_extract_plotted_values.py [directory]")
        sys.exit(1)
    
    directory = sys.argv[1] if len(sys.argv) == 2 else '.'
    
    print("🚀 Batch Plotted Values Extraction")
    print("=" * 70)
    
    batch_extract(directory)
    
    print("\n✨ Done!")


if __name__ == "__main__":
    main()
