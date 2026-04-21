#!/usr/bin/env python3
"""
Test script to verify multi-source CRLB computation
"""
import sys
from pathlib import Path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

import numpy as np
from crlb_evaluation import (
    compute_crlb_single_source,
    compute_crlb_multiple_sources,
    unit_norm_steering_vector,
    unit_norm_steering_vector_derivative
)

print("=" * 60)
print("Testing Multi-Source CRLB Implementation")
print("=" * 60)

# Test parameters
M = 8  # Array elements
T = 512  # Snapshots
SNR_dB = 10  # SNR in dB

# Test 1: Single source (compare with original implementation)
print("\n📍 Test 1: Single Source")
theta1 = np.radians(60)
crlb_single = compute_crlb_single_source(theta1, M, T, SNR_dB, method='conditional')
print(f"   Single source at 60°: CRLB = {crlb_single:.6e} rad²")
print(f"   RMSE LB = {np.sqrt(crlb_single) * 180/np.pi:.4f}°")

# Also test with multi-source function (should give same result)
crlb_multi_single = compute_crlb_multiple_sources([theta1], M, T, SNR_dB)
print(f"   Same source via multi-source function: CRLB = {crlb_multi_single[0]:.6e} rad²")
print(f"   RMSE LB = {np.sqrt(crlb_multi_single[0]) * 180/np.pi:.4f}°")
print(f"   ✓ Match: {np.allclose(crlb_single, crlb_multi_single[0])}")

# Test 2: Two sources with large separation (should behave independently)
print("\n📍 Test 2: Two Sources - Large Separation (60° apart)")
theta_array = np.radians([50, 110])
crlb_two_far = compute_crlb_multiple_sources(theta_array, M, T, SNR_dB)
print(f"   Source 1 at 50°: CRLB = {crlb_two_far[0]:.6e} rad²")
print(f"   Source 2 at 110°: CRLB = {crlb_two_far[1]:.6e} rad²")
print(f"   RMSE LB = [{np.sqrt(crlb_two_far[0])*180/np.pi:.4f}°, {np.sqrt(crlb_two_far[1])*180/np.pi:.4f}°]")
print(f"   Average RMSE LB = {np.sqrt(np.mean(crlb_two_far))*180/np.pi:.4f}°")

# Test 3: Two sources with close separation (should degrade)
print("\n📍 Test 3: Two Sources - Close Separation (10° apart)")
theta_array = np.radians([70, 80])
crlb_two_close = compute_crlb_multiple_sources(theta_array, M, T, SNR_dB)
print(f"   Source 1 at 70°: CRLB = {crlb_two_close[0]:.6e} rad²")
print(f"   Source 2 at 80°: CRLB = {crlb_two_close[1]:.6e} rad²")
print(f"   RMSE LB = [{np.sqrt(crlb_two_close[0])*180/np.pi:.4f}°, {np.sqrt(crlb_two_close[1])*180/np.pi:.4f}°]")
print(f"   Average RMSE LB = {np.sqrt(np.mean(crlb_two_close))*180/np.pi:.4f}°")

# Test 4: Two sources with very close separation (should degrade significantly)
print("\n📍 Test 4: Two Sources - Very Close Separation (3° apart)")
theta_array = np.radians([70, 73])
crlb_two_veryclose = compute_crlb_multiple_sources(theta_array, M, T, SNR_dB)
print(f"   Source 1 at 70°: CRLB = {crlb_two_veryclose[0]:.6e} rad²")
print(f"   Source 2 at 73°: CRLB = {crlb_two_veryclose[1]:.6e} rad²")
print(f"   RMSE LB = [{np.sqrt(crlb_two_veryclose[0])*180/np.pi:.4f}°, {np.sqrt(crlb_two_veryclose[1])*180/np.pi:.4f}°]")
print(f"   Average RMSE LB = {np.sqrt(np.mean(crlb_two_veryclose))*180/np.pi:.4f}°")

# Test 5: Three sources
print("\n📍 Test 5: Three Sources")
theta_array = np.radians([50, 80, 110])
crlb_three = compute_crlb_multiple_sources(theta_array, M, T, SNR_dB)
print(f"   Source 1 at 50°: CRLB = {crlb_three[0]:.6e} rad²")
print(f"   Source 2 at 80°: CRLB = {crlb_three[1]:.6e} rad²")
print(f"   Source 3 at 110°: CRLB = {crlb_three[2]:.6e} rad²")
print(f"   RMSE LB = [{np.sqrt(crlb_three[0])*180/np.pi:.4f}°, "
      f"{np.sqrt(crlb_three[1])*180/np.pi:.4f}°, {np.sqrt(crlb_three[2])*180/np.pi:.4f}°]")
print(f"   Average RMSE LB = {np.sqrt(np.mean(crlb_three))*180/np.pi:.4f}°")

print("\n" + "=" * 60)
print("✅ Multi-Source CRLB Tests Complete")
print("=" * 60)
