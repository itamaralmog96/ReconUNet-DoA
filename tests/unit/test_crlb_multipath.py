#!/usr/bin/env python3
"""Test CRLB implementations for multipath scenarios."""

import numpy as np
import sys
sys.path.insert(0, 'src')

from controlled_angle_evaluation import (
    compute_crlb_single_source,
    compute_crlb_multiple_sources, 
    compute_crlb_coherent_multipath
)

print('🧪 Testing CRLB Implementations')
print('=' * 70)

# Parameters
M = 8  # Array elements
T = 512  # Snapshots
SNR_dB = 10  # 10 dB
theta0_deg = 45
theta0_rad = np.deg2rad(theta0_deg)

# ULA steering vector
def ula_steering(theta_rad, M):
    m = np.arange(M)
    return (1.0 / np.sqrt(M)) * np.exp(1j * np.pi * m * np.cos(theta_rad))

a0 = ula_steering(theta0_rad, M)

print(f'\n📐 Test Setup:')
print(f'   Array: {M} elements ULA')
print(f'   Snapshots: {T}')
print(f'   SNR: {SNR_dB} dB')
print(f'   Direct path angle: {theta0_deg}°')

# Test 1: Single source, no multipath
print(f'\n📊 Test 1: Single source, NO multipath')
print('-' * 70)
crlb_var_single = compute_crlb_single_source(theta0_rad, M, T, SNR_dB, a0)
crlb_deg_single = np.sqrt(crlb_var_single) * (180/np.pi)
print(f'   CRLB (standard): {crlb_deg_single:.4f}°')

# Test 2: Single source with weak multipath (Case B)
print(f'\n📊 Test 2: Single source with WEAK coherent multipath (Case B)')
print('-' * 70)
num_mp = 3
theta_mp_deg = [60, 80, 120]
theta_mp_rad = np.deg2rad(theta_mp_deg)
alpha_mp = np.array([0.1 * np.exp(1j*np.pi/4), 
                     0.05 * np.exp(1j*np.pi/3), 
                     0.03 * np.exp(1j*np.pi/6)])

A_mp = np.zeros((M, num_mp), dtype=complex)
for l in range(num_mp):
    A_mp[:, l] = ula_steering(theta_mp_rad[l], M)

crlb_var_coherent_weak = compute_crlb_coherent_multipath(
    theta0_rad, theta_mp_rad, alpha_mp, M, T, SNR_dB, a0, A_mp
)
crlb_deg_coherent_weak = np.sqrt(crlb_var_coherent_weak) * (180/np.pi)
print(f'   Multipath angles: {theta_mp_deg}°')
print(f'   Multipath gains: |alpha| = {[f"{abs(a):.2f}" for a in alpha_mp]}')
print(f'   CRLB (coherent multipath): {crlb_deg_coherent_weak:.4f}°')
print(f'   Ratio vs no multipath: {crlb_deg_coherent_weak/crlb_deg_single:.2f}x')

# Test 3: Single source with strong/close multipath (Case B)
print(f'\n📊 Test 3: Single source with STRONG/CLOSE coherent multipath (Case B)')
print('-' * 70)
num_mp_close = 2
theta_mp_close_deg = [47, 43]  # Very close to 45 degrees!
theta_mp_close_rad = np.deg2rad(theta_mp_close_deg)
alpha_mp_close = np.array([0.5 * np.exp(1j*np.pi/4), 
                           0.5 * np.exp(1j*np.pi/3)])

A_mp_close = np.zeros((M, num_mp_close), dtype=complex)
for l in range(num_mp_close):
    A_mp_close[:, l] = ula_steering(theta_mp_close_rad[l], M)

crlb_var_coherent_strong = compute_crlb_coherent_multipath(
    theta0_rad, theta_mp_close_rad, alpha_mp_close, M, T, SNR_dB, a0, A_mp_close
)
crlb_deg_coherent_strong = np.sqrt(crlb_var_coherent_strong) * (180/np.pi)
print(f'   Multipath angles: {theta_mp_close_deg}° (only 2° separation each side)')
print(f'   Multipath gains: |alpha| = {[f"{abs(a):.2f}" for a in alpha_mp_close]}')
print(f'   CRLB (coherent multipath): {crlb_deg_coherent_strong:.4f}°')
print(f'   Ratio vs no multipath: {crlb_deg_coherent_strong/crlb_deg_single:.2f}x')

# Test 4: Decorrelated model (Case A) - same angles as Test 2
print(f'\n📊 Test 4: Multiple DECORRELATED sources (Case A, for comparison)')
print('-' * 70)
all_angles_deg = [theta0_deg] + theta_mp_deg
all_angles_rad = np.deg2rad(all_angles_deg)
A_all = np.zeros((M, len(all_angles_rad)), dtype=complex)
for k, theta in enumerate(all_angles_rad):
    A_all[:, k] = ula_steering(theta, M)

crlb_vars_decorr = compute_crlb_multiple_sources(all_angles_rad, M, T, SNR_dB, A_all)
crlb_deg_decorr = np.sqrt(crlb_vars_decorr[0]) * (180/np.pi)  # CRLB for first source
print(f'   All angles treated as uncorrelated: {all_angles_deg}°')
print(f'   CRLB for first source (decorrelated): {crlb_deg_decorr:.4f}°')
print(f'   Ratio vs Case B (coherent): {crlb_deg_decorr/crlb_deg_coherent_weak:.2f}x')

print(f'\n✅ Summary:')
print('=' * 70)
print(f'No multipath:                {crlb_deg_single:.4f}°')
print(f'Weak coherent multipath:     {crlb_deg_coherent_weak:.4f}° ({crlb_deg_coherent_weak/crlb_deg_single:.2f}x worse)')
print(f'Strong/close multipath:      {crlb_deg_coherent_strong:.4f}° ({crlb_deg_coherent_strong/crlb_deg_single:.2f}x worse)')
print(f'Decorrelated (Case A):       {crlb_deg_decorr:.4f}° ({crlb_deg_decorr/crlb_deg_single:.2f}x vs no MP)')
print()
print('💡 Observation: Coherent multipath (Case B) gives HIGHER (worse) CRLB')
print('   than decorrelated (Case A), especially when paths are close!')
