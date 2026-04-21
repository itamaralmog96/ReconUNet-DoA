import sys
sys.path.insert(0, 'src')
import numpy as np
from signalgen import ArrayConfig, ArrayModel

# Test [6,0] case: 6 main sources, no interference
num_sources = 6
M = 8
T = 512
SNR_dB = 20.0

# 6 sources with 10° separation
source_angles = [30, 50, 70, 90, 110, 130]
print(f"Testing 6 main sources at: {source_angles}°")

config = ArrayConfig(array_type='linear', num_elements=M, carrier_freq=6e9, element_spacing=0.5)
array_model = ArrayModel(config)

steering_vecs = array_model.steering_matrix(source_angles, nominal=True)
theta_rad_array = np.deg2rad(source_angles)

# Compute CRLB
sigma2 = 10**(-SNR_dB / 10)
K = len(theta_rad_array)
A = steering_vecs.copy()

m = np.arange(M, dtype=float)
D = np.zeros((M, K), dtype=complex)
for k, th in enumerate(theta_rad_array):
    D[:, k] = (-1j * np.pi * m * np.sin(th)) * A[:, k]

I = np.eye(M, dtype=complex)
AH_A = A.conj().T @ A
AH_A_inv = np.linalg.pinv(AH_A)
Pi_perp = I - A @ AH_A_inv @ A.conj().T

G = D.conj().T @ Pi_perp @ D
G = (G + G.conj().T) / 2
G_real = np.real(G)

J_diag = np.diag(np.diag(G_real))
J_matrix = (2.0 * T / sigma2) * J_diag

diag_J = np.diag(J_matrix)
crlb_vars = np.where(diag_J > 0, 1.0 / diag_J, np.inf)
crlb_deg = np.sqrt(crlb_vars) * (180/np.pi)

print(f"\nCRLB per source:")
for i, (angle, crlb) in enumerate(zip(source_angles, crlb_deg)):
    print(f"  Source {i+1} at {angle}°: {crlb:.6f}°")

print(f"\nCurrent approach (average variances then sqrt):")
avg_var = np.mean(crlb_vars)
print(f"  CRLB = √(mean(variances)) = {np.sqrt(avg_var) * (180/np.pi):.6f}°")

print(f"\nAlternative: RMS across all sources:")
print(f"  RMS = √(mean(errors²)) across all {K} sources")
print(f"  This is: √(mean([CRLB₁², CRLB₂², ..., CRLB₆²]))")
rms_crlb = np.sqrt(np.mean(crlb_deg**2))
print(f"  RMS CRLB = {rms_crlb:.6f}°")

print(f"\nThey're the same: {np.isclose(np.sqrt(avg_var) * (180/np.pi), rms_crlb)}")

