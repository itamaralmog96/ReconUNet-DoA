# Array Imperfection Parameters - Complete Specification

This document provides the complete specification of all hardware imperfection parameters used in the dataset generation for DOA estimation, addressing reproducibility requirements.

## Hardware Imperfection Model

The array model includes three types of hardware imperfections:

### 1. Gain and Phase Errors

**Location**: `src/signalgen/array_processing.py`, `ArrayConfig` class (lines 36-38)

**Model**: Each antenna element has independent gain and phase errors applied through a diagonal matrix:
```
G = diag(g₁·e^(jφ₁), g₂·e^(jφ₂), ..., g_M·e^(jφ_M))
```

**Parameters**:
- **Gain Error Range**: `gain_error_range = (0.975, 1.025)`
  - Distribution: Uniform
  - Physical meaning: ±2.5% gain variation
  - Linear scale (not dB)
  - Applied per antenna element independently
  
- **Phase Error Range**: `phase_error_range_deg = (-5.0, 5.0)` degrees
  - Distribution: Uniform
  - Physical meaning: ±5° phase mismatch
  - Applied per antenna element independently
  
**Enable/Disable**: `enable_gain_phase_errors = True` (default)

**Implementation** (lines 211-222):
```python
gain_errors = rng.uniform(*gain_error_range, size=M)  # [g₁, ..., g_M]
phase_errors_rad = np.deg2rad(rng.uniform(*phase_error_range_deg, size=M))  # [φ₁, ..., φ_M]
G = diag(gain_errors * exp(1j * phase_errors))
```

---

### 2. Mutual Coupling

**Location**: `src/signalgen/array_processing.py`, `ArrayConfig` class (lines 40-43)

**Model**: Toeplitz mutual coupling matrix modeling electromagnetic coupling between array elements:
```
C = I + E_mc
```
where E_mc is a symmetric Toeplitz matrix with:
```
E[i,j] = γ^|i-j| · v_|i-j|,  for i ≠ j
E[i,i] = 0
```

**Parameters**:
- **Coupling Strength**: `coupling_strength = 0.3`
  - Base coupling coefficient magnitude |γ|
  - Unitless, represents fractional signal coupling
  - Physical meaning: 30% of signal from adjacent elements couples to each element
  
- **Coupling Phase**: `coupling_phase_deg = -100.0` degrees
  - Phase angle of base coupling coefficient ∠γ
  - Physical meaning: Phase shift in coupled signal
  - Combined: γ = 0.3 · e^(j·(-100°))
  
- **Coupling Variation**: `coupling_variation = 0.9`
  - Random variation factor: v_k ~ Uniform(1 - 0.9/2, 1 + 0.9/2) = Uniform(0.55, 1.45)
  - Physical meaning: ±90% manufacturing variation in coupling
  - One random factor per diagonal order k (shared across that diagonal to preserve Toeplitz structure)
  
**Enable/Disable**: `enable_mutual_coupling = True` (default)

**Implementation** (lines 224-266):
```python
gamma = coupling_strength * exp(1j * deg2rad(coupling_phase_deg))  # γ = 0.3·e^(-j100°)
v = rng.uniform(1 - variation/2, 1 + variation/2, size=M-1)  # [v₁, ..., v_{M-1}]
c[k] = gamma^k * v[k]  # Geometric decay with variation
E[i,j] = c[|i-j|]  for i ≠ j
C = I + E
```

**Geometric Decay**: Coupling decreases geometrically with element separation:
- 1st order (adjacent): c₁ = 0.3·e^(-j100°) · v₁
- 2nd order: c₂ = 0.09·e^(-j200°) · v₂
- kth order: c_k = 0.3^k·e^(-j100k°) · v_k

---

### 3. Position Errors

**Location**: `src/signalgen/array_processing.py`, `ArrayConfig` class (line 46)

**Model**: Additive Gaussian noise on element positions:
```
p_actual[i] = p_nominal[i] + ε_i
where ε_i ~ N(0, σ²I₂)  (2D Gaussian)
```

**Parameters**:
- **Position Error Std**: `position_error_std = 0.001` meters
  - Standard deviation: σ = 1 mm
  - Distribution: Gaussian/Normal, zero mean
  - Applied independently to x and y coordinates
  - Physical meaning: Mechanical placement error
  
**Relative to Wavelength** (at f_c = 2.45 GHz):
- Wavelength λ ≈ 0.122 meters (12.2 cm)
- Position error ≈ 1 mm = 0.0082λ (≈0.8% of wavelength)

**Implementation** (lines 205-209):
```python
errors = rng.normal(0, position_error_std, size=(M, 2))  # [ε_x, ε_y] per element
positions_actual = positions_nominal + errors
```

---

## Combined Imperfection Model

**Total Model**: The combined effect is applied as:
```
y_imperfect = G · C · y_ideal
```
where:
- G: Gain and phase error matrix (diagonal)
- C: Mutual coupling matrix (Toeplitz)
- y_ideal: Ideal received signal vector

**Implementation** (line 95):
```python
imperfection_matrix = gain_phase_matrix @ mutual_coupling_matrix
```

---

## Operating Frequencies and Physical Context

**Default Carrier Frequency**: `carrier_freq = 2.45 GHz` (ISM band)
- Wavelength λ = c/f ≈ 12.2 cm
- Default element spacing: 0.5λ ≈ 6.1 cm

**Imperfections in Physical Context**:
1. **Gain**: ±2.5% typical for commercial RF amplifiers
2. **Phase**: ±5° typical for antenna array calibration residuals
3. **Coupling**: 30% strength typical for half-wavelength spacing
4. **Position**: ±1 mm achievable with precision mechanical assembly

---

## Reproducibility Guarantees

**Seeding Mechanism**:
```python
ArrayModel(config, seed=42)  # Deterministic imperfections
```
- Each sample uses a deterministic seed
- Seeds stored in HDF5 dataset: `array_models/seed`
- Enables exact reproduction of any dataset sample

**Verification**:
```python
# Example: Reproduce exact imperfections
config = ArrayConfig(...)
array1 = ArrayModel(config, seed=12345)
array2 = ArrayModel(config, seed=12345)
assert np.allclose(array1.imperfection_matrix, array2.imperfection_matrix)
```

---

## Dataset Storage

All imperfection parameters are stored per sample in HDF5 format:

**Stored Fields** (`array_models/` group):
- `gain_error_range_min`, `gain_error_range_max`: (0.975, 1.025)
- `phase_error_range_deg_min`, `phase_error_range_deg_max`: (-5.0, 5.0)
- `coupling_strength`: 0.3
- `coupling_phase_deg`: -100.0
- `coupling_variation`: 0.9
- `position_error_std`: 0.001
- `seed`: Per-sample random seed for exact reproducibility
- `imperfection_matrix`: Complete M×M complex matrix (if `save_array_metadata=True`)

**Loading Example**:
```python
with h5py.File('dataset.h5', 'r') as f:
    gain_range = (f['array_models/gain_error_range_min'][idx],
                  f['array_models/gain_error_range_max'][idx])
    phase_range = (f['array_models/phase_error_range_deg_min'][idx],
                   f['array_models/phase_error_range_deg_max'][idx])
    coupling_str = f['array_models/coupling_strength'][idx]
    coupling_phase = f['array_models/coupling_phase_deg'][idx]
    coupling_var = f['array_models/coupling_variation'][idx]
    position_std = f['array_models/position_error_std'][idx]
    seed = f['array_models/seed'][idx]
```

---

## Summary Table

| Imperfection Type | Parameter | Symbol | Value | Distribution | Physical Unit |
|-------------------|-----------|--------|-------|--------------|---------------|
| **Gain Error** | Min gain | g_min | 0.975 | Uniform | Linear scale |
| | Max gain | g_max | 1.025 | Uniform | Linear scale |
| | Variation | - | ±2.5% | - | Percent |
| **Phase Error** | Min phase | φ_min | -5.0 | Uniform | Degrees |
| | Max phase | φ_max | +5.0 | Uniform | Degrees |
| **Mutual Coupling** | Strength | \|γ\| | 0.3 | Deterministic | Unitless |
| | Phase | ∠γ | -100.0 | Deterministic | Degrees |
| | Variation | v | ±90% | Uniform | Percent |
| | Variation range | - | [0.55, 1.45] | Uniform | Unitless |
| **Position Error** | Std deviation | σ_pos | 0.001 | Gaussian | Meters |
| | Relative to λ | - | 0.0082λ | Gaussian | Wavelengths |

---

## References

- Implementation: `Tri4Net/src/signalgen/array_processing.py`
- Dataset Generator: `Tri4Net/src/data/dataset_generator.py`
- Configuration: `ArrayConfig` dataclass (lines 23-58)
- Gain/Phase Implementation: `_generate_gain_phase_matrix()` (lines 211-222)
- Coupling Implementation: `_generate_mutual_coupling_matrix()` (lines 224-266)
- Position Implementation: `_add_position_errors()` (lines 205-209)

---

## Citation-Ready Format

For use in papers:

> **Hardware Imperfections**: We model three types of array imperfections: (1) gain and phase errors sampled uniformly from [0.975, 1.025] and [-5°, +5°] respectively, (2) mutual coupling with Toeplitz structure, base coefficient γ = 0.3·exp(-j100°), and ±90% random variation per diagonal order, and (3) Gaussian position errors with σ = 1 mm (≈0.008λ at 2.45 GHz). All imperfections are applied deterministically using per-sample random seeds stored in the dataset for exact reproducibility.

