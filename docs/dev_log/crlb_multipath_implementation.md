# CRLB for Multipath Scenarios: Implementation Guide

## Overview

The Cramér-Rao Lower Bound (CRLB) provides a theoretical lower bound on the variance of unbiased estimators. For Direction of Arrival (DOA) estimation with multipath, the appropriate CRLB formula depends on **how we model the multipath**.

## Two Cases Implemented

### Case A: Uncorrelated Multipath (Decorrelated Model)

**When to use:**
- You apply **spatial smoothing** or other decorrelation techniques
- Multipath paths are treated as **independent sources**
- Applicable when paths are well-separated

**Model:**
```
y(t) = A * s(t) + e(t)
```
where:
- `A = [a(θ₀), a(θ₁), ..., a(θₖ₋₁)]` contains K steering vectors (direct + multipath)
- Each path has uncorrelated signal `s_k(t)`
- Full-rank signal covariance

**CRLB Formula:**

The Fisher Information Matrix for the angle vector **θ** = [θ₀, θ₁, ..., θₖ₋₁] is:

```
J = (2T/σ²) * Re{(D^H * Π_⊥ * D) ⊙ P^T}
```

where:
- `D = [d(θ₀), ..., d(θₖ₋₁)]` with `d(θ) = ∂a(θ)/∂θ`
- `Π_⊥ = I - A(A^H A)^(-1) A^H` projects onto signal subspace's orthogonal complement
- `P = diag(P₀, ..., Pₖ₋₁)` are signal powers
- `⊙` is Hadamard (element-wise) product

**CRLB for direct path angle θ₀:**
```
var(θ̂₀) ≥ [J⁻¹]₁₁
```

**Implementation:** `compute_crlb_multiple_sources()`

---

### Case B: Coherent Multipath (Same Waveform)

**When to use:**
- **No** decorrelation applied (raw specular multipath)
- All paths share the **same source waveform** s(t)
- You want to estimate **only θ₀** (direct path)
- Other paths are nuisance parameters

**Model:**
```
y(t) = s(t) * [α₀*a(θ₀) + Σ αₗ*a(θₗ)] + e(t)
      = s(t) * v(ψ) + e(t)
```
where:
- Single waveform `s(t)` (rank-1 signal subspace)
- `v(ψ)` is composite steering vector
- `ψ = [θ₀, Re(α), Im(α), θ₁, ..., θₗ]` (nuisance parameters)

**CRLB Formula (Schur Complement):**

The Fisher information for **θ₀ only**, after eliminating nuisance parameters:

```
J_θ₀ = (2T/σ²) * Re{g₀^H * Π_⊥ * (I - G(G^H Π_⊥ G)⁻¹ G^H Π_⊥) * Π_⊥ * g₀}
```

where:
- `Π_⊥ = I - vv^H/|v|²` projects onto v's orthogonal complement
- `g₀ = ∂v/∂θ₀` is gradient w.r.t. direct path angle
- `G` contains gradients w.r.t. nuisance parameters:
  - `∂v/∂Re(αₗ), ∂v/∂Im(αₗ)` for each multipath gain
  - `∂v/∂θₗ` for each multipath angle

**CRLB:**
```
var(θ̂₀) ≥ 1/J_θ₀
```

**Key insight:** The Schur complement term subtracts the information that can be "explained" by nuisance parameters. If nuisance derivatives span similar directions as g₀, the bound **increases** (harder to estimate θ₀).

**Implementation:** `compute_crlb_coherent_multipath()`

---

## Implementation Details

### Function Signatures

#### Case A (Decorrelated):
```python
def compute_crlb_multiple_sources(theta_rad_array, M, T, SNR_dB, steering_vectors=None):
    """
    Parameters
    ----------
    theta_rad_array : array-like
        All source angles [θ₀, θ₁, ..., θₖ₋₁] in radians
    M : int
        Number of array elements
    T : int
        Number of snapshots
    SNR_dB : float
        Signal-to-noise ratio in dB
    steering_vectors : ndarray, optional
        Pre-computed steering matrix [M x K]
    
    Returns
    -------
    crlb_vars : ndarray
        CRLB variance for each angle [K,]
    """
```

#### Case B (Coherent):
```python
def compute_crlb_coherent_multipath(theta0_rad, theta_multipath_rad, alpha_multipath, 
                                     M, T, SNR_dB, steering_vector_main=None, 
                                     steering_vectors_multipath=None):
    """
    Parameters
    ----------
    theta0_rad : float
        Direct path angle in radians
    theta_multipath_rad : array-like
        Multipath angles [θ₁, ..., θₗ] in radians
    alpha_multipath : array-like
        Complex multipath gains [α₁, ..., αₗ] (α₀=1 implicit)
    M : int
        Number of array elements
    T : int
        Number of snapshots
    SNR_dB : float
        Signal-to-noise ratio in dB
    steering_vector_main : ndarray, optional
        Steering vector for direct path [M,]
    steering_vectors_multipath : ndarray, optional
        Steering vectors for multipath [M x L]
    
    Returns
    -------
    crlb : float
        CRLB variance for θ₀
    """
```

### Automatic Selection Logic

In `evaluate_sample_with_algorithms()`, the code automatically selects the appropriate CRLB:

```python
num_multipath = int(sample['labels'].get('num_multipath', 0))
has_multipath = num_multipath > 0

if has_multipath:
    if len(true_doas_clean) == 1:
        # Single source + multipath → Use Case B (coherent)
        crlb = compute_crlb_coherent_multipath(...)
    else:
        # Multiple sources + multipath → Use Case A (decorrelated)
        crlb = compute_crlb_multiple_sources(...)
else:
    # No multipath → Standard CRLB
    if len(true_doas_clean) == 1:
        crlb = compute_crlb_single_source(...)
    else:
        crlb = compute_crlb_multiple_sources(...)
```

---

## Mathematical Derivations

### Case A Derivation

Starting from the deterministic signal model with K uncorrelated sources:

1. **Log-likelihood (conditional):**
   ```
   L(θ, P) = -MT log(σ²) - (1/σ²) Σₜ ||y(t) - A(θ)s(t)||²
   ```

2. **Fisher Information Matrix:**
   ```
   [J]ᵢⱼ = E[∂²L/∂θᵢ∂θⱼ] = (2T/σ²) Re{dᵢ^H Π_⊥ dⱼ} Pⱼ
   ```

3. **Matrix form:**
   ```
   J = (2T/σ²) Re{D^H Π_⊥ D ⊙ P^T}
   ```

### Case B Derivation (Schur Complement)

Starting with parameter vector ψ = [θ₀, nuisance]:

1. **Full FIM:**
   ```
   J_full = [J_θθ  J_θν]
            [J_νθ  J_νν]
   ```

2. **Schur complement for θ₀:**
   ```
   J_θ₀ = J_θθ - J_θν J_νν⁻¹ J_νθ
   ```

3. **In terms of gradients:**
   ```
   J_θ₀ = (2T/σ²) Re{g₀^H Π_⊥ (I - G(G^H Π_⊥ G)⁻¹ G^H Π_⊥) Π_⊥ g₀}
   ```

The Schur complement effectively removes the information contribution from nuisance parameters.

---

## Practical Considerations

### 1. Identifiability

**Case B can fail** when:
- Multipath angles are too close to direct path
- Gains are too similar (ambiguity between α and θ)
- Near endfire (90° or -90°) where ∂a/∂θ ≈ 0

When `G^H Π_⊥ G` is singular → CRLB = ∞ (not identifiable)

### 2. Multipath Parameters

In practice, multipath angles and gains are unknown. The implementation uses a representative model:
- **Angles:** Random uniform [0, 2π]
- **Gains:** Exponential decay `αₗ = 0.5^(l+1) * e^(jφₗ)`

This gives a **pessimistic** (higher) CRLB estimate, since actual multipath may be more favorable.

### 3. Computational Complexity

- **Case A:** O(K³) for K×K matrix inversion
- **Case B:** O((2L+L)³) = O(27L³) for nuisance elimination

### 4. When to Use Which?

| Scenario | Use Case | Why |
|----------|----------|-----|
| Spatial smoothing applied | A | Paths decorrelated |
| ESPRIT/Root-MUSIC | A | Assume uncorrelated |
| Raw MUSIC/Capon, single source | B | Coherent, want θ₀ only |
| Multiple sources + multipath | A | Too complex for Case B |
| Research comparison | Both | Shows effect of decorrelation |

---

## Validation & Testing

### Expected Behavior

1. **No multipath:**
   - Case A and standard CRLB should match

2. **With multipath:**
   - Case B (coherent) should give **higher** (worse) CRLB than Case A
   - Why? Coherent multipath is harder to resolve than decorrelated

3. **SNR dependence:**
   - CRLB ∝ 1/SNR (should decrease linearly on log-log plot)

4. **Snapshot dependence:**
   - CRLB ∝ 1/√T (inversely proportional to √T)

### Test Cases

```python
# Test 1: No multipath (both should match)
theta0 = 45°
num_multipath = 0
# Case A and single-source CRLB should be identical

# Test 2: Weak multipath (Case B >> Case A)
theta0 = 45°
theta_mp = [60°, 80°, 120°]  # Well-separated
alpha_mp = [0.1, 0.05, 0.03]  # Weak
# Case B should be higher but finite

# Test 3: Strong coherent multipath (Case B >>> Case A)
theta0 = 45°
theta_mp = [47°, 43°]  # Very close!
alpha_mp = [0.5, 0.5]  # Strong
# Case B should be much higher (near-ambiguity)
```

---

## Output Interpretation

When running `controlled_angle_evaluation.py` with multipath:

```
📊 Results Summary
====================================================================================
SNR (dB)  MUSIC  MVDR  ...  UNet_MUSIC  CRLB           
------------------------------------------------------------------------------------
-20.0     12.89  15.25 ...  11.51       3.27    ← Case B (with multipath)
-15.0     11.82  9.81  ...  9.87        1.83
...
```

- **CRLB column** now uses Case B for single source + multipath
- Values are **higher** than without multipath (as expected)
- Algorithms that exceed CRLB need improvement
- Algorithms below CRLB may be:
  - Using decorrelation (spatial smoothing)
  - Exploiting multipath constructively
  - Or have bias (CRLB is for unbiased estimators)

---

## References

1. **Standard CRLB**: Stoica & Nehorai (1989) "MUSIC, Maximum Likelihood, and Cramér-Rao Bound"
2. **Coherent signals**: Ottersten et al. (1992) "Covariance matching estimation techniques"
3. **Schur complement**: Kay (1993) "Fundamentals of Statistical Signal Processing: Estimation Theory"

---

## File Locations

- **Implementation**: `controlled_angle_evaluation.py` lines 280-420
- **Case A**: `compute_crlb_multiple_sources()` (line 302)
- **Case B**: `compute_crlb_coherent_multipath()` (line 344)
- **Auto-selection**: `evaluate_sample_with_algorithms()` (line 561)
