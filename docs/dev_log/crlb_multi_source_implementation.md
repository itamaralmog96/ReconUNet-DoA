# Multi-Source CRLB Implementation

## Overview

The `crlb_evaluation.py` script now supports **both single-source and multi-source** CRLB computation. It automatically detects the number of sources from the `num_sources` configuration and uses the appropriate formulation.

## Implementation Details

### Single Source (K=1)

Uses three different CRLB formulations:
1. **Stochastic (Slepian-Bangs)**: R = a·a^H + σ²I
2. **Conditional (Deterministic)**: J = 2T·(P/σ²)·d^H·(I - a·a^H)·d
3. **Closed-form ULA**: Analytical formula for uniform linear arrays

### Multiple Sources (K>1)

Uses the **conditional/deterministic multi-source CRLB**:

#### Mathematical Formulation

For K sources with angles **θ** = [θ₁, ..., θₖ]:

- **A** = [a₁, ..., aₖ] — Steering matrix with unit-norm columns
- **D** = [d₁, ..., dₖ] — Derivative matrix, dₖ = ∂a(θₖ)/∂θ
- **Π⊥** = I - A(A^H A)^{-1}A^H — Projector orthogonal to ALL steering vectors
- **P** = diag(P₁, ..., Pₖ) — Source covariance (diagonal for uncorrelated)

**Fisher Information Matrix (K×K):**
```
J_cond(θ) = (2T/σ²) * Re[(D^H Π⊥ D) ⊙ P^T]
```
where ⊙ is the Hadamard (element-wise) product.

**CRLB Matrix:**
```
CRLB(θ) = J_cond^{-1}
```

Per-angle variance bounds are the **diagonal elements** of CRLB(θ).

#### Key Properties

1. **Coupling:** Sources are coupled through the projector Π⊥ even when uncorrelated
2. **Degradation:** Close sources degrade the bound significantly
3. **Singularity:** When sources are too close, the matrix becomes singular (infinite CRLB)

## Test Results

From `test_multisource_crlb.py` (SNR=10dB, M=8, T=512):

| Configuration | RMSE Lower Bound |
|--------------|------------------|
| Single source at 60° | 0.0908° |
| Two sources 60° apart (50°, 110°) | 0.0959° (avg) |
| Two sources 10° apart (70°, 80°) | 0.1488° (avg) |
| Two sources 3° apart (70°, 73°) | 0.5298° (avg) ⚠️ |
| Three sources (50°, 80°, 110°) | 0.0970° (avg) |

**Note:** Very close sources (3° separation) degrade the CRLB by ~6x compared to single source!

## Configuration

To use multi-source CRLB, set `num_sources` in `crlb_evaluation.py`:

```python
# Single source (default)
num_sources = [[1, 0]]  # 1 main source, 0 interference

# Two sources
num_sources = [[2, 0]]  # 2 main sources, 0 interference

# Three sources
num_sources = [[3, 0]]  # 3 main sources, 0 interference
```

The script will automatically:
1. Detect the number of sources
2. Generate appropriate datasets
3. Compute CRLB using the correct formulation
4. Report averaged bounds for multi-source cases

## Output

For multi-source scenarios, the script reports:
- **Average CRLB** over all sources
- All three CRLB "methods" show the same value (conditional)
- Algorithm performance (MUSIC, ESPRIT, etc.) evaluated against all sources

## Functions

### `compute_crlb_multiple_sources()`

```python
def compute_crlb_multiple_sources(theta_rad_array, M, T, SNR_dB, 
                                 steering_vectors=None, 
                                 source_powers=None, 
                                 method='conditional'):
    """
    Compute multi-source conditional CRLB.
    
    Args:
        theta_rad_array: Array of K angles in radians
        M: Number of array elements
        T: Number of snapshots
        SNR_dB: Signal-to-Noise Ratio (same for all sources)
        steering_vectors: Optional [M, K] steering matrix
        source_powers: Optional [K] source power array (default: ones)
        
    Returns:
        Array of K CRLB values (rad²) for each source
    """
```

### `compute_crlb_single_source()`

Original function, unchanged, for single-source scenarios.

## References

1. **Stoica & Nehorai (1989)** - "MUSIC, Maximum Likelihood, and Cramér-Rao Bound"
2. **Van Trees (2002)** - "Optimum Array Processing", Chapter 8
3. **Slepian & Bangs (1954)** - Original stochastic CRLB formulation

## Validation

Run the test suite:
```bash
python test_multisource_crlb.py
```

This verifies:
- Single-source consistency
- Multi-source degradation with separation
- Three-source scenarios
- Numerical stability
