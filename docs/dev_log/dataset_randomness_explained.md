# Dataset Randomness Behavior

## Current Configuration

```python
DATASET_GENERATION_SEED = int(time.time())  # Line 122 in controlled_angle_evaluation.py
```

## Answer: YES, the dataset is random each time

### What Gets Randomized Each Run?

With `DATASET_GENERATION_SEED = int(time.time())`, the seed changes every second, so **each run generates a completely different dataset**:

1. **Angles** 🎯
   - Mode 3: Random angles within (30°, 150°) range
   - Different integer angles each run (e.g., 45°, 78°, 120° vs 33°, 89°, 145°)

2. **Multipath Components** 🌊
   - Random multipath angles (scattered paths)
   - Random delays, gains, and phases
   - Different for each run

3. **Noise** 🔊
   - Random noise realizations for each SNR
   - Different noise patterns each run

4. **Array Imperfections** 📡
   - Random gain/phase errors
   - Random mutual coupling
   - Different per run (if `ARRAY_IMPERFECTIONS = True`)

### Example: Two Consecutive Runs

**Run 1** (seed = 1760165482):
```
Sample 1: DOA = 105°
Sample 2: DOA = 94°
Sample 3: DOA = 140°
```

**Run 2** (seed = 1760165483):
```
Sample 1: DOA = 78°
Sample 2: DOA = 133°
Sample 3: DOA = 51°
```

## How to Make Dataset Reproducible

### Option 1: Fixed Integer Seed (Recommended for repeatability)

```python
DATASET_GENERATION_SEED = 42  # Or any fixed integer
```

**Result**: 
- ✅ Same angles every run
- ✅ Same noise every run
- ✅ Same multipath every run
- ✅ Perfect reproducibility for debugging/comparison

### Option 2: Keep Current Time-Based Seed (Good for diversity)

```python
DATASET_GENERATION_SEED = int(time.time())  # Current setting
```

**Result**:
- ✅ Different dataset every run
- ✅ Tests algorithm robustness across many scenarios
- ⚠️ Cannot reproduce exact same results
- ⚠️ Need to save seed if you want to regenerate

### Option 3: Random Each Time

```python
DATASET_GENERATION_SEED = None
```

**Result**:
- ✅ Uses system random seed
- ✅ Maximum randomness
- ⚠️ No reproducibility at all

## What the Seed Controls

The `DATASET_GENERATION_SEED` initializes the random number generator (`self.rng`) which controls:

### 1. Angle Generation (Mode 3)
```python
# Line 1416 in dataset_generator.py
main_angle = float(self.rng.integers(int(angle_range[0]), int(angle_range[1]) + 1))
```

### 2. Per-Sample Deterministic Generation
Even with a fixed generator seed, each sample gets its own deterministic seed:
```python
# Line 1628-1638 in dataset_generator.py
seed_tuple = (
    self.generator_seed,  # Your DATASET_GENERATION_SEED
    tuple(custom_angles),
    param_combo['snr'],
    param_combo['num_snapshots'],
    # ... other parameters
    example_idx,
)
seed_str = "_".join(map(str, seed_tuple))
base_seed = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
```

This means:
- **Same generator seed** → **Same angles** → **Same per-sample noise/multipath**
- **Different generator seed** → **Different angles** → **Different everything**

## Recommendation

### For Research/Publication:
```python
DATASET_GENERATION_SEED = 42  # Fixed seed for reproducibility
```
Save this in your paper/code so others can reproduce your exact results.

### For Algorithm Testing:
```python
# Run multiple times with different seeds
for seed in [42, 123, 456, 789, 1001]:
    DATASET_GENERATION_SEED = seed
    # Generate and evaluate
```
Tests robustness across different angle configurations.

### For Quick Testing:
```python
DATASET_GENERATION_SEED = int(time.time())  # Current - fine for exploration
```

## Current Behavior Summary

| Aspect | Behavior | Why |
|--------|----------|-----|
| **Angles** | Random each run | `self.rng.integers()` with time-based seed |
| **Noise** | Random each run | Per-sample seed derived from angles + generator seed |
| **Multipath** | Random each run | Per-sample seed derived from angles + generator seed |
| **SNR levels** | Same each run | Fixed: `[-20, -15, -10, -5, 0, 5, 10, 15, 20]` |
| **Number of samples** | Same each run | Fixed: `SAMPLES_PER_SNR = 100` |
| **Angle range** | Same each run | Fixed: `(30.0, 150.0)` |

**Bottom line**: The *structure* is the same (9 SNRs, 100 samples each), but the *content* (specific angles, noise, multipath) is different every run.
