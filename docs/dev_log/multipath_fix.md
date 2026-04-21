# Multipath Handling Fix for Controlled Angle Evaluation

## Problem

When `NUM_MULTIPATH = 6` was configured, the dataset generation failed with:

```
ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0, 
with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 7 is different from 1)
```

**Root Cause**: Multipath components are treated as additional signal paths with their own angles of arrival. With 1 main source and 6 multipath components, there are 7 total signal paths.

## Understanding Multipath in the System

### How Multipath Works

1. **Direct Path**: Original signal from source at specified angle
2. **Multipath Components**: Scattered/reflected signals arriving from different angles with:
   - Random angles of arrival (AOA)
   - Time delays
   - Amplitude attenuation
   - Phase shifts

### Signal Flow with Multipath

```
Initial State:
- custom_angles = [main_angle]           # 1 angle
- signals.shape = (1, T)                 # 1 source

After add_multipath():
- signal_config.angles = [main_angle, mp1_angle, mp2_angle, ..., mp6_angle]  # 7 angles
- signals.shape = (7, T)                 # 7 paths (1 main + 6 multipath)
```

### Key Insight

The `add_multipath()` method modifies `signal_config.angles` to include multipath angles:

```python
# From signal_generator.py, line 372-374
if hasattr(self.config, 'angles') and self.config.angles is not None:
    self.config.angles = np.concatenate([self.config.angles, np.array(all_angles)])
```

## The Fix

### Changes Made to `_generate_sample_custom()`

#### 1. Steering Matrix Creation (Line ~1695)

**Before** (incorrect):
```python
# Used custom_angles which only has main source angles
steering_matrix = array_model.steering_matrix(custom_angles, nominal=False)
# Result: (8, 1) - only 1 column for main source
```

**After** (correct):
```python
# Use signal_config.angles which includes multipath angles added by add_multipath()
steering_matrix = array_model.steering_matrix(signal_config.angles, nominal=False)
# Result: (8, 7) - 7 columns for all paths
```

#### 2. ReceivedSignal Creation (Line ~1702)

**Before** (incorrect):
```python
received_signal = ReceivedSignal(steering_matrix, signals, custom_angles, signal_config)
# Mismatch: steering_matrix has 1 angle, signals has 7 paths
```

**After** (correct):
```python
received_signal = ReceivedSignal(steering_matrix, signals, signal_config.angles, signal_config)
# Match: both use all 7 angles/paths
```

#### 3. Steering Vector Storage (Line ~1752)

**No change needed** - steering vectors should only be saved for main/interference sources, not multipath:
```python
if self.config.save_steering_vectors:
    # Save steering vectors for main/interference sources only (not multipath)
    nominal_steering = array_model.steering_matrix(custom_angles, nominal=True)
    if param_combo['array_imperfections']:
        actual_steering = array_model.steering_matrix(custom_angles, nominal=False)
    else:
        actual_steering = array_model.steering_matrix(custom_angles, nominal=True)
```

#### 4. DOA Labels (Line ~1709)

**No change needed** - labels should only include main/interference angles:
```python
'doas': np.array(custom_angles, dtype=np.float32),  # Only main source(s)
```

## Summary

### What Gets Multipath Angles?

✅ **Signal processing** (steering_matrix @ signals):
- Uses `signal_config.angles` (includes multipath)
- Creates realistic received signal with multipath effects

### What Stays with Original Angles?

✅ **Labels and ground truth**:
- Uses `custom_angles` (only main/interference sources)
- DOA estimation targets the original sources, not multipath

✅ **Steering vectors for evaluation**:
- Uses `custom_angles` 
- Used by algorithms to estimate source directions

## Configuration Example

```python
NUM_SOURCES = [1, 0]      # 1 main source, 0 interference
NUM_MULTIPATH = 6          # 6 multipath components

# Results in:
# - 1 angle in custom_angles (main source)
# - 7 angles in signal_config.angles after add_multipath() (1 main + 6 multipath)
# - 7 signal paths in received signal
# - 1 DOA label for algorithms to estimate
```

## Verification

Test passed with:
- **Dataset generation**: ✅ 900 samples in ~1.7 seconds
- **All algorithms**: ✅ MUSIC, MVDR, Beamformer, ESPRIT, UnitaryESPRIT, RootMUSIC, UNet+MUSIC, CRLB
- **Integer angles**: ✅ All angles are clean integers (30°, 45°, 90°, etc.)
- **Multipath**: ✅ 6 multipath components correctly added per source

## Related Files

- `src/data/dataset_generator.py`: Lines 1617-1760 (`_generate_sample_custom`)
- `src/signalgen/signal_generator.py`: Lines 184-394 (`add_multipath`)
- `DOA_est_Master/controlled_angle_evaluation.py`: Main evaluation script
