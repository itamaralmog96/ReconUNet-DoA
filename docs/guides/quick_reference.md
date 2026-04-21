# Quick Reference: Controlled Angle Evaluation

## 🚀 Quick Start

```bash
cd Tri4Net
python controlled_angle_evaluation.py
```

## 📋 Three Modes at a Glance

| Mode | Reference | Delta | Use Case |
|------|-----------|-------|----------|
| **1** | Fixed list | Fixed | Test specific angles |
| **2** | Random | Fixed | General eval with controlled separation |
| **3** | Random | Min separation | Fully random (like existing scripts) |

## ⚙️ Basic Configuration Template

```python
# === MODE SELECTION ===
EVALUATION_MODE = 2  # 1, 2, or 3

# === ANGLES ===
REFERENCE_ANGLES = [90.0]  # Mode 1 only
ANGLE_DELTA = 10.0         # Modes 1 & 2 (None for Mode 3)
ANGLE_RANGE = (30.0, 150.0)  # Modes 2 & 3

# === SNR & SAMPLES ===
SNR_DB = list(range(-20, 21, 5))
SAMPLES_PER_SNR = 100

# === SOURCES ===
NUM_SOURCES = [1, 2]  # [main, interference]

# === ALGORITHMS ===
EVALUATE_UNET_MUSIC = True
EVALUATE_MUSIC = True
EVALUATE_CRLB = True
```

## 🎯 Common Scenarios

### Single Source
```python
NUM_SOURCES = [1, 0]
ANGLE_DELTA = None  # Not needed
```

### Two Sources with Controlled Separation
```python
EVALUATION_MODE = 2
NUM_SOURCES = [2, 0]
ANGLE_DELTA = 10.0  # Sources 10° apart
```

### Multiple Sources + Interference
```python
NUM_SOURCES = [1, 2]  # 1 main + 2 interference
ANGLE_DELTA = 10.0
SIR_DB = 0.0
```

### Test Close Sources
```python
EVALUATION_MODE = 1
REFERENCE_ANGLES = [90.0]
NUM_SOURCES = [2, 0]
ANGLE_DELTA = 3.0  # Very close!
```

## 📊 Output Files

```
Results JSON:
src/evaluation/controlled_angle_evaluation/results_modeX.json

RMSE Plot:
src/evaluation/controlled_angle_evaluation/rmse_vs_snr_modeX.png

Dataset:
Data/datasets/linear/controlled_eval_modeX/...h5
```

## 🔧 Key Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `ANGLE_DELTA` | Exact separation (°) | `10.0` |
| `SAMPLES_PER_SNR` | Samples per SNR level | `100` |
| `REFERENCE_ANGLES` | Fixed refs (Mode 1) | `[45, 90, 135]` |
| `NUM_SOURCES` | [main, interference] | `[1, 2]` |
| `ARRAY_TYPE` | Array geometry | `'linear'` |
| `NUM_ELEMENTS` | Array size | `8` |

## 💡 Tips

- **Mode 1:** Quick tests at specific angles
- **Mode 2:** Best for robust statistics
- **Mode 3:** Most general evaluation
- **Large SAMPLES_PER_SNR:** Smooth curves
- **Small ANGLE_DELTA:** Test resolution limits

## ⚠️ Common Issues

| Error | Solution |
|-------|----------|
| Cannot fit sources | Reduce `ANGLE_DELTA` or increase `ANGLE_RANGE` |
| Reference out of range | Choose ref further from boundaries |
| UNet not found | Update `UNET_MODEL_PATH` or disable |

## 📚 Full Documentation

See: `CONTROLLED_ANGLE_EVALUATION_README.md`
