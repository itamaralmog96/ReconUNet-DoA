# Controlled Angle Evaluation - Visual Guide

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                 controlled_angle_evaluation.py                   │
│                    (Main Evaluation Script)                       │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 │ Uses
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│        ControlledDatasetGenerator.generate_test_dataset()        │
│                  (New Method in dataset_generator.py)            │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 │ Generates
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DOA Dataset (HDF5)                            │
│   - Received signals                                             │
│   - Covariance matrices                                          │
│   - Clean covariance (for UNet targets)                          │
│   - Autocorrelation (for UNet inputs)                            │
│   - Labels (angles, SNR, etc.)                                   │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 │ Evaluated by
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DOA Algorithms                                │
│   - UNet + MUSIC                                                 │
│   - Classic MUSIC                                                │
│   - MVDR / Beamformer / ESPRIT / Root-MUSIC                      │
│   - CRLB (theoretical bounds)                                    │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 │ Produces
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Results & Plots                               │
│   - results_modeX.json                                           │
│   - rmse_vs_snr_modeX.png                                        │
└─────────────────────────────────────────────────────────────────┘
```

## Three Modes Visualized

### Mode 1: Fixed Reference + Fixed Delta

```
Configuration:
  REFERENCE_ANGLES = [90.0]
  ANGLE_DELTA = 10.0
  NUM_SOURCES = [3, 0]
  SAMPLES_PER_SNR = 3

Result (all samples identical per SNR):
  
  Sample 1:  ●────10°───●────10°───●
             90°       100°       110°
  
  Sample 2:  ●────10°───●────10°───●  (same!)
             90°       100°       110°
  
  Sample 3:  ●────10°───●────10°───●  (same!)
             90°       100°       110°

✅ Perfect for: Testing specific critical angles
```

### Mode 2: Random Reference + Fixed Delta

```
Configuration:
  REFERENCE_ANGLES = 'random'
  ANGLE_DELTA = 10.0
  NUM_SOURCES = [3, 0]
  SAMPLES_PER_SNR = 3

Result (different reference, same delta):
  
  Sample 1:  ●────10°───●────10°───●
             73°        83°        93°
  
  Sample 2:       ●────10°───●────10°───●
                 112°       122°       132°
  
  Sample 3:  ●────10°───●────10°───●
             41°        51°        61°

✅ Perfect for: Robust evaluation with controlled separation
```

### Mode 3: Random Angles + Minimum Separation

```
Configuration:
  REFERENCE_ANGLES = None
  ANGLE_DELTA = None
  min_angle_separation_deg = 10.0
  NUM_SOURCES = [3, 0]
  SAMPLES_PER_SNR = 3

Result (fully random with constraint):
  
  Sample 1:  ●──────23°──────●────────47°────────●
             42°              65°                 112°
  
  Sample 2:  ●───15°────●──────────────54°──────────●
             68°         83°                       137°
  
  Sample 3:       ●─────────32°──────────●────18°────●
                 95°                     127°        145°

✅ Perfect for: General performance across all scenarios
```

## Boundary Handling Visualization

### Problem: Sources Near Boundary

```
Angle Range: [30°, 150°]
Reference: 145°
Delta: 10°
Sources: 3

Naive Approach (FAILS):
  30°                                           150°
  |─────────────────────────────────────────────|
                                      ●───10°──●───10°──●
                                    145°     155° 165° ❌
                                                     │
                                                     └─> EXCEEDS!
```

### Solution: Automatic Adjustment

```
Smart Approach (WORKS):
  30°                                           150°
  |─────────────────────────────────────────────|
                                  ●───10°──●───10°──●
                                135°     145°     150° ✅
                                 │
                                 └─> ADJUSTED BACKWARD
```

## Data Flow Diagram

```
Configuration
     │
     │ define: mode, angles, delta, SNR, sources
     ▼
┌──────────────────────┐
│  Angle Generation    │
│                      │
│  Mode 1: Use fixed   │
│  Mode 2: Random ref  │
│  Mode 3: Fully random│
└──────────┬───────────┘
           │
           │ generate angles for each sample
           ▼
┌──────────────────────┐
│  Signal Generation   │
│                      │
│  - Array steering    │
│  - Source signals    │
│  - Multipath         │
│  - Noise (SNR)       │
└──────────┬───────────┘
           │
           │ create samples
           ▼
┌──────────────────────┐
│  Dataset (HDF5)      │
│                      │
│  Saved to disk       │
└──────────┬───────────┘
           │
           │ load with DOADataset
           ▼
┌──────────────────────┐
│  Algorithm Eval      │
│                      │
│  For each sample:    │
│  - MUSIC             │
│  - UNet+MUSIC        │
│  - MVDR              │
│  - ESPRIT            │
│  - CRLB              │
└──────────┬───────────┘
           │
           │ compute RMSE per SNR
           ▼
┌──────────────────────┐
│  Results & Plots     │
│                      │
│  - JSON with stats   │
│  - RMSE vs SNR plot  │
└──────────────────────┘
```

## Sample Configuration Examples

### Example 1: Resolution Limit Test

```
┌─────────────────────────────────────────────────────────────┐
│ Goal: Find minimum resolvable separation                     │
├─────────────────────────────────────────────────────────────┤
│ EVALUATION_MODE = 1                                          │
│ REFERENCE_ANGLES = [90.0]  # Broadside                       │
│ ANGLE_DELTA = 3.0  # Very close!                             │
│ NUM_SOURCES = [2, 0]                                         │
│ SNR_DB = [0, 10, 20]  # High SNR                             │
│ SAMPLES_PER_SNR = 200                                        │
└─────────────────────────────────────────────────────────────┘

Visual:
  ●───3°───●
  90°      93°
  
Question: Can algorithms distinguish sources only 3° apart?
```

### Example 2: Interference Impact Study

```
┌─────────────────────────────────────────────────────────────┐
│ Goal: Measure interference impact on main source            │
├─────────────────────────────────────────────────────────────┤
│ EVALUATION_MODE = 2                                          │
│ ANGLE_DELTA = 10.0                                           │
│ NUM_SOURCES = [1, 3]  # 1 main + 3 interference              │
│ SIR_DB = 0.0  # Equal power                                  │
│ SNR_DB = range(-20, 21, 5)                                   │
│ SAMPLES_PER_SNR = 100                                        │
└─────────────────────────────────────────────────────────────┘

Visual:
  ●───10°──●───10°──●───10°──●
  ^        ^        ^        ^
  main    int1     int2     int3
  
Question: How do 3 interferers affect main source estimation?
```

### Example 3: Angular Coverage Test

```
┌─────────────────────────────────────────────────────────────┐
│ Goal: Compare performance across all angles                 │
├─────────────────────────────────────────────────────────────┤
│ EVALUATION_MODE = 1                                          │
│ REFERENCE_ANGLES = [30, 60, 90, 120, 150]  # Full range     │
│ ANGLE_DELTA = 10.0                                           │
│ NUM_SOURCES = [2, 0]                                         │
│ SAMPLES_PER_SNR = 50  # Per angle per SNR                   │
└─────────────────────────────────────────────────────────────┘

Visual:
  30°:  ●───10°───●
  60°:        ●───10°───●
  90°:              ●───10°───●
  120°:                   ●───10°───●
  150°:                          ●──10°─●
  
Question: Do algorithms perform uniformly across all angles?
```

## Output Structure Visualization

```
Tri4Net/
│
├── controlled_angle_evaluation.py  ← Main script
├── QUICK_REFERENCE.md             ← This guide!
├── CONTROLLED_ANGLE_EVALUATION_README.md  ← Full docs
│
├── Data/datasets/linear/
│   ├── controlled_eval_mode1_delta10/
│   │   ├── controlled_eval_mode1_delta10.h5  ← Dataset
│   │   └── controlled_eval_mode1_delta10_info.json
│   │
│   ├── controlled_eval_mode2_delta10/
│   │   └── ...
│   │
│   └── controlled_eval_mode3_random/
│       └── ...
│
└── src/evaluation/controlled_angle_evaluation/
    ├── results_mode1.json           ← RMSE values
    ├── results_mode2.json
    ├── results_mode3.json
    ├── rmse_vs_snr_mode1.png       ← Plots
    ├── rmse_vs_snr_mode2.png
    └── rmse_vs_snr_mode3.png
```

## Workflow Visualization

```
┌─────────────┐
│   Step 1    │  Configure parameters in script
│   CONFIG    │  - Mode, angles, delta, SNR, sources
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Step 2    │  Run script
│   GENERATE  │  $ python controlled_angle_evaluation.py
└──────┬──────┘
       │
       ├──────────────┐
       │              │
       ▼              ▼
┌─────────────┐  ┌─────────────┐
│   Step 3a   │  │   Step 3b   │
│   DATASET   │  │   EVALUATE  │
│             │  │             │
│ Generate    │  │ Apply       │
│ samples     │  │ algorithms  │
└──────┬──────┘  └──────┬──────┘
       │                │
       └────────┬───────┘
                ▼
       ┌─────────────┐
       │   Step 4    │  Results appear in:
       │   RESULTS   │  - JSON files
       │             │  - PNG plots
       └─────────────┘  - Terminal summary
```

## Performance Comparison Example

```
Typical RMSE vs SNR Plot:

RMSE (°)
  100 │                    CRLB ─ ─ ─ ─ (lower bound)
      │                        ╲
   10 │                         ╲
      │              Beamformer──●──●──●──●
      │                   MVDR───○──○──○──○──○
      │                  MUSIC───□──□──□──□──□──□
    1 │           Root-MUSIC────△──△──△──△──△──△──△
      │         UNet+MUSIC──────✱──✱──✱──✱──✱──✱──✱──✱
      │                                           (best!)
  0.1 │
      └────────────────────────────────────────────────
     -20  -15  -10   -5    0    5   10   15   20  SNR(dB)

Legend:
  ─ ─ ─  CRLB (theoretical lower bound)
  ──●──  Beamformer
  ──○──  MVDR
  ──□──  MUSIC
  ──△──  Root-MUSIC
  ──✱──  UNet+MUSIC (usually best at high SNR)
```

## Quick Decision Tree

```
Need to evaluate DOA performance?
    │
    ├─ Test specific angles? ────────────────► Mode 1
    │   (e.g., boundaries, broadside)
    │
    ├─ Need controlled separation? ──────────► Mode 2
    │   (e.g., delta analysis)
    │
    └─ General characterization? ────────────► Mode 3
        (e.g., overall RMSE curve)
```

---

**Remember:**
- Mode 1 = **Specific** angles (deterministic)
- Mode 2 = **Random** reference with **fixed** delta
- Mode 3 = **Fully random** (most general)

🎯 Choose based on your research question!
