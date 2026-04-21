# Tri4Net Subspace Models Training

This module provides training functionality for subspace-based deep learning models ported from SubspaceNet to Tri4Net. It supports multiple model types and handles data preprocessing from Tri4Net datasets.

## 🎯 Supported Models

| Model | Description | Input Format | Key Features |
|-------|-------------|--------------|--------------|
| **SubspaceNet** | Generalized subspace network | Autocorrelation tensor `[B, τ, 2N, N]` | Supports Root-MUSIC or ESPRIT |
| **SubspaceNetEsprit** | SubspaceNet with ESPRIT | Autocorrelation tensor `[B, τ, 2N, N]` | Hard-wired to ESPRIT |
| **DeepRootMUSIC** | CNN + Root-MUSIC | Autocorrelation tensor `[B, τ, 2N, N]` | Original SubspaceNet architecture |
| **DeepAugmentedMUSIC** | GRU + MUSIC spectrum | Raw signals `[B, N, T]` | RNN-based approach |
| **DeepCNN** | Baseline CNN | Covariance tensor `[B, N, N, 3]` | Grid-based DoA prediction |

Where:
- `B` = batch size
- `N` = number of array elements  
- `T` = number of time snapshots
- `τ` = number of autocorrelation lags

## 🚀 Quick Start

### 1. Basic Training

```python
from src.training.subspace_training import train_subspace_model

# Train SubspaceNet with Root-MUSIC
results = train_subspace_model(
    dataset_path="Data/datasets/triangular/my_dataset/my_dataset.h5",
    model_type="subspace_net",
    tau=5,
    diff_method="root_music",
    epochs=100,
    batch_size=32,
    learning_rate=1e-3,
    save_dir="models/trained"
)
```

### 2. Using Example Configurations

```python
from src.training.train_examples import train_subspace_net_example

# Train with optimized parameters
results = train_subspace_net_example(
    dataset_path="Data/datasets/triangular/my_dataset/my_dataset.h5",
    save_dir="models/subspace_net"
)
```

### 3. Command Line Interface

```bash
# Train a specific model
python -m src.training.train_examples \
    --dataset Data/datasets/triangular/my_dataset/my_dataset.h5 \
    --model subspace_net \
    --save-dir models/trained

# Quick test with minimal epochs
python -m src.training.train_examples \
    --dataset Data/datasets/triangular/my_dataset/my_dataset.h5 \
    --model quick_test

# Train all models
python -m src.training.train_examples \
    --dataset Data/datasets/triangular/my_dataset/my_dataset.h5 \
    --model all \
    --save-dir models/comparison
```

## 📋 Prerequisites

### 1. Generate Dataset

First, create a dataset using the Tri4Net dataset generator:

```bash
cd Tri4Net
python -m src.data.dataset_generator --config triangular_base.yaml
```

### 2. Required Dependencies

- PyTorch
- NumPy
- Matplotlib
- tqdm
- h5py

## 🔧 Configuration Options

### Training Parameters

```python
from src.training.subspace_training import SubspaceTrainingParams

params = SubspaceTrainingParams(
    # Model configuration
    model_type="subspace_net",       # Model type
    tau=5,                           # Autocorrelation lags (SubspaceNet models)
    diff_method="root_music",        # "root_music" or "esprit"
    
    # Training parameters
    batch_size=32,                   # Batch size
    epochs=100,                      # Number of epochs
    learning_rate=1e-3,              # Learning rate
    weight_decay=1e-4,               # L2 regularization
    
    # Learning rate scheduler
    step_size=30,                    # LR decay step size
    gamma=0.1,                       # LR decay factor
    
    # Data splitting
    train_test_split=0.9,            # Train/validation split
    
    # Device
    device="auto"                    # "auto", "cpu", or "cuda"
)
```

### Model-Specific Recommendations

#### SubspaceNet / DeepRootMUSIC
```python
# Recommended parameters
tau=5                    # Good balance of performance/memory
batch_size=32           # Standard batch size
learning_rate=1e-3      # Works well for most datasets
epochs=100              # Usually sufficient
```

#### DeepAugmentedMUSIC
```python
# Requires more memory and training time
batch_size=16           # Smaller due to GRU memory requirements
learning_rate=5e-4      # Lower learning rate
epochs=150              # Needs more epochs to converge
```

#### DeepCNN
```python
# Simpler model, faster training
batch_size=64           # Can use larger batches
learning_rate=1e-3      # Standard learning rate
epochs=80               # Converges faster
weight_decay=1e-3       # Higher regularization needed
```

## 📊 Data Preprocessing

The training module automatically handles data preprocessing based on model type:

### Autocorrelation Tensor (SubspaceNet models)
- Input: Complex received signals `[N, T]`
- Output: Autocorrelation tensor `[τ, 2N, N]`
- Process: Computes autocorrelation matrices for lags 0 to τ-1, stacks real/imaginary parts

### Covariance Tensor (CNN models)  
- Input: Complex received signals `[N, T]`
- Output: Covariance tensor `[N, N, 3]`
- Process: Computes sample covariance, stacks real/imaginary/phase components

### Raw Signals (DeepAugmentedMUSIC)
- Input: Complex received signals `[N, T]`
- Output: Same format `[N, T]`
- Process: Direct input to GRU layers

## 📈 Training Output

The training process provides:

1. **Real-time Progress**: tqdm progress bars with loss updates
2. **Training Curves**: Automatically saved plots of train/validation loss
3. **Model Checkpoints**: Best model saved based on validation loss
4. **Training Statistics**: 
   - Best validation loss
   - Training time
   - Model parameter count

Example output:
```
=== Training subspace_net ===
Dataset size: 10800
Training samples: 9720
Validation samples: 1080
Array size (N): 4
Snapshots (T): 256
Sources (M): 1
Model parameters: 143,233

Epoch 1/100
Training: 100%|████| 304/304 [01:23<00:00, 3.65it/s, loss=2.451]
New best validation loss: 1.823456
Train Loss: 2.451234, Val Loss: 1.823456

🎉 Training completed!
Best validation loss: 0.123456
Training time: 456.78 seconds
```

## 🔍 Example Configurations

### Quick Test (5 epochs)
```python
train_quick_test_example(dataset_path, save_dir)
```

### Production Training
```python
train_subspace_net_example(dataset_path, save_dir)
```

### Model Comparison
```python
train_all_models_example(dataset_path, base_save_dir)
```

## 📁 Output Structure

```
models/
├── subspace_net/
│   ├── subspace_net_20241201_143022.pth     # Model weights
│   └── subspace_net_20241201_143022_curves.png  # Training curves
├── deep_root_music/
│   ├── deep_root_music_20241201_143122.pth
│   └── deep_root_music_20241201_143122_curves.png
└── ...
```

## 🐛 Troubleshooting

### Common Issues

1. **Dataset not found**
   ```bash
   # Generate dataset first
   python -m src.data.dataset_generator --config triangular_base.yaml
   ```

2. **CUDA out of memory**
   ```python
   # Reduce batch size
   batch_size=16  # or smaller
   ```

3. **Import errors**
   ```bash
   # Ensure you're in the Tri4Net directory
   cd Tri4Net
   python -m src.training.train_examples --help
   ```

4. **Loss not decreasing**
   - Try lower learning rate: `learning_rate=1e-4`
   - Increase regularization: `weight_decay=1e-3`
   - Check dataset quality and labeling

### Debug Mode

Use the quick test configuration for debugging:
```python
train_quick_test_example(dataset_path, save_dir)
```

This runs only 5 epochs with small batch size for fast iteration.

## 🔗 Integration with Evaluation

After training, use the evaluation module to test your models:

```python
from src.evaluation.evaluation import DOAEvaluator, EvaluationConfig

# Configure evaluation
cfg = EvaluationConfig(
    dataset_path="path/to/test_dataset.h5",
    augmented_models=["subspace_net"],
    model_weight_paths={"subspace_net": "models/trained/subspace_net_timestamp.pth"}
)

# Run evaluation
evaluator = DOAEvaluator(cfg)
evaluator.run()
```

## 📖 API Reference

See the docstrings in `subspace_training.py` for detailed API documentation of all classes and functions. 