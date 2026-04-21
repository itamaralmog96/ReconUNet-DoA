# Tri4Net: DOA Estimation with Controlled Datasets

A comprehensive framework for Direction of Arrival (DOA) estimation using controlled synthetic datasets and deep learning models.

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Create and activate virtual environment
python -m venv DOA_env
source DOA_env/bin/activate  # On Windows: DOA_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Your First Dataset
```bash
# Quick test dataset (small, fast generation)
python scripts/generate_dataset.py --config configs/dataset_configs/basic_dataset.yaml --dry-run

# Generate the dataset
python scripts/generate_dataset.py --config configs/dataset_configs/basic_dataset.yaml
```

### 3. Train a Model
```bash
# Train CNN on the generated dataset
python scripts/train_model.py --dataset Data/datasets/basic_controlled_dataset --model cnn --epochs 10
```

### 4. Monitor Training
```bash
# View training progress in Tensorboard
tensorboard --logdir Data/experiments
```

## 📁 Project Structure

```
Tri4Net/
├── configs/
│   └── dataset_configs/          # YAML configuration files
│       ├── basic_dataset.yaml    # Quick testing configuration
│       └── comprehensive_dataset.yaml  # Full research configuration
├── scripts/
│   ├── generate_dataset.py       # Configuration-based dataset generation
│   └── train_model.py            # Model training with experiments
├── src/
│   ├── data/
│   │   ├── controlled_dataset_generator.py    # Core dataset generator
│   │   └── controlled_dataset_dataloader.py  # PyTorch dataloaders
│   └── signalgen/                # Signal generation utilities
├── Data/
│   ├── datasets/                 # Generated datasets
│   └── experiments/              # Training experiments & logs
└── notebooks/                    # Jupyter notebooks for exploration
```

## 🔧 Configuration System

### Dataset Configurations

**Basic Dataset** (`configs/dataset_configs/basic_dataset.yaml`):
- Small parameter space for quick testing
- Single file output (no train/val/test splits)
- Positive SNR only
- ~1,800 samples

**Comprehensive Dataset** (`configs/dataset_configs/comprehensive_dataset.yaml`):
- Full parameter space for research
- Automatic train/val/test splits
- Includes multipath and interference
- ~50,000+ samples

### Custom Configurations

Create your own YAML config file:

```yaml
dataset_name: "my_custom_dataset"
create_splits: true

array_config:
  array_type: "triangular"
  num_elements: 4
  carrier_freq: 2.45e9

angle_sweep:
  min_deg: 0
  max_deg: 180
  step_deg: 5
  enabled: true

# ... more parameters
```

## 🎯 Usage Examples

### 1. Dataset Generation

```bash
# Preview what will be generated
python scripts/generate_dataset.py --config configs/dataset_configs/basic_dataset.yaml --dry-run

# Generate dataset
python scripts/generate_dataset.py --config configs/dataset_configs/basic_dataset.yaml

# Generate comprehensive research dataset
python scripts/generate_dataset.py --config configs/dataset_configs/comprehensive_dataset.yaml
```

### 2. Model Training

```bash
# Basic training
python scripts/train_model.py --dataset Data/datasets/basic_controlled_dataset --model cnn

# Advanced training with custom parameters
python scripts/train_model.py \
    --dataset Data/datasets/comprehensive_controlled_dataset \
    --model cnn \
    --epochs 50 \
    --batch-size 64 \
    --lr 0.0001 \
    --experiment-name "cnn_comprehensive_v1"
```

### 3. Using the DataLoader Directly

```python
from src.data.controlled_dataset_dataloader import create_dataloaders, DataLoaderConfig

# Configure dataloader
config = DataLoaderConfig(
    batch_size=32,
    data_format='real',  # or 'complex'
    normalize_data=True,
    include_raw_signals=True,
    include_covariance_matrices=True
)

# Create dataloaders
train_loader, val_loader, test_loader = create_dataloaders(
    dataset_path="Data/datasets/my_dataset",
    config=config
)

# Use in training loop
for batch in train_loader:
    signals = batch['raw_signals']      # [batch, elements, time, 2]
    angles = batch['angles_deg']        # [batch]
    covariance = batch['covariance_matrices']  # [batch, elements, elements, 2]
    # ... your training code
```

## 📊 Data Formats

### Dataset Structure
```
Data/datasets/my_dataset/
├── dataset.h5              # Single file (if create_splits=false)
├── train.h5               # Training split (if create_splits=true)
├── val.h5                 # Validation split
├── test.h5                # Test split
├── README.md              # Dataset documentation
├── config.json            # Generation configuration
└── dataset_info.json      # Dataset statistics
```

### HDF5 Data Fields
- `raw_signals`: Complex-valued received signals `[samples, elements, time_steps]`
- `covariance_matrices`: Sample covariance matrices `[samples, elements, elements]`
- `steering_vectors`: Array steering vectors `[samples, elements]`
- `angles_deg`: True DOA angles in degrees `[samples]`
- `snr_db`: Signal-to-noise ratios `[samples]`
- `metadata`: Additional parameters per sample

## 🔬 Advanced Usage

### Custom Array Configurations

Modify the array configuration in your YAML file:

```yaml
array_config:
  array_type: "linear"        # or "triangular", "circular"
  num_elements: 8
  carrier_freq: 5.8e9
  enable_gain_phase_errors: true
  enable_mutual_coupling: true
  position_error_std: 0.005   # Smaller errors for precision arrays
```

### Multipath and Interference

Enable realistic channel conditions:

```yaml
multipath_sweep:
  enabled: true
  num_paths_range: [1, 4]
  delay_range_ns: [5, 200]
  amplitude_range_db: [-25, -3]

sir_sweep:
  enabled: true
  min_db: -15
  max_db: 25
  step_db: 5
```

### Classical Signal Processing Integration

```python
# Load data for classical algorithms (MUSIC, ESPRIT, etc.)
config = DataLoaderConfig(
    data_format='complex',           # Keep complex format
    include_covariance_matrices=True,
    include_steering_vectors=True,
    normalize_data=False             # No normalization for classical methods
)

train_loader, _, _ = create_dataloaders(dataset_path, config)

for batch in train_loader:
    # Use with classical DOA algorithms
    R = batch['covariance_matrices']  # Sample covariance matrices
    A = batch['steering_vectors']     # Steering vectors
    # ... apply MUSIC, ESPRIT, etc.
```

## 🧪 Experiment Management

### Automatic Experiment Tracking

Training automatically creates organized experiments:

```
Data/experiments/
└── cnn_basic_controlled_dataset_1703123456/
    ├── config.json           # Training configuration
    ├── best_model.pth        # Best model checkpoint
    └── tensorboard/          # Tensorboard logs
        └── events.out.tfevents...
```

### Tensorboard Monitoring

```bash
# View all experiments
tensorboard --logdir Data/experiments

# View specific experiment
tensorboard --logdir Data/experiments/cnn_basic_controlled_dataset_1703123456/tensorboard
```

## 🛠️ Development Workflow

### 1. Always Run from Project Root
```bash
cd /path/to/Tri4Net
python scripts/generate_dataset.py --config ...
python scripts/train_model.py --dataset ...
```

### 2. Use Configuration Files
- Never hardcode parameters in scripts
- Create YAML configs for reproducibility
- Version control your configurations

### 3. Organize Experiments
- Use descriptive experiment names
- Keep training logs and configurations
- Document significant results

### 4. Data Management
- Use hierarchical dataset organization
- Include comprehensive metadata
- Validate datasets before training

## 🔍 Troubleshooting

### Common Issues

**Import Errors**: Make sure you're running from the project root (`Tri4Net/`)

**Memory Issues**: Reduce `batch_size` or `examples_per_combination` in configs

**CUDA Issues**: Use `--device cpu` to force CPU training

**Dataset Not Found**: Check that dataset path exists and contains `.h5` files

### Performance Tips

- Use `--dry-run` to preview dataset generation
- Start with basic configs before comprehensive ones
- Monitor GPU memory usage during training
- Use multiple workers for faster data loading

## 📚 References

- Array signal processing fundamentals
- DOA estimation algorithms (MUSIC, ESPRIT, etc.)
- Deep learning for signal processing
- PyTorch documentation

## 🤝 Contributing

1. Follow the established project structure
2. Use configuration files for parameters
3. Include comprehensive documentation
4. Test with both basic and comprehensive configs
5. Maintain backward compatibility

---

**Happy DOA Estimation! 🎯** 