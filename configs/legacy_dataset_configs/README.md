# Base Dataset Configurations

This directory contains three base configurations for generating DOA estimation datasets with different complexity levels and use cases.

## Quick Start

```bash
# From Tri4Net directory:

# Small dataset for testing
python src/data/simple_controlled_generator.py --config base_small

# Medium dataset for training  
python src/data/simple_controlled_generator.py --config base_medium

# Large dataset for research
python src/data/simple_controlled_generator.py --config base_large
```

## Configuration Overview

| Configuration | Samples | Size | Time | Use Case |
|---------------|---------|------|------|----------|
| `base_small` | ~8,600 | 2-3 GB | 2-5 min | Quick testing, debugging, development |
| `base_medium` | ~540,000 | 20-30 GB | 15-30 min | Training, evaluation, validation |
| `base_large` | ~324,000 | 100-150 GB | 1-3 hours | Comprehensive research, benchmarking |

## Configuration Details

### Base Small (`base_small.yaml`)
- **Purpose**: Quick testing and development
- **Parameters**:
  - Angles: Every 45° (8 values)
  - SNR: 3 levels (-10, 0, 10 dB)
  - Snapshots: 2 sizes (128, 256)
  - Sources: 3 configurations ([1,0], [1,1], [1,2])
  - SIR: 3 levels (-10, 0, 10 dB)
  - Multipath: 2 values (0, 1)
  - Examples: 5 per combination
- **Ideal for**: Algorithm prototyping, quick validation, CI/CD testing

### Base Medium (`base_medium.yaml`)
- **Purpose**: Training and evaluation
- **Parameters**:
  - Angles: Every 20° (18 values)
  - SNR: 5 levels (-20 to 20 dB)
  - Snapshots: 3 sizes (128, 256, 512)
  - Sources: 6 configurations (all practical scenarios)
  - SIR: 5 levels (-20 to 20 dB)
  - Multipath: 3 values (0, 1, 2)
  - Examples: 20 per combination
- **Ideal for**: Model training, performance evaluation, method comparison

### Base Large (`base_large.yaml`)
- **Purpose**: Comprehensive research
- **Parameters**:
  - Angles: Every 10° (36 values) with **random sampling**
  - SNR: 13 levels (-30 to 30 dB)
  - Snapshots: 4 sizes (64, 128, 256, 512)
  - Sources: 6 configurations (all practical scenarios)
  - SIR: 13 levels (-30 to 30 dB)
  - Multipath: 3 values (0, 1, 2)
  - Examples: 50 per combination
- **Ideal for**: Benchmarking, robustness testing, publication-quality results

## Key Features

### Automatic Multipath Constraint
All configurations automatically enforce the constraint: **main + interference + multipath ≤ 3**

This ensures realistic signal scenarios while preventing invalid combinations.

### Range Specifications
Configurations use convenient range specifications:
```yaml
# Every 20 degrees
angles_deg:
  start: 0
  stop: 360
  step: 20

# 5 equally spaced SNR levels
snr_db:
  start: -20
  stop: 21
  step: 10
```

### Smart Random Sampling
The large configuration uses random angle sampling to reduce dataset size while maintaining diversity:
```yaml
random_sampling_mode: true
random_sample_params: ["angles_deg"]
```

## Customization

You can create custom configurations by:

1. **Copying a base config**: Start with the closest base configuration
2. **Modifying parameters**: Adjust ranges, add/remove values
3. **Changing generation settings**: Modify `examples_per_combination`
4. **Adding random sampling**: Enable for specific parameters

### Example Customization
```yaml
# Custom config based on base_medium
angles_deg: "0:360:15"  # Every 15 degrees instead of 20
snr_db: [-15, -5, 5, 15]  # Specific SNR values
examples_per_combination: 10  # Fewer examples for faster generation
```

## Output Structure

Generated datasets are saved to:
```
Tri4Net/Data/datasets/{dataset_name}/
├── {dataset_name}.h5        # HDF5 dataset file
└── {dataset_name}_info.json # Configuration and metadata
```

## Loading and Using Datasets

```python
from src.data.simple_controlled_generator import DOADataset

# Load dataset
dataset = DOADataset("Tri4Net/Data/datasets/base_small/base_small.h5")

# Filter examples
high_snr = dataset.filter(snr=10.0)
single_source = dataset.filter(num_sources=[1, 0])
imperfect_arrays = dataset.filter(array_imperfections=True)

# Get sample
sample = dataset[0]
received_signal = sample['received_signal']
labels = sample['labels']
```

## Performance Tips

1. **Start small**: Always test with `base_small` first
2. **Monitor resources**: Large datasets require significant disk space and memory
3. **Use random sampling**: For very large parameter spaces, enable random sampling
4. **Parallel generation**: Run multiple smaller configs in parallel if needed
5. **Storage planning**: Ensure sufficient disk space before starting large generations

## Troubleshooting

- **Out of memory**: Reduce `examples_per_combination` or enable random sampling
- **Slow generation**: Start with smaller configurations to estimate timing
- **Invalid combinations**: Check multipath constraint (main + interference + multipath ≤ 3)
- **Missing files**: Ensure you're running from the `Tri4Net` directory 