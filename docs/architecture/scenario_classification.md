# Scenario Classification for DoA Estimation

This module implements deep learning models for classifying DoA scenarios into 36 different classes based on source configurations, array conditions, and SNR levels. This can be used for adaptive DoA processing and scenario-aware algorithm selection.

## 📊 Scenario Classification Problem

### **36 Total Scenarios**
The classification system identifies scenarios based on three factors:

1. **6 Source/Multipath Combinations:**
   - 1 main source + 0 interference + 0 multipath
   - 1 main source + 0 interference + 1 multipath  
   - 1 main source + 1 interference + 0 multipath
   - 1 main source + 1 interference + 1 multipath
   - 1 main source + 2 interference + 0 multipath
   - 1 main source + 2 interference + 1 multipath

2. **2 Array Conditions:**
   - Perfect array (no imperfections)
   - Imperfect array (with array imperfections)

3. **3 SNR Levels:**
   - Low SNR: < 0 dB
   - Mid SNR: 0-10 dB  
   - High SNR: > 10 dB

**Total Classes:** 6 × 2 × 3 = **36 scenarios**

### **Input/Output Format**
- **Input:** Complex received signals `[B, N, T]` where B=batch, N=antennas, T=time samples
- **Output:** Scenario class probabilities `[B, 36]`

## 🏗️ Model Architectures

### 1. **ScenarioClassificationCNN** (Recommended)
Multi-scale CNN optimized for spatial-temporal signal patterns.

```python
from src.models.deep_learning.classification import ScenarioClassificationCNN

model = ScenarioClassificationCNN(N=8, T=1024, num_classes=36)
```

**Features:**
- Multi-scale spatial processing across antennas
- Multi-scale temporal processing for different time patterns
- Fast training and inference (~100k parameters)
- Excellent performance for array signal processing

### 2. **CovarianceResNet**
ResNet-based model processing sample covariance matrices.

```python
from src.models.deep_learning.classification import CovarianceResNet

model = CovarianceResNet(N=8, num_classes=36, resnet_variant='resnet18')
```

**Features:**
- Leverages domain knowledge (covariance structure)
- ResNet backbone with skip connections
- 3-channel representation: [Real, Imaginary, Magnitude]
- Robust to noise variations

### 3. **SignalTransformer**
Transformer with spatial-temporal attention for long-range dependencies.

```python
from src.models.deep_learning.classification import SignalTransformer

model = SignalTransformer(N=8, T=1024, num_classes=36, d_model=256)
```

**Features:**
- Captures long-range temporal dependencies
- Spatial-temporal attention mechanisms
- Patch-based processing for efficiency
- State-of-the-art potential performance

### 4. **HybridCNNRNN**
Efficient CNN-RNN hybrid for balanced spatial-temporal modeling.

```python
from src.models.deep_learning.classification import HybridCNNRNN

model = HybridCNNRNN(N=8, T=1024, num_classes=36)
```

**Features:**
- CNN for spatial feature extraction
- Bidirectional LSTM for temporal modeling
- Memory efficient design
- Good balance of performance and speed

## 🚀 Quick Start

### **Command Line Training**

```bash
# Train Multi-Scale CNN (recommended for starting)
python src/training/classification_training.py \
    --dataset Data/datasets/linear/ula_dataset_90_all_scenarios/ula_dataset_90_all_scenarios.h5 \
    --model scenario_cnn \
    --epochs 50 \
    --batch_size 64 \
    --lr 2e-3 \
    --save_dir models/classification/scenario_cnn

# Train Covariance ResNet
python src/training/classification_training.py \
    --dataset Data/datasets/linear/ula_dataset_90_all_scenarios/ula_dataset_90_all_scenarios.h5 \
    --model scenario_resnet \
    --epochs 40 \
    --batch_size 32 \
    --lr 1e-3

# Train Transformer (advanced)
python src/training/classification_training.py \
    --dataset Data/datasets/linear/ula_dataset_90_all_scenarios/ula_dataset_90_all_scenarios.h5 \
    --model scenario_transformer \
    --epochs 50 \
    --batch_size 16 \
    --lr 5e-4
```

### **Python API Training**

```python
from src.training.classification_training import ClassificationTrainer, ClassificationTrainingParams

# Configure training parameters
params = ClassificationTrainingParams(
    dataset_path="path/to/dataset.h5",
    model_type="scenario_cnn",
    epochs=50,
    batch_size=32,
    learning_rate=1e-3,
    save_dir="models/classification"
)

# Train model
trainer = ClassificationTrainer(params)
results = trainer.train(params.dataset_path)

print(f"Best validation accuracy: {results['best_val_acc']:.2f}%")
```

### **Example Script**

Run the comprehensive example that trains and compares multiple models:

```bash
python train_classification_example.py
```

This script will:
- Train Multi-Scale CNN, ResNet, and Hybrid models
- Compare performance and training times
- Generate confusion matrices and training plots
- Save all results with TensorBoard logs

## 📈 Training Features

### **Comprehensive Training Pipeline**
- Automatic train/validation/test splitting (80/10/10%)
- Multiple optimizers: AdamW, Adam, SGD
- Learning rate schedulers: Cosine, Step, Plateau
- Early stopping with patience
- Gradient clipping for stability

### **Advanced Regularization**
- Dropout with configurable rates
- Label smoothing for better generalization
- Weight decay for parameter regularization
- Data augmentation potential

### **Logging and Visualization**
- TensorBoard integration for real-time monitoring
- Confusion matrix visualization
- Classification reports with per-class metrics
- Training history and model checkpoints

### **Evaluation Metrics**
- Overall accuracy
- Per-class precision, recall, F1-score
- Confusion matrix analysis
- Model parameter counting
- Training time tracking

## 🎯 Expected Performance

Based on the model architectures and problem complexity:

| **Model** | **Expected Accuracy** | **Training Time** | **Parameters** |
|-----------|----------------------|-------------------|----------------|
| Multi-Scale CNN | 85-92% | Fast (~30 min) | ~100k |
| Covariance ResNet | 88-94% | Medium (~45 min) | ~200k |
| Hybrid CNN-RNN | 86-91% | Fast (~35 min) | ~150k |
| Signal Transformer | 90-96% | Slow (~90 min) | ~500k |

*Estimates based on 8-antenna ULA with 1024 time samples*

## 📁 Output Structure

After training, the following files are generated:

```
models/classification/
├── best_model.pth                    # Best model checkpoint
├── training_history.json             # Training metrics and results
├── confusion_matrix_epoch_X.png      # Confusion matrix visualization
└── logs/                             # TensorBoard logs
    ├── events.out.tfevents.*          # Training/validation curves
    └── ...
```

## 🔧 Customization

### **Adding New Scenarios**
To modify the scenario classification scheme, edit the `ScenarioLabelEncoder` class:

```python
# In src/training/classification_training.py
def _create_scenario_mapping(self) -> Dict[Tuple, int]:
    # Modify source_multipath_combos for your use case
    source_multipath_combos = [
        (1, 0, 0),  # Your scenario definitions
        (2, 0, 0),
        # ... add more scenarios
    ]
    # Update SNR thresholds if needed
    def _categorize_snr(self, snr: float) -> str:
        if snr < -5:    # Adjust thresholds
            return "low"
        elif snr < 5:
            return "mid"
        else:
            return "high"
```

### **Custom Model Architecture**
Create new models by inheriting from `nn.Module`:

```python
class CustomScenarioClassifier(nn.Module):
    def __init__(self, N=8, T=1024, num_classes=36):
        super().__init__()
        # Your custom architecture
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, T] complex received signals
        # return: [B, num_classes] logits
        pass
```

### **Training Configuration**
All training parameters can be customized via `ClassificationTrainingParams`:

```python
params = ClassificationTrainingParams(
    # Data splits
    train_split=0.8,
    val_split=0.1,
    test_split=0.1,
    
    # Training
    epochs=100,
    batch_size=32,
    learning_rate=1e-3,
    
    # Regularization
    dropout_rate=0.5,
    label_smoothing=0.1,
    weight_decay=1e-4,
    
    # Hardware
    device="auto",  # "cuda", "cpu", or "auto"
    num_workers=4,
)
```

## 🔍 Troubleshooting

### **Common Issues**

1. **Out of Memory Error**
   ```python
   # Reduce batch size
   params.batch_size = 16
   
   # Or use gradient accumulation
   # Implement in training loop if needed
   ```

2. **Poor Convergence**
   ```python
   # Try different learning rate
   params.learning_rate = 5e-4
   
   # Use different scheduler
   params.scheduler = "plateau"
   
   # Increase regularization
   params.dropout_rate = 0.6
   params.label_smoothing = 0.2
   ```

3. **Unbalanced Classes**
   ```python
   # Add class weights to loss function
   criterion = nn.CrossEntropyLoss(
       weight=class_weights,
       label_smoothing=params.label_smoothing
   )
   ```

### **Performance Optimization**

1. **Fast Training**
   - Use `scenario_cnn` or `scenario_hybrid`
   - Increase batch size
   - Use mixed precision training (if CUDA available)

2. **High Accuracy**
   - Use `scenario_transformer` with more epochs
   - Ensemble multiple models
   - Add data augmentation

3. **Memory Efficiency**
   - Use `scenario_hybrid` with smaller RNN hidden size
   - Reduce temporal downsampling
   - Use gradient checkpointing

## 📚 Citation

If you use this scenario classification system in your research, please cite:

```bibtex
@software{tri4net_scenario_classification,
  title={Scenario Classification for DoA Estimation},
  author={Tri4Net Team},
  year={2024},
  url={https://github.com/your-repo/Tri4Net}
}
```

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- New model architectures
- Data augmentation techniques  
- Multi-modal scenarios (e.g., including frequency info)
- Real-time inference optimization
- Transfer learning capabilities

## 📄 License

This project is licensed under the same terms as the main Tri4Net project. 