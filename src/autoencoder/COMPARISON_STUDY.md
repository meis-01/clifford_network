# Weight Initialization Comparison Study

This module provides tools to train and compare a large complex autoencoder model with different weight initialization schemes.

## Overview

The weight initialization is a critical factor in deep learning performance. This study compares six different initialization methods for complex-valued neural networks:

1. **Random**: Simple random initialization with fixed standard deviation
2. **Xavier (Glorot)**: Uniform variance initialization adapted for complex networks
3. **He Initialization**: Tailored for ReLU-like activations, adapted for complex networks
4. **Unitary**: Random unitary matrix initialization (preserves information flow)
5. **Trabelsi**: Rayleigh-distributed initialization specifically designed for complex networks
6. **Structured Preserve**: Signal-preserving initialization with structured components

## Model Architecture

The large autoencoder model has:
- **Encoder**: 1 → 64 → 128 → 256 → 512 → 1024 → 512 → 256
- **Bottleneck**: 1024 complex-valued features
- **Decoder**: 256 → 512 → 1024 → 512 → 256 → 128 → 64 → 1
- **Total Parameters**: ~500M+ for complex tensor operations
- **Activation**: ModReLU (complex-valued activation)

## Components

### 1. Configuration File
**File**: `configs/large_autoencoder.yaml`

Defines the large model architecture and training parameters:
- Deep encoder/decoder with increasing/decreasing channels
- 1024-dimensional latent space
- Extended training (100 epochs with early stopping)
- Optimized learning rates for large models

### 2. Comparison Training Script
**File**: `src/autoencoder/comparison_train.py`

Trains the autoencoder with different weight initialization schemes.

#### Usage:
```bash
# Train all initialization methods
python -m src.autoencoder.comparison_train \
    --config configs/large_autoencoder.yaml \
    --log-level INFO

# Train specific methods only
python -m src.autoencoder.comparison_train \
    --config configs/large_autoencoder.yaml \
    --methods xavier he unitary \
    --log-level INFO
```

#### Features:
- Trains each initialization method independently in separate directories
- Saves best and last model checkpoints for each method
- Logs detailed training metrics (loss, NMSE, convergence) per epoch
- Early stopping with patience-based stopping criteria
- Generates individual summaries for each initialization method

#### Output Structure:
```
runs/weight_init_comparison/
├── comparison.log                    # Main comparison log
├── comparison_summary.json           # Overall results summary
├── init_random/
│   ├── train.log
│   ├── config.yaml
│   ├── history.json                 # Epoch-by-epoch metrics
│   ├── summary.json                 # Training summary
│   ├── best_model.pt                # Best checkpoint
│   └── last_model.pt                # Final checkpoint
├── init_xavier/
├── init_he/
├── init_unitary/
├── init_trabelsi/
└── init_structured_preserve/
```

### 3. Analysis and Visualization Script
**File**: `src/autoencoder/comparison_analysis.py`

Analyzes and visualizes results from the comparison study.

#### Usage:
```bash
# Generate full analysis and plots
python -m src.autoencoder.comparison_analysis \
    --results-dir runs/weight_init_comparison \
    --methods random xavier he unitary trabelsi structured_preserve \
    --output-dir runs/weight_init_comparison/plots

# Specify custom output directory
python -m src.autoencoder.comparison_analysis \
    --results-dir runs/weight_init_comparison \
    --methods xavier he \
    --output-dir custom/output/path
```

#### Generated Outputs:

1. **comparison_overview.png** (4-subplot figure):
   - Training and validation loss curves
   - Training and validation NMSE curves
   - Best validation loss bar chart
   - Training time and convergence epoch comparison

2. **detailed_loss_curves.png** (2-subplot figure):
   - Validation loss curves (all methods)
   - Generalization gap over training

3. **comparison_summary.csv**:
   - Tabular summary with metrics for all methods
   - Columns: Method, Best Val Loss, Best Epoch, Total Epochs, Training Time, Final Metrics, etc.
   - Sorted by best validation loss

4. **Console Output**:
   - Summary table with key metrics
   - Convergence analysis including:
     - Initial and final loss
     - Improvement percentage
     - Epochs to convergence
     - Stability metrics

## Metrics Explained

### Loss and NMSE
- **Loss**: Complex MSE (Mean Squared Error) between reconstruction and target
- **NMSE**: Normalized MSE relative to target signal power

### Convergence Metrics
- **Best Epoch**: Epoch with lowest validation loss
- **Improvement %**: Percentage reduction in loss from initial to final epoch
- **Loss Gap**: Difference between validation and training loss (generalization gap)
- **Convergence Rate**: Fraction of training budget needed to reach best performance

### Training Properties
- **Training Time**: Total wall-clock time in minutes
- **Stability**: Standard deviation of loss in final 10% of training
- **Epochs to Convergence**: How quickly best loss is achieved

## Complete Workflow

### Step 1: Run Comparison Training
```bash
cd /home/mad07/clifford_network

# Run training with all methods (recommended for first run)
python -m src.autoencoder.comparison_train \
    --config configs/large_autoencoder.yaml \
    --log-level INFO

# Monitor progress in separate terminal
tail -f runs/weight_init_comparison/comparison.log
```

Expected duration: 5-10 hours depending on hardware (GPU recommended)

### Step 2: Analyze Results
```bash
# After training completes, run analysis
python -m src.autoencoder.comparison_analysis \
    --results-dir runs/weight_init_comparison \
    --methods random xavier he unitary trabelsi structured_preserve
```

### Step 3: Review Results
The generated visualizations and tables will show:
1. Which initialization method provides fastest convergence
2. Which achieves the lowest final loss
3. Generalization properties (gap between train/val loss)
4. Stability and variance in training curves
5. Computational efficiency (training time per method)

## Tips for Running

### For Quick Evaluation
```bash
# Train only the most promising methods
python -m src.autoencoder.comparison_train \
    --config configs/large_autoencoder.yaml \
    --methods xavier he unitary
```

### Using GPU
The scripts automatically detect and use CUDA if available. Set in config:
```yaml
training:
  device: auto  # Automatically selects GPU if available
```

### Adjusting Training Parameters
Edit `configs/large_autoencoder.yaml`:
- Reduce `epochs` for quick testing
- Increase `batch_size` if GPU memory allows (faster but more GPU VRAM)
- Adjust `learning_rate` if convergence is problematic
- Modify `patience` for more/less aggressive early stopping

## Expected Results

For the large autoencoder on complex MRI data:
- **Best Methods**: Xavier and He typically perform well
- **Convergence**: Most methods converge within 30-50 epochs
- **NMSE Range**: 0.001-0.01 depending on initialization quality
- **Training Time**: 20-40 minutes per method on GPU

## Files Modified/Created

### New Files:
- `configs/large_autoencoder.yaml` - Large model configuration
- `src/autoencoder/comparison_train.py` - Comparison training script
- `src/autoencoder/comparison_analysis.py` - Analysis and visualization

### Existing Files (Not Modified):
- `src/autoencoder/model.py` - Supports weight_init parameter
- `src/autoencoder/weight_initializations.py` - Contains all init methods
- `src/autoencoder/train.py` - Original training script
- `src/autoencoder/config.py` - Configuration loading

## Advanced: Custom Initialization Methods

To add a new initialization method:

1. **Add to `weight_initializations.py`**:
   ```python
   elif method == "my_custom":
       # Your initialization logic
       values = torch.complex(...)
   ```

2. **Test in comparison**:
   ```bash
   python -m src.autoencoder.comparison_train \
       --config configs/large_autoencoder.yaml \
       --methods my_custom
   ```

## Troubleshooting

### Out of Memory
- Reduce `batch_size` in config (4 → 2)
- Reduce model size by modifying `channels` in config
- Use gradient accumulation (advanced)

### Slow Training
- Increase `num_workers` in config (0 → 4)
- Use GPU (set `device: auto`)
- Increase `batch_size` if memory allows

### Loss Not Decreasing
- Check learning rate (try 1e-3 to 1e-5)
- Verify data is properly normalized
- Check for NaN in metrics files

## References

- [Complex-Valued Neural Networks](https://arxiv.org/abs/2402.01298)
- [Initialization for Complex Networks](https://arxiv.org/abs/1504.04724)
- [Trabelsi Initialization](https://arxiv.org/abs/1705.09792)
