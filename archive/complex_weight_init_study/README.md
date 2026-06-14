# Complex Weight Initialization Study

This project evaluates robust weight initialization strategies for complex-valued dense neural networks. It compares how different complex initialization schemes affect forward-pass activation magnitudes, saturation, and backward gradient stability across many layers during training.

## Goals

- Build a deep complex dense network using PyTorch complex tensors
- Implement multiple complex activation functions
- Implement multiple complex initialization strategies
- Train models and monitor activation statistics in deeper layers over epochs
- Analyze activation value distributions, saturation, and vanishing signals
- Compare different weight initialization schemes to determine which performs better

## Features

- **Modular Design**: Separate modules for data, layers, initialization, training, and analysis
- **Training Monitoring**: Track activation statistics in specified deeper layers during training
- **Distribution Analysis**: Plot activation value distributions and evolution over epochs
- **Saturation/Vanishing Analysis**: Detect and quantify saturation and vanishing activation problems
- **Comparative Analysis**: Compare initialization methods across multiple metrics

## Requirements

- Python 3.10+
- PyTorch
- NumPy
- Matplotlib
- Seaborn
- Pandas

You can install dependencies with:

```bash
pip install -r requirements.txt
```

## Project Structure

```
complex_weight_init_study/
├── README.md
├── requirements.txt
├── complex_layers.py          # Complex neural network layers and activations
├── initialization.py          # Complex weight initialization methods
├── data_module.py             # Data generation and loading
├── training_module.py         # Training loop with activation monitoring
├── analysis_module.py         # Analysis and plotting functions
├── train_experiments.py       # Main script for running training experiments
└── run_experiment.py          # Original forward-pass only experiments
```

## Available Initialization Methods

The project supports the following complex weight initialization methods:

- **`xavier`**: Xavier/Glorot initialization with complex Gaussian noise
- **`he`**: He/Kaiming initialization for complex networks  
- **`unitary`**: Random unitary matrices (complex phases only)
- **`random`**: Standard normal complex initialization
- **`trabelsi`**: Rayleigh magnitude with random phase (from Trabelsi et al.)
- **`structured_preserve`**: Structured signal preservation initialization

### Structured Signal Preservation (`structured_preserve`)

This initialization implements a structured approach that ensures direct signal preservation across layers:

**Mathematical Formulation:**
```
D_{i,j} = e^{iθ}  if i ≡ j (mod N_{ℓ-1}) and θ ∈ (−π,π)
        = 0       otherwise

W = D + Z
```

Where:
- `D` is the structured matrix with one-to-one neuron mappings
- `Z_{ij} ∼ N(0, σ_z²)` is additive complex Gaussian noise
- `σ_z = α / √(N_{ℓ-1})` with empirical value `α = 0.085`

**Purpose:** Maintains structured connectivity patterns that preserve signal flow while adding controlled noise for regularization.

## Run Training Experiments

### Using YAML Configuration (Recommended)

The project now supports YAML-based configuration for easy experiment setup:

```bash
# Create default configuration files
python train_experiments.py --create-configs

# Run experiments using YAML config
python train_experiments.py --config configs/comparison.yaml
python train_experiments.py --config configs/deep_network.yaml
python train_experiments.py --config configs/cifar10_fft.yaml
python train_experiments.py --config configs/mnist_fft.yaml
python train_experiments.py --config configs/comparison.yaml
python train_experiments.py --config configs/deep_network.yaml
python train_experiments.py --config configs/cifar10_fft.yaml
python train_experiments.py --config configs/mnist_fft.yaml
python train_experiments.py --config configs/single_experiment.yaml
```

### Available Configuration Files

- **`configs/single_experiment.yaml`**: Single activation/init method combination
- **`configs/comparison.yaml`**: Compare multiple methods with multiple activations
- **`configs/deep_network.yaml`**: Deep network analysis with many layers

### Custom Configuration

Create your own YAML config files to specify:
- Desired activation functions
- Selected weight initialization methods
- Number of layers
- Training parameters
- Layers to monitor for analysis
- Dataset source and root directory

Example custom config:
```yaml
activations:
- modrelu
- zrelu
init_methods:
- xavier
- he
- structured_preserve
in_features: 64
hidden_size: 128
n_layers: 10
epochs: 50
batch_size: 128
device: cpu
monitor_layers: [4, 7, 9]
dataset_name: synthetic
dataset_root: data
output_dir: my_custom_experiment
```

For FFT-based complex image experiments on MNIST or CIFAR10, set `dataset_name` accordingly and use the flattened FFT input size:
```yaml
activations:
- modrelu
- zrelu
init_methods:
- xavier
- he
- structured_preserve
in_features: 784
hidden_size: 128
n_layers: 10
epochs: 50
batch_size: 128
device: cpu
monitor_layers: [4, 7, 9]
dataset_name: mnist
dataset_root: data
output_dir: mnist_fft_experiment
```
```yaml
activations:
- modrelu
- zrelu
init_methods:
- xavier
- he
- structured_preserve
in_features: 3072
hidden_size: 128
n_layers: 10
epochs: 50
batch_size: 128
device: cpu
monitor_layers: [4, 7, 9]
dataset_name: cifar10
dataset_root: data
output_dir: cifar10_fft_experiment
```

### Legacy Command Line Usage

For backward compatibility, command line arguments are still supported:

```bash
python train_experiments.py --hidden-size 128 --n-layers 100 --epochs 100 --batch-size 256
```

To compare specific activations and initializations:

```bash
python train_experiments.py --activations modrelu zrelu tanh --inits xavier he unitary --epochs 50
```

To monitor specific layers:

```bash
python train_experiments.py --monitor-layers 4 7 9 --n-layers 10
```

## Quick Initialization Comparison

For a focused comparison of initialization methods:

```bash
python compare_initializations.py
```

This uses the `configs/comparison.yaml` configuration to compare multiple initialization methods.

## Run Forward-Pass Only Experiments

For quick analysis without training:

```bash
python run_experiment.py --hidden-size 128 --n-layers 10 --batch-size 256
```

## Output

The training experiments save:

- **Training curves**: Loss evolution over epochs
- **Activation evolution**: How activation statistics change during training
- **Distribution plots**: Activation value distributions at different epochs
- **Saturation/vanishing analysis**: Ratios of saturated/vanishing activations
- **Comparison plots**: Side-by-side comparison of initialization methods
- **CSV summaries**: Tabular data for further analysis

All plots and data are saved in the `training_results/` directory (or custom output directory).

## Analysis Focus

The project specifically analyzes:

1. **Activation Magnitude Evolution**: How activation magnitudes change over training epochs
2. **Saturation Detection**: High activation values that may cause numerical instability
3. **Vanishing Signals**: Very small activation values that may cause gradient issues
4. **Distribution Shifts**: How activation distributions evolve and stabilize
5. **Initialization Robustness**: Which methods provide stable training across different activations
