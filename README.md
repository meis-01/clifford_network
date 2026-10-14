# Clifford Network

Reproducible experiments for complex-valued neural networks, focused on how
weight initialization and model depth affect training success, activation flow,
and gradient flow.

The main comparison is between `structured_preserve` and `trabelsi`, with
standard baselines included for context. The intended paper experiments cover:

- fastMRI T2 complex autoencoder reconstruction
- MNIST FFT classification
- Fashion-MNIST FFT classification
- CIFAR10 FFT classification
- CIFAR100 FFT classification

All experiments are driven through one CLI:

```powershell
python -m clifford_network run --config configs/smoke/synthetic_classification.yaml
python -m clifford_network sweep --config configs/experiments/mnist_fft_init_depth.yaml
python -m clifford_network analyze --results-dir results/runs/synthetic_classification_smoke
python -m clifford_network report --results-dir results/runs/synthetic_classification_smoke
```
## Repository Layout

```text
src/clifford_network/
├─ activations/       # complex activation modules and registry
├─ analysis/          # HoloViews/Bokeh plots, aggregation, tables, reports
├─ dataset/           # dataset adapters: synthetic, FFT vision, fastMRI T2
├─ experiments/       # config loading, single-run and sweep orchestration
├─ initialization/    # shared complex initialization methods
├─ models/            # shared complex layers, MLPs, and autoencoder MLP
├─ training/          # losses, metrics, local activation/gradient/weight monitoring
└─ utils/             # IO, device, seed helpers
```


The previous repository state is kept under `archive/` for reference during the
migration.


## Complex Batch Normalization

This module implements **Complex Batch Normalization** for complex-valued neural networks, following the whitening-based approach introduced by Trabelsi et al. in *Deep Complex Networks* (ICLR 2018).

Unlike standard batch normalization, which is designed for real-valued data, complex batch normalization jointly normalizes the real and imaginary components using their covariance matrix. This accounts for differences in their variances and their correlation.

### Features

- Supports native PyTorch complex tensors (`torch.complex64` and `torch.complex128`).
- Performs joint whitening of real and imaginary components using a \(2 \times 2\) covariance matrix.
- Includes learnable affine scaling and complex bias parameters.
- Tracks running means, variances, and covariance for evaluation.
- Supports configurable numerical stability (`eps`), momentum, affine transformation, and running-statistics tracking.
- Supports inputs with arbitrary leading dimensions, with the feature dimension as the final dimension.

### Mathematical Formulation

For each complex feature \(z=x+iy\), the layer computes the means of the real and imaginary components and constructs the covariance matrix

\[
V =
\begin{pmatrix}
V_{rr} & V_{ri}\\
V_{ri} & V_{ii}
\end{pmatrix},
\]

where \(V_{rr}\) and \(V_{ii}\) denote the component variances and \(V_{ri}\) denotes their covariance.

The centered components are jointly whitened using the inverse square root of the regularized covariance matrix:

\[
\begin{pmatrix}
\hat{x}\\
\hat{y}
\end{pmatrix}
=
(V+\epsilon I)^{-1/2}
\begin{pmatrix}
x-\mathbb{E}[x]\\
y-\mathbb{E}[y]
\end{pmatrix}.
\]

A learnable symmetric scaling matrix and complex bias are then applied:

\[
\begin{pmatrix}
x_{\mathrm{out}}\\
y_{\mathrm{out}}
\end{pmatrix}
=
\begin{pmatrix}
\gamma_{rr} & \gamma_{ri}\\
\gamma_{ri} & \gamma_{ii}
\end{pmatrix}
\begin{pmatrix}
\hat{x}\\
\hat{y}
\end{pmatrix}
+
\begin{pmatrix}
\beta_r\\
\beta_i
\end{pmatrix}.
\]

The layer computes the inverse square root analytically for the \(2 \times 2\) covariance matrix.

### Usage

Import the layer from `layers.py`:

```python
from your_package.layers import ComplexBatchNorm
```

Initialize a layer with 64 complex features:

```python
import torch
from your_package.layers import ComplexBatchNorm

bn = ComplexBatchNorm(num_features=64)

x = torch.randn(128, 64, dtype=torch.complex64)
y = bn(x)

print(y.shape)  # torch.Size([128, 64])
print(y.dtype)  # torch.complex64
```

Replace `your_package` with the appropriate package name in your project.

### Configuration

| Parameter | Default | Description |
|---|---:|---|
| `num_features` | Required | Number of complex features in the final dimension |
| `eps` | `1e-5` | Numerical regularization for covariance whitening |
| `momentum` | `0.1` | Weight assigned to the current batch statistics when updating running statistics |
| `affine` | `True` | Enables learnable scaling and bias parameters |
| `track_running_stats` | `True` | Tracks running statistics for evaluation |

### Integration with Complex Neural Networks

Complex batch normalization can be applied after a complex linear layer and before a complex activation function:

```python
self.linear = ComplexLinear(64, 64)
self.bn = ComplexBatchNorm(64)

def forward(self, x):
    x = self.linear(x)
    x = self.bn(x)
    x = torch.complex(
        torch.tanh(x.real),
        torch.tanh(x.imag),
    )
    return x
```

### Training and Evaluation

During training, the layer uses statistics computed from the current batch and updates its running estimates. During evaluation, it uses the accumulated running statistics when `track_running_stats=True`.

Call `model.train()` before training and `model.eval()` during evaluation.

### Experimental Considerations

For weight-initialization experiments, compare models with and without complex batch normalization under otherwise identical settings. This helps distinguish the effects of covariance whitening from the effects of the initialization method.

The learnable affine matrix is symmetric, but its positive-semidefiniteness is not explicitly enforced during training.

### Reference

Trabelsi, C., et al. (2018). *Deep Complex Networks*. International Conference on Learning Representations (ICLR).

Paper: https://openreview.net/forum?id=H1T2hmZAb
