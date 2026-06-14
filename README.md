# Clifford Network

Reproducible experiments for complex-valued neural networks, focused on how
weight initialization and model depth affect training success, activation flow,
and gradient flow.

The main comparison is between `structured_preserve` and `trabelsi`, with
standard baselines included for context. The intended paper experiments cover:

- fastMRI T2 complex autoencoder reconstruction
- MNIST FFT classification
- CIFAR10 FFT classification

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
├─ data/              # dataset adapters: synthetic, MNIST FFT, CIFAR10 FFT, fastMRI T2
├─ experiments/       # config loading, single-run and sweep orchestration
├─ initialization/    # shared complex initialization methods
├─ models/            # shared complex layers, MLPs, and autoencoder MLP
├─ training/          # losses, metrics, local activation/gradient/weight monitoring
└─ utils/             # IO, device, seed helpers
```

The previous repository state is kept under `archive/` for reference during the
migration.
