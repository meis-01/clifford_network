python -m clifford_network sweep --config configs/experiments/mnist_fft_init_depth.yaml
python -m clifford_network analyze --results-dir results/runs/mnist_fft_init_depth
python -m clifford_network report --results-dir results/runs/mnist_fft_init_depth