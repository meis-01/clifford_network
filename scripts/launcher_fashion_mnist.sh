python -m clifford_network sweep --config configs/experiments/fashion_mnist_fft_init_depth.yaml
python -m clifford_network analyze --results-dir results/run_128/fashion_mnist_fft_init_depth
python -m clifford_network report --results-dir results/run_128/fashion_mnist_fft_init_depth