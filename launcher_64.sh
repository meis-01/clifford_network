# python -m clifford_network sweep --config configs/experiments_64/mnist_fft_init_depth_64.yaml
# python -m clifford_network analyze --results-dir results/run_64/mnist_fft_init_depth
# python -m clifford_network report --results-dir results/run_64/mnist_fft_init_depth


# python -m clifford_network sweep --config configs/experiments_64/cifar10_fft_init_depth_64.yaml
# python -m clifford_network analyze --results-dir results/run_64/cifar10_fft_init_depth
# python -m clifford_network report --results-dir results/run_64/cifar10_fft_init_depth


python -m clifford_network sweep --config configs/experiments_64/cifar100_fft_init_depth_64.yaml
python -m clifford_network analyze --results-dir results/run_64/cifar100_fft_init_depth
python -m clifford_network report --results-dir results/run_64/cifar100_fft_init_depth


python -m clifford_network sweep --config configs/experiments_64/fashion_mnist_fft_init_depth_64.yaml
python -m clifford_network analyze --results-dir results/run_64/fashion_mnist_fft_init_depth
python -m clifford_network report --results-dir results/run_64/fashion_mnist_fft


