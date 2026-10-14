python -m clifford_network sweep --config configs/experiments_32/mnist_fft_init_depth_32.yaml
python -m clifford_network analyze --results-dir results/run_32/mnist_fft_init_depth
python -m clifford_network report --results-dir results/run_32/mnist_fft_init_depth


python -m clifford_network sweep --config configs/experiments_32/cifar10_fft_init_depth_32.yaml
python -m clifford_network analyze --results-dir results/run_32/cifar10_fft_init_depth
python -m clifford_network report --results-dir results/run_32/cifar10_fft_init_depth


python -m clifford_network sweep --config configs/experiments_32/cifar100_fft_init_depth_32 .yaml
python -m clifford_network analyze --results-dir results/run_32/cifar100_fft_init_depth
python -m clifford_network report --results-dir results/run_32/cifar100_fft_init_depth


python -m clifford_network sweep --config configs/experiments_32/fashion_mnist_fft_init_depth_32.yaml
python -m clifford_network analyze --results-dir results/run_32/fashion_mnist_fft_init_depth
python -m clifford_network report --results-dir results/run_32/fashion_mnist_fft


