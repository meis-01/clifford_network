python -m clifford_network sweep --config configs/experiments/mnist_fft_init_depth_512.yaml
python -m clifford_network analyze --results-dir results/run_512/mnist_fft_init_depth
python -m clifford_network report --results-dir results/run_512 /mnist_fft_init_depth


python -m clifford_network sweep --config configs/experiments/cifar10_fft_init_depth_512.yaml
python -m clifford_network analyze --results-dir results/run_512/cifar10_fft_init_depth
python -m clifford_network report --results-dir results/run_512/cifar10_fft_init_depth


python -m clifford_network sweep --config configs/experiments/cifar100_fft_init_depth_512.yaml
python -m clifford_network analyze --results-dir results/run_512/cifar100_fft_init_depth
python -m clifford_network report --results-dir results/run_512/cifar100_fft_init_depth


python -m clifford_network sweep --config configs/experiments/fashion_mnist_fft_init_depth_512.yaml
python -m clifford_network analyze --results-dir results/run_512/fashion_mnist_fft_init_depth
python -m clifford_network report --results-dir results/run_512/fashion_mnist_fft


