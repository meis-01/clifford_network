python -m clifford_network sweep --config configs/experiments/fastmri_t2_init_depth.yaml --results-dir results/runs/fastmri_t2_init_depth
python -m clifford_network analyze --results-dir results/runs/fastmri_t2_init_depth
python -m clifford_network report --results-dir results/runs/fastmri_t2_init_depth