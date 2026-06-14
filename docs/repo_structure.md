# Refactored Repository Structure

The refactored repository is organized around one experiment runner. Dataset,
model, activation, and initialization choices are selected by YAML config.

```text
configs/
├─ smoke/              # fast checks for runner correctness
├─ experiments/        # active experiment designs
└─ paper_experiments/  # frozen paper-producing configs

src/clifford_network/
├─ initialization/
├─ activations/
├─ models/
├─ data/
├─ training/
├─ experiments/
├─ analysis/
└─ utils/
```

The old implementation is kept in `archive/` as migration reference.
