# Reproduction Workflow

Run a quick local smoke sweep:

```powershell
python -m clifford_network sweep --config configs/smoke/synthetic_classification.yaml
python -m clifford_network analyze --results-dir results/runs/synthetic_classification_smoke
python -m clifford_network report --results-dir results/runs/synthetic_classification_smoke
```

Main experiment configs live under `configs/experiments/`. Frozen paper configs
will live under `configs/paper_experiments/` once the final protocol is fixed.

All visualizations are generated as Bokeh-backed HTML files through HoloViews.
