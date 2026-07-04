# Alfa activation analysis

This folder contains the paper-style activation histogram experiment for the
complex classifier.

Defaults:

- 3000 complex samples with independent real and imaginary values from
  `Uniform(-1, 1)`
- MNIST-shaped classifier input: `28 * 28`
- 1000 hidden layers
- 32 hidden nodes per layer
- `structured_preserve(alpha=alfa)`
- alfa values: `0.00001`, `0.001`, `0.0085`, `0.1`, `1`, `10`
- HoloViews/Bokeh histogram output

Run from the repository root:

```powershell
$env:PYTHONPATH = "src"
python -m clifford_network.alfa_analysis
```

The default output directory is `src/clifford_network/alfa_analysis/results`.
It contains compressed raw activations, `activation_summary.csv`,
`metadata.json`, and `plots/activation_histograms.html`.

For a fast smoke run:

```powershell
$env:PYTHONPATH = "src"
python -m clifford_network.alfa_analysis --alfa-values 0.00001 --num-samples 8 --depth 3 --batch-size 8 --output-dir src/clifford_network/alfa_analysis/smoke_results
```
