# clifford_network

Minimal scaffold for a fastMRI prostate classification pipeline with modular data loading, reconstruction, training, and experiment tracking.

## What is included

- A package layout under `src/` for data, reconstruction, models, training, experiments, visualization, and utilities.
- Namespace-safe ISMRMRD header parsing to avoid brittle XML lookups.
- A modular T2 reconstruction pipeline built from GRAPPA, padding, inverse FFT, and RSS coil combination.
- Deterministic volume-wise splitting utilities to avoid slice leakage.
- A dependency-light `main.py` that can run a synthetic smoke test without dataset access.

## Quick start

Create or activate your environment, then install the project from the repository root:

```powershell
pip install -e .
```

If you prefer installing from the raw dependency list instead, you can still use `requirements.txt`.

After installation, run:

```powershell
c:/Users/meisa/Projects/clifford_network/.venv/Scripts/python.exe main.py --smoke-test
```

To scan a local dataset tree:

```powershell
c:/Users/meisa/Projects/clifford_network/.venv/Scripts/python.exe main.py --config configs/default.yaml
```

## Project layout

- `src/data`: discovery, label handling, loaders, datasets, and split logic.
- `src/reconstruction`: header parsing, padding, regridding, GRAPPA, coil combination, and derived maps.
- `src/models`: baseline model interfaces and optional PyTorch models.
- `src/training`: training and evaluation helpers.
- `src/experiments`: experiment tracking adapters.
- `src/visualization`: HoloViews-based plotting helpers.
- `tests`: unit and integration smoke tests.

## Notes

- The current scratch notebook remains in the repository root as `canvas.ipynb`.
- The included reconstruction code is a solid baseline for T2 experiments, but DWI regridding and downstream modeling still need dataset-specific validation before training results should be trusted.

## Paired K-space Classification

The repository now includes a slice-level classifier under `src/classification` that pairs T2 and DWI volumes and can run four experiment modes:

- `kspace x real`: two channels, T2 and DWI magnitude from filled k-space.
- `kspace x complex`: two native complex channels, `T2` and `DWI`, from physically coil-combined filled k-space.
- `reconstruction x real`: two channels, T2 and DWI magnitude after inverse FFT reconstruction.
- `reconstruction x complex`: two native complex channels, `T2` and `DWI`, after physically coil-combined reconstruction.

The real-valued variants reduce coils after physical sensitivity-weighted image-domain combination and then take magnitude. The complex variants keep native `torch.complex64` tensors through the convolutional backbone and optimize a real-valued objective with PyTorch's built-in conjugate Wirtinger gradients.

The default loader joins the original fastMRI-style T2 and DWI slice CSVs on `(fastmri_pt_id, slice, data_split)` and expects each row to resolve to a real file via `root / folder / fastmri_rawfile`.

Run the real k-space baseline with:

```powershell
c:/Users/meisa/Projects/clifford_network/.venv/Scripts/python.exe -m src.classification.train --config configs/kspace_joint_classifier.yaml
```

Run the other experiment variants with:

```powershell
c:/Users/meisa/Projects/clifford_network/.venv/Scripts/python.exe -m src.classification.train --config configs/kspace_joint_classifier_complex.yaml
c:/Users/meisa/Projects/clifford_network/.venv/Scripts/python.exe -m src.classification.train --config configs/reconstruction_joint_classifier.yaml
c:/Users/meisa/Projects/clifford_network/.venv/Scripts/python.exe -m src.classification.train --config configs/reconstruction_joint_classifier_complex.yaml
```

Evaluate the best checkpoint with:

```powershell
c:/Users/meisa/Projects/clifford_network/.venv/Scripts/python.exe -m src.classification.evaluate --config configs/kspace_joint_classifier.yaml
```

If you want to bypass the separate modality CSVs, point `data.paired_manifest_csv` at a custom CSV with columns `fastmri_pt_id`, `slice`, `data_split`, `label`, `t2_path`, and `dwi_path`.