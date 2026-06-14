# Running on a small data subset — quickstart guide

Two files need to go into your project, then you pick a command below.

---

## Step 1 — drop in the two files

```
your_project/
├── configs/
│   └── debug_small.yaml        ← new
└── src/autoencoder/
    └── dataset.py              ← replace with patched version
```

The only change to `dataset.py` is three lines that read `config["data"].get("max_samples")`
and slice `paths[:max_samples]`.  Everything else is identical to the original.

---

## Step 2 — adjust `debug_small.yaml` for your machine

Open `configs/debug_small.yaml` and set:

```yaml
data:
  max_samples: 50      # how many files per split (train/val/test each get 50)

model:
  channels: [64, 32, 16]   # tiny model; raise back to [1024,512,256,128,64] for full run

training:
  batch_size: 4        # 4 is safe on CPU; try 16 or 32 if you have a GPU
  epochs: 5            # just enough to see if it trains
```

---

## Step 3 — pick a command

### A) Single model, one init method (fastest sanity check — ~1 min on CPU)

```bash
python -m src.autoencoder.train \
    --config configs/debug_small.yaml
```

### B) Compare two or three methods (recommended first run)

```bash
python -m src.autoencoder.comparison_train \

    --config configs/debug_small.yaml \

    --methods structured_preserve trabelsi \ 

    --log-level INFO
```

### C) Full 6-method comparison on subset

```bash
python -m src.autoencoder.comparison_train \
    --config configs/debug_small.yaml \
    --log-level INFO
```

### D) Analysis only (after training finishes)

```bash
python -m src.autoencoder.comparison_analysis \
    --results-dir runs/debug_small \
    --methods xavier he trabelsi
```

---

## What changes between debug and production

| Setting | debug_small.yaml | production |
|---|---|---|
| `data.max_samples` | 50 per split | null (all data) |
| `model.channels` | `[64, 32, 16]` | `[1024, 512, 256, 128, 64]` |
| `model.latent_channels` | 16 | 64 |
| `training.epochs` | 5 | 50 |
| `training.patience` | 3 | 10 |
| `training.batch_size` | 4 | 128 |
| `training.output_dir` | `runs/debug_small` | your production path |

To switch back to the full run, use your original production config — no other code changes needed.

---

## How `max_samples` works

`dataset.py` reads `config["data"].get("max_samples")` and slices:

```python
all_paths = all_paths[:max_samples]
```

It applies to each split independently, so with `max_samples: 50` you get
50 train + 50 val + 50 test files.  The files are picked in sorted order
(deterministic), so results are reproducible.

Setting `max_samples: null` (or omitting the key) uses all available data —
no behaviour change from the original.

---

## Tips

- **Just checking the pipeline works?** Use `max_samples: 10` and `epochs: 2` —
  a full comparison of 3 methods finishes in under 2 minutes on CPU.

- **Checking init method ranking?** Use `max_samples: 200` and `epochs: 20` —
  the relative ranking of methods is usually stable on a 200-sample subset.

- **GPU available?** Set `batch_size: 32` or higher and `num_workers: 4` for
  faster iteration.