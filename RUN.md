# Running the Benchmarks

All commands should be run from the **project root** (`hsi-foodanomaly/`).

## `benchmark.py` — Run Everything at Once

Runs all four architectures in sequence, saves per-arch JSON, and prints a combined summary table.

```bash
# All architectures, all food types
python benchmark.py

# Specific food types
python benchmark.py --food Almond Pistachio

# Select specific architectures
python benchmark.py --only ours pa2e
python benchmark.py --skip GT-HAD

# Override epochs/iters
python benchmark.py --phase1-epochs 10 --phase2-epochs 50 --num-iters 100

# Custom output directory
python benchmark.py --output-dir ./results/run1
```

### Output

| File | Contents |
|------|----------|
| `results/ours.json` | Conv1D results |
| `results/pa2e.json` | FC results |
| `results/gt-had.json` | GT-HAD results |
| `results/bocknet.json` | BockNet results |
| `results/benchmark_all.json` | All architectures combined |

---

## Individual Architectures

| Script | Description |
|--------|-------------|
| `scripts/ours` | PA2E with **Conv1D** partial encoders + Deep k-NN inference |
| `scripts/pa2e` | PA2E with **FC** partial encoders + linear layer fusion |
| `scripts/GT-HAD` | Graph Transformer (GT-HAD) |
| `scripts/bocknet` | Blind-Block Reconstruction Network with a Guard Window |

---

## `scripts/ours` — Conv1D variant

```bash
# All food types (default)
python -m scripts.ours

# Single food type
python -m scripts.ours --food Almond

# Multiple food types
python -m scripts.ours --food Almond Pistachio

# Custom hyperparameters
python -m scripts.ours --food Almond --batch-size 256 --phase1-epochs 30 --phase2-epochs 150

# Custom output directory
python -m scripts.ours --output-dir ./results/ours
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--food` | all | Food type(s): `Almond`, `Pistachio`, `GarlicStems` |
| `--batch-size` | `512` | DataLoader batch size |
| `--phase1-epochs` | `20` | Pre-training epochs (reconstruction loss) |
| `--phase2-epochs` | `100` | Main training epochs (DSVDD-style + CosineAnnealingLR) |
| `--output-dir` | `./results` | Directory for `ours.json` results file |

---

## `scripts/pa2e` — FC variant

```bash
# All food types (default)
python -m scripts.pa2e

# Single food type
python -m scripts.pa2e --food Pistachio

# Multiple food types
python -m scripts.pa2e --food Almond GarlicStems

# Custom hyperparameters
python -m scripts.pa2e --food Almond --batch-size 256 --phase1-epochs 30 --phase2-epochs 50

# Custom output directory
python -m scripts.pa2e --output-dir ./results/pa2e
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--food` | all | Food type(s): `Almond`, `Pistachio`, `GarlicStems` |
| `--batch-size` | `512` | DataLoader batch size |
| `--phase1-epochs` | `20` | Pre-training epochs (reconstruction loss) |
| `--phase2-epochs` | `20` | Main training epochs (DSVDD-style) |
| `--output-dir` | `./results` | Directory for `pa2e.json` results file |

---

## `scripts/GT-HAD` — Graph Transformer (GT-HAD)

> Image-level self-supervised model — no pixel train/test split.
> The full test image is the self-supervised context; the train image is used only for loading.

```bash
# All food types (default)
python -m scripts.GT-HAD

# Single food type
python -m scripts.GT-HAD --food Almond

# Multiple food types
python -m scripts.GT-HAD --food Almond Pistachio

# Custom hyperparameters
python -m scripts.GT-HAD --food Almond --num-iters 200 --search-iter 50 --batch-size 128

# Custom output directory
python -m scripts.GT-HAD --output-dir ./results/gt-had
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--food` | all | Food type(s): `Almond`, `Pistachio`, `GarlicStems` |
| `--num-iters` | `150` | Total training iterations |
| `--search-iter` | `25` | CMM search interval (every N iterations) |
| `--batch-size` | `64` | DataLoader batch size |
| `--lr` | `1e-3` | Adam learning rate |
| `--output-dir` | `./results` | Directory for `gt-had.json` results file |

---

## `scripts/bocknet` — BockNet

> Blind-Block Reconstruction Network with a Guard Window for Hyperspectral Anomaly Detection.

```bash
# All food types (default)
python -m scripts.bocknet

# Single food type
python -m scripts.bocknet --food Almond

# Multiple food types
python -m scripts.bocknet --food Almond Pistachio

# Custom hyperparameters
python -m scripts.bocknet --food Almond --epochs 3000 --lr 1e-4 --blindspot 15

# Custom output directory
python -m scripts.bocknet --output-dir ./results/bocknet
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--food` | all | Food type(s): `Almond`, `Pistachio`, `GarlicStems` |
| `--epochs` | `3000` | Training epochs |
| `--lr` | `1e-4` | Learning rate |
| `--blindspot` | `15` | Blindspot guard window size |
| `--retrain` | `yes` | Retrain model if present (`yes` or `no`) |
| `--output-dir` | `./results` | Directory for `bocknet.json` results file |

---

## Dataset Structure

The dataset can be acquired here: [IEEE DataPort - Anomaly Detection in Hyperspectral Imaging for Food Safety Inspection](https://ieee-dataport.org/documents/anomaly-detection-hyperspectral-imaging-food-safety-inspection)

Both scripts expect the dataset at `AnomalyonFood/Dataset/` relative to the project root:

```
AnomalyonFood/Dataset/
  Almond/
    Train/  data.hdr  WHITEREF.hdr  DARKREF.hdr
    Test/   data.hdr  WHITEREF.hdr  DARKREF.hdr  label.npy
  Pistachio/   ...
  GarlicStems/ ...
```

## Output

Results are saved as JSON to `--output-dir`:

- `ours.json` — ROC-AUC, PR-AUC, inference time for original and Conv+BN-fused model
- `pa2e.json` — ROC-AUC, PR-AUC, inference time for original and Linear-fused model
- `gt-had.json` — ROC-AUC, PR-AUC, inference time for GT-HAD
- `bocknet.json` — ROC-AUC, PR-AUC, inference time for BockNet
