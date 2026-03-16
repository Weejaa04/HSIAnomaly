# BENCH.md – API & Module Integration Guide

A comprehensive reference for integrating new anomaly detection architectures into the HSI Food Anomaly Detection benchmark framework.

---

## Getting Started (3-Step Quickstart)

If you're cloning this repo and want to add a new architecture:

### Step 1: Create Your Module

Create `scripts/<ARCH>/__main__.py` with a `benchmark_food_type()` function (details in [Section 1](#1-core-api-contract)).

### Step 2: Register in Benchmark

Edit `benchmark.py` and add two lines:

```python
# Line ~24: Add to ARCH_MODULE_MAP
ARCH_MODULE_MAP = {
    'BockNet':'scripts.bocknet.__main__',
    'our': 'scripts.our.__main__',
    '<ARCH>': 'scripts.<ARCH>.__main__',     # ← Add this
}

# Line ~30: Add to ARCH_RESULT_FILE
ARCH_RESULT_FILE = {
    'BockNet':'bocknet.json',
    'our': 'our.json',
    '<ARCH>': '<arch>.json',                 # ← Add this
}
```

### Step 3: Test It

```bash
# Quick sanity check (runs ~10-20 seconds, tests full pipeline)
python benchmark.py --only <ARCH> --dry-run

# Full benchmark (runs all food types)
python benchmark.py --only <ARCH>
```

**Results** are automatically saved to:
- Per-architecture: `results/<arch>.json`
- Combined: `results/benchmark_all.json`

### Project Structure

```
hsi-food-ad/
├── AnomalyonFood/Dataset/     ← HSI data (Food Type / Train,Test / data.hdr, label.npy, refs)
├── scripts/
│   ├── bocknet/               ← Existing architecture
│   ├── our/                   ← Existing architecture
│   └── <your-arch>/           ← YOUR MODULE GOES HERE
│       ├── __main__.py        ← REQUIRED: benchmark_food_type()
│       ├── model.py           ← Optional: your model classes
│       ├── data.py            ← Optional: data loading utilities
│       └── utils.py           ← Optional: helpers
├── weights/                   ← Auto-created: saved model weights
├── results/                   ← Auto-created: JSON results
├── benchmark.py               ← Main orchestrator
├── BENCH.md                   ← This file (API reference)
└── DATA.md                    ← Spatial partitioning protocol
```

### Key Concepts

- **Data Partitioning:** All models use the same spatial guillotine protocol (see [DATA.md](DATA.md))
  - Train: `Y[50:200]`  
  - Val: `Y[300:350]`  
  - Test: Remaining pixels  

- **Weight Saving:** Models automatically save to `./weights/<ARCH>/<food_type>.pt`

- **Dry Run:** Tests with `--dry-run` flag: 1 epoch/iteration, Almond only, full pipeline validated

---

## 1. Core API Contract

### benchmark_food_type Function

Every architecture module must expose a function with this signature:

```python
def benchmark_food_type(food_type: str, 
                        base_dir: str = 'AnomalyonFood/Dataset',
                        dry_run: bool = False, 
                        retrain: bool = True,
                        **kwargs) -> dict:
    """
    Train and evaluate an anomaly detection model on a single food type.
    
    Args:
        food_type (str): One of 'Almond', 'Pistachio', 'GarlicStems'
        base_dir (str): Root path to the dataset directory
        dry_run (bool): If True, run minimal iterations (1 epoch/iteration) to validate script
        retrain (bool): If True, train from scratch. If False, attempt to load saved weights
        **kwargs: Architecture-specific hyperparameters
    
    Returns:
        dict: Result dictionary (see Return Format below)
    """
```

#### Performance Notes
- Function is called once per food type, with `retrain=True` by default
- When `dry_run=True`, models should train for minimal iterations (1 epoch/iteration)
- Function blocks until training/inference completes (no async/background required)

---

## 2. Return Format

Your `benchmark_food_type` function must return a result dictionary matching one of these structures:

```json
{
  "roc_auc": 0.8234,
  "pr_auc": 0.7456,
  "detectmap_shape": [400, 512],
  "infer_time_sec": 12.5,
  "n_params": 1234567,
  "max_vram_gb": 2.1
}
```

#### Required Fields
- `roc_auc` (float): ROC-AUC score [0, 1]
- `pr_auc` (float): PR-AUC score [0, 1]

#### Optional Fields
- `detectmap_shape`: Shape of the anomaly score map (for debugging)
- `infer_time_sec`: Wall-clock time spent on inference (float seconds)
- `n_params`: Total trainable parameter count
- `max_vram_gb`: Peak GPU memory usage in GB

---

## 3. Module Structure

Each architecture must be a Python package under `scripts/<ARCH>/` with:

```
scripts/<ARCH>/
  __init__.py              # Can be empty
  __main__.py              # REQUIRED: contains benchmark_food_type()
  model.py                 # (optional) Your model class(es)
  data.py                  # (optional) Data loading utilities
  utils.py                 # (optional) Helper functions & constants
  __pycache__/             # Auto-generated
```

### __main__.py Requirements

```python
# The module MUST be importable as:
# module = importlib.import_module('scripts.<ARCH>.__main__')
# and expose:
# module.benchmark_food_type(food_type, ...)

from . import data, model, utils  # (optional, if you have these modules)

def benchmark_food_type(food_type: str, **kwargs) -> dict:
    # Your implementation
    return {...}

# Optional: if you want to use the module as a standalone CLI
if __name__ == '__main__':
    result = benchmark_food_type('Almond', dry_run=False, retrain=True)
    print(result)
```

---

## 4. Weight Management

### Saving Trained Weights

Convention: Save weights in `./weights/<ARCH>/` directory.

```python
import os
import torch

WEIGHT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'weights', '<ARCH>')
os.makedirs(WEIGHT_DIR, exist_ok=True)

def save_weights(model, food_type: str):
    """
    Args:
        model: PyTorch model or dict of models
        food_type: 'Almond', 'Pistachio', or 'GarlicStems'
    """
    weight_path = os.path.join(WEIGHT_DIR, f'{food_type}.pt')
    torch.save(model.state_dict(), weight_path)
    print(f"✅ Weights saved to {weight_path}")
```

### Loading Trained Weights

```python
def load_weights(model, food_type: str, device='cpu'):
    """
    Args:
        model: PyTorch model to load into
        food_type: 'Almond', 'Pistachio', or 'GarlicStems'
        device: 'cpu' or 'cuda'
    
    Returns:
        bool: True if weights loaded, False if file not found
    """
    weight_path = os.path.join(WEIGHT_DIR, f'{food_type}.pt')
    
    if os.path.exists(weight_path):
        model.load_state_dict(
            torch.load(weight_path, map_location=device)
        )
        print(f"✅ Weights loaded from {weight_path}")
        return True
    else:
        print(f"⚠️  No weights found at {weight_path}")
        return False

def weights_exist(food_type: str) -> bool:
    """Check if saved weights exist for a food type."""
    weight_path = os.path.join(WEIGHT_DIR, f'{food_type}.pt')
    return os.path.exists(weight_path)
```

---

## 5. Data Partitioning & Spatial Guillotine Protocol

**CRITICAL:** The benchmark enforces strict spatial data partitioning (see `DATA.md`). All models must respect these boundaries to ensure zero data leakage.

### Spatial Map

For all images with shape `(H=400, W=512, C=224)`:

```
┌─────────────────────────────────┐
│ Y[0:50]      : DISCARDED        │  Sensor startup noise
├─────────────────────────────────┤
│ Y[50:200]    : TRAIN BLOCK      │  150 rows – exclusive training zone
├─────────────────────────────────┤
│ Y[200:300]   : DEAD ZONE        │  100 rows – buffer for patch leakage
├─────────────────────────────────┤
│ Y[300:350]   : VAL BLOCK        │  50 rows – unsupervised early stopping
├─────────────────────────────────┤
│ Y[350:400]   : REMAINING        │  50 rows – can be used for final test inference
└─────────────────────────────────┘
```

### Implementation Requirements

1. **Training Pass:**
   - Only compute gradients from pixels in `Y[50:200]`
   - Do NOT backpropagate from validation/test zones
   - Use a spatial mask when masking losses

2. **Validation Pass:**
   - Use `Y[300:350]` for unsupervised early stopping (reconstruction loss, contrastive loss, etc.)
   - Do NOT use ground truth labels during validation

3. **Spatial Mask Examples:**

check [DATA.md](DATA.md)

---

## 6. Data Loading & Calibration

### HSI Data Format

The dataset uses ENVI format (spectroscopic image format):

```
AnomalyonFood/Dataset/
└── <FoodType>/
    ├── Train/
    │   ├── data.hdr              # ENVI metadata
    │   ├── WHITEREF.hdr          # White reference (for calibration)
    │   └── DARKREF.hdr           # Dark reference (for calibration)
    │   └── label.npy             # Binary labels (0=normal, 1=anomaly)
    └── Test/
        ├── data.hdr
        ├── WHITEREF.hdr
        ├── DARKREF.hdr
        └── label.npy
```

### Loading Data

```python
import spectral
import numpy as np

def load_hsi_data(data_path):
    """Load HSI data from ENVI format."""
    img = spectral.open_image(data_path)
    data = img.load()
    return np.array(data, dtype=np.float32)  # Shape: (H, W, C)

def load_label(label_path):
    """Load binary anomaly labels."""
    return np.load(label_path)  # Shape: (H, W)

def load_white_ref(path):
    return spectral.open_image(path).load()

def load_dark_ref(path):
    return spectral.open_image(path).load()
```

### Calibration (White-Dark Correction)

Recommended practice to preserve lighting gradients:

```python
def calibrate_hsi(data, white_ref_path, dark_ref_path):
    """
    Calibrate HSI data while preserving physical spatial lighting gradient.
    
    Args:
        data: (H, W, C) raw HSI cube
        white_ref_path: Path to white reference ENVI file
        dark_ref_path: Path to dark reference ENVI file
    
    Returns:
        calibrated: (H, W, C) normalized to [0, 1]
    """
    white_ref = np.array(spectral.open_image(white_ref_path).load(), dtype=np.float32)
    dark_ref = np.array(spectral.open_image(dark_ref_path).load(), dtype=np.float32)
    
    # Average across time dimension (axis=0) if multi-frame refs
    white_profile = white_ref.mean(axis=0)  # (W, C)
    dark_profile = dark_ref.mean(axis=0)    # (W, C)
    
    # Calibrate with broadcasting to preserve spatial gradients
    calibrated = (data - dark_profile) / (white_profile - dark_profile + 1e-8)
    calibrated = np.clip(calibrated, 0, 1)
    
    return calibrated
```

---

## 5.5. Training Loop Pattern (Critical for Model Convergence)

### The Standard Pattern: While Loop + Early Stopping + Dry Run Break

**ALL models must follow this pattern to ensure proper convergence without artificial epoch caps:**

```python
# REQUIRED PATTERN for all architectures
def train_model(model, train_loader, val_loader, optimizer, device, dry_run=False):
    """
    Train with early stopping - allows models to converge naturally.
    Dry run support ensures quick validation of the training pipeline.
    """
    
    # Initialize early stopper
    early_stopper = UniversalEarlyStopping(patience=50, min_delta=1e-4)
    
    # Use while loop, NOT for loop with fixed epochs
    epoch = 0
    while not early_stopper.early_stop:
        epoch += 1
        
        # Training phase
        model.train()
        for batch in train_loader:
            # Your training code here
            loss.backward()
            optimizer.step()
        
        # Validation phase
        model.eval()
        with torch.inference_mode():
            val_loss = evaluate(model, val_loader)
        
        # Early stopping check
        early_stopper(val_loss)
        if early_stopper.early_stop:
            print(f"✅ Convergence reached at epoch {epoch}. Stopping training.")
        
        # CRITICAL: Support dry run (exits after 1 epoch)
        if dry_run:
            print(f"Dry run mode: completed 1 epoch (full pipeline validated)")
            break
    
    return model, epoch
```

### Why This Pattern?

| Aspect | Old Pattern (for loop) | New Pattern (while loop) |
|--------|------------------------|-------------------------|
| **Convergence** | Hard-capped at N epochs (e.g., `for epoch in range(150)`) | Trains until early stopping triggers (natural stopping point) |
| **Overfitting Risk** | Forced to run full N epochs even after convergence | Stops immediately after validation loss plateaus |
| **Dry Run Support** | Requires parameter changes (`max_epochs=1`) | Uses simple `if dry_run: break` |
| **Code Clarity** | Epoch limit mixed with early stopping logic | Single, clear exit condition |
| **Scalability** | Different food types need different epoch counts | One pattern works for all |

### Migration Guide

If your model currently uses:
```python
# ❌ OLD PATTERN - Do not use
for epoch in range(num_epochs):  # Hard cap
    train()
    val_loss = validate()
    early_stopper(val_loss)
    if early_stopper.early_stop:
        break
```

Convert to:
```python
# ✅ NEW PATTERN - Required
epoch = 0
while not early_stopper.early_stop:  # Soft cap (early stopping only)
    epoch += 1
    train()
    val_loss = validate()
    early_stopper(val_loss)
    if early_stopper.early_stop:
        print(f"Convergence reached at epoch {epoch}. Stopping training.")
    if dry_run:  # Add this for dry run support
        break
```

### Examples in Codebase

Reference implementations:
- [BockNet](scripts/bocknet/__main__.py#L143) – Lines 143-176
- [PA2E](scripts/pa2e/__main__.py#L153) – Lines 153-216
- [Our Model](scripts/our/__main__.py#L175) – Lines 175-240
- [GT-HAD](scripts/gthad/__main__.py#L188) – Lines 188-270

---

## 7. Recommended Utility Implementations

### Early Stopping Monitor

For unsupervised training:

```python
class UniversalEarlyStopping:
    """Monitor unsupervised loss (reconstruction, contrastive, etc.)."""
    
    def __init__(self, patience=50, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"Early stopping triggered (patience {self.patience} reached)")
```

### Random Seed Management

For reproducibility:

```python
def set_seed(seed: int):
    """Set seed for all random sources."""
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
```

---

## 8. Benchmark.py Integration

The main `benchmark.py` script automatically:

1. **Discovers** your module via `scripts.<ARCH>.__main__.benchmark_food_type`
2. **Calls** `benchmark_food_type(food_type, dry_run=..., retrain=...)`
3. **Saves** results to `./results/<ARCH>.json` using JSON serialization
4. **Aggregates** all results in `./results/benchmark_all.json`

### Hyperparameter Passing

Custom hyperparameters can be passed via `extra_kwargs` in `benchmark.py`:

```python
# In benchmark.py:
extra_kwargs = {
    'common': {
        'dry_run': False,
        'retrain': True,
    },
    '<ARCH>': {'param1': value1, 'param2': value2},
}

# In benchmark_food_type:
def benchmark_food_type(food_type, param1=default1, param2=default2, **kwargs):
    ...
```

---

## 9. Checklist for New Architecture

- [ ] Create `scripts/<ARCH>/__main__.py` with `benchmark_food_type(food_type, dry_run, retrain, **kwargs)`
- [ ] Return dictionary with `roc_auc` and `pr_auc` keys
- [ ] Respect spatial boundaries: train on `Y[50:200]`, validate on `Y[300:350]`
- [ ] Add weight saving/loading to `./weights/<ARCH>/<food_type>.pt`
- [ ] **[CRITICAL]** Use while loop + early stopping + if dry_run break pattern (see [Section 5.5](#55-training-loop-pattern-critical-for-model-convergence))
  - ✅ `while not early_stopper.early_stop:` (NOT `for epoch in range(num_epochs)`)
  - ✅ `if dry_run: break` at end of loop
  - ✅ No hard epoch cap – let model converge naturally
- [ ] Implement early stopping using unsupervised loss (no GT labels in validation)
- [ ] Support `dry_run=True` for quick validation (1 epoch/iteration)
- [ ] Support `retrain=False` to load pre-trained weights
- [ ] Calibrate HSI data using white/dark references
- [ ] Test: `python -m scripts.<ARCH>` should work standalone
- [ ] Verify: `python benchmark.py --only <ARCH>` completes successfully
- [ ] Document architecture-specific hyperparameters in module docstring

---

## 10. Example: Minimal Template

```python
# scripts/<ARCH>/__main__.py

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.ndimage import median_filter
import time

from .utils import load_hsi_data, load_label, calibrate_hsi, \
                   get_spatial_train_val_masks, save_weights, load_weights
from .model import AnomalyDetectionModel

def benchmark_food_type(food_type: str,
                        base_dir: str = 'AnomalyonFood/Dataset',
                        dry_run: bool = False,
                        retrain: bool = True,
                        **kwargs) -> dict:
    """Train and evaluate your architecture on a food type."""
    
    print(f"\n{'='*70}\nBENCHMARKING: {food_type}\n{'='*70}")
    
    # Load data
    train_dir = os.path.join(base_dir, food_type, 'Train')
    data = load_hsi_data(os.path.join(train_dir, 'data.hdr'))
    gt = load_label(os.path.join(train_dir, 'label.npy'))
    
    # Calibrate
    data = calibrate_hsi(
        data,
        os.path.join(train_dir, 'WHITEREF.hdr'),
        os.path.join(train_dir, 'DARKREF.hdr')
    )
    
    # Setup device, model, optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AnomalyDetectionModel(num_channels=data.shape[2]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Training
    start_time = time.time()
    train_mask, val_mask = get_spatial_train_val_masks(H=400, W=512)
    
    if not retrain and load_weights(model, food_type, device):
        train_time = 0.0
    else:
        # REQUIRED: Use while loop with early stopping (no hard epoch cap)
        early_stopper = UniversalEarlyStopping(patience=50, min_delta=1e-4)
        epoch = 0
        while not early_stopper.early_stop:
            epoch += 1
            
            # Train on Y[50:200]
            loss = model.train_step(data, train_mask, optimizer)
            
            # Validate on Y[300:350]
            val_loss = model.val_step(data, val_mask)
            
            # Early stopping check
            early_stopper(val_loss)
            if early_stopper.early_stop:
                print(f"✅ Convergence reached at epoch {epoch}. Stopping training.")
            
            # Support dry run mode (exit after 1 epoch)
            if dry_run:
                break
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: loss={loss:.4f}, val_loss={val_loss:.4f}")
        
        train_time = time.time() - start_time
        save_weights(model, food_type)
    
    # Inference
    anomaly_scores = model.predict(data)
    
    # Compute metrics
    roc_auc = roc_auc_score(gt.flatten(), anomaly_scores.flatten())
    pr_auc = average_precision_score(gt.flatten(), anomaly_scores.flatten())
    
    return {
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'detectmap_shape': list(anomaly_scores.shape),
        'infer_time_sec': infer_time,
        'n_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'max_vram_gb': max_vram
    }

if __name__ == '__main__':
    result = benchmark_food_type('Almond', dry_run=False, retrain=True)
    print(json.dumps(result, indent=2, default=str))
```

---

## 11. JSON Serialization Notes

The benchmark.py script uses custom JSON serialization to handle numpy/torch types:

```python
def convert_numpy(obj):
    import numpy as np
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.floating, np.integer)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

json.dump(arch_results, f, indent=2, default=convert_numpy)
```

Ensure all values in your returned dict are JSON-serializable (float, int, str, list, dict).

