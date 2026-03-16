# HSI Anomaly Detection: Unbiased Early Stopping Protocol

Self-supervised Hyperspectral Anomaly Detection (HAD) models are highly susceptible to the **Identity Mapping Problem (IMP)**. If a model has excessive capacity (e.g., massive 3D transformers), it will eventually overfit the background *and* the sparse anomalies, driving its reconstruction error for contaminants to zero and destroying its PR-AUC.

To evaluate architectures fairly, we cannot rely on manual epoch limits, nor can we use PR-AUC/ROC-AUC to trigger early stopping (which constitutes severe data leakage).

All models in this benchmark are governed by a strict, automated **Validation Reconstruction Loss** monitor. The model is allowed to train until it mathematically stops learning the background manifold of the spatially isolated Validation Block.

## 1. Directory Structure Rule

Do **not** place this in a globally shared `utils.py`. To maintain self-contained executable modules for each architecture, drop the early stopping class directly into the architecture's specific utility file:
`scripts/<arch>/utils.py`

## 2. The Early Stopping Class

Add the following `UniversalEarlyStopping` class to your script-level `utils.py`. This class tracks the validation loss and enforces a strict patience window.

```python
import torch
import numpy as np

class UniversalEarlyStopping:
    def __init__(self, patience=50, min_delta=1e-4):
        """
        Monitors validation reconstruction loss to prevent Identity Mapping.
        
        Args:
            patience (int): How many epochs/evals to wait after the last 
                            meaningful improvement in validation loss.
            min_delta (float): The minimum absolute change required to qualify
                               as an improvement. Prevents micro-oscillations 
                               from extending the training loop indefinitely.
        """
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
            print(f"  [EarlyStopping] Patience {self.counter}/{self.patience} (Best: {self.best_loss:.5f})")
            if self.counter >= self.patience:
                self.early_stop = True

```

## 3. Training Loop Integration

In your architecture's main training script (e.g., `scripts/<arch>/__main__.py`), instantiate the stopper before the loop and feed it the validation loss at the end of each epoch.

**Critical Requirements for Integration:**

1. You **must** calculate the loss exclusively on the spatially isolated Validation Block (e.g., `Y[300:350]`).
2. You **must** wrap the validation pass in `torch.inference_mode()` to prevent gradient accumulation and VRAM leaks.

### Integration Example:

```python
import torch
from .utils import UniversalEarlyStopping

# 1. Initialize the referee
# Use 50 patience to give heavy networks runway to escape saddle points
early_stopper = UniversalEarlyStopping(patience=50, min_delta=1e-4)

# 2. Inside the training loop
for epoch in range(MAX_EPOCHS):
    
    # --- TRAINING PASS ---
    model.train()
    # ... execute training step on Train Block (Y[50:200]) ...
    
    # --- VALIDATION PASS ---
    model.eval()
    val_loss_total = 0.0
    
    with torch.inference_mode():
        for val_batch in val_dataloader:
            # val_dataloader MUST only yield data from Validation Block (Y[300:350])
            inputs = val_batch.to(device)
            outputs = model(inputs)
            
            # Calculate standard reconstruction loss (e.g., MSE)
            loss = criterion(outputs, inputs)
            val_loss_total += loss.item()
            
    avg_val_loss = val_loss_total / len(val_dataloader)
    
    # 3. Check for Early Stopping
    early_stopper(avg_val_loss)
    
    if early_stopper.early_stop:
        print(f"Convergence reached at epoch {epoch}. Terminating training.")
        break

```

### The Expected Behavior

By strictly adhering to this protocol:

* **Efficient, regularized models (like 1D CNNs):** Will learn the general background rules quickly, flatline on the validation set, and trigger early stopping while preserving a high anomaly detection capability (PR-AUC).
* **Bloated, over-parameterized models:** Will aggressively overfit the Train Block. Once they memorize the training anomalies, their validation loss will stagnate, triggering early stopping. Their final PR-AUC will expose their susceptibility to the Identity Mapping Problem.

---