# HSI Food Anomaly Detection: Universal Data Partitioning Protocol

This document outlines the strict spatial data partitioning protocol used for benchmarking Hyperspectral Imaging (HSI) anomaly detection models.

Because we are evaluating architectures with vastly different receptive fields (from $1 \times 1$ spectral pixels to massive 3D spatial patches and global transformers) on continuous pushbroom scans, standard random train/test splits are invalid. A random pixel split causes catastrophic spatial data leakage for patch-based models, as validation pixels will physically fall inside their training receptive fields.

To ensure a mathematically fair, zero-leakage benchmark, all models are subjected to a unified **Horizontal Spatial Guillotine** protocol.

## 1. The Universal Spatial Map

Our HSI scans (e.g., conveyor belt data) have highly consistent background manifolds along the Y-axis (height). We partition the image horizontally. Assuming an image shape of `(H=400, W=512, C=224)`:

* **Train Block (`Y[50:200]`):** The exclusive zone for weight updates. No model is allowed to backpropagate gradients from pixels outside this block. The `0:50` block is discarded to avoid sensor startup noise.
* **The Dead Zone (`Y[200:300]`):** A massive 100-pixel margin. This guarantees that even the largest 3D patch-based models cannot accidentally ingest validation pixels during training.
* **Validation Block (`Y[300:350]`):** Strictly reserved for the unsupervised Early Stopping monitor (reconstruction loss).
* **Test Inference:** Depending on the dataset, final PR-AUC/ROC-AUC is calculated either on a completely standalone HSI scan or the remaining unseen coordinates of the image.

---

## 2. Implementation Guide: Adding a New Model

If you are integrating a new model into this benchmark, you must adapt its PyTorch `Dataset` or training loop to respect the spatial boundaries based on its architectural paradigm.

### Paradigm A: 1D Pixel-Based Models

*Models that operate purely on the $1 \times 1 \times C$ spectral vector (e.g., standard MLPs, 1D CNNs).*

These models are highly efficient and don't need contiguous spatial blocks. However, to maintain fairness, you must restrict their random sampling to the defined spatial zones.

```python
import numpy as np

def get_1d_pixel_coords(mode='train'):
    if mode == 'train':
        valid_y = np.arange(50, 200)
    elif mode == 'val':
        valid_y = np.arange(300, 350)
    else:
        raise ValueError("Invalid mode")
        
    valid_x = np.arange(0, 512)
    yy, xx = np.meshgrid(valid_y, valid_x, indexing='ij')
    coords = np.column_stack((yy.ravel(), xx.ravel()))
    
    return coords

```

### Paradigm B: 3D Patch-Based Models

*Models that extract spatial-spectral cubes (e.g., $15 \times 15 \times C$, $27 \times 27 \times C$) using 3D convolutions or local window attention.*

You must restrict the center pixel of the patch so that the outer edges of the patch never cross the spatial boundaries.

```python
import numpy as np

def get_3d_patch_centers(mode='train', patch_size=27):
    half_p = patch_size // 2
    
    if mode == 'train':
        # Ensure patch doesn't cross Y=50 or Y=200
        valid_y = np.arange(50 + half_p, 200 - half_p)
    elif mode == 'val':
        # Ensure patch doesn't cross Y=300 or Y=350
        valid_y = np.arange(300 + half_p, 350 - half_p)
        
    valid_x = np.arange(0 + half_p, 512 - half_p)
    
    # Yield these coordinates to your DataLoader to extract the patches
    yy, xx = np.meshgrid(valid_y, valid_x, indexing='ij')
    return np.column_stack((yy.ravel(), xx.ravel()))

```

### Paradigm C: Global Full-Image Models

*Models that ingest the entire $H \times W \times C$ tensor in a single forward pass (e.g., global Vision Transformers, large U-Nets).*

You cannot crop the image without destroying their global receptive field. Instead, you feed them the entire image but apply a binary spatial mask to the loss function. This allows them to see global context but prevents them from updating weights based on the validation/test areas.

```python
import torch

# 1. Generate masks once before the training loop
train_mask = torch.zeros((400, 512), device='cuda')
train_mask[50:200, :] = 1.0

val_mask = torch.zeros((400, 512), device='cuda')
val_mask[300:350, :] = 1.0

# 2. Inside the training loop
outputs = global_model(full_image) 
raw_loss = criterion(outputs, full_image) # e.g., MSE loss per pixel

# 3. Apply the Spatial Information Bottleneck
masked_train_loss = (raw_loss * train_mask).sum() / train_mask.sum()
masked_train_loss.backward()

```

---

## 3. The Early Stopping Trap

All models in this benchmark must be trained using the `UniversalEarlyStopping` monitor tracking the **Validation Block Reconstruction Loss**.

Because unsupervised anomaly detection models are prone to the Identity Mapping Problem (overfitting the background *and* the anomalies), we do not use an arbitrary epoch limit. Models are allowed to train until their validation reconstruction loss fails to improve by `1e-4` for 50 epochs/iterations.

Architectures that lack proper inductive biases or regularization will inevitably overfit the Train Block, fail to improve on the Validation Block, and trigger early stopping—leaving them vulnerable to catastrophic PR-AUC collapse during final inference.

---