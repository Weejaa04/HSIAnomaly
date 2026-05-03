"""
SGLNet Benchmark - Spectral-Graph Learning Network for HSI Anomaly Detection

Architecture: Patch-based autoencoder with spectral, local, and graph branches
Training: Reconstruction loss with early stopping on validation set
Inference: Reconstruction error as anomaly score

Usage:
    from scripts.sglnet.__main__ import benchmark_food_type
    result = benchmark_food_type('Almond', dry_run=False, retrain=True)
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime
from scipy.ndimage import median_filter
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    auc,
)
from tqdm import tqdm
import random
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scripts.sglnet.utils import (
    UniversalEarlyStopping,
    load_hsi_data,
    load_label,
    calibrate_hsi,
    get_spatial_train_val_mask,
    save_weights,
    load_weights,
    weights_exist,
)
from scripts.sglnet.model import SGLNetAE
from scripts.sglnet.data import HSIPatchDataset

SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=True)


set_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"Using device: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU")


def train_model(
    model, train_loader, val_loader, optimizer, device, dry_run=False, food_type=None
):
    """
    Train SGLNet autoencoder with reconstruction loss.

    Uses while loop + early stopping pattern per BENCH.md requirements.
    """
    print("\n=== Training SGLNet Autoencoder ===")

    early_stopper = UniversalEarlyStopping(patience=50, min_delta=1e-4)

    epoch = 0
    while not early_stopper.early_stop:
        epoch += 1

        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            x = batch if isinstance(batch, torch.Tensor) else batch[0]
            x = x.to(device)

            optimizer.zero_grad()
            reconstruction = model(x)
            loss = F.mse_loss(reconstruction, x)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.inference_mode():
            for batch in val_loader:
                x = batch if isinstance(batch, torch.Tensor) else batch[0]
                x = x.to(device)
                reconstruction = model(x)
                loss = F.mse_loss(reconstruction, x)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(
            f"Epoch {epoch}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}"
        )

        early_stopper(avg_val_loss)
        if early_stopper.early_stop:
            print(f"✅ Convergence reached at epoch {epoch}. Stopping training.")

        if dry_run:
            print("Dry run mode: completed 1 epoch (full pipeline validated)")
            break

    if food_type:
        save_weights(model, food_type)

    return epoch


def inference(model, test_data, device, patch_size=9):
    """Run inference on test data and compute anomaly scores."""
    print("\n--- Inference ---")
    start = datetime.now()

    model.eval()
    H, W, B = test_data.shape
    pad = patch_size // 2

    test_padded = np.pad(test_data, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")

    scores = np.zeros(H * W)

    batch_size = 512
    batch_patches = []
    batch_indices = []

    with torch.inference_mode():
        for row in range(H):
            for col in range(W):
                patch = test_padded[row : row + patch_size, col : col + patch_size, :]
                patch_tensor = torch.from_numpy(patch).float()
                patch_tensor = patch_tensor.permute(2, 0, 1).unsqueeze(0).to(device)

                batch_patches.append(patch_tensor)
                batch_indices.append(row * W + col)

                if len(batch_patches) >= batch_size:
                    patches = torch.cat(batch_patches, dim=0)
                    reconstructions = model(patches)
                    errors = torch.mean((reconstructions - patches) ** 2, dim=(1, 2, 3))

                    for idx, err in zip(batch_indices, errors.cpu().numpy()):
                        scores[idx] = err

                    batch_patches = []
                    batch_indices = []

        if batch_patches:
            patches = torch.cat(batch_patches, dim=0)
            reconstructions = model(patches)
            errors = torch.mean((reconstructions - patches) ** 2, dim=(1, 2, 3))

            for idx, err in zip(batch_indices, errors.cpu().numpy()):
                scores[idx] = err

    score_map = scores.reshape(H, W)
    smoothed = median_filter(score_map, size=3).flatten()

    end = datetime.now()
    elapsed = (end - start).total_seconds()

    print(f"⚡ Inference time: {elapsed:.2f}s")

    return smoothed, elapsed, (H, W)


def evaluate(scores, labels):
    """Compute ROC-AUC and PR-AUC."""
    roc_auc = roc_auc_score(labels, scores)
    precision, recall, _ = precision_recall_curve(labels, scores)
    pr_auc = auc(recall, precision)

    return roc_auc, pr_auc


def benchmark_food_type(
    food_type: str,
    base_dir: str = "AnomalyonFood/Dataset",
    dry_run: bool = False,
    retrain: bool = True,
    **kwargs,
) -> dict:
    """
    Run SGLNet benchmark on a specific food type.

    Args:
        food_type: 'Almond', 'Pistachio', or 'GarlicStems'
        base_dir: Root path to the dataset directory
        dry_run: Run for 1 epoch only (optional)
        retrain: Whether to retrain or load saved model (optional)

    Returns:
        dict with results including roc_auc, pr_auc, etc.
    """
    print(f"\n{'=' * 80}")
    print(f"SGLNet Benchmark: {food_type}")
    print(f"{'=' * 80}")

    total_start_time = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    base_path = os.path.join(os.path.dirname(__file__), "..", "..", base_dir, food_type)

    train_data_path = os.path.join(base_path, "Train", "data.hdr")
    train_white_path = os.path.join(base_path, "Train", "WHITEREF.hdr")
    train_dark_path = os.path.join(base_path, "Train", "DARKREF.hdr")

    print(f"Loading {food_type} training data...")
    train_data = load_hsi_data(train_data_path)
    train_data = calibrate_hsi(train_data, train_white_path, train_dark_path)

    H, W, B = train_data.shape

    train_mask, val_mask = get_spatial_train_val_mask(H, W)

    test_data_path = os.path.join(base_path, "Test", "data.hdr")
    test_white_path = os.path.join(base_path, "Test", "WHITEREF.hdr")
    test_dark_path = os.path.join(base_path, "Test", "DARKREF.hdr")
    test_label_path = os.path.join(base_path, "Test", "label.npy")

    print(f"Loading {food_type} test data...")
    test_data = load_hsi_data(test_data_path)
    test_data = calibrate_hsi(test_data, test_white_path, test_dark_path)
    test_labels = load_label(test_label_path)

    test_labels_flat = test_labels.reshape(-1)
    binary_labels = (test_labels_flat != 2).astype(int)

    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    print(f"Number of bands: {B}")
    print(f"Train pixels: {train_mask.sum()}")
    print(f"Val pixels: {val_mask.sum()}")

    patch_size = 9
    batch_size = 128

    train_dataset = HSIPatchDataset(train_data, mask=train_mask, patch_size=patch_size)
    val_dataset = HSIPatchDataset(train_data, mask=val_mask, patch_size=patch_size)
    test_dataset = HSIPatchDataset(
        test_data, labels=test_labels_flat, patch_size=patch_size
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    model = SGLNetAE(in_ch=B, head_ch=48, local_blc=2, sq=10, mb=64).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")

    weights_loaded = False
    train_time_sec = 0.0

    if not retrain and weights_exist(food_type):
        print(f"\n[LoadWeights] Trained weights found. Loading {food_type} model...")
        load_weights(model, food_type, device=device)
        weights_loaded = True

    if not weights_loaded:
        training_start = time.time()

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        train_model(
            model,
            train_loader,
            val_loader,
            optimizer,
            device,
            dry_run=dry_run,
            food_type=food_type,
        )

        train_time_sec = time.time() - training_start

    scores, infer_time, shape = inference(
        model, test_data, device, patch_size=patch_size
    )

    roc_auc, pr_auc = evaluate(scores, binary_labels)

    max_vram_gb = 0.0
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated(device)
        max_vram_gb = peak_memory / (1024**3)

    total_elapsed = time.time() - total_start_time

    print(f"\n{'=' * 80}")
    print(f"Results: ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f}")
    print(
        f"Training time: {train_time_sec:.2f}s | Inference time: {infer_time:.2f}s | Total time: {total_elapsed:.2f}s"
    )
    if max_vram_gb > 0:
        print(f"Peak VRAM: {max_vram_gb:.2f} GB")
    print(f"{'=' * 80}")

    return {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "detectmap_shape": [int(shape[0]), int(shape[1])],
        "infer_time_sec": float(infer_time),
        "n_params": int(total_params),
        "max_vram_gb": float(max_vram_gb),
    }


if __name__ == "__main__":
    result = benchmark_food_type("Almond", dry_run=False, retrain=True)
    print(json.dumps(result, indent=2, default=str))
