"""
SuperAD: Self-supervised Hyperspectral Anomaly Detection
Reference: TGRS2025 - Overcoming the Identity Mapping Problem in Self-Supervised Hyperspectral Anomaly Detection

Usage:
    from scripts.superad.__main__ import benchmark_food_type
    result = benchmark_food_type('Almond', dry_run=False, retrain=True)
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import numpy as np
import time
import random

from .model import SuperADNetwork, OBPM
from .data import get_food_data
from .utils import (
    set_seed,
    get_auc,
    TensorToHSI,
    UniversalEarlyStopping,
    save_weights,
    load_weights,
    weights_exist,
)

SEED = 42


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


set_all_seeds(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"Using device: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU")


def benchmark_food_type(
    food_type: str,
    base_dir: str = "AnomalyonFood/Dataset",
    dry_run: bool = False,
    retrain: bool = True,
    kernel_size: int = 3,
    window_size: int = 5,
    lr: float = 1e-3,
    n_segments: int = 500,
    compactness: int = 10,
    alpha: float = 1.0,
    beta: float = 1.0,
    th_idx: float = 0.25,
    loss_type: str = "OBPM",
    **kwargs,
) -> dict:
    """
    Train and evaluate SuperAD on a single food type.

    Args:
        food_type: One of 'Almond', 'Pistachio', 'GarlicStems'
        base_dir: Root path to the dataset directory
        dry_run: If True, run minimal iterations to validate script
        retrain: If True, train from scratch. If False, load saved weights
        kernel_size: Size of the convolutional kernel
        window_size: Size of the sliding window
        lr: Learning rate
        n_segments: Number of superpixel segments
        compactness: Compactness parameter for SLIC
        alpha: OBPM loss alpha parameter
        beta: OBPM loss beta parameter
        th_idx: OBPM loss threshold index
        loss_type: Loss type ('OBPM', 'l1', 'l2')

    Returns:
        dict: Result dictionary with roc_auc, pr_auc, etc.
    """
    print(f"\n{'=' * 70}")
    print(f"SUPERAD BENCHMARK: {food_type}")
    print(f"{'=' * 70}")

    total_start_time = time.time()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    base_path = os.path.join(base_dir, food_type)

    print(f"Loading {food_type} data...")
    train_data, test_data, test_labels, segments, H, W, B, train_mask, val_mask = (
        get_food_data(
            food_type, base_dir, n_segments=n_segments, compactness=compactness
        )
    )

    train_data = train_data.to(device)
    test_data = test_data.to(device)
    test_labels = test_labels.to(device)
    segments = segments.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)

    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    print(f"Number of bands: {B}")
    print(f"Superpixel segments: {n_segments}")

    model = SuperADNetwork(
        nch_in=B, nch_out=B, kernel_size=kernel_size, window_size=window_size
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.5, 0.8), weight_decay=1e-4
    )
    scheduler = StepLR(optimizer, step_size=100, gamma=0.8)

    last_loss = None
    weights_loaded = False

    if not retrain and weights_exist(food_type):
        print(f"\n[LoadWeights] Loading trained weights...")
        if load_weights(model, food_type, device):
            weights_loaded = True
            train_time = 0.0

    if not weights_loaded:
        print(f"\n=== Training SuperAD ===")

        if dry_run:
            patience = 1
            min_delta = 1e-8
        else:
            patience = 50
            min_delta = 1e-4

        early_stopper = UniversalEarlyStopping(patience=patience, min_delta=min_delta)

        training_start = time.time()
        epoch = 0

        while not early_stopper.early_stop:
            epoch += 1

            model.train()
            optimizer.zero_grad()

            pred = model(train_data, segments, last_loss)
            loss = F.l1_loss(pred, train_data, reduction="none")
            loss_masked = (loss * train_mask).mean()

            total_loss, plt_loss = OBPM(
                loss.mean(dim=1, keepdim=True),
                segments,
                beta=beta,
                alpha=alpha,
                th_idx=th_idx,
                loss_type=loss_type,
            )

            total_loss.backward()
            optimizer.step()
            scheduler.step()

            last_loss = loss.detach().mean(dim=1, keepdim=True)

            model.eval()
            with torch.inference_mode():
                val_pred = model(train_data, segments, last_loss)
                val_loss = F.l1_loss(val_pred, train_data, reduction="none")
                val_loss_masked = (val_loss * val_mask).mean()

            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch}: train_loss={loss_masked.item():.6f}, val_loss={val_loss_masked.item():.6f}"
                )

            early_stopper(val_loss_masked.item())

            if early_stopper.early_stop:
                print(f"\nConvergence reached at epoch {epoch}. Stopping training.")

            if dry_run:
                break

        train_time = time.time() - training_start
        print(f"\nTraining time: {train_time:.2f}s")
        print(f"Epochs trained: {epoch}")

        save_weights(model, food_type)

    print("\n=== Inference ===")
    model.eval()

    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    infer_start = time.time()

    with torch.inference_mode():
        test_pred = model(test_data, segments, last_loss)

    if torch.cuda.is_available():
        torch.cuda.synchronize(device)

    infer_time = time.time() - infer_start

    HSI_old = TensorToHSI(test_data)
    HSI_new = TensorToHSI(test_pred)
    gt = test_labels.cpu().numpy()

    roc_auc, detectmap = get_auc(HSI_old, HSI_new, gt)

    from sklearn.metrics import average_precision_score

    test_labels_flat = gt.reshape(-1)
    binary_labels = (test_labels_flat != 2).astype(int)
    pr_auc = average_precision_score(binary_labels, detectmap.flatten())

    peak_vram_mib = (
        torch.cuda.max_memory_allocated(device) / 1024**2
        if torch.cuda.is_available()
        else 0.0
    )
    max_vram_gb = peak_vram_mib / 1024

    total_elapsed = time.time() - total_start_time

    print(f"\n{'=' * 70}")
    print(f"Results: ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f}")
    print(
        f"Training time: {train_time:.2f}s | Inference time: {infer_time:.2f}s | Total time: {total_elapsed:.2f}s"
    )
    if max_vram_gb > 0:
        print(f"Peak VRAM: {max_vram_gb:.2f} GB")
    print(f"{'=' * 70}")

    return {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "detectmap_shape": list(detectmap.shape),
        "infer_time_sec": float(infer_time),
        "n_params": int(total_params),
        "max_vram_gb": float(max_vram_gb),
    }


if __name__ == "__main__":
    result = benchmark_food_type("Almond", dry_run=False, retrain=True)
    print(json.dumps(result, indent=2, default=str))
