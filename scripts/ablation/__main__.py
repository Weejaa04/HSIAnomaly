"""
Ablation benchmark for the 'our' architecture.

Runs all eight ablation axes and returns results per variant per food type.

Usage:
    from scripts.ablation.__main__ import benchmark_food_type
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
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc
from tqdm import tqdm
import random
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from scripts.ablation.utils import (
    UniversalEarlyStopping,
    load_hsi_data, load_label,
    calibrate_hsi_mean_per_column, calibrate_hsi_mean, calibrate_hsi_none,
    get_spatial_train_val_mask,
    save_weights, load_weights, weights_exist,
    CALIBRATION_FNS,
)
from scripts.ablation.model import (
    PA2EPartial, PA2EPartialFT, build_model
)
from scripts.ablation.data import HSIPixelDataset

# ─────────────────────────────────────────────────────────────────────────────
# SEED & DEVICE
# ─────────────────────────────────────────────────────────────────────────────
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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f'Using device: {torch.cuda.get_device_name(0)}')
else:
    print('Using CPU')


# ─────────────────────────────────────────────────────────────────────────────
# Ablation Variant Definitions
# ─────────────────────────────────────────────────────────────────────────────

# Each variant is a dict describing one axis change vs baseline.
# Baseline = partial=True, encoder='cnn', loss='mse', scoring='nnmb', calib='mean_per_column'

ABLATION_AXES = {
    # 1. HSI Calibration
    "calib_mean":            {"calib": "mean",            "partial": True,  "encoder": "cnn",    "loss": "mse", "scoring": "nnmb"},
    "calib_mean_per_column": {"calib": "mean_per_column", "partial": True,  "encoder": "cnn",    "loss": "mse", "scoring": "nnmb"},  # BASELINE
    "calib_none":            {"calib": "none",            "partial": True,  "encoder": "cnn",    "loss": "mse", "scoring": "nnmb"},
    # 2. Encoder type
    "encoder_cnn":           {"calib": "mean_per_column", "partial": True,  "encoder": "cnn",    "loss": "mse", "scoring": "nnmb"},  # BASELINE (same as calib_mean_per_column)
    "encoder_linear":        {"calib": "mean_per_column", "partial": True,  "encoder": "linear", "loss": "mse", "scoring": "nnmb"},
    # 3. Loss function
    "loss_mse":              {"calib": "mean_per_column", "partial": True,  "encoder": "cnn",    "loss": "mse", "scoring": "nnmb"},  # BASELINE
    "loss_mae":              {"calib": "mean_per_column", "partial": True,  "encoder": "cnn",    "loss": "mae", "scoring": "nnmb"},
    # 4. Scoring method
    "scoring_nnmb":          {"calib": "mean_per_column", "partial": True,  "encoder": "cnn",    "loss": "mse", "scoring": "nnmb"},  # BASELINE
    "scoring_dsvdd":         {"calib": "mean_per_column", "partial": True,  "encoder": "cnn",    "loss": "mse", "scoring": "dsvdd"},
    # 5. Partial vs Non-Partial
    "partial_yes":           {"calib": "mean_per_column", "partial": True,  "encoder": "cnn",    "loss": "mse", "scoring": "nnmb"},  # BASELINE
    "partial_no_cnn":        {"calib": "mean_per_column", "partial": False, "encoder": "cnn",    "loss": "mse", "scoring": "nnmb"},
    "partial_no_linear":     {"calib": "mean_per_column", "partial": False, "encoder": "linear", "loss": "mse", "scoring": "nnmb"},
}

# ── 6. Loss Weights (alpha × beta) ─────────────────────────────────────────────
# Phase 2 loss = alpha * center_loss + beta * recon_loss.
# Default (best from grid) is alpha=0.2, beta=1.0.

WEIGHT_GRID = [0.2, 0.4, 0.6, 0.8, 1.0]

def _build_weight_variants():
    variants = {}
    for alpha in WEIGHT_GRID:
        for beta in WEIGHT_GRID:
            variants[f"weight_a{alpha:.1f}_b{beta:.1f}"] = {
                "calib": "mean_per_column", "partial": True, "encoder": "cnn",
                "loss": "mse", "scoring": "nnmb",
                "alpha": alpha, "beta": beta,
            }
    return variants

ABLATION_AXES.update(_build_weight_variants())

# ── 7. Post-processing (Median kernel) ────────────────────────────────────────
# Median filter applied to the raw score map before evaluation.
# Baseline default is 3x3 (median_filter(size=3)); kernel=0 disables smoothing.

PP_AXES = {
    "pp_zero": {"calib": "mean_per_column", "partial": True, "encoder": "cnn",
                "loss": "mse", "scoring": "nnmb", "median_kernel": 0},
    "pp_3":    {"calib": "mean_per_column", "partial": True, "encoder": "cnn",
                "loss": "mse", "scoring": "nnmb", "median_kernel": 3},  # BASELINE
    "pp_5":    {"calib": "mean_per_column", "partial": True, "encoder": "cnn",
                "loss": "mse", "scoring": "nnmb", "median_kernel": 5},
    "pp_7":    {"calib": "mean_per_column", "partial": True, "encoder": "cnn",
                "loss": "mse", "scoring": "nnmb", "median_kernel": 7},
}
ABLATION_AXES.update(PP_AXES)

# ── 8. Training Phases (Phase 1 pre-training) ─────────────────────────────────
# Baseline default trains Phase 1 (reconstruction + OneCycleLR) then Phase 2
# (DSVDD-style). The ablation removes Phase 1 entirely (Phase 2 only, from
# random init).

PHASE_AXES = {
    "phase_both":  {"calib": "mean_per_column", "partial": True, "encoder": "cnn",
                    "loss": "mse", "scoring": "nnmb", "phase1": True, "phase2":True, "center": True},   # BASELINE
    "phase_no_p1": {"calib": "mean_per_column", "partial": True, "encoder": "cnn",
                    "loss": "mse", "scoring": "nnmb", "phase1": False, "phase2":True, "center": True },
    "phase_p2_no_center": {"calib": "mean_per_column", "partial": True, "encoder": "cnn",
                "loss": "mse", "scoring": "nnmb", "phase1": True, "phase2":True, "center": False},
    "phase_no_p2": {"calib": "mean_per_column", "partial": True, "encoder": "cnn",
                    "loss": "mse", "scoring": "nnmb", "phase1": True, "phase2":False, "center": False},

    

    
}
ABLATION_AXES.update(PHASE_AXES)

# Deduplicated set of actually distinct configs (avoids re-training the baseline 5 times)
# Key = canonical config tuple; value = list of variant names sharing that config
def _config_key(cfg):
    return (cfg["calib"], cfg["partial"], cfg["encoder"], cfg["loss"], cfg["scoring"], cfg["center"],
            cfg.get("alpha", 0.2), cfg.get("beta", 1.0), cfg.get("median_kernel", 3),
            cfg.get("phase1", True), cfg.get("phase2", True))


# ─────────────────────────────────────────────────────────────────────────────
# Loss helper
# ─────────────────────────────────────────────────────────────────────────────

def recon_loss_fn(predicted, target, loss_type):
    if loss_type == "mse":
        return F.mse_loss(predicted, target)
    elif loss_type == "mae":
        return F.l1_loss(predicted, target)
    else:
        raise ValueError(f"Unknown loss: {loss_type!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def _get_windows_or_full(model, x):
    """Return (z, recon, target) regardless of partial/non-partial model."""
    if isinstance(model, (PA2EPartial, PA2EPartialFT)):
        target = model.window_extractor(x)
        z, recon = model(x)
        return z, recon, target
    else:
        # Non-partial: forward returns (z, recon_full)
        z, recon = model(x)
        return z, recon, x   # target = x itself


def train_phase1(model, train_loader, num_epochs, lr, loss_type):
    print('\n=== Phase 1: Pre-training (Autoencoder + OneCycleLR) ===')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3, anneal_strategy='cos',
        div_factor=25.0, final_div_factor=1e4,
    )

    losses = []
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            x = batch.to(device)
            optimizer.zero_grad()
            _, recon, target = _get_windows_or_full(model, x)
            loss = recon_loss_fn(recon, target, loss_type)
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.2e}')
    return losses


def initialize_center(model, train_loader):
    print('\nInitializing center...')
    model.eval()
    latents = []
    with torch.no_grad():
        for batch in tqdm(train_loader, desc='Computing center'):
            x = batch.to(device)
            z, _ = model.encode(x)
            latents.append(z)
    latents = torch.cat(latents, dim=0)
    center = latents.mean(dim=0)
    model.center = center
    model.center_initialized = True
    print(f'Center shape: {center.shape}, norm: {center.norm().item():.4f}')
    return center


def train_phase2(model, train_loader, val_loader, num_epochs, lr, alpha, beta, loss_type, food_type, variant_name, dry_run):
    print('\n=== Phase 2: Main Training (DSVDD-style + Early Stopping) ===')
    initialize_center(model, train_loader)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr * 1e-2)
    early_stopper = UniversalEarlyStopping(patience=50, min_delta=1e-4)

    losses, center_losses, recon_losses, val_losses = [], [], [], []
    epoch = 0
    while not early_stopper.early_stop:
        epoch += 1
        model.train()
        epoch_loss = epoch_center = epoch_recon = 0

        for batch in tqdm(train_loader, desc=f'Epoch {epoch}'):
            x = batch.to(device)
            optimizer.zero_grad()

            z, recon, target = _get_windows_or_full(model, x)
            r_loss = recon_loss_fn(recon, target, loss_type)
            c_loss = torch.mean(torch.sum((z - model.center) ** 2, dim=1))
            
            #print ("no center loss: alpha", alpha)
            loss = alpha * c_loss + beta * r_loss

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_center += c_loss.item()
            epoch_recon += r_loss.item()

        # Validation
        model.eval()
        val_total = 0
        with torch.inference_mode():
            for val_batch in val_loader:
                x = val_batch.to(device)
                _, recon, target = _get_windows_or_full(model, x)
                val_total += recon_loss_fn(recon, target, loss_type).item()

        avg_val = val_total / len(val_loader)
        scheduler.step()

        avg_loss = epoch_loss / len(train_loader)
        avg_center = epoch_center / len(train_loader)
        avg_recon = epoch_recon / len(train_loader)

        losses.append(avg_loss)
        center_losses.append(avg_center)
        recon_losses.append(avg_recon)
        val_losses.append(avg_val)

        print(f'Epoch {epoch}, Loss: {avg_loss:.6f}, Center: {avg_center:.6f}, '
              f'Recon: {avg_recon:.6f}, Val: {avg_val:.6f}, LR: {scheduler.get_last_lr()[0]:.2e}')

        early_stopper(avg_val)
        if early_stopper.early_stop:
            print(f"\n✅ Convergence at epoch {epoch}. Stopping.")

        if dry_run:
            break

    save_weights(model, food_type, variant_name)
    return losses, center_losses, recon_losses, val_losses, epoch


# ─────────────────────────────────────────────────────────────────────────────
# Inference / Scoring
# ─────────────────────────────────────────────────────────────────────────────

def build_memory_bank(model, loader):
    model.eval()
    bank = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Building Memory Bank"):
            x = batch.to(device)
            z, _ = model.encode(x)
            bank.append(z)
    return torch.cat(bank, dim=0)


def inference_nnmb(model, test_loader, memory_bank, H, W, kernel=3):
    """Nearest-Neighbour Memory Bank scoring (k-NN, k=1)."""
    print("\n--- NNMB Inference ---")
    start = datetime.now()
    model.eval()
    all_scores = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="k-NN distances"):
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device)
            z, _ = model.encode(x)
            distances = torch.cdist(z, memory_bank, p=2.0)
            min_dist, _ = torch.min(distances, dim=1)
            all_scores.append(min_dist.cpu())

    scores = torch.cat(all_scores).numpy()
    score_map = scores.reshape(H, W)
    if kernel and kernel > 1:
        smoothed = median_filter(score_map, size=kernel).flatten()
    else:
        smoothed = score_map.flatten()
    elapsed = (datetime.now() - start).total_seconds()
    print(f"⚡ NNMB inference: {elapsed:.2f}s (median kernel={kernel})")
    return smoothed, elapsed


def inference_dsvdd(model, test_loader, H, W, kernel=3):
    """DSVDD scoring: distance to center in latent space."""
    print("\n--- DSVDD Inference ---")
    start = datetime.now()
    model.eval()
    all_scores = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="DSVDD distances"):
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device)
            z, _ = model.encode(x)
            dist = torch.sum((z - model.center) ** 2, dim=1)
            all_scores.append(dist.cpu())

    scores = torch.cat(all_scores).numpy()
    score_map = scores.reshape(H, W)
    smoothed = median_filter(score_map, size=3).flatten()
    elapsed = (datetime.now() - start).total_seconds()
    print(f"⚡ DSVDD inference: {elapsed:.2f}s (median kernel={kernel})")
    return smoothed, elapsed


def evaluate(scores, labels):
    roc_auc = roc_auc_score(labels, scores)
    precision, recall, _ = precision_recall_curve(labels, scores)
    pr_auc = auc(recall, precision)
    return roc_auc, pr_auc


# ─────────────────────────────────────────────────────────────────────────────
# Single-variant runner
# ─────────────────────────────────────────────────────────────────────────────

def run_variant(variant_name, cfg, food_type, base_path, dry_run, retrain):
    """
    Train and evaluate a single ablation variant.

    Returns a dict compatible with benchmark_all.json schema.
    """
    print(f"\n{'─'*80}")
    print(f"  Variant: {variant_name}")
    print(f"  Config : {cfg}")
    print(f"{'─'*80}")

    set_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    total_start = time.time()

    # ── Load Data ─────────────────────────────────────────────────────────────
    calib_fn = CALIBRATION_FNS[cfg["calib"]]

    train_data_path  = os.path.join(base_path, 'Train', 'data.hdr')
    train_white_path = os.path.join(base_path, 'Train', 'WHITEREF.hdr')
    train_dark_path  = os.path.join(base_path, 'Train', 'DARKREF.hdr')
    test_data_path   = os.path.join(base_path, 'Test', 'data.hdr')
    test_white_path  = os.path.join(base_path, 'Test', 'WHITEREF.hdr')
    test_dark_path   = os.path.join(base_path, 'Test', 'DARKREF.hdr')
    test_label_path  = os.path.join(base_path, 'Test', 'label.npy')

    print(f'Loading {food_type} data (calib={cfg["calib"]})...')
    train_data = calib_fn(load_hsi_data(train_data_path), train_white_path, train_dark_path)
    test_data  = calib_fn(load_hsi_data(test_data_path),  test_white_path,  test_dark_path)
    test_labels = load_label(test_label_path)

    H, W, B = train_data.shape
    test_h, test_w = test_data.shape[:2]

    train_mask, val_mask = get_spatial_train_val_mask(H, W)
    test_labels_flat = test_labels.reshape(-1)
    binary_labels = (test_labels_flat != 2).astype(int)

    batch_size = 512
    train_dataset = HSIPixelDataset(train_data, mask=train_mask)
    val_dataset   = HSIPixelDataset(train_data, mask=val_mask)
    test_dataset  = HSIPixelDataset(test_data, test_labels_flat)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=8)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=8)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=8)

    # ── Build Model ────────────────────────────────────────────────────────────
    model = build_model(
        input_dim=B,
        partial=cfg["partial"],
        encoder_type=cfg["encoder"],
        window_size=56,
        stride=14,
        latent_local_dim=28,
        latent_global_dim=32,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters: {total_params:,}')

    # ── Load or Train ──────────────────────────────────────────────────────────
    train_time_sec = 0.0
    converged_epoch = None
    weights_loaded = False

    if not retrain and weights_exist(food_type, variant_name):
        print(f'[LoadWeights] Loading {variant_name}/{food_type}...')
        if isinstance(model, PA2EPartial):
            model_ft = PA2EPartialFT(model).to(device)
        else:
            model_ft = model
        load_weights(model_ft, food_type, variant_name, device=device)
        weights_loaded = True

    if not weights_loaded:
        training_start = time.time()
        num_epochs_p1 = 1 if dry_run else 20
        num_epochs_p2 = 1 if dry_run else 500

        if cfg.get("phase1", True):
            train_phase1(model, train_loader, num_epochs=num_epochs_p1, lr=1e-3, loss_type=cfg["loss"])
        else:
            print('\n=== Skipping Phase 1 (no pre-training) ===')

        if isinstance(model, PA2EPartial):
            model_ft = PA2EPartialFT(model).to(device)
        else:
            model_ft = model

        if cfg.get ("phase2",True):
            if cfg.get("center",True):
                print ("using center")
                _, _, _, _, converged_epoch = train_phase2(
                    model_ft, train_loader, val_loader,
                    num_epochs=num_epochs_p2,
                    lr=1e-3,
                    alpha=cfg.get("alpha", 0.2),
                    beta=cfg.get("beta", 1.0),
                    loss_type=cfg["loss"],
                    food_type=food_type,
                    variant_name=variant_name,
                    dry_run=dry_run)
            else:
                print ("no center")
                _, _, _, _, converged_epoch = train_phase2(
                    model_ft, train_loader, val_loader,
                    num_epochs=num_epochs_p2,
                    lr=1e-3,
                    alpha=cfg.get("alpha", 0),
                    beta=cfg.get("beta", 1.0),
                    loss_type=cfg["loss"],
                    food_type=food_type,
                    variant_name=variant_name,
                    dry_run=dry_run)
        else:
            print ('\n=== Skipping Phase 2 (no phase2) ===')
            
        train_time_sec = time.time() - training_start

    # ── Inference ──────────────────────────────────────────────────────────────
    kernel = cfg.get("median_kernel", 3)
    if cfg["scoring"] == "nnmb":
        memory_bank = build_memory_bank(model_ft, train_loader)
        print(f"Memory Bank Shape: {memory_bank.shape}")
        smoothed_scores, infer_time = inference_nnmb(model_ft, test_loader, memory_bank, test_h, test_w, kernel=kernel)
    elif cfg["scoring"] == "dsvdd":
        smoothed_scores, infer_time = inference_dsvdd(model_ft, test_loader, test_h, test_w, kernel=kernel)
    else:
        raise ValueError(f"Unknown scoring: {cfg['scoring']!r}")

    # ── Evaluate ───────────────────────────────────────────────────────────────
    roc_auc, pr_auc = evaluate(smoothed_scores, binary_labels)

    max_vram_gb = 0.0
    if torch.cuda.is_available():
        max_vram_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

    total_elapsed = time.time() - total_start

    print(f"\n✅ {variant_name} [{food_type}]: ROC={roc_auc:.4f}, PR={pr_auc:.4f}, "
          f"train={train_time_sec:.1f}s, infer={infer_time:.2f}s, VRAM={max_vram_gb:.2f}GB")

    return {
        "roc_auc":          float(roc_auc),
        "pr_auc":           float(pr_auc),
        "detectmap_shape":  [int(test_h), int(test_w)],
        "converged_epoch":  int(converged_epoch) if converged_epoch is not None else None,
        "infer_time_sec":   float(infer_time),
        "train_time_sec":   float(train_time_sec),
        "n_params":         int(total_params),
        "max_vram_gb":      float(max_vram_gb),
        "config":           dict(cfg),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public API (called by ablation.py)
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_food_type(food_type, **kwargs):
    """
    Run all ablation variants for a given food type.

    Args:
        food_type: 'Almond', 'Pistachio', or 'GarlicStems'
        dry_run: bool (default False)
        retrain: bool (default True)
        variants: optional list of variant names; restricts the run to those
                  variants only (e.g. ['loss_mse', 'loss_mae'])

    Returns:
        dict mapping variant_name → result_dict
    """
    dry_run = kwargs.get('dry_run', False)
    retrain = kwargs.get('retrain', True)
    variants = kwargs.get('variants', None)

    print(f"\n{'='*80}")
    print(f"  Ablation Benchmark: {food_type}")
    print(f"  dry_run={dry_run}  retrain={retrain}")
    print(f"={'='*79}")

    base_path = os.path.join(
        os.path.dirname(__file__), '..', '..', 'AnomalyonFood', 'Dataset', food_type
    )

    # Deduplicate: run each unique config only once, store result under all matching names
    seen_configs: dict = {}    # config_key → result_dict
    all_results: dict = {}

    for variant_name, cfg in ABLATION_AXES.items():
        if variants is not None and variant_name not in variants:
            continue

        key = _config_key(cfg)
        if key in seen_configs:
            all_results[variant_name] = seen_configs[key]
            print(f"\n[skip dupe] {variant_name} → reusing result for config {key}")
            continue

        result = run_variant(variant_name, cfg, food_type, base_path, dry_run, retrain)
        seen_configs[key] = result
        all_results[variant_name] = result

    return all_results


if __name__ == '__main__':
    result = benchmark_food_type('Almond', dry_run=True, retrain=True)
    print(json.dumps(result, indent=2, default=str))
