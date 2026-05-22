"""
PA2E BASED (Partial-window Autoencoder with DSVDD) Benchmark

Architecture: Conv1D partial encoders/decoders + DSVDD-style center loss
Training: Phase 1 (20 epochs, reconstruction), Phase 2 (DSVDD with Early Stopping)
Inference: Deep GPU k-NN on memory bank

Usage:
    from scripts.our.__main__ import benchmark_food_type
    result = benchmark_food_type('Almond', dry_run=False, retrain=True)
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Force FP32 precision (disable TF32)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
if hasattr(torch, 'set_float32_matmul_precision'):
    torch.set_float32_matmul_precision('highest')

import numpy as np
from datetime import datetime
from scipy.ndimage import median_filter
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, auc
from tqdm import tqdm
import random
import time

# Add to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from scripts.our.utils import (
    UniversalEarlyStopping, load_hsi_data, load_label, calibrate_hsi,
    get_spatial_train_val_mask, get_random_train_val_indices, save_weights, load_weights, weights_exist
)
from scripts.our.model import PA2E, PA2EFT
from scripts.our.data import HSIPixelDataset

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
# TRAINING FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def compute_reconstruction_loss(model, x):
    """Compute reconstruction loss for sliding windows."""
    z, reconstructed_windows = model(x)
    original_windows = model.window_extractor(x)
    loss = F.mse_loss(reconstructed_windows, original_windows)
    return loss, z


def train_phase1(model, train_loader, num_epochs=20, lr=1e-3):
    """
    Phase 1: Pre-training with reconstruction loss + OneCycleLR.
    
    OneCycleLR drives aggressive exploration so the Conv1D filters can
    learn the broad chemical-signature shapes without collapsing onto
    background noise.
    """
    print('\n=== Phase 1: Pre-training (Autoencoder + OneCycleLR) ===')
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=1e4,
    )
    
    losses = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            x = batch.to(device)
            
            optimizer.zero_grad()
            loss, _ = compute_reconstruction_loss(model, x)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        
        current_lr = scheduler.get_last_lr()[0]
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}, LR: {current_lr:.2e}')
    
    return losses


def initialize_center(model, train_loader):
    """Compute center as mean of latent representations."""
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
    
    print(f'Center shape: {center.shape}')
    print(f'Center norm: {center.norm().item():.4f}')
    
    return center


def train_phase2(model, train_loader, val_loader, num_epochs=500, lr=1e-4, alpha=0.5, food_type=None, suffix=''):
    """
    Phase 2: DSVDD-style manifold refinement + CosineAnnealingLR + Early Stopping.
    
    Note: num_epochs is a hard upper limit, but early stopping (patience=50, min_delta=1e-4)
    will typically terminate training well before this cap. This ensures the model trains
    until it provably stops learning the background manifold.
    
    Args:
        food_type: If provided, saves final weights after training
    """
    print('\n=== Phase 2: Main Training (DSVDD-style + Early Stopping) ===')
    
    initialize_center(model, train_loader)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=lr * 1e-2,
    )
    
    early_stopper = UniversalEarlyStopping(patience=50, min_delta=1e-4)
    
    losses = []
    center_losses = []
    recon_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_center_loss = 0
        epoch_recon_loss = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            x = batch.to(device)
            
            optimizer.zero_grad()
            
            recon_loss, z = compute_reconstruction_loss(model, x)
            center_loss = torch.mean(torch.sum((z - model.center) ** 2, dim=1))
            loss = center_loss + alpha * recon_loss
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_center_loss += center_loss.item()
            epoch_recon_loss += recon_loss.item()
        
        # Validation pass
        model.eval()
        val_loss_total = 0.0
        
        with torch.inference_mode():
            for val_batch in val_loader:
                x = val_batch.to(device)
                _, reconstructed = model(x)
                original_windows = model.window_extractor(x)
                loss = F.mse_loss(reconstructed, original_windows)
                val_loss_total += loss.item()
        
        avg_val_loss = val_loss_total / len(val_loader)
        
        scheduler.step()
        
        avg_loss = epoch_loss / len(train_loader)
        avg_center_loss = epoch_center_loss / len(train_loader)
        avg_recon_loss = epoch_recon_loss / len(train_loader)
        
        losses.append(avg_loss)
        center_losses.append(avg_center_loss)
        recon_losses.append(avg_recon_loss)
        val_losses.append(avg_val_loss)
        
        current_lr = scheduler.get_last_lr()[0]
        print(f'Epoch {epoch+1}/{num_epochs}, '
              f'Loss: {avg_loss:.6f}, '
              f'Center: {avg_center_loss:.6f}, '
              f'Recon: {avg_recon_loss:.6f}, '
              f'Val: {avg_val_loss:.6f}, '
              f'LR: {current_lr:.2e}')
        
        # Early stopping check
        early_stopper(avg_val_loss)
        if early_stopper.early_stop:
            print(f"\n✅ Convergence reached at epoch {epoch+1}. Stopping training.")
            break
    
    if food_type:
        save_weights(model, food_type, suffix=suffix)
    return losses, center_losses, recon_losses, val_losses


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE & EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def build_gpu_memory_bank(model, loader, device):
    """Build memory bank on GPU for deep k-NN inference."""
    model.eval()
    bank = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Building GPU Memory Bank"):
            x = batch.to(device)
            z, _ = model.encode(x)
            bank.append(z)
    
    return torch.cat(bank, dim=0)


def inference_deep_knn(model, test_loader, memory_bank, device, H, W):
    """Run deep k-NN inference on test set. Returns (scores, elapsed_time)."""
    print("\n--- Deep k-NN Inference ---")
    start = datetime.now()
    
    model.eval()
    all_scores = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Computing k-NN distances"):
            # Unpack batch (test_loader yields (features, labels))
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device)
            z, _ = model.encode(x)
            distances = torch.cdist(z, memory_bank, p=2.0)
            min_distances, _ = torch.min(distances, dim=1)
            all_scores.append(min_distances.cpu())
    
    scores = torch.cat(all_scores).numpy()
    
    # Spatial smoothing
    score_map = scores.reshape(H, W)
    smoothed = median_filter(score_map, size=3).flatten()
    
    end = datetime.now()
    elapsed = (end - start).total_seconds()
    
    print(f"⚡ Inference time: {elapsed:.2f}s")
    
    return smoothed, elapsed


def evaluate(scores, labels):
    """Compute ROC-AUC and PR-AUC with full curve data."""
    roc_auc = roc_auc_score(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)
    precision, recall, _ = precision_recall_curve(labels, scores)
    pr_auc = auc(recall, precision)
    
    return roc_auc, pr_auc, fpr.tolist(), tpr.tolist(), precision.tolist(), recall.tolist()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN BENCHMARK FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def benchmark_food_type(food_type, **kwargs):
    """
    Run PA2E benchmark on a specific food type.
    
    Args:
        food_type: 'Almond', 'Pistachio', or 'GarlicStems'
        dry_run: Run for 1 epoch only (optional)
        retrain: Whether to retrain or load saved model (optional)
        split_method: "spatial" (guillotine) or "random" (random sampling) (optional)
    
    Returns:
        dict with results including original_model and fused_model metrics
    """
    dry_run = kwargs.get('dry_run', False)
    retrain = kwargs.get('retrain', True)
    split_method = kwargs.get('split_method', 'spatial')
    suffix = '_random' if split_method == 'random' else ''
    suffix += kwargs.get('noise_suffix', '')
    print(f"\n{'='*80}")
    print(f"PA2E Benchmark: {food_type} (split_method={split_method})")
    print(f"{'='*80}")
    
    # Track timing
    total_start_time = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    
    base_path = os.path.join(os.path.dirname(__file__), '..', '..', 'AnomalyonFood', 'Dataset', food_type)
    
    # ─── Load Data ───────────────────────────────────────────────────────────
    train_data_path = os.path.join(base_path, 'Train', 'data.hdr')
    train_white_path = os.path.join(base_path, 'Train', 'WHITEREF.hdr')
    train_dark_path = os.path.join(base_path, 'Train', 'DARKREF.hdr')
    
    print(f'Loading {food_type} training data...')
    train_data = load_hsi_data(train_data_path)
    train_data = calibrate_hsi(train_data, train_white_path, train_dark_path)
    
    H, W, B = train_data.shape
    
    # Get masks based on split method
    if split_method == 'random':
        train_indices, val_indices = get_random_train_val_indices(H, W, seed=42)
        print(f'Random sampling protocol:')
        print(f'Train pixels: {len(train_indices)}  |  Val pixels: {len(val_indices)}')
    else:  # spatial
        train_indices, val_indices = get_spatial_train_val_mask(H, W)
        print(f'Spatial guillotine protocol:')
        print(f'Train pixels (spatial Y[50:200]): {train_indices.sum()}')
        print(f'Val pixels (spatial Y[300:350]): {val_indices.sum()}')
    
    # For compatibility with dataset creation, convert to boolean masks
    train_mask = np.zeros(H * W, dtype=bool)
    val_mask = np.zeros(H * W, dtype=bool)
    train_mask[train_indices] = True
    val_mask[val_indices] = True
    
    # Load test data
    test_data_path = os.path.join(base_path, 'Test', 'data.hdr')
    test_white_path = os.path.join(base_path, 'Test', 'WHITEREF.hdr')
    test_dark_path = os.path.join(base_path, 'Test', 'DARKREF.hdr')
    test_label_path = os.path.join(base_path, 'Test', 'label.npy')
    
    print(f'Loading {food_type} test data...')
    test_data = load_hsi_data(test_data_path)
    test_data = calibrate_hsi(test_data, test_white_path, test_dark_path)
    test_labels = load_label(test_label_path)
    
    # Flatten labels
    test_labels_flat = test_labels.reshape(-1)
    binary_labels = (test_labels_flat != 2).astype(int)
    
    print(f'Train data shape: {train_data.shape}')
    print(f'Test data shape: {test_data.shape}')
    print(f'Number of bands: {B}')
    print(f'Train pixels (spatial Y[50:200]): {train_mask.sum()}')
    print(f'Val pixels (spatial Y[300:350]): {val_mask.sum()}')
    
    # Create datasets
    batch_size = 512
    train_dataset = HSIPixelDataset(train_data, mask=train_mask)
    val_dataset = HSIPixelDataset(train_data, mask=val_mask)
    test_dataset = HSIPixelDataset(test_data, test_labels_flat)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    
    print(f'Train samples: {len(train_dataset)}')
    print(f'Val samples: {len(val_dataset)}')
    print(f'Test samples: {len(test_dataset)}')
    
    # ─── Initialize Model ────────────────────────────────────────────────────
    model = PA2E(
        input_dim=B,
        window_size=56,
        stride=14,
        latent_local_dim=28,
        latent_global_dim=32
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\nTotal parameters: {total_params:,}')
    
    # ─── Check if weights exist and retrain=False ────────────────────────────
    weights_loaded = False
    train_time_sec = 0.0
    
    if not retrain and weights_exist(food_type, suffix=suffix):
        print(f'\n[LoadWeights] Trained weights found. Loading {food_type} model...')
        # Create PA2EFT wrapper and load final weights
        model_ft = PA2EFT(model).to(device)
        load_weights(model_ft, food_type, device=device, suffix=suffix)
        weights_loaded = True
        phase1_losses = []
        phase2_losses, phase2_center_losses, phase2_recon_losses, phase2_val_losses = [], [], [], []
    
    # ─── Phase 1: Pre-training ──────────────────────────────────────────────
    if not weights_loaded:
        training_start = time.time()
        if dry_run:
            num_epochs_p1 = 1
            num_epochs_p2 = 1
        else:
            num_epochs_p1 = 20
            num_epochs_p2 = 500
        
        phase1_losses = train_phase1(model, train_loader, num_epochs=num_epochs_p1, lr=1e-3)
        
        # ── Phase 2: DSVDD-style Training with Early Stopping ──────────────────
        model_ft = PA2EFT(model).to(device)
        phase2_losses, phase2_center_losses, phase2_recon_losses, phase2_val_losses = train_phase2(
            model_ft, train_loader, val_loader, num_epochs=num_epochs_p2, lr=1e-3, alpha=0.5, food_type=food_type, suffix=suffix
        )
        train_time_sec = time.time() - training_start
    else:
        # Already loaded weights, model_ft is ready to use
        pass

    
    # ─── Build Memory Bank & Inference ──────────────────────────────────────
    memory_bank = build_gpu_memory_bank(model_ft, train_loader, device)
    print(f"Memory Bank Shape: {memory_bank.shape}")
    
    test_h, test_w = test_data.shape[:2]
    smoothed_scores, infer_time = inference_deep_knn(model_ft, test_loader, memory_bank, device, test_h, test_w)
    
    # ─── Evaluation ─────────────────────────────────────────────────────────
    roc_auc, pr_auc, fpr, tpr, precision, recall = evaluate(smoothed_scores, binary_labels)
    
    # Calculate memory stats
    max_vram_gb = 0.0
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated(device)
        max_vram_gb = peak_memory / (1024 ** 3)
    
    total_elapsed = time.time() - total_start_time
    
    print(f"\n{'='*80}")
    print(f"Results: ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f}")
    print(f"Training time: {train_time_sec:.2f}s | Inference time: {infer_time:.2f}s | Total time: {total_elapsed:.2f}s")
    if max_vram_gb > 0:
        print(f"Peak VRAM: {max_vram_gb:.2f} GB")
    print(f"{'='*80}")
    
    anomaly_dir = kwargs.get('anomaly_dir')
    if anomaly_dir:
        np.save(os.path.join(anomaly_dir, 'our_' + food_type + '_scores.npy'), smoothed_scores.reshape(test_h, test_w))
        label_path = os.path.join(anomaly_dir, f'{food_type}_labels.npy')
        if not os.path.exists(label_path):
            np.save(label_path, binary_labels.reshape(test_h, test_w).astype(np.uint8))

    # Return results matching benchmark.py API (single model format)
    return {
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'detectmap_shape': [int(test_h), int(test_w)],
        'infer_time_sec': float(infer_time),
        'n_params': int(total_params),
        'max_vram_gb': float(max_vram_gb),
        'fpr': fpr,
        'tpr': tpr,
        'precision': precision,
        'recall': recall,
    }


if __name__ == '__main__':
    # Quick test
    result = benchmark_food_type('Almond', dry_run=False, retrain=True)
    print(json.dumps(result, indent=2, default=str))
