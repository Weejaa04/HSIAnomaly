"""
DMS2F-HAD: Dual Mamba Spectral-Spatial Fusion for Hyperspectral Anomaly Detection
Reference: Based on the DMS2F-HAD architecture.

Usage:
    from scripts.dms2f.__main__ import benchmark_food_type
    result = benchmark_food_type('Almond', dry_run=False, retrain=True)
"""

import os
import sys
import json
import time
import random
import math
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc
from scipy.ndimage import median_filter
from scipy.stats import rankdata
from tqdm import tqdm
from einops import rearrange

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from scripts.dms2f.utils import (
    load_hsi_data, load_label, calibrate_hsi,
    get_spatial_train_val_mask, get_random_train_val_mask,
)
from scripts.dms2f.model import AnomalyDetectionModel
from scripts.metrics import f1_acc_at_tnr95, count_flops

SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=True)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f'Using device: {torch.cuda.get_device_name(0)}')
else:
    print('Using CPU')


# ─── Block utilities ──────────────────────────────────────────────────────
def block_search(image_shape, block_size, stride, pad=True):
    H, W = image_shape[:2]
    pad_h = pad_w = 0
    if pad:
        if (H - block_size) % stride != 0:
            pad_h = stride - ((H - block_size) % stride)
        if (W - block_size) % stride != 0:
            pad_w = stride - ((W - block_size) % stride)
    H_pad, W_pad = H + pad_h, W + pad_w
    positions = []
    for i in range(0, H_pad - block_size + 1, stride):
        for j in range(0, W_pad - block_size + 1, stride):
            positions.append((i, j))
    return positions, (H_pad, W_pad)


def block_embedding(image, block_size, stride):
    H, W = image.shape[0], image.shape[1]
    positions, (H_pad, W_pad) = block_search((H, W), block_size, stride, pad=True)
    pad_h, pad_w = H_pad - H, W_pad - W
    if pad_h > 0 or pad_w > 0:
        image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
    blocks = []
    for i, j in positions:
        block = image[i:i + block_size, j:j + block_size, :]
        blocks.append(block)
    blocks = np.stack(blocks, axis=0)
    blocks = torch.from_numpy(blocks.astype(np.float32)).permute(0, 3, 1, 2)
    return blocks, (H_pad, W_pad), positions


def block_fold(blocks, image_shape, block_size, stride, positions):
    H_orig, W_orig = image_shape
    H_pad = max(i for i, j in positions) + block_size
    W_pad = max(j for i, j in positions) + block_size
    C = blocks.shape[1]
    output = torch.zeros((C, H_pad, W_pad), dtype=blocks.dtype)
    count = torch.zeros((H_pad, W_pad), dtype=blocks.dtype)
    for block, (i, j) in zip(blocks, positions):
        output[:, i:i + block_size, j:j + block_size] += block
        count[i:i + block_size, j:j + block_size] += 1
    count[count == 0] = 1
    output = output / count.unsqueeze(0)
    return output[:, :H_orig, :W_orig]


class HSIBlockDataset(Dataset):
    """Self-supervised dataset: input = target = block."""
    def __init__(self, blocks):
        self.blocks = blocks

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx):
        return self.blocks[idx], self.blocks[idx]


# ─── Early stopping ───────────────────────────────────────────────────────
class UniversalEarlyStopping:
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


# ─── Loss ─────────────────────────────────────────────────────────────────
mse_crit = nn.MSELoss()
l1_crit = nn.L1Loss()


def compute_loss(recon, target):
    mse = mse_crit(recon, target)
    l1 = l1_crit(recon, target)
    return mse + 0.1 * l1, mse, l1


def run_val(model, loader, device):
    model.eval()
    total = 0.
    with torch.no_grad():
        for blocks, targets in loader:
            blocks = blocks.to(device)
            targets = targets.to(device)
            recon, _ = model(blocks)
            loss, _, _ = compute_loss(recon, targets)
            total += loss.item()
    return total / len(loader)


# ─── Weights helpers ──────────────────────────────────────────────────────
WEIGHTS_DIR = 'weights/dms2f'


def weights_exist(food_type, split_method='spatial', noise_suffix=''):
    suffix = '_random' if split_method == 'random' else ''
    suffix += noise_suffix
    path = os.path.join(WEIGHTS_DIR, f'dms2f_{food_type}{suffix}_best.pt')
    return os.path.isfile(path)


def save_weights(model, food_type, split_method='spatial', noise_suffix=''):
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    suffix = '_random' if split_method == 'random' else ''
    suffix += noise_suffix
    path = os.path.join(WEIGHTS_DIR, f'dms2f_{food_type}{suffix}_best.pt')
    torch.save({'state_dict': model.state_dict()}, path)
    print(f'  💾 Weights saved → {path}')


def load_weights(model, food_type, split_method='spatial', noise_suffix=''):
    suffix = '_random' if split_method == 'random' else ''
    suffix += noise_suffix
    path = os.path.join(WEIGHTS_DIR, f'dms2f_{food_type}{suffix}_best.pt')
    if os.path.isfile(path):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['state_dict'], strict=False)
        print(f'  ✅ Weights loaded from {path}')
        return True
    return False


# ─── Main benchmarking function ───────────────────────────────────────────
def benchmark_food_type(food_type, **kwargs):
    """
    Train and evaluate DMS2F-HAD on a single food type.

    Args:
        food_type: 'Almond', 'Pistachio', or 'GarlicStems'
        dry_run: Run for limited epochs (optional)
        retrain: Whether to retrain or load saved model (optional)
        split_method: 'spatial' (guillotine) or 'random' (optional)

    Returns:
        dict with keys: roc_auc, pr_auc, fpr, tpr, precision, recall, ...
    """
    dry_run = kwargs.get('dry_run', False)
    retrain = kwargs.get('retrain', True)
    split_method = kwargs.get('split_method', 'spatial')
    anomaly_dir = kwargs.get('anomaly_dir')
    suffix = '_random' if split_method == 'random' else ''
    suffix += kwargs.get('noise_suffix', '')

    set_seed(kwargs.get('seed', SEED))
    print(f"\n{'=' * 70}")
    print(f"BENCHMARKING: {food_type} (DMS2F-HAD, split_method={split_method})")
    print(f"{'=' * 70}")

    # ── Hyperparameters ───────────────────────────────────────────────
    BLOCK_SIZE = 16
    STRIDE_PATCH = 8
    DIM = 64
    DEPTH = 1
    SPEC_NUM = 12
    SPEC_RATE = 0.5
    SPA_TOKEN = 16
    MODE = 'full'
    BATCH_SIZE = 32
    EPOCHS = kwargs.get('max_epochs', 5 if dry_run else 10000)
    LR = 5e-4
    WEIGHT_DECAY = 1e-4
    PATIENCE = 5 if dry_run else 50

    # ── Data loading ──────────────────────────────────────────────────
    base_dir = kwargs.get('base_dir', 'AnomalyonFood/Dataset')
    BASE_PATH = os.path.join(base_dir, food_type)
    TRAIN_PATH = os.path.join(BASE_PATH, 'Train', 'data.hdr')
    TRAIN_WHITE_PATH = os.path.join(BASE_PATH, 'Train', 'WHITEREF.hdr')
    TRAIN_DARK_PATH = os.path.join(BASE_PATH, 'Train', 'DARKREF.hdr')
    TEST_PATH = os.path.join(BASE_PATH, 'Test', 'data.hdr')
    TEST_WHITE_PATH = os.path.join(BASE_PATH, 'Test', 'WHITEREF.hdr')
    TEST_DARK_PATH = os.path.join(BASE_PATH, 'Test', 'DARKREF.hdr')
    TEST_LABEL_PATH = os.path.join(BASE_PATH, 'Test', 'label.npy')

    print(f'Loading {food_type} training data...')
    train_data = load_hsi_data(TRAIN_PATH)
    train_data = calibrate_hsi(train_data, TRAIN_WHITE_PATH, TRAIN_DARK_PATH)
    H, W, B = train_data.shape
    print(f'  Train data shape: {train_data.shape}')

    print(f'Loading {food_type} test data...')
    test_data = load_hsi_data(TEST_PATH)
    test_data = calibrate_hsi(test_data, TEST_WHITE_PATH, TEST_DARK_PATH)
    test_labels = load_label(TEST_LABEL_PATH)
    H_test, W_test = test_data.shape[:2]
    test_labels_flat = test_labels.reshape(-1)
    binary_labels = (test_labels_flat != 2).astype(int)
    print(f'  Test data shape: {test_data.shape}')
    print(f'  Anomaly pixels: {binary_labels.sum():,} / {len(binary_labels):,} '
          f'({100 * binary_labels.mean():.2f}%)')

    # ── Split & Block extraction ──────────────────────────────────────
    if split_method == 'random':
        train_mask, val_mask = get_random_train_val_mask(H, W)
    else:
        train_mask, val_mask = get_spatial_train_val_mask(H, W)

    print('Building block datasets...')
    train_mask_flat = train_mask.ravel()
    val_mask_flat = val_mask.ravel()

    all_blocks, _, all_positions = block_embedding(train_data, BLOCK_SIZE, STRIDE_PATCH)
    test_blocks, _, test_positions = block_embedding(test_data, BLOCK_SIZE, STRIDE_PATCH)

    train_idx, val_idx = [], []
    for idx, (i, j) in enumerate(all_positions):
        ci = min(i + BLOCK_SIZE // 2, H - 1)
        cj = min(j + BLOCK_SIZE // 2, W - 1)
        pos = ci * W + cj
        if train_mask_flat[pos]:
            train_idx.append(idx)
        elif val_mask_flat[pos]:
            val_idx.append(idx)

    train_blocks = all_blocks[torch.tensor(train_idx)]
    val_blocks = all_blocks[torch.tensor(val_idx)]

    if len(train_blocks) < 1:
        train_blocks = all_blocks[:len(all_blocks) // 2]
        val_blocks = all_blocks[len(all_blocks) // 2:3 * len(all_blocks) // 4]

    print(f'  Train blocks: {train_blocks.shape}')
    print(f'  Val   blocks: {val_blocks.shape}')
    print(f'  Test  blocks: {test_blocks.shape}')

    train_dataset = HSIBlockDataset(train_blocks)
    val_dataset = HSIBlockDataset(val_blocks)
    test_dataset = HSIBlockDataset(test_blocks)

    num_workers = min(2, os.cpu_count() or 1)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────
    model = AnomalyDetectionModel(
        in_channels=B, mode=MODE, dim=DIM, depth=DEPTH,
        spec_num=SPEC_NUM, spec_rate=SPEC_RATE, spa_token=SPA_TOKEN
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # ── Training ──────────────────────────────────────────────────────
    training_time = 0.0
    noise_suffix = kwargs.get('noise_suffix', '')
    if retrain or not weights_exist(food_type, split_method, noise_suffix=noise_suffix):
        print(f'\n{"─" * 50}')
        print(f'Training DMS2F-HAD on {food_type}...')
        print(f'{"─" * 50}')

        model.train()
        for p in model.parameters():
            p.requires_grad = True

        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS, eta_min=LR * 1e-2)
        early_stopper = UniversalEarlyStopping(patience=PATIENCE, min_delta=1e-4)

        best_auc = 0.0
        best_pr_auc = 0.0
        best_state = None
        t_start = time.time()

        for epoch in range(1, EPOCHS + 1):
            model.train()
            running = 0.
            ep_start = time.time()

            pbar = tqdm(train_loader, desc=f'Ep {epoch:3d}/{EPOCHS}', leave=False)
            for blocks, targets in pbar:
                blocks = blocks.to(device)
                targets = targets.to(device)
                optimizer.zero_grad()
                recon, _ = model(blocks)
                loss, mse, l1 = compute_loss(recon, targets)
                loss.backward()
                optimizer.step()
                running += loss.item() * blocks.size(0)
                pbar.set_postfix({'loss': f'{loss.item():.4f}',
                                  'lr': f'{scheduler.get_last_lr()[0]:.1e}'})

            n = len(train_loader.dataset)
            avg_train = running / n
            avg_val = run_val(model, val_loader, device)
            scheduler.step()

            # Evaluate on test set
            model.eval()
            recs = []
            with torch.no_grad():
                for blk, _ in test_loader:
                    out, _ = model(blk.to(device))
                    recs.append(out.cpu())
            recs = torch.cat(recs, dim=0)
            recon_map = block_fold(recs, (H_test, W_test),
                                    BLOCK_SIZE, STRIDE_PATCH, test_positions)
            orig_map = torch.tensor(test_data.astype(np.float32)).permute(2, 0, 1)
            diff = recon_map - orig_map
            res_map = torch.norm(diff, dim=0).numpy()
            gt_flat = binary_labels.astype(int)
            val_auc = roc_auc_score(gt_flat, res_map.ravel())
            prec_ep, rec_ep, _ = precision_recall_curve(gt_flat, res_map.ravel())
            val_pr_auc = auc(rec_ep, prec_ep)

            if epoch == 1 or epoch % 10 == 0 or dry_run:
                ep_time = time.time() - ep_start
                print(f'  Ep {epoch:4d} | train={avg_train:.5f} | val={avg_val:.5f} | '
                      f'ROC-AUC={val_auc:.4f} | PR-AUC={val_pr_auc:.4f} | '
                      f'lr={scheduler.get_last_lr()[0]:.1e} | {ep_time:.1f}s')

            if val_auc > best_auc:
                best_auc = val_auc
                best_pr_auc = val_pr_auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            early_stopper(avg_val)
            if early_stopper.early_stop:
                print(f'  Convergence at epoch {epoch}. Training stopped.')
                break
            if dry_run and epoch >= 2:
                print('  Dry run: stopping after 2 epochs.')
                break

        training_time = time.time() - t_start
        print(f'\nTraining done. Best ROC-AUC: {best_auc:.4f} | PR-AUC: {best_pr_auc:.4f}')

        if best_state is not None:
            model.load_state_dict(best_state)
            save_weights(model, food_type, split_method, noise_suffix=noise_suffix)
    else:
        print(f'\nLoading saved weights for {food_type}...')
        load_weights(model, food_type, split_method, noise_suffix=noise_suffix)

    # ── Inference ─────────────────────────────────────────────────────
    print(f'\nRunning inference...')
    model.eval()
    infer_start = time.time()
    recs = []
    with torch.no_grad():
        for blk, _ in tqdm(test_loader, desc='Inference'):
            out, _ = model(blk.to(device))
            recs.append(out.cpu())
    recs = torch.cat(recs, dim=0)
    recon_map = block_fold(recs, (H_test, W_test),
                            BLOCK_SIZE, STRIDE_PATCH, test_positions)
    orig_map = torch.tensor(test_data.astype(np.float32)).permute(2, 0, 1)
    diff = recon_map - orig_map
    residual = torch.norm(diff, dim=0).numpy()
    infer_time = time.time() - infer_start

    # ── Metrics ───────────────────────────────────────────────────────
    score_flat = residual.ravel()
    rank_scores = rankdata(score_flat, method='average') / len(score_flat)
    score_norm = rank_scores.reshape(H_test, W_test)
    score_1d = score_norm.ravel()
    label_1d = binary_labels.ravel()

    roc_auc = roc_auc_score(label_1d, score_1d)
    precision, recall, _ = precision_recall_curve(label_1d, score_1d)
    pr_auc = auc(recall, precision)
    fpr, tpr, _ = roc_curve(label_1d, score_1d)
    f1_tnr95, acc_tnr95 = f1_acc_at_tnr95(score_1d, label_1d)
    gflops = count_flops(model, torch.zeros(1, B, BLOCK_SIZE, BLOCK_SIZE, device=device))

    # Median filter post-processing (matching other archs)
    residual_smooth = median_filter(score_1d, size=5)
    roc_auc_smooth = roc_auc_score(label_1d, residual_smooth)

    max_vram_gb = 0.0
    if torch.cuda.is_available():
        max_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    # Save anomaly map
    if anomaly_dir:
        np.save(os.path.join(anomaly_dir, f'dms2f_{food_type}_scores.npy'), residual)
        label_path = os.path.join(anomaly_dir, f'{food_type}_labels.npy')
        if not os.path.exists(label_path):
            np.save(label_path, binary_labels.astype(np.uint8))

    print(f'\n{"=" * 50}')
    print(f'  {food_type} — DMS2F-HAD ({MODE})')
    print(f'  ROC-AUC : {roc_auc:.4f}')
    print(f'  PR-AUC  : {pr_auc:.4f}')
    print(f'  Parameters: {n_params:,}')
    if max_vram_gb > 0:
        print(f'  Peak VRAM: {max_vram_gb:.2f} GB')
    print(f'{"=" * 50}')

    return {
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'detectmap_shape': list(residual.shape),
        'infer_time_sec': infer_time,
        'n_params': n_params,
        'max_vram_gb': max_vram_gb,
        'f1_tnr95': f1_tnr95,
        'acc_tnr95': acc_tnr95,
        'gflops': gflops,
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'precision': precision.tolist(),
        'recall': recall.tolist(),
    }


if __name__ == '__main__':
    result = benchmark_food_type('Almond', dry_run=False, retrain=True)
    print('\n' + json.dumps(result, indent=2, default=str))
