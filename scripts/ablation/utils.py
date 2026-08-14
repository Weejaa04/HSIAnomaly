"""
Utilities for ablation module.

Extends scripts/our/utils.py with HSI calibration variants:
- calibrate_hsi_mean: normalize by global mean per band
- calibrate_hsi_mean_per_column: (current) normalize per-column per-band  
- no_calibrate_hsi: raw data (clip 0-1 only)
"""
import torch
import numpy as np
import spectral
import os


class UniversalEarlyStopping:
    """
    Monitors validation reconstruction loss to prevent Identity Mapping.

    Args:
        patience (int): How many epochs to wait after the last meaningful
                        improvement in validation loss.
        min_delta (float): The minimum absolute change required to qualify
                           as an improvement.
    """

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
            print(f"  [EarlyStopping] Patience {self.counter}/{self.patience} (Best: {self.best_loss:.5f})")
            if self.counter >= self.patience:
                self.early_stop = True


def load_hsi_data(data_path):
    """Load HSI data from ENVI format."""
    img = spectral.open_image(data_path)
    data = img.load()
    return np.array(data, dtype=np.float32)


def load_label(label_path):
    """Load label data."""
    return np.load(label_path)


# ─── Calibration Variants ────────────────────────────────────────────────────

def calibrate_hsi_mean_per_column(data, white_ref_path, dark_ref_path):
    """
    (Current/default) Calibrate HSI while preserving spatial lighting gradient.
    White/dark reference averaged across scan-line axis (axis=0), broadcast over all rows.
    """
    white_ref = np.array(spectral.open_image(white_ref_path).load(), dtype=np.float32)
    dark_ref = np.array(spectral.open_image(dark_ref_path).load(), dtype=np.float32)

    # Average across the time/scan-line dimension (axis=0) → (1, W, B)
    white_profile = white_ref.mean(axis=0)
    dark_profile = dark_ref.mean(axis=0)

    calibrated = (data - dark_profile) / (white_profile - dark_profile + 1e-8)
    calibrated = np.clip(calibrated, 0, 1)

    return calibrated


def calibrate_hsi_mean(data, white_ref_path, dark_ref_path):
    """
    Calibrate HSI using global mean white/dark reference (no spatial gradient preserved).
    White/dark reference averaged across both spatial axes → scalar per band.
    """
    white_ref = np.array(spectral.open_image(white_ref_path).load(), dtype=np.float32)
    dark_ref = np.array(spectral.open_image(dark_ref_path).load(), dtype=np.float32)

    # Average across all spatial dims → (B,)
    white_mean = white_ref.mean(axis=(0, 1))
    dark_mean = dark_ref.mean(axis=(0, 1))

    calibrated = (data - dark_mean) / (white_mean - dark_mean + 1e-8)
    calibrated = np.clip(calibrated, 0, 1)

    return calibrated


def calibrate_hsi_none(data, white_ref_path=None, dark_ref_path=None):
    """
    No calibration — just clip raw sensor values to [0, 1].
    """
    return np.clip(data, 0, 1)


CALIBRATION_FNS = {
    "mean_per_column": calibrate_hsi_mean_per_column,
    "mean": calibrate_hsi_mean,
    "none": calibrate_hsi_none,
}


# ─── Spatial masks ───────────────────────────────────────────────────────────

def get_spatial_train_val_mask(H=400, W=512, train_y_start=None, train_y_end=None,
                               val_y_start=None, val_y_end=None):
    """
    Generate spatial masks for training and validation blocks.

    Following DATA.md spatial guillotine protocol (percentages of H):
    - Train Block: Y[12.5%:50%]   (37.5% of rows)
    - Dead Zone  : Y[50%:75%]
    - Val Block  : Y[75%:87.5%]   (12.5% of rows)

    For H=400 this equals the classic Y[50:200] / Y[300:350]; larger cubes
    (e.g. AgriFood, H=1000) scale to Y[125:500] / Y[750:875].

    Returns:
        train_mask: shape (H*W,), boolean array for train pixels
        val_mask: shape (H*W,), boolean array for val pixels
    """
    if train_y_start is None:
        train_y_start = round(0.125 * H)
    if train_y_end is None:
        train_y_end = round(0.5 * H)
    if val_y_start is None:
        val_y_start = round(0.75 * H)
    if val_y_end is None:
        val_y_end = round(0.875 * H)

    mask_2d_train = np.zeros((H, W), dtype=bool)
    mask_2d_val = np.zeros((H, W), dtype=bool)

    mask_2d_train[train_y_start:train_y_end, :] = True
    mask_2d_val[val_y_start:val_y_end, :] = True

    train_mask = mask_2d_train.flatten()
    val_mask = mask_2d_val.flatten()

    return train_mask, val_mask


# ─── Weight helpers (namespaced per variant) ─────────────────────────────────

def get_weight_paths(food_type, variant_name):
    """
    Get weight file path for a given food type and ablation variant.
    """
    model_dir = os.path.join(
        os.path.dirname(__file__), '..', '..', 'weights', 'ablation', variant_name
    )
    os.makedirs(model_dir, exist_ok=True)
    weight_path = os.path.join(model_dir, f'{food_type}.pt')
    return weight_path


def save_weights(model, food_type, variant_name):
    weight_path = get_weight_paths(food_type, variant_name)
    torch.save(model.state_dict(), weight_path)
    print(f"✅ Weights saved → {weight_path}")


def load_weights(model, food_type, variant_name, device=None):
    weight_path = get_weight_paths(food_type, variant_name)
    if os.path.exists(weight_path):
        if device is None:
            device = 'cpu'
        model.load_state_dict(torch.load(weight_path, map_location=device))
        print(f"✅ Weights loaded from {weight_path}")
        return True
    else:
        print(f"⚠️  Weights not found at {weight_path}")
        return False


def weights_exist(food_type, variant_name):
    weight_path = get_weight_paths(food_type, variant_name)
    return os.path.exists(weight_path)
