"""
Utilities for SGLNet training: Early Stopping, data loading, and helpers.
"""

import torch
import numpy as np
import spectral
import os


class UniversalEarlyStopping:
    """
    Monitors validation reconstruction loss to prevent identity mapping.

    Args:
        patience (int): How many epochs to wait after the last improvement
        min_delta (float): Minimum absolute change to qualify as improvement
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
            print(
                f"  [EarlyStopping] Patience {self.counter}/{self.patience} (Best: {self.best_loss:.5f})"
            )
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


def calibrate_hsi(data, white_ref_path, dark_ref_path):
    """Calibrate HSI data while preserving the physical spatial lighting gradient."""
    white_ref = np.array(spectral.open_image(white_ref_path).load(), dtype=np.float32)
    dark_ref = np.array(spectral.open_image(dark_ref_path).load(), dtype=np.float32)

    white_profile = white_ref.mean(axis=0)
    dark_profile = dark_ref.mean(axis=0)

    calibrated = (data - dark_profile) / (white_profile - dark_profile + 1e-8)
    calibrated = np.clip(calibrated, 0, 1)

    return calibrated


def get_spatial_train_val_mask(
    H=400, W=512, train_y_start=50, train_y_end=200, val_y_start=300, val_y_end=350
):
    """
    Generate spatial masks for training and validation blocks.

    Following DATA.md spatial guillotine protocol:
    - Train Block: Y[50:200]
    - Dead Zone: Y[200:300]
    - Val Block: Y[300:350]

    Returns:
        train_mask: shape (H*W,), boolean array for train pixels
        val_mask: shape (H*W,), boolean array for val pixels
    """
    train_mask = np.zeros(H * W, dtype=bool)
    val_mask = np.zeros(H * W, dtype=bool)

    mask_2d_train = np.zeros((H, W), dtype=bool)
    mask_2d_val = np.zeros((H, W), dtype=bool)

    mask_2d_train[train_y_start:train_y_end, :] = True
    mask_2d_val[val_y_start:val_y_end, :] = True

    train_mask = mask_2d_train.flatten()
    val_mask = mask_2d_val.flatten()

    return train_mask, val_mask


def get_random_train_val_mask(H, W, seed=42):
    """Generate random masks for training and validation pixels."""
    np.random.seed(seed)
    total_pixels = H * W
    rand_map = np.random.rand(total_pixels)
    train_mask = (rand_map < 0.375).reshape(H, W).flatten()
    val_mask = ((rand_map >= 0.375) & (rand_map < 0.500)).reshape(H, W).flatten()
    return train_mask, val_mask


def get_weight_dir():
    """Get weight directory path for SGLNet."""
    return os.path.join(os.path.dirname(__file__), "..", "..", "weights", "sglnet")


def get_weight_paths(food_type, suffix=''):
    """Get weight file path for a given food type."""
    model_dir = get_weight_dir()
    os.makedirs(model_dir, exist_ok=True)
    return os.path.join(model_dir, f"{food_type}{suffix}.pt")


def save_weights(model, food_type, suffix=''):
    """Save model weights to disk."""
    weight_path = get_weight_paths(food_type, suffix=suffix)
    torch.save(model.state_dict(), weight_path)
    print(f"✅ Weights saved to {weight_path}")


def load_weights(model, food_type, device=None, suffix=''):
    """Load trained weights from saved checkpoint."""
    weight_path = get_weight_paths(food_type, suffix=suffix)

    if os.path.exists(weight_path):
        if device is None:
            device = getattr(model, "device", None)
            if device is None:
                device = "cpu"
        model.load_state_dict(torch.load(weight_path, map_location=device))
        print(f"✅ Weights loaded from {weight_path}")
        return True
    else:
        print(f"⚠️  Weights not found at {weight_path}")
        return False


def weights_exist(food_type, suffix=''):
    """Check if weights exist for a given food type."""
    return os.path.exists(get_weight_paths(food_type, suffix=suffix))
