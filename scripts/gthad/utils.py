"""Utility functions for GT-HAD module."""

import os
import torch
import numpy as np
import spectral


def load_hsi_data(data_path):
    """Load HSI data from ENVI format."""
    img = spectral.open_image(data_path)
    data = img.load()
    return np.array(data, dtype=np.float32)


def load_label(label_path):
    """Load binary anomaly labels."""
    return np.load(label_path)


def load_white_ref(path):
    """Load white reference for calibration."""
    img = spectral.open_image(path)
    return np.array(img.load(), dtype=np.float32)


def load_dark_ref(path):
    """Load dark reference for calibration."""
    img = spectral.open_image(path)
    return np.array(img.load(), dtype=np.float32)


def calibrate_hsi(data, white_ref_path, dark_ref_path):
    """
    Calibrate HSI data (white-dark correction).
    Preserves spatial lighting gradients.
    
    Args:
        data: (H, W, C) raw HSI cube
        white_ref_path: Path to white reference ENVI file
        dark_ref_path: Path to dark reference ENVI file
    
    Returns:
        calibrated: (H, W, C) normalized to [0, 1]
    """
    white_ref = load_white_ref(white_ref_path)
    dark_ref = load_dark_ref(dark_ref_path)
    
    # Average across spatial dimension if needed
    if len(white_ref.shape) == 3:
        white_profile = white_ref.mean(axis=0)
        dark_profile = dark_ref.mean(axis=0)
    else:
        white_profile = white_ref
        dark_profile = dark_ref
    
    # Calibrate with broadcasting to preserve spatial gradients
    calibrated = (data - dark_profile) / (white_profile - dark_profile + 1e-8)
    calibrated = np.clip(calibrated, 0, 1)
    
    return calibrated


def get_spatial_train_val_mask(H=400, W=512):
    """
    Get spatial guillotine masks for train and validation zones.
    
    Protocol:
        Y[0:50]:     Discarded (sensor startup noise)
        Y[50:200]:   Train block (150 rows)
        Y[200:300]:  Dead zone (buffer)
        Y[300:350]:  Val block (50 rows)
        Y[350:400]:  Remaining (test zone)
    
    Returns:
        train_mask: (H, W) boolean mask
        val_mask: (H, W) boolean mask
    """
    train_mask = np.zeros((H, W), dtype=bool)
    train_mask[50:200, :] = True
    
    val_mask = np.zeros((H, W), dtype=bool)
    val_mask[300:350, :] = True
    
    return train_mask, val_mask


def get_random_train_val_mask(H, W, seed=42):
    """Generate random masks for training and validation.
    
    Random Sampling: 37.5% train, 12.5% val, remaining unlabeled.
    
    Args:
        H, W: Image height and width
        seed: Random seed for reproducibility
    
    Returns:
        train_mask: (H, W) boolean mask
        val_mask: (H, W) boolean mask
    """
    np.random.seed(seed)
    rand_map = np.random.rand(H, W)
    
    train_mask = rand_map < 0.375
    val_mask = (rand_map >= 0.375) & (rand_map < 0.500)
    
    return train_mask, val_mask


def img2mask(img):
    """Convert residual map to anomaly score map."""
    img = img[0].sum(0)  # Sum over channels and remove batch
    img = img - img.min()
    img = img / (img.max() + 1e-8)
    img = img.detach().cpu().numpy()
    return img


def save_weights(model, food_type: str, arch='gthad', suffix=''):
    """Save model weights to disk."""
    weight_dir = os.path.join(
        os.path.dirname(__file__), '..', '..', 'weights', arch
    )
    os.makedirs(weight_dir, exist_ok=True)
    weight_path = os.path.join(weight_dir, f'{food_type}{suffix}.pt')
    
    if isinstance(model, dict):
        torch.save(model, weight_path)
    else:
        torch.save(model.state_dict(), weight_path)
    print(f"✅ Weights saved to {weight_path}")


def load_weights(model, food_type: str, device='cpu', arch='gthad', suffix=''):
    """Load model weights from disk."""
    weight_path = os.path.join(
        os.path.dirname(__file__), '..', '..', 'weights', arch, f'{food_type}{suffix}.pt'
    )
    
    if os.path.exists(weight_path):
        if isinstance(model, dict):
            state = torch.load(weight_path, map_location=device)
            model.update(state)
        else:
            model.load_state_dict(
                torch.load(weight_path, map_location=device)
            )
        print(f"✅ Weights loaded from {weight_path}")
        return True
    else:
        print(f"⚠️  No weights found at {weight_path}")
        return False


def weights_exist(food_type: str, arch='gthad', suffix='') -> bool:
    """Check if saved weights exist."""
    weight_path = os.path.join(
        os.path.dirname(__file__), '..', '..', 'weights', arch, f'{food_type}{suffix}.pt'
    )
    return os.path.exists(weight_path)


class UniversalEarlyStopping:
    """
    Monitors validation reconstruction loss to prevent Identity Mapping Problem.
    
    References ES.md: Unbiased Early Stopping Protocol
    """
    def __init__(self, patience=50, min_delta=1e-4):
        """
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
        """Check if early stopping criterion is met."""
        if self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            print(f"  [EarlyStopping] Patience {self.counter}/{self.patience} (Best: {self.best_loss:.5f})")
            if self.counter >= self.patience:
                self.early_stop = True
