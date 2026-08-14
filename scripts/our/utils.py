"""
Utilities for PA2E training: Early Stopping, data loading, and helpers.
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


def calibrate_hsi(data, white_ref_path, dark_ref_path):
    """Calibrate HSI data while preserving the physical spatial lighting gradient."""
    white_ref = np.array(spectral.open_image(white_ref_path).load(), dtype=np.float32)
    dark_ref = np.array(spectral.open_image(dark_ref_path).load(), dtype=np.float32)
    
    # Average across the time/scan-line dimension (axis=0)
    white_profile = white_ref.mean(axis=0) 
    dark_profile = dark_ref.mean(axis=0)
    
    # Calibrate with broadcasting
    calibrated = (data - dark_profile) / (white_profile - dark_profile + 1e-8)
    calibrated = np.clip(calibrated, 0, 1)
    
    return calibrated


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
    
    train_mask = np.zeros(H * W, dtype=bool)
    val_mask = np.zeros(H * W, dtype=bool)
    
    # Reshape for indexing
    mask_2d_train = np.zeros((H, W), dtype=bool)
    mask_2d_val = np.zeros((H, W), dtype=bool)
    
    mask_2d_train[train_y_start:train_y_end, :] = True
    mask_2d_val[val_y_start:val_y_end, :] = True
    
    train_mask = mask_2d_train.flatten()
    val_mask = mask_2d_val.flatten()
    
    return train_mask, val_mask


def get_random_train_val_indices(H, W, seed=42):
    """Generate random indices for training and validation.
    
    Random Sampling: 37.5% train, 12.5% val, remaining unlabeled.
    
    Args:
        H, W: Image height and width
        seed: Random seed for reproducibility
    
    Returns:
        train_indices: 1D array of flattened pixel indices for training
        val_indices: 1D array of flattened pixel indices for validation
    """
    np.random.seed(seed)
    total_pixels = H * W
    rand_map = np.random.rand(total_pixels)
    
    train_indices = np.where(rand_map < 0.375)[0]
    val_indices = np.where((rand_map >= 0.375) & (rand_map < 0.500))[0]
    
    return train_indices, val_indices


def get_weight_paths(food_type, suffix=''):
    """
    Get weight file path for a given food type.
    
    Args:
        food_type: 'Almond', 'Pistachio', or 'GarlicStems'
        suffix: optional suffix (e.g., '_random') to distinguish weight files
    
    Returns:
        weight_path: path to the trained model weights (after both Phase 1 and Phase 2)
    """
    model_dir = os.path.join(
        os.path.dirname(__file__), '..', '..', 'weights', 'our'
    )
    os.makedirs(model_dir, exist_ok=True)
    
    weight_path = os.path.join(model_dir, f'{food_type}{suffix}.pt')
    
    return weight_path


def save_weights(model, food_type, suffix=''):
    """
    Save model weights to disk.
    
    Args:
        model: PyTorch model
        food_type: 'Almond', 'Pistachio', or 'GarlicStems'
        suffix: optional suffix (e.g., '_random') to distinguish weight files
    """
    weight_path = get_weight_paths(food_type, suffix=suffix)
    
    torch.save(model.state_dict(), weight_path)
    print(f"✅ Weights saved to {weight_path}")


def load_weights(model, food_type, device=None, suffix=''):
    """
    Load trained weights from saved checkpoint.
    
    Args:
        model: PyTorch model to load weights into
        food_type: 'Almond', 'Pistachio', or 'GarlicStems'
        device: torch device to load weights to (defaults to model.device if available)
        suffix: optional suffix (e.g., '_random') to distinguish weight files
    
    Returns:
        True if weights were loaded, False if file doesn't exist
    """
    weight_path = get_weight_paths(food_type, suffix=suffix)
    
    if os.path.exists(weight_path):
        # Use provided device or try to get from model
        if device is None:
            device = getattr(model, 'device', None)
            if device is None:
                device = 'cpu'
        model.load_state_dict(torch.load(weight_path, map_location=device))
        print(f"✅ Weights loaded from {weight_path}")
        return True
    else:
        print(f"⚠️  Weights not found at {weight_path}")
        return False


def weights_exist(food_type, suffix=''):
    """Check if weights exist for a given food type."""
    weight_path = get_weight_paths(food_type, suffix=suffix)
    return os.path.exists(weight_path)
