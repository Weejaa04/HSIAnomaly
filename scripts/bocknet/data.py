import numpy as np
import torch

def load_hsi_data(data_path: str) -> np.ndarray:
    """Load HSI data from ENVI format."""
    import spectral
    return np.array(spectral.open_image(data_path).load(), dtype=np.float32)

def calibrate_hsi(data: np.ndarray, white_ref_path: str, dark_ref_path: str) -> np.ndarray:
    """Calibrate HSI data."""
    import spectral
    white = np.array(spectral.open_image(white_ref_path).load(), dtype=np.float32).mean(axis=0)
    dark  = np.array(spectral.open_image(dark_ref_path).load(),  dtype=np.float32).mean(axis=0)
    return np.clip((data - dark) / (white - dark + 1e-8), 0, 1)

def get_food_data(food_type: str, base_dir: str = 'AnomalyonFood/Dataset', device=None):
    """Load, calibrate and return (img_var, gt, H, W, B) for a food type.
    Note: BockNet expects inputs in (1, B, H, W).
    
    Also returns spatial masks for the Global Full-Image Model paradigm:
    - train_mask: Y[50:200]  (exclusive zone for weight updates)
    - val_mask:   Y[300:350] (for validation loss only)
    """
    base = f'{base_dir}/{food_type}'

    test_data = calibrate_hsi(
        load_hsi_data(f'{base}/Test/data.hdr'),
        f'{base}/Test/WHITEREF.hdr',
        f'{base}/Test/DARKREF.hdr',
    )  # (H, W, B)
    gt_raw = np.load(f'{base}/Test/label.npy')  # (H, W)

    H, W, B = test_data.shape
    gt = (gt_raw != 2)  # binary: anomaly=1, background=0

    # Convert test image to (1, B, H, W) tensor for the network
    img_np  = test_data.transpose(2, 0, 1)   # (B, H, W)
    img_np  = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
    
    img_var = torch.from_numpy(img_np).float()
    if device is not None:
        img_var = img_var.to(device)
    img_var = img_var.unsqueeze(0)  # (1, B, H, W)

    # Create spatial masks for the Global Full-Image Model paradigm
    train_mask = torch.zeros((H, W), dtype=torch.float32, device=device)
    train_mask[50:200, :] = 1.0
    
    val_mask = torch.zeros((H, W), dtype=torch.float32, device=device)
    val_mask[300:350, :] = 1.0

    return img_var, gt, H, W, B, train_mask, val_mask

