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
    """
    base = f'{base_dir}/{food_type}'

    # For BockNet, we might just train on the Test image as an unsupervised anomaly detection task,
    # or follow the exact convention of whether Train data is used.
    # The original BockNet main.py just loads a single test image `image = input_data['data']` and uses it as both train and test.
    # We will use the Test image (`data.hdr` and `label.npy`) as the target for blind-spot reconstruction.
    
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

    return img_var, gt, H, W, B
