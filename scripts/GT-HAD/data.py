import numpy as np
import torch
import torch.utils.data as data
from .block import Block_embedding


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


def load_food_data(food_type: str, base_dir: str = 'AnomalyonFood/Dataset', device=None):
    """Load, calibrate and return (img_var, gt, H, W, B) for a food type.

    img_var : torch.FloatTensor  shape (1, B, H, W)  — net input
    gt      : np.ndarray         shape (H, W)         — binary ground truth
    """
    base = f'{base_dir}/{food_type}'

    # Training image (whole image is the self-supervised context)
    train_data = calibrate_hsi(
        load_hsi_data(f'{base}/Train/data.hdr'),
        f'{base}/Train/WHITEREF.hdr',
        f'{base}/Train/DARKREF.hdr',
    )  # (H, W, B)

    # Test image + label
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


class DatasetHsi(data.Dataset):
    def __init__(self, hsi_data, wsize=15, wstride=5):
        super().__init__()
        self.data_processer = Block_embedding(wsize=wsize, wstride=wstride)
        self.block_gt, self.block_input, self.padding = self.data_processer(hsi_data)

    def __getitem__(self, index):
        return {
            'block_gt':    self.block_gt[index],
            'block_input': self.block_input[index],
            'index':       index,
        }

    def __len__(self):
        return self.block_gt.size(0)