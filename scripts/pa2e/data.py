import numpy as np
import torch
from torch.utils.data import Dataset


# ============================= DATA LOADING ==============================

def load_hsi_data(data_path: str) -> np.ndarray:
    """Load HSI data from ENVI format."""
    import spectral
    img = spectral.open_image(data_path)
    data = img.load()
    return np.array(data, dtype=np.float32)


def load_label(label_path: str) -> np.ndarray:
    """Load label data."""
    return np.load(label_path)


def calibrate_hsi(data: np.ndarray, white_ref_path: str, dark_ref_path: str) -> np.ndarray:
    """Calibrate HSI data while preserving the physical spatial lighting gradient."""
    import spectral

    white_ref = np.array(spectral.open_image(white_ref_path).load(), dtype=np.float32)
    dark_ref  = np.array(spectral.open_image(dark_ref_path).load(),  dtype=np.float32)

    white_profile = white_ref.mean(axis=0)
    dark_profile  = dark_ref.mean(axis=0)

    calibrated = (data - dark_profile) / (white_profile - dark_profile + 1e-8)
    return np.clip(calibrated, 0, 1)


class HSIPixelDataset(Dataset):
    """Dataset that treats each pixel as an independent sample.

    Accepts data of shape (H, W, B); reshapes internally.
    """

    def __init__(self, data: np.ndarray, labels: np.ndarray = None):
        """
        Args:
            data:   HSI data of shape (H, W, B)
            labels: Label array of shape (H, W) or None
        """
        self.data   = data
        self.labels = labels

        H, W, B = data.shape
        self.pixels      = data.reshape(-1, B)
        self.pixel_labels = labels.reshape(-1) if labels is not None else None
        self.shape = (H, W)

    def __len__(self):
        return len(self.pixels)

    def __getitem__(self, idx):
        pixel = torch.from_numpy(self.pixels[idx]).float()
        if self.pixel_labels is not None:
            label = torch.tensor(self.pixel_labels[idx], dtype=torch.long)
            return pixel, label
        return pixel
