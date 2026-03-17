import numpy as np
import torch

SEED_DICT = {
    "Almond": 42,
    "Pistachio": 42,
    "GarlicStems": 42,
}


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
            if self.counter % 10 == 0:
                print(
                    f"  [EarlyStopping] Patience {self.counter}/{self.patience} (Best: {self.best_loss:.5f})"
                )
            if self.counter >= self.patience:
                self.early_stop = True


def load_hsi_data(data_path: str) -> np.ndarray:
    """Load HSI data from ENVI format."""
    import spectral

    return np.array(spectral.open_image(data_path).load(), dtype=np.float32)


def calibrate_hsi(
    data: np.ndarray, white_ref_path: str, dark_ref_path: str
) -> np.ndarray:
    """Calibrate HSI data using white-dark correction."""
    import spectral

    white = np.array(spectral.open_image(white_ref_path).load(), dtype=np.float32).mean(
        axis=0
    )
    dark = np.array(spectral.open_image(dark_ref_path).load(), dtype=np.float32).mean(
        axis=0
    )
    return np.clip((data - dark) / (white - dark + 1e-8), 0, 1)


def get_food_data(food_type: str, base_dir: str = "AnomalyonFood/Dataset", device=None):
    """Load, calibrate and return data for a food type.

    Returns:
        img: (1, B, H, W) tensor
        gt: (H, W) binary ground truth
        H, W, B: spatial and spectral dimensions
    """
    base = f"{base_dir}/{food_type}"

    test_data = calibrate_hsi(
        load_hsi_data(f"{base}/Test/data.hdr"),
        f"{base}/Test/WHITEREF.hdr",
        f"{base}/Test/DARKREF.hdr",
    )
    gt_raw = np.load(f"{base}/Test/label.npy")

    H, W, B = test_data.shape
    gt = gt_raw != 2

    from .model import hyper_norm

    img_np = test_data.transpose(2, 0, 1)
    img_np = hyper_norm(img_np)

    img_var = torch.from_numpy(img_np).float().unsqueeze(0)
    if device is not None:
        img_var = img_var.to(device)

    return img_var, gt, H, W, B
