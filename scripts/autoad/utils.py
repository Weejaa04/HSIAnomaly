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


def get_random_train_val_mask(H, W, seed=42):
    """Generate random masks for training and validation.
    
    Random Sampling: 37.5% train, 12.5% val, remaining unlabeled.
    
    Args:
        H, W: Image height and width
        seed: Random seed for reproducibility
    
    Returns:
        train_mask: (H, W) boolean mask for training
        val_mask: (H, W) boolean mask for validation
    """
    np.random.seed(seed)
    rand_map = np.random.rand(H, W)
    
    train_mask = rand_map < 0.375
    val_mask = (rand_map >= 0.375) & (rand_map < 0.500)
    
    train_mask = torch.from_numpy(train_mask).float()
    val_mask = torch.from_numpy(val_mask).float()
    
    return train_mask, val_mask


def get_food_data(food_type: str, base_dir: str = "AnomalyonFood/Dataset", device=None, split_method: str = "spatial"):
    """Load, calibrate and return data for a food type.

    Args:
        food_type: Food type name
        base_dir: Base directory for data
        device: Torch device
        split_method: "spatial" (guillotine) or "random" (random sampling)

    Returns:
        img_var: (1, B, H, W) tensor
        gt: (H, W) binary ground truth
        H, W, B: spatial and spectral dimensions
        train_mask: (H, W) mask for training region
        val_mask: (H, W) mask for validation region
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

    img_np = test_data.transpose(2, 0, 1)
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)

    img_var = torch.from_numpy(img_np).float()
    if device is not None:
        img_var = img_var.to(device)
    img_var = img_var.unsqueeze(0)

    # Generate masks based on split method
    if split_method == "random":
        train_mask, val_mask = get_random_train_val_mask(H, W, seed=42)
    else:  # spatial (guillotine)
        train_mask = torch.zeros((H, W), dtype=torch.float32)
        train_mask[50:200, :] = 1.0
        val_mask = torch.zeros((H, W), dtype=torch.float32)
        val_mask[300:350, :] = 1.0
    
    if device is not None:
        train_mask = train_mask.to(device)
        val_mask = val_mask.to(device)

    return img_var, gt, H, W, B, train_mask, val_mask


def get_noise(input_depth, method, shape):
    if method == "noise":
        return torch.randn((1, input_depth, shape[0], shape[1])).normal_()
    else:
        return torch.zeros((1, input_depth, shape[0], shape[1]))
