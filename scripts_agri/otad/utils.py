"""AgriFood-adapted data loading for OT-AD (trains on Train/ cube, tests on Test/ cube)."""
import numpy as np
import torch

SEED_DICT = {
    "Almond": 42,
    "Pistachio": 42,
    "GarlicStems": 42,
    "AgriFood": 42,
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
    """Calibrate HSI data using white-dark correction (identity for AgriFood)."""
    import spectral

    white = np.array(spectral.open_image(white_ref_path).load(), dtype=np.float32).mean(
        axis=0
    )
    dark = np.array(spectral.open_image(dark_ref_path).load(), dtype=np.float32).mean(
        axis=0
    )
    return np.clip((data - dark) / (white - dark + 1e-8), 0, 1)


def get_spatial_train_val_mask(H=1000, W=900):
    """Spatial guillotine masks scaled by image height (H=400 -> Y[50:200]/Y[300:350])."""
    train_y_start, train_y_end = round(0.125 * H), round(0.5 * H)
    val_y_start, val_y_end = round(0.75 * H), round(0.875 * H)

    train_mask = np.zeros((H, W), dtype=bool)
    train_mask[train_y_start:train_y_end, :] = True

    val_mask = np.zeros((H, W), dtype=bool)
    val_mask[val_y_start:val_y_end, :] = True

    return train_mask, val_mask


def get_random_train_val_mask(H, W, seed=42):
    """Random sampling: 37.5% train, 12.5% val, remaining unlabeled."""
    np.random.seed(seed)
    rand_map = np.random.rand(H, W)

    train_mask = rand_map < 0.375
    val_mask = (rand_map >= 0.375) & (rand_map < 0.500)

    return train_mask, val_mask


def get_food_data(food_type: str = "AgriFood", base_dir: str = "AgriFood/Dataset", device=None, split_method: str = "spatial"):
    """AgriFood protocol: train on the Train/ cube, infer on the Test/ cube.

    Returns:
        img_train: (1, B, Ht, Wt) training cube tensor (Normal_3)
        img_var: (1, B, H, W) test cube tensor (anomaly scene)
        gt: (H, W) binary ground truth from Test/label.npy
        H, W, B: test cube spatial/spectral dims
        train_mask, val_mask: scaled guillotine masks over the train cube
    """
    base = f"{base_dir}/{food_type}"

    # Train cube (All-Normal: UseCase_1_(Avoine1)_Normal_3)
    train_data = calibrate_hsi(
        load_hsi_data(f"{base}/Train/data.hdr"),
        f"{base}/Train/WHITEREF.hdr",
        f"{base}/Train/DARKREF.hdr",
    )
    Ht, Wt = train_data.shape[:2]
    train_np = train_data.transpose(2, 0, 1)
    train_np = (train_np - train_np.min()) / (train_np.max() - train_np.min() + 1e-8)
    img_train = torch.from_numpy(train_np).float()
    if device is not None:
        img_train = img_train.to(device)
    img_train = img_train.unsqueeze(0)

    # Test cube (anomaly scene) + ground truth
    test_data = calibrate_hsi(
        load_hsi_data(f"{base}/Test/data.hdr"),
        f"{base}/Test/WHITEREF.hdr",
        f"{base}/Test/DARKREF.hdr",
    )
    gt_raw = np.load(f"{base}/Test/label.npy")

    H, W, B = test_data.shape
    gt = gt_raw != 2

    from scripts.otad.model import hyper_norm

    img_np = test_data.transpose(2, 0, 1)
    img_np = hyper_norm(img_np)

    img_var = torch.from_numpy(img_np).float().unsqueeze(0)
    if device is not None:
        img_var = img_var.to(device)

    # Scaled spatial guillotine masks over the train cube
    if split_method == "random":
        train_mask, val_mask = get_random_train_val_mask(Ht, Wt, seed=42)
    else:
        train_mask, val_mask = get_spatial_train_val_mask(Ht, Wt)

    return img_train, img_var, gt, H, W, B, train_mask, val_mask