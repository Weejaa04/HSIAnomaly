import os
import numpy as np
import torch
import random
from sklearn.metrics import roc_auc_score


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def map01(img):
    img_01 = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img_01


def get_auc(HSI_old, HSI_new, gt):
    n_row, n_col, n_band = HSI_old.shape
    n_pixels = n_row * n_col

    img_olds = np.reshape(HSI_old, (n_pixels, n_band), order="F")
    img_news = np.reshape(HSI_new, (n_pixels, n_band), order="F")
    sub_img = img_olds - img_news

    detectmap = np.linalg.norm(sub_img, ord=2, axis=1, keepdims=True) ** 2
    detectmap = detectmap / n_band

    detectmap = map01(detectmap)

    label = np.reshape(gt, (n_pixels, 1), order="F")
    # Convert to binary labels: 2 is normal (0), everything else is anomaly (1)
    binary_label = (label != 2).astype(int).flatten()
    try:
        auc = roc_auc_score(binary_label, detectmap.flatten())
    except:
        auc = 0.0
    detectmap = np.reshape(detectmap, (n_row, n_col), order="F")

    return auc, detectmap


def TensorToHSI(img):
    HSI = img.squeeze().cpu().data.numpy().transpose((1, 2, 0))
    return HSI


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
            if self.counter >= self.patience:
                self.early_stop = True


WEIGHT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "weights", "SuperAD")


def save_weights(model, food_type: str):
    os.makedirs(WEIGHT_DIR, exist_ok=True)
    weight_path = os.path.join(WEIGHT_DIR, f"{food_type}.pt")
    torch.save(model.state_dict(), weight_path)
    print(f"Weights saved to {weight_path}")


def load_weights(model, food_type: str, device="cpu"):
    weight_path = os.path.join(WEIGHT_DIR, f"{food_type}.pt")
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location=device))
        print(f"Weights loaded from {weight_path}")
        return True
    else:
        print(f"No weights found at {weight_path}")
        return False


def weights_exist(food_type: str) -> bool:
    weight_path = os.path.join(WEIGHT_DIR, f"{food_type}.pt")
    return os.path.exists(weight_path)


def get_spatial_train_val_masks(H=400, W=512):
    train_mask = np.zeros((H, W), dtype=np.float32)
    val_mask = np.zeros((H, W), dtype=np.float32)

    train_mask[50:200, :] = 1.0
    val_mask[300:350, :] = 1.0

    train_mask = torch.from_numpy(train_mask).unsqueeze(0).unsqueeze(0)
    val_mask = torch.from_numpy(val_mask).unsqueeze(0).unsqueeze(0)

    return train_mask, val_mask
