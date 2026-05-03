import os
import numpy as np
import torch
from torch.utils.data import Dataset
import spectral
from skimage.segmentation import slic


def load_hsi_data(data_path):
    img = spectral.open_image(data_path)
    data = img.load()
    return np.array(data, dtype=np.float32)


def load_label(label_path):
    return np.load(label_path)


def calibrate_hsi(data, white_ref_path, dark_ref_path):
    white_ref = np.array(spectral.open_image(white_ref_path).load(), dtype=np.float32)
    dark_ref = np.array(spectral.open_image(dark_ref_path).load(), dtype=np.float32)

    white_profile = white_ref.mean(axis=0)
    dark_profile = dark_ref.mean(axis=0)

    calibrated = (data - dark_profile) / (white_profile - dark_profile + 1e-8)
    calibrated = np.clip(calibrated, 0, 1)

    return calibrated


def compute_superpixels(image, n_segments=500, compactness=10):
    H, W, C = image.shape
    image_norm = (image - image.min()) / (image.max() - image.min() + 1e-8)
    segments = slic(
        image_norm,
        n_segments=n_segments,
        compactness=compactness,
        channel_axis=-1,
        start_label=1,
    )
    return segments.astype(np.int64)


class HSISuperpixelDataset(Dataset):
    def __init__(self, data, gt, segments, train_mask=None, val_mask=None):
        self.data = torch.from_numpy(data.transpose(2, 0, 1)).float()
        self.gt = torch.from_numpy(gt).float()
        self.segments = torch.from_numpy(segments).unsqueeze(0).long()
        self.train_mask = train_mask
        self.val_mask = val_mask

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return {
            "hsi": self.data,
            "gt": self.gt,
            "segs": self.segments,
            "train_mask": self.train_mask,
            "val_mask": self.val_mask,
        }


def get_random_train_val_masks(H, W, seed=42):
    """Generate random masks for training and validation.
    
    Random Sampling: 37.5% train, 12.5% val, remaining unlabeled.
    
    Args:
        H, W: Image height and width
        seed: Random seed for reproducibility
    
    Returns:
        train_mask: (1, 1, H, W) tensor
        val_mask: (1, 1, H, W) tensor
    """
    np.random.seed(seed)
    rand_map = np.random.rand(H, W)
    
    train_mask = rand_map < 0.375
    val_mask = (rand_map >= 0.375) & (rand_map < 0.500)
    
    train_mask = torch.from_numpy(train_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    val_mask = torch.from_numpy(val_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    
    return train_mask, val_mask


def get_food_data(food_type, base_dir, n_segments=500, compactness=10, split_method="spatial"):
    base_path = os.path.join(base_dir, food_type)

    train_data_path = os.path.join(base_path, "Train", "data.hdr")
    train_white_path = os.path.join(base_path, "Train", "WHITEREF.hdr")
    train_dark_path = os.path.join(base_path, "Train", "DARKREF.hdr")

    train_data = load_hsi_data(train_data_path)
    train_data = calibrate_hsi(train_data, train_white_path, train_dark_path)

    H, W, B = train_data.shape

    # Generate masks based on split method
    if split_method == "random":
        train_mask, val_mask = get_random_train_val_masks(H, W, seed=42)
    else:  # spatial (guillotine)
        train_mask, val_mask = get_spatial_train_val_masks(H, W)

    segments = compute_superpixels(
        train_data, n_segments=n_segments, compactness=compactness
    )

    test_data_path = os.path.join(base_path, "Test", "data.hdr")
    test_white_path = os.path.join(base_path, "Test", "WHITEREF.hdr")
    test_dark_path = os.path.join(base_path, "Test", "DARKREF.hdr")
    test_label_path = os.path.join(base_path, "Test", "label.npy")

    test_data = load_hsi_data(test_data_path)
    test_data = calibrate_hsi(test_data, test_white_path, test_dark_path)
    test_labels = load_label(test_label_path)

    train_data_tensor = (
        torch.from_numpy(train_data.transpose(2, 0, 1)).float().unsqueeze(0)
    )
    test_data_tensor = (
        torch.from_numpy(test_data.transpose(2, 0, 1)).float().unsqueeze(0)
    )
    test_labels_tensor = torch.from_numpy(test_labels).float()

    segments_tensor = torch.from_numpy(segments).unsqueeze(0).long()

    return (
        train_data_tensor,
        test_data_tensor,
        test_labels_tensor,
        segments_tensor,
        H,
        W,
        B,
        train_mask,
        val_mask,
    )


def get_spatial_train_val_masks(H=400, W=512):
    train_mask = np.zeros((H, W), dtype=np.float32)
    val_mask = np.zeros((H, W), dtype=np.float32)

    train_mask[50:200, :] = 1.0
    val_mask[300:350, :] = 1.0

    train_mask = torch.from_numpy(train_mask).unsqueeze(0).unsqueeze(0)
    val_mask = torch.from_numpy(val_mask).unsqueeze(0).unsqueeze(0)

    return train_mask, val_mask
