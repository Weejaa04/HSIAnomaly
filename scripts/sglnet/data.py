"""Data loading for SGLNet architecture."""

import torch
from torch.utils.data import Dataset
import numpy as np


class HSIPixelDataset(Dataset):
    """Dataset that treats each pixel as an independent sample."""

    def __init__(self, data, labels=None, mask=None):
        """
        Args:
            data: HSI data of shape (H, W, B)
            labels: Label array of shape (H, W) or None
            mask: Boolean mask of shape (H*W,) to select pixels
        """
        H, W, B = data.shape
        data_flat = data.reshape(-1, B)

        if mask is not None:
            data_flat = data_flat[mask]
            if labels is not None:
                labels = labels.flatten()[mask]

        self.pixels = data_flat
        self.labels = labels
        self.H = H
        self.W = W

    def __len__(self):
        return len(self.pixels)

    def __getitem__(self, idx):
        pixel = torch.from_numpy(self.pixels[idx]).float()

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return pixel, label
        else:
            return pixel


class HSIPatchDataset(Dataset):
    """Dataset that extracts small patches around each pixel for local context."""

    def __init__(self, data, labels=None, mask=None, patch_size=9):
        """
        Args:
            data: HSI data of shape (H, W, B)
            labels: Label array of shape (H, W) or None
            mask: Boolean mask of shape (H*W,) to select pixel centers
            patch_size: Size of patch to extract around each pixel
        """
        self.H, self.W, self.B = data.shape
        self.patch_size = patch_size
        self.pad = patch_size // 2

        data_padded = np.pad(
            data, ((self.pad, self.pad), (self.pad, self.pad), (0, 0)), mode="reflect"
        )
        self.data_padded = data_padded

        valid_coords = []
        if mask is not None:
            for i, m in enumerate(mask.flatten()):
                if m:
                    valid_coords.append(i)
        else:
            valid_coords = list(range(self.H * self.W))

        self.valid_coords = valid_coords

        if labels is not None:
            self.labels = labels.flatten()[mask]
        else:
            self.labels = None

    def __len__(self):
        return len(self.valid_coords)

    def __getitem__(self, idx):
        flat_idx = self.valid_coords[idx]
        row = flat_idx // self.W
        col = flat_idx % self.W

        r_center = row + self.pad
        c_center = col + self.pad

        patch = self.data_padded[
            r_center - self.pad : r_center + self.pad + 1,
            c_center - self.pad : c_center + self.pad + 1,
            :,
        ]

        patch_tensor = torch.from_numpy(patch).float()
        patch_tensor = patch_tensor.permute(2, 0, 1)

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return patch_tensor, label
        else:
            return patch_tensor
