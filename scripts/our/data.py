"""Data loading for 'our' architecture."""
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
