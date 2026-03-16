"""Data loading for GT-HAD."""

import torch
import torch.utils.data as data
import numpy as np
from .block import BlockEmbedding


class DatasetHsi(data.Dataset):
    """HSI dataset with sliding window blocks for GT-HAD."""
    
    def __init__(self, hsi_data, wsize=15, wstride=5, spatial_mask=None):
        """
        Args:
            hsi_data: (batch=1, bands, height, width)
            wsize: window/block size
            wstride: window stride
            spatial_mask: Optional (H, W) boolean mask. Only blocks with centers 
                         in the masked region are included. If None, all blocks used.
        """
        super(DatasetHsi, self).__init__()
        self.data_processer = BlockEmbedding(wsize=wsize, wstride=wstride)
        self.block_gt, self.block_input, self.padding = self.data_processer(hsi_data)
        
        # Apply spatial masking if provided
        self.valid_indices = None
        if spatial_mask is not None:
            self.valid_indices = self._get_valid_block_indices(
                hsi_data.shape[2], hsi_data.shape[3], wsize, wstride, spatial_mask
            )
            # Filter blocks
            self.block_gt = self.block_gt[self.valid_indices]
            self.block_input = self.block_input[self.valid_indices]
    
    def _get_valid_block_indices(self, H, W, wsize, wstride, spatial_mask):
        """
        Get indices of blocks whose centers fall within the spatial mask.
        
        Args:
            H, W: Image height and width
            wsize: Block size
            wstride: Block stride
            spatial_mask: (H, W) boolean array
        
        Returns:
            valid_indices: Array of block indices to keep
        """
        half_p = wsize // 2
        valid_y = np.arange(0 + half_p, H - half_p, wstride)
        valid_x = np.arange(0 + half_p, W - half_p, wstride)
        
        yy, xx = np.meshgrid(valid_y, valid_x, indexing='ij')
        centers_y = yy.ravel()
        centers_x = xx.ravel()
        
        # Check which centers fall within the spatial mask
        valid_mask = spatial_mask[centers_y, centers_x]
        valid_indices = np.where(valid_mask)[0]
        
        return torch.from_numpy(valid_indices).long()

    def __getitem__(self, index):
        block_gt = self.block_gt[index]
        block_input = self.block_input[index]
        return {
            'block_gt': block_gt, 
            'block_input': block_input, 
            'index': index
        }

    def __len__(self):
        return self.block_gt.size(0)

