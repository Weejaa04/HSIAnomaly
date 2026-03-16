"""Block operations for GT-HAD: embedding, folding, and searching."""

import torch
import torch.nn.functional as F
import torch.nn as nn


class BlockEmbedding(nn.Module):
    """Extract sliding window blocks from HSI data."""
    
    def __init__(self, wsize=15, wstride=5):
        super(BlockEmbedding, self).__init__()
        self.ksize = wsize
        self.stride = wstride

    def same_padding(self, images, ksizes, strides, rates=(1, 1)):
        """Apply same padding to images before unfold."""
        assert len(images.size()) == 4
        batch_size, channel, rows, cols = images.size()
        out_rows = (rows + strides[0] - 1) // strides[0]
        out_cols = (cols + strides[1] - 1) // strides[1]
        effective_k_row = (ksizes[0] - 1) * rates[0] + 1
        effective_k_col = (ksizes[1] - 1) * rates[1] + 1
        padding_rows = max(0, (out_rows - 1) * strides[0] + effective_k_row - rows)
        padding_cols = max(0, (out_cols - 1) * strides[1] + effective_k_col - cols)
        
        padding_top = int(padding_rows / 2.)
        padding_left = int(padding_cols / 2.)
        padding_bottom = padding_rows - padding_top
        padding_right = padding_cols - padding_left
        paddings = (padding_left, padding_right, padding_top, padding_bottom)
        images = F.pad(images, paddings, mode='replicate')
        
        return images, paddings

    def extract_image_blocks(self, images, ksizes, strides):
        """Extract overlapping blocks from image using unfold."""
        assert len(images.size()) == 4
        images, paddings = self.same_padding(images, ksizes, strides)
        unfold = torch.nn.Unfold(kernel_size=ksizes,
                                 padding=0,
                                 stride=strides)
        blocks = unfold(images)
        return blocks, paddings

    def forward(self, x):
        """
        Extract blocks from HSI.
        Args:
            x: (batch=1, bands, height, width)
        Returns:
            blocks: (num_blocks, bands, ksize, ksize)
            paddings: padding info for later reconstruction
        """
        _, band, row, col = x.size()
        blocks, paddings = self.extract_image_blocks(
            x, ksizes=[self.ksize, self.ksize],
            strides=[self.stride, self.stride]
        )
        blocks = blocks.squeeze(0).permute(1, 0)
        blocks = blocks.view(blocks.size(0), band, self.ksize, self.ksize)
        return blocks, blocks, paddings


class BlockFold(nn.Module):
    """Reconstruct HSI from overlapping blocks."""
    
    def __init__(self, wsize=15, wstride=5):
        super(BlockFold, self).__init__()
        self.ksize = wsize
        self.stride = wstride

    def forward(self, x, paddings, row, col):
        """
        Fold blocks back to image using averaging in overlap regions.
        Args:
            x: (num_blocks, channels, ksize, ksize)
            paddings: (left, right, top, bottom)
            row, col: original image dimensions
        Returns:
            reconstructed: (1, channels, height, width)
        """
        num = x.size(0)
        back = x.view(num, -1).permute(1, 0).unsqueeze(0)
        
        block_size1 = (row + 2 * paddings[2] - (self.ksize - 1) - 1) / self.stride + 1
        block_size2 = (col + 2 * paddings[0] - (self.ksize - 1) - 1) / self.stride + 1
        
        if block_size1 * block_size2 != num:
            pad = [paddings[3], paddings[1]]
        else:
            pad = [paddings[2], paddings[0]]

        ori = F.fold(back, (row, col), (self.ksize, self.ksize), 
                     padding=pad, stride=self.stride)
        
        # Average overlapping regions
        tmp = torch.ones_like(ori)
        tmp_unfold = F.unfold(tmp, (self.ksize, self.ksize), 
                              padding=pad, stride=self.stride)
        fold_mask = F.fold(tmp_unfold, (row, col), (self.ksize, self.ksize), 
                          padding=pad, stride=self.stride)
        out = ori / fold_mask

        return out


class BlockSearch(nn.Module):
    """Content Matching Module (CMM) for GT-HAD."""
    
    def __init__(self, x_ori, wsize=15, wstride=5):
        super(BlockSearch, self).__init__()
        self.ksize = wsize
        self.stride = wstride
        
        self.block_embedding = BlockEmbedding(wsize=self.ksize, wstride=self.stride)
        block_query, _ = self.block_embedding.extract_image_blocks(
            x_ori, ksizes=[self.ksize, self.ksize], 
            strides=[self.stride, self.stride]
        )
        self.block_query = block_query.squeeze(0).permute(1, 0)

    def matrix_get_dis(self, x1, x2):
        """
        Compute pairwise Euclidean distances (vectorized with memory-aware batching).
        
        Phase 1 Optimization: Use torch.cdist for vectorization, but batch process
        to avoid memory overflow on large block counts.
        
        Previous: O(N) loops → N individual CUDA launches + sync points (slow)
        New: Batched cdist → single kernel per batch (fast + memory safe)
        
        Args:
            x1: (num_blocks_key, flat_features) - reconstructed blocks
            x2: (num_blocks_query, flat_features) - original blocks
        
        Returns:
            dis_map: (num_blocks_key, num_blocks_query) distance matrix
        """
        batch_size = 256  # Process 256 blocks at a time
        num_blocks = x1.size(0)
        
        # Process in batches to avoid allocating massive distance matrices
        dis_list = []
        for i in range(0, num_blocks, batch_size):
            batch_end = min(i + batch_size, num_blocks)
            x1_batch = x1[i:batch_end]
            
            # Compute distances for this batch
            dis_batch = torch.cdist(x1_batch, x2, p=2)
            dis_list.append(dis_batch)
        
        # Concatenate all batches
        dis_map = torch.cat(dis_list, dim=0)
        return dis_map

    def forward(self, x, match_vec, idx):
        """
        Find most similar blocks and update match vector.
        
        Phase 3 Optimization: Detach all computations to prevent gradient graph buildup.
        
        Args:
            x: reconstructed HSI (detached for CMM phase)
            match_vec: vector tracking which blocks matched
            idx: indices for blocks
        Returns:
            updated match_vec
        """
        block_key, _ = self.block_embedding.extract_image_blocks(
            x, ksizes=[self.ksize, self.ksize], 
            strides=[self.stride, self.stride]
        )
        block_key = block_key.squeeze(0).permute(1, 0)
        num = block_key.size(0)
        
        # Phase 3: Detach all computations to prevent gradient graph buildup
        # and enable memory release after CMM search
        dis_map = self.matrix_get_dis(block_key.detach(), self.block_query.detach())
        _, index = torch.topk(dis_map, 1, dim=1, largest=False, sorted=True)
        index = index.squeeze()
        
        flag = (index == idx).int()
        match_vec[flag == 1] = 1

        return match_vec
