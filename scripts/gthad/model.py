"""GT-HAD model architecture: Global-local Transformer for Hyperspectral Anomaly Detection."""

import torch
import torch.nn as nn


class Mlp(nn.Module):
    """MLP (FFN) block."""
    
    def __init__(self, in_features, hidden_features=None, 
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """Global-local Transformer Attention with AFB/BFB modes."""
    
    def __init__(self, dim, patch_size=3, patch_stride=3, attn_drop=0.):
        super().__init__()
        self.psize = patch_size
        self.pstride = patch_stride

        self.attn_drop = nn.Dropout(attn_drop)
        self.softmax = nn.Softmax(dim=-1)

        self.embed_dim = dim 
        self.hidden_dim = self.embed_dim // 2
        self.fc = nn.Linear(self.embed_dim, self.hidden_dim, bias=True)
        self.scale = (self.hidden_dim * self.psize ** 2) ** -0.5
        
        self.N = self.psize * self.pstride
        self.mask = torch.eye(self.N).cuda()
        self.mask[self.mask == 1] = -100.0

    def calculate_mask(self, block_idx=0, match_vec=None):
        """Calculate attention mask for AFB (All-to-First) or BFB (Block-to-First)."""
        B = block_idx.size(0)
        cur_match = torch.index_select(match_vec, dim=0, index=block_idx).squeeze()

        if cur_match.sum() == 0:
            mask = self.mask
        else:
            mask = self.mask.unsqueeze(0).repeat(B, 1, 1)
            mask[cur_match == 1] = 0

        return mask

    def attn_cal(self, attn, mask, v, shape):
        """Calculate attention output."""
        B, H, W, C = shape
        attn = attn + mask
        attn = self.softmax(attn)

        x_attn = (attn @ v)
        x_back = x_attn.view(B, self.pstride, self.pstride, self.psize, self.psize, C)
        x = x_back.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, C)

        return x

    def forward(self, x, block_idx=0, match_vec=None):
        """
        Args:
            x: (B, H, W, C) where H=W=patch_size*patch_stride
            block_idx: indices to select blocks
            match_vec: gating vector for AFB/BFB selection
        """
        B, H, W, C = x.shape
        N = self.N
        P = self.psize ** 2

        x_view = x.view(B, self.pstride, self.psize, self.pstride, self.psize, C)
        x_fc = x_view.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, P, C) 
        x_q = self.fc(x_fc).view(B, N, -1)
        attn = x_q @ x_q.transpose(-2, -1)
        attn = attn * self.scale
        attn = self.attn_drop(attn)

        v = x_view.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, N, P * C)
        mask = self.calculate_mask(block_idx=block_idx, match_vec=match_vec)
        x = self.attn_cal(attn, mask, v, [B, H, W, C])

        return x


class TransformerBlock(nn.Module):
    """Transformer block with Global-local Attention + FFN."""
    
    def __init__(self, dim, patch_size=3, patch_stride=3, mlp_ratio=4., 
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_drop=0., drop=0.):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, patch_size=patch_size, patch_stride=patch_stride, attn_drop=attn_drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, block_idx=0, match_vec=None):
        x = x + self.attn(self.norm1(x), block_idx=block_idx, match_vec=match_vec)
        x = x + self.mlp(self.norm2(x))
        return x


class Net(nn.Module):
    """GT-HAD: Global-local Transformer for Hyperspectral Anomaly Detection."""
    
    def __init__(self, in_chans=224, embed_dim=64, patch_size=3, 
                 patch_stride=3, mlp_ratio=2.0, attn_drop=0.0, drop=0.0,
                 depth=4, norm_layer=nn.LayerNorm):
        super().__init__()
        
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        
        # Input projection
        self.proj = nn.Linear(in_chans, embed_dim)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                patch_size=patch_size,
                patch_stride=patch_stride,
                mlp_ratio=mlp_ratio,
                attn_drop=attn_drop,
                drop=drop,
                norm_layer=norm_layer
            )
            for _ in range(depth)
        ])
        
        # Output projection back to input dimension
        self.norm = norm_layer(embed_dim)
        self.out_proj = nn.Linear(embed_dim, in_chans)

    def forward(self, x, block_idx=0, match_vec=None):
        """
        Args:
            x: (B, C, H, W) block from HSI
            block_idx: indices for blocks in batch
            match_vec: Content Matching Module gating vector
        Returns:
            reconstructed: (B, C, H, W) reconstructed block
        """
        B, C, H, W = x.shape
        
        # Reshape to (B, H, W, C)
        x = x.permute(0, 2, 3, 1).contiguous()
        
        # Project input
        x = self.proj(x)  # (B, H, W, embed_dim)
        
        # Transformer blocks with GTB (Global-local Transformer Block)
        for block in self.blocks:
            x = block(x, block_idx=block_idx, match_vec=match_vec)
        
        # Normalize and project back
        x = self.norm(x)
        x = self.out_proj(x)  # (B, H, W, in_chans)
        
        # Reshape back to (B, C, H, W)
        x = x.permute(0, 3, 1, 2).contiguous()
        
        return x
