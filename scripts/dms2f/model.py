"""
DMS2F-HAD: Dual Mamba Spectral-Spatial Fusion for Hyperspectral Anomaly Detection.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# ─── Mamba imports / fallback ─────────────────────────────────────────────
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
    try:
        from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
    except ImportError:
        causal_conv1d_fn = causal_conv1d_update = None
    try:
        from mamba_ssm.ops.triton.selective_state_update import selective_state_update
    except ImportError:
        selective_state_update = None
    try:
        from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
    except ImportError:
        RMSNorm = layer_norm_fn = rms_norm_fn = None
    MAMBA_FULL = True
except ImportError:
    MAMBA_FULL = False


class MambaSimple(nn.Module):
    """Pure-PyTorch Mamba fallback when mamba_ssm CUDA ops are unavailable."""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, **kwargs):
        super().__init__()
        self.d_inner = int(expand * d_model)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                groups=self.d_inner, padding=d_conv - 1)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, **kwargs):
        B, L, _ = x.shape
        residual = x
        x = self.norm(x)
        xz = self.in_proj(x)
        x_part, z = xz.chunk(2, dim=-1)
        xc = self.conv1d(x_part.transpose(1, 2))[:, :, :L].transpose(1, 2)
        xc = F.silu(xc)
        y = xc * torch.sigmoid(z) + xc * self.D
        return self.out_proj(y) + residual


def make_mamba(d_model, d_state=64, d_conv=4, expand=2):
    """Factory: use mamba_ssm if available, fallback to MambaSimple."""
    if MAMBA_FULL:
        try:
            from mamba_ssm import Mamba
            return Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv,
                         expand=expand, use_fast_path=False)
        except Exception:
            pass
    return MambaSimple(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)


# ─── Mamba blocks ─────────────────────────────────────────────────────────
class Residual_SSMN(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class LayerNormalize_SSMN(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class mamba_block1(nn.Module):
    """Spatial Mamba block with residual + LayerNorm."""
    def __init__(self, dim, depth):
        super().__init__()
        self.layers = nn.ModuleList([
            Residual_SSMN(LayerNormalize_SSMN(dim, make_mamba(dim, d_state=64)))
            for _ in range(depth)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class mamba_block2(nn.Module):
    """Spectral Mamba block with residual + LayerNorm."""
    def __init__(self, dim, depth):
        super().__init__()
        self.layers = nn.ModuleList([
            Residual_SSMN(LayerNormalize_SSMN(dim, make_mamba(dim, d_state=64)))
            for _ in range(depth)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# ─── Utilities ────────────────────────────────────────────────────────────
def split_band(x, move_num, spec_num):
    """Split spectral dimension into overlapping segments."""
    b, c, h, w = x.shape
    slices = []
    for i in range(0, c, move_num):
        if i + spec_num > c:
            sl = x[:, c - spec_num:c, :, :]
        else:
            sl = x[:, i:i + spec_num, :, :]
        slices.append(sl)
    return torch.stack(slices, dim=1)


class RandomMasking(nn.Module):
    """In-model random masking to prevent identity mapping."""
    def __init__(self, mask_prob=0.5, mask_size=0.2, mode='full_spectrum'):
        super().__init__()
        self.mask_prob = mask_prob
        self.mask_size = mask_size
        self.mode = mode

    def forward(self, x):
        if not self.training or torch.rand(1) > self.mask_prob:
            return x
        B, C, H, W = x.shape
        mask = torch.ones_like(x)
        if self.mode == 'full_spectrum':
            for b in range(B):
                ph = max(1, int(H * self.mask_size))
                pw = max(1, int(W * self.mask_size))
                i = torch.randint(0, H - ph + 1, (1,)).item()
                j = torch.randint(0, W - pw + 1, (1,)).item()
                mask[b, :, i:i + ph, j:j + pw] = 0.
        elif self.mode == 'random_channels':
            num_mask = int(C * self.mask_size)
            idx = torch.randperm(C)[:num_mask]
            mask[:, idx, :, :] = 0.
        return x * mask


class LSSDecoderBlock(nn.Module):
    """Decoder block: Mamba (global) + Conv2d 3x3 + Conv2d 5x5 + fuse."""
    def __init__(self, channels, hss_depth=2):
        super().__init__()
        self.hss = mamba_block1(channels, hss_depth)
        self.local3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.local5 = nn.Conv2d(channels, channels, kernel_size=5, padding=2)
        self.fuse = nn.Conv2d(channels * 3, channels, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x):
        B, C, H, W = x.shape
        g = rearrange(x, 'b c h w -> b (h w) c')
        g = self.hss(g)
        g = rearrange(g, 'b (h w) c -> b c h w', h=H, w=W)
        l3 = self.local3(x)
        l5 = self.local5(x)
        cat = torch.cat([g, l3, l5], dim=1)
        return self.act(self.fuse(cat)) + x


# ─── Main Model ───────────────────────────────────────────────────────────
class AnomalyDetectionModel(nn.Module):
    """
    DMS2F-HAD: Dual Mamba Spectral-Spatial Fusion.

    mode: 'spatial'  — spatial branch only
          'spectral' — spectral branch only
          'full'     — dual branch + gated fusion
    """
    def __init__(self, in_channels, mode='full', dim=64, depth=1,
                 spec_num=12, spec_rate=0.5, spa_token=16):
        super().__init__()
        self.mode = mode
        self.spec_num = spec_num
        self.move_num = int(math.ceil(spec_num * spec_rate))
        self.spa_token = spa_token
        self.dim = dim

        self.preprocess = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )

        self.spatial_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )
        self.spatial_mamba = mamba_block1(dim, depth)

        self.dropout = nn.Dropout(0.1)
        num_patch = (
            math.floor((dim - (spec_num - self.move_num)) / self.move_num) +
            math.ceil(
                (((dim - (spec_num - self.move_num)) % self.move_num)
                 + (spec_num - self.move_num)) / self.move_num
            )
        )
        self.spe_token1 = nn.Sequential(
            nn.Conv3d(1, 1, (1, 1, 7), stride=(1, 1, 1), padding=(0, 0, 3)),
            nn.LayerNorm(dim), nn.GELU(),
        )
        self.spe_token2 = nn.Sequential(
            nn.Conv3d(1, 1, (1, 1, 3), stride=(1, 1, 1), padding=(0, 0, 1)),
            nn.LayerNorm(dim), nn.GELU(),
        )
        self.spectral_mamba = mamba_block2(spec_num, depth)
        self.nn2 = nn.Sequential(
            nn.Linear(spec_num * num_patch, dim),
            nn.LayerNorm(dim), nn.GELU(),
        )

        self.lss1 = LSSDecoderBlock(dim, hss_depth=2)
        self.lss2 = LSSDecoderBlock(dim, hss_depth=2)
        self.out_conv = nn.Conv2d(dim, in_channels, 1)

        self.use_random_mask = True
        self.random_mask = RandomMasking()

        self.fusion_fc = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1), nn.GELU(),
        )
        self.gate_conv = nn.Sequential(
            nn.Conv2d(2 * dim, 1, kernel_size=1), nn.Sigmoid(),
        )

    def forward(self, x):
        b, C, H, W = x.shape

        if self.use_random_mask:
            x = self.random_mask(x)

        feat = self.preprocess(x).permute(0, 2, 3, 1)

        spa_feat = feat.permute(0, 3, 1, 2)
        spatial_feat = self.spatial_conv(spa_feat)
        spatial_feat = self.spatial_mamba(
            rearrange(spatial_feat, 'b c h w -> b (h w) c')
        )
        spatial_feat = rearrange(spatial_feat,
                                  'b (h w) c -> b c h w', h=H, w=W)

        x_s = feat.unsqueeze(1)
        x_s = self.spe_token1(x_s) + self.spe_token2(x_s) + x_s
        x_s = self.dropout(x_s).squeeze(1).permute(0, 3, 1, 2)

        Patch_pool = nn.AvgPool2d((H, W)).to(x.device)
        x_s = Patch_pool(x_s)

        spec_slices = split_band(x_s, self.move_num, self.spec_num)
        bb, nn_seg, cc, hh, ww = spec_slices.shape
        spec_slices = rearrange(spec_slices, 'b n c h w -> (b h w) n c')

        spec_feat = self.spectral_mamba(spec_slices)
        spec_feat = rearrange(spec_feat,
                               '(b h w) n c -> b (n c) h w',
                               h=hh, w=ww).mean(-1)
        spec_feat = (
            self.nn2(spec_feat.squeeze(2))
            .unsqueeze(1).unsqueeze(1)
            .permute(0, 3, 1, 2)
        )
        spec_feat = spec_feat.view(b, self.dim, 1, 1).expand(b, self.dim, H, W)

        if self.mode == 'spatial':
            fused = self.fusion_fc(spatial_feat)
        elif self.mode == 'spectral':
            fused = self.fusion_fc(spec_feat)
        else:
            cat = torch.cat([spatial_feat, spec_feat], dim=1)
            gate = self.gate_conv(cat)
            fused_raw = gate * spatial_feat + (1. - gate) * spec_feat
            fused = self.fusion_fc(fused_raw)

        z = self.lss1(fused)
        z = self.lss2(z)
        out = self.out_conv(z)
        return out, fused

    @torch.no_grad()
    def anomaly_score_patch(self, x):
        """Reconstruction error per pixel (mean over bands)."""
        self.eval()
        recon, _ = self.forward(x)
        return ((x - recon) ** 2).mean(dim=1)
