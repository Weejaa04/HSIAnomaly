import torch
import torch.nn as nn
import torch.nn.functional as F


def wasserstein_distance(features_real, features_fake):
    """Calculate the Wasserstein distance"""
    w_distance = torch.mean(torch.abs(features_real - features_fake))
    return w_distance


def l2_normalize(tensor, dim=-1, eps=1e-12):
    """L2 normalization is applied to a given tensor"""
    min_val = tensor.min()
    if min_val < 0:
        tensor = tensor - min_val + eps

    norm = torch.norm(tensor, p=2, dim=dim, keepdim=True)
    return tensor / (norm + eps)


class MLP(nn.Module):
    def __init__(
        self, in_features, hidden_features=None, act_layer=nn.GELU, drop_ratio=0.0
    ):
        super().__init__()
        out_features = in_features
        hidden_features = hidden_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop_ratio)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self, dim, num_heads=4, patch_size=3, patch_grid=3, attn_drop_ratio=0.0
    ):
        super().__init__()
        self.patch_size = patch_size
        self.patch_grid = patch_grid
        self.num_heads = num_heads
        self.attn_drop_ratio = nn.Dropout(attn_drop_ratio)
        self.softmax = nn.Softmax(dim=-1)
        self.hidden_dim = dim // num_heads
        self.proj = nn.Linear(dim, dim, bias=True)
        self.scale = (self.hidden_dim * self.patch_size**2) ** -0.5
        self.N = self.patch_grid * self.patch_grid
        self.qkv = nn.Linear(dim, dim * 3, bias=False)

    def forward(self, x):
        B, H, W, C = x.shape
        N = self.N
        P = self.patch_size**2

        x_view = x.view(
            B, self.patch_grid, self.patch_size, self.patch_grid, self.patch_size, C
        )
        x_fc = x_view.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, N, P, C)
        qkv = (
            self.qkv(x_fc)
            .reshape(B, N, 3, self.num_heads, P, C // self.num_heads)
            .permute(2, 0, 3, 1, 4, 5)
        )
        qkv = qkv.view(3, B, self.num_heads, N, -1)
        q, k, v = qkv[0], qkv[1], qkv[2]
        k = k.transpose(-2, -1)
        attn = q @ k
        attn = attn * self.scale
        attn = self.softmax(attn)
        x_attn = (
            (attn @ v)
            .transpose(1, 2)
            .reshape(B, N, P, C // self.num_heads, self.num_heads)
        )
        x_attn = x_attn.view(B, N, P, C)
        x_attn = self.proj(x_attn)
        x_back = x_attn.view(
            B, self.patch_grid, self.patch_grid, self.patch_size, self.patch_size, C
        )
        x = x_back.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, C)
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        patch_size=3,
        patch_grid=3,
        mlp_ratio=4.0,
        num_heads=4,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        attn_drop_ratio=0.0,
        drop_ratio=0.0,
    ):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            patch_size=patch_size,
            patch_grid=patch_grid,
            attn_drop_ratio=attn_drop_ratio,
        )
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop_ratio=drop_ratio,
        )

    def forward(self, x):
        B, H, W, C = x.shape
        x1 = x.view(B, H * W, C)
        x_norm1 = self.norm1(x1)
        x_norm1 = x_norm1.view(B, H, W, C)
        x_attn = self.attn(x_norm1)

        x = x_attn.view(B, H * W, C)
        x_norm2 = self.norm2(x)
        x_mlp = x + self.mlp(x_norm2)
        x = x_mlp.view(B, H, W, C)

        return x


class OTADNet(nn.Module):
    def __init__(
        self,
        input_channels=3,
        embedding_dim=64,
        patch_size=3,
        patch_grid=3,
        num_heads=4,
        mlp_ratio=2.0,
        attn_drop_ratio=0.0,
        drop_ratio=0.0,
    ):
        super().__init__()
        self.conv_head = nn.Conv2d(
            input_channels, embedding_dim, kernel_size=3, stride=1, padding=1
        )
        self.attn_layer = Transformer(
            embedding_dim,
            patch_size=patch_size,
            patch_grid=patch_grid,
            mlp_ratio=mlp_ratio,
            num_heads=num_heads,
            attn_drop_ratio=attn_drop_ratio,
            drop_ratio=drop_ratio,
        )
        self.conv_middle = nn.Conv2d(
            embedding_dim, embedding_dim, kernel_size=3, stride=1, padding=1
        )
        self.conv_tail = nn.Conv2d(
            embedding_dim, input_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        x_conv = self.conv_head(x)
        x_conv_reshaped = x_conv.permute(0, 2, 3, 1).contiguous()
        x_attn = self.attn_layer(x_conv_reshaped)
        x_attn_reshaped = x_attn.permute(0, 3, 1, 2).contiguous()
        x_middle = self.conv_middle(x_attn_reshaped)
        x_reconstructed = self.conv_tail(x_middle)

        x_normalized = l2_normalize(x.view(x.size(0), -1))
        x_reconstructed_normalized = l2_normalize(
            x_reconstructed.view(x_reconstructed.size(0), -1)
        )

        x_conv_normalized = l2_normalize(x_conv.view(x_conv.size(0), -1))
        x_middle_normalized = l2_normalize(x_middle.view(x_middle.size(0), -1))

        w_distance1 = wasserstein_distance(x_normalized, x_reconstructed_normalized)
        w_distance2 = wasserstein_distance(x_conv_normalized, x_middle_normalized)

        w = w_distance1 + w_distance2

        return x_reconstructed, w


class BlockGeneration(nn.Module):
    def __init__(self, block_size=15, stride=5):
        super().__init__()
        self.block_size = block_size
        self.stride = stride

    def pixel_padding(self, hsis, kernel_sizes, strides, rates=(1, 1)):
        assert len(hsis.size()) == 4
        batch_size, band, height, width = hsis.size()
        out_height = (height + strides[0] - 1) // strides[0]
        out_width = (width + strides[1] - 1) // strides[1]
        effective_k_row = (kernel_sizes[0] - 1) * rates[0] + 1
        effective_k_col = (kernel_sizes[1] - 1) * rates[1] + 1
        padding_rows = max(0, (out_height - 1) * strides[0] + effective_k_row - height)
        padding_cols = max(0, (out_width - 1) * strides[1] + effective_k_col - width)

        padding_top = int(padding_rows // 2)
        padding_bottom = padding_rows - padding_top
        padding_left = int(padding_cols // 2)
        padding_right = padding_cols - padding_left

        paddings = (padding_left, padding_right, padding_top, padding_bottom)
        hsis = F.pad(hsis, paddings, mode="replicate")

        return hsis, paddings

    def extract_image_blocks(self, hsis, kernel_sizes, strides):
        images, paddings = self.pixel_padding(hsis, kernel_sizes, strides)
        unfold = torch.nn.Unfold(kernel_size=kernel_sizes, padding=0, stride=strides)
        blocks = unfold(images)
        return blocks, paddings

    def forward(self, x):
        _, band, height, width = x.size()
        blocks, paddings = self.extract_image_blocks(
            x,
            kernel_sizes=[self.block_size, self.block_size],
            strides=[self.stride, self.stride],
        )
        blocks = blocks.squeeze(0).permute(1, 0)
        blocks = blocks.view(blocks.size(0), band, self.block_size, self.block_size)
        return blocks, blocks, paddings


class BlockRestore(nn.Module):
    def __init__(self, block_size=15, stride=5):
        super().__init__()
        self.block_size = block_size
        self.stride = stride

    def forward(self, x, paddings, height, width):
        num_blocks = x.size(0)
        blocks_reshaped = x.view(num_blocks, -1).permute(1, 0).unsqueeze(0)

        block_size_height = (
            height + 2 * paddings[2] - self.block_size
        ) / self.stride + 1
        block_size_width = (width + 2 * paddings[0] - self.block_size) / self.stride + 1

        if block_size_height * block_size_width != num_blocks:
            pad = [paddings[3], paddings[1]]
        else:
            pad = [paddings[2], paddings[0]]

        original_image = F.fold(
            blocks_reshaped,
            (height, width),
            (self.block_size, self.block_size),
            padding=pad,
            stride=self.stride,
        )

        overlapping_mask = torch.ones_like(original_image)
        mask_unfold = F.unfold(
            overlapping_mask,
            (self.block_size, self.block_size),
            padding=pad,
            stride=self.stride,
        )
        fold_mask = F.fold(
            mask_unfold,
            (height, width),
            (self.block_size, self.block_size),
            padding=pad,
            stride=self.stride,
        )

        out = original_image / fold_mask
        return out


class OTADDataset(torch.utils.data.Dataset):
    def __init__(self, data, block_size=15, stride=5):
        super().__init__()
        self.data_processer = BlockGeneration(block_size=block_size, stride=stride)
        self.gt_blocks, self.input_blocks, self.padding = self.data_processer(data)

    def __getitem__(self, index):
        block_gt = self.gt_blocks[index]
        block_input = self.input_blocks[index]
        return {"block_gt": block_gt, "block_input": block_input}

    def __len__(self):
        return self.gt_blocks.size(0)


def hyper_norm(data):
    """Normalize the input data to [0, 1]."""
    data = data.astype(float)
    min_val = data.min()
    max_val = data.max()
    norm = data - min_val
    if max_val == min_val:
        norm = data * 0
    else:
        norm /= max_val - min_val
    return norm
