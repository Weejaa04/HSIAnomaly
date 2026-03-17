import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class AdaConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, window_size):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.window_size = window_size
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size),
            requires_grad=True,
        )
        self.bias = nn.Parameter(
            torch.empty(
                out_channels,
            ),
            requires_grad=True,
        )

    def forward(self, x, l):
        B, C, H, W = x.shape
        window_size = self.window_size
        kernel_size = self.kernel_size
        if l is None:
            gathered_inp = F.unfold(
                x, (kernel_size, kernel_size), padding=(kernel_size - 1) // 2
            )
        else:
            p = (window_size - 1) // 2
            padded_inp = F.pad(x, (p, p, p, p), "reflect")
            l = F.pad(l, (p, p, p, p), "constant", 999)
            l = F.unfold(l, (window_size, window_size))
            min_idx = torch.topk(l, k=kernel_size**2, dim=1, largest=False).indices
            unfold_inp = (
                padded_inp.unfold(2, window_size, 1)
                .unfold(3, window_size, 1)
                .reshape(B, x.shape[1], H * W, window_size**2)
                .permute(0, 1, 3, 2)
            )
            gathered_inp = torch.gather(
                unfold_inp,
                dim=-2,
                index=min_idx.unsqueeze(1).repeat(1, x.shape[1], 1, 1),
            ).view(B, -1, H * W)

        out_unf = (
            gathered_inp.transpose(1, 2)
            .matmul(self.weight.view(self.weight.size(0), -1).t())
            .transpose(1, 2)
        )
        out = out_unf.view(B, -1, H, W) + self.bias.unsqueeze(0).repeat(B, 1).view(
            B, -1, 1, 1
        )
        return out, None


class SequentialMultiInput(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class LeakyReLUWrapper(nn.LeakyReLU):
    def __init__(self, negative_slope=0.01, inplace=False):
        super().__init__(negative_slope=negative_slope, inplace=inplace)

    def forward(self, input1, input2=None):
        output1 = super().forward(input1)
        return output1, input2


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class SuperADNetwork(nn.Module):
    def __init__(self, nch_in=189, nch_out=189, kernel_size=3, window_size=5):
        super(SuperADNetwork, self).__init__()
        in_channels = nch_in
        out_channels = nch_out
        dim = 32

        self.patch_embed = SequentialMultiInput(
            nn.Conv2d(in_channels, dim, 3, 1, 1, padding_mode="reflect"),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(dim, dim, 3, 1, 1, padding_mode="reflect"),
        )

        embed_dim = dim
        num_heads = 1
        mlp_ratio = 2
        depth = 1
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, mlp_ratio) for i in range(depth)]
        )

        self.norm = nn.LayerNorm(embed_dim)

        decoder_embed_dim = 32
        decoder_num_heads = 1
        decoder_depth = 1
        self.middle_conv = nn.Conv2d(
            dim, decoder_embed_dim, 3, stride=1, padding=1, padding_mode="reflect"
        )
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.decoder_blocks = nn.ModuleList(
            [
                TransformerBlock(decoder_embed_dim, decoder_num_heads, mlp_ratio)
                for i in range(decoder_depth)
            ]
        )

        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)

        self.window_size = window_size
        self.kernel_size = kernel_size

        self.smconv = SequentialMultiInput(
            AdaConv(decoder_embed_dim, decoder_embed_dim, kernel_size, window_size),
            LeakyReLUWrapper(negative_slope=0.1, inplace=True),
            AdaConv(decoder_embed_dim, decoder_embed_dim, kernel_size, window_size),
            LeakyReLUWrapper(negative_slope=0.1, inplace=True),
            AdaConv(decoder_embed_dim, decoder_embed_dim, kernel_size, window_size),
            LeakyReLUWrapper(negative_slope=0.1, inplace=True),
            AdaConv(decoder_embed_dim, decoder_embed_dim, kernel_size, window_size),
            LeakyReLUWrapper(negative_slope=0.1, inplace=True),
            AdaConv(decoder_embed_dim, decoder_embed_dim, kernel_size, window_size),
            LeakyReLUWrapper(negative_slope=0.1, inplace=True),
            AdaConv(decoder_embed_dim, decoder_embed_dim, kernel_size, window_size),
            LeakyReLUWrapper(negative_slope=0.1, inplace=True),
            AdaConv(decoder_embed_dim, decoder_embed_dim, kernel_size, window_size),
        )

        self.output_block = SequentialMultiInput(
            AdaConv(decoder_embed_dim, dim, kernel_size, window_size),
            LeakyReLUWrapper(negative_slope=0.1, inplace=True),
            AdaConv(dim, dim, kernel_size, window_size),
            LeakyReLUWrapper(negative_slope=0.1, inplace=True),
            AdaConv(dim, dim, kernel_size, window_size),
            LeakyReLUWrapper(negative_slope=0.1, inplace=True),
            AdaConv(dim, dim, kernel_size, window_size),
            LeakyReLUWrapper(negative_slope=0.1, inplace=True),
            AdaConv(dim, dim, kernel_size, window_size),
            LeakyReLUWrapper(negative_slope=0.1, inplace=True),
            AdaConv(dim, dim, kernel_size, window_size),
            LeakyReLUWrapper(negative_slope=0.1, inplace=True),
            AdaConv(dim, dim, kernel_size, window_size),
        )

        self.post_conv = SequentialMultiInput(
            nn.Conv2d(dim, dim, 3, 1, 1, padding_mode="reflect"),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(dim, out_channels, 1),
        )

        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, AdaConv):
            init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                fan_in, _ = init._calculate_fan_in_and_fan_out(m.weight)
                if fan_in != 0:
                    bound = 1 / math.sqrt(fan_in)
                    init.uniform_(m.bias, -bound, bound)

    def forward(self, x, segs, err_map):
        B, C, H, W = x.shape

        x = self.patch_embed(x)

        x_pooled = F.adaptive_avg_pool2d(x, (8, 8))
        x_suppool = x_pooled.flatten(2).transpose(1, 2)

        for blk in self.blocks:
            x_suppool = blk(x_suppool)
        x_suppool = self.norm(x_suppool)
        x_suppool = self.decoder_embed(x_suppool)

        for blk in self.decoder_blocks:
            x_suppool = blk(x_suppool)

        x_suppool = self.decoder_norm(x_suppool).transpose(1, 2)

        uppooled_x = F.interpolate(
            x_suppool.reshape(1, x_suppool.shape[1], 8, 8),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )

        x = self.middle_conv(x)

        if err_map is not None:
            out, l = self.smconv(x, err_map)
            x, l = self.output_block(out * uppooled_x, err_map)
        else:
            x, l = self.output_block(x * uppooled_x, err_map)

        return self.post_conv(x)


def OBPM(loss, segs, beta, alpha, th_idx, loss_type):
    if segs.dim() == 3:
        segs = segs.unsqueeze(1)
    if loss.dim() == 4 and loss.shape[1] != 1:
        loss = loss.mean(dim=1, keepdim=True)

    if loss_type == "l1":
        loss = loss
    elif loss_type == "l2":
        loss = loss * loss
    else:
        loss = torch.exp(beta * loss) / beta + loss * alpha

    plt_loss = loss.sum(dim=1, keepdim=False)

    N = segs.max().item()
    expanded_mask = segs.expand(-1, N, -1, -1)
    bool_mask = expanded_mask == torch.arange(1, N + 1).unsqueeze(0).unsqueeze(
        -1
    ).unsqueeze(-1).to(loss.device)

    B, _, H, W = loss.shape
    select_loss = loss.unsqueeze(1) * bool_mask.float()
    flat_select_loss = select_loss.reshape(B, N, -1)

    min_val = (
        torch.where(bool_mask.reshape(B, N, -1), flat_select_loss, torch.inf)
        .min(-1)
        .values
    )
    is_inf = torch.isinf(min_val)
    min_val = torch.where(is_inf, torch.tensor(0.0), min_val)

    sorted_flat_select_loss, _ = flat_select_loss.sort(dim=-1)
    mask = sorted_flat_select_loss == 0
    min_val_expanded = min_val.unsqueeze(-1).expand_as(sorted_flat_select_loss)
    sorted_flat_select_loss.masked_scatter_(mask, min_val_expanded[mask])

    diff1 = torch.diff(
        sorted_flat_select_loss,
        dim=-1,
        prepend=sorted_flat_select_loss.permute(2, 0, 1)[0].unsqueeze(-1),
    )

    K = 7
    kernel = torch.ones(1, 1, K).to(loss.device) / K
    acc_diff1 = F.conv1d(
        diff1.view(-1, 1, diff1.shape[-1]), kernel, stride=1, padding=(K - 1) // 2
    ).view(diff1.shape)

    max_idx = acc_diff1.argmax(dim=-1)
    th_loss = sorted_flat_select_loss.gather(-1, max_idx.unsqueeze(-1))

    ignore_idx = (H * W - max_idx) >= (bool_mask.sum((-1, -2)) * (1 - th_idx))
    th_loss[ignore_idx] = 1e4
    loss = torch.where(
        flat_select_loss <= th_loss, flat_select_loss, torch.tensor(0.0).to(loss.device)
    )
    plt_loss = loss.sum(dim=1, keepdim=False).reshape(H, W)

    total_loss = plt_loss.mean()

    return total_loss, plt_loss
