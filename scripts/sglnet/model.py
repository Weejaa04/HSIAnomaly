"""
SGLNet Model for HSI Anomaly Detection.

Adapts the SGLNet architecture from ../SGLNet/ for pixel-level anomaly detection
using patch-based reconstruction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=1),
            nn.BatchNorm2d(in_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, kernel_size=1),
        )

    def forward(self, x):
        return x + self.body(x)


class LocalBranch(nn.Module):
    def __init__(self, in_ch, num_module=2):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.BatchNorm2d(in_ch),
            nn.LeakyReLU(inplace=True),
            *[ResidualBlock(in_ch) for _ in range(num_module)],
        )
        self.Conv_out = nn.Conv2d(
            in_ch, in_ch, kernel_size=3, padding=1, padding_mode="reflect"
        )

    def forward(self, x):
        out = self.body(x)
        return self.Conv_out(out)


class GRAPHLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class GraphBranch(nn.Module):
    def __init__(self, hidden_node):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(hidden_node, hidden_node, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_node),
            nn.LeakyReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_node, hidden_node, kernel_size=1),
            nn.BatchNorm2d(hidden_node),
            nn.LeakyReLU(),
        )
        self.conv_out = nn.Conv2d(hidden_node, hidden_node, kernel_size=1)
        self.reps_graph = nn.Sequential(
            nn.Linear(hidden_node, hidden_node // 2),
            GRAPHLayerNorm(hidden_node // 2, eps=1e-12),
            nn.ReLU(),
            nn.Linear(hidden_node // 2, hidden_node),
            GRAPHLayerNorm(hidden_node, eps=1e-12),
            nn.ReLU(),
        )

    def forward(self, x):
        org = x
        B, C, H, W = x.shape

        x_reshaped = x.permute(0, 2, 3, 1).reshape(B, H * W, C)

        reps_graph = torch.matmul(x_reshaped, x_reshaped.transpose(1, 2))
        reps_graph = F.softmax(reps_graph, dim=-1)
        rel_reps = torch.matmul(reps_graph, x_reshaped)
        rel_reps = self.reps_graph(x_reshaped + rel_reps)

        rel_reps = rel_reps.reshape(B, H, W, C).permute(0, 3, 1, 2)
        rel_reps = self.conv2(rel_reps)
        return self.conv_out(rel_reps + org)


class ChannelAttention(nn.Module):
    def __init__(self, num_feat, squeeze_factor=8, memory_blocks=256):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.subnet = nn.Sequential(nn.Linear(num_feat, num_feat // squeeze_factor))
        self.upnet = nn.Sequential(
            nn.Linear(num_feat // squeeze_factor, num_feat), nn.Sigmoid()
        )
        self.mb = nn.Parameter(torch.randn(num_feat // squeeze_factor, memory_blocks))
        self.low_dim = num_feat // squeeze_factor

    def forward(self, x):
        b, n, c = x.shape
        t = x.transpose(1, 2)
        y = self.pool(t).squeeze(-1)
        low_rank_f = self.subnet(y).unsqueeze(2)
        mbg = self.mb.unsqueeze(0).repeat(b, 1, 1)
        f1 = torch.bmm(low_rank_f.transpose(1, 2), mbg)
        f_dic_c = F.softmax(f1 * (int(self.low_dim) ** (-0.5)), dim=-1)
        y1 = torch.bmm(f_dic_c, mbg.transpose(1, 2))
        y2 = self.upnet(y1.squeeze(1))
        return x * y2.unsqueeze(1)


class LRRBlock(nn.Module):
    def __init__(self, num_feat, squeeze_factor=8, memory_blocks=256):
        super().__init__()
        self.num_feat = num_feat
        self.cab = ChannelAttention(num_feat, squeeze_factor, memory_blocks)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        out = self.cab(x)
        return out.reshape(B, H, W, C).permute(0, 3, 1, 2)


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa = nn.Conv2d(
            2, 1, kernel_size=3, padding=1, padding_mode="reflect", bias=True
        )

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        return self.sa(torch.cat([x_avg, x_max], dim=1))


class ChannelAttention2(nn.Module):
    def __init__(self, dim, reduction=8):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.map = nn.AdaptiveMaxPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, bias=True),
        )
        self.ma = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, bias=True),
        )
        self.Conv = nn.Conv2d(dim, dim, 1, bias=True)

    def forward(self, x):
        return self.Conv(self.ca(self.gap(x)) + self.ma(self.map(x)))


class SFFM(nn.Module):
    def __init__(self, dim, reduction=8):
        super().__init__()
        self.sa = SpatialAttention()
        self.convfusion = nn.Conv2d(dim, dim, 3, padding=1, padding_mode="reflect")
        self.ca = ChannelAttention2(dim, reduction)
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, LRR_feat, local_feat, graph_feat):
        lrr_feat = self.convfusion(LRR_feat)
        initial = lrr_feat + local_feat + graph_feat
        spa_out = initial * self.sa(initial)
        spe_out = spa_out * self.ca(spa_out)
        pattn2 = self.sigmoid(spe_out)
        return self.conv(pattn2 * local_feat + (1 - pattn2) * graph_feat)


class SGLNetAE(nn.Module):
    """
    SGLNet Autoencoder for patch-based anomaly detection.

    Takes patches of shape (B, patch_size, patch_size) as input and reconstructs them.
    Anomaly score = reconstruction error.
    """

    def __init__(self, in_ch=189, head_ch=48, local_blc=2, sq=10, mb=64):
        super().__init__()

        assert head_ch % 2 == 0, "head_ch should be divisible by 2"
        n1 = in_ch // 2

        self.Encoder1 = nn.Sequential(
            nn.Conv2d(in_ch, n1, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.BatchNorm2d(n1),
            nn.LeakyReLU(),
        )
        self.Encoder2 = nn.Sequential(
            nn.Conv2d(n1, head_ch, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.BatchNorm2d(head_ch),
            nn.Sigmoid(),
        )

        self.Decoder1 = nn.Sequential(
            nn.Conv2d(head_ch, n1, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.BatchNorm2d(n1),
            nn.LeakyReLU(),
        )
        self.Decoder2 = nn.Sequential(
            nn.Conv2d(n1, in_ch, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.BatchNorm2d(in_ch),
        )

        self.branch1 = LocalBranch(head_ch, local_blc)
        self.branch2 = GraphBranch(head_ch)
        self.LR_Prior = LRRBlock(head_ch, sq, mb)
        self.CGAFusion = SFFM(head_ch)
        self.Conv1x1 = nn.Conv2d(head_ch, head_ch, kernel_size=1)

    def forward(self, x):
        x1 = self.Encoder1(x)
        x2 = self.Encoder2(x1)

        local_br = self.branch1(x2)
        global_br = self.branch2(x2)
        LR_prior = self.LR_Prior(x2)
        fuse = self.CGAFusion(LR_prior, local_br, global_br)

        out1 = self.Decoder1(fuse)
        out2 = self.Decoder2(out1)

        return out2

    def encode(self, x):
        x1 = self.Encoder1(x)
        x2 = self.Encoder2(x1)
        return x2

    def decode(self, z):
        out1 = self.Decoder1(z)
        out2 = self.Decoder2(out1)
        return out2
