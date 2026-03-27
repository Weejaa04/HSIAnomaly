"""
Model architectures for ablation study.

Variants vs the baseline (scripts/our/model.py):
  1. Encoder type  : Conv1D (baseline) vs Linear
  2. Partial mode  : Partial (baseline) vs Non-Partial (full-band single encoder)

The loss variant (MAE vs MSE) and scoring variant (DSVDD vs NNMB) are
handled inside __main__.py, not inside the model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Shared: Sliding Window Extractor
# ─────────────────────────────────────────────────────────────────────────────

class SlidingWindowExtractor(nn.Module):
    """Extract sliding windows using masked FC (not convolution)."""

    def __init__(self, input_dim, window_size, stride):
        super().__init__()
        self.input_dim = input_dim
        self.window_size = window_size
        self.stride = stride

        self.num_windows = 1 + (input_dim - window_size) // stride
        self.register_buffer('extract_matrix', self._create_extract_matrix())

    def _create_extract_matrix(self):
        matrix = torch.zeros(self.num_windows * self.window_size, self.input_dim)
        for i in range(self.num_windows):
            start_idx = i * self.stride
            end_idx = start_idx + self.window_size
            for j in range(self.window_size):
                matrix[i * self.window_size + j, start_idx + j] = 1.0
        return matrix

    def forward(self, x):
        """
        Args:
            x: (batch, input_dim)
        Returns:
            windows: (batch, num_windows, window_size)
        """
        windows = torch.matmul(x, self.extract_matrix.t())
        batch_size = x.size(0)
        windows = windows.view(batch_size, self.num_windows, self.window_size)
        return windows


# ─────────────────────────────────────────────────────────────────────────────
# Variant A: Conv1D partial encoders/decoders (BASELINE)
# ─────────────────────────────────────────────────────────────────────────────

class PartialEncodersConv1D(nn.Module):
    """Lean Conv1D encoders with GAP to 2 bins (14 channels × 2 bins = 28 dims)."""

    def __init__(self, num_windows, window_size, latent_local_dim):
        super().__init__()
        self.num_windows = num_windows
        self.window_size = window_size
        self.latent_local_dim = latent_local_dim

        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.Conv1d(in_channels=16, out_channels=14, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(14),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(output_size=2),
                nn.Flatten(start_dim=1, end_dim=-1)
            )
            for _ in range(num_windows)
        ])

    def forward(self, windows):
        """
        Args:
            windows: (batch, num_windows, window_size)
        Returns:
            local_latents: (batch, num_windows, latent_local_dim)
        """
        local_latents = []
        for i in range(self.num_windows):
            window_i = windows[:, i, :].unsqueeze(1)
            latent_i = self.encoders[i](window_i)
            local_latents.append(latent_i)
        return torch.stack(local_latents, dim=1)


class PartialDecodersConv1D(nn.Module):
    """Lean Conv1D decoders symmetric to PartialEncodersConv1D."""

    def __init__(self, num_windows, window_size, latent_local_dim):
        super().__init__()
        self.num_windows = num_windows

        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Unflatten(dim=1, unflattened_size=(14, 2)),
                nn.Upsample(size=14, mode='nearest'),
                nn.ConvTranspose1d(in_channels=14, out_channels=16, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.ConvTranspose1d(in_channels=16, out_channels=1, kernel_size=4, stride=2, padding=1)
            )
            for _ in range(num_windows)
        ])

    def forward(self, local_latents):
        """
        Args:
            local_latents: (batch, num_windows, latent_local_dim)
        Returns:
            reconstructed_windows: (batch, num_windows, window_size)
        """
        reconstructed = []
        for i in range(self.num_windows):
            latent_i = local_latents[:, i, :]
            recon_i = self.decoders[i](latent_i).squeeze(1)
            reconstructed.append(recon_i)
        return torch.stack(reconstructed, dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# Variant B: Linear partial encoders/decoders
# Same dimensionality reduction ratio as Conv1D: window_size → latent_local_dim
# ─────────────────────────────────────────────────────────────────────────────

class PartialEncodersLinear(nn.Module):
    """Linear encoders as drop-in replacement for PartialEncodersConv1D."""

    def __init__(self, num_windows, window_size, latent_local_dim):
        super().__init__()
        self.num_windows = num_windows
        # Same latent_local_dim output as Conv1D variant
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(window_size, window_size // 2),
                nn.ReLU(),
                nn.Linear(window_size // 2, latent_local_dim),
            )
            for _ in range(num_windows)
        ])

    def forward(self, windows):
        local_latents = []
        for i in range(self.num_windows):
            window_i = windows[:, i, :]       # (batch, window_size)
            latent_i = self.encoders[i](window_i)
            local_latents.append(latent_i)
        return torch.stack(local_latents, dim=1)


class PartialDecodersLinear(nn.Module):
    """Linear decoders symmetric to PartialEncodersLinear."""

    def __init__(self, num_windows, window_size, latent_local_dim):
        super().__init__()
        self.num_windows = num_windows
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_local_dim, window_size // 2),
                nn.ReLU(),
                nn.Linear(window_size // 2, window_size),
            )
            for _ in range(num_windows)
        ])

    def forward(self, local_latents):
        reconstructed = []
        for i in range(self.num_windows):
            latent_i = local_latents[:, i, :]
            recon_i = self.decoders[i](latent_i)
            reconstructed.append(recon_i)
        return torch.stack(reconstructed, dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# Partial model (BASELINE) — Conv1D or Linear partial encoders
# ─────────────────────────────────────────────────────────────────────────────

class PA2EPartial(nn.Module):
    """
    PA2E with partial (windowed) encoders.
    encoder_type: 'cnn' (Conv1D, baseline) or 'linear'
    """

    def __init__(self, input_dim, window_size=56, stride=14,
                 latent_local_dim=28, latent_global_dim=32,
                 encoder_type='cnn'):
        super().__init__()

        self.input_dim = input_dim
        self.window_size = window_size
        self.stride = stride
        self.latent_global_dim = latent_global_dim
        self.encoder_type = encoder_type

        self.window_extractor = SlidingWindowExtractor(input_dim, window_size, stride)
        num_windows = self.window_extractor.num_windows

        if encoder_type == 'cnn':
            self.partial_encoders = PartialEncodersConv1D(num_windows, window_size, latent_local_dim)
            self.partial_decoders = PartialDecodersConv1D(num_windows, window_size, latent_local_dim)
        elif encoder_type == 'linear':
            self.partial_encoders = PartialEncodersLinear(num_windows, window_size, latent_local_dim)
            self.partial_decoders = PartialDecodersLinear(num_windows, window_size, latent_local_dim)
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type!r}")

        concat_dim = num_windows * latent_local_dim
        self.aggregate_encoder = nn.Linear(concat_dim, latent_global_dim)
        self.aggregate_decoder = nn.Linear(latent_global_dim, concat_dim)

        self.register_buffer('center', torch.zeros(latent_global_dim))
        self.center_initialized = False

    def encode(self, x):
        windows = self.window_extractor(x)
        local_latents = self.partial_encoders(windows)
        batch_size = x.size(0)
        z_concat = local_latents.view(batch_size, -1)
        z = self.aggregate_encoder(z_concat)
        return z, local_latents

    def decode_from_z(self, z):
        """Reconstruct windows from global latent z (used in Phase 1 / PA2E forward)."""
        z_concat = self.aggregate_decoder(z)
        batch_size = z.size(0)
        num_windows = self.window_extractor.num_windows
        latent_local_dim = z_concat.size(1) // num_windows
        local_latents = z_concat.view(batch_size, num_windows, latent_local_dim)
        return self.partial_decoders(local_latents)

    def decode_from_local(self, local_latents):
        """Reconstruct windows from local latents (used in Phase 2 fine-tuning)."""
        return self.partial_decoders(local_latents)

    def forward(self, x):
        """Phase 1 forward: encode → decode via z."""
        z, local_latents = self.encode(x)
        reconstructed_windows = self.decode_from_z(z)
        return z, reconstructed_windows

    def window_extractor_fwd(self, x):
        return self.window_extractor(x)

    def compute_anomaly_score(self, x):
        z, _ = self.encode(x)
        return torch.sum((z - self.center) ** 2, dim=1)


class PA2EPartialFT(PA2EPartial):
    """Fine-tuning wrapper: decode from local latents instead of z."""

    def __init__(self, pa2e: PA2EPartial):
        super().__init__(
            input_dim=pa2e.input_dim,
            window_size=pa2e.window_size,
            stride=pa2e.stride,
            latent_local_dim=28,
            latent_global_dim=pa2e.latent_global_dim,
            encoder_type=pa2e.encoder_type,
        )
        self.load_state_dict(pa2e.state_dict())

    def forward(self, x):
        z, local_latents = self.encode(x)
        reconstructed_windows = self.decode_from_local(local_latents)
        return z, reconstructed_windows


# ─────────────────────────────────────────────────────────────────────────────
# Non-Partial model — same total latent capacity, no windowing
# ─────────────────────────────────────────────────────────────────────────────

def _compute_nonpartial_dims(input_dim, window_size, stride, latent_local_dim, latent_global_dim):
    """
    Compute non-partial dims so total parameter budget matches partial:
    num_windows * latent_local_dim == hidden_dim (concat_dim in partial)
    """
    num_windows = 1 + (input_dim - window_size) // stride
    hidden_dim = num_windows * latent_local_dim   # same as partial concat_dim
    return hidden_dim


class PA2ENonPartialCNN(nn.Module):
    """
    Non-partial model using Conv1D on the entire spectrum.

    If the partial model's concat_dim = num_windows * latent_local_dim,
    this model uses an equivalent Conv1D that processes the full input_dim at
    once with the same target hidden dim.

    Down/up scale:  input_dim → hidden_dim → latent_global_dim (same as partial)
    """

    def __init__(self, input_dim, window_size=56, stride=14,
                 latent_local_dim=28, latent_global_dim=32):
        super().__init__()
        self.input_dim = input_dim
        self.latent_global_dim = latent_global_dim

        hidden_dim = _compute_nonpartial_dims(input_dim, window_size, stride,
                                              latent_local_dim, latent_global_dim)
        self.hidden_dim = hidden_dim

        # Encoder: (batch,1,input_dim) → (batch, hidden_dim) via Conv1D
        # Two conv layers applying same stride-2 downsampling, then adaptive pool
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 14, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(14),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(output_size=hidden_dim // 14),
            nn.Flatten(start_dim=1),
        )
        actual_hidden = 14 * (hidden_dim // 14)

        self.aggregate_encoder = nn.Linear(actual_hidden, latent_global_dim)
        self.aggregate_decoder = nn.Linear(latent_global_dim, actual_hidden)

        # Decoder: unflatten → upsample → ConvTranspose1d
        half_size = input_dim // 4
        self.decoder = nn.Sequential(
            nn.Unflatten(dim=1, unflattened_size=(14, hidden_dim // 14)),
            nn.Upsample(size=half_size, mode='nearest'),
            nn.ConvTranspose1d(14, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.AdaptiveAvgPool1d(output_size=input_dim),
        )

        self.register_buffer('center', torch.zeros(latent_global_dim))
        self.center_initialized = False

    def encode(self, x):
        """x: (batch, input_dim) → z: (batch, latent_global_dim)"""
        x_3d = x.unsqueeze(1)                           # (batch,1,input_dim)
        h = self.encoder(x_3d)                          # (batch, actual_hidden)
        z = self.aggregate_encoder(h)                   # (batch, latent_global_dim)
        return z, h                                      # h plays role of local_latents

    def decode(self, z):
        h = self.aggregate_decoder(z)
        x_3d = self.decoder(h)                          # (batch, 1, input_dim)
        return x_3d.squeeze(1)                          # (batch, input_dim)

    def forward(self, x):
        z, h = self.encode(x)
        recon = self.decode(z)
        return z, recon

    def compute_anomaly_score(self, x):
        z, _ = self.encode(x)
        return torch.sum((z - self.center) ** 2, dim=1)


class PA2ENonPartialLinear(nn.Module):
    """
    Non-partial model using Linear layers.
    Down/up scale: input_dim → hidden_dim → latent_global_dim
    hidden_dim = num_windows * latent_local_dim (same as partial's concat_dim)
    """

    def __init__(self, input_dim, window_size=56, stride=14,
                 latent_local_dim=28, latent_global_dim=32):
        super().__init__()
        self.input_dim = input_dim
        self.latent_global_dim = latent_global_dim

        hidden_dim = _compute_nonpartial_dims(input_dim, window_size, stride,
                                              latent_local_dim, latent_global_dim)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, hidden_dim),
        )
        self.aggregate_encoder = nn.Linear(hidden_dim, latent_global_dim)
        self.aggregate_decoder = nn.Linear(latent_global_dim, hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim),
        )

        self.register_buffer('center', torch.zeros(latent_global_dim))
        self.center_initialized = False

    def encode(self, x):
        h = self.encoder(x)
        z = self.aggregate_encoder(h)
        return z, h

    def decode(self, z):
        h = self.aggregate_decoder(z)
        return self.decoder(h)

    def forward(self, x):
        z, h = self.encode(x)
        recon = self.decode(z)
        return z, recon

    def compute_anomaly_score(self, x):
        z, _ = self.encode(x)
        return torch.sum((z - self.center) ** 2, dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_model(input_dim, partial=True, encoder_type='cnn',
                window_size=56, stride=14, latent_local_dim=28, latent_global_dim=32):
    """
    Build the appropriate model for the given ablation configuration.

    Args:
        input_dim: number of spectral bands
        partial: if True use windowed partial model (baseline)
        encoder_type: 'cnn' or 'linear'
        window_size, stride, latent_local_dim, latent_global_dim: architecture params

    Returns:
        model (nn.Module), is_partial (bool)
    """
    if partial:
        model = PA2EPartial(
            input_dim=input_dim,
            window_size=window_size,
            stride=stride,
            latent_local_dim=latent_local_dim,
            latent_global_dim=latent_global_dim,
            encoder_type=encoder_type,
        )
    else:
        if encoder_type == 'cnn':
            model = PA2ENonPartialCNN(
                input_dim=input_dim,
                window_size=window_size,
                stride=stride,
                latent_local_dim=latent_local_dim,
                latent_global_dim=latent_global_dim,
            )
        elif encoder_type == 'linear':
            model = PA2ENonPartialLinear(
                input_dim=input_dim,
                window_size=window_size,
                stride=stride,
                latent_local_dim=latent_local_dim,
                latent_global_dim=latent_global_dim,
            )
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type!r}")

    return model
