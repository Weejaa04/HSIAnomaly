import copy

import torch
import torch.nn as nn

from .block import SlidingWindowExtractor, PartialEncoders, PartialDecoders


# ============================= MODEL ARCHITECTURE ========================

class PA2E(nn.Module):
    """PA2E model with FC partial encoders/decoders."""

    def __init__(self, input_dim: int, window_size: int = 56, stride: int = 14,
                 hidden_dims: list = None, latent_local_dim: int = 32,
                 latent_global_dim: int = 64):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [52, 48, 44, 40, 36]

        self.input_dim        = input_dim
        self.window_size      = window_size
        self.stride           = stride
        self.latent_global_dim = latent_global_dim

        self.window_extractor = SlidingWindowExtractor(input_dim, window_size, stride)
        num_windows = self.window_extractor.num_windows
        print(f'Number of windows: {num_windows}')

        self.partial_encoders = PartialEncoders(num_windows, window_size,
                                                hidden_dims, latent_local_dim)

        concat_dim = num_windows * latent_local_dim
        self.aggregate_encoder = nn.Linear(concat_dim, latent_global_dim)
        self.aggregate_decoder = nn.Linear(latent_global_dim, concat_dim)

        self.partial_decoders = PartialDecoders(num_windows, window_size,
                                                hidden_dims, latent_local_dim)

        self.register_buffer('center', torch.zeros(latent_global_dim))
        self.center_initialized = False

    def encode(self, x: torch.Tensor):
        windows      = self.window_extractor(x)
        local_latents = self.partial_encoders(windows)
        z_concat     = local_latents.view(x.size(0), -1)
        z            = self.aggregate_encoder(z_concat)
        return z, local_latents

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z_concat       = self.aggregate_decoder(z)
        num_windows    = self.window_extractor.num_windows
        latent_local_dim = z_concat.size(1) // num_windows
        local_latents  = z_concat.view(z.size(0), num_windows, latent_local_dim)
        return self.partial_decoders(local_latents)

    def forward(self, x: torch.Tensor):
        z, local_latents     = self.encode(x)
        reconstructed_windows = self.decode(z)
        return z, reconstructed_windows

    def compute_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        z, _ = self.encode(x)
        return torch.sum((z - self.center) ** 2, dim=1)


class PA2EFT(PA2E):
    """Fine-tuning wrapper for PA2E (decodes from local latents directly)."""

    def __init__(self, pa2e: PA2E):
        super().__init__(
            input_dim=pa2e.input_dim,
            window_size=pa2e.window_size,
            stride=pa2e.stride,
            hidden_dims=pa2e.partial_encoders.hidden_dims,
            latent_local_dim=pa2e.partial_encoders.latent_local_dim,
            latent_global_dim=pa2e.latent_global_dim,
        )
        self.load_state_dict(pa2e.state_dict())

    def decode(self, local_latents: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.partial_decoders(local_latents)

    def forward(self, x: torch.Tensor):
        z, local_latents      = self.encode(x)
        reconstructed_windows = self.decode(local_latents)
        return z, reconstructed_windows

    def compute_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        z, _ = self.encode(x)
        return torch.sum((z - self.center) ** 2, dim=1)
