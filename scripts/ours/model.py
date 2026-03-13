import torch
import torch.nn as nn

from .block import SlidingWindowExtractor, PartialEncodersConv1D, PartialDecodersConv1D


# ============================= MODEL ARCHITECTURE ========================

class PA2E(nn.Module):
    """PA2E model with Conv1D partial encoders/decoders."""

    def __init__(self, input_dim: int, window_size: int = 56, stride: int = 14,
                 latent_local_dim: int = 32, latent_global_dim: int = 64):
        super().__init__()

        self.input_dim = input_dim
        self.window_size = window_size
        self.stride = stride
        self.latent_global_dim = latent_global_dim

        self.window_extractor = SlidingWindowExtractor(input_dim, window_size, stride)
        num_windows = self.window_extractor.num_windows

        self.partial_encoders = PartialEncodersConv1D(num_windows, window_size, latent_local_dim)

        concat_dim = num_windows * latent_local_dim
        self.aggregate_encoder = nn.Linear(concat_dim, latent_global_dim)
        self.aggregate_decoder = nn.Linear(latent_global_dim, concat_dim)

        self.partial_decoders = PartialDecodersConv1D(num_windows, window_size, latent_local_dim)

        self.register_buffer('center', torch.zeros(latent_global_dim))
        self.center_initialized = False

    def encode(self, x: torch.Tensor):
        """Encode input to latent representation."""
        windows = self.window_extractor(x)
        local_latents = self.partial_encoders(windows)

        batch_size = x.size(0)
        z_concat = local_latents.view(batch_size, -1)
        z = self.aggregate_encoder(z_concat)

        return z, local_latents

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to reconstruction."""
        z_concat = self.aggregate_decoder(z)

        batch_size = z.size(0)
        num_windows = self.window_extractor.num_windows
        latent_local_dim = z_concat.size(1) // num_windows
        local_latents = z_concat.view(batch_size, num_windows, latent_local_dim)

        return self.partial_decoders(local_latents)

    def forward(self, x: torch.Tensor):
        """Forward pass."""
        z, local_latents = self.encode(x)
        reconstructed_windows = self.decode(z)
        return z, reconstructed_windows

    def compute_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Compute anomaly score as distance to center."""
        z, _ = self.encode(x)
        return torch.sum((z - self.center) ** 2, dim=1)


class PA2EFT(PA2E):
    """Fine-tuning wrapper for PA2E with Conv1D components."""

    def __init__(self, pa2e: PA2E):
        super().__init__(
            input_dim=pa2e.input_dim,
            window_size=pa2e.window_size,
            stride=pa2e.stride,
            latent_local_dim=28,
            latent_global_dim=pa2e.latent_global_dim,
        )
        self.load_state_dict(pa2e.state_dict())

    def encode(self, x: torch.Tensor):
        """Encode input to latent representation."""
        windows = self.window_extractor(x)
        local_latents = self.partial_encoders(windows)

        batch_size = x.size(0)
        z_concat = local_latents.view(batch_size, -1)
        z = self.aggregate_encoder(z_concat)

        return z, local_latents

    def decode(self, local_latents: torch.Tensor) -> torch.Tensor:
        """Decode local latents directly to reconstructed windows."""
        return self.partial_decoders(local_latents)

    def forward(self, x: torch.Tensor):
        """Forward pass."""
        z, local_latents = self.encode(x)
        reconstructed_windows = self.decode(local_latents)
        return z, reconstructed_windows

    def compute_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Compute anomaly score as distance to center."""
        z, _ = self.encode(x)
        return torch.sum((z - self.center) ** 2, dim=1)
