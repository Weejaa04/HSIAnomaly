import torch
import torch.nn as nn


# ============================= MODEL BLOCKS ==============================

class SlidingWindowExtractor(nn.Module):
    """Extract sliding windows using masked FC (not convolution)."""

    def __init__(self, input_dim: int, window_size: int, stride: int):
        super().__init__()
        self.input_dim = input_dim
        self.window_size = window_size
        self.stride = stride

        self.num_windows = 1 + (input_dim - window_size) // stride
        self.register_buffer('extract_matrix', self._create_extract_matrix())

    def _create_extract_matrix(self) -> torch.Tensor:
        """Create a matrix that extracts sliding windows."""
        matrix = torch.zeros(self.num_windows * self.window_size, self.input_dim)
        for i in range(self.num_windows):
            start_idx = i * self.stride
            for j in range(self.window_size):
                matrix[i * self.window_size + j, start_idx + j] = 1.0
        return matrix

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim)
        Returns:
            windows: (batch, num_windows, window_size)
        """
        windows = torch.matmul(x, self.extract_matrix.t())
        batch_size = x.size(0)
        return windows.view(batch_size, self.num_windows, self.window_size)


class PartialEncodersConv1D(nn.Module):
    """Lean Conv1D encoders with GAP to 2 bins (14 channels × 2 bins = 28 dims)."""

    def __init__(self, num_windows: int, window_size: int, latent_local_dim: int):
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
                nn.Flatten(start_dim=1, end_dim=-1),
            )
            for _ in range(num_windows)
        ])

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        """
        Args:
            windows: (batch, num_windows, window_size)
        Returns:
            local_latents: (batch, num_windows, latent_local_dim)
        """
        local_latents = []
        for i in range(self.num_windows):
            window_i = windows[:, i, :].unsqueeze(1)
            local_latents.append(self.encoders[i](window_i))
        return torch.stack(local_latents, dim=1)


class PartialDecodersConv1D(nn.Module):
    """Lean Conv1D decoders symmetric to encoder with 14 channels × 2 bins input."""

    def __init__(self, num_windows: int, window_size: int, latent_local_dim: int):
        super().__init__()
        self.num_windows = num_windows
        self.window_size = window_size
        self.latent_local_dim = latent_local_dim

        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Unflatten(dim=1, unflattened_size=(14, 2)),
                nn.Upsample(size=14, mode='nearest'),
                nn.ConvTranspose1d(in_channels=14, out_channels=16, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.ConvTranspose1d(in_channels=16, out_channels=1, kernel_size=4, stride=2, padding=1),
            )
            for _ in range(num_windows)
        ])

    def forward(self, local_latents: torch.Tensor) -> torch.Tensor:
        """
        Args:
            local_latents: (batch, num_windows, latent_local_dim)
        Returns:
            reconstructed_windows: (batch, num_windows, window_size)
        """
        reconstructed = []
        for i in range(self.num_windows):
            latent_i = local_latents[:, i, :]
            reconstructed.append(self.decoders[i](latent_i).squeeze(1))
        return torch.stack(reconstructed, dim=1)
