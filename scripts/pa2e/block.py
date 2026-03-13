import torch
import torch.nn as nn


# ============================= MODEL BLOCKS ==============================

class SlidingWindowExtractor(nn.Module):
    """Extract sliding windows using masked FC (not convolution)."""

    def __init__(self, input_dim: int, window_size: int, stride: int):
        super().__init__()
        self.input_dim  = input_dim
        self.window_size = window_size
        self.stride     = stride

        self.num_windows = 1 + (input_dim - window_size) // stride
        self.register_buffer('extract_matrix', self._create_extract_matrix())

    def _create_extract_matrix(self) -> torch.Tensor:
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
        return windows.view(x.size(0), self.num_windows, self.window_size)


class PartialEncoders(nn.Module):
    """Independent FC encoders for each window (block-diagonal masked FC)."""

    def __init__(self, num_windows: int, window_size: int,
                 hidden_dims: list, latent_local_dim: int):
        super().__init__()
        self.num_windows     = num_windows
        self.window_size     = window_size
        self.hidden_dims     = hidden_dims
        self.latent_local_dim = latent_local_dim

        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(window_size,    hidden_dims[0]),
                nn.BatchNorm1d(hidden_dims[0]),
                nn.Linear(hidden_dims[0], hidden_dims[1]),
                nn.BatchNorm1d(hidden_dims[1]),
                nn.Linear(hidden_dims[1], hidden_dims[2]),
                nn.BatchNorm1d(hidden_dims[2]),
                nn.Linear(hidden_dims[2], hidden_dims[3]),
                nn.BatchNorm1d(hidden_dims[3]),
                nn.ReLU(),
                nn.Linear(hidden_dims[3], hidden_dims[4]),
                nn.BatchNorm1d(hidden_dims[4]),
                nn.ReLU(),
                nn.Linear(hidden_dims[4], latent_local_dim),
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
            local_latents.append(self.encoders[i](windows[:, i, :]))
        return torch.stack(local_latents, dim=1)


class PartialDecoders(nn.Module):
    """Independent FC decoders for each window (mirror of PartialEncoders)."""

    def __init__(self, num_windows: int, window_size: int,
                 hidden_dims: list, latent_local_dim: int):
        super().__init__()
        self.num_windows = num_windows
        self.window_size = window_size

        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_local_dim, hidden_dims[4]),
                nn.BatchNorm1d(hidden_dims[4]),
                nn.ReLU(),
                nn.Linear(hidden_dims[4], hidden_dims[3]),
                nn.BatchNorm1d(hidden_dims[3]),
                nn.Linear(hidden_dims[3], hidden_dims[2]),
                nn.BatchNorm1d(hidden_dims[2]),
                nn.Linear(hidden_dims[2], hidden_dims[1]),
                nn.BatchNorm1d(hidden_dims[1]),
                nn.Linear(hidden_dims[1], hidden_dims[0]),
                nn.BatchNorm1d(hidden_dims[0]),
                nn.Linear(hidden_dims[0], window_size),
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
            reconstructed.append(self.decoders[i](local_latents[:, i, :]))
        return torch.stack(reconstructed, dim=1)
