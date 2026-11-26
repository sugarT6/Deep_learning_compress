"""Shared FiLM conditioning from explicitly supplied, transmitted block features."""

from __future__ import annotations

import torch
from torch import nn


BLOCK_FEATURE_DIM = 16


class BlockQualityConditioning(nn.Module):
    """Map one normalized block descriptor to affine parameters for all layers.

    The output projection starts at zero, so enabling this module initially
    leaves the Transformer unchanged. Neither this module nor the model obtains
    descriptors from quality targets: callers must supply decoded wire features.
    """

    def __init__(self, d_model: int, num_layers: int, hidden_dim: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.encoder = nn.Sequential(
            nn.Linear(BLOCK_FEATURE_DIM, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_layers * 2 * d_model),
        )
        nn.init.zeros_(self.encoder[-1].weight)
        nn.init.zeros_(self.encoder[-1].bias)

    def forward(self, features: torch.Tensor):
        affine = self.encoder(features).reshape(
            features.shape[0], self.num_layers, 2, self.d_model
        )
        return affine[:, :, 0], affine[:, :, 1]
