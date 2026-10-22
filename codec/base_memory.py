"""Quality-independent base memory and causal layers that read that memory."""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch import nn

from .fastq_stream import BASE_ALPHABET_SIZE, BASE_PAD_ID


def _padding_positions(mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    return mask if mask.dtype == torch.bool else torch.isneginf(mask)


def _safe_padding_mask(mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Expose zero-valued keys for empty reads so softmax has a finite row."""
    padding = _padding_positions(mask)
    if padding is None or padding.shape[1] == 0:
        return mask
    empty = padding.all(dim=1)
    if not bool(empty.any()):
        return mask
    safe = mask.clone()
    safe[empty, 0] = False if safe.dtype == torch.bool else 0.0
    return safe


class BaseMemoryEncoder(nn.Module):
    """Bidirectional base-only memory; target qualities never enter this module."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.memory_dim = config.base_memory_dim
        self.embedding = nn.Embedding(
            BASE_ALPHABET_SIZE, config.base_embed_dim, padding_idx=BASE_PAD_ID
        )
        self.input_projection = nn.Linear(config.base_embed_dim, self.memory_dim)
        self.position_projection = nn.Linear(3, self.memory_dim, bias=False)
        layer = nn.TransformerEncoderLayer(
            d_model=self.memory_dim,
            nhead=config.base_memory_heads,
            dim_feedforward=2 * self.memory_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=config.base_memory_layers,
            norm=nn.LayerNorm(self.memory_dim),
            enable_nested_tensor=False,
        )

    def forward(
        self,
        bases: torch.Tensor,
        active_mask: torch.Tensor,
        position_features: torch.Tensor,
    ) -> torch.Tensor:
        if bases.ndim != 2 or active_mask.shape != bases.shape:
            raise ValueError("bases and active_mask must have shape [batch, sequence]")
        if active_mask.dtype != torch.bool:
            raise ValueError("active_mask must be bool")
        if position_features.shape != (*bases.shape, 3):
            raise ValueError("position_features must have shape [batch, sequence, 3]")
        if bases.shape[0] == 0 or bases.shape[1] == 0:
            return self.input_projection.weight.new_zeros(
                (*bases.shape, self.memory_dim)
            )
        padding = ~active_mask
        safe_bases = bases.masked_fill(padding, BASE_PAD_ID)
        positions = position_features.masked_fill(padding.unsqueeze(-1), 0.0)
        hidden = self.input_projection(self.embedding(safe_bases))
        hidden = hidden + self.position_projection(positions.to(hidden.dtype))
        hidden = hidden.masked_fill(padding.unsqueeze(-1), 0.0)
        hidden = self.transformer(
            hidden, src_key_padding_mask=_safe_padding_mask(padding)
        )
        return hidden.masked_fill(padding.unsqueeze(-1), 0.0)


class BaseMemoryDecoderLayer(nn.TransformerEncoderLayer):
    """Pre-LN causal Q self-attention, base cross-attention, then feedforward.

    Existing self_attn/linear1/linear2 names remain compatible with suffix LoRA.
    A handwritten forward always executes cross-attention, including no_grad
    inference where the parent class can otherwise select a fused encoder path.
    Pass the original memory padding mask: all-padding rows are handled here.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Any = "gelu",
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
        norm_first: bool = True,
        device: Any = None,
        dtype: Any = None,
        *,
        memory_dim: int,
    ) -> None:
        if not batch_first or not norm_first:
            raise ValueError("base-memory decoder requires batch_first and norm_first")
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            norm_first=norm_first,
            device=device,
            dtype=dtype,
        )
        self.cross_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            kdim=memory_dim,
            vdim=memory_dim,
            batch_first=True,
            device=device,
            dtype=dtype,
        )
        self.cross_norm = nn.LayerNorm(
            d_model, eps=layer_norm_eps, device=device, dtype=dtype
        )
        self.cross_dropout = nn.Dropout(dropout)
        self.activation_relu_or_gelu = 0

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        *,
        memory: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        if src.shape[0] == 0 or src.shape[1] == 0:
            return src
        padding = _padding_positions(src_key_padding_mask)
        hidden = src
        if padding is not None:
            hidden = hidden.masked_fill(padding.unsqueeze(-1), 0.0)
        if is_causal and src_mask is None:
            src_mask = torch.ones(
                src.shape[1], src.shape[1], dtype=torch.bool, device=src.device
            ).triu(1)
        self_padding_mask = src_key_padding_mask
        if padding is not None and bool(padding.any()):
            # A padded query beyond a short read's sliding window can otherwise
            # have no visible keys. Masking its NaN output fixes inference but
            # still leaves NaN softmax gradients. Give only discarded queries
            # unrestricted rows; retained queries keep the exact causal mask.
            batch, length = src.shape[:2]
            heads = self.self_attn.num_heads
            if src_mask is None:
                combined = torch.zeros(
                    batch, heads, length, length, dtype=torch.bool, device=src.device
                )
            elif src_mask.ndim == 2:
                combined = src_mask[None, None].expand(batch, heads, -1, -1).clone()
            else:
                combined = src_mask.reshape(batch, heads, length, length).clone()
            if combined.dtype == torch.bool and src_key_padding_mask.dtype == torch.bool:
                combined |= padding[:, None, None, :]
                combined.masked_fill_(padding[:, None, :, None], False)
            else:
                if combined.dtype == torch.bool:
                    combined = torch.zeros_like(combined, dtype=src.dtype).masked_fill(
                        combined, float("-inf")
                    )
                key_bias = src_key_padding_mask
                if key_bias.dtype == torch.bool:
                    key_bias = torch.zeros_like(key_bias, dtype=combined.dtype).masked_fill(
                        key_bias, float("-inf")
                    )
                combined = combined + key_bias[:, None, None, :]
                combined.masked_fill_(padding[:, None, :, None], 0.0)
            src_mask = combined.reshape(batch * heads, length, length)
            self_padding_mask = None
        normalized = self.norm1(hidden)
        attended = self.dropout1(self.self_attn(
            normalized,
            normalized,
            normalized,
            attn_mask=src_mask,
            key_padding_mask=self_padding_mask,
            need_weights=False,
        )[0])
        if padding is not None:
            attended = attended.masked_fill(padding.unsqueeze(-1), 0.0)
        hidden = hidden + attended
        if memory.shape[1]:
            memory_padding = _padding_positions(memory_key_padding_mask)
            safe_memory = memory
            if memory_padding is not None:
                safe_memory = memory.masked_fill(memory_padding.unsqueeze(-1), 0.0)
            attended = self.cross_attn(
                self.cross_norm(hidden),
                safe_memory,
                safe_memory,
                key_padding_mask=_safe_padding_mask(memory_key_padding_mask),
                need_weights=False,
            )[0]
            if padding is not None:
                attended = attended.masked_fill(padding.unsqueeze(-1), 0.0)
            hidden = hidden + self.cross_dropout(attended)
        hidden = hidden + self._ff_block(self.norm2(hidden))
        if padding is not None:
            hidden = hidden.masked_fill(padding.unsqueeze(-1), 0.0)
        return hidden
