"""Causal self-attention with a learned bias for each past distance."""

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


class RelativePositionTransformerEncoderLayer(nn.TransformerEncoderLayer):
    """Keep the baseline projection names and learn head-specific distances.

    The window includes the current, already right-shifted Q input. Future
    positions and distances outside ``context_length`` remain inaccessible,
    regardless of the supplied mask or the learned bias. Parameters inherited
    from ``TransformerEncoderLayer`` keep their names for suffix LoRA adapters.
    """

    def __init__(self, *args, context_length: int, **kwargs) -> None:
        if type(context_length) is not int or context_length <= 0:
            raise ValueError("context_length must be a positive integer")
        super().__init__(*args, **kwargs)
        self.context_length = context_length
        weight = self.self_attn.in_proj_weight
        self.relative_position_bias = nn.Parameter(
            torch.zeros(
                self.self_attn.num_heads,
                context_length,
                device=weight.device,
                dtype=weight.dtype,
            )
        )
        # TransformerEncoder also inspects this guard before selecting its
        # nested/fused path. Such paths do not know about the distance bias.
        self.activation_relu_or_gelu = 0

    @staticmethod
    def _additive_mask(mask: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        if mask.dtype == torch.bool:
            return torch.zeros_like(mask, device=src.device, dtype=src.dtype).masked_fill(
                mask.to(device=src.device), float("-inf")
            )
        if not torch.is_floating_point(mask):
            raise ValueError("attention masks must be bool or floating point")
        return mask.to(device=src.device, dtype=src.dtype)

    def _relative_mask(
        self, src: torch.Tensor, src_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        unbatched = src.ndim == 2
        sequence_axis = 0 if unbatched or not self.self_attn.batch_first else 1
        length = src.shape[sequence_axis]
        batch = 1 if unbatched else src.shape[1 - sequence_axis]
        positions = torch.arange(length, device=src.device)
        distances = positions[:, None] - positions[None, :]
        blocked = (distances < 0) | (distances >= self.context_length)
        indices = distances.clamp(min=0, max=self.context_length - 1)
        bias = self.relative_position_bias[:, indices].to(dtype=src.dtype)
        bias = bias.unsqueeze(0).expand(batch, -1, -1, -1).reshape(
            batch * self.self_attn.num_heads, length, length
        )
        if src_mask is not None:
            if src_mask.shape not in (
                torch.Size((length, length)),
                torch.Size((batch * self.self_attn.num_heads, length, length)),
            ):
                raise ValueError("src_mask must have shape [S, S] or [B * H, S, S]")
            bias = bias + self._additive_mask(src_mask, src)
        # Mask after addition so even very large biases cannot reopen a key.
        return bias.masked_fill(blocked.unsqueeze(0), float("-inf"))

    def _relative_sa_block(
        self,
        src: torch.Tensor,
        mask: torch.Tensor,
        padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        unbatched = src.ndim == 2
        query = src.transpose(0, 1) if self.self_attn.batch_first and not unbatched else src
        # Call the functional implementation directly: PyTorch 2.0's native
        # MHA eval/no_grad path can convert additive masks to boolean masks,
        # silently discarding finite relative biases. The functional path also
        # reads parametrized weights, including training-time LoRA factors.
        attended, _ = F.multi_head_attention_forward(
            query,
            query,
            query,
            self.self_attn.embed_dim,
            self.self_attn.num_heads,
            self.self_attn.in_proj_weight,
            self.self_attn.in_proj_bias,
            self.self_attn.bias_k,
            self.self_attn.bias_v,
            self.self_attn.add_zero_attn,
            self.self_attn.dropout,
            self.self_attn.out_proj.weight,
            self.self_attn.out_proj.bias,
            training=self.self_attn.training,
            key_padding_mask=padding_mask,
            need_weights=False,
            attn_mask=mask,
            is_causal=False,
        )
        if self.self_attn.batch_first and not unbatched:
            attended = attended.transpose(0, 1)
        return self.dropout1(attended)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        # Causality is enforced in the explicit additive mask rather than
        # passed as an SDPA hint, which could omit the learned distance bias.
        if src.ndim not in (2, 3) or src.is_nested:
            raise ValueError("src must be a dense [S, D], [B, S, D] or [S, B, D] tensor")
        mask = self._relative_mask(src, src_mask)
        padding = (
            None
            if src_key_padding_mask is None
            else self._additive_mask(src_key_padding_mask, src)
        )
        hidden = src
        if self.norm_first:
            hidden = hidden + self._relative_sa_block(self.norm1(hidden), mask, padding)
            hidden = hidden + self._ff_block(self.norm2(hidden))
        else:
            hidden = self.norm1(hidden + self._relative_sa_block(hidden, mask, padding))
            hidden = self.norm2(hidden + self._ff_block(hidden))
        return hidden
