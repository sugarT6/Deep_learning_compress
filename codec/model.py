"""Minimal no-SeqArc causal model for direct Q0..Q41 prediction."""

from __future__ import annotations

import math
import warnings
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .base_memory import BaseMemoryDecoderLayer, BaseMemoryEncoder
from .relative_attention import RelativePositionTransformerEncoderLayer
from .fastq_stream import (
    BASE_ALPHABET_SIZE,
    BASE_OTHER_ID,
    BASE_PAD_ID,
    PHRED_OFFSET,
    QUALITY_ALPHABET_SIZE,
    QUALITY_PAD_ID,
    FastqBatch,
)


MODEL_ARCHITECTURE = "direct-quality-causal-transformer"
MODEL_ARCHITECTURE_VERSION = 1
FEATURE_SCHEMA_VERSION = 1
Q_BOS_ID = QUALITY_ALPHABET_SIZE
Q_TOKEN_COUNT = QUALITY_ALPHABET_SIZE + 1
QMER_KS = (2, 3, 4)
QMER_BUCKET_BOUNDARIES = (10, 20, 25, 30, 35, 40)
QMER_BUCKET_COUNT = 7
QMER_BOS_BUCKET = QMER_BUCKET_COUNT
QMER_BASE = QMER_BUCKET_COUNT + 1
QMER_VOCAB_SIZE = QMER_BASE ** max(QMER_KS)
POSITION_FEATURE_DIM = 3
QUALITY_PREFIX_FEATURE_DIM = 4


@dataclass(frozen=True)
class DirectQualityModelConfig:
    architecture: str = MODEL_ARCHITECTURE
    architecture_version: int = MODEL_ARCHITECTURE_VERSION
    quality_alphabet_size: int = QUALITY_ALPHABET_SIZE
    phred_offset: int = PHRED_OFFSET
    qmer_ks: Tuple[int, ...] = QMER_KS
    prev_q_embed_dim: int = 16
    qmer_embed_dim: int = 8
    base_embed_dim: int = 16
    base_conv_kernels: Tuple[int, ...] = (3, 5, 7)
    base_conv_channels: int = 16
    base_context_dim: int = 32
    position_length_scale: float = 512.0
    d_model: int = 256
    num_heads: int = 4
    num_layers: int = 4
    feedforward_dim: int = 512
    context_length: int = 256
    dropout: float = 0.1
    base_encoder_type: str = "local_cnn"
    ffn_type: str = "gelu"
    quality_prefix_features: bool = False
    relative_position_bias: bool = False
    base_cross_attention: bool = False
    base_memory_dim: int = 64
    base_memory_heads: int = 4
    base_memory_layers: int = 2
    base_cross_attention_layers: int = 2

    def __post_init__(self) -> None:
        if self.architecture != MODEL_ARCHITECTURE:
            raise ValueError(f"unsupported model architecture {self.architecture!r}")
        if self.architecture_version != MODEL_ARCHITECTURE_VERSION:
            raise ValueError("unsupported model architecture version")
        if self.quality_alphabet_size != QUALITY_ALPHABET_SIZE:
            raise ValueError("direct-quality model must output exactly 42 classes")
        if self.phred_offset != PHRED_OFFSET:
            raise ValueError("direct-quality model requires Phred+33")
        if tuple(self.qmer_ks) != QMER_KS:
            raise ValueError("direct-quality model requires causal Q-mers k=2,3,4")
        if self.prev_q_embed_dim <= 0 or self.qmer_embed_dim <= 0:
            raise ValueError("quality embedding dimensions must be positive")
        if self.base_embed_dim <= 0 or self.base_conv_channels <= 0:
            raise ValueError("base embedding and convolution dimensions must be positive")
        if self.base_context_dim <= 0:
            raise ValueError("base_context_dim must be positive")
        if not self.base_conv_kernels or any(
            kernel <= 0 or kernel % 2 == 0 for kernel in self.base_conv_kernels
        ):
            raise ValueError("base convolution kernels must be positive odd integers")
        if self.position_length_scale <= 0:
            raise ValueError("position_length_scale must be positive")
        if self.d_model <= 0 or self.num_heads <= 0:
            raise ValueError("d_model and num_heads must be positive")
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if self.num_layers <= 0 or self.feedforward_dim <= 0:
            raise ValueError("Transformer layer dimensions must be positive")
        if self.context_length <= 0:
            raise ValueError("context_length must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.base_encoder_type not in ("local_cnn", "dilated_cnn"):
            raise ValueError("base_encoder_type must be local_cnn or dilated_cnn")
        if self.ffn_type not in ("gelu", "swiglu"):
            raise ValueError("ffn_type must be gelu or swiglu")
        if not isinstance(self.quality_prefix_features, bool):
            raise ValueError("quality_prefix_features must be bool")
        for name in ("relative_position_bias", "base_cross_attention"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be bool")
        if self.relative_position_bias and self.base_cross_attention:
            raise ValueError("relative position and base cross-attention are independent experiments")
        if (self.relative_position_bias or self.base_cross_attention) and self.ffn_type != "gelu":
            raise ValueError("conditioning architecture experiments require the GELU FFN")
        for name in ("base_memory_dim", "base_memory_heads", "base_memory_layers",
                     "base_cross_attention_layers"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.base_memory_dim % self.base_memory_heads:
            raise ValueError("base_memory_dim must be divisible by base_memory_heads")
        if self.base_cross_attention and self.base_cross_attention_layers > self.num_layers:
            raise ValueError("base_cross_attention_layers cannot exceed num_layers")

    def to_dict(self) -> Dict[str, Any]:
        values = asdict(self)
        # Keep the serialized legacy configuration identical. Checkpoint and
        # strict-resume contracts from before these opt-in variants stay valid.
        for name, default in (
            ("base_encoder_type", "local_cnn"),
            ("ffn_type", "gelu"),
            ("quality_prefix_features", False),
            ("relative_position_bias", False),
            ("base_cross_attention", False),
            ("base_memory_dim", 64),
            ("base_memory_heads", 4),
            ("base_memory_layers", 2),
            ("base_cross_attention_layers", 2),
        ):
            if values[name] == default:
                values.pop(name)
        return values

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "DirectQualityModelConfig":
        normalized = dict(values)
        for name in ("qmer_ks", "base_conv_kernels"):
            if name in normalized:
                normalized[name] = tuple(int(value) for value in normalized[name])
        return cls(**normalized)


def feature_schema(config: Optional[DirectQualityModelConfig] = None) -> Dict[str, Any]:
    """Return the exact decoder-available feature contract saved in checkpoints."""

    schema = {
        "version": FEATURE_SCHEMA_VERSION,
        "prediction_target": "quality_id",
        "quality_classes": list(range(QUALITY_ALPHABET_SIZE)),
        "phred_offset": PHRED_OFFSET,
        "features": {
            "prev_q": {"bos_id": Q_BOS_ID, "strictly_shifted": True},
            "causal_qmer": {
                "ks": list(QMER_KS),
                "bucket_boundaries": list(QMER_BUCKET_BOUNDARIES),
                "bos_bucket": QMER_BOS_BUCKET,
                "base": QMER_BASE,
                "vocab_size": QMER_VOCAB_SIZE,
            },
            "complete_base_sequence": {
                "base_ids": "A=0,C=1,G=2,T=3,N=4,other=5,pad=6",
                "future_base_allowed": True,
            },
            "position_read_length": {
                "values": ["absolute_scaled", "relative", "read_length_scaled"]
            },
            "active_mask": {"padding_enters_loss": False},
        },
        "excluded": [
            "SeqArc",
            "q_hat",
            "prev_r",
            "R-mer",
            "quality_distribution_prior",
            "platform_embedding",
            "accession_id",
        ],
    }
    if config is not None and config.quality_prefix_features:
        schema["version"] = FEATURE_SCHEMA_VERSION + 1
        schema["features"]["causal_quality_prefix"] = {
            "values": ["mean", "variance", "low_quality_fraction", "low_quality_run_length"],
            "history": "qualities_before_prediction_position_only",
            "empty_prefix": "all_zero",
            "variance": "population",
            "low_quality_threshold_exclusive": 20,
            "quality_scale": QUALITY_ALPHABET_SIZE - 1,
            "run_length_scale": config.position_length_scale,
            "side_information_bytes": 0,
        }
    if config is not None and config.relative_position_bias:
        schema["version"] = FEATURE_SCHEMA_VERSION + 2
        schema["features"]["relative_position_bias"] = {
            "type": "learned_per_head_causal_distance",
            "distance_bins": config.context_length,
            "future_quality_allowed": False,
            "side_information_bytes": 0,
        }
    if config is not None and config.base_cross_attention:
        schema["version"] = FEATURE_SCHEMA_VERSION + 2
        schema["features"]["base_memory"] = {
            "inputs": ["complete_base_sequence", "position_read_length"],
            "bidirectional": True,
            "quality_input": False,
            "padding_masked": True,
            "dimension": config.base_memory_dim,
            "heads": config.base_memory_heads,
            "encoder_layers": config.base_memory_layers,
            "cross_attention_layers": list(range(
                config.num_layers - config.base_cross_attention_layers, config.num_layers
            )),
            "side_information_bytes": 0,
        }
    return schema


def fastq_batch_to_tensors(
    batch: FastqBatch, device: torch.device
) -> Dict[str, torch.Tensor]:
    return {
        "bases": torch.as_tensor(batch.bases, dtype=torch.long, device=device),
        "qualities": torch.as_tensor(
            batch.qualities, dtype=torch.long, device=device
        ),
        "lengths": torch.as_tensor(batch.lengths, dtype=torch.long, device=device),
        "active_mask": torch.as_tensor(
            batch.active_mask, dtype=torch.bool, device=device
        ),
    }


def _validate_target_values(
    qualities: torch.Tensor, active_mask: torch.Tensor, *, name: str
) -> None:
    active_values = qualities[active_mask]
    if active_values.numel():
        minimum = int(active_values.min().item())
        maximum = int(active_values.max().item())
        if minimum < 0 or maximum >= QUALITY_ALPHABET_SIZE:
            raise ValueError(
                f"{name} contains quality id outside Q0-Q41: Q{minimum}..Q{maximum}"
            )
    inactive_values = qualities[~active_mask]
    if inactive_values.numel() and torch.any(inactive_values.ne(QUALITY_PAD_ID)):
        raise ValueError(f"{name} padding positions must use id {QUALITY_PAD_ID}")


def build_quality_history_tokens(
    qualities: torch.Tensor,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
    """Build exact previous-Q and bucketed causal 2/3/4-mer tokens."""

    if qualities.ndim != 2:
        raise ValueError("qualities must have shape [batch, sequence]")
    batch_size, sequence_length = qualities.shape
    prev_q = torch.full_like(qualities, Q_BOS_ID)
    if sequence_length > 1:
        previous_values = qualities[:, :-1]
        prev_q[:, 1:] = torch.where(
            previous_values.lt(QUALITY_ALPHABET_SIZE),
            previous_values,
            torch.full_like(previous_values, Q_BOS_ID),
        )

    boundaries = torch.tensor(
        QMER_BUCKET_BOUNDARIES, device=qualities.device, dtype=qualities.dtype
    )
    buckets = (qualities.unsqueeze(-1) >= boundaries).sum(dim=-1)
    max_k = max(QMER_KS)
    padded = torch.full(
        (batch_size, max_k + sequence_length),
        QMER_BOS_BUCKET,
        dtype=torch.long,
        device=qualities.device,
    )
    padded[:, max_k:] = buckets
    qmer_tokens = []
    for k in QMER_KS:
        code = torch.zeros_like(qualities)
        start = max_k - k
        for digit in range(k):
            history = padded[:, start + digit : start + digit + sequence_length]
            code = code * QMER_BASE + history
        qmer_tokens.append(code)
    return prev_q, tuple(qmer_tokens)


def build_quality_prefix_features(
    qualities: torch.Tensor,
    active_mask: torch.Tensor,
    position_length_scale: float = 512.0,
) -> torch.Tensor:
    """Four decoder-reconstructible statistics of Q[:t], never Q[t:].

    Mean and population variance are scaled by Q41 and Q41 squared. The
    consecutive-low-Q length is scaled like the absolute cycle feature.
    Padded positions and the empty prefix are exactly zero.
    """
    if qualities.ndim != 2 or active_mask.shape != qualities.shape:
        raise ValueError("prefix qualities and active_mask must have shape [batch, sequence]")
    if active_mask.dtype != torch.bool:
        raise ValueError("prefix active_mask must be bool")
    if position_length_scale <= 0:
        raise ValueError("position_length_scale must be positive")
    batch_size, sequence_length = qualities.shape
    if sequence_length == 0:
        return torch.zeros(
            (batch_size, 0, QUALITY_PREFIX_FEATURE_DIM),
            dtype=torch.float32,
            device=qualities.device,
        )
    mask = active_mask.to(torch.float32)
    values = qualities.to(torch.float32) * mask
    zeros = values.new_zeros((batch_size, 1))

    def exclusive_cumsum(tensor: torch.Tensor) -> torch.Tensor:
        return torch.cat((zeros, tensor.cumsum(dim=1)[:, :-1]), dim=1)

    counts = exclusive_cumsum(mask).clamp_min(1.0)
    mean = exclusive_cumsum(values) / counts
    variance = (exclusive_cumsum(values.square()) / counts - mean.square()).clamp_min(0.0)
    low = qualities.lt(20) & active_mask
    low_fraction = exclusive_cumsum(low.to(torch.float32)) / counts

    # At each position, record the last non-low cycle. cummax reconstructs a
    # run without a Python loop, and shifting excludes the current target Q.
    cycles = torch.arange(1, sequence_length + 1, device=qualities.device).unsqueeze(0)
    reset_cycles = torch.where(low, torch.zeros_like(cycles), cycles)
    run_including_current = cycles - reset_cycles.cummax(dim=1).values
    previous_run = torch.cat((zeros, run_including_current[:, :-1].to(torch.float32)), dim=1)
    quality_scale = float(QUALITY_ALPHABET_SIZE - 1)
    features = torch.stack(
        (
            mean / quality_scale,
            variance / (quality_scale * quality_scale),
            low_fraction,
            previous_run / position_length_scale,
        ),
        dim=-1,
    )
    return features * mask.unsqueeze(-1)


def masked_cross_entropy(
    logits: torch.Tensor,
    qualities: torch.Tensor,
    active_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean natural-log cross entropy over active quality symbols only."""

    if logits.shape[:-1] != qualities.shape or qualities.shape != active_mask.shape:
        raise ValueError("logit, quality, and active-mask shapes do not agree")
    if logits.shape[-1] != QUALITY_ALPHABET_SIZE:
        raise ValueError("direct-quality logits must contain 42 classes")
    _validate_target_values(qualities, active_mask, name="targets")
    if not torch.any(active_mask):
        raise ValueError("cannot compute loss for a batch without quality symbols")
    return F.cross_entropy(logits[active_mask], qualities[active_mask], reduction="mean")


def theoretical_bits(
    logits: torch.Tensor,
    qualities: torch.Tensor,
    active_mask: torch.Tensor,
) -> Tuple[float, int]:
    """Return summed ideal code length in bits and active symbol count."""

    if logits.shape[:-1] != qualities.shape or qualities.shape != active_mask.shape:
        raise ValueError("logit, quality, and active-mask shapes do not agree")
    if logits.shape[-1] != QUALITY_ALPHABET_SIZE:
        raise ValueError("direct-quality logits must contain 42 classes")
    _validate_target_values(qualities, active_mask, name="targets")
    symbol_count = int(active_mask.sum().item())
    if symbol_count == 0:
        return 0.0, 0
    log_probabilities = F.log_softmax(logits[active_mask], dim=-1)
    selected = log_probabilities.gather(
        1, qualities[active_mask].unsqueeze(1)
    ).squeeze(1)
    total_bits = float((-selected.sum() / math.log(2.0)).item())
    return total_bits, symbol_count


class BaseContextEncoder(nn.Module):
    """Centered convolutions over the complete, decoder-known base sequence."""

    def __init__(self, config: DirectQualityModelConfig) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            BASE_ALPHABET_SIZE, config.base_embed_dim, padding_idx=BASE_PAD_ID
        )
        self.convolutions = nn.ModuleList(
            [
                nn.Conv1d(
                    config.base_embed_dim,
                    config.base_conv_channels,
                    kernel_size=kernel,
                    padding=kernel // 2,
                )
                for kernel in config.base_conv_kernels
            ]
        )
        self.projection = nn.Sequential(
            nn.Linear(
                config.base_conv_channels * len(config.base_conv_kernels),
                config.base_context_dim,
            ),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

    def forward(self, bases: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(bases).transpose(1, 2)
        branches = [F.gelu(convolution(embedded)) for convolution in self.convolutions]
        context = self.projection(torch.cat(branches, dim=1).transpose(1, 2))
        return context * active_mask.unsqueeze(-1).to(context.dtype)


class DilatedBaseResidualBlock(nn.Module):
    """Position-wise normalization and a masked, bidirectional convolution."""

    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.convolution = nn.Conv1d(
            channels, channels, kernel_size=3, dilation=dilation, padding=dilation
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # LN has an affine bias. Mask before convolution as well as after the
        # residual, so artificial padding never reenters active positions.
        branch = self.norm(hidden) * mask
        branch = self.convolution(branch.transpose(1, 2)).transpose(1, 2)
        branch = self.dropout(F.gelu(branch))
        return (hidden + branch) * mask


class DilatedBaseContextEncoder(nn.Module):
    """Encode known bases with a 255-cycle bidirectional receptive field.

    The quality Transformer remains strictly causal; future *bases* are
    available at decode time and may be used by this independent encoder.
    """

    def __init__(self, config: DirectQualityModelConfig) -> None:
        super().__init__()
        self.context_dim = config.base_context_dim
        self.embedding = nn.Embedding(
            BASE_ALPHABET_SIZE, config.base_embed_dim, padding_idx=BASE_PAD_ID
        )
        self.projection = nn.Linear(config.base_embed_dim, config.base_context_dim)
        self.blocks = nn.ModuleList(
            DilatedBaseResidualBlock(config.base_context_dim, dilation, config.dropout)
            for dilation in (1, 2, 4, 8, 16, 32, 64)
        )
        self.norm = nn.LayerNorm(config.base_context_dim)

    def forward(self, bases: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
        if bases.shape[1] == 0:
            return self.projection.weight.new_zeros((bases.shape[0], 0, self.context_dim))
        mask = active_mask.unsqueeze(-1).to(self.projection.weight.dtype)
        hidden = self.projection(self.embedding(bases)) * mask
        for block in self.blocks:
            hidden = block(hidden, mask)
        return self.norm(hidden) * mask


class SwiGLUTransformerEncoderLayer(nn.TransformerEncoderLayer):
    """Keep familiar FFN parameter names and add an independently learned gate."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.linear_gate = nn.Linear(
            self.linear1.in_features,
            self.linear1.out_features,
            device=self.linear1.weight.device,
            dtype=self.linear1.weight.dtype,
        )
        # The fused Transformer eval path hardcodes ReLU/GELU and would silently
        # bypass _ff_block and linear_gate. Its documented internal guard must
        # remain false, including inference under torch.no_grad().
        self.activation_relu_or_gelu = 0

    def _ff_block(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = self.linear1(hidden) * F.silu(self.linear_gate(hidden))
        return self.dropout2(self.linear2(self.dropout(hidden)))


class ResidualOutputHead(nn.Linear):
    """Linear logits plus a small GELU residual; installed only by a wire adapter.

    Keep weight/bias at the root for the base linear head. Zero-initialized up
    projection makes the initial function exactly the supplied linear head.
    """

    def __init__(self, d_model: int, residual_dim: int):
        super().__init__(d_model, QUALITY_ALPHABET_SIZE)
        self.down = nn.Linear(d_model, residual_dim)
        self.up = nn.Linear(residual_dim, QUALITY_ALPHABET_SIZE)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden, self.weight, self.bias) + self.up(
            F.gelu(self.down(hidden), approximate="none")
        )


class CrossLayerOutputHead(ResidualOutputHead):
    """Residual head on final features plus a zero-start branch on layer 3."""

    def __init__(self, d_model, residual_dim, cross_dim=16):
        super().__init__(d_model, residual_dim)
        self.cross_down = nn.Linear(d_model, cross_dim)
        self.cross_up = nn.Linear(cross_dim, QUALITY_ALPHABET_SIZE)
        nn.init.zeros_(self.cross_up.weight)
        nn.init.zeros_(self.cross_up.bias)

    def forward(self, packed):
        return super().forward(packed[..., :self.in_features]) + self.cross_up(
            F.gelu(self.cross_down(packed[..., self.in_features:]), approximate="none"))


class DirectQualityTransformer(nn.Module):
    """Causal quality model using only decoder-synchronous stage-B features."""

    def __init__(self, config: Optional[DirectQualityModelConfig] = None) -> None:
        super().__init__()
        self.config = config or DirectQualityModelConfig()
        self.prev_q_embedding = nn.Embedding(
            Q_TOKEN_COUNT, self.config.prev_q_embed_dim
        )
        self.qmer_embeddings = nn.ModuleList(
            [
                nn.Embedding(QMER_VOCAB_SIZE, self.config.qmer_embed_dim)
                for _ in QMER_KS
            ]
        )
        self.base_encoder = (
            BaseContextEncoder(self.config)
            if self.config.base_encoder_type == "local_cnn"
            else DilatedBaseContextEncoder(self.config)
        )
        combined_dim = (
            self.config.prev_q_embed_dim
            + len(QMER_KS) * self.config.qmer_embed_dim
            + self.config.base_context_dim
            + POSITION_FEATURE_DIM
            + (QUALITY_PREFIX_FEATURE_DIM if self.config.quality_prefix_features else 0)
        )
        self.input_projection = nn.Sequential(
            nn.Linear(combined_dim, self.config.d_model),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
        )
        layer_class = (
            nn.TransformerEncoderLayer
            if self.config.ffn_type == "gelu"
            else SwiGLUTransformerEncoderLayer
        )
        layer_options = dict(
            d_model=self.config.d_model,
            nhead=self.config.num_heads,
            dim_feedforward=self.config.feedforward_dim,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        if self.config.relative_position_bias:
            layer_class = RelativePositionTransformerEncoderLayer
        layer = layer_class(
            **layer_options,
            **({"context_length": self.config.context_length}
               if self.config.relative_position_bias else {}),
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=self.config.num_layers,
            norm=nn.LayerNorm(self.config.d_model),
        )
        if self.config.base_cross_attention:
            self.base_memory_encoder = BaseMemoryEncoder(self.config)
            first_cross = self.config.num_layers - self.config.base_cross_attention_layers
            for index in range(first_cross, self.config.num_layers):
                self.transformer.layers[index] = BaseMemoryDecoderLayer(
                    **layer_options, memory_dim=self.config.base_memory_dim
                )
        self.output_head = nn.Linear(
            self.config.d_model, QUALITY_ALPHABET_SIZE
        )

    def _validate_common_inputs(
        self,
        bases: torch.Tensor,
        lengths: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> None:
        if bases.ndim != 2:
            raise ValueError("bases must have shape [batch, sequence]")
        if lengths.ndim != 1 or lengths.shape[0] != bases.shape[0]:
            raise ValueError("lengths must have shape [batch]")
        if active_mask.shape != bases.shape or active_mask.dtype != torch.bool:
            raise ValueError("active_mask must be bool with the same shape as bases")
        if lengths.numel() and (
            int(lengths.min().item()) < 0
            or int(lengths.max().item()) > bases.shape[1]
        ):
            raise ValueError("read length outside padded sequence width")
        expected_mask = (
            torch.arange(bases.shape[1], device=bases.device).unsqueeze(0)
            < lengths.unsqueeze(1)
        )
        if not torch.equal(active_mask, expected_mask):
            raise ValueError("active_mask does not match lengths")
        if bases.numel():
            minimum = int(bases.min().item())
            maximum = int(bases.max().item())
            if minimum < 0 or maximum >= BASE_ALPHABET_SIZE:
                raise ValueError("base id outside supported range")
            if torch.any(bases[active_mask].gt(BASE_OTHER_ID)):
                raise ValueError("active base positions may not contain padding")
            if torch.any(bases[~active_mask].ne(BASE_PAD_ID)):
                raise ValueError("inactive base positions must contain padding")

    def _position_features(
        self, lengths: torch.Tensor, active_mask: torch.Tensor
    ) -> torch.Tensor:
        sequence_length = active_mask.shape[1]
        position = torch.arange(
            sequence_length, device=lengths.device, dtype=torch.float32
        ).unsqueeze(0)
        length_float = lengths.to(torch.float32).unsqueeze(1)
        relative_denominator = torch.clamp(length_float - 1.0, min=1.0)
        features = torch.stack(
            (
                position.expand(lengths.shape[0], -1)
                / self.config.position_length_scale,
                position.expand(lengths.shape[0], -1) / relative_denominator,
                length_float.expand(-1, sequence_length)
                / self.config.position_length_scale,
            ),
            dim=-1,
        )
        return features * active_mask.unsqueeze(-1).to(features.dtype)

    def _attention_mask(self, sequence_length: int, device: torch.device) -> torch.Tensor:
        query = torch.arange(sequence_length, device=device).unsqueeze(1)
        key = torch.arange(sequence_length, device=device).unsqueeze(0)
        distance = query - key
        return (distance < 0) | (distance >= self.config.context_length)

    def _hidden_impl(
        self,
        bases: torch.Tensor,
        qualities: torch.Tensor,
        lengths: torch.Tensor,
        active_mask: torch.Tensor,
        read_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if hasattr(self, "read_state_conditioning"):
            self._validate_read_states(read_states, bases.shape[0])
            if read_states.device != bases.device:
                raise ValueError("read states and bases must be on the same device")
        elif read_states is not None:
            raise ValueError("read states require a state-conditioned adapter")
        sequence_length = bases.shape[1]
        if sequence_length == 0:
            return self.output_head.weight.new_empty(
                (bases.shape[0], 0, self.config.d_model)
            )
        prev_q, qmer_tokens = build_quality_history_tokens(qualities)
        pieces = [self.prev_q_embedding(prev_q)]
        pieces.extend(
            embedding(tokens)
            for embedding, tokens in zip(self.qmer_embeddings, qmer_tokens)
        )
        pieces.append(self.base_encoder(bases, active_mask))
        position_features = self._position_features(lengths, active_mask)
        pieces.append(position_features)
        if self.config.quality_prefix_features:
            pieces.append(
                build_quality_prefix_features(
                    qualities, active_mask, self.config.position_length_scale
                )
            )
        hidden = self.input_projection(torch.cat(pieces, dim=-1))

        # Active positions are a contiguous prefix of every read. A causal
        # query at an active cycle can therefore see only active keys, so a
        # separate key-padding mask cannot change any retained output.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Converting mask without torch.bool dtype to bool.*",
                category=UserWarning,
            )
            mask = self._attention_mask(sequence_length, bases.device)
            if self.config.base_cross_attention:
                memory = self.base_memory_encoder(bases, active_mask, position_features)
                for index, layer in enumerate(self.transformer.layers):
                    if (hasattr(self, "read_state_conditioning")
                            and index in self.read_state_conditioning.layers):
                        hidden = self.read_state_conditioning(hidden, read_states, index)
                    if isinstance(layer, BaseMemoryDecoderLayer):
                        hidden = layer(
                            hidden, src_mask=mask, memory=memory,
                            memory_key_padding_mask=~active_mask,
                        )
                    else:
                        hidden = layer(hidden, src_mask=mask)
                hidden = self.transformer.norm(hidden)
            elif hasattr(self, "read_state_conditioning"):
                for index, layer in enumerate(self.transformer.layers):
                    if index in self.read_state_conditioning.layers:
                        hidden = self.read_state_conditioning(hidden, read_states, index)
                    hidden = layer(hidden, src_mask=mask)
                hidden = self.transformer.norm(hidden)
            else:
                hidden = self.transformer(hidden, mask=mask)
        return hidden

    def _validate_read_states(self, read_states, batch_size):
        class_count = self.read_state_conditioning.class_count
        if (not isinstance(read_states, torch.Tensor) or read_states.dtype != torch.long
                or read_states.shape != (batch_size,)
                or torch.any((read_states < 0) | (read_states >= class_count))):
            raise ValueError(f"conditioned inference requires one read-state ID in [0, {class_count - 1}] per read")

    def _forward_impl(self, bases, qualities, lengths, active_mask, read_states=None):
        hidden = (self._cross_hidden_impl(bases, qualities, lengths, active_mask, read_states)
                  if isinstance(self.output_head, CrossLayerOutputHead)
                  else self._hidden_impl(bases, qualities, lengths, active_mask, read_states))
        logits = self.output_head(hidden)
        return logits * active_mask.unsqueeze(-1).to(logits.dtype)

    def forward_features(self, bases, qualities, lengths, active_mask, read_states=None):
        """Return causal output-head inputs, for frozen-feature adaptation only."""
        self._validate_common_inputs(bases, lengths, active_mask)
        if qualities.shape != bases.shape:
            raise ValueError("qualities must have the same shape as bases")
        _validate_target_values(qualities, active_mask, name="qualities")
        return self._hidden_impl(bases, qualities, lengths, active_mask, read_states)

    def forward_cross_layer_features(self, bases, qualities, lengths, active_mask, read_states=None):
        self._validate_common_inputs(bases, lengths, active_mask)
        if qualities.shape != bases.shape:
            raise ValueError("qualities must have the same shape as bases")
        _validate_target_values(qualities, active_mask, name="qualities")
        return self._cross_hidden_impl(bases, qualities, lengths, active_mask, read_states)

    def _cross_hidden_impl(self, bases, qualities, lengths, active_mask, read_states=None):
        """Pack final features and per-position, affine-free LN of block 3.

        Capture the existing forward instead of rerunning the backbone. The
        temporary hook is always removed, including on failed inference.
        """
        if self.config.num_layers < 4:
            raise ValueError("cross-layer adaptation requires at least 4 Transformer layers")
        if bases.shape[1] == 0:
            return self._hidden_impl(bases, qualities, lengths, active_mask, read_states).repeat(1, 1, 2)
        captured = []
        def capture(module, inputs, output):
            captured.append(output)
        handle = self.transformer.layers[2].register_forward_hook(capture)
        try:
            final = self._hidden_impl(bases, qualities, lengths, active_mask, read_states)
        finally:
            handle.remove()
        middle = F.layer_norm(captured[0], (self.config.d_model,), weight=None, bias=None, eps=1e-5)
        return torch.cat((final, middle), dim=-1)

    def forward_full(
        self,
        bases: torch.Tensor,
        qualities: torch.Tensor,
        lengths: torch.Tensor,
        active_mask: torch.Tensor,
        read_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict all positions with teacher-forced, strictly causal history."""

        self._validate_common_inputs(bases, lengths, active_mask)
        if qualities.shape != bases.shape:
            raise ValueError("qualities must have the same shape as bases")
        _validate_target_values(qualities, active_mask, name="qualities")
        return self._forward_impl(bases, qualities, lengths, active_mask, read_states)

    def forward_step(
        self,
        bases: torch.Tensor,
        quality_prefix: torch.Tensor,
        lengths: torch.Tensor,
        active_mask: torch.Tensor,
        read_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict one cycle from already-decoded quality history.

        The first implementation deliberately recomputes the full padded
        sequence.  Keeping the same tensor shape and masks as ``forward_full``
        makes their eval-mode outputs directly comparable before a future KV
        cache optimization is attempted.
        """

        self._validate_common_inputs(bases, lengths, active_mask)
        if quality_prefix.ndim != 2 or quality_prefix.shape[0] != bases.shape[0]:
            raise ValueError("quality_prefix must have shape [batch, decoded_cycles]")
        cycle = quality_prefix.shape[1]
        if cycle >= bases.shape[1]:
            raise ValueError("quality prefix already covers the padded sequence")
        prefix_mask = active_mask[:, :cycle]
        _validate_target_values(quality_prefix, prefix_mask, name="quality_prefix")
        qualities = torch.full_like(bases, QUALITY_PAD_ID)
        qualities[:, :cycle] = quality_prefix
        logits = self._forward_impl(bases, qualities, lengths, active_mask, read_states)
        return logits[:, cycle, :]

    def forward(
        self,
        bases: torch.Tensor,
        qualities: torch.Tensor,
        lengths: torch.Tensor,
        active_mask: torch.Tensor,
        read_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.forward_full(bases, qualities, lengths, active_mask, read_states)
