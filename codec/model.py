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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "DirectQualityModelConfig":
        normalized = dict(values)
        for name in ("qmer_ks", "base_conv_kernels"):
            if name in normalized:
                normalized[name] = tuple(int(value) for value in normalized[name])
        return cls(**normalized)


def feature_schema() -> Dict[str, Any]:
    """Return the exact decoder-available feature contract saved in checkpoints."""

    return {
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
        self.base_encoder = BaseContextEncoder(self.config)
        combined_dim = (
            self.config.prev_q_embed_dim
            + len(QMER_KS) * self.config.qmer_embed_dim
            + self.config.base_context_dim
            + POSITION_FEATURE_DIM
        )
        self.input_projection = nn.Sequential(
            nn.Linear(combined_dim, self.config.d_model),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.config.d_model,
            nhead=self.config.num_heads,
            dim_feedforward=self.config.feedforward_dim,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=self.config.num_layers,
            norm=nn.LayerNorm(self.config.d_model),
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

    def _forward_impl(
        self,
        bases: torch.Tensor,
        qualities: torch.Tensor,
        lengths: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        sequence_length = bases.shape[1]
        if sequence_length == 0:
            return self.output_head.weight.new_empty(
                (bases.shape[0], 0, QUALITY_ALPHABET_SIZE)
            )
        prev_q, qmer_tokens = build_quality_history_tokens(qualities)
        pieces = [self.prev_q_embedding(prev_q)]
        pieces.extend(
            embedding(tokens)
            for embedding, tokens in zip(self.qmer_embeddings, qmer_tokens)
        )
        pieces.append(self.base_encoder(bases, active_mask))
        pieces.append(self._position_features(lengths, active_mask))
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
            hidden = self.transformer(
                hidden,
                mask=self._attention_mask(sequence_length, bases.device),
            )
        logits = self.output_head(hidden)
        return logits * active_mask.unsqueeze(-1).to(logits.dtype)

    def forward_full(
        self,
        bases: torch.Tensor,
        qualities: torch.Tensor,
        lengths: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Predict all positions with teacher-forced, strictly causal history."""

        self._validate_common_inputs(bases, lengths, active_mask)
        if qualities.shape != bases.shape:
            raise ValueError("qualities must have the same shape as bases")
        _validate_target_values(qualities, active_mask, name="qualities")
        return self._forward_impl(bases, qualities, lengths, active_mask)

    def forward_step(
        self,
        bases: torch.Tensor,
        quality_prefix: torch.Tensor,
        lengths: torch.Tensor,
        active_mask: torch.Tensor,
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
        logits = self._forward_impl(bases, qualities, lengths, active_mask)
        return logits[:, cycle, :]

    def forward(
        self,
        bases: torch.Tensor,
        qualities: torch.Tensor,
        lengths: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_full(bases, qualities, lengths, active_mask)
