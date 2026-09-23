"""Decode-only compatibility for retired Q-history adapters (wire version 3).

Do not use this module for new training or encoding experiments.
"""

import json

import torch
from torch.nn import functional as F

from .model import ResidualOutputHead


HISTORY_FEATURE_DIM = 8


class LegacyHistoryHead(ResidualOutputHead):
    """Only instantiated when restoring an existing adapter-v3 container."""

    decode_only = True

    def __init__(self, d_model, residual_dim):
        super().__init__(d_model, residual_dim)
        self.down = torch.nn.Linear(d_model + HISTORY_FEATURE_DIM, residual_dim)

    def forward(self, packed):
        return F.linear(packed[..., :self.in_features], self.weight, self.bias) + self.up(
            F.gelu(self.down(packed), approximate="none"))

    def forward_with_quality(self, hidden, qualities, active_mask):
        return self(torch.cat((hidden, quality_history_features(qualities, active_mask)), dim=-1))


def history_feature_schema():
    return {
        "name": "causal_q_summary", "version": 1,
        "features": ["prev2", "prev3", "prev1_minus_prev2", "prefix_mean",
                     "prefix_std", "window_mean", "window_minus_prefix_mean", "history_fill"],
        "quality_scale": 41, "window": 8, "missing_lag": -1,
        "prefix_accumulation": "int64", "std": "population",
        "scope": "same_read_strict_prefix",
    }


def validate_history_feature_schema(schema):
    # Canonical JSON also distinguishes bool/int and float/int protocol values.
    try:
        valid = isinstance(schema, dict) and json.dumps(schema, sort_keys=True, allow_nan=False) == json.dumps(
            history_feature_schema(), sort_keys=True, allow_nan=False)
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError("unsupported quality history feature schema")


def quality_history_features(qualities, active_mask):
    """Return [batch, length, 8] FP32 features; never use current/future Q.

    During step decoding, unknown future qualities contain the padding id.
    Integer prefix sums ensure the result for a known prefix does not depend
    on floating-point scan grouping or values beyond that prefix.
    Missing exact lags are -1; empty summaries/delta are zero. Inactive rows
    are all zero. Means/std/delta use scale 41; fill = min(t, 8) / 8.
    """
    if qualities.ndim != 2 or qualities.shape != active_mask.shape:
        raise ValueError("quality history requires matching [batch, length] tensors")
    batch, length = qualities.shape
    if length == 0:
        return torch.zeros((batch, 0, HISTORY_FEATURE_DIM), device=qualities.device, dtype=torch.float32)
    q = torch.where(active_mask & (qualities >= 0) & (qualities <= 41), qualities, 0).to(torch.int64)
    zeros = torch.zeros((batch, 1), device=q.device, dtype=torch.int64)
    sums = torch.cat((zeros, q.cumsum(dim=1)[:, :-1]), dim=1)
    squares = torch.cat((zeros, (q * q).cumsum(dim=1)[:, :-1]), dim=1)
    t = torch.arange(length, device=q.device, dtype=torch.int64).unsqueeze(0)
    n = t.clamp_min(1)
    mean = sums.to(torch.float32) / n.to(torch.float32) / 41.0
    # Compute variance numerator in integers before converting to FP32.
    variance = (n * squares - sums * sums).clamp_min(0).to(torch.float32)
    std = torch.sqrt(variance / (n * n).to(torch.float32)) / 41.0
    starts = (t[0] - 8).clamp_min(0)
    window_sum = sums - sums.index_select(1, starts)
    window_mean = window_sum.to(torch.float32) / t.clamp(min=1, max=8).to(torch.float32) / 41.0

    def lag(k):
        values = torch.full_like(q, -41)
        if length > k:
            values[:, k:] = q[:, :-k]
        return values

    prev1, prev2, prev3 = lag(1), lag(2), lag(3)
    delta = torch.where(t >= 2, prev1 - prev2, 0).to(torch.float32) / 41.0
    features = torch.stack((prev2.to(torch.float32) / 41.0, prev3.to(torch.float32) / 41.0,
                            delta, mean, std, window_mean, window_mean - mean,
                            t.clamp(max=8).to(torch.float32).expand(batch, -1) / 8.0), dim=-1)
    return features * active_mask.unsqueeze(-1).to(torch.float32)
