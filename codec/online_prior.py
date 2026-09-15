"""Deterministic causal online file prior for Q0--Q41 qualities.

The state is reset for every FASTQ file and is updated only after a complete
codec batch.  Within the next batch it provides a hierarchical distribution:

``cycle_bin, prev_q -> prev_q -> global q -> uniform``.

All probability calculations use finite CPU float64 values.  Integer counts
use int64 and only active qualities are observed.
"""

from __future__ import annotations

import math
import operator
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from .fastq_stream import QUALITY_ALPHABET_SIZE, QUALITY_PAD_ID


ONLINE_PRIOR_VERSION = 1
ONLINE_PRIOR_PROFILE = "causal_online_hierarchical_v1"
ONLINE_PRIOR_UPDATE_GRANULARITY = "completed_batch"
ONLINE_PRIOR_FLOAT_CONTRACT = "finite_cpu_float64"
ONLINE_PRIOR_COUNT_DTYPE = "int64"
ONLINE_PRIOR_BACKOFF_ORDER = (
    "cycle_bin_prev_q",
    "prev_q",
    "global_q",
    "uniform",
)
ONLINE_PRIOR_FUSION = "logit_adjustment_prior_to_uniform_ratio"
BOS_QUALITY_ID = QUALITY_ALPHABET_SIZE
PREV_Q_CONTEXTS = QUALITY_ALPHABET_SIZE + 1


class OnlinePriorError(ValueError):
    """Raised when an online-prior configuration or state input is invalid."""


def _positive_finite(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise OnlinePriorError(f"{name} must be a positive finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise OnlinePriorError(f"{name} must be a positive finite number") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise OnlinePriorError(f"{name} must be a positive finite number")
    return result


@dataclass(frozen=True)
class OnlinePriorConfig:
    """Version-1 hierarchical prior and neural-fusion parameters.

    A backoff strength is the effective sample size assigned to the parent
    distribution.  The default global strength of 42 is exactly add-one
    smoothing because its parent is uniform over 42 qualities.
    """

    cycle_bin_width: int = 8
    global_backoff_strength: float = 42.0
    prev_q_backoff_strength: float = 42.0
    cycle_backoff_strength: float = 42.0
    prior_weight: float = 0.25

    def __post_init__(self) -> None:
        if isinstance(self.cycle_bin_width, (bool, np.bool_)):
            raise OnlinePriorError("cycle_bin_width must be a positive integer")
        try:
            width = operator.index(self.cycle_bin_width)
        except TypeError as exc:
            raise OnlinePriorError(
                "cycle_bin_width must be a positive integer"
            ) from exc
        if width <= 0:
            raise OnlinePriorError("cycle_bin_width must be a positive integer")
        object.__setattr__(self, "cycle_bin_width", width)
        for name in (
            "global_backoff_strength",
            "prev_q_backoff_strength",
            "cycle_backoff_strength",
        ):
            object.__setattr__(
                self, name, _positive_finite(getattr(self, name), name=name)
            )
        if isinstance(self.prior_weight, (bool, np.bool_)):
            raise OnlinePriorError("prior_weight must be strictly between 0 and 1")
        try:
            weight = float(self.prior_weight)
        except (TypeError, ValueError) as exc:
            raise OnlinePriorError(
                "prior_weight must be strictly between 0 and 1"
            ) from exc
        if not math.isfinite(weight) or not 0.0 < weight < 1.0:
            raise OnlinePriorError("prior_weight must be strictly between 0 and 1")
        object.__setattr__(self, "prior_weight", weight)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "OnlinePriorConfig":
        if not isinstance(values, Mapping):
            raise OnlinePriorError("online-prior config must be an object")
        expected = {
            "cycle_bin_width",
            "global_backoff_strength",
            "prev_q_backoff_strength",
            "cycle_backoff_strength",
            "prior_weight",
        }
        if set(values) != expected:
            missing = sorted(expected - set(values))
            extra = sorted(set(values) - expected)
            raise OnlinePriorError(
                f"online-prior config fields mismatch: missing={missing}, extra={extra}"
            )
        return cls(**dict(values))

    def to_profile_metadata(self) -> Dict[str, Any]:
        return {
            "name": ONLINE_PRIOR_PROFILE,
            "version": ONLINE_PRIOR_VERSION,
            "update_granularity": ONLINE_PRIOR_UPDATE_GRANULARITY,
            "float_contract": ONLINE_PRIOR_FLOAT_CONTRACT,
            "count_dtype": ONLINE_PRIOR_COUNT_DTYPE,
            "backoff_order": list(ONLINE_PRIOR_BACKOFF_ORDER),
            "fusion": ONLINE_PRIOR_FUSION,
            "config": self.to_dict(),
        }

    @classmethod
    def from_profile_metadata(cls, values: Mapping[str, Any]) -> "OnlinePriorConfig":
        if not isinstance(values, Mapping):
            raise OnlinePriorError("probability_profile must be an object")
        expected = {
            "name",
            "version",
            "update_granularity",
            "float_contract",
            "count_dtype",
            "backoff_order",
            "fusion",
            "config",
        }
        if set(values) != expected:
            missing = sorted(expected - set(values))
            extra = sorted(set(values) - expected)
            raise OnlinePriorError(
                f"probability_profile fields mismatch: missing={missing}, extra={extra}"
            )
        if values["name"] != ONLINE_PRIOR_PROFILE:
            raise OnlinePriorError("unsupported probability profile name")
        if (
            type(values["version"]) is not int
            or values["version"] != ONLINE_PRIOR_VERSION
        ):
            raise OnlinePriorError("unsupported online-prior version")
        if values["update_granularity"] != ONLINE_PRIOR_UPDATE_GRANULARITY:
            raise OnlinePriorError("unsupported online-prior update granularity")
        if values["float_contract"] != ONLINE_PRIOR_FLOAT_CONTRACT:
            raise OnlinePriorError("unsupported online-prior float contract")
        if values["count_dtype"] != ONLINE_PRIOR_COUNT_DTYPE:
            raise OnlinePriorError("unsupported online-prior count dtype")
        if not isinstance(values["backoff_order"], list) or tuple(
            values["backoff_order"]
        ) != ONLINE_PRIOR_BACKOFF_ORDER:
            raise OnlinePriorError("unsupported online-prior backoff order")
        if values["fusion"] != ONLINE_PRIOR_FUSION:
            raise OnlinePriorError("unsupported online-prior fusion")
        return cls.from_dict(values["config"])


DEFAULT_ONLINE_PRIOR_CONFIG = OnlinePriorConfig()


def _integer_matrix(values: Sequence[Sequence[int]], *, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 2:
        raise OnlinePriorError(f"{name} must be a two-dimensional array")
    if array.dtype.kind not in "iu":
        raise OnlinePriorError(f"{name} must contain integers")
    return array.astype(np.int64, copy=False)


class OnlinePriorState:
    """Cumulative prior counts visible only to later completed batches."""

    def __init__(self, config: OnlinePriorConfig = DEFAULT_ONLINE_PRIOR_CONFIG) -> None:
        if not isinstance(config, OnlinePriorConfig):
            raise OnlinePriorError("config must be an OnlinePriorConfig")
        self.config = config
        self.global_counts = np.zeros(QUALITY_ALPHABET_SIZE, dtype=np.int64)
        self.prev_q_counts = np.zeros(
            (PREV_Q_CONTEXTS, QUALITY_ALPHABET_SIZE), dtype=np.int64
        )
        self.cycle_prev_q_counts = np.zeros(
            (0, PREV_Q_CONTEXTS, QUALITY_ALPHABET_SIZE), dtype=np.int64
        )
        self.completed_batches = 0
        self.observed_symbols = 0

    def _validate_contexts(
        self, previous_quality_ids: Sequence[int], cycles: Sequence[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        previous = np.asarray(previous_quality_ids)
        cycle_array = np.asarray(cycles)
        if previous.ndim != 1 or cycle_array.shape != previous.shape:
            raise OnlinePriorError(
                "previous_quality_ids and cycles must be equal 1-D arrays"
            )
        if previous.dtype.kind not in "iu" or cycle_array.dtype.kind not in "iu":
            raise OnlinePriorError(
                "previous_quality_ids and cycles must contain integers"
            )
        previous = previous.astype(np.int64, copy=False)
        cycle_array = cycle_array.astype(np.int64, copy=False)
        if np.any(cycle_array < 0):
            raise OnlinePriorError("cycles must be nonnegative")
        if np.any(previous < 0) or np.any(previous > BOS_QUALITY_ID):
            raise OnlinePriorError("previous_quality_ids must be Q0..Q41 or BOS")
        if np.any((cycle_array == 0) != (previous == BOS_QUALITY_ID)):
            raise OnlinePriorError(
                "cycle 0 must use BOS and later cycles must use Q0..Q41"
            )
        return previous, cycle_array

    def probabilities(
        self, previous_quality_ids: Sequence[int], cycles: Sequence[int]
    ) -> np.ndarray:
        """Return float64 hierarchical prior probabilities for active positions."""

        previous, cycle_array = self._validate_contexts(previous_quality_ids, cycles)
        row_count = previous.size
        if row_count == 0:
            return np.empty((0, QUALITY_ALPHABET_SIZE), dtype=np.float64)

        uniform = 1.0 / QUALITY_ALPHABET_SIZE
        global_strength = self.config.global_backoff_strength
        global_probability = (
            self.global_counts.astype(np.float64) + global_strength * uniform
        ) / (float(self.global_counts.sum(dtype=np.int64)) + global_strength)

        previous_counts = self.prev_q_counts[previous].astype(np.float64)
        previous_totals = previous_counts.sum(axis=1, keepdims=True, dtype=np.float64)
        previous_strength = self.config.prev_q_backoff_strength
        previous_probability = (
            previous_counts + previous_strength * global_probability[None, :]
        ) / (previous_totals + previous_strength)

        cycle_bins = cycle_array // self.config.cycle_bin_width
        cycle_counts = np.zeros_like(previous_counts)
        available = cycle_bins < self.cycle_prev_q_counts.shape[0]
        if np.any(available):
            cycle_counts[available] = self.cycle_prev_q_counts[
                cycle_bins[available], previous[available]
            ]
        cycle_totals = cycle_counts.sum(axis=1, keepdims=True, dtype=np.float64)
        cycle_strength = self.config.cycle_backoff_strength
        result = (
            cycle_counts + cycle_strength * previous_probability
        ) / (cycle_totals + cycle_strength)
        if not np.isfinite(result).all() or np.any(result <= 0.0):
            raise OnlinePriorError("online prior produced invalid probabilities")
        return result

    def fuse_logits(
        self,
        neural_logits: Sequence[Sequence[float]],
        previous_quality_ids: Sequence[int],
        cycles: Sequence[int],
    ) -> np.ndarray:
        """Return prior-ratio-adjusted neural scores in deterministic float64."""

        logits = np.asarray(neural_logits, dtype=np.float64)
        if logits.ndim != 2 or logits.shape[1] != QUALITY_ALPHABET_SIZE:
            raise OnlinePriorError(
                f"neural_logits must have shape [N, {QUALITY_ALPHABET_SIZE}]"
            )
        if not np.isfinite(logits).all():
            raise OnlinePriorError("neural_logits must not contain NaN or Inf")
        prior = self.probabilities(previous_quality_ids, cycles)
        if prior.shape[0] != logits.shape[0]:
            raise OnlinePriorError(
                "neural logits and prior contexts have different rows"
            )
        weight = self.config.prior_weight
        return logits + weight * np.log(prior * QUALITY_ALPHABET_SIZE)

    def update_batch(
        self,
        qualities: Sequence[Sequence[int]],
        active_mask: Sequence[Sequence[bool]],
    ) -> None:
        """Observe one completed batch in cycle-major active-position order."""

        quality_array = _integer_matrix(qualities, name="qualities")
        mask = np.asarray(active_mask)
        if mask.shape != quality_array.shape or mask.dtype.kind != "b":
            raise OnlinePriorError(
                "active_mask must be a boolean array matching qualities"
            )
        if quality_array.shape[1] > 1 and np.any(mask[:, 1:] & ~mask[:, :-1]):
            raise OnlinePriorError("active_mask rows must be contiguous prefixes")
        active_values = quality_array[mask]
        if active_values.size and (
            np.any(active_values < 0) or np.any(active_values >= QUALITY_ALPHABET_SIZE)
        ):
            raise OnlinePriorError("active qualities must be in Q0..Q41")
        if np.any(~mask) and np.any(quality_array[~mask] != QUALITY_PAD_ID):
            raise OnlinePriorError(
                f"inactive qualities must use pad id {QUALITY_PAD_ID}"
            )

        cycles, rows = np.nonzero(mask.T)
        symbols = quality_array[rows, cycles]
        previous = np.full(symbols.shape, BOS_QUALITY_ID, dtype=np.int64)
        later = cycles > 0
        previous[later] = quality_array[rows[later], cycles[later] - 1]
        cycle_bins = cycles // self.config.cycle_bin_width

        if cycle_bins.size:
            required_bins = int(cycle_bins.max()) + 1
            if required_bins > self.cycle_prev_q_counts.shape[0]:
                extension = np.zeros(
                    (
                        required_bins - self.cycle_prev_q_counts.shape[0],
                        PREV_Q_CONTEXTS,
                        QUALITY_ALPHABET_SIZE,
                    ),
                    dtype=np.int64,
                )
                self.cycle_prev_q_counts = np.concatenate(
                    (self.cycle_prev_q_counts, extension), axis=0
                )
            np.add.at(self.global_counts, symbols, 1)
            np.add.at(self.prev_q_counts, (previous, symbols), 1)
            np.add.at(
                self.cycle_prev_q_counts, (cycle_bins, previous, symbols), 1
            )
        self.completed_batches += 1
        self.observed_symbols += int(symbols.size)


__all__ = [
    "BOS_QUALITY_ID",
    "DEFAULT_ONLINE_PRIOR_CONFIG",
    "ONLINE_PRIOR_BACKOFF_ORDER",
    "ONLINE_PRIOR_PROFILE",
    "ONLINE_PRIOR_VERSION",
    "OnlinePriorConfig",
    "OnlinePriorError",
    "OnlinePriorState",
]
