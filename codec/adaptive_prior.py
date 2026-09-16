"""Completed-batch online EM weights for versioned neural/order-2/run experts.

Prediction is pure unless capture=True. Captured probabilities are consumed
only after their symbols have been encoded/decoded. Integer responsibilities
make accumulation independent of full-batch versus per-cycle partitioning.
"""

from dataclasses import asdict, dataclass
import math

import numpy as np

from .mixture_prior import MixturePriorConfig, MixturePriorState
from .online_prior import OnlinePriorError


ADAPTIVE_PROFILE = "causal_adaptive_mixture_v1"
POSITION_ADAPTIVE_PROFILE = "causal_adaptive_mixture_v2"
RESPONSIBILITY_TOTAL = 1 << 24


@dataclass(frozen=True)
class AdaptivePriorConfig(MixturePriorConfig):
    context_mode: str = "position_enriched"
    adaptation_rate: float = 0.1
    weight_floor: float = 0.01

    def __post_init__(self):
        super().__post_init__()
        if self.context_mode not in ("enriched", "position_enriched"):
            raise OnlinePriorError("adaptive weights require enriched contexts")
        for name in ("adaptation_rate", "weight_floor"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise OnlinePriorError(f"invalid adaptive {name}")
        if not 0 < self.adaptation_rate <= 1 or not 0 < self.weight_floor < 1 / 3:
            raise OnlinePriorError("invalid adaptive rate or weight floor")
        if min(1 - self.alpha, self.alpha / 2) < self.weight_floor:
            raise OnlinePriorError("initial weights must respect weight_floor")

    def to_profile_metadata(self):
        positioned = self.context_mode == "position_enriched"
        metadata = {"name": POSITION_ADAPTIVE_PROFILE if positioned else ADAPTIVE_PROFILE,
            "version": 2 if positioned else 1,
            "update_granularity": "completed_batch", "float_contract": "finite_cpu_float64",
            "count_dtype": "int64", "fusion": "three_expert_arithmetic_mixture_temperature",
            "cold_start": "temperature_neural_only_skip_weight_learning",
            "run_length_cap": 16, "experts": ["neural", "cycle_order2" if positioned else "order2", "cycle_prev_run"],
            "initial_weights": "1-alpha,alpha/2,alpha/2",
            "weight_update": "damped_mean_responsibility_floor_projection_v1",
            "responsibility_total": RESPONSIBILITY_TOTAL,
            "responsibility_rounding": "floor_then_remainder_to_largest_responsibility_first_tie",
            "config": asdict(self)}
        if positioned:
            metadata["order2_rule"] = "cycle_bin_prev2_prev_backoff_to_order2_backoff_to_hierarchy_context_strength"
        return metadata

    @classmethod
    def from_profile_metadata(cls, values):
        try:
            if not isinstance(values, dict) or not isinstance(values.get("config"), dict):
                raise ValueError("profile must be an object")
            if set(values["config"]) != set(cls.__dataclass_fields__):
                raise ValueError("adaptive config fields mismatch")
            result = cls(**values["config"])
            expected = result.to_profile_metadata()
            for key in ("version", "run_length_cap", "responsibility_total"):
                if type(values.get(key)) is not int:
                    raise ValueError(f"invalid adaptive {key}")
            if set(values) != set(expected) or values != expected:
                raise ValueError("unsupported adaptive profile")
            return result
        except (TypeError, ValueError) as exc:
            raise OnlinePriorError(f"invalid adaptive profile: {exc}") from exc


def responsibility_units(selected_probabilities, weights):
    """Quantize each symbol's expert responsibilities before any reduction."""
    weighted = np.asarray(selected_probabilities, dtype=np.float64) * weights[None, :]
    denominator = weighted.sum(axis=1, keepdims=True, dtype=np.float64)
    if not np.isfinite(weighted).all() or np.any(weighted < 0) or np.any(denominator <= 0):
        raise OnlinePriorError("invalid adaptive expert probabilities")
    responsibilities = weighted / denominator
    units = np.floor(responsibilities * RESPONSIBILITY_TOTAL).astype(np.int64)
    remainder = RESPONSIBILITY_TOTAL - units.sum(axis=1, dtype=np.int64)
    if np.any(remainder < 0) or np.any(remainder > weighted.shape[1]):
        raise OnlinePriorError("invalid responsibility rounding")
    units[np.arange(units.shape[0]), np.argmax(responsibilities, axis=1)] += remainder
    return units


class AdaptivePriorState(MixturePriorState):
    def __init__(self, config):
        super().__init__(config)
        count = len(config.to_profile_metadata()["experts"])
        self._weights = np.array([1 - config.alpha] + [config.alpha / (count - 1)] * (count - 1), dtype=np.float64)
        self._pending = None
        self._units = np.zeros(count, dtype=np.int64)
        self._recorded_positions = 0
        self._learned_positions = 0
        self.weight_updates = 0

    @property
    def weights(self):
        return self._weights.copy()

    def _prediction_experts(self, qualities, rows, cycles):
        return self.expert_probabilities(qualities, rows, cycles)

    def fuse_positions(self, logits, qualities, rows, cycles, *, capture=False):
        if capture and self._pending is not None:
            raise OnlinePriorError("adaptive predictions must be consumed before recapture")
        scores = np.asarray(logits, dtype=np.float64) / self.config.temperature
        if scores.shape != (cycles.size, 42) or not np.isfinite(scores).all():
            raise OnlinePriorError("invalid adaptive neural logits")
        if not self.observed_symbols:
            if capture:
                self._pending = (None, cycles.size)
            return scores
        neural = np.exp(scores - scores.max(axis=1, keepdims=True))
        neural /= neural.sum(axis=1, keepdims=True)
        online = self._prediction_experts(qualities, rows, cycles)
        order2, runs = online[:2]
        mixture = self._weights[0] * neural + self._weights[1] * order2 + self._weights[2] * runs
        for index, expert in enumerate(online[2:], start=3):
            mixture = mixture + self._weights[index] * expert
        if capture:
            self._pending = ((neural, *online), cycles.size)
        return np.log(mixture)

    def observe_symbols(self, symbols):
        """Accumulate feedback; never change the weights or counts used by this batch."""
        if self._pending is None:
            raise OnlinePriorError("no captured adaptive predictions")
        experts, count = self._pending
        symbols = np.asarray(symbols)
        if symbols.shape != (count,) or symbols.dtype.kind not in "iu" or np.any(symbols < 0) or np.any(symbols > 41):
            raise OnlinePriorError("invalid adaptive observed symbols")
        if (self._recorded_positions + count) > np.iinfo(np.int64).max // RESPONSIBILITY_TOTAL:
            raise OnlinePriorError("adaptive batch responsibility accumulator would overflow")
        if experts is not None and count:
            positions = np.arange(count)
            selected = np.column_stack([p[positions, symbols] for p in experts])
            self._units += responsibility_units(selected, self._weights).sum(axis=0, dtype=np.int64)
            self._learned_positions += count
        self._recorded_positions += count
        self._pending = None

    def update_batch(self, qualities, active_mask):
        expected = int(np.count_nonzero(active_mask))
        if self._pending is not None or self._recorded_positions != expected:
            raise OnlinePriorError("adaptive batch requires feedback for every active symbol")
        # Validate and commit counts only at the original completed-batch boundary.
        super().update_batch(qualities, active_mask)
        if self._learned_positions:
            mean = self._units.astype(np.float64) / (self._learned_positions * RESPONSIBILITY_TOTAL)
            rate, floor = self.config.adaptation_rate, self.config.weight_floor
            proposed = (1 - rate) * self._weights + rate * mean
            free = np.maximum(proposed - floor, 0.0)
            self._weights = floor + (1 - len(self._weights) * floor) * (free / free.sum(dtype=np.float64))
            self.weight_updates += 1
        self._units[:] = 0
        self._recorded_positions = self._learned_positions = 0

    def diagnostics(self):
        return {"experts": self.config.to_profile_metadata()["experts"],
            "final_weights": self.weights.tolist(), "weight_updates": self.weight_updates,
            "adaptation_rate": self.config.adaptation_rate, "weight_floor": self.config.weight_floor}
