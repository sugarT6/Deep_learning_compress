"""Opt-in causal probability mixture; all state updates are completed-batch only."""

from dataclasses import asdict, dataclass
import math

import numpy as np

from .online_prior import OnlinePriorConfig, OnlinePriorError, OnlinePriorState


MIXTURE_PROFILE = "causal_quality_mixture_v1"


@dataclass(frozen=True)
class MixturePriorConfig:
    alpha: float = 0.5
    temperature: float = 1.0
    context_mode: str = "enriched"
    context_strength: float = 42.0
    cycle_bin_width: int = 8
    global_backoff_strength: float = 42.0
    prev_q_backoff_strength: float = 42.0
    cycle_backoff_strength: float = 42.0

    def __post_init__(self):
        for name in ("alpha", "temperature", "context_strength"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise OnlinePriorError(f"invalid mixture {name}")
        if not 0 < self.alpha < 1 or not 0.25 <= self.temperature <= 4 or self.context_strength <= 0:
            raise OnlinePriorError("invalid mixture alpha, temperature or context strength")
        if self.context_mode not in ("hierarchical", "enriched"):
            raise OnlinePriorError("unsupported mixture context mode")
        base = self.base_config()
        for name in ("cycle_bin_width", "global_backoff_strength", "prev_q_backoff_strength", "cycle_backoff_strength"):
            object.__setattr__(self, name, getattr(base, name))

    def base_config(self):
        return OnlinePriorConfig(cycle_bin_width=self.cycle_bin_width,
            global_backoff_strength=self.global_backoff_strength,
            prev_q_backoff_strength=self.prev_q_backoff_strength,
            cycle_backoff_strength=self.cycle_backoff_strength)

    def to_profile_metadata(self):
        return {"name": MIXTURE_PROFILE, "version": 1,
            "update_granularity": "completed_batch", "float_contract": "finite_cpu_float64",
            "count_dtype": "int64", "fusion": "arithmetic_mixture_temperature",
            "cold_start": "temperature_neural_only", "run_length_cap": 16,
            "enriched_rule": "mean_order2_and_cycle_prev_run_backoff_to_hierarchy",
            "config": asdict(self)}

    @classmethod
    def from_profile_metadata(cls, values):
        try:
            if not isinstance(values, dict) or not isinstance(values.get("config"), dict):
                raise ValueError("profile must be an object")
            if set(values["config"]) != set(cls.__dataclass_fields__):
                raise ValueError("mixture config fields mismatch")
            result = cls(**values["config"])
            expected = result.to_profile_metadata()
            if set(values) != set(expected) or type(values.get("version")) is not int:
                raise ValueError("mixture profile fields/version mismatch")
            if type(values.get("run_length_cap")) is not int or values != expected:
                raise ValueError("unsupported mixture protocol")
            return result
        except (TypeError, ValueError) as exc:
            raise OnlinePriorError(f"invalid mixture profile: {exc}") from exc


def history_contexts(qualities, rows, cycles):
    """Read only cycles strictly before the predicted position; cap runs at 16."""
    qualities = np.asarray(qualities)
    previous = np.full(cycles.size, 42, dtype=np.int64)
    previous2 = previous.copy()
    later = cycles > 0
    previous[later] = qualities[rows[later], cycles[later] - 1]
    later2 = cycles > 1
    previous2[later2] = qualities[rows[later2], cycles[later2] - 2]
    run = np.zeros(cycles.size, dtype=np.int64)
    continuing = later.copy()
    for distance in range(1, 17):
        eligible = continuing & (cycles >= distance)
        selected = np.flatnonzero(eligible)
        continuing[:] = False
        continuing[selected] = qualities[rows[selected], cycles[selected] - distance] == previous[selected]
        run += continuing
    # 0/1 -> 0; 2/3 -> 1; 4..7 -> 2; 8..15 -> 3; 16 -> 4.
    run_bin = np.searchsorted([2, 4, 8, 16], run, side="right")
    return previous, previous2, run_bin


class MixturePriorState:
    def __init__(self, config):
        self.config = config
        self.base = OnlinePriorState(config.base_config())
        self.order2_counts = np.zeros((43, 43, 42), dtype=np.int64)
        self.run_counts = np.zeros((0, 43, 5, 42), dtype=np.int64)

    @property
    def observed_symbols(self):
        return self.base.observed_symbols

    def probabilities(self, qualities, rows, cycles):
        previous, previous2, run_bin = history_contexts(qualities, rows, cycles)
        bins = cycles // self.config.cycle_bin_width
        # Group all contexts needed by the hierarchy and its two richer children.
        keys = ((bins * 43 + previous) * 43 + previous2) * 5 + run_bin
        _, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
        parent = self.base.probabilities(previous[first], cycles[first])
        if self.config.context_mode == "enriched":
            order2 = self.order2_counts[previous2[first], previous[first]].astype(np.float64)
            runs = np.zeros_like(order2)
            available = bins[first] < len(self.run_counts)
            ix = first[available]
            runs[available] = self.run_counts[bins[ix], previous[ix], run_bin[ix]]
            strength = self.config.context_strength
            order2 = (order2 + strength * parent) / (order2.sum(axis=1, keepdims=True) + strength)
            runs = (runs + strength * parent) / (runs.sum(axis=1, keepdims=True) + strength)
            parent = (order2 + runs) * 0.5
        return parent[inverse]

    def fuse_positions(self, logits, qualities, rows, cycles):
        scores = np.asarray(logits, dtype=np.float64) / self.config.temperature
        if scores.shape != (cycles.size, 42) or not np.isfinite(scores).all():
            raise OnlinePriorError("invalid mixture neural logits")
        if not self.observed_symbols:
            return scores
        neural = np.exp(scores - scores.max(axis=1, keepdims=True))
        neural /= neural.sum(axis=1, keepdims=True)
        online = self.probabilities(qualities, rows, cycles)
        mixture = (1 - self.config.alpha) * neural + self.config.alpha * online
        return np.log(mixture)

    def update_batch(self, qualities, active_mask):
        # Reference validator runs before either enriched table can change.
        self.base.update_batch(qualities, active_mask)
        if self.config.context_mode != "enriched":
            return
        cycles, rows = np.nonzero(np.asarray(active_mask).T)
        previous, previous2, run_bin = history_contexts(qualities, rows, cycles)
        if not cycles.size:
            return
        bins = cycles // self.config.cycle_bin_width
        required = int(bins.max()) + 1
        if required > len(self.run_counts):
            self.run_counts = np.concatenate((self.run_counts,
                np.zeros((required - len(self.run_counts), 43, 5, 42), dtype=np.int64)))
        symbols = np.asarray(qualities)[rows, cycles]
        np.add.at(self.order2_counts, (previous2, previous, symbols), 1)
        np.add.at(self.run_counts, (bins, previous, run_bin, symbols), 1)


def parse_probability_profile(values):
    if isinstance(values, dict) and values.get("name") == MIXTURE_PROFILE:
        return MixturePriorConfig.from_profile_metadata(values)
    return OnlinePriorConfig.from_profile_metadata(values)


def make_prior_state(config):
    if config is None:
        return None
    return MixturePriorState(config) if isinstance(config, MixturePriorConfig) else OnlinePriorState(config)


def fuse_profile_positions(state, logits, qualities, rows, cycles):
    if state is None:
        return logits
    if isinstance(state, MixturePriorState):
        return state.fuse_positions(logits, qualities, rows, cycles)
    previous = np.full(cycles.size, 42, dtype=np.int64)
    later = cycles > 0
    previous[later] = qualities[rows[later], cycles[later] - 1]
    return state.fuse_logits(logits, previous, cycles)
