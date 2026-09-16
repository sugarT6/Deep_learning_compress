"""FQZ-style cumulative downward-quality context, frozen per completed batch."""

from dataclasses import dataclass, asdict

import numpy as np

from .adaptive_prior import AdaptivePriorConfig, AdaptivePriorState
from .online_prior import OnlinePriorError


RUNNING_DELTA_PROFILE = "causal_adaptive_mixture_v3"


@dataclass(frozen=True)
class RunningDeltaPriorConfig(AdaptivePriorConfig):
    delta_initial: int = 5
    delta_bin_width: int = 8
    delta_max_bin: int = 7

    def __post_init__(self):
        super().__post_init__()
        if self.context_mode != "position_enriched":
            raise OnlinePriorError("running delta requires position_enriched")
        for name, minimum, maximum in (("delta_initial", 0, 65535),
                ("delta_bin_width", 1, 65535), ("delta_max_bin", 1, 31)):
            value = getattr(self, name)
            if type(value) is not int or not minimum <= value <= maximum:
                raise OnlinePriorError(f"invalid running delta {name}")
        if not 0 < self.weight_floor < 1 / 4 or min(1 - self.alpha, self.alpha / 3) < self.weight_floor:
            raise OnlinePriorError("four initial weights must respect weight floor")

    def to_profile_metadata(self):
        metadata = super().to_profile_metadata()
        metadata.update(name=RUNNING_DELTA_PROFILE, version=3,
            fusion="four_expert_arithmetic_mixture_temperature",
            experts=["neural", "cycle_order2", "cycle_prev_run", "running_delta"],
            initial_weights="1-alpha,alpha/3,alpha/3,alpha/3",
            delta_rule="initial_plus_sum_positive_previous_quality_drops_strict_prefix_floor_div_clamp",
            delta_context="cycle_bin_prev2_prev_delta_bin_backoff_to_cycle_order2_context_strength",
            config=asdict(self))
        return metadata


def running_delta_bins(qualities, rows, cycles, config):
    """D(c)=initial+sum(max(q[j-1]-q[j],0), j=1..c-1); never include q[c]."""
    rows, cycles = np.asarray(rows), np.asarray(cycles)
    if not cycles.size:
        return np.zeros(0, dtype=np.int64)
    read_ids, inverse = np.unique(rows, return_inverse=True)
    # Integer prefix scan: later entries never enter an earlier position's result.
    maximum = int(cycles.max())
    prefix = np.full((read_ids.size, maximum + 1), config.delta_initial, dtype=np.int64)
    if maximum > 1:
        history = np.asarray(qualities)[read_ids, :maximum].astype(np.int64)
        drops = np.maximum(history[:, :-1] - history[:, 1:], 0)
        prefix[:, 2:] += np.cumsum(drops, axis=1, dtype=np.int64)
    return np.minimum(prefix[inverse, cycles] // config.delta_bin_width, config.delta_max_bin)


class RunningDeltaPriorState(AdaptivePriorState):
    def __init__(self, config):
        super().__init__(config)
        self.delta_counts = np.zeros((0, 43, 43, config.delta_max_bin + 1, 42), dtype=np.int64)

    def _delta_contexts(self, qualities, rows, cycles):
        # No second run-length scan: this expert needs only two quality lags.
        q = np.asarray(qualities)
        previous = np.full(cycles.size, 42, dtype=np.int64)
        previous2 = previous.copy()
        later, later2 = cycles > 0, cycles > 1
        previous[later] = q[rows[later], cycles[later] - 1]
        previous2[later2] = q[rows[later2], cycles[later2] - 2]
        return (cycles // self.config.cycle_bin_width, previous2, previous,
            running_delta_bins(qualities, rows, cycles, self.config))

    def _prediction_experts(self, qualities, rows, cycles):
        order2, runs = super()._prediction_experts(qualities, rows, cycles)
        bins, previous2, previous, delta = self._delta_contexts(qualities, rows, cycles)
        keys = ((bins * 43 + previous2) * 43 + previous) * (self.config.delta_max_bin + 1) + delta
        _, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
        counts = np.zeros((first.size, 42), dtype=np.float64)
        available = bins[first] < len(self.delta_counts)
        ix = first[available]
        counts[available] = self.delta_counts[bins[ix], previous2[ix], previous[ix], delta[ix]]
        strength = self.config.context_strength
        probabilities = (counts + strength * order2[first]) / (counts.sum(axis=1, keepdims=True) + strength)
        return order2, runs, probabilities[inverse]

    def update_batch(self, qualities, active_mask):
        # Parent validates full feedback/mask and commits all previous tables/weights.
        super().update_batch(qualities, active_mask)
        cycles, rows = np.nonzero(np.asarray(active_mask).T)
        if not cycles.size:
            return
        bins, previous2, previous, delta = self._delta_contexts(qualities, rows, cycles)
        required = int(bins.max()) + 1
        if required > len(self.delta_counts):
            self.delta_counts = np.concatenate((self.delta_counts,
                np.zeros((required - len(self.delta_counts), 43, 43, self.config.delta_max_bin + 1, 42), dtype=np.int64)))
        np.add.at(self.delta_counts, (bins, previous2, previous, delta, np.asarray(qualities)[rows, cycles]), 1)
