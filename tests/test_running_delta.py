import copy
import unittest

import numpy as np

from codec.running_delta import RunningDeltaPriorConfig, RunningDeltaPriorState, running_delta_bins
from codec.mixture_prior import parse_probability_profile, make_prior_state
from codec.online_prior import OnlinePriorError
from codec.probability_quantization import logits_to_cdfs


def commit(state, q, mask):
    cycles, rows = np.nonzero(mask.T)
    state.fuse_positions(np.zeros((cycles.size, 42)), q, rows, cycles, capture=True)
    state.observe_symbols(q[rows, cycles])
    state.update_batch(q, mask)


class RunningDeltaTest(unittest.TestCase):
    def test_manual_drop_prefix_bins_and_saturation(self):
        config = RunningDeltaPriorConfig()
        q = np.array([[30, 30, 20, 25, 0, 41, 0]])
        # D before each quality: 5,5,5,15,15,40,40. Rises add nothing.
        result = running_delta_bins(q, np.zeros(7, dtype=int), np.arange(7), config)
        np.testing.assert_array_equal(result, [0, 0, 0, 1, 1, 5, 5])
        changed = q.copy(); changed[0, 3:] = 41
        np.testing.assert_array_equal(running_delta_bins(changed, np.zeros(4, dtype=int), np.arange(4), config), result[:4])
        long = np.tile([41, 0], (1, 20))
        self.assertEqual(running_delta_bins(long, np.array([0]), np.array([39]), config)[0], 7)
        self.assertEqual(running_delta_bins(q, np.array([], dtype=int), np.array([], dtype=int), config).size, 0)

    def test_bos_position_counts_and_padding(self):
        state = RunningDeltaPriorState(RunningDeltaPriorConfig(cycle_bin_width=2))
        q = np.array([[41, 0, 41, 0], [0, 42, 42, 42]])
        commit(state, q, q != 42)
        self.assertEqual(int(state.delta_counts.sum()), 5)
        self.assertEqual(state.delta_counts[0, 42, 42, 0, 41], 1)
        self.assertEqual(state.delta_counts[0, 42, 42, 0, 0], 1)
        self.assertEqual(state.delta_counts[0, 42, 41, 0, 0], 1)
        self.assertEqual(state.delta_counts[1, 41, 0, 5, 41], 1)
        self.assertEqual(state.delta_counts[1, 0, 41, 5, 0], 1)
        reset = RunningDeltaPriorState(state.config)
        self.assertEqual(reset.delta_counts.size, 0)
        np.testing.assert_allclose(reset.weights, [0.5, 1/6, 1/6, 1/6])

    def test_smoothing_missing_context_and_batch_freeze(self):
        state = RunningDeltaPriorState(RunningDeltaPriorConfig(context_strength=2, cycle_bin_width=2))
        q = np.array([[41, 0, 41, 0]])
        commit(state, q, q != 42)
        rows, cycles = np.array([0]), np.array([2])
        order2, _, delta = state._prediction_experts(q, rows, cycles)
        counts = np.zeros(42); counts[41] = 1
        np.testing.assert_array_equal(delta[0], (counts + 2 * order2[0]) / 3)
        before = logits_to_cdfs(state.fuse_positions(np.zeros((1, 42)), q, rows, cycles))
        altered = q.copy(); altered[0, 2:] = 0
        np.testing.assert_array_equal(before, logits_to_cdfs(state.fuse_positions(np.zeros((1, 42)), altered, rows, cycles)))
        self.assertEqual(int(state.delta_counts.sum()), 4)
        missing = np.array([[41, 0, 41, 0, 41, 0, 41]])
        parent, _, empty = state._prediction_experts(missing, rows, np.array([6]))
        np.testing.assert_array_equal(empty, parent)
        self.assertTrue((delta > 0).all())
        commit(state, altered, altered != 42)
        self.assertFalse(np.array_equal(before, logits_to_cdfs(state.fuse_positions(np.zeros((1, 42)), q, rows, cycles))))

    def test_full_step_cdfs_integer_feedback_and_all_tables(self):
        rng = np.random.default_rng(901)
        config = RunningDeltaPriorConfig()
        encoder, decoder = RunningDeltaPriorState(config), RunningDeltaPriorState(config)
        for _ in range(3):
            q = rng.integers(0, 42, (65, 19))
            mask = np.arange(19)[None, :] < rng.integers(1, 20, (65, 1)); q[~mask] = 42
            logits = rng.normal(size=(65, 19, 42)).astype(np.float32)
            cycles, rows = np.nonzero(mask.T); frozen = encoder.weights
            full = logits_to_cdfs(encoder.fuse_positions(logits[rows, cycles], q, rows, cycles, capture=True))
            encoder.observe_symbols(q[rows, cycles])
            decoded = np.full_like(q, 42); step = []
            for cycle in range(19):
                active = np.flatnonzero(mask[:, cycle]); positions = np.full(active.size, cycle)
                step.append(logits_to_cdfs(decoder.fuse_positions(logits[active, cycle], decoded, active, positions, capture=True)))
                decoded[active, cycle] = q[active, cycle]
                decoder.observe_symbols(q[active, cycle])
                np.testing.assert_array_equal(decoder.weights, frozen)
            np.testing.assert_array_equal(full, np.concatenate(step))
            np.testing.assert_array_equal(encoder._units, decoder._units)
            encoder.update_batch(q, mask); decoder.update_batch(decoded, mask)
            np.testing.assert_array_equal(encoder.weights, decoder.weights)
            for name in ("delta_counts", "order2_counts", "cycle_order2_counts", "run_counts"):
                np.testing.assert_array_equal(getattr(encoder, name), getattr(decoder, name))
            np.testing.assert_array_equal(encoder.base.cycle_prev_q_counts, decoder.base.cycle_prev_q_counts)

    def test_delta_separates_same_order2_and_enters_final_mixture(self):
        state = RunningDeltaPriorState(RunningDeltaPriorConfig(context_strength=2))
        q = np.array([[41, 0, 30, 30, 41], [30, 30, 30, 30, 0]])
        commit(state, q, q != 42)
        rows, cycles = np.array([0, 1]), np.array([4, 4])
        order2, runs, delta = state._prediction_experts(q, rows, cycles)
        np.testing.assert_array_equal(order2[0], order2[1])
        self.assertGreater(delta[0, 41], delta[1, 41])
        self.assertGreater(delta[1, 0], delta[0, 0])
        weights = state.weights
        expected = weights[0] / 42 + weights[1] * order2 + weights[2] * runs + weights[3] * delta
        actual = state.fuse_positions(np.zeros((2, 42)), q, rows, cycles)
        np.testing.assert_array_equal(actual, np.log(expected))
        omitted = (weights[0] / 42 + weights[1] * order2 + weights[2] * runs) / (1 - weights[3])
        self.assertFalse(np.array_equal(logits_to_cdfs(actual), logits_to_cdfs(np.log(omitted))))

    def test_strict_metadata_and_legacy_rejection(self):
        from codec.adaptive_prior import AdaptivePriorConfig
        config = RunningDeltaPriorConfig(); profile = config.to_profile_metadata()
        self.assertEqual(parse_probability_profile(profile), config)
        self.assertIsInstance(make_prior_state(config), RunningDeltaPriorState)
        with self.assertRaises(OnlinePriorError):
            AdaptivePriorConfig.from_profile_metadata(profile)
        for key in profile:
            broken = copy.deepcopy(profile); del broken[key]
            with self.assertRaises(OnlinePriorError):
                parse_probability_profile(broken)
        for key, value in (("name", "causal_adaptive_mixture_v2"), ("version", 2), ("delta_rule", "absolute"), ("delta_context", "wrong")):
            broken = copy.deepcopy(profile); broken[key] = value
            with self.assertRaises(OnlinePriorError):
                parse_probability_profile(broken)
        for key, value in (("delta_initial", True), ("delta_bin_width", 0), ("delta_max_bin", 32),
                ("context_mode", "enriched"), ("weight_floor", 0.25)):
            broken = copy.deepcopy(profile); broken["config"][key] = value
            with self.assertRaises(OnlinePriorError):
                parse_probability_profile(broken)

    def test_four_expert_floor_projection(self):
        state = RunningDeltaPriorState(RunningDeltaPriorConfig(adaptation_rate=1))
        for _ in range(3):
            state._pending = ((np.full((1, 42), 1e-30),) * 3 + (np.ones((1, 42)),), 1)
            state.observe_symbols(np.array([0])); state.update_batch(np.array([[0]]), np.array([[True]]))
        np.testing.assert_allclose(state.weights, [0.01, 0.01, 0.01, 0.97])
