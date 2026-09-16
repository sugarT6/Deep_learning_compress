import copy
from dataclasses import replace
import unittest

import numpy as np

from codec.mixture_prior import (MixturePriorConfig, MixturePriorState,
    history_contexts, parse_probability_profile)
from codec.online_prior import OnlinePriorConfig, OnlinePriorError, OnlinePriorState
from codec.evaluate_fusion import candidate_configs, candidate_scores, validate_manifest
from codec.datasets import TRAIN_DATASETS, UNSEEN_DATASETS
from codec.probability_quantization import logits_to_cdfs


class MixturePriorTest(unittest.TestCase):
    def test_history_counts_bos_runs_tail_mask(self):
        q = np.array([[41, 41, 41, 0], [0, 42, 42, 42]])
        mask = q != 42
        cycles, rows = np.nonzero(mask.T)
        previous, previous2, runs = history_contexts(q, rows, cycles)
        np.testing.assert_array_equal(previous, [42, 42, 41, 41, 41])
        np.testing.assert_array_equal(previous2, [42, 42, 42, 41, 41])
        np.testing.assert_array_equal(runs, [0, 0, 0, 1, 1])
        state = MixturePriorState(MixturePriorConfig(cycle_bin_width=2))
        state.update_batch(q, mask)
        self.assertEqual(state.observed_symbols, 5)
        self.assertEqual(state.order2_counts[42, 42, 41], 1)
        self.assertEqual(state.order2_counts[41, 41, 0], 1)
        self.assertEqual(state.run_counts[1, 41, 1, 0], 1)
        self.assertEqual(state.run_counts.sum(), 5)
        long_q = np.full((1, 25), 41)
        _, _, run = history_contexts(long_q, np.array([0, 0]), np.array([16, 24]))
        np.testing.assert_array_equal(run, [4, 4])

    def test_smoothing_and_mixture_manual(self):
        state = MixturePriorState(MixturePriorConfig(alpha=0.5, context_strength=2))
        q = np.array([[0, 0]])
        mask = np.ones_like(q, dtype=bool)
        state.update_batch(q, mask)
        rows, cycles = np.array([0]), np.array([1])
        parent = state.base.probabilities([0], [1])[0]
        counts = np.zeros(42); counts[0] = 1
        expected = (counts + 2 * parent) / 3
        actual = state.probabilities(q, rows, cycles)[0]
        np.testing.assert_array_equal(actual, expected)
        mixed = np.exp(state.fuse_positions(np.zeros((1, 42)), q, rows, cycles))[0]
        np.testing.assert_allclose(mixed, 0.5 / 42 + 0.5 * expected, rtol=1e-15)
        self.assertTrue((actual > 0).all())

    def test_full_step_cdfs_causality_and_batch_state(self):
        rng = np.random.default_rng(444)
        for mode in ("hierarchical", "enriched", "position_enriched"):
            config = MixturePriorConfig(context_mode=mode, temperature=0.85)
            encoder, decoder = MixturePriorState(config), MixturePriorState(config)
            for _ in range(3):
                q = rng.integers(0, 42, (65, 19))
                mask = np.arange(19)[None, :] < rng.integers(1, 20, (65, 1))
                q[~mask] = 42
                logits = rng.normal(size=(65, 19, 42)).astype(np.float32)
                cycles, rows = np.nonzero(mask.T)
                all_cdfs = logits_to_cdfs(encoder.fuse_positions(logits[rows, cycles], q, rows, cycles))
                decoded = np.full_like(q, 42)
                step_cdfs = []
                for cycle in range(19):
                    active = np.flatnonzero(mask[:, cycle])
                    positions = np.full(active.size, cycle)
                    step_cdfs.append(logits_to_cdfs(decoder.fuse_positions(logits[active, cycle], decoded, active, positions)))
                    decoded[active, cycle] = q[active, cycle]
                np.testing.assert_array_equal(all_cdfs, np.concatenate(step_cdfs))
                before = encoder.base.global_counts.copy()
                encoder.fuse_positions(logits[rows, cycles], q, rows, cycles)
                np.testing.assert_array_equal(before, encoder.base.global_counts)
                encoder.update_batch(q, mask); decoder.update_batch(decoded, mask)
                for attr in ("order2_counts", "run_counts", "cycle_order2_counts"):
                    np.testing.assert_array_equal(getattr(encoder, attr), getattr(decoder, attr))
                np.testing.assert_array_equal(encoder.base.cycle_prev_q_counts, decoder.base.cycle_prev_q_counts)

    def test_position_order2_hand_counts_smoothing_and_unseen_bin(self):
        config = MixturePriorConfig(context_mode="position_enriched", cycle_bin_width=2,
            context_strength=2)
        state = MixturePriorState(config)
        q = np.array([[0, 0, 41, 0], [41, 42, 42, 42]])
        mask = q != 42
        state.update_batch(q, mask)
        self.assertEqual(int(state.cycle_order2_counts.sum()), 5)
        self.assertEqual(state.cycle_order2_counts[0, 42, 42, 0], 1)
        self.assertEqual(state.cycle_order2_counts[0, 42, 42, 41], 1)
        self.assertEqual(state.cycle_order2_counts[0, 42, 0, 0], 1)
        self.assertEqual(state.cycle_order2_counts[1, 0, 0, 41], 1)
        self.assertEqual(state.cycle_order2_counts[1, 0, 41, 0], 1)
        query = np.array([[0, 0, 41, 0, 0, 0, 0]])
        for cycle in (2, 6):
            parent = state.base.probabilities([0], [cycle])[0]
            counts = state.order2_counts[0, 0].astype(np.float64)
            fallback = (counts + 2 * parent) / (counts.sum() + 2)
            local = state.cycle_order2_counts[cycle // 2, 0, 0].astype(np.float64) if cycle == 2 else np.zeros(42)
            expected = (local + 2 * fallback) / (local.sum() + 2)
            actual, _ = state.expert_probabilities(query, np.array([0]), np.array([cycle]))
            np.testing.assert_array_equal(actual[0], expected)
            self.assertTrue((actual > 0).all())
            self.assertAlmostEqual(float(actual.sum()), 1)

    def test_position_order2_frozen_batch_and_position_separation(self):
        state = MixturePriorState(MixturePriorConfig(context_mode="position_enriched",
            cycle_bin_width=2, context_strength=2))
        # Same history (0,0), but Q41 at cycle 2 and Q0 at cycle 6.
        q = np.array([[0, 0, 41, 0, 0, 0, 0]])
        mask = np.ones_like(q, dtype=bool)
        rows, cycles = np.array([0, 0]), np.array([2, 6])
        before = state.expert_probabilities(q, rows, cycles)[0]
        changed = q.copy(); changed[0, 2] = 0
        np.testing.assert_array_equal(before, state.expert_probabilities(changed, rows, cycles)[0])
        self.assertEqual(state.cycle_order2_counts.size, 0)
        state.update_batch(q, mask)
        after = state.expert_probabilities(q, rows, cycles)[0]
        self.assertGreater(after[0, 41], after[1, 41])
        self.assertGreater(after[1, 0], after[0, 0])
        self.assertFalse(np.array_equal(before, after))

    def test_position_mixture_profile_strict_validation(self):
        config = MixturePriorConfig(context_mode="position_enriched")
        profile = config.to_profile_metadata()
        self.assertEqual(profile["name"], "causal_quality_mixture_v2")
        self.assertEqual(parse_probability_profile(profile), config)
        for key in profile:
            broken = copy.deepcopy(profile); del broken[key]
            with self.assertRaises(OnlinePriorError):
                parse_probability_profile(broken)
        for key, value in (("name", "causal_quality_mixture_v1"), ("version", 1), ("order2_rule", "wrong")):
            broken = copy.deepcopy(profile); broken[key] = value
            with self.assertRaises(OnlinePriorError):
                parse_probability_profile(broken)

    def test_evaluation_candidates_match_deployed_profiles(self):
        configs = candidate_configs()
        q = np.array([[0, 0, 41], [41, 0, 41]])
        mask = np.ones_like(q, dtype=bool)
        cycles, rows = np.nonzero(mask.T)
        logits = np.zeros((cycles.size, 42), dtype=np.float32)
        for warmed in (False, True):
            base, rich = OnlinePriorState(), MixturePriorState(MixturePriorConfig())
            if warmed:
                base.update_batch(q, mask); rich.update_batch(q, mask)
            scores = candidate_scores(logits, q, rows, cycles, base, rich, configs)
            for name, config in configs.items():
                if not isinstance(config, MixturePriorConfig):
                    continue
                state = MixturePriorState(config)
                if warmed:
                    state.update_batch(q, mask)
                np.testing.assert_array_equal(logits_to_cdfs(scores[name]),
                    logits_to_cdfs(state.fuse_positions(logits, q, rows, cycles)))

    def test_profile_strict_validation_and_old_decoder_rejection(self):
        original = MixturePriorConfig().to_profile_metadata()
        self.assertEqual(parse_probability_profile(original), MixturePriorConfig())
        for key in original:
            broken = copy.deepcopy(original); del broken[key]
            with self.assertRaises(OnlinePriorError):
                parse_probability_profile(broken)
        for key, value in (("version", True), ("version", 2), ("fusion", "wrong"), ("run_length_cap", 32)):
            broken = copy.deepcopy(original); broken[key] = value
            with self.assertRaises(OnlinePriorError):
                parse_probability_profile(broken)
        for key, value in (("alpha", 0), ("alpha", float("nan")), ("temperature", 0),
                           ("context_mode", "unknown"), ("context_strength", -1)):
            broken = copy.deepcopy(original); broken["config"][key] = value
            with self.assertRaises(OnlinePriorError):
                parse_probability_profile(broken)
        with self.assertRaises(OnlinePriorError):
            OnlinePriorConfig.from_profile_metadata(original)

    def test_unseen_and_overlapping_splits_rejected(self):
        entries = {d.accession: {"train_range": [0, 10], "validation_range": [10, 20], "read_count": 20} for d in TRAIN_DATASETS}
        payload = {"data_split": {"datasets": entries}}
        validate_manifest(payload, TRAIN_DATASETS)
        with self.assertRaises(ValueError):
            validate_manifest(payload, (*TRAIN_DATASETS[:-1], UNSEEN_DATASETS[0]))
        entries[TRAIN_DATASETS[0].accession]["validation_range"] = [9, 20]
        with self.assertRaises(ValueError):
            validate_manifest(payload, TRAIN_DATASETS)
