import copy
import contextlib
import io
import hashlib
import unittest

import numpy as np

from codec.adaptive_prior import (AdaptivePriorConfig, AdaptivePriorState,
    RESPONSIBILITY_TOTAL, responsibility_units)
from codec.mixture_prior import MixturePriorConfig, MixturePriorState, parse_probability_profile, make_prior_state
from codec.online_prior import OnlinePriorError
from codec.probability_quantization import logits_to_cdfs


class AdaptivePriorTest(unittest.TestCase):
    def test_encoder_rejects_retired_expert_options(self):
        from codec.encode import build_parser
        parser = build_parser()
        self.assertFalse(hasattr(parser.parse_args(["a.fq", "b.fqdc", "c.pt"]), "adaptive_weights"))
        for flags in (["--adaptive-weights"], ["--probability-profile", "x.json"],
                      ["--probability-profile=x.json"], ["--prior-weight", "0.25"],
                      ["--prior-cycle-bin-width=8"]):
            with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()) as stderr:
                with self.assertRaises(SystemExit) as error:
                    parser.parse_args(["a.fq", "b.fqdc", "c.pt"] + flags)
                self.assertEqual(error.exception.code, 2)
                self.assertIn("has been retired", stderr.getvalue())

    def test_manual_responsibility_and_completed_batch_update(self):
        config = AdaptivePriorConfig(adaptation_rate=0.5)
        state = AdaptivePriorState(config)
        old = state.weights
        experts = []
        for p in (0.8, 0.1, 0.1):
            row = np.full((1, 42), (1 - p) / 41)
            row[0, 0] = p
            experts.append(row)
        state._pending = (tuple(experts), 1)
        units = responsibility_units(np.array([[0.8, 0.1, 0.1]]), old)
        self.assertEqual(int(units.sum()), RESPONSIBILITY_TOTAL)
        np.testing.assert_allclose(units[0] / RESPONSIBILITY_TOTAL,
            [8 / 9, 1 / 18, 1 / 18], atol=2 / RESPONSIBILITY_TOTAL, rtol=0)
        state.observe_symbols(np.array([0]))
        np.testing.assert_array_equal(state.weights, old)
        self.assertEqual(state.observed_symbols, 0)
        proposed = 0.5 * old + 0.5 * units[0] / RESPONSIBILITY_TOTAL
        free = np.maximum(proposed - config.weight_floor, 0)
        expected = config.weight_floor + (1 - 3 * config.weight_floor) * free / free.sum()
        state.update_batch(np.array([[0]]), np.array([[True]]))
        np.testing.assert_allclose(state.weights, expected, rtol=1e-15)
        self.assertGreater(state.weights[0], old[0])
        self.assertEqual(state.weight_updates, 1)

    def test_integer_feedback_partition_and_ties(self):
        rng = np.random.default_rng(6)
        selected = rng.uniform(0.0001, 1, (1000, 3))
        weights = np.array([0.5, 0.25, 0.25])
        full = responsibility_units(selected, weights)
        chunks = [responsibility_units(part, weights) for part in np.array_split(selected, 23)]
        np.testing.assert_array_equal(full.sum(axis=0), sum(c.sum(axis=0) for c in chunks))
        np.testing.assert_array_equal(full.sum(axis=1), np.full(1000, RESPONSIBILITY_TOTAL))
        tie = responsibility_units(np.ones((1, 3)), np.ones(3) / 3)[0]
        self.assertEqual(tie[0], tie[1] + 1)
        self.assertEqual(tie[1], tie[2])

    def test_cold_start_feedback_contract_and_reset(self):
        config = AdaptivePriorConfig(temperature=0.85)
        state = AdaptivePriorState(config)
        q = np.array([[0, 41]])
        rows, cycles = np.array([0, 0]), np.array([0, 1])
        logits = np.arange(84).reshape(2, 42).astype(np.float64)
        initial = state.weights
        scores = state.fuse_positions(logits, q, rows, cycles, capture=True)
        np.testing.assert_array_equal(scores, logits / 0.85)
        with self.assertRaises(OnlinePriorError):
            state.update_batch(q, np.ones_like(q, dtype=bool))
        with self.assertRaises(OnlinePriorError):
            state.fuse_positions(logits, q, rows, cycles, capture=True)
        with self.assertRaises(OnlinePriorError):
            state.observe_symbols(np.array([0, 42]))
        state.observe_symbols(q[rows, cycles])
        with self.assertRaises(OnlinePriorError):
            state.observe_symbols(q[rows, cycles])
        state.update_batch(q, np.ones_like(q, dtype=bool))
        np.testing.assert_array_equal(state.weights, initial)
        self.assertEqual(state.weight_updates, 0)
        reset = AdaptivePriorState(config)
        np.testing.assert_array_equal(reset.weights, initial)
        self.assertEqual(reset.observed_symbols, 0)

    def test_full_step_weights_cdfs_counts_and_verification_purity(self):
        rng = np.random.default_rng(55)
        config = AdaptivePriorConfig()
        encoder, decoder = AdaptivePriorState(config), AdaptivePriorState(config)
        reference = MixturePriorState(MixturePriorConfig(context_mode=config.context_mode))
        for _ in range(5):
            q = rng.integers(0, 42, (65, 19))
            mask = np.arange(19)[None, :] < rng.integers(1, 20, (65, 1))
            q[~mask] = 42
            cycles, rows = np.nonzero(mask.T)
            symbols = q[rows, cycles]
            logits = rng.normal(size=(65, 19, 42)).astype(np.float32)
            frozen = encoder.weights
            cdfs = logits_to_cdfs(encoder.fuse_positions(logits[rows, cycles], q, rows, cycles, capture=True))
            # Debug full/step predictions must never capture/consume feedback.
            pure = logits_to_cdfs(encoder.fuse_positions(logits[rows, cycles], q, rows, cycles))
            np.testing.assert_array_equal(cdfs, pure)
            encoder.observe_symbols(symbols)
            np.testing.assert_array_equal(encoder.weights, frozen)
            decoded = np.full_like(q, 42)
            per_cycle = []
            for cycle in range(19):
                active = np.flatnonzero(mask[:, cycle])
                positions = np.full(active.size, cycle)
                per_cycle.append(logits_to_cdfs(decoder.fuse_positions(logits[active, cycle], decoded,
                    active, positions, capture=True)))
                decoded[active, cycle] = q[active, cycle]
                decoder.observe_symbols(q[active, cycle])
                np.testing.assert_array_equal(decoder.weights, frozen)
            np.testing.assert_array_equal(cdfs, np.concatenate(per_cycle))
            np.testing.assert_array_equal(encoder._units, decoder._units)
            encoder.update_batch(q, mask); decoder.update_batch(decoded, mask)
            reference.update_batch(q, mask)
            np.testing.assert_array_equal(encoder.weights, decoder.weights)
            for name in ("order2_counts", "run_counts", "cycle_order2_counts"):
                np.testing.assert_array_equal(getattr(encoder, name), getattr(reference, name))
                np.testing.assert_array_equal(getattr(encoder, name), getattr(decoder, name))
            np.testing.assert_array_equal(encoder.base.cycle_prev_q_counts, reference.base.cycle_prev_q_counts)

    def test_floor_and_stability(self):
        state = AdaptivePriorState(AdaptivePriorConfig(adaptation_rate=1.0))
        for _ in range(100):
            state._pending = ((np.ones((1, 42)), np.full((1, 42), 1e-30), np.full((1, 42), 1e-30)), 1)
            state.observe_symbols(np.array([0]))
            state.update_batch(np.array([[0]]), np.array([[True]]))
            self.assertTrue(np.isfinite(state.weights).all())
            self.assertTrue((state.weights >= 0.01).all())
            self.assertAlmostEqual(float(state.weights.sum()), 1.0)
        np.testing.assert_allclose(state.weights, [0.98, 0.01, 0.01])

    def test_strict_profile_validation_and_old_rejection(self):
        config = AdaptivePriorConfig()
        profile = config.to_profile_metadata()
        self.assertEqual(parse_probability_profile(profile), config)
        self.assertIsInstance(make_prior_state(config), AdaptivePriorState)
        with self.assertRaises(OnlinePriorError):
            MixturePriorConfig.from_profile_metadata(profile)
        for key in profile:
            broken = copy.deepcopy(profile); del broken[key]
            with self.assertRaises(OnlinePriorError):
                parse_probability_profile(broken)
        self.assertEqual(profile["name"], "causal_adaptive_mixture_v2")
        for key, value in (("version", 1), ("version", True), ("responsibility_total", 65536),
                           ("name", "causal_adaptive_mixture_v1"), ("order2_rule", "wrong"),
                           ("cold_start", "learn"), ("experts", ["run", "neural", "order2"])):
            broken = copy.deepcopy(profile); broken[key] = value
            with self.assertRaises(OnlinePriorError):
                parse_probability_profile(broken)
        for key, value in (("adaptation_rate", 0), ("weight_floor", True), ("context_mode", "enriched")):
            broken = copy.deepcopy(profile); broken["config"][key] = value
            with self.assertRaises(OnlinePriorError):
                parse_probability_profile(broken)

    def test_legacy_cdfs_match_pre_position_commit(self):
        # Frozen digests generated from e41165a, not from the implementation under test.
        cases = ((MixturePriorConfig(), MixturePriorState,
            "28596821b552b8649677a2be9917d1527ed8d041a1bcbce95ff7f9d979a4c58f"),
            (AdaptivePriorConfig(context_mode="enriched"), AdaptivePriorState,
            "f1c96d2cc843cbd04bb3e50735d1469a9695ad75412f0c8c8f9f1acb57e82116"),
            # Position v2 baseline generated from 880b595.
            (AdaptivePriorConfig(), AdaptivePriorState,
            "fbf0d5d2ff4538d8bcdc25e82cb07d67dccd5dcb54e6f1bf709e87361d5cc854"))
        for config, state_class, expected in cases:
            state = state_class(config); digest = hashlib.sha256()
            rng = np.random.default_rng(1729)
            for _ in range(3):
                q = rng.integers(0, 42, (5, 19))
                mask = np.arange(19)[None, :] < np.array([1, 8, 9, 17, 19])[:, None]
                q[~mask] = 42
                cycles, rows = np.nonzero(mask.T)
                logits = rng.normal(size=(cycles.size, 42)).astype(np.float32)
                if isinstance(state, AdaptivePriorState):
                    scores = state.fuse_positions(logits, q, rows, cycles, capture=True)
                    state.observe_symbols(q[rows, cycles])
                else:
                    scores = state.fuse_positions(logits, q, rows, cycles)
                digest.update(logits_to_cdfs(scores).astype("<i8").tobytes())
                state.update_batch(q, mask)
            self.assertEqual(digest.hexdigest(), expected)

    def test_legacy_adaptive_profile_preserves_nonposition_expert(self):
        config = AdaptivePriorConfig(context_mode="enriched")
        profile = config.to_profile_metadata()
        self.assertEqual(profile["name"], "causal_adaptive_mixture_v1")
        self.assertEqual(profile["version"], 1)
        self.assertNotIn("order2_rule", profile)
        self.assertEqual(parse_probability_profile(profile), config)
        state = AdaptivePriorState(parse_probability_profile(profile))
        q = np.array([[0, 0, 41]])
        state.fuse_positions(np.zeros((3, 42)), q, np.array([0, 0, 0]), np.arange(3), capture=True)
        state.observe_symbols(q[0]); state.update_batch(q, q != 42)
        self.assertEqual(state.cycle_order2_counts.size, 0)
        reference = MixturePriorState(MixturePriorConfig())
        reference.update_batch(q, q != 42)
        args = (q, np.array([0]), np.array([2]))
        for actual, expected in zip(state.expert_probabilities(*args), reference.expert_probabilities(*args)):
            np.testing.assert_array_equal(actual, expected)
        for key, value in (("adaptation_rate", 0), ("adaptation_rate", float("nan")),
                           ("weight_floor", 0.34), ("weight_floor", True), ("context_mode", "hierarchical")):
            broken = copy.deepcopy(profile); broken["config"][key] = value
            with self.assertRaises(OnlinePriorError):
                parse_probability_profile(broken)
