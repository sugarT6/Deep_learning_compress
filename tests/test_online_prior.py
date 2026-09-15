import unittest

import numpy as np

from codec.online_prior import (
    BOS_QUALITY_ID,
    ONLINE_PRIOR_BACKOFF_ORDER,
    ONLINE_PRIOR_PROFILE,
    OnlinePriorConfig,
    OnlinePriorError,
    OnlinePriorState,
)


class OnlinePriorTest(unittest.TestCase):
    def setUp(self):
        self.config = OnlinePriorConfig(
            cycle_bin_width=2,
            global_backoff_strength=42.0,
            prev_q_backoff_strength=3.0,
            cycle_backoff_strength=5.0,
            prior_weight=0.25,
        )

    def test_manual_global_prev_q_cycle_bin_counts_and_bos(self):
        state = OnlinePriorState(self.config)
        qualities = np.asarray(
            [[0, 1, 41, 42], [41, 41, 42, 42]], dtype=np.int64
        )
        active = np.asarray(
            [[True, True, True, False], [True, True, False, False]]
        )
        state.update_batch(qualities, active)

        expected_global = np.zeros(42, dtype=np.int64)
        expected_global[[0, 1, 41]] = [1, 1, 3]
        np.testing.assert_array_equal(state.global_counts, expected_global)
        self.assertEqual(state.prev_q_counts[BOS_QUALITY_ID, 0], 1)
        self.assertEqual(state.prev_q_counts[BOS_QUALITY_ID, 41], 1)
        self.assertEqual(state.prev_q_counts[0, 1], 1)
        self.assertEqual(state.prev_q_counts[1, 41], 1)
        self.assertEqual(state.prev_q_counts[41, 41], 1)
        self.assertEqual(state.cycle_prev_q_counts[0, BOS_QUALITY_ID, 0], 1)
        self.assertEqual(state.cycle_prev_q_counts[0, BOS_QUALITY_ID, 41], 1)
        self.assertEqual(state.cycle_prev_q_counts[0, 0, 1], 1)
        self.assertEqual(state.cycle_prev_q_counts[0, 41, 41], 1)
        self.assertEqual(state.cycle_prev_q_counts[1, 1, 41], 1)
        self.assertEqual(int(state.cycle_prev_q_counts.sum()), 5)
        self.assertEqual(state.completed_batches, 1)
        self.assertEqual(state.observed_symbols, 5)

    def test_hierarchical_backoff_and_smoothing_match_formula(self):
        state = OnlinePriorState(self.config)
        qualities = np.asarray([[0, 1, 41]], dtype=np.int64)
        active = np.ones_like(qualities, dtype=bool)
        state.update_batch(qualities, active)

        global_probability = (state.global_counts.astype(np.float64) + 1.0) / 45.0
        previous_counts = state.prev_q_counts[0].astype(np.float64)
        previous_probability = (
            previous_counts + 3.0 * global_probability
        ) / (previous_counts.sum() + 3.0)

        # Cycle 4 is in an unseen bin, so its cycle row backs off exactly to
        # the prev-Q distribution.
        actual_unseen = state.probabilities([0], [4])[0]
        np.testing.assert_allclose(actual_unseen, previous_probability, rtol=0, atol=0)

        cycle_counts = state.cycle_prev_q_counts[0, 0].astype(np.float64)
        expected_seen = (
            cycle_counts + 5.0 * previous_probability
        ) / (cycle_counts.sum() + 5.0)
        actual_seen = state.probabilities([0], [1])[0]
        np.testing.assert_allclose(actual_seen, expected_seen, rtol=0, atol=0)
        self.assertTrue(np.all(actual_seen > 0.0))
        self.assertAlmostEqual(float(actual_seen.sum()), 1.0, places=15)

    def test_current_batch_is_invisible_until_completed_update(self):
        state = OnlinePriorState(self.config)
        before = state.probabilities([BOS_QUALITY_ID, 0], [0, 1])
        repeated = state.probabilities([BOS_QUALITY_ID, 0], [0, 1])
        np.testing.assert_array_equal(before, repeated)

        qualities = np.asarray([[0, 41]], dtype=np.int64)
        active = np.asarray([[True, True]])
        state.update_batch(qualities, active)
        after = state.probabilities([BOS_QUALITY_ID, 0], [0, 1])
        self.assertFalse(np.array_equal(before, after))

    def test_uniform_initial_prior_is_neutral_to_neural_logits(self):
        state = OnlinePriorState(self.config)
        logits = np.linspace(-2.0, 2.0, 84, dtype=np.float64).reshape(2, 42)
        fused = state.fuse_logits(logits, [BOS_QUALITY_ID, 0], [0, 1])
        np.testing.assert_array_equal(fused, logits)

    def test_q0_q41_variable_lengths_and_padding(self):
        state = OnlinePriorState(self.config)
        qualities = np.asarray(
            [[0, 41, 42], [41, 42, 42], [0, 0, 0]], dtype=np.int64
        )
        active = np.asarray(
            [[True, True, False], [True, False, False], [True, True, True]]
        )
        state.update_batch(qualities, active)
        self.assertEqual(state.observed_symbols, 6)
        self.assertEqual(state.global_counts[0], 4)
        self.assertEqual(state.global_counts[41], 2)
        self.assertEqual(int(state.global_counts.sum()), 6)

    def test_encoder_decoder_states_remain_identical_per_batch(self):
        encoder = OnlinePriorState(self.config)
        decoder = OnlinePriorState(self.config)
        batches = (
            (
                np.asarray([[0, 1], [41, 42]], dtype=np.int64),
                np.asarray([[True, True], [True, False]]),
            ),
            (
                np.asarray([[41, 41, 0]], dtype=np.int64),
                np.asarray([[True, True, True]]),
            ),
        )
        for qualities, active in batches:
            contexts = ([BOS_QUALITY_ID, 41], [0, 2])
            np.testing.assert_array_equal(
                encoder.probabilities(*contexts), decoder.probabilities(*contexts)
            )
            encoder.update_batch(qualities, active)
            decoder.update_batch(qualities.copy(), active.copy())
            np.testing.assert_array_equal(encoder.global_counts, decoder.global_counts)
            np.testing.assert_array_equal(encoder.prev_q_counts, decoder.prev_q_counts)
            np.testing.assert_array_equal(
                encoder.cycle_prev_q_counts, decoder.cycle_prev_q_counts
            )

    def test_profile_round_trip_and_invalid_configuration(self):
        metadata = self.config.to_profile_metadata()
        self.assertEqual(metadata["name"], ONLINE_PRIOR_PROFILE)
        self.assertEqual(tuple(metadata["backoff_order"]), ONLINE_PRIOR_BACKOFF_ORDER)
        self.assertEqual(
            OnlinePriorConfig.from_profile_metadata(metadata), self.config
        )

        for kwargs in (
            {"cycle_bin_width": 0},
            {"global_backoff_strength": 0.0},
            {"prev_q_backoff_strength": float("nan")},
            {"cycle_backoff_strength": -1.0},
            {"prior_weight": 0.0},
            {"prior_weight": 1.0},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(OnlinePriorError):
                    OnlinePriorConfig(**kwargs)

    def test_invalid_context_and_mask_are_rejected(self):
        state = OnlinePriorState(self.config)
        with self.assertRaises(OnlinePriorError):
            state.probabilities([0], [0])
        with self.assertRaises(OnlinePriorError):
            state.probabilities([BOS_QUALITY_ID], [1])
        with self.assertRaises(OnlinePriorError):
            state.update_batch([[42]], [[True]])
        with self.assertRaises(OnlinePriorError):
            state.update_batch([[42, 0]], [[False, True]])


if __name__ == "__main__":
    unittest.main()
