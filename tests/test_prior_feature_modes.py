import unittest

import numpy as np

from predict_sequence_residual_transformer import mer_params_from_config
from sequence_residual_transformer_model import (
    ALPHABET_SIZE,
    DEFAULT_QUALITY_ALPHABET_SIZE,
    QHAT_ONLY,
    QHAT_ONLY_CONTINUOUS_FEATURE_DIM,
    QUALITY_TARGET,
    RESIDUAL_CLASSES,
    RESIDUAL_MIN,
    RESIDUAL_TARGET,
    build_sequence_batch,
    prediction_output_dim,
)


class PriorFeatureModeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.freqs = np.ones((3, 95), dtype=np.uint32)
        self.freqs[0, 30] = 30
        self.freqs[1, 31] = 40
        self.freqs[2, 29] = 50
        self.observed = np.asarray([31, 30, 29], dtype=np.uint8)
        self.offsets = np.asarray([0, 2, 3], dtype=np.int64)

    def build(self, mode: str, target: str = QUALITY_TARGET):
        return build_sequence_batch(
            freqs=self.freqs,
            observed=self.observed,
            local_offsets=self.offsets,
            qmer_ks=(),
            rmer_ks=(),
            prior_feature_mode=mode,
            prediction_target=target,
        )

    def test_qhat_only_omits_full_probability_features(self) -> None:
        batch = self.build(QHAT_ONLY)
        self.assertEqual(batch.continuous.shape, (2, 2, QHAT_ONLY_CONTINUOUS_FEATURE_DIM))
        np.testing.assert_array_equal(batch.q_hat[0], np.asarray([30, 31]))
        np.testing.assert_allclose(batch.continuous[0, :, 0], np.asarray([0.0, 1.0]))
        np.testing.assert_allclose(batch.continuous[0, :, 1], np.asarray([0.0002, 0.0002]))
        np.testing.assert_array_equal(batch.targets[0], self.observed[:2])

    def test_distribution_shape_cannot_enter_qhat_only_features(self) -> None:
        before = self.build(QHAT_ONLY)
        self.freqs[:, 0] += 1  # Change distribution shape, preserving every argmax.
        after = self.build(QHAT_ONLY)
        np.testing.assert_array_equal(before.continuous, after.continuous)
        np.testing.assert_array_equal(before.q_hat, after.q_hat)
        np.testing.assert_array_equal(before.prev_r, after.prev_r)
        self.assertNotEqual(before.baseline_bits, after.baseline_bits)

    def test_retired_features_are_rejected_without_affecting_qhat_checkpoints(self) -> None:
        for mode in ("full_prior", "compact_prior"):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.build(mode)
        for fields in ({"continuous_dim": 194}, {"exact_q_lags": [2]},
                       {"exact_r_lags": [3]}, {"platform_embed_dim": 8}):
            with self.subTest(fields=fields), self.assertRaisesRegex(ValueError, "retired"):
                mer_params_from_config(fields)


    def test_prediction_restores_new_42_class_checkpoint_alphabet(self) -> None:
        params = mer_params_from_config(
            {
                "continuous_dim": QHAT_ONLY_CONTINUOUS_FEATURE_DIM,
                "prediction_target": QUALITY_TARGET,
                "quality_alphabet_size": DEFAULT_QUALITY_ALPHABET_SIZE,
                "output_dim": DEFAULT_QUALITY_ALPHABET_SIZE,
            }
        )
        self.assertEqual(
            params["quality_alphabet_size"],
            DEFAULT_QUALITY_ALPHABET_SIZE,
        )

    def test_prediction_infers_legacy_95_class_quality_checkpoint(self) -> None:
        params = mer_params_from_config(
            {
                "continuous_dim": QHAT_ONLY_CONTINUOUS_FEATURE_DIM,
                "prediction_target": QUALITY_TARGET,
                "output_dim": ALPHABET_SIZE,
            }
        )
        self.assertEqual(params["quality_alphabet_size"], ALPHABET_SIZE)

    def test_quality_target_keeps_residual_history(self) -> None:
        quality_batch = self.build(QHAT_ONLY, QUALITY_TARGET)
        residual_batch = self.build(QHAT_ONLY, RESIDUAL_TARGET)

        np.testing.assert_array_equal(quality_batch.targets[0], np.asarray([31, 30]))
        np.testing.assert_array_equal(
            residual_batch.targets[0],
            np.asarray([31 - 30 - RESIDUAL_MIN, 30 - 31 - RESIDUAL_MIN]),
        )
        np.testing.assert_array_equal(quality_batch.prev_r, residual_batch.prev_r)
        np.testing.assert_array_equal(quality_batch.rmer_tokens, residual_batch.rmer_tokens)
        self.assertEqual(
            prediction_output_dim(QUALITY_TARGET),
            DEFAULT_QUALITY_ALPHABET_SIZE,
        )
        self.assertEqual(
            prediction_output_dim(QUALITY_TARGET, ALPHABET_SIZE),
            ALPHABET_SIZE,
        )
        self.assertEqual(prediction_output_dim(RESIDUAL_TARGET), RESIDUAL_CLASSES)

    def test_support_matched_h5_baseline_keeps_full_qhat_but_renormalizes_42_classes(self) -> None:
        batch = self.build(QHAT_ONLY)
        np.testing.assert_array_equal(batch.q_hat[0], np.asarray([30, 31]))
        self.assertLess(batch.support_matched_baseline_bits, batch.baseline_bits)

        first_true_frequency = float(self.freqs[0, self.observed[0]])
        expected_full = first_true_frequency / float(self.freqs[0].sum())
        expected_supported = first_true_frequency / float(self.freqs[0, :42].sum())
        self.assertAlmostEqual(float(batch.h5_true_prob[0, 0]), expected_full)
        self.assertAlmostEqual(
            float(batch.h5_support_matched_true_prob[0, 0]),
            expected_supported,
        )

    def test_direct_quality_rejects_symbols_above_configured_max(self) -> None:
        observed = self.observed.copy()
        observed[0] = 42
        with self.assertRaisesRegex(ValueError, r"model range \[0, 41\]"):
            build_sequence_batch(
                freqs=self.freqs,
                observed=observed,
                local_offsets=self.offsets,
                qmer_ks=(),
                rmer_ks=(),
                prior_feature_mode=QHAT_ONLY,
                prediction_target=QUALITY_TARGET,
            )


if __name__ == "__main__":
    unittest.main()
