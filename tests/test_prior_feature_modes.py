import unittest
from unittest import mock

import numpy as np

import sequence_residual_transformer_model as residual_model
from predict_sequence_residual_transformer import mer_params_from_config
from sequence_residual_transformer_model import (
    ALPHABET_SIZE,
    FULL_PRIOR,
    FULL_PRIOR_CONTINUOUS_FEATURE_DIM,
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

    def test_qhat_only_does_not_construct_log_probability_matrix(self) -> None:
        with mock.patch.object(
            residual_model,
            "_quality_probs",
            side_effect=AssertionError("full prior path was used"),
        ):
            self.build(QHAT_ONLY)

    def test_full_prior_and_qhat_only_share_targets_and_baseline(self) -> None:
        qhat_only = self.build(QHAT_ONLY)
        full_prior = self.build(FULL_PRIOR)

        self.assertEqual(
            full_prior.continuous.shape,
            (2, 2, FULL_PRIOR_CONTINUOUS_FEATURE_DIM),
        )
        np.testing.assert_array_equal(qhat_only.q_hat, full_prior.q_hat)
        np.testing.assert_array_equal(qhat_only.targets, full_prior.targets)
        np.testing.assert_allclose(qhat_only.h5_true_prob, full_prior.h5_true_prob)
        self.assertAlmostEqual(qhat_only.baseline_bits, full_prior.baseline_bits)

        for read_idx, length in enumerate(full_prior.lengths):
            for pos in range(int(length)):
                q_hat = int(full_prior.q_hat[read_idx, pos])
                observed = int(full_prior.targets[read_idx, pos])
                residual_class = observed - q_hat - RESIDUAL_MIN
                log_p0_true = float(
                    full_prior.continuous[read_idx, pos, residual_class]
                )
                self.assertAlmostEqual(
                    float(np.exp(log_p0_true)),
                    float(full_prior.h5_true_prob[read_idx, pos]),
                    places=6,
                )
        self.assertEqual(full_prior.continuous.shape[-1], RESIDUAL_CLASSES + 5)

    def test_old_checkpoint_without_mode_is_inferred_as_full_prior(self) -> None:
        old_config = {"continuous_dim": FULL_PRIOR_CONTINUOUS_FEATURE_DIM}
        params = mer_params_from_config(old_config)
        self.assertEqual(params["prior_feature_mode"], FULL_PRIOR)
        self.assertEqual(params["prediction_target"], RESIDUAL_TARGET)

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
        self.assertEqual(prediction_output_dim(QUALITY_TARGET), ALPHABET_SIZE)
        self.assertEqual(prediction_output_dim(RESIDUAL_TARGET), RESIDUAL_CLASSES)


if __name__ == "__main__":
    unittest.main()
