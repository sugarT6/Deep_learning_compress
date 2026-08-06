import unittest
from pathlib import Path

import numpy as np

from sequence_residual_transformer_model import (
    DEFAULT_EXACT_Q_LAGS,
    DEFAULT_EXACT_R_LAGS,
    DEFAULT_HISTORY_RUN_EMBED_DIM,
    DEFAULT_PRIOR_FEATURE_MODE,
    DEFAULT_QMER_KS,
    DEFAULT_RMER_KS,
    build_exact_lag_tokens_for_read,
    build_causal_run_length_tokens,
    build_mer_tokens_for_read,
    run_length_to_bucket,
)
from train_sequence_residual_transformer import build_parser


def reference_mer_tokens(
    buckets: np.ndarray,
    ks: tuple[int, ...],
    stride: int,
    base: int,
    bos_bucket: int,
    vocab_size: int,
) -> np.ndarray:
    """Original scalar implementation retained as an equivalence oracle."""

    tokens = np.zeros((len(buckets), len(ks)), dtype=np.int64)
    for pos in range(len(buckets)):
        for mer_idx, k in enumerate(ks):
            code = 0
            for distance in range(k, 0, -1):
                hist_pos = pos - distance * stride
                bucket = int(buckets[hist_pos]) if hist_pos >= 0 else bos_bucket
                code = (code * base + bucket) % vocab_size
            tokens[pos, mer_idx] = code
    return tokens


class MerTokenTest(unittest.TestCase):
    def test_default_windows(self) -> None:
        self.assertEqual(DEFAULT_QMER_KS, (2, 3, 4))
        self.assertEqual(DEFAULT_RMER_KS, (2, 3, 4))
        self.assertEqual(DEFAULT_EXACT_Q_LAGS, ())
        self.assertEqual(DEFAULT_EXACT_R_LAGS, ())

    def test_training_defaults(self) -> None:
        args = build_parser().parse_args([])
        self.assertEqual(args.batch_reads, 64)
        self.assertEqual(args.eval_batch_reads, 64)
        self.assertEqual(args.qmer_ks, (2, 3, 4))
        self.assertEqual(args.rmer_ks, (2, 3, 4))
        self.assertEqual(args.history_run_embed_dim, DEFAULT_HISTORY_RUN_EMBED_DIM)
        self.assertEqual(args.history_run_embed_dim, 0)
        self.assertEqual(args.prior_feature_mode, DEFAULT_PRIOR_FEATURE_MODE)
        self.assertEqual(args.base_conv_kernels, (3, 5, 7))
        self.assertEqual(args.num_layers, 4)
        self.assertEqual(
            args.output_dir,
            Path(
                "runs/transformer_residual_4layer_qhatonly_qrmer234_"
                "baseconv357_b64_e15"
            ),
        )

    def test_vectorized_tokens_match_scalar_reference(self) -> None:
        rng = np.random.default_rng(20260727)
        configurations = (
            ((4, 6), 1, 8, 7, 4096, 8),
            ((4, 6), 1, 12, 11, 4096, 12),
            ((1, 3, 7), 2, 12, 11, 97, 12),
            ((2, 5), 3, 8, 7, 31, 8),
        )

        for length in (0, 1, 2, 5, 17, 101, 388):
            for ks, stride, base, bos_bucket, vocab_size, bucket_count in configurations:
                with self.subTest(length=length, ks=ks, stride=stride, base=base):
                    buckets = rng.integers(0, bucket_count, size=length, dtype=np.int64)
                    expected = reference_mer_tokens(
                        buckets,
                        ks,
                        stride,
                        base,
                        bos_bucket,
                        vocab_size,
                    )
                    actual = build_mer_tokens_for_read(
                        buckets,
                        ks,
                        stride,
                        base,
                        bos_bucket,
                        vocab_size,
                    )
                    np.testing.assert_array_equal(actual, expected)

    def test_empty_window_list(self) -> None:
        actual = build_mer_tokens_for_read(
            np.asarray([1, 2, 3], dtype=np.int64),
            (),
            stride=1,
            base=8,
            bos_bucket=7,
            vocab_size=4096,
        )
        self.assertEqual(actual.shape, (3, 0))

    def test_exact_lags_are_causal_and_use_bos_for_missing_history(self) -> None:
        actual = build_exact_lag_tokens_for_read(
            np.asarray([10, 20, 30, 40, 50], dtype=np.int64),
            (2, 3, 4),
            bos_token=95,
        )
        expected = np.asarray(
            [
                [95, 95, 95],
                [95, 95, 95],
                [10, 95, 95],
                [20, 10, 95],
                [30, 20, 10],
            ],
            dtype=np.int64,
        )
        np.testing.assert_array_equal(actual, expected)

    def test_exact_lags_reject_lag_one_and_duplicates(self) -> None:
        values = np.asarray([10, 20, 30], dtype=np.int64)
        with self.assertRaisesRegex(ValueError, "at least 2"):
            build_exact_lag_tokens_for_read(values, (1,), bos_token=95)
        with self.assertRaisesRegex(ValueError, "unique"):
            build_exact_lag_tokens_for_read(values, (2, 2), bos_token=95)

    def test_run_length_bucket_boundaries(self) -> None:
        lengths = np.asarray([0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 31, 32, 99])
        expected = np.asarray([0, 1, 2, 3, 4, 5, 5, 6, 6, 7, 7, 8, 8])
        np.testing.assert_array_equal(run_length_to_bucket(lengths), expected)

    def test_run_lengths_use_only_positions_before_current_target(self) -> None:
        zero_run, same_q_run = build_causal_run_length_tokens(
            qualities=np.asarray([10, 10, 20, 20, 20, 30]),
            residuals=np.asarray([0, 0, -1, 0, 0, 0]),
        )
        np.testing.assert_array_equal(zero_run, np.asarray([0, 1, 2, 0, 1, 2]))
        np.testing.assert_array_equal(same_q_run, np.asarray([0, 1, 2, 1, 2, 3]))

    def test_rejects_invalid_parameters(self) -> None:
        buckets = np.asarray([1, 2, 3], dtype=np.int64)
        with self.assertRaises(ValueError):
            build_mer_tokens_for_read(buckets, (0,), 1, 8, 7, 4096)
        with self.assertRaises(ValueError):
            build_mer_tokens_for_read(buckets, (4,), 0, 8, 7, 4096)
        with self.assertRaises(ValueError):
            build_mer_tokens_for_read(buckets, (4,), 1, 8, 7, 0)
        with self.assertRaises(ValueError):
            build_mer_tokens_for_read(buckets.reshape(1, -1), (4,), 1, 8, 7, 4096)


if __name__ == "__main__":
    unittest.main()
