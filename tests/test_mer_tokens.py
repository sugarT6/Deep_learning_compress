import unittest
from pathlib import Path

import numpy as np

from sequence_residual_transformer_model import (
    DEFAULT_QMER_KS,
    DEFAULT_RMER_KS,
    build_mer_tokens_for_read,
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

    def test_training_defaults(self) -> None:
        args = build_parser().parse_args([])
        self.assertEqual(args.batch_reads, 128)
        self.assertEqual(args.qmer_ks, (2, 3, 4))
        self.assertEqual(args.rmer_ks, (2, 3, 4))
        self.assertEqual(args.num_layers, 4)
        self.assertEqual(
            args.output_dir,
            Path("runs/transformer_residual_4layer_qrmer_234_b128"),
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
