import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch

from sequence_residual_transformer_model import (
    DEFAULT_QUALITY_ALPHABET_SIZE,
    DIRECT_LOGITS,
    Q_BOS_TOKEN,
    R_BOS_TOKEN,
    ResidualTransformer,
    inspect_quality_distribution,
    quality_distribution_bits_for_batch,
)
from train_sequence_residual_transformer import build_parser


def make_model() -> ResidualTransformer:
    return ResidualTransformer(
        continuous_dim=2,
        q_hat_embed_dim=2,
        prev_q_embed_dim=2,
        prev_r_embed_dim=2,
        exact_q_lags=[],
        exact_r_lags=[],
        qmer_ks=[],
        rmer_ks=[],
        base_context_dim=0,
        platform_embed_dim=0,
        quality_distribution_prior=True,
        quality_distribution_embed_dim=4,
        quality_distribution_hidden_dim=8,
        quality_delta_limit=4.0,
        d_model=8,
        num_heads=2,
        num_layers=1,
        feedforward_dim=16,
        context_length=4,
        dropout=0.0,
        output_dim=DEFAULT_QUALITY_ALPHABET_SIZE,
        output_parameterization=DIRECT_LOGITS,
    )


def model_inputs() -> dict[str, torch.Tensor]:
    return {
        "continuous": torch.zeros((2, 3, 2), dtype=torch.float32),
        "q_hat": torch.zeros((2, 3), dtype=torch.long),
        "prev_q": torch.full((2, 3), Q_BOS_TOKEN, dtype=torch.long),
        "prev_r": torch.full((2, 3), R_BOS_TOKEN, dtype=torch.long),
        "lengths": torch.tensor([3, 2], dtype=torch.long),
    }


class QualityDistributionPriorTest(unittest.TestCase):
    def test_add_one_distribution_uses_42_body_quality_classes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "qualities.h5"
            with h5py.File(path, "w") as handle:
                handle.create_dataset(
                    "observed",
                    data=np.array([0, 0, 1], dtype=np.uint8),
                )

            info = inspect_quality_distribution(path, chunk_rows=2)

            self.assertEqual(info.total_symbols, 3)
            self.assertEqual(info.alphabet_size, DEFAULT_QUALITY_ALPHABET_SIZE)
            self.assertEqual(info.counts.shape, (DEFAULT_QUALITY_ALPHABET_SIZE,))
            self.assertEqual(info.counts.tolist()[:3], [2, 1, 0])
            denominator = 3.0 + DEFAULT_QUALITY_ALPHABET_SIZE
            self.assertAlmostEqual(float(info.probabilities[0]), 3.0 / denominator)
            self.assertAlmostEqual(float(info.probabilities[1]), 2.0 / denominator)
            self.assertAlmostEqual(float(info.probabilities[2]), 1.0 / denominator)
            self.assertAlmostEqual(float(info.probabilities.sum()), 1.0)

            batch = SimpleNamespace(
                targets=np.array([[0, 1, -100]], dtype=np.int64),
                valid_mask=np.array([[True, True, False]]),
            )
            expected_bits = -math.log2(3.0 / denominator) - math.log2(
                2.0 / denominator
            )
            self.assertAlmostEqual(
                quality_distribution_bits_for_batch(
                    batch,
                    info.log_probabilities,
                ),
                expected_bits,
                places=5,
            )

    def test_training_flag_uses_recommended_dimensions_and_c4(self) -> None:
        args = build_parser().parse_args(["--quality-distribution-prior"])
        self.assertTrue(args.quality_distribution_prior)
        self.assertEqual(args.quality_distribution_embed_dim, 32)
        self.assertEqual(args.quality_distribution_hidden_dim, 64)
        self.assertEqual(args.quality_delta_limit, 4.0)
        self.assertEqual(
            args.quality_alphabet_size,
            DEFAULT_QUALITY_ALPHABET_SIZE,
        )

    def test_distribution_rejects_quality_above_q41(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "qualities.h5"
            with h5py.File(path, "w") as handle:
                handle.create_dataset(
                    "observed",
                    data=np.array([41, 42], dtype=np.uint8),
                )

            with self.assertRaisesRegex(ValueError, r"model range \[0, 41\]"):
                inspect_quality_distribution(path)

    def test_zero_initialized_delta_reproduces_file_histogram(self) -> None:
        model = make_model().eval()
        probabilities = torch.arange(
            1,
            DEFAULT_QUALITY_ALPHABET_SIZE + 1,
            dtype=torch.float32,
        )
        probabilities /= probabilities.sum()
        log_probabilities = probabilities.log()

        logits = model(
            **model_inputs(),
            quality_distribution_log_probs=log_probabilities,
        )

        expected = log_probabilities.view(1, 1, -1).expand(2, 3, -1)
        self.assertTrue(torch.allclose(logits, expected, atol=1e-6, rtol=0.0))
        self.assertTrue(
            torch.allclose(
                torch.softmax(logits, dim=-1),
                probabilities.view(1, 1, -1).expand(2, 3, -1),
                atol=1e-6,
                rtol=0.0,
            )
        )

    def test_delta_logits_are_bounded_to_minus4_plus4(self) -> None:
        model = make_model().eval()
        with torch.no_grad():
            model.output_head.bias.copy_(
                torch.linspace(
                    -100.0,
                    100.0,
                    DEFAULT_QUALITY_ALPHABET_SIZE,
                )
            )
        log_probabilities = torch.full(
            (DEFAULT_QUALITY_ALPHABET_SIZE,),
            -math.log(DEFAULT_QUALITY_ALPHABET_SIZE),
            dtype=torch.float32,
        )

        logits = model(
            **model_inputs(),
            quality_distribution_log_probs=log_probabilities,
        )
        delta = logits - log_probabilities.view(1, 1, -1)

        self.assertGreaterEqual(float(delta.min()), -4.000001)
        self.assertLessEqual(float(delta.max()), 4.000001)
        self.assertAlmostEqual(float(delta[0, 0, 0]), -4.0, places=5)
        self.assertAlmostEqual(float(delta[0, 0, -1]), 4.0, places=5)

    def test_distribution_is_required_by_conditioned_model(self) -> None:
        model = make_model().eval()
        with self.assertRaisesRegex(ValueError, "quality_distribution_log_probs"):
            model(**model_inputs())


if __name__ == "__main__":
    unittest.main()
