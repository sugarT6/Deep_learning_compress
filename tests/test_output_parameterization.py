import unittest

import torch

from sequence_residual_transformer_model import (
    CONTINUOUS_FEATURE_DIM,
    DIRECT_RESIDUAL_LOGITS,
    LOG_P0_PLUS_DELTA,
    RESIDUAL_CLASSES,
    ResidualTransformer,
)
from train_sequence_residual_transformer import build_parser


def make_small_model(output_parameterization: str) -> ResidualTransformer:
    return ResidualTransformer(
        continuous_dim=CONTINUOUS_FEATURE_DIM,
        qmer_ks=(),
        rmer_ks=(),
        d_model=16,
        num_heads=4,
        num_layers=1,
        feedforward_dim=32,
        context_length=8,
        dropout=0.0,
        output_parameterization=output_parameterization,
    )


class OutputParameterizationTest(unittest.TestCase):
    def test_parser_defaults_to_direct_logits_and_accepts_delta(self) -> None:
        parser = build_parser()
        self.assertEqual(
            parser.parse_args([]).output_parameterization,
            DIRECT_RESIDUAL_LOGITS,
        )
        self.assertEqual(
            parser.parse_args(
                ["--output-parameterization", LOG_P0_PLUS_DELTA]
            ).output_parameterization,
            LOG_P0_PLUS_DELTA,
        )

    def test_zero_initialized_delta_model_reproduces_log_p0(self) -> None:
        torch.manual_seed(7)
        model = make_small_model(LOG_P0_PLUS_DELTA)
        continuous = torch.randn(2, 3, CONTINUOUS_FEATURE_DIM)
        q_hat = torch.randint(0, 95, (2, 3))
        prev_q = torch.randint(0, 96, (2, 3))
        prev_r = torch.randint(0, 190, (2, 3))

        logits = model(
            continuous=continuous,
            q_hat=q_hat,
            prev_q=prev_q,
            prev_r=prev_r,
        )

        torch.testing.assert_close(logits, continuous[..., :RESIDUAL_CLASSES])
        torch.testing.assert_close(
            model.output_head.weight,
            torch.zeros_like(model.output_head.weight),
        )
        torch.testing.assert_close(
            model.output_head.bias,
            torch.zeros_like(model.output_head.bias),
        )

    def test_direct_model_keeps_normal_output_head_initialization(self) -> None:
        torch.manual_seed(7)
        model = make_small_model(DIRECT_RESIDUAL_LOGITS)
        self.assertGreater(float(model.output_head.weight.abs().sum()), 0.0)

    def test_unknown_output_parameterization_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "output_parameterization"):
            make_small_model("unknown")


if __name__ == "__main__":
    unittest.main()
