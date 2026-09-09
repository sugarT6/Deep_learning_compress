import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from codec.checkpoint import load_training_checkpoint, save_training_checkpoint
from codec.fastq_stream import QUALITY_PAD_ID, make_fastq_batch
from codec.model import (
    DirectQualityModelConfig,
    DirectQualityTransformer,
    build_quality_history_tokens,
    fastq_batch_to_tensors,
    feature_schema,
    masked_cross_entropy,
    theoretical_bits,
)


def tiny_config():
    return DirectQualityModelConfig(
        prev_q_embed_dim=4,
        qmer_embed_dim=3,
        base_embed_dim=4,
        base_conv_channels=4,
        base_context_dim=8,
        d_model=16,
        num_heads=4,
        num_layers=1,
        feedforward_dim=32,
        context_length=8,
        dropout=0.0,
    )


def variable_batch():
    return make_fastq_batch(
        [
            np.asarray([0, 1, 2, 3], dtype=np.uint8),
            np.asarray([4, 0], dtype=np.uint8),
        ],
        [
            np.asarray([0, 10, 20, 41], dtype=np.uint8),
            np.asarray([41, 0], dtype=np.uint8),
        ],
        [0, 1],
        source_name="synthetic.fastq",
    )


class DirectQualityModelTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)
        self.model = DirectQualityTransformer(tiny_config())
        self.batch = variable_batch()
        self.tensors = fastq_batch_to_tensors(self.batch, torch.device("cpu"))

    def test_forward_shape_and_feature_schema(self):
        logits = self.model.forward_full(**self.tensors)
        self.assertEqual(tuple(logits.shape), (2, 4, 42))
        self.assertEqual(self.model.output_head.out_features, 42)
        schema = feature_schema()
        self.assertEqual(schema["features"]["causal_qmer"]["ks"], [2, 3, 4])
        self.assertEqual(
            schema["excluded"],
            [
                "SeqArc",
                "q_hat",
                "prev_r",
                "R-mer",
                "quality_distribution_prior",
                "platform_embedding",
                "accession_id",
            ],
        )

    def test_prev_q_and_qmer_tokens_use_only_strict_history(self):
        qualities = torch.tensor([[0, 10, 20, 40]], dtype=torch.long)
        prev_q, qmers = build_quality_history_tokens(qualities)
        torch.testing.assert_close(
            prev_q, torch.tensor([[42, 0, 10, 20]]), rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(
            qmers[0], torch.tensor([[63, 56, 1, 10]]), rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(
            qmers[1], torch.tensor([[511, 504, 449, 10]]), rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(
            qmers[2],
            torch.tensor([[4095, 4088, 4033, 3594]]),
            rtol=0.0,
            atol=0.0,
        )

    def test_quality_history_is_strictly_causal(self):
        self.model.eval()
        changed = self.tensors["qualities"].clone()
        changed[0, 2:] = torch.tensor([35, 36])
        with torch.no_grad():
            original_logits = self.model.forward_full(**self.tensors)
            changed_logits = self.model.forward_full(
                self.tensors["bases"],
                changed,
                self.tensors["lengths"],
                self.tensors["active_mask"],
            )
        torch.testing.assert_close(
            original_logits[0, :3], changed_logits[0, :3], rtol=0.0, atol=0.0
        )
        self.assertFalse(torch.equal(original_logits[0, 3], changed_logits[0, 3]))

    def test_complete_future_base_can_affect_current_logits(self):
        self.model.eval()
        changed_bases = self.tensors["bases"].clone()
        changed_bases[0, 1] = 4
        with torch.no_grad():
            original = self.model.forward_full(**self.tensors)
            changed = self.model.forward_full(
                changed_bases,
                self.tensors["qualities"],
                self.tensors["lengths"],
                self.tensors["active_mask"],
            )
        self.assertFalse(torch.equal(original[0, 0], changed[0, 0]))

    def test_padding_is_zeroed_and_does_not_enter_loss_or_bits(self):
        logits = self.model.forward_full(**self.tensors)
        self.assertTrue(torch.equal(logits[1, 2:], torch.zeros_like(logits[1, 2:])))
        first_loss = masked_cross_entropy(
            logits, self.tensors["qualities"], self.tensors["active_mask"]
        )
        first_bits = theoretical_bits(
            logits, self.tensors["qualities"], self.tensors["active_mask"]
        )
        changed_logits = logits.clone()
        changed_logits[~self.tensors["active_mask"]] = 1_000.0
        second_loss = masked_cross_entropy(
            changed_logits,
            self.tensors["qualities"],
            self.tensors["active_mask"],
        )
        second_bits = theoretical_bits(
            changed_logits,
            self.tensors["qualities"],
            self.tensors["active_mask"],
        )
        torch.testing.assert_close(first_loss, second_loss, rtol=0.0, atol=0.0)
        self.assertEqual(first_bits, second_bits)
        self.assertEqual(first_bits[1], int(self.batch.active_mask.sum()))

    def test_q0_q41_are_valid_and_q42_active_target_is_rejected(self):
        logits = self.model.forward_full(**self.tensors)
        self.assertTrue(torch.isfinite(logits).all())
        invalid = self.tensors["qualities"].clone()
        invalid[0, 0] = QUALITY_PAD_ID
        with self.assertRaisesRegex(ValueError, "outside Q0-Q41"):
            self.model.forward_full(
                self.tensors["bases"],
                invalid,
                self.tensors["lengths"],
                self.tensors["active_mask"],
            )

    def test_forward_full_matches_every_forward_step_in_eval_mode(self):
        self.model.eval()
        with torch.no_grad():
            full = self.model.forward_full(**self.tensors)
            for cycle in range(full.shape[1]):
                step = self.model.forward_step(
                    self.tensors["bases"],
                    self.tensors["qualities"][:, :cycle],
                    self.tensors["lengths"],
                    self.tensors["active_mask"],
                )
                torch.testing.assert_close(
                    step, full[:, cycle], rtol=0.0, atol=0.0
                )

    def test_checkpoint_saves_and_restores_full_contract(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "checkpoint.pt"
            save_training_checkpoint(
                path,
                model=self.model,
                optimizer=optimizer,
                epoch=2,
                global_step=7,
                data_split={"strategy": "test", "datasets": {}},
                sampler_config={"strategy": "balanced"},
                sampler_statistics={"total_batches": 7},
                best_validation_bits_per_quality=3.5,
                validation_metrics={"symbol_micro_bits_per_quality": 3.5},
            )
            loaded = load_training_checkpoint(path)

        self.assertEqual(loaded.payload["model_config"], tiny_config().to_dict())
        self.assertEqual(loaded.payload["feature_schema"], feature_schema())
        self.assertEqual(loaded.payload["cache_schema"]["schema_version"], 1)
        self.assertEqual(loaded.payload["data_split"]["strategy"], "test")
        self.assertEqual(loaded.payload["sampler_statistics"]["total_batches"], 7)
        self.model.eval()
        loaded.model.eval()
        with torch.no_grad():
            expected = self.model.forward_full(**self.tensors)
            actual = loaded.model.forward_full(**self.tensors)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()
