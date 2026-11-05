"""Final-step adaptation without holdout scoring, including paid block LoRA."""

import contextlib
import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

import codec.backbone_adapter as backbone_module
import codec.head_adapter as head_module
from codec.block_features import decode_block_features, encode_block_features
from codec.checkpoint import load_training_checkpoint
from codec.container import BLOCK_FEATURE_CONTAINER_VERSION, read_container, sha256_file
from codec.decode import decode_fastq
from codec.encode import encode_fastq
from codec.fastq_stream import iter_fastq_batches
from codec.head_adapter import HeadAdaptationConfig, adapt_output_head, apply_head_adapter
from codec.model import fastq_batch_to_tensors
from codec.probability_quantization import logits_to_cdfs
import test_block_codec as block_codec_helpers
import test_head_adapter as head_helpers


def adaptation_config(**overrides):
    values = dict(
        max_reads=16, max_symbols=512, steps=2, symbols_per_step=64,
        learning_rate=.01, anchor_strength=0, max_seconds=0,
        head_type="residual", residual_dim=8, cross_layer=True, cross_dim=4,
        quantization_version=2, backbone_layers=4, backbone_lora_rank=4,
        backbone_steps=3, backbone_batch_reads=4, backbone_learning_rate=.003,
        backbone_eval_every=1, backbone_patience=0,
        backbone_max_reads=16, backbone_max_symbols=512,
    )
    values.update(overrides)
    return HeadAdaptationConfig(**values)


@contextlib.contextmanager
def forbid_adaptation_validation():
    """A hidden scorer, scheduler or head gain gate must fail immediately."""
    with mock.patch.object(backbone_module, "score_model", side_effect=AssertionError("unexpected holdout score")) as score, \
            mock.patch.object(backbone_module, "TrainingControl", side_effect=AssertionError("unexpected validation control")) as control, \
            mock.patch.object(head_module, "logits_to_cdfs", side_effect=AssertionError("unexpected head holdout score")) as head_score:
        yield score, control, head_score


class UnvalidatedAdaptationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def fit(self, model, source, checkpoint, config, batch_reads):
        return adapt_output_head(model, source, device=torch.device("cpu"),
                                 batch_reads=batch_reads, total=65536,
                                 base_sha256=sha256_file(checkpoint), config=config)

    def assert_final_report(self, report, config):
        self.assertTrue(report["accepted"])
        self.assertTrue(report["backbone_accepted"])
        self.assertFalse(report["backbone_fallback"])
        self.assertEqual(report["steps_completed"], config.steps)
        self.assertEqual(report["backbone_steps_completed"], config.backbone_steps)
        self.assertEqual(report["backbone_selected_step"], config.backbone_steps)
        self.assertEqual(report["backbone_selection_policy"], "final_trained_step_without_validation")
        self.assertEqual(report["reason"], "backbone_final_trained_step_without_validation")
        self.assertEqual(report["backbone_validation_history"], [])
        self.assertEqual(report["backbone_validation_scope"], "disabled")
        self.assertEqual(report["backbone_validation_reads"], 0)
        self.assertEqual(report["backbone_train_reads"], report["backbone_sampling"]["sampled_reads"])
        self.assertIsNone(report["backbone_before_validation_bits_per_quality"])
        self.assertIsNone(report["backbone_after_validation_bits_per_quality"])
        self.assertIsNone(report["estimated_quality_saving_bytes"])
        self.assertIsNone(report["estimated_net_saving_bytes"])
        self.assertNotIn("wire_validation", report["backbone_stage_seconds"])
        warmup = report["warmup_report"]
        self.assertEqual(warmup["reason"], "final_step_without_validation")
        self.assertEqual(warmup["train_reads"], warmup["prefix_reads"])
        self.assertEqual(warmup["validation_reads"], 0)
        self.assertEqual(warmup["validation_symbols"], 0)
        self.assertIsNone(warmup["before_validation_bits_per_quality"])
        self.assertIsNone(warmup["after_validation_bits_per_quality"])
        self.assertIsNone(warmup["selection_metric"])

    def test_default_still_uses_holdout_periodic_scores_and_best_trained_selection(self):
        self.assertTrue(HeadAdaptationConfig().validation_enabled)
        with self.assertRaises(ValueError):
            HeadAdaptationConfig(validation_enabled=1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, checkpoint = head_helpers.model_and_checkpoint(root, num_layers=4)
            source, _ = head_helpers.source_file(root, count=80)
            original_score = backbone_module.score_model
            original_control = backbone_module.TrainingControl
            cfg = adaptation_config()
            with mock.patch.object(backbone_module, "score_model", wraps=original_score) as score, \
                    mock.patch.object(backbone_module, "TrainingControl", wraps=original_control) as control:
                adapter, report = self.fit(model, source, checkpoint, cfg, 4)
            self.assertEqual(adapter["version"], 8)
            self.assertEqual(control.call_count, 1)
            self.assertEqual(score.call_count, 1 + cfg.backbone_steps + 2)
            self.assertEqual([entry["step"] for entry in report["backbone_validation_history"]], [0, 1, 2, 3])
            self.assertEqual(report["backbone_selection_policy"], "best_trained_state_without_gain_gate")
            self.assertGreater(report["backbone_validation_reads"], 0)
            self.assertGreater(report["warmup_report"]["validation_reads"], 0)
            self.assertIsNotNone(report["backbone_before_validation_bits_per_quality"])
            self.assertIsNotNone(report["backbone_after_validation_bits_per_quality"])
            self.assertIn(report["backbone_selected_step"], (1, 2, 3))

    def test_disabled_plain_all4_lora_never_scores_or_constructs_validation_control(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, checkpoint = head_helpers.model_and_checkpoint(root, num_layers=4)
            source, _ = head_helpers.source_file(root, count=80)
            cfg = adaptation_config(validation_enabled=False, min_gain_bits_per_quality=1e6)
            with forbid_adaptation_validation() as forbidden:
                adapter, report = self.fit(model, source, checkpoint, cfg, 4)
            self.assertEqual(adapter["version"], 8)
            self.assertEqual(adapter["rank"], 4)
            self.assertEqual(adapter["layers"], [0, 1, 2, 3])
            self.assert_final_report(report, cfg)
            for mocked in forbidden:
                mocked.assert_not_called()

    def test_disabled_block_all4_lora_is_deterministic_and_freezes_block_conditioner(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = block_codec_helpers.BlockCodecTest().checkpoint(root, num_layers=4)
            base = load_training_checkpoint(checkpoint).model
            source = root / "source.fastq"
            source.write_bytes(block_codec_helpers.fixture_bytes(41))
            cfg = adaptation_config(validation_enabled=False)
            fitted = []
            for _ in range(2):
                model = copy.deepcopy(base)
                with forbid_adaptation_validation():
                    adapter, report = self.fit(model, source, checkpoint, cfg, 4)
                self.assert_final_report(report, cfg)
                changed_names = {entry["name"] for entry in adapter["matrices"]}
                changed_names.update(("transformer.norm.weight", "transformer.norm.bias"))
                for name, value in base.state_dict().items():
                    if not name.startswith("output_head.") and name not in changed_names:
                        self.assertTrue(torch.equal(value, model.state_dict()[name]), name)
                for name, value in base.block_conditioning.state_dict().items():
                    self.assertTrue(torch.equal(value, model.block_conditioning.state_dict()[name]), name)
                self.assertEqual(adapter["version"], 8)
                self.assertEqual(adapter["layers"], [0, 1, 2, 3])
                self.assertEqual(adapter["rank"], 4)
                fitted.append((model, adapter))
            self.assertEqual(fitted[0][1], fitted[1][1])
            for name, value in fitted[0][0].state_dict().items():
                self.assertTrue(torch.equal(value, fitted[1][0].state_dict()[name]), name)
            restored = copy.deepcopy(base)
            for _ in range(2):
                apply_head_adapter(restored, fitted[0][1], sha256_file(checkpoint))
                for name, value in fitted[0][0].state_dict().items():
                    self.assertTrue(torch.equal(value, restored.state_dict()[name]), name)
            batch = next(iter_fastq_batches(source, batch_reads=4))
            t = fastq_batch_to_tensors(batch, torch.device("cpu"))
            z = torch.as_tensor(decode_block_features(encode_block_features(batch)).copy()).unsqueeze(0).expand(4, -1)
            with torch.no_grad():
                a = fitted[0][0].eval().forward_full(**t, block_features=z)
                b = restored.eval().forward_full(**t, block_features=z)
            np.testing.assert_array_equal(logits_to_cdfs(a[t["active_mask"]].numpy(), version=2),
                                          logits_to_cdfs(b[t["active_mask"]].numpy(), version=2))

    def test_block_all4_lora_final_adapter_native_codec_roundtrip_and_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = block_codec_helpers.BlockCodecTest().checkpoint(root, num_layers=4)
            source = root / "source.fastq"
            original = block_codec_helpers.fixture_bytes(41)
            source.write_bytes(original)
            cfg = adaptation_config(validation_enabled=False)
            artifact, target = root / "trained.adapter.json", root / "trained.fqdc"
            common = dict(device=torch.device("cpu"), batch_reads=4,
                          progress=False, verify_cdf=True)
            with forbid_adaptation_validation():
                stats = encode_fastq(source, target, checkpoint, **common,
                                     head_adaptation_config=cfg, save_head_adapter_path=artifact,
                                     quantization_version=2, coder_backend="native")
            self.assert_final_report(stats.head_adaptation, cfg)
            info = read_container(target, verify_checksums=True)
            self.assertEqual(info.metadata["format_version"], BLOCK_FEATURE_CONTAINER_VERSION)
            self.assertEqual(info.metadata["head_adapter"]["version"], 8)
            self.assertEqual(info.metadata["head_adapter"]["rank"], 4)
            decoded = root / "decoded.fastq"
            decode_fastq(target, decoded, checkpoint, **common)
            self.assertEqual(original, decoded.read_bytes())
            reloaded = root / "reloaded.fqdc"
            reloaded_stats = encode_fastq(source, reloaded, checkpoint, **common,
                                          head_adapter_path=artifact,
                                          quantization_version=2, coder_backend="native")
            self.assertTrue(reloaded_stats.head_adaptation["accepted"])
            restored_info = read_container(reloaded)
            self.assertEqual(info.read_section("quality_range"), restored_info.read_section("quality_range"))
            self.assertEqual(info.read_section("block_features"), restored_info.read_section("block_features"))
            decoded_again = root / "decoded_again.fastq"
            decode_fastq(reloaded, decoded_again, checkpoint, **common)
            self.assertEqual(original, decoded_again.read_bytes())
            self.assertEqual(json.loads(artifact.read_text())["adapter"], info.metadata["head_adapter"])


if __name__ == "__main__":
    unittest.main()
