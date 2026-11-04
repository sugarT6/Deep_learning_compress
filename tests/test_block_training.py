import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from codec.block_features import CacheBlockFeatures
from codec.block_training import (
    BLOCK_SELECTION_METRIC,
    BlockDatasetMetrics,
    evaluate_block_reader_range,
    initialize_block_model,
    summarize_block_metrics,
    transfer_block_initialization,
    validate_parent_split,
)
from codec.datasets import DirectQualityDataset
from codec.fastq_stream import make_fastq_batch
from codec.model import DirectQualityModelConfig, DirectQualityTransformer, fastq_batch_to_tensors
from codec.training_cache import TrainingCacheReader, create_training_cache


def tiny_config(**changes):
    config = DirectQualityModelConfig(
        prev_q_embed_dim=4, qmer_embed_dim=2, base_embed_dim=3,
        base_conv_channels=3, base_context_dim=4, d_model=8,
        num_heads=2, num_layers=1, feedforward_dim=16,
        context_length=8, dropout=0.0,
    )
    return replace(config, **changes)


def manifest(train_stop=8, *, source_sha="abc", read_count=12):
    return {"datasets": {"A": {
        "source_sha256": source_sha, "read_count": read_count,
        "train_range": [0, train_stop], "validation_range": [train_stop, read_count],
    }}}


class RecordingModel(torch.nn.Module):
    """A deterministic causal distribution which exposes the received wire input."""

    def __init__(self, *, mode="features"):
        super().__init__()
        self.config = SimpleNamespace(
            block_quality_features=True, block_reads=4, block_feature_mode=mode,
        )
        self.conditions = []

    def forward_full(self, *, bases, qualities, lengths, active_mask, block_features):
        self.conditions.append(block_features.detach().cpu().numpy().copy())
        logits = torch.zeros((*bases.shape, 42), device=bases.device)
        logits[:, :, 0] = 2 * block_features[:, :1]
        return logits


class BlockTrainingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_ordinary_transfer_preserves_new_identity_conditioner_and_predictions(self):
        ordinary = DirectQualityTransformer(tiny_config()).eval()
        conditioned = DirectQualityTransformer(tiny_config(block_quality_features=True)).eval()
        new_state = {k: v.clone() for k, v in conditioned.block_conditioning.state_dict().items()}
        transfer_block_initialization(conditioned, ordinary)
        for key, value in ordinary.state_dict().items():
            self.assertTrue(torch.equal(value, conditioned.state_dict()[key]), key)
        for key, value in new_state.items():
            self.assertTrue(torch.equal(value, conditioned.block_conditioning.state_dict()[key]), key)
        batch = make_fastq_batch(
            [np.zeros(4, np.uint8)] * 2,
            [np.arange(4, dtype=np.uint8), np.arange(4, 8, dtype=np.uint8)],
            [0, 1], source_name="sample.fastq",
        )
        tensors = fastq_batch_to_tensors(batch, torch.device("cpu"))
        with torch.inference_mode():
            expected = ordinary.forward_full(**tensors)
            actual = conditioned.forward_full(**tensors, block_features=torch.rand(2, 16))
        self.assertTrue(torch.equal(expected, actual))

    def test_transfer_rejects_unrelated_configs_keys_and_nonzero_new_conditioner(self):
        source = DirectQualityTransformer(tiny_config())
        target = DirectQualityTransformer(tiny_config(block_quality_features=True))
        incompatible = DirectQualityTransformer(tiny_config(block_quality_features=True, context_length=9))
        with self.assertRaisesRegex(ValueError, "only add"):
            transfer_block_initialization(incompatible, source)
        source.register_buffer("unexpected", torch.zeros(1))
        with self.assertRaisesRegex(ValueError, "tensor keys"):
            transfer_block_initialization(target, source)
        source = DirectQualityTransformer(tiny_config())
        original_weight = source.output_head.weight
        source.output_head.weight = torch.nn.Parameter(original_weight[:, :4].clone())
        with self.assertRaisesRegex(ValueError, "shape/dtype"):
            transfer_block_initialization(target, source)
        source.output_head.weight = original_weight
        with torch.no_grad():
            target.block_conditioning.encoder[-1].bias.fill_(1)
        with self.assertRaisesRegex(ValueError, "zero-initialized"):
            transfer_block_initialization(target, source)

    def test_identical_block_parent_restores_trained_conditioner_and_rejects_mode_change(self):
        config = tiny_config(block_quality_features=True)
        source = DirectQualityTransformer(config)
        with torch.no_grad():
            source.block_conditioning.encoder[-1].weight.fill_(0.125)
        target = DirectQualityTransformer(config)
        transfer_block_initialization(target, source)
        for key, value in source.state_dict().items():
            self.assertTrue(torch.equal(value, target.state_dict()[key]), key)
        control = DirectQualityTransformer(replace(config, block_feature_mode="constant"))
        with self.assertRaisesRegex(ValueError, "identical"):
            transfer_block_initialization(control, source)

    def test_parent_split_checks_identity_and_prevents_downward_alignment_leak(self):
        parent = {"data_split": manifest(7)}
        validate_parent_split(parent, manifest(8))
        for changed in (manifest(4), manifest(8, source_sha="changed"), manifest(8, read_count=13)):
            with self.assertRaises(ValueError):
                validate_parent_split(parent, changed)
        with self.assertRaisesRegex(ValueError, "absent"):
            validate_parent_split(parent, {"datasets": {}})
        with self.assertRaisesRegex(ValueError, "lacks"):
            validate_parent_split({}, manifest())

    def test_initialization_validates_split_before_mutation_and_preserves_rng(self):
        source = DirectQualityTransformer(tiny_config())
        target = DirectQualityTransformer(tiny_config(block_quality_features=True))
        payload = {"model_config": source.config.to_dict(),
                   "model_state_dict": source.state_dict(), "data_split": manifest(7)}
        before = {k: v.clone() for k, v in target.state_dict().items()}
        with self.assertRaises(ValueError):
            initialize_block_model(target, payload, manifest(4))
        self.assertTrue(all(torch.equal(value, target.state_dict()[key]) for key, value in before.items()))
        random_state = torch.get_rng_state().clone()
        initialize_block_model(target, payload, manifest(8))
        self.assertTrue(torch.equal(random_state, torch.get_rng_state()))
        self.assertTrue(torch.equal(target.output_head.weight, source.output_head.weight))

    def _cache(self, root):
        fastq = root / "A.fastq"
        fastq.write_bytes(b"".join(
            b"@read\nAAAA\n+\n" + bytes([33 + q] * 4) + b"\n"
            for q in (0, 1, 2, 3, 20, 30, 41, 35, 5, 15)
        ))
        cache = root / "A.h5"
        create_training_cache(fastq, cache)
        return cache

    def _evaluate(self, model, reader, **changes):
        arguments = dict(category="validation", start=4, stop=10, batch_reads=1,
                         max_reads=3, device=torch.device("cpu"))
        arguments.update(changes)
        return evaluate_block_reader_range(
            model, reader, DirectQualityDataset("A", "Illumina", True), **arguments,
        )

    def test_validation_rounds_limit_up_and_uses_whole_block_features_once(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = self._cache(Path(directory))
            with TrainingCacheReader(cache) as reader:
                model = RecordingModel().train()
                metrics = self._evaluate(model, reader)
                self.assertTrue(model.training)
                self.assertEqual((metrics.reads, metrics.symbols, metrics.block_count), (4, 16, 1))
                self.assertEqual(metrics.block_side_bits, 128)
                expected = CacheBlockFeatures(reader, 4, split_start=4, split_stop=8).for_read_indices([4, 5, 6, 7])
                np.testing.assert_array_equal(np.concatenate(model.conditions), expected)
                # The microbatch contains one read, but all four share the full-block descriptor.
                self.assertTrue(np.array_equal(expected[0], expected[-1]))
                self.assertAlmostEqual(
                    metrics.quantized_Q_plus_block_side_bits_per_quality,
                    (metrics.quantized_bits + 128) / 16,
                )
                self.assertNotAlmostEqual(metrics.quantized_bits, metrics.total_bits, places=7)
                complete = self._evaluate(RecordingModel(), reader, max_reads=5, batch_reads=3)
                self.assertEqual((complete.reads, complete.block_count, complete.block_side_bits), (6, 2, 256))

    def test_constant_control_receives_zeros_and_has_no_transmitted_side_bits(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = self._cache(Path(directory))
            with TrainingCacheReader(cache) as reader:
                model = RecordingModel(mode="constant")
                metrics = self._evaluate(model, reader, max_reads=0, batch_reads=2)
                self.assertEqual(metrics.block_side_bits, 0)
                self.assertEqual(metrics.block_count, 2)
                self.assertTrue(all(np.count_nonzero(condition) == 0 for condition in model.conditions))
                self.assertAlmostEqual(metrics.bits_per_quality, math.log2(42), places=5)
                self.assertAlmostEqual(metrics.quantized_bits / metrics.symbols, math.log2(42), places=12)

    def test_evaluation_rejects_incomplete_boundaries_and_restores_training_on_error(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = self._cache(Path(directory))
            with TrainingCacheReader(cache) as reader:
                for changed in ({"start": 2}, {"stop": 9}, {"max_reads": -1}, {"batch_reads": 257}):
                    with self.assertRaises(ValueError):
                        self._evaluate(RecordingModel(), reader, **changed)
                model = RecordingModel().train()
                def fail(**kwargs):
                    raise RuntimeError("intentional model error")
                model.forward_full = fail
                with self.assertRaisesRegex(RuntimeError, "intentional"):
                    self._evaluate(model, reader)
                self.assertTrue(model.training)

    def test_summary_retains_float_q_metrics_and_selects_quantized_q_plus_side(self):
        metrics = [
            BlockDatasetMetrics("A", "Illumina", "validation", False,
                                2, 10, 20.0, 2.0, 21.0, 1, 128, 14.9),
            BlockDatasetMetrics("B", "MGI/BGI", "validation", False,
                                2, 30, 120.0, 4.0, 123.0, 1, 128, 251 / 30),
        ]
        summary = summarize_block_metrics(metrics)
        self.assertEqual(summary["dataset_macro_bits_per_quality"], 3.0)
        self.assertEqual(summary["block_side_bits"], 256)
        self.assertAlmostEqual(summary[BLOCK_SELECTION_METRIC], (14.9 + 251 / 30) / 2)
        self.assertAlmostEqual(summary["dataset_macro_quantized_Q_bits_per_quality"], 3.1)
        self.assertAlmostEqual(summary["symbol_micro_quantized_Q_plus_block_side_bits_per_quality"], 10.0)
        self.assertEqual(summary["per_dataset"]["A"]["bits_per_quality"], 2.0)
        self.assertIsNone(summarize_block_metrics([])[BLOCK_SELECTION_METRIC])


if __name__ == "__main__":
    unittest.main()
