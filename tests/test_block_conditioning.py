"""Explicit side-information conditioning, causality, and lossless Q checks."""

import dataclasses
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from codec.checkpoint import load_training_checkpoint, save_training_checkpoint
from codec.fastq_stream import make_fastq_batch
from codec.model import (
    CrossLayerOutputHead, DirectQualityModelConfig, DirectQualityTransformer,
    fastq_batch_to_tensors, feature_schema, masked_cross_entropy,
)
from codec.probability_quantization import logits_to_cdfs
from codec.range_decoder import RangeDecoder
from codec.range_encoder import RangeEncoder
from codec.read_state import ReadStateConditioning


def config(**changes):
    values = dict(
        d_model=8, num_heads=2, num_layers=4, feedforward_dim=16,
        prev_q_embed_dim=4, qmer_embed_dim=2, base_embed_dim=3,
        base_conv_channels=3, base_context_dim=4, context_length=8,
        dropout=0.0, block_conditioning_dim=5,
    )
    values.update(changes)
    return DirectQualityModelConfig(**values)


def tensors(lengths=(13, 5, 0)):
    batch = make_fastq_batch(
        [(np.arange(n, dtype=np.uint8) * 3 + 1) % 5 for n in lengths],
        [(np.arange(n, dtype=np.uint8) * 7 + 2) % 42 for n in lengths],
        list(range(len(lengths))), source_name="block-conditioning-test.fq",
    )
    return fastq_batch_to_tensors(batch, torch.device("cpu"))


def features(batch_size):
    return torch.linspace(0.05, 0.95, batch_size * 16).reshape(batch_size, 16)


def train_conditioner(model, batch, z):
    model.requires_grad_(False)
    model.block_conditioning.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.block_conditioning.parameters(), lr=.02, weight_decay=0)
    model.train()
    gradients = set()
    for _ in range(3):
        optimizer.zero_grad()
        loss = masked_cross_entropy(model.forward_full(**batch, block_features=z),
                                    batch["qualities"], batch["active_mask"])
        loss.backward()
        gradients.update(name for name, value in model.block_conditioning.named_parameters()
                         if value.grad is not None and bool(value.grad.abs().sum() > 0))
        optimizer.step()
    model.eval()
    return gradients


class BlockConditioningTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(20261104)

    def test_defaults_preserve_legacy_serialization_module_state_and_rng(self):
        ordinary = DirectQualityModelConfig()
        serialized = ordinary.to_dict()
        for name in ("block_quality_features", "block_reads", "block_feature_mode", "block_conditioning_dim"):
            self.assertNotIn(name, serialized)
        self.assertEqual(feature_schema(ordinary), feature_schema())
        torch.manual_seed(13)
        legacy = DirectQualityTransformer(config())
        next_legacy = torch.rand(4)
        torch.manual_seed(13)
        disabled = DirectQualityTransformer(config(block_reads=128, block_feature_mode="constant",
                                                    block_conditioning_dim=11))
        next_disabled = torch.rand(4)
        self.assertFalse(hasattr(disabled, "block_conditioning"))
        self.assertEqual(set(legacy.state_dict()), set(disabled.state_dict()))
        for name, value in legacy.state_dict().items():
            self.assertTrue(torch.equal(value, disabled.state_dict()[name]), name)
        self.assertTrue(torch.equal(next_legacy, next_disabled))

    def test_config_and_feature_schema_are_explicit_and_checkpointed(self):
        active = config(block_quality_features=True, block_reads=128)
        self.assertEqual(DirectQualityModelConfig.from_dict(active.to_dict()), active)
        schema = feature_schema(active)
        self.assertEqual(schema["version"], 4)
        self.assertEqual(schema["features"]["block_quality"]["block_reads"], 128)
        self.assertEqual(schema["features"]["block_quality"]["conditioning"]["mode"], "features")
        self.assertNotEqual(schema, feature_schema(dataclasses.replace(active, block_feature_mode="constant")))
        for changes in (
            dict(block_quality_features=1), dict(block_reads=0), dict(block_reads=True),
            dict(block_conditioning_dim=0), dict(block_feature_mode="unknown"),
            dict(block_quality_features=True, relative_position_bias=True),
            dict(block_quality_features=True, base_cross_attention=True),
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                config(**changes)

    def test_every_public_prediction_path_requires_explicit_z_and_legacy_rejects_it(self):
        batch = tensors()
        z = features(3)
        for enabled in (False, True):
            model = DirectQualityTransformer(config(block_quality_features=enabled)).eval()
            invalid_kwargs = {} if enabled else dict(block_features=z)
            for method in (model.forward_full, model.forward, model.forward_features,
                           model.forward_cross_layer_features):
                with self.subTest(enabled=enabled, method=method.__name__), self.assertRaisesRegex(ValueError, "block"):
                    method(**batch, **invalid_kwargs)
            with self.assertRaisesRegex(ValueError, "block"):
                model.forward_step(batch["bases"], batch["qualities"][:, :0],
                                   batch["lengths"], batch["active_mask"], **invalid_kwargs)

    def test_descriptor_validation_and_incompatible_read_states(self):
        batch = tensors()
        model = DirectQualityTransformer(config(block_quality_features=True)).eval()
        good = features(3)
        invalid = [good[:, :15], good.double(), good.long(), good.numpy(),
                   good + 1, good - 1,
                   torch.full_like(good, float("nan")), torch.full_like(good, float("inf")),
                   torch.empty((3, 16), device="meta")]
        for z in invalid:
            with self.subTest(value_type=type(z).__name__), self.assertRaises(ValueError):
                model.forward_full(**batch, block_features=z)
        with self.assertRaisesRegex(ValueError, "state"):
            model.forward_full(**batch, block_features=good, read_states=torch.zeros(3, dtype=torch.long))
        model.read_state_conditioning = ReadStateConditioning(8, [0, 1, 2, 3], [10000, 20000, 30000])
        with self.assertRaisesRegex(ValueError, "cannot combine"):
            model.forward_full(**batch, block_features=good)

    def test_zero_initialization_is_identity_and_control_has_same_parameters(self):
        torch.manual_seed(19)
        ordinary = DirectQualityTransformer(config()).eval()
        torch.manual_seed(19)
        conditioned = DirectQualityTransformer(config(block_quality_features=True)).eval()
        for name, value in ordinary.state_dict().items():
            self.assertTrue(torch.equal(value, conditioned.state_dict()[name]), name)
        torch.manual_seed(19)
        constant = DirectQualityTransformer(config(block_quality_features=True, block_feature_mode="constant")).eval()
        self.assertEqual(set(conditioned.state_dict()), set(constant.state_dict()))
        for name, value in conditioned.state_dict().items():
            self.assertTrue(torch.equal(value, constant.state_dict()[name]), name)
        batch = tensors()
        with torch.no_grad():
            expected = ordinary.forward_full(**batch)
            actual = conditioned.forward_full(**batch, block_features=features(3))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        np.testing.assert_array_equal(logits_to_cdfs(actual.reshape(-1, 42).numpy(), version=2),
                                      logits_to_cdfs(expected.reshape(-1, 42).numpy(), version=2))

    def test_training_uses_descriptor_and_constant_control_ignores_descriptor(self):
        batch = tensors()
        model = DirectQualityTransformer(config(block_quality_features=True))
        gradients = train_conditioner(model, batch, features(3))
        self.assertEqual(gradients, {name for name, _ in model.block_conditioning.named_parameters()})
        constant = DirectQualityTransformer(config(block_quality_features=True, block_feature_mode="constant"))
        constant.load_state_dict(model.state_dict(), strict=True)
        constant.eval()
        z0, z1 = torch.zeros((3, 16)), torch.ones((3, 16))
        with torch.no_grad():
            different = model.forward_full(**batch, block_features=z1) - model.forward_full(**batch, block_features=z0)
            a = constant.forward_full(**batch, block_features=z0)
            b = constant.forward_full(**batch, block_features=z1)
        self.assertGreater(float(different.abs().max()), 1e-6)
        self.assertTrue(torch.equal(a, b))

    def test_fixed_descriptor_preserves_quality_causality_for_plain_and_cross_heads(self):
        batch = tensors((17, 9, 0))
        z = features(3)
        model = DirectQualityTransformer(config(block_quality_features=True))
        train_conditioner(model, batch, z)
        changed = dict(batch)
        changed["qualities"] = batch["qualities"].clone()
        changed["qualities"][0, 5:] = 41 - changed["qualities"][0, 5:]
        changed["qualities"][1, 5:9] = 41 - changed["qualities"][1, 5:9]
        for cross in (False, True):
            if cross:
                model.output_head = CrossLayerOutputHead(8, 8, 4)
                model.eval()
            with self.subTest(cross=cross), torch.no_grad():
                a = model.forward_full(**batch, block_features=z)
                b = model.forward_full(**changed, block_features=z)
                torch.testing.assert_close(a[:, :6], b[:, :6], rtol=0, atol=0)
                final = model.forward_features(**batch, block_features=z)
                packed = model.forward_cross_layer_features(**batch, block_features=z)
                torch.testing.assert_close(packed[:, :, :8], final, rtol=0, atol=0)

    def test_trained_nonzero_film_has_identical_full_step_cdfs_and_real_range_roundtrip(self):
        batch = tensors((13, 5, 0))
        z = features(3)
        model = DirectQualityTransformer(config(block_quality_features=True))
        train_conditioner(model, batch, z)
        coder = RangeEncoder(backend="native")
        with torch.no_grad():
            full = model.forward_full(**batch, block_features=z)
            cdfs = []
            for cycle in range(full.shape[1]):
                active = batch["active_mask"][:, cycle].numpy()
                step = model.forward_step(batch["bases"], batch["qualities"][:, :cycle],
                                          batch["lengths"], batch["active_mask"], block_features=z)
                torch.testing.assert_close(full[:, cycle], step, rtol=0, atol=0)
                cdf = logits_to_cdfs(full[:, cycle].numpy()[active], version=2)
                np.testing.assert_array_equal(cdf, logits_to_cdfs(step.numpy()[active], version=2))
                y = batch["qualities"][:, cycle].numpy()[active]
                rows = np.arange(len(y))
                coder.encode_prevalidated_intervals(cdf[rows, y], cdf[rows, y + 1], cdf[:, -1])
                cdfs.append(cdf)
            decoder = RangeDecoder(coder.finish())
            decoded = torch.full_like(batch["qualities"], 42)
            for cycle in range(full.shape[1]):
                active = batch["active_mask"][:, cycle]
                step = model.forward_step(batch["bases"], decoded[:, :cycle], batch["lengths"],
                                          batch["active_mask"], block_features=z)
                cdf = logits_to_cdfs(step.numpy()[active.numpy()], version=2)
                np.testing.assert_array_equal(cdf, cdfs[cycle])
                decoded[active, cycle] = torch.as_tensor(decoder.decode_prevalidated_batch(cdf))
            decoder.finish()
        self.assertTrue(torch.equal(decoded, batch["qualities"]))

    def test_empty_reads_padding_and_checkpoint_roundtrip(self):
        batch = tensors()
        z = features(3)
        model = DirectQualityTransformer(config(block_quality_features=True))
        train_conditioner(model, batch, z)
        with torch.no_grad():
            expected = model.forward_full(**batch, block_features=z)
            self.assertTrue(torch.equal(expected[~batch["active_mask"]],
                                        torch.zeros_like(expected[~batch["active_mask"]])))
            self.assertEqual(model.forward_full(**tensors((0, 0)), block_features=features(2)).shape, (2, 0, 42))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "block.pt"
            save_training_checkpoint(path, model=model, optimizer=None, epoch=1, global_step=3,
                                     data_split={}, sampler_config={}, sampler_statistics={},
                                     best_validation_bits_per_quality=2.0, validation_metrics={})
            loaded = load_training_checkpoint(path).model.eval()
            self.assertEqual(loaded.config, model.config)
            with torch.no_grad():
                actual = loaded.forward_full(**batch, block_features=z)
            self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
