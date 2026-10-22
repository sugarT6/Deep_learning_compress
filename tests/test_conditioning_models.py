"""Causality and lossless-wire checks for the two conditioning experiments."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from codec.checkpoint import load_training_checkpoint, save_training_checkpoint
from codec.container import sha256_file
from codec.decode import decode_fastq
from codec.encode import encode_fastq
from codec.fastq_stream import make_fastq_batch
from codec.head_adapter import (
    adapt_output_head, adapter_container_bytes, apply_head_adapter,
)
from codec.model import (
    CrossLayerOutputHead, DirectQualityModelConfig, DirectQualityTransformer,
    fastq_batch_to_tensors, feature_schema, masked_cross_entropy,
)
from codec.probability_quantization import logits_to_cdfs
from codec.suffix_lora_adapter import install_suffix_lora, serialize_suffix_lora
from test_head_adapter import config as adaptation_config, source_file


VARIANTS = ("relative_position_bias", "base_cross_attention")


def tiny_config(variant=None, **overrides):
    values = dict(
        d_model=8, num_heads=2, num_layers=4, feedforward_dim=16,
        prev_q_embed_dim=4, qmer_embed_dim=2, base_embed_dim=3,
        base_conv_channels=3, base_context_dim=4, context_length=8,
        dropout=0.0,
    )
    if variant:
        values[variant] = True
    if variant == "base_cross_attention":
        values.update(base_memory_dim=8, base_memory_heads=2,
                      base_memory_layers=1, base_cross_attention_layers=2)
    values.update(overrides)
    return DirectQualityModelConfig(**values)


def tensors_for(lengths=(13, 5, 0)):
    bases = [(np.arange(length, dtype=np.uint8) * 3 + 1) % 5
             for length in lengths]
    qualities = [(np.arange(length, dtype=np.uint8) * 7 + 2) % 42
                 for length in lengths]
    batch = make_fastq_batch(bases, qualities, list(range(len(lengths))),
                             source_name="conditioning-test.fq")
    return fastq_batch_to_tensors(batch, torch.device("cpu"))


def save_checkpoint(path, model, optimizer=None, step=1):
    save_training_checkpoint(
        path, model=model, optimizer=optimizer, epoch=1, global_step=step,
        data_split={}, sampler_config={}, sampler_statistics={},
        best_validation_bits_per_quality=2.0, validation_metrics={},
    )


def set_nonzero_relative_bias(model):
    """Exercise real additive masks, rather than the zero-bias special case."""
    parameters = [(name, parameter) for name, parameter in model.named_parameters()
                  if "relative" in name and "bias" in name]
    if model.config.relative_position_bias and not parameters:
        raise AssertionError("relative position experiment has no learned bias")
    with torch.no_grad():
        for _, parameter in parameters:
            parameter.copy_(torch.linspace(-.7, .8, parameter.numel(),
                                           dtype=parameter.dtype).reshape_as(parameter))


class ConditioningModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(20261022)

    def test_variant_configs_roundtrip_and_feature_contracts_are_distinct(self):
        schemas = []
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                config = tiny_config(variant)
                self.assertEqual(DirectQualityModelConfig.from_dict(config.to_dict()), config)
                self.assertIs(config.to_dict()[variant], True)
                schema = feature_schema(config)
                self.assertEqual(schema["version"], 3)
                self.assertNotEqual(schema, feature_schema(tiny_config()))
                schemas.append(schema)
        self.assertNotEqual(*schemas)

        invalid = (
            dict(relative_position_bias=True, base_cross_attention=True),
            dict(relative_position_bias=1),
            dict(base_cross_attention=1),
            dict(relative_position_bias=True, ffn_type="swiglu"),
            dict(base_cross_attention=True, ffn_type="swiglu"),
            dict(base_cross_attention=True, base_memory_dim=7, base_memory_heads=2),
            dict(base_cross_attention=True, base_memory_layers=0),
            dict(base_cross_attention=True, base_cross_attention_layers=0),
            dict(base_cross_attention=True, base_cross_attention_layers=5),
        )
        for options in invalid:
            with self.subTest(invalid=options), self.assertRaises(ValueError):
                tiny_config(**options)

    def test_future_qualities_cannot_affect_any_current_prediction_or_cross_head(self):
        tensors = tensors_for((17, 9, 0))
        changed = dict(tensors)
        changed["qualities"] = tensors["qualities"].clone()
        changed["qualities"][0, 5:] = 41 - changed["qualities"][0, 5:]
        changed["qualities"][1, 5:9] = 41 - changed["qualities"][1, 5:9]
        for variant in VARIANTS:
            for cross_head in (False, True):
                with self.subTest(variant=variant, cross_head=cross_head):
                    model = DirectQualityTransformer(tiny_config(variant)).eval()
                    set_nonzero_relative_bias(model)
                    if cross_head:
                        model.output_head = CrossLayerOutputHead(8, 8, 4)
                        with torch.no_grad():
                            model.output_head.up.weight.normal_(std=.07)
                            model.output_head.cross_up.weight.normal_(std=.07)
                    with torch.no_grad():
                        original = model.forward_full(**tensors)
                        different = model.forward_full(**changed)
                    # Cycle five predicts Q[5] from Q[:5], not Q[5] itself.
                    torch.testing.assert_close(original[:, :6], different[:, :6],
                                               rtol=0, atol=0)

    def test_base_memory_reads_distant_future_bases_known_to_the_decoder(self):
        model = DirectQualityTransformer(tiny_config("base_cross_attention")).eval()
        tensors = tensors_for((31,))
        # Also covers implementations that initialize a cross residual at zero.
        optimizer = torch.optim.Adam(model.parameters(), lr=.003)
        model.train()
        for _ in range(2):
            optimizer.zero_grad()
            loss = masked_cross_entropy(model.forward_full(**tensors),
                                        tensors["qualities"], tensors["active_mask"])
            loss.backward()
            optimizer.step()
        model.eval()
        changed = dict(tensors)
        changed["bases"] = tensors["bases"].clone()
        changed["bases"][0, 23:] = (changed["bases"][0, 23:] + 2) % 5
        with torch.no_grad():
            original = model.forward_full(**tensors)
            different = model.forward_full(**changed)
        self.assertGreater(float((original[0, 0] - different[0, 0]).abs().max()), 1e-7)
        # The retained local CNN alone cannot reach these future positions.
        ordinary = DirectQualityTransformer(tiny_config()).eval()
        with torch.no_grad():
            torch.testing.assert_close(ordinary.forward_full(**tensors)[:, 0],
                                       ordinary.forward_full(**changed)[:, 0],
                                       rtol=0, atol=0)

    def test_right_padding_and_empty_reads_remain_finite_and_do_not_change_valid_logits(self):
        short = tensors_for((5,))
        padded = tensors_for((5, 17, 0))
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                model = DirectQualityTransformer(tiny_config(variant)).eval()
                set_nonzero_relative_bias(model)
                with torch.no_grad():
                    short_logits = model.forward_full(**short)
                    padded_logits = model.forward_full(**padded)
                    empty_logits = model.forward_full(**tensors_for((0, 0)))
                    empty_batch_logits = model.forward_full(
                        bases=torch.empty((0, 0), dtype=torch.long),
                        qualities=torch.empty((0, 0), dtype=torch.long),
                        lengths=torch.empty((0,), dtype=torch.long),
                        active_mask=torch.empty((0, 0), dtype=torch.bool))
                for logits in (short_logits, padded_logits, empty_logits, empty_batch_logits):
                    self.assertTrue(bool(torch.isfinite(logits).all()))
                self.assertEqual(empty_logits.shape, (2, 0, 42))
                self.assertEqual(empty_batch_logits.shape, (0, 0, 42))
                torch.testing.assert_close(short_logits[0], padded_logits[0, :5],
                                           rtol=1e-5, atol=2e-6)
                self.assertTrue(torch.equal(padded_logits[~padded["active_mask"]],
                                            torch.zeros_like(padded_logits[~padded["active_mask"]])))

    def test_full_and_incremental_logits_and_integer_cdfs_are_identical(self):
        tensors = tensors_for((13, 5, 0))
        for variant in VARIANTS:
            for cross_head in (False, True):
                with self.subTest(variant=variant, cross_head=cross_head):
                    model = DirectQualityTransformer(tiny_config(variant)).eval()
                    set_nonzero_relative_bias(model)
                    if cross_head:
                        model.output_head = CrossLayerOutputHead(8, 8, 4)
                        with torch.no_grad():
                            model.output_head.up.weight.normal_(std=.07)
                            model.output_head.cross_up.weight.normal_(std=.07)
                    with torch.no_grad():
                        full = model.forward_full(**tensors)
                        for cycle in range(full.shape[1]):
                            step = model.forward_step(
                                tensors["bases"], tensors["qualities"][:, :cycle],
                                tensors["lengths"], tensors["active_mask"])
                            torch.testing.assert_close(full[:, cycle], step, rtol=0, atol=0)
                            for version in (1, 2):
                                np.testing.assert_array_equal(
                                    logits_to_cdfs(full[:, cycle].numpy(), version=version),
                                    logits_to_cdfs(step.numpy(), version=version))

    def test_new_parameters_receive_gradients_and_survive_checkpoint_roundtrip(self):
        tensors = tensors_for((13, 5, 0))
        legacy_names = set(DirectQualityTransformer(tiny_config()).state_dict())
        for variant in VARIANTS:
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as directory:
                model = DirectQualityTransformer(tiny_config(variant))
                new_parameters = {name: value for name, value in model.named_parameters()
                                  if name not in legacy_names}
                self.assertTrue(new_parameters)
                before = {name: value.detach().clone() for name, value in new_parameters.items()}
                optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=0)
                active_gradients = set()
                for _ in range(2):
                    optimizer.zero_grad()
                    loss = masked_cross_entropy(model.forward_full(**tensors),
                                                tensors["qualities"], tensors["active_mask"])
                    loss.backward()
                    active_gradients.update(name for name, value in new_parameters.items()
                                            if value.grad is not None
                                            and bool(value.grad.abs().sum() > 0))
                    optimizer.step()
                self.assertTrue(active_gradients, variant)
                self.assertTrue(any(not torch.equal(before[name], new_parameters[name])
                                    for name in active_gradients))
                if variant == "base_cross_attention":
                    # Gradients must reach both the base-only memory encoder
                    # and the Q decoder's query-to-memory attention.
                    self.assertTrue(any("memory" in name for name in active_gradients),
                                    sorted(active_gradients))
                    self.assertTrue(any("cross" in name for name in active_gradients),
                                    sorted(active_gradients))
                model.eval()
                checkpoint = Path(directory) / "trained.pt"
                save_checkpoint(checkpoint, model, optimizer, step=2)
                loaded = load_training_checkpoint(checkpoint)
                self.assertEqual(loaded.model.config, model.config)
                self.assertEqual(loaded.payload["global_step"], 2)
                self.assertEqual(loaded.payload["feature_schema"], feature_schema(model.config))
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(value, loaded.model.state_dict()[name], rtol=0, atol=0)
                with torch.no_grad():
                    torch.testing.assert_close(model.forward_full(**tensors),
                                               loaded.model.eval().forward_full(**tensors),
                                               rtol=0, atol=0)
                payload = torch.load(checkpoint, map_location="cpu")
                payload["feature_schema"] = feature_schema(tiny_config())
                bad = Path(directory) / "mismatched-schema.pt"
                torch.save(payload, bad)
                with self.assertRaisesRegex(ValueError, "feature schema"):
                    load_training_checkpoint(bad)

    def test_nonzero_all_layer_rank4_adapters_roundtrip_without_sending_new_modules(self):
        legacy_names = set(DirectQualityTransformer(tiny_config()).state_dict())
        for variant in VARIANTS:
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                base = DirectQualityTransformer(tiny_config(variant)).eval()
                set_nonzero_relative_bias(base)
                checkpoint = root / "base.pt"
                save_checkpoint(checkpoint, base)
                digest = sha256_file(checkpoint)
                candidate = copy.deepcopy(base).requires_grad_(False)
                candidate.output_head = CrossLayerOutputHead(8, 8, 4)
                layout = install_suffix_lora(candidate, 4, 4, 77)
                with torch.no_grad():
                    for item in layout:
                        module, name = item["name"].rsplit(".", 1)
                        factors = candidate.get_submodule(module).parametrizations[name][0]
                        factors.b.normal_(std=.03)
                    candidate.transformer.norm.bias.add_(.01)
                    candidate.output_head.up.weight.normal_(std=.03)
                    candidate.output_head.cross_up.weight.normal_(std=.03)
                for name, parameter in candidate.named_parameters():
                    if name not in legacy_names and "parametrizations" not in name:
                        if not name.startswith("output_head."):
                            self.assertFalse(parameter.requires_grad, name)
                adapter = serialize_suffix_lora(candidate, 4, 4, digest, base_model=base)
                self.assertEqual(adapter["version"], 8)
                self.assertEqual(adapter["layers"], [0, 1, 2, 3])
                self.assertTrue(all(item["name"] in legacy_names for item in adapter["matrices"]))
                restored = copy.deepcopy(base)
                apply_head_adapter(restored, adapter, digest)
                for name, tensor in base.state_dict().items():
                    if name not in legacy_names:
                        torch.testing.assert_close(restored.state_dict()[name], tensor, rtol=0, atol=0)
                artifact = root / "adapter.json"
                artifact.write_text(json.dumps(dict(
                    format="direct-quality-head-adaptation-artifact", version=1,
                    base_checkpoint_sha256=digest, adapter=adapter, report={})))
                source, raw = source_file(root, count=13, compressed=True)
                kwargs = dict(device=torch.device("cpu"), batch_reads=4,
                              progress=False, verify_cdf=True)
                for quantizer in (1, 2):
                    archive = root / ("archive%d.fqdc" % quantizer)
                    decoded = root / ("decoded%d.fq" % quantizer)
                    stats = encode_fastq(source, archive, checkpoint,
                                         head_adapter_path=artifact, quantization_version=quantizer,
                                         coder_backend="native", **kwargs)
                    # The decoder uses the Python range coder, independently
                    # checking the native encoder's integer-CDF compatibility.
                    decode_fastq(archive, decoded, checkpoint, **kwargs)
                    self.assertEqual(decoded.read_bytes(), raw)
                    self.assertEqual(stats.quality_and_adapter_bytes - stats.range_stream_bytes,
                                     adapter_container_bytes(adapter))

    def test_dense_suffix_fit_is_rejected_before_it_can_drop_new_parameters(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                model = DirectQualityTransformer(tiny_config(variant)).eval()
                checkpoint = root / "base.pt"
                save_checkpoint(checkpoint, model)
                source, _ = source_file(root, count=17)
                before = {name: value.detach().clone() for name, value in model.state_dict().items()}
                with self.assertRaisesRegex(ValueError, "LoRA|lora|dense|conditioning|relative|cross"):
                    adapt_output_head(
                        model, source, device=torch.device("cpu"), batch_reads=4,
                        total=65536, base_sha256=sha256_file(checkpoint),
                        config=adaptation_config(
                            steps=1, max_seconds=0, backbone_layers=2, backbone_lora_rank=0,
                            backbone_steps=1, backbone_eval_every=1))
                for name, tensor in model.state_dict().items():
                    torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)

    def test_real_per_file_lora_training_keeps_conditioning_parameters_frozen(self):
        legacy_names = set(DirectQualityTransformer(tiny_config()).state_dict())
        for variant in VARIANTS:
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                model = DirectQualityTransformer(tiny_config(variant)).eval()
                set_nonzero_relative_bias(model)
                before = {name: value.detach().clone() for name, value in model.state_dict().items()}
                checkpoint = root / "base.pt"
                save_checkpoint(checkpoint, model)
                source, _ = source_file(root, count=65)
                adapter, report = adapt_output_head(
                    model, source, device=torch.device("cpu"), batch_reads=4,
                    total=65536, base_sha256=sha256_file(checkpoint),
                    config=adaptation_config(
                        max_reads=8, max_symbols=100, steps=1, symbols_per_step=32,
                        learning_rate=.003, max_seconds=0,
                        head_type="residual", residual_dim=8, cross_layer=True, cross_dim=4,
                        backbone_layers=4, backbone_lora_rank=4,
                        backbone_steps=2, backbone_eval_every=1, backbone_batch_reads=4,
                        backbone_max_reads=32, backbone_max_symbols=1000))
                self.assertEqual(adapter["version"], 8)
                self.assertTrue(report["backbone_accepted"], report)
                self.assertEqual(report["backbone_steps_completed"], 2)
                for name, tensor in model.state_dict().items():
                    if name not in legacy_names and not name.startswith("output_head."):
                        torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)
                self.assertTrue(any(not torch.equal(model.state_dict()[item["name"]],
                                                    before[item["name"]])
                                    for item in adapter["matrices"]))


if __name__ == "__main__":
    unittest.main()
