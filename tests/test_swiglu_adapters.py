"""SwiGLU gate weights participate in suffix fitting and lossless wire restore."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch

from codec.backbone_adapter import (
    PROTOCOL as DENSE_GELU_PROTOCOL,
    SWIGLU_PROTOCOL as DENSE_SWIGLU_PROTOCOL,
    serialize_backbone, validate_backbone_adapter,
)
from codec.checkpoint import save_training_checkpoint
from codec.container import ContainerError, read_container, sha256_file
from codec.decode import decode_fastq
from codec.encode import encode_fastq
from codec.head_adapter import (
    adapt_output_head, apply_head_adapter, serialize_head, validate_head_adapter,
)
from codec.model import CrossLayerOutputHead, DirectQualityModelConfig, DirectQualityTransformer
from codec.read_state import ReadStateConditioning, serialize_read_state
from codec.suffix_lora_adapter import (
    PROTOCOL as LORA_GELU_PROTOCOL,
    SWIGLU_PROTOCOL as LORA_SWIGLU_PROTOCOL,
    install_suffix_lora, serialize_suffix_lora, validate_suffix_lora,
)
from test_head_adapter import config, rewrite_metadata, source_file


class SwiGLUAdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def make_base(self, root, ffn_type="swiglu"):
        torch.manual_seed(253)
        cfg = DirectQualityModelConfig(
            d_model=8, num_heads=2, num_layers=4, feedforward_dim=16,
            prev_q_embed_dim=4, qmer_embed_dim=2, base_embed_dim=3,
            base_conv_channels=3, base_context_dim=4, dropout=.1,
            ffn_type=ffn_type,
        )
        base = DirectQualityTransformer(cfg).eval()
        checkpoint = root / f"base_{ffn_type}.pt"
        save_training_checkpoint(
            checkpoint, model=base, optimizer=None, epoch=1, global_step=1,
            data_split={}, sampler_config={}, sampler_statistics={},
            best_validation_bits_per_quality=1., validation_metrics={},
        )
        return base, checkpoint

    def make_adapter(self, base, digest, rank):
        candidate = copy.deepcopy(base).eval().requires_grad_(False)
        candidate.output_head = CrossLayerOutputHead(8, 16, 4)
        with torch.no_grad():
            candidate.output_head.up.weight.normal_(std=.01)
            candidate.output_head.cross_up.weight.normal_(std=.01)
            candidate.transformer.norm.bias.add_(.01)
        if rank:
            matrices = install_suffix_lora(candidate, 2, rank, 97)
            with torch.no_grad():
                for item in matrices:
                    module_name, name = item["name"].rsplit(".", 1)
                    candidate.get_submodule(module_name).parametrizations[name][0].b.normal_(std=.02)
            adapter = serialize_suffix_lora(candidate, 2, rank, digest, base_model=base)
        else:
            with torch.no_grad():
                for layer in candidate.transformer.layers[-2:]:
                    layer.linear1.weight.add_(.01)
                    if base.config.ffn_type == "swiglu":
                        layer.linear_gate.weight.add_(.02)
                        layer.linear_gate.bias.add_(.03)
            adapter = serialize_backbone(candidate, 2, digest, base_model=base)
        return candidate, adapter

    def assert_parameters_equal(self, actual, expected):
        actual_parameters = dict(actual.named_parameters())
        expected_parameters = dict(expected.named_parameters())
        self.assertEqual(set(actual_parameters), set(expected_parameters))
        for name, parameter in expected_parameters.items():
            torch.testing.assert_close(actual_parameters[name], parameter, rtol=0, atol=0)

    def test_gate_replacements_roundtrip_repeat_switch_and_restore(self):
        with tempfile.TemporaryDirectory() as directory:
            base, checkpoint = self.make_base(Path(directory))
            digest = sha256_file(checkpoint)
            dense_candidate, dense = self.make_adapter(base, digest, 0)
            _, lora = self.make_adapter(base, digest, 8)
            self.assertEqual(dense["protocol"], DENSE_SWIGLU_PROTOCOL)
            self.assertEqual(lora["protocol"], LORA_SWIGLU_PROTOCOL)
            dense_gates = [i for i in dense["tensors"] if ".linear_gate." in i["name"]]
            lora_gates = [i for i in lora["matrices"] if ".linear_gate." in i["name"]]
            self.assertEqual(len(dense_gates), 4)  # weight and bias in two layers
            self.assertEqual(len(lora_gates), 2)  # weight only; gate biases stay frozen
            self.assertTrue(all(i["name"].endswith(".weight") for i in lora_gates))
            restored = copy.deepcopy(base)
            for adapter in (dense, lora, dense, lora, lora):
                apply_head_adapter(restored, adapter, digest)
                expected = copy.deepcopy(base)
                apply_head_adapter(expected, adapter, digest)
                self.assert_parameters_equal(restored, expected)
                for index in (0, 1):
                    self.assertTrue(torch.equal(restored.transformer.layers[index].linear_gate.weight,
                                                base.transformer.layers[index].linear_gate.weight))
                for index in (2, 3):
                    self.assertFalse(torch.equal(restored.transformer.layers[index].linear_gate.weight,
                                                 base.transformer.layers[index].linear_gate.weight))
                    if adapter is lora:
                        self.assertTrue(torch.equal(restored.transformer.layers[index].linear_gate.bias,
                                                    base.transformer.layers[index].linear_gate.bias))
                if adapter is dense:
                    self.assert_parameters_equal(restored, dense_candidate)
            apply_head_adapter(restored, serialize_head(base.output_head, digest), digest)
            self.assert_parameters_equal(restored, base)
            self.assertFalse(hasattr(restored, "_backbone_original_parameters"))

    def test_protocol_type_and_gate_layout_are_strict_and_architecture_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, checkpoint = self.make_base(root)
            gelu, _ = self.make_base(root, "gelu")
            digest = sha256_file(checkpoint)
            for rank, validator, old_protocol in (
                (0, validate_backbone_adapter, DENSE_GELU_PROTOCOL),
                (8, validate_suffix_lora, LORA_GELU_PROTOCOL),
            ):
                with self.subTest(rank=rank):
                    _, adapter = self.make_adapter(base, digest, rank)
                    validate_head_adapter(adapter, 8, digest)
                    for change in ({"ffn_type": "gelu"}, {"ffn_type": True}, {"protocol": old_protocol}):
                        with self.assertRaises(ValueError):
                            validator(dict(adapter, **change), 8, digest)
                    missing_type = dict(adapter)
                    del missing_type["ffn_type"]
                    with self.assertRaises(ValueError):
                        validator(missing_type, 8, digest)
                    layout_field = "matrices" if rank else "tensors"
                    missing_gate = dict(adapter, **{
                        layout_field: [i for i in adapter[layout_field] if ".linear_gate." not in i["name"]],
                    })
                    with self.assertRaisesRegex(ValueError, "layout"):
                        validator(missing_gate, 8, digest)
                    original = copy.deepcopy(gelu)
                    with self.assertRaisesRegex(ValueError, "configuration mismatch"):
                        apply_head_adapter(gelu, adapter, digest)
                    self.assert_parameters_equal(gelu, original)
                    # A legacy payload keeps its exact old field set and cannot
                    # acquire even an explicit default-architecture field.
                    _, legacy = self.make_adapter(gelu, digest, rank)
                    self.assertEqual(legacy["protocol"], old_protocol)
                    self.assertNotIn("ffn_type", legacy)
                    with self.assertRaises(ValueError):
                        validator(dict(legacy, ffn_type="gelu"), 8, digest)
                    original = copy.deepcopy(base)
                    with self.assertRaisesRegex(ValueError, "configuration mismatch"):
                        apply_head_adapter(base, legacy, digest)
                    self.assert_parameters_equal(base, original)

    def test_dense_lora_and_read_state_encode_decode_have_exact_cdfs(self):
        for rank, read_state in ((0, False), (8, False), (0, True), (8, True)):
            with self.subTest(rank=rank, read_state=read_state), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                base, checkpoint = self.make_base(root)
                digest = sha256_file(checkpoint)
                candidate, adapter = self.make_adapter(base, digest, rank)
                if read_state:
                    candidate.read_state_conditioning = ReadStateConditioning(8, [2, 3], [10000])
                    with torch.no_grad():
                        candidate.read_state_conditioning.shift.normal_(std=.01)
                    adapter = serialize_read_state(candidate, adapter, digest)
                artifact = root / "adapter.json"
                artifact.write_text(json.dumps({
                    "format": "direct-quality-head-adaptation-artifact", "version": 1,
                    "base_checkpoint_sha256": digest, "adapter": adapter, "report": {},
                }))
                source, raw = source_file(root, count=17, compressed=True)
                target, output = root / "data.fqdc", root / "decoded.fq"
                options = dict(device=torch.device("cpu"), batch_reads=8, progress=False, verify_cdf=True)
                encode_fastq(source, target, checkpoint, head_adapter_path=artifact,
                             quantization_version=2, coder_backend="python", **options)
                decode_fastq(target, output, checkpoint, **options)
                self.assertEqual(output.read_bytes(), raw)
                bad = root / "bad_ffn.fqdc"
                rewrite_metadata(target, bad, lambda meta: meta["checkpoint"]["model_config"].update(ffn_type="gelu"))
                with self.assertRaisesRegex(ContainerError, "architecture"):
                    read_container(bad)

    def test_real_dense_and_lora_training_updates_gate_weights_only_in_suffix(self):
        for rank in (0, 8):
            with self.subTest(rank=rank), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                model, checkpoint = self.make_base(root)
                before = copy.deepcopy(model)
                source = root / "fit.fq"
                source.write_bytes(b"@r\nACGTACGT\n+\n++++++++\n" * 96)
                cfg = config(
                    max_reads=32, max_symbols=256, steps=1, max_seconds=0,
                    head_type="residual", residual_dim=16, cross_layer=True, cross_dim=4,
                    backbone_layers=2, backbone_lora_rank=rank, backbone_steps=10,
                    backbone_batch_reads=8, backbone_learning_rate=.01, backbone_eval_every=10,
                    backbone_max_reads=64, backbone_max_symbols=512, quantization_version=2,
                )
                adapter, report = adapt_output_head(
                    model, source, device=torch.device("cpu"), batch_reads=16,
                    total=65536, base_sha256=sha256_file(checkpoint), config=cfg,
                )
                self.assertTrue(report["backbone_accepted"], report)
                self.assertEqual(adapter["ffn_type"], "swiglu")
                for index in range(4):
                    actual, previous = model.transformer.layers[index], before.transformer.layers[index]
                    self.assertEqual(torch.equal(actual.linear_gate.weight, previous.linear_gate.weight), index < 2)
                    if rank or index < 2:
                        self.assertTrue(torch.equal(actual.linear_gate.bias, previous.linear_gate.bias))
                if rank:
                    self.assertEqual(report["backbone_trainable_parameters"] * 4, adapter["raw_bytes"])


if __name__ == "__main__":
    unittest.main()
