import base64
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zlib

import numpy as np
import torch

import codec.backbone_adapter as backbone_module
from codec.container import ContainerError, read_container, sha256_file
from codec.decode import decode_fastq
from codec.encode import encode_fastq, main
from codec.head_adapter import (
    adapt_output_head, adapter_container_bytes, apply_head_adapter, serialize_head,
)
from codec.model import CrossLayerOutputHead
from codec.suffix_lora_adapter import (
    install_suffix_lora, serialize_suffix_lora, validate_suffix_lora,
)
import test_backbone_adapter as backbone_tests
from test_head_adapter import config, model_and_checkpoint, rewrite_metadata, source_file


class SuffixLoRATest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def candidate(self, root, depth=4, rank=2):
        base, checkpoint = model_and_checkpoint(root, num_layers=depth)
        candidate = copy.deepcopy(base).eval().requires_grad_(False)
        candidate.output_head = CrossLayerOutputHead(8, 16, 4)
        rng = torch.get_rng_state().clone()
        layout = install_suffix_lora(candidate, 2, rank, 77)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        with torch.no_grad():
            for item in layout:
                module, name = item["name"].rsplit(".", 1)
                factors = candidate.get_submodule(module).parametrizations[name][0]
                factors.b.normal_(std=.03)
            candidate.transformer.norm.bias.add_(.01)
        digest = sha256_file(checkpoint)
        adapter = serialize_suffix_lora(candidate, 2, rank, digest, base_model=base)
        return base, candidate, checkpoint, adapter

    def test_wire_roundtrip_and_exact_cdfs_for_four_and_six_layers(self):
        for depth, rank in ((4, 8), (6, 16)):
            with self.subTest(depth=depth, rank=rank), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                base, _, checkpoint, adapter = self.candidate(root, depth, rank)
                digest = sha256_file(checkpoint)
                source, raw = source_file(root, count=17, compressed=True)
                artifact = root / "adapter.json"
                artifact.write_text(json.dumps(dict(format="direct-quality-head-adaptation-artifact",
                    version=1, base_checkpoint_sha256=digest, adapter=adapter, report={})))
                for quantizer in (1, 2):
                    target, output = root / f"out{quantizer}.fqdc", root / f"decoded{quantizer}.fq"
                    kwargs = dict(device=torch.device("cpu"), batch_reads=8, progress=False, verify_cdf=True)
                    stats = encode_fastq(source, target, checkpoint, head_adapter_path=artifact,
                        quantization_version=quantizer, coder_backend="native", **kwargs)
                    decode_fastq(target, output, checkpoint, **kwargs)
                    self.assertEqual(output.read_bytes(), raw)
                    self.assertEqual(stats.quality_and_adapter_bytes - stats.range_stream_bytes,
                                     adapter_container_bytes(adapter))
                    bad = root / f"bad{quantizer}.fqdc"
                    rewrite_metadata(target, bad, lambda m: m["checkpoint"]["model_config"].update(feedforward_dim=32))
                    with self.assertRaisesRegex(ContainerError, "architecture"):
                        read_container(bad)
                restored = copy.deepcopy(base)
                apply_head_adapter(restored, adapter, digest)
                first = {n: p.detach().clone() for n, p in restored.named_parameters()}
                apply_head_adapter(restored, adapter, digest)
                self.assertFalse(any("parametrizations" in n for n, _ in restored.named_parameters()))
                for name, p in restored.named_parameters():
                    torch.testing.assert_close(p, first[name], rtol=0, atol=0)
                # All six projection slices participate, including K, O and both FFN matrices.
                base_params = dict(base.named_parameters())
                for item in adapter["matrices"]:
                    old, new = base_params[item["name"]], first[item["name"]]
                    for a, b in zip(old.chunk(item["parts"]), new.chunk(item["parts"])):
                        self.assertFalse(torch.equal(a, b), item["name"])
                apply_head_adapter(restored, serialize_head(base.output_head, digest), digest)
                for name, p in restored.named_parameters():
                    torch.testing.assert_close(p, base_params[name], rtol=0, atol=0)

    def test_switching_dense_legacy_lora_and_suffix_lora_uses_pristine_base(self):
        from codec.backbone_adapter import serialize_backbone
        from codec.lora_adapter import QVLoRA, serialize_lora
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, _, checkpoint, suffix = self.candidate(root)
            digest = sha256_file(checkpoint)
            dense_model = copy.deepcopy(base)
            with torch.no_grad():
                dense_model.transformer.layers[2].linear1.weight.add_(.05)
            dense = serialize_backbone(dense_model, 2, digest, base_model=base)
            old_lora = serialize_lora(CrossLayerOutputHead(8, 16, 4), QVLoRA(8, 2, 9), digest)
            changed = copy.deepcopy(base)
            for previous, next_adapter in ((suffix, dense), (dense, suffix), (suffix, old_lora),
                                           (old_lora, suffix), (suffix, suffix)):
                apply_head_adapter(changed, previous, digest)
                apply_head_adapter(changed, next_adapter, digest)
                expected = copy.deepcopy(base)
                apply_head_adapter(expected, next_adapter, digest)
                for name, p in expected.named_parameters():
                    torch.testing.assert_close(p, dict(changed.named_parameters())[name], rtol=0, atol=0)

    def test_rejects_invalid_and_nonfinite_payloads_before_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, _, checkpoint, adapter = self.candidate(root)
            digest = sha256_file(checkpoint)
            changes = [dict(version=True), dict(rank=True), dict(rank=17), dict(scaling=2),
                       dict(num_layers=True), dict(layers=[0, 3]), dict(extra=True),
                       dict(raw_bytes=adapter["raw_bytes"] + 4), dict(data_base64="!"),
                       dict(sha256="0" * 64), dict(parameter_bytes=True),
                       dict(matrices=[{**adapter["matrices"][0], "parts": True}] + adapter["matrices"][1:]),
                       dict(head={**adapter["head"], "version": 8})]
            for change in changes:
                with self.subTest(change=list(change)), self.assertRaises(ValueError):
                    validate_suffix_lora({**adapter, **change}, 8, digest)
            for suffix in (b"trailing", zlib.compress(b"second stream")):
                bad = {**adapter, "data_base64": base64.b64encode(
                    base64.b64decode(adapter["data_base64"]) + suffix).decode()}
                with self.assertRaises(ValueError):
                    validate_suffix_lora(bad, 8, digest)
            before = {n: p.detach().clone() for n, p in base.named_parameters()}
            bad = {**adapter, "merged_sha256": "0" * 64}
            with self.assertRaisesRegex(ValueError, "merged.*SHA"):
                apply_head_adapter(base, bad, digest)
            for name, p in base.named_parameters():
                torch.testing.assert_close(p, before[name], rtol=0, atol=0)
            planes = zlib.decompress(base64.b64decode(adapter["data_base64"]))
            raw = bytearray(np.frombuffer(planes, dtype=np.uint8).reshape(4, -1).T.copy().tobytes())
            raw[:4] = np.asarray([float("nan")], dtype="<f4").tobytes()
            bad = {**adapter, "sha256": hashlib.sha256(raw).hexdigest(), "data_base64": base64.b64encode(
                zlib.compress(np.frombuffer(raw, dtype=np.uint8).reshape(-1, 4).T.copy().tobytes())).decode()}
            with self.assertRaisesRegex(ValueError, "finite"):
                apply_head_adapter(base, bad, digest)
            # Bounded decompression must also reject a stream with excess output.
            bad = {**adapter, "data_base64": base64.b64encode(zlib.compress(
                bytes(adapter["raw_bytes"] + 4))).decode()}
            with self.assertRaisesRegex(ValueError, "length"):
                validate_suffix_lora(bad, 8, digest)
            different, _ = model_and_checkpoint(root, num_layers=6)
            with self.assertRaisesRegex(ValueError, "configuration mismatch"):
                apply_head_adapter(different, adapter, digest)

    def fit(self, root, count=10000, steps=40):
        model, before, source, checkpoint, cfg = backbone_tests.BackboneAdapterTest().fit_case(root, count, steps)
        return model, before, source, checkpoint, replace(cfg, backbone_lora_rank=2)

    def test_real_training_updates_only_target_matrices_norm_and_head(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, before, source, checkpoint, cfg = self.fit(root)
            adapter, report = adapt_output_head(model, source, device=torch.device("cpu"),
                batch_reads=16, total=65536, base_sha256=sha256_file(checkpoint), config=cfg)
            self.assertTrue(report["backbone_accepted"], report)
            self.assertEqual(adapter["version"], 8)
            self.assertEqual(report["selected_adapter_type"], "suffix_lora")
            self.assertFalse(report["backbone_fallback"])
            self.assertEqual(report["backbone_steps_completed"], 40)
            self.assertGreater(report["estimated_net_saving_bytes"], 0)
            targets = {i["name"] for i in adapter["matrices"]}
            for name, p in model.named_parameters():
                if not name.startswith(("output_head.", "transformer.norm.")) and name not in targets:
                    torch.testing.assert_close(p, before[name], rtol=0, atol=0)
            for layer in (2, 3):
                for matrix in ("self_attn.in_proj_weight", "self_attn.out_proj.weight", "linear1.weight", "linear2.weight"):
                    name = f"transformer.layers.{layer}.{matrix}"
                    self.assertFalse(torch.equal(dict(model.named_parameters())[name], before[name]), name)
            self.assertEqual(report["backbone_trainable_parameters"] * 4, adapter["raw_bytes"])

    def test_best_factors_are_restored_and_weight_cost_can_reject(self):
        for reject in (False, True):
            with self.subTest(reject=reject), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                model, before, source, checkpoint, cfg = self.fit(root, 96 if reject else 10000, 30)
                values = [6., 4., 5., 6., 6., 3.]
                selected = {}
                def score(candidate, records, *, quantized, **kwargs):
                    value = values.pop(0)
                    if not quantized and value == 4.:
                        for item in candidate.transformer.layers[2:]:
                            selected[len(selected)] = item.linear1.weight.detach().clone()
                    return value
                with mock.patch.object(backbone_module, "score_model", side_effect=score):
                    adapter, report = adapt_output_head(model, source, device=torch.device("cpu"),
                        batch_reads=16, total=65536, base_sha256=sha256_file(checkpoint), config=cfg)
                self.assertEqual(report["backbone_selected_step"], 10)
                self.assertEqual(report["backbone_accepted"], not reject)
                if reject:
                    self.assertTrue(report["backbone_fallback"])
                    self.assertIn(report["selected_adapter_type"], ("output_head", "base_model"))
                    self.assertEqual(report["reason"], "backbone_weight_cost_exceeds_estimated_saving")
                    for name, p in model.named_parameters():
                        if not name.startswith("output_head."):
                            torch.testing.assert_close(p, before[name], rtol=0, atol=0)
                else:
                    self.assertEqual(adapter["version"], 8)
                    for i, layer in enumerate(model.transformer.layers[2:]):
                        # Training FP32 matmul and canonical FP64 merge may differ by one rounding.
                        torch.testing.assert_close(layer.linear1.weight, selected[i], rtol=1e-6, atol=1e-7)

    def test_cli_guards_and_rank_parameter_cost(self):
        common = ["codec.encode", "in.fq", "out.fqdc", "base.pt", "--finetune-head"]
        with mock.patch("sys.argv", common + ["--finetune-last-layers", "2", "--backbone-lora-rank", "8"]), \
             mock.patch("codec.encode.encode_fastq") as encode, mock.patch("builtins.print"):
            encode.return_value.to_dict.return_value = {}
            main()
            self.assertEqual(encode.call_args.kwargs["head_adaptation_config"].backbone_lora_rank, 8)
        for flags in (["--backbone-lora-rank", "8"], ["--finetune-last-layers", "2", "--backbone-lora-rank", "17"]):
            with mock.patch("sys.argv", common + flags), self.assertRaises(SystemExit):
                main()
        with self.assertRaises(ValueError):
            config(backbone_layers=2, backbone_lora_rank=True)
        from codec.suffix_lora_adapter import matrix_layout
        layout = matrix_layout(256, 512, [2, 3])
        for rank, expected in ((8, 231424), (16, 460800)):
            size = 4 * (sum(rank * (i["shape"][0] + i["parts"] * i["shape"][1]) for i in layout) + 512)
            self.assertEqual(size, expected)


if __name__ == "__main__":
    unittest.main()
