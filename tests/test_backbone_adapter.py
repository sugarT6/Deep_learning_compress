import base64
import copy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock
import zlib

import torch

import codec.backbone_adapter as backbone_module
from codec.backbone_adapter import (
    apply_backbone_adapter, sample_records, serialize_backbone, validate_backbone_adapter,
)
from codec.container import ContainerError, read_container, sha256_file
from codec.decode import decode_fastq
from codec.encode import encode_fastq, main
from codec.head_adapter import (
    adapt_output_head, adapter_container_bytes, apply_head_adapter, serialize_head,
)
from codec.model import CrossLayerOutputHead
from test_head_adapter import config, model_and_checkpoint, rewrite_metadata, source_file


class BackboneAdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def candidate(self, root, depth=4):
        model, ckpt = model_and_checkpoint(root, num_layers=depth)
        base = copy.deepcopy(model)
        head = CrossLayerOutputHead(8, 16, 4)
        with torch.no_grad():
            head.weight.copy_(model.output_head.weight)
            head.bias.copy_(model.output_head.bias)
            head.up.weight.normal_(std=.03)
            head.cross_up.weight.normal_(std=.03)
            for layer in model.transformer.layers[-2:]:
                layer.self_attn.in_proj_weight.add_(.01)
                layer.linear1.weight.add_(.02)
            model.transformer.norm.weight.add_(.01)
        model.output_head = head
        return base, model, ckpt

    def test_wire_replacements_roundtrip_both_quantizers_and_restore_base(self):
        for depth in (4, 6):
            with self.subTest(depth=depth), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                base, candidate, ckpt = self.candidate(root, depth)
                digest = sha256_file(ckpt)
                adapter = serialize_backbone(candidate, 2, digest, base_model=base)
                restored = copy.deepcopy(base)
                for _ in range(2):
                    apply_head_adapter(restored, adapter, digest)
                    for name, parameter in candidate.named_parameters():
                        torch.testing.assert_close(parameter, dict(restored.named_parameters())[name], rtol=0, atol=0)
                source, raw = source_file(root, count=17, compressed=True)
                artifact = root / "adapter.json"
                artifact.write_text(json.dumps(dict(format="direct-quality-head-adaptation-artifact",
                    version=1, base_checkpoint_sha256=digest, adapter=adapter, report={})))
                for quantizer in (1, 2):
                    target, output = root / f"out{quantizer}.fqdc", root / f"decoded{quantizer}.fq"
                    kwargs = dict(device=torch.device("cpu"), batch_reads=8, progress=False, verify_cdf=True)
                    stats = encode_fastq(source, target, ckpt, head_adapter_path=artifact,
                        quantization_version=quantizer, coder_backend="python", **kwargs)
                    decode_fastq(target, output, ckpt, **kwargs)
                    self.assertEqual(output.read_bytes(), raw)
                    self.assertEqual(stats.quality_and_adapter_bytes - stats.range_stream_bytes,
                                     adapter_container_bytes(adapter))
                    bad = root / f"bad_architecture{quantizer}.fqdc"
                    rewrite_metadata(target, bad, lambda m: m["checkpoint"]["model_config"].update(feedforward_dim=32))
                    with self.assertRaisesRegex(ContainerError, "architecture"):
                        read_container(bad)
                apply_head_adapter(restored, serialize_head(base.output_head, digest), digest)
                for name, parameter in base.named_parameters():
                    torch.testing.assert_close(parameter, dict(restored.named_parameters())[name], rtol=0, atol=0)
                self.assertFalse(hasattr(restored, "_backbone_original_parameters"))
                # Architecture binding fails before a changed tensor can be installed.
                different, _ = model_and_checkpoint(root, num_layers=depth + 1)
                with self.assertRaisesRegex(ValueError, "configuration mismatch"):
                    apply_backbone_adapter(different, adapter, digest)

    def test_switching_lora_and_dense_adapters_preserves_pristine_base(self):
        from codec.lora_adapter import QVLoRA, serialize_lora
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, candidate, ckpt = self.candidate(root)
            digest = sha256_file(ckpt)
            dense = serialize_backbone(candidate, 2, digest, base_model=base)
            factors = QVLoRA(8, 4, 17)
            with torch.no_grad():
                factors.qb.fill_(.01)
                factors.vb.fill_(.02)
            lora = serialize_lora(candidate.output_head, factors, digest)
            restored, expected_lora = copy.deepcopy(base), copy.deepcopy(base)
            apply_head_adapter(restored, lora, digest)
            apply_head_adapter(restored, dense, digest)
            for name, parameter in candidate.named_parameters():
                torch.testing.assert_close(parameter, dict(restored.named_parameters())[name], rtol=0, atol=0)
            apply_head_adapter(restored, lora, digest)
            apply_head_adapter(expected_lora, lora, digest)
            for name, parameter in expected_lora.named_parameters():
                torch.testing.assert_close(parameter, dict(restored.named_parameters())[name], rtol=0, atol=0)

    def test_wire_rejects_invalid_layout_payload_and_nonfinite_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, candidate, ckpt = self.candidate(root)
            digest = sha256_file(ckpt)
            adapter = serialize_backbone(candidate, 2, digest, base_model=base)
            changes = [dict(version=True), dict(num_layers=True), dict(layers=[0, 3]),
                       dict(feedforward_dim=32), dict(raw_bytes=adapter["raw_bytes"] + 4),
                       dict(parameter_bytes=True), dict(delta_sha256="0" * 64), dict(data_base64="!"),
                       dict(dtype="float16"), dict(compression="unknown"), dict(extra=True),
                       dict(head={**adapter["head"], "version": 7})]
            for change in changes:
                with self.subTest(change=list(change)), self.assertRaises(ValueError):
                    validate_backbone_adapter({**adapter, **change}, 8, digest)
            for suffix in (b"trailing", zlib.compress(b"second stream")):
                bad = {**adapter, "data_base64": base64.b64encode(
                    base64.b64decode(adapter["data_base64"]) + suffix).decode()}
                with self.assertRaises(ValueError):
                    validate_backbone_adapter(bad, 8, digest)
            bad = {**adapter, "sha256": "0" * 64}
            with self.assertRaisesRegex(ValueError, "reconstructed.*SHA"):
                apply_backbone_adapter(base, bad, digest)
            with torch.no_grad():
                candidate.transformer.layers[-1].linear1.weight[0, 0] = float("nan")
            with self.assertRaisesRegex(ValueError, "finite"):
                serialize_backbone(candidate, 2, digest, base_model=base)

    def test_reservoir_uses_later_reads_reproducibly_and_respects_caps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _ = source_file(root, count=200)
            cfg = config(backbone_layers=2, backbone_max_reads=40, backbone_max_symbols=100)
            a, info_a = sample_records(source, cfg, 48, float("inf"))
            b, info_b = sample_records(source, cfg, 48, float("inf"))
            self.assertEqual([r.read_index for r in a], [r.read_index for r in b])
            self.assertEqual(info_a, info_b)
            self.assertTrue(info_a["sampling_complete"])
            self.assertEqual(info_a["total_file_reads"], 200)
            self.assertTrue(all(r.read_index >= 48 and len(r.quality) for r in a))
            self.assertLessEqual(len(a), 40)
            self.assertLessEqual(sum(len(r.quality) for r in a), 100)
            a, info = sample_records(source, cfg, 48, time.perf_counter() - 1)
            self.assertFalse(info["sampling_complete"])
            self.assertFalse(a)

    def fit_case(self, root, count=10000, steps=40):
        model, ckpt = model_and_checkpoint(root, num_layers=4)
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        source = root / "training.fq"
        source.write_bytes(b"@r\nACGTACGT\n+\n++++++++\n" * count)
        cfg = config(max_reads=32, max_symbols=256, steps=1, max_seconds=0,
                     head_type="residual", residual_dim=16, cross_layer=True, cross_dim=4,
                     backbone_layers=2, backbone_steps=steps, backbone_batch_reads=8,
                     backbone_learning_rate=.01, backbone_eval_every=10,
                     backbone_max_reads=64, backbone_max_symbols=512,
                     quantization_version=2)
        return model, before, source, ckpt, cfg

    def test_dense_training_improves_and_changes_only_suffix_norm_and_head(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, before, source, ckpt, cfg = self.fit_case(root)
            adapter, report = adapt_output_head(model, source, device=torch.device("cpu"),
                batch_reads=16, total=65536, base_sha256=sha256_file(ckpt), config=cfg)
            self.assertEqual(report["backbone_steps_completed"], 40)
            self.assertFalse(report["budget_exceeded"])
            self.assertTrue(report["backbone_accepted"], report)
            self.assertEqual(adapter["version"], 7)
            self.assertGreater(report["estimated_net_saving_bytes"], 0)
            changed = []
            for name, parameter in model.named_parameters():
                if name.startswith(("output_head.", "transformer.layers.2.", "transformer.layers.3.", "transformer.norm.")):
                    if name in before and not torch.equal(parameter, before[name]):
                        changed.append(name)
                else:
                    torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
            self.assertTrue(any(n.startswith("transformer.layers.2.") for n in changed))
            self.assertTrue(any(n.startswith("transformer.layers.3.") for n in changed))
            self.assertGreaterEqual(report["backbone_sampling"]["total_file_symbols"], 80000)

    def test_best_state_is_selected_before_later_worse_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, _, source, ckpt, cfg = self.fit_case(root, steps=30)
            values = iter((6., 4., 5., 6.))
            selected, quantized_calls = {}, []
            def score(candidate, records, *, quantized, **kwargs):
                if quantized:
                    quantized_calls.append(True)
                    return 6. if len(quantized_calls) == 1 else 3.
                value = next(values)
                if value == 4.:
                    selected.update({name: p.detach().clone() for name, p in candidate.named_parameters()})
                return value
            with mock.patch.object(backbone_module, "score_model", side_effect=score):
                adapter, report = adapt_output_head(model, source, device=torch.device("cpu"),
                    batch_reads=16, total=65536, base_sha256=sha256_file(ckpt), config=cfg)
            self.assertEqual(report["backbone_selected_step"], 10)
            self.assertEqual(adapter["version"], 7)
            for name, parameter in model.named_parameters():
                torch.testing.assert_close(parameter, selected[name], rtol=0, atol=0)

    def test_gain_that_cannot_pay_for_weights_falls_back_without_changing_backbone(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, before, source, ckpt, cfg = self.fit_case(root, count=96, steps=10)
            with mock.patch.object(backbone_module, "score_model", side_effect=[6., 4., 6., 3.]):
                adapter, report = adapt_output_head(model, source, device=torch.device("cpu"),
                    batch_reads=16, total=65536, base_sha256=sha256_file(ckpt), config=cfg)
            self.assertFalse(report["backbone_accepted"])
            self.assertEqual(report["reason"], "backbone_weight_cost_exceeds_estimated_saving")
            self.assertLess(report["estimated_net_saving_bytes"], 0)
            for name, parameter in model.named_parameters():
                if not name.startswith("output_head."):
                    torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
            self.assertTrue(adapter is None or adapter["version"] != 7)

    def test_cli_wiring_and_guards(self):
        common = ["codec.encode", "in.fq", "out.fqdc", "base.pt"]
        with mock.patch("sys.argv", common + ["--finetune-head", "--finetune-last-layers", "2", "--head-max-seconds", "0"]), \
             mock.patch("codec.encode.encode_fastq") as encode, mock.patch("builtins.print"):
            encode.return_value.to_dict.return_value = {}
            main()
            cfg = encode.call_args.kwargs["head_adaptation_config"]
            self.assertEqual(cfg.backbone_layers, 2)
            self.assertEqual(cfg.max_seconds, 0)
        for flags in (["--finetune-last-layers", "2"], ["--backbone-steps", "10"],
                      ["--finetune-head", "--finetune-last-layers", "5"],
                      ["--finetune-head", "--head-type", "residual", "--head-cross-layer",
                       "--head-lora", "--finetune-last-layers", "2"]):
            with mock.patch("sys.argv", common + flags), self.assertRaises(SystemExit):
                main()


if __name__ == "__main__":
    unittest.main()
