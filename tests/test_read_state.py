import base64
import copy
import gzip
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

import codec.backbone_adapter as backbone_module
from codec.backbone_adapter import serialize_backbone
from codec.container import ContainerError, read_container, sha256_file
from codec.decode import decode_fastq
from codec.encode import encode_fastq, main
from codec.fastq_stream import iter_fastq_batches
from codec.head_adapter import adapt_output_head, adapter_container_bytes, apply_head_adapter, serialize_head
from codec.model import CrossLayerOutputHead, fastq_batch_to_tensors
from codec.read_state import (
    ReadStateConditioning, StateReader, StateWriter, classify_reads,
    conditioned_tensors, fit_thresholds, mean_milli, serialize_read_state,
    state_container_overhead,
)
from codec.suffix_lora_adapter import install_suffix_lora, serialize_suffix_lora
from test_head_adapter import config, model_and_checkpoint, rewrite_metadata, source_file


class ReadStateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def candidate(self, root, depth=4, rank=8):
        base, checkpoint = model_and_checkpoint(root, num_layers=depth)
        candidate = copy.deepcopy(base).eval().requires_grad_(False)
        candidate.output_head = CrossLayerOutputHead(8, 16, 4)
        layers = list(range(depth - 2, depth))
        candidate.read_state_conditioning = ReadStateConditioning(8, layers, [5000, 15000, 30000])
        if rank:
            layout = install_suffix_lora(candidate, 2, rank, 77)
            with torch.no_grad():
                for item in layout:
                    module, name = item["name"].rsplit(".", 1)
                    candidate.get_submodule(module).parametrizations[name][0].b.normal_(std=.03)
        with torch.no_grad():
            candidate.read_state_conditioning.scale.normal_(std=.03)
            candidate.read_state_conditioning.shift.normal_(std=.03)
        digest = sha256_file(checkpoint)
        nested = (serialize_suffix_lora(candidate, 2, rank, digest, base_model=base) if rank
                  else serialize_backbone(candidate, 2, digest, base_model=base))
        adapter = serialize_read_state(candidate, nested, digest)
        return base, candidate, checkpoint, adapter

    def artifact(self, root, checkpoint, adapter):
        path = root / "adapter.json"
        path.write_text(json.dumps(dict(format="direct-quality-head-adaptation-artifact",
            version=1, base_checkpoint_sha256=sha256_file(checkpoint), adapter=adapter, report={})))
        return path

    def test_integer_means_thresholds_and_tensor_classification(self):
        qualities = np.array([[0, 41, 42], [10, 42, 42], [41, 41, 41], [42, 42, 42]])
        lengths = np.array([2, 1, 3, 0])
        np.testing.assert_array_equal(mean_milli(qualities, lengths), [20500, 10000, 41000, 0])
        expected = classify_reads(qualities, lengths, [10000, 20000, 41000])
        np.testing.assert_array_equal(expected, [2, 1, 3, 0])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, _ = model_and_checkpoint(root, 4)
            model.read_state_conditioning = ReadStateConditioning(8, [2, 3], [10000, 20000, 41000])
            tensors = dict(qualities=torch.tensor(qualities), lengths=torch.tensor(lengths),
                active_mask=torch.arange(3)[None, :] < torch.tensor(lengths)[:, None])
            np.testing.assert_array_equal(conditioned_tensors(model, tensors)["read_states"].numpy(), expected)
            source, _ = source_file(root)
            from codec.fastq_stream import iter_fastq_records
            records = list(iter_fastq_records(source))
            thresholds = fit_thresholds(records[:48])
            self.assertEqual(thresholds, [10000, 10000, 10000])
            self.assertEqual(classify_reads(np.full((1, 2), 10), [2], thresholds).tolist(), [3])

    def test_continuous_packing_and_strict_padding(self):
        handle = io.BytesIO()
        writer = StateWriter(handle)
        writer.write(np.array([0, 1, 2]))
        writer.write(np.array([3, 3, 2]))
        writer.write(np.array([1]))
        writer.finish()
        self.assertEqual(handle.getvalue(), bytes([0x1b, 0xe4]))
        self.assertEqual(writer.class_counts.tolist(), [1, 2, 2, 2])
        reader = StateReader(io.BytesIO(handle.getvalue()), 7)
        self.assertEqual(reader.read(5).tolist() + reader.read(2).tolist(), [0, 1, 2, 3, 3, 2, 1])
        reader.finish()
        for raw in (bytes([0x1b, 0xe5]), bytes([0x1b, 0xe4, 0])):
            reader = StateReader(io.BytesIO(raw), 7)
            reader.read(7)
            with self.assertRaises(ValueError):
                reader.finish()
        with self.assertRaisesRegex(ValueError, "truncated"):
            StateReader(io.BytesIO(b""), 1).read(1)

    def test_roundtrip_both_quantizers_depths_and_cross_batch_state_packing(self):
        for depth, rank in ((4, 8), (6, 16), (4, 0)):
            with self.subTest(depth=depth, rank=rank), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                base, _, checkpoint, adapter = self.candidate(root, depth, rank)
                source, raw = source_file(root, count=17, compressed=True)
                artifact = self.artifact(root, checkpoint, adapter)
                for quantizer in (1, 2):
                    target, output = root / f"out{quantizer}.fqdc", root / f"decoded{quantizer}.fq"
                    kwargs = dict(device=torch.device("cpu"), batch_reads=5, progress=False, verify_cdf=True)
                    stats = encode_fastq(source, target, checkpoint, head_adapter_path=artifact,
                        quantization_version=quantizer, coder_backend="native", **kwargs)
                    decode_fastq(target, output, checkpoint, **kwargs)
                    self.assertEqual(output.read_bytes(), raw)
                    info = read_container(target)
                    self.assertEqual(info.metadata["format_version"], 4)
                    self.assertEqual(stats.read_state_stream_bytes, 5)
                    self.assertEqual(stats.quality_and_adapter_bytes, stats.range_stream_bytes
                                     + adapter_container_bytes(adapter) + sum(state_container_overhead(info)))
                    # Label JSON and the fifth descriptor are charged exactly.
                    stripped = copy.deepcopy(info.metadata)
                    del stripped["read_state"]
                    stripped["sections"].pop()
                    dump = lambda v: json.dumps(v, sort_keys=True, separators=(",", ":")).encode()
                    self.assertEqual(len(dump(info.metadata)) - len(dump(stripped)), stats.read_state_metadata_bytes)
                # Loaded adapters generate states afresh for another input; no
                # source labels are hidden in the exported adapter artifact.
                source.write_bytes(gzip.compress(b"@different\nAC\n+\nJJ\n"))
                target, output = root / "other.fqdc", root / "other.fq"
                encode_fastq(source, target, checkpoint, head_adapter_path=artifact, **kwargs)
                artifact.unlink()
                decode_fastq(target, output, checkpoint, **kwargs)
                self.assertEqual(output.read_bytes(), b"@different\nAC\n+\nJJ\n")

    def test_loaded_state_adapter_handles_empty_files_and_all_empty_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, checkpoint, adapter = self.candidate(root)
            artifact = self.artifact(root, checkpoint, adapter)
            for i, raw in enumerate((b"", b"@empty\n\n+\n\n" * 7)):
                source, target, output = root / "empty.fq", root / f"empty{i}.fqdc", root / f"empty{i}.fq"
                source.write_bytes(raw)
                kwargs = dict(device=torch.device("cpu"), batch_reads=3, progress=False, verify_cdf=True)
                encode_fastq(source, target, checkpoint, head_adapter_path=artifact, **kwargs)
                decode_fastq(target, output, checkpoint, **kwargs)
                self.assertEqual(output.read_bytes(), raw)

    def test_state_affects_main_backbone_and_future_q_is_causal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, _, checkpoint, adapter = self.candidate(root)
            apply_head_adapter(base, adapter, sha256_file(checkpoint))
            source, _ = source_file(root, count=17)
            batch = next(iter_fastq_batches(source, batch_reads=17))
            tensors = fastq_batch_to_tensors(batch, torch.device("cpu"))
            states = torch.zeros(17, dtype=torch.long)
            base.eval()
            with torch.inference_mode():
                zero = base.forward_full(**tensors, read_states=states)
                three = base.forward_full(**tensors, read_states=states + 3)
                self.assertFalse(torch.equal(zero[tensors["active_mask"]], three[tensors["active_mask"]]))
                changed = dict(tensors)
                changed["qualities"] = tensors["qualities"].clone()
                changed["qualities"][:, 3:][tensors["active_mask"][:, 3:]] = 20
                future = base.forward_full(**changed, read_states=states)
                torch.testing.assert_close(zero[:, :4], future[:, :4], rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, "requires one read-state"):
                base.forward_full(**tensors)

    def test_idempotence_switching_and_invalid_payloads_do_not_mutate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, _, checkpoint, adapter = self.candidate(root)
            digest = sha256_file(checkpoint)
            changed = copy.deepcopy(base)
            for next_adapter in (adapter, adapter["backbone"], adapter,
                                 serialize_head(base.output_head, digest), adapter):
                apply_head_adapter(changed, next_adapter, digest)
                expected = copy.deepcopy(base)
                apply_head_adapter(expected, next_adapter, digest)
                self.assertEqual(set(changed.state_dict()), set(expected.state_dict()))
                for name, value in expected.state_dict().items():
                    torch.testing.assert_close(changed.state_dict()[name], value, rtol=0, atol=0)
            before = copy.deepcopy(changed.state_dict())
            mutations = [dict(version=True), dict(class_count=True), dict(layers=[2, True]),
                         dict(thresholds_milli=[30000, 20000, 10000]), dict(thresholds_milli=[0, 1, 41001]),
                         dict(raw_bytes=True), dict(parameter_bytes=1), dict(data_base64="!"),
                         dict(backbone=adapter), dict(extra=1), dict(sha256="0" * 64)]
            raw = bytearray(base64.b64decode(adapter["data_base64"]))
            raw[:4] = np.array([float("nan")], dtype="<f4").tobytes()
            mutations.append(dict(data_base64=base64.b64encode(raw).decode(), sha256=hashlib.sha256(raw).hexdigest()))
            for mutation in mutations:
                with self.subTest(mutation=list(mutation)), self.assertRaises(ValueError):
                    apply_head_adapter(changed, {**adapter, **mutation}, digest)
                for name, value in before.items():
                    torch.testing.assert_close(changed.state_dict()[name], value, rtol=0, atol=0)

    def test_container_rejects_bad_label_contract_and_architecture(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, checkpoint, adapter = self.candidate(root)
            source, _ = source_file(root, count=17)
            target = root / "valid.fqdc"
            encode_fastq(source, target, checkpoint, device=torch.device("cpu"), batch_reads=5,
                         progress=False, head_adapter_path=self.artifact(root, checkpoint, adapter))
            changes = [lambda m: m["read_state"].update(label_count=18),
                       lambda m: m["read_state"].update(valid_bits=True),
                       lambda m: m["read_state"].update(thresholds_milli=[1, 2, 3]),
                       lambda m: m["sections"][-1].update(uncompressed_length=6),
                       lambda m: m["checkpoint"]["model_config"].update(feedforward_dim=32),
                       lambda m: m.update(head_adapter=adapter["backbone"])]
            for i, change in enumerate(changes):
                bad = root / f"bad{i}.fqdc"
                rewrite_metadata(target, bad, change)
                with self.assertRaises(ContainerError):
                    read_container(bad)

    def test_joint_training_transmits_conditioning_and_uses_best_trained_state(self):
        for rank in (0, 2):
            with self.subTest(rank=rank), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                model, checkpoint = model_and_checkpoint(root, 4)
                before = copy.deepcopy(model.state_dict())
                source = root / "training.fq"
                source.write_bytes(b"".join(b"@r\nACGTACGT\n+\n" + bytes([33 + q]) * 8 + b"\n"
                                             for q in (5, 15, 25, 35) for _ in range(64)))
                cfg = config(max_reads=32, max_symbols=256, steps=2, max_seconds=0,
                    head_type="residual", residual_dim=16, cross_layer=True, cross_dim=4,
                    backbone_layers=2, backbone_steps=6, backbone_batch_reads=8,
                    backbone_learning_rate=.01, backbone_eval_every=2,
                    backbone_max_reads=64, backbone_max_symbols=512,
                    backbone_lora_rank=rank, read_quality_state=True)
                # Every trained checkpoint is worse than step zero and projected
                # total savings are negative: the state adapter must still be used.
                with mock.patch.object(backbone_module, "score_model", side_effect=[1., 3., 2., 4., 1., 2.]):
                    adapter, report = adapt_output_head(model, source, device=torch.device("cpu"),
                        batch_reads=16, total=65536, base_sha256=sha256_file(checkpoint), config=cfg)
                self.assertEqual(adapter["version"], 9)
                self.assertEqual(report["backbone_selected_step"], 4)
                self.assertFalse(report["backbone_fallback"])
                self.assertEqual(report["selected_adapter_type"], "read_state_suffix")
                self.assertLess(report["estimated_net_saving_bytes"], 0)
                self.assertTrue(any(torch.count_nonzero(p) for p in model.read_state_conditioning.parameters()))
                for name, value in before.items():
                    if not name.startswith(("transformer.layers.2.", "transformer.layers.3.", "transformer.norm.", "output_head.")):
                        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
                target, output = root / "trained.fqdc", root / "trained.fq"
                kwargs = dict(device=torch.device("cpu"), batch_reads=17, progress=False, verify_cdf=True)
                encode_fastq(source, target, checkpoint, head_adapter_path=self.artifact(root, checkpoint, adapter), **kwargs)
                decode_fastq(target, output, checkpoint, **kwargs)
                self.assertEqual(output.read_bytes(), source.read_bytes())

    def test_cli_requires_joint_adaptation(self):
        common = ["codec.encode", "in.fq", "out.fqdc", "base.pt", "--read-quality-state"]
        for flags in ([], ["--finetune-head"]):
            with mock.patch("sys.argv", common + flags), self.assertRaises(SystemExit):
                main()
        with mock.patch("sys.argv", common + ["--finetune-head", "--finetune-last-layers", "2"]), \
             mock.patch("codec.encode.encode_fastq") as encode, mock.patch("builtins.print"):
            encode.return_value.to_dict.return_value = {}
            main()
            self.assertTrue(encode.call_args.kwargs["head_adaptation_config"].read_quality_state)

    def test_end_to_end_fitting_export_and_real_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, checkpoint = model_and_checkpoint(root, 4)
            source, raw = source_file(root, count=80, compressed=True)
            cfg = config(max_reads=16, max_symbols=256, steps=2, max_seconds=0,
                head_type="residual", residual_dim=16, cross_layer=True, cross_dim=4,
                backbone_layers=2, backbone_steps=4, backbone_batch_reads=8,
                backbone_learning_rate=.01, backbone_eval_every=2,
                backbone_max_reads=32, backbone_max_symbols=256,
                backbone_lora_rank=2, read_quality_state=True)
            target, output, artifact = root / "fit.fqdc", root / "fit.fq", root / "fit.adapter.json"
            kwargs = dict(device=torch.device("cpu"), batch_reads=7, progress=False, verify_cdf=True)
            stats = encode_fastq(source, target, checkpoint, quantization_version=2,
                head_adaptation_config=cfg, save_head_adapter_path=artifact, **kwargs)
            self.assertEqual(json.loads(artifact.read_text())["adapter"]["version"], 9)
            self.assertTrue(stats.head_adaptation["backbone_accepted"])
            self.assertIsNotNone(stats.head_adaptation["backbone_after_validation_bits_per_quality"])
            self.assertEqual(stats.read_state_stream_bytes, 20)
            artifact.unlink()
            decode_fastq(target, output, checkpoint, **kwargs)
            self.assertEqual(output.read_bytes(), raw)


if __name__ == "__main__":
    unittest.main()
