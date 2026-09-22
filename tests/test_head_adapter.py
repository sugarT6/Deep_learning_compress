import base64
import copy
import gzip
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock
import zlib

import numpy as np
import torch

from codec.checkpoint import save_training_checkpoint
from codec.container import ContainerError, read_container, sha256_file
from codec.decode import decode_fastq
from codec.encode import encode_fastq, _encode_fastq_compat, main
from codec.fastq_stream import iter_fastq_batches
from codec.head_adapter import (
    HeadAdaptationConfig, adapt_output_head, apply_head_adapter,
    serialize_head, validate_head_adapter,
)
from codec.model import DirectQualityModelConfig, DirectQualityTransformer, ResidualOutputHead, fastq_batch_to_tensors
from codec.running_delta import RunningDeltaPriorConfig
from codec.quality_history_features import quality_history_features


def model_and_checkpoint(root):
    torch.manual_seed(27)
    model = DirectQualityTransformer(DirectQualityModelConfig(
        d_model=8, num_heads=2, num_layers=1, feedforward_dim=16,
        prev_q_embed_dim=4, qmer_embed_dim=2, base_embed_dim=3,
        base_conv_channels=3, base_context_dim=4, dropout=0.1))
    checkpoint = root / "base.pt"
    save_training_checkpoint(checkpoint, model=model, optimizer=None, epoch=1, global_step=1,
        data_split={}, sampler_config={}, sampler_statistics={},
        best_validation_bits_per_quality=1, validation_metrics={})
    return model, checkpoint


def source_file(root, count=65, compressed=False):
    rows = []
    for i in range(count):
        length = i % 11  # includes empty reads and variable lengths
        q = bytes([33 + (41 if i == 1 else 0 if i == 2 else 10)]) * length
        ending = b"\r\n" if i % 2 else b"\n"
        final = b"" if i == count - 1 and length else ending
        rows.append(b"@r" + str(i).encode() + ending + b"A" * length + ending
                    + b"+description" + ending + q + final)
    raw = b"".join(rows)
    path = root / ("input.fq.gz" if compressed else "input.fq")
    path.write_bytes(gzip.compress(raw) if compressed else raw)
    return path, raw


def config(**overrides):
    values = dict(max_reads=48, max_symbols=1000, steps=30, symbols_per_step=64,
                  learning_rate=0.03, max_seconds=60)
    values.update(overrides)
    return HeadAdaptationConfig(**values)


def rewrite_metadata(source, target, change, physical=None):
    raw = source.read_bytes()
    prefix = struct.Struct("<8sHHII")
    magic, version, flags, size, _ = prefix.unpack(raw[:prefix.size])
    meta = json.loads(raw[prefix.size:prefix.size + size])
    change(meta)
    payload = json.dumps(meta, sort_keys=True, separators=(",", ":")).encode()
    target.write_bytes(prefix.pack(magic, version if physical is None else physical, flags,
        len(payload), zlib.crc32(payload) & 0xffffffff) + payload + raw[prefix.size + size:])


class HeadAdapterTest(unittest.TestCase):
    def test_frozen_backbone_whole_read_split_and_determinism(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, ckpt = model_and_checkpoint(root)
            source, _ = source_file(root)
            original = {k: v.clone() for k, v in model.state_dict().items()}
            twin = copy.deepcopy(model)
            rng = torch.get_rng_state().clone()
            adapter, report = adapt_output_head(model, source, device=torch.device("cpu"),
                batch_reads=8, total=65536, base_sha256=sha256_file(ckpt), config=config())
            self.assertTrue(report["accepted"])
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertEqual(report["train_read_range"], [0, 36])
            self.assertEqual(report["validation_read_range"], [36, 48])
            self.assertEqual(report["prefix_symbols"], report["train_symbols"] + report["validation_symbols"])
            self.assertLess(report["after_validation_bits_per_quality"], report["before_validation_bits_per_quality"])
            for name, value in model.state_dict().items():
                if not name.startswith("output_head."):
                    self.assertTrue(torch.equal(original[name], value), name)
            self.assertFalse(torch.equal(original["output_head.weight"], model.output_head.weight))
            self.assertTrue(model.training)
            self.assertTrue(all(p.requires_grad and p.grad is None for p in model.parameters()))
            other, _ = adapt_output_head(twin, source, device=torch.device("cpu"), batch_reads=8,
                total=65536, base_sha256=sha256_file(ckpt), config=config())
            self.assertEqual(adapter, other)

    def test_features_match_logits_and_are_causal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, _ = model_and_checkpoint(root)
            model.eval()
            source, _ = source_file(root)
            tensors = fastq_batch_to_tensors(next(iter_fastq_batches(source, batch_reads=8)), "cpu")
            with torch.no_grad():
                features = model.forward_features(**tensors)
                reconstructed = model.output_head(features) * tensors["active_mask"].unsqueeze(-1)
                self.assertTrue(torch.equal(model.forward_full(**tensors), reconstructed))
                changed = {k: v.clone() for k, v in tensors.items()}
                changed["qualities"][:, 2:] = torch.where(changed["active_mask"][:, 2:], 5, 42)
                other = model.forward_features(**changed)
                self.assertTrue(torch.equal(features[:, :3], other[:, :3]))

    def test_fallback_leaves_entire_model_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, ckpt = model_and_checkpoint(root)
            source, _ = source_file(root)
            original = {k: v.clone() for k, v in model.state_dict().items()}
            for cfg, reason in ((config(min_gain_bits_per_quality=1000), "no_validation_gain"),
                                (config(max_seconds=1e-12), "insufficient_prefix"),
                                (config(max_symbols=1), "insufficient_prefix")):
                adapter, report = adapt_output_head(model, source, device=torch.device("cpu"), batch_reads=8,
                    total=65536, base_sha256=sha256_file(ckpt), config=cfg)
                self.assertIsNone(adapter)
                self.assertEqual(report["reason"], reason)
                for name, value in model.state_dict().items():
                    self.assertTrue(torch.equal(value, original[name]))

    def test_adapter_validation_corruption_and_finite_values(self):
        with tempfile.TemporaryDirectory() as directory:
            model, ckpt = model_and_checkpoint(Path(directory))
            digest = sha256_file(ckpt)
            adapter = serialize_head(model.output_head, digest)
            for name, value in (("version", True), ("dtype", "float16"), ("weight_shape", [42, 9]),
                                ("parameter_bytes", 1), ("base_checkpoint_sha256", "0" * 64),
                                ("sha256", "0" * 64), ("data_base64", "?" * len(adapter["data_base64"]))):
                bad = dict(adapter, **{name: value})
                with self.assertRaises(ValueError, msg=name):
                    validate_head_adapter(bad, 8, digest)
            bad = dict(adapter, extra=1)
            with self.assertRaises(ValueError):
                validate_head_adapter(bad, 8, digest)
            raw = bytearray(base64.b64decode(adapter["data_base64"]))
            raw[:4] = struct.pack("<f", float("nan"))
            bad = dict(adapter, sha256=hashlib.sha256(raw).hexdigest(),
                       data_base64=base64.b64encode(raw).decode())
            with self.assertRaisesRegex(ValueError, "finite"):
                validate_head_adapter(bad, 8, digest)

    def test_adapted_roundtrip_both_modes_and_reused_head(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, ckpt = model_and_checkpoint(root)
            source, raw = source_file(root, compressed=True)
            digest = sha256_file(ckpt)
            artifact = root / "adapter.json"
            pure = root / "pure.fqdc"
            stats = encode_fastq(source, pure, ckpt, device=torch.device("cpu"), batch_reads=8,
                progress=False, verify_cdf=True,
                head_adaptation_config=config(), save_head_adapter_path=artifact)
            self.assertTrue(stats.head_adaptation["accepted"])
            self.assertEqual(stats.quality_and_adapter_bytes,
                stats.range_stream_bytes + stats.head_adaptation["container_adapter_bytes"])
            metadata = read_container(pure).metadata
            self.assertEqual(metadata["format_version"], 3)
            self.assertIsNone(metadata["probability_profile"])
            for name, profile in (("repeat", None), ("mixed", RunningDeltaPriorConfig(cycle_bin_width=2))):
                target = root / (name + ".fqdc")
                replay = _encode_fastq_compat(source, target, ckpt, device=torch.device("cpu"), batch_reads=8,
                    progress=False, verify_cdf=True, online_prior_config=profile, head_adapter_path=artifact)
                self.assertEqual(metadata["head_adapter"], read_container(target).metadata["head_adapter"])
                if profile is None:
                    self.assertEqual(pure.read_bytes(), target.read_bytes())
                else:
                    self.assertEqual(replay.online_adaptation["weight_updates"], 8)
                output = root / (name + ".fq")
                with mock.patch("codec.head_adapter.adapt_output_head", side_effect=AssertionError("decoder trained")):
                    decode_fastq(target, output, ckpt, device=torch.device("cpu"), batch_reads=8,
                                 progress=False, verify_cdf=True)
                self.assertEqual(output.read_bytes(), raw)
            self.assertEqual(digest, sha256_file(ckpt))
            with self.assertRaises(FileExistsError):
                encode_fastq(source, root / "overwrite.fqdc", ckpt, device=torch.device("cpu"),
                    head_adaptation_config=config(), save_head_adapter_path=artifact, progress=False)

    def test_container_rejects_missing_or_downgraded_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, ckpt = model_and_checkpoint(root)
            source, _ = source_file(root)
            container = root / "valid.fqdc"
            encode_fastq(source, container, ckpt, device=torch.device("cpu"), batch_reads=8,
                progress=False, head_adaptation_config=config())
            changes = [lambda m: m.pop("head_adapter"), lambda m: m.pop("probability_profile"),
                       lambda m: m["head_adapter"].update(sha256="0" * 64)]
            for i, mutate in enumerate(changes):
                bad = root / f"bad{i}.fqdc"
                rewrite_metadata(container, bad, mutate)
                with self.assertRaises(ContainerError):
                    read_container(bad)
            bad = root / "downgraded.fqdc"
            rewrite_metadata(container, bad, lambda m: m.update(format_version=1), physical=1)
            with self.assertRaisesRegex(ContainerError, "version 3"):
                read_container(bad)

    def test_small_input_and_rejection_keep_legacy_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, ckpt = model_and_checkpoint(root)
            for n in (0, 1, 3, 65):
                source, raw = source_file(root, n)
                container, out = root / f"{n}.fqdc", root / f"{n}.fq"
                artifact = root / f"{n}.json"
                stats = encode_fastq(source, container, ckpt, device=torch.device("cpu"), batch_reads=8,
                    progress=False, head_adaptation_config=config(min_gain_bits_per_quality=1000),
                    save_head_adapter_path=artifact)
                self.assertFalse(stats.head_adaptation["accepted"])
                self.assertEqual(read_container(container).metadata["format_version"], 1)
                decode_fastq(container, out, ckpt, device=torch.device("cpu"), batch_reads=8, progress=False)
                self.assertEqual(raw, out.read_bytes())
                repeated = root / f"repeat{n}.fqdc"
                encode_fastq(source, repeated, ckpt, device=torch.device("cpu"), batch_reads=8,
                             progress=False, head_adapter_path=artifact)
                self.assertEqual(container.read_bytes(), repeated.read_bytes())

    def test_cli_modes_and_invalid_training_options(self):
        common = ["codec.encode", "in.fq", "out.fqdc", "base.pt"]
        for flags in ([], ["--finetune-head"], ["--neural-only"], ["--head-adapter", "head.json"]):
            with mock.patch("sys.argv", common + flags), mock.patch("codec.encode.encode_fastq") as encode:
                encode.return_value.to_dict.return_value = {}
                with mock.patch("builtins.print"):
                    main()
                self.assertNotIn("online_prior_config", encode.call_args.kwargs)
        with mock.patch("sys.argv", common + ["--head-steps", "8"]):
            with self.assertRaisesRegex(SystemExit, "require --finetune-head"):
                main()

    def test_production_is_neural_only_and_bit_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, ckpt = model_and_checkpoint(root)
            source, raw = source_file(root, compressed=True)
            for adapted in (False, True):
                with self.subTest(adapted=adapted):
                    target = root / f"production{adapted}.fqdc"
                    reference = root / f"reference{adapted}.fqdc"
                    kwargs = dict(device=torch.device("cpu"), batch_reads=8, progress=False,
                                  verify_cdf=True, head_adaptation_config=config() if adapted else None)
                    with mock.patch("codec.encode.make_prior_state", side_effect=AssertionError("expert created")), \
                         mock.patch("codec.encode.fuse_profile_positions", side_effect=AssertionError("expert fused")), \
                         mock.patch("codec.encode.fuse_batch_logits", side_effect=AssertionError("expert fused")):
                        stats = encode_fastq(source, target, ckpt, **kwargs)
                    self.assertIsNone(stats.online_adaptation)
                    self.assertIsNone(read_container(target).metadata.get("probability_profile"))
                    _encode_fastq_compat(source, reference, ckpt, online_prior_config=None, **kwargs)
                    self.assertEqual(target.read_bytes(), reference.read_bytes())
                    restored = root / f"restored{adapted}.fq"
                    decode_fastq(target, restored, ckpt, device=torch.device("cpu"), batch_reads=8,
                                 progress=False, verify_cdf=True)
                    self.assertEqual(restored.read_bytes(), raw)
            with self.assertRaisesRegex(TypeError, "online_prior_config"):
                encode_fastq(source, root / "forbidden.fqdc", ckpt, device=torch.device("cpu"),
                             online_prior_config=RunningDeltaPriorConfig())

    def test_invalid_config(self):
        for override in ({"steps": True}, {"max_seconds": float("nan")}, {"max_seconds": 0},
                         {"max_reads": 3}, {"learning_rate": 0}, {"anchor_strength": -1},
                         {"head_type": "unknown"}, {"residual_dim": 0}, {"residual_dim": 129},
                         {"residual_dim": True}, {"history_features": True},
                         {"head_type": "residual", "history_features": 1}):
            with self.assertRaises(ValueError):
                config(**override)

    def test_residual_initial_function_wire_parameters_and_linear_restore(self):
        with tempfile.TemporaryDirectory() as directory:
            model, ckpt = model_and_checkpoint(Path(directory))
            digest = sha256_file(ckpt)
            base = serialize_head(model.output_head, digest)
            head = ResidualOutputHead(8, 32)
            with torch.no_grad():
                head.weight.copy_(model.output_head.weight)
                head.bias.copy_(model.output_head.bias)
            hidden = torch.randn(3, 5, 8)
            self.assertTrue(torch.equal(head(hidden), model.output_head(hidden)))
            with torch.no_grad():
                head.up.weight.normal_()
                head.up.bias.normal_()
            adapter = serialize_head(head, digest)
            self.assertEqual(adapter["version"], 2)
            self.assertEqual(adapter["parameter_bytes"], sum(p.numel() for p in head.parameters()) * 4)
            self.assertEqual(serialize_head(ResidualOutputHead(256, 32), digest)["parameter_bytes"], 81616)
            rng = torch.get_rng_state().clone()
            apply_head_adapter(model, adapter, digest)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertTrue(torch.equal(head(hidden), model.output_head(hidden)))
            self.assertEqual(adapter, serialize_head(model.output_head, digest))
            apply_head_adapter(model, base, digest)
            self.assertNotIsInstance(model.output_head, ResidualOutputHead)
            self.assertEqual(base, serialize_head(model.output_head, digest))

    def test_residual_fit_frozen_backbone_rng_and_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, ckpt = model_and_checkpoint(root)
            source, _ = source_file(root)
            original = copy.deepcopy(model)
            twin = copy.deepcopy(model)
            rng = torch.get_rng_state().clone()
            kwargs = dict(device=torch.device("cpu"), batch_reads=8, total=65536,
                          base_sha256=sha256_file(ckpt), config=config(head_type="residual"))
            adapter, report = adapt_output_head(model, source, **kwargs)
            self.assertTrue(report["accepted"])
            self.assertEqual(report["steps_completed"], 30)
            self.assertEqual(report["train_read_range"], [0, 36])
            self.assertEqual(report["validation_read_range"], [36, 48])
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertTrue(model.training)
            self.assertTrue(all(p.requires_grad and p.grad is None for p in model.parameters()))
            self.assertGreater(int(torch.count_nonzero(model.output_head.up.weight)), 0)
            for name, value in original.state_dict().items():
                if not name.startswith("output_head."):
                    self.assertTrue(torch.equal(value, model.state_dict()[name]), name)
            other, _ = adapt_output_head(twin, source, **kwargs)
            self.assertEqual(adapter, other)
            for override in ({"min_gain_bits_per_quality": 1000}, {"max_seconds": 1e-12}):
                fallback = copy.deepcopy(original)
                fallback.requires_grad_(False)
                kwargs["config"] = config(head_type="residual", **override)
                rejected, report = adapt_output_head(fallback, source, **kwargs)
                self.assertIsNone(rejected)
                self.assertNotIsInstance(fallback.output_head, ResidualOutputHead)
                self.assertTrue(all(not p.requires_grad for p in fallback.parameters()))
                for name, value in original.state_dict().items():
                    self.assertTrue(torch.equal(value, fallback.state_dict()[name]), name)

    def test_residual_roundtrip_reuse_and_no_external_artifact_at_decode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, ckpt = model_and_checkpoint(root)
            digest = sha256_file(ckpt)
            source, raw = source_file(root, compressed=True)
            artifact, target = root / "adapter.json", root / "residual.fqdc"
            kwargs = dict(device=torch.device("cpu"), batch_reads=8, progress=False, verify_cdf=True)
            stats = encode_fastq(source, target, ckpt, **kwargs,
                                head_adaptation_config=config(head_type="residual"), save_head_adapter_path=artifact)
            self.assertTrue(stats.head_adaptation["accepted"])
            meta = read_container(target).metadata
            self.assertEqual(meta["format_version"], 3)
            self.assertEqual(meta["head_adapter"]["version"], 2)
            self.assertEqual(meta["head_adapter"]["residual_dim"], 32)
            self.assertIsNone(meta["probability_profile"])
            self.assertEqual(stats.quality_and_adapter_bytes,
                             stats.range_stream_bytes + stats.head_adaptation["container_adapter_bytes"])
            repeated = root / "repeat.fqdc"
            encode_fastq(source, repeated, ckpt, **kwargs, head_adapter_path=artifact)
            self.assertEqual(target.read_bytes(), repeated.read_bytes())
            artifact.unlink()  # This test-owned export is deliberately absent at decode.
            with mock.patch("codec.head_adapter.adapt_output_head", side_effect=AssertionError("decoder trained")):
                restored = root / "restored.fq"
                decode_fastq(target, restored, ckpt, **kwargs)
            self.assertEqual(restored.read_bytes(), raw)
            self.assertEqual(digest, sha256_file(ckpt))

    def test_residual_metadata_rejects_invalid_protocol_and_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, ckpt = model_and_checkpoint(root)
            source, _ = source_file(root)
            target = root / "valid.fqdc"
            encode_fastq(source, target, ckpt, device=torch.device("cpu"), batch_reads=8,
                         progress=False, head_adaptation_config=config(head_type="residual"))
            changes = [lambda a: a.update(version=1), lambda a: a.update(version=True),
                       lambda a: a.update(residual_dim=129), lambda a: a.update(residual_dim=True),
                       lambda a: a.update(activation="gelu_tanh"), lambda a: a.pop("up_bias_shape"),
                       lambda a: a.update(down_weight_shape=[32, 9]), lambda a: a.update(up_bias_shape=[True]),
                       lambda a: a.update(layout="weight_row_major_then_bias"),
                       lambda a: a.update(parameter_bytes=a["parameter_bytes"] - 4),
                       lambda a: a.update(sha256="0" * 64)]
            for i, change in enumerate(changes):
                bad = root / f"bad{i}.fqdc"
                rewrite_metadata(target, bad, lambda m: change(m["head_adapter"]))
                with self.subTest(change=i), self.assertRaises(ContainerError):
                    read_container(bad)
            adapter = copy.deepcopy(read_container(target).metadata["head_adapter"])
            data = bytearray(base64.b64decode(adapter["data_base64"]))
            data[-4:] = struct.pack("<f", float("nan"))
            adapter.update(data_base64=base64.b64encode(data).decode(), sha256=hashlib.sha256(data).hexdigest())
            with self.assertRaisesRegex(ValueError, "finite"):
                validate_head_adapter(adapter, 8, sha256_file(ckpt))

    def test_residual_cli_is_opt_in_and_requires_training(self):
        common = ["codec.encode", "in.fq", "out.fqdc", "base.pt"]
        with mock.patch("sys.argv", common + ["--finetune-head", "--head-type", "residual", "--head-residual-dim", "16"]), \
             mock.patch("codec.encode.encode_fastq") as encode, mock.patch("builtins.print"):
            encode.return_value.to_dict.return_value = {}
            main()
            cfg = encode.call_args.kwargs["head_adaptation_config"]
            self.assertEqual((cfg.head_type, cfg.residual_dim), ("residual", 16))
        for flags in (["--head-type", "residual"], ["--head-adapter", "head.json", "--head-type", "residual"],
                      ["--finetune-head", "--head-residual-dim", "16"]):
            with mock.patch("sys.argv", common + flags), self.assertRaises(SystemExit):
                main()

    def test_history_head_initialization_and_wire_restore(self):
        with tempfile.TemporaryDirectory() as directory:
            model, ckpt = model_and_checkpoint(Path(directory))
            torch.manual_seed(17)
            baseline = ResidualOutputHead(8, 32)
            torch.manual_seed(17)
            head = ResidualOutputHead(8, 32, history_features=True)
            for name in ("weight", "bias"):
                self.assertTrue(torch.equal(getattr(baseline, name), getattr(head, name)))
            self.assertTrue(torch.equal(baseline.down.weight, head.down.weight[:, :8]))
            self.assertTrue(torch.equal(baseline.down.bias, head.down.bias))
            self.assertEqual(int(torch.count_nonzero(head.down.weight[:, 8:])), 0)
            h = torch.randn(4, 7, 8)
            packed = torch.cat((h, torch.randn(4, 7, 8)), dim=-1)
            self.assertTrue(torch.equal(baseline(h), head(packed)))
            with torch.no_grad():
                head.down.weight[:, 8:].normal_()
                head.up.weight.normal_()
            digest = sha256_file(ckpt)
            adapter = serialize_head(head, digest)
            self.assertEqual(adapter["version"], 3)
            self.assertEqual(adapter["down_weight_shape"], [32, 16])
            apply_head_adapter(model, adapter, digest)
            self.assertTrue(torch.equal(head(packed), model.output_head(packed)))
            self.assertEqual(adapter, serialize_head(model.output_head, digest))
            self.assertEqual(serialize_head(ResidualOutputHead(256, 32, True), digest)["parameter_bytes"], 82640)

    def test_history_fit_freezes_backbone_and_preserves_rng(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, ckpt = model_and_checkpoint(root)
            source, _ = source_file(root)
            original, twin = copy.deepcopy(model), copy.deepcopy(model)
            rng = torch.get_rng_state().clone()
            kwargs = dict(device=torch.device("cpu"), batch_reads=8, total=65536,
                          base_sha256=sha256_file(ckpt), config=config(head_type="residual", history_features=True))
            adapter, report = adapt_output_head(model, source, **kwargs)
            self.assertTrue(report["accepted"])
            self.assertEqual(report["steps_completed"], 30)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertTrue(all(p.requires_grad and p.grad is None for p in model.parameters()))
            self.assertGreater(int(torch.count_nonzero(model.output_head.down.weight[:, 8:])), 0)
            for name, value in original.state_dict().items():
                if not name.startswith("output_head."):
                    self.assertTrue(torch.equal(value, model.state_dict()[name]), name)
            other, _ = adapt_output_head(twin, source, **kwargs)
            self.assertEqual(adapter, other)
            model.eval()
            tensors = fastq_batch_to_tensors(next(iter_fastq_batches(source, batch_reads=8)), "cpu")
            with torch.no_grad():
                h = model.forward_features(**tensors)
                s = quality_history_features(tensors["qualities"], tensors["active_mask"])
                expected = model.output_head(torch.cat((h, s), -1)) * tensors["active_mask"].unsqueeze(-1)
                self.assertTrue(torch.equal(model.forward_full(**tensors), expected))

    def test_history_roundtrip_artifact_reuse_and_schema_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, ckpt = model_and_checkpoint(root)
            digest = sha256_file(ckpt)
            source, raw = source_file(root, compressed=True)
            target, artifact = root / "history.fqdc", root / "head.json"
            kwargs = dict(device=torch.device("cpu"), batch_reads=8, progress=False, verify_cdf=True)
            stats = encode_fastq(source, target, ckpt, **kwargs,
                                head_adaptation_config=config(head_type="residual", history_features=True),
                                save_head_adapter_path=artifact)
            self.assertTrue(stats.head_adaptation["accepted"])
            self.assertEqual(read_container(target).metadata["head_adapter"]["version"], 3)
            repeated = root / "repeat.fqdc"
            encode_fastq(source, repeated, ckpt, **kwargs, head_adapter_path=artifact)
            self.assertEqual(target.read_bytes(), repeated.read_bytes())
            artifact.unlink()
            restored = root / "out.fq"
            decode_fastq(target, restored, ckpt, **kwargs)
            self.assertEqual(raw, restored.read_bytes())
            self.assertEqual(digest, sha256_file(ckpt))
            changes = [lambda a: a.pop("history_features"), lambda a: a.update(version=2),
                       lambda a: a["history_features"].update(window=16),
                       lambda a: a["history_features"].update(version=True),
                       lambda a: a["history_features"]["features"].reverse(),
                       lambda a: a.update(down_weight_shape=[32, 8])]
            for i, change in enumerate(changes):
                bad = root / f"bad{i}.fqdc"
                rewrite_metadata(target, bad, lambda m: change(m["head_adapter"]))
                with self.subTest(change=i), self.assertRaises(ContainerError):
                    read_container(bad)

    def test_history_cli_requires_residual_training(self):
        common = ["codec.encode", "in.fq", "out.fqdc", "base.pt"]
        with mock.patch("sys.argv", common + ["--finetune-head", "--head-type", "residual", "--head-history-features"]), \
             mock.patch("codec.encode.encode_fastq") as encode, mock.patch("builtins.print"):
            encode.return_value.to_dict.return_value = {}
            main()
            self.assertTrue(encode.call_args.kwargs["head_adaptation_config"].history_features)
        for flags in (["--head-history-features"], ["--finetune-head", "--head-history-features"],
                      ["--head-adapter", "head.json", "--head-history-features"]):
            with mock.patch("sys.argv", common + flags), self.assertRaises(SystemExit):
                main()


if __name__ == "__main__":
    unittest.main()
