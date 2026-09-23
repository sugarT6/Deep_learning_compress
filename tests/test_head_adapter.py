import base64
import copy
import gzip
import hashlib
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
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
from codec.model import CrossLayerOutputHead, DirectQualityModelConfig, DirectQualityTransformer, ResidualOutputHead, fastq_batch_to_tensors
from codec.running_delta import RunningDeltaPriorConfig


def model_and_checkpoint(root, num_layers=1):
    torch.manual_seed(27)
    model = DirectQualityTransformer(DirectQualityModelConfig(
        d_model=8, num_heads=2, num_layers=num_layers, feedforward_dim=16,
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
    def test_sequence_windows_causality_and_seeded_zero_start(self):
        from codec.model import CausalSequenceOutputHead, causal_quality_windows
        from codec.head_adapter import _new_head
        q = torch.arange(12).reshape(1, 12)
        windows = causal_quality_windows(q)
        self.assertEqual(windows[0, 0].tolist(), [42] * 8)
        self.assertEqual(windows[0, 9].tolist(), list(range(1, 9)))
        altered = q.clone()
        altered[:, 5:] = 41
        self.assertTrue(torch.equal(windows[:, :6], causal_quality_windows(altered)[:, :6]))
        self.assertEqual(tuple(causal_quality_windows(q[:, :0]).shape), (1, 0, 8))
        kw = dict(device="cpu", dtype=torch.float32, seed=123, cross_dim=16)
        baseline = _new_head(8, 64, **kw)
        head = _new_head(8, 64, sequence_branch=True, **kw)
        self.assertIsInstance(head, CausalSequenceOutputHead)
        for name, value in baseline.state_dict().items():
            self.assertTrue(torch.equal(value, head.state_dict()[name]))
        h = torch.randn(1, 12, 16)
        self.assertTrue(torch.equal(baseline(h), head.forward_with_quality(h, q, None)))
        with torch.no_grad():
            head.sequence_up.weight.normal_()
        left = head.forward_with_quality(h, q, None)
        right = head.forward_with_quality(h, altered, None)
        self.assertTrue(torch.equal(left[:, :6], right[:, :6]))
        self.assertFalse(torch.equal(left[:, 6:], right[:, 6:]))

    def test_sequence_fit_roundtrip_serialization_and_frozen_backbone(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, ckpt = model_and_checkpoint(root, num_layers=4)
            original = copy.deepcopy(model)
            source, raw = source_file(root)
            cfg = config(head_type="residual", cross_layer=True, sequence_branch=True)
            kw = dict(device=torch.device("cpu"), batch_reads=8, total=65536,
                      base_sha256=sha256_file(ckpt), config=cfg)
            rng = torch.get_rng_state().clone()
            adapter, report = adapt_output_head(model, source, **kw)
            self.assertTrue(report["accepted"])
            self.assertEqual(adapter["version"], 5)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertGreater(model.output_head.sequence_up.weight.abs().sum().item(), 0)
            for name, value in original.state_dict().items():
                if not name.startswith("output_head."):
                    self.assertTrue(torch.equal(value, model.state_dict()[name]), name)
            twin, _ = adapt_output_head(copy.deepcopy(original), source, **kw)
            self.assertEqual(adapter, twin)
            for change in ({"sequence_protocol": "future_q"}, {"version": 4},
                           {"parameter_bytes": adapter["parameter_bytes"] - 4}):
                with self.assertRaises(ValueError):
                    validate_head_adapter(dict(adapter, **change), 8, sha256_file(ckpt))
            artifact = root / "sequence.json"
            artifact.write_text(json.dumps(dict(format="direct-quality-head-adaptation-artifact",
                version=1, base_checkpoint_sha256=sha256_file(ckpt), adapter=adapter, report={})))
            target = root / "sequence.fqdc"
            io_kw = dict(device=torch.device("cpu"), batch_reads=8, progress=False, verify_cdf=True)
            encode_fastq(source, target, ckpt, **io_kw, head_adapter_path=artifact)
            self.assertEqual(read_container(target).metadata["head_adapter"]["version"], 5)
            artifact.unlink()
            restored = root / "restored.fq"
            decode_fastq(target, restored, ckpt, **io_kw)
            self.assertEqual(raw, restored.read_bytes())

    def test_sequence_cli_and_config_guards(self):
        for overrides in (dict(sequence_branch=True), dict(sequence_branch=1),
                          dict(sequence_branch=True, cross_layer=True, head_type="linear")):
            with self.assertRaises(ValueError):
                config(**overrides)
        common = ["codec.encode", "in.fq", "out.fqdc", "base.pt"]
        with mock.patch("sys.argv", common + ["--head-sequence-branch"]), self.assertRaises(SystemExit):
            main()
        flags = ["--finetune-head", "--head-type", "residual", "--head-cross-layer", "--head-sequence-branch"]
        with mock.patch("sys.argv", common + flags), mock.patch("codec.encode.encode_fastq") as encode, mock.patch("builtins.print"):
            encode.return_value.to_dict.return_value = {}
            main()
            self.assertTrue(encode.call_args.kwargs["head_adaptation_config"].sequence_branch)

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

    def test_cross_layer_features_causal_exact_and_hooks_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, _ = model_and_checkpoint(root, num_layers=4)
            model.eval()
            source, _ = source_file(root)
            tensors = fastq_batch_to_tensors(next(iter_fastq_batches(source, batch_reads=8)), "cpu")
            with torch.no_grad():
                base = model.forward_features(**tensors)
                packed = model.forward_cross_layer_features(**tensors)
                self.assertTrue(torch.equal(base, packed[..., :8]))
                self.assertEqual(packed.shape[-1], 16)
                changed = {k: v.clone() for k, v in tensors.items()}
                changed["qualities"][:, 3:] = torch.where(changed["active_mask"][:, 3:], 5, 42)
                other = model.forward_cross_layer_features(**changed)
                self.assertTrue(torch.equal(packed[:, :4], other[:, :4]))
                empty = {k: (v[:, :0] if v.ndim == 2 else v * 0) for k, v in tensors.items()}
                self.assertEqual(model.forward_cross_layer_features(**empty).shape, (8, 0, 16))
            self.assertFalse(model.transformer.layers[2]._forward_hooks)
            with mock.patch.object(model, "_hidden_impl", side_effect=RuntimeError("fixture")):
                with self.assertRaises(RuntimeError):
                    model.forward_cross_layer_features(**tensors)
            self.assertFalse(model.transformer.layers[2]._forward_hooks)

    def test_cross_layer_fit_roundtrip_and_exact_wire(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, ckpt = model_and_checkpoint(root, num_layers=4)
            source, raw = source_file(root, compressed=True)
            original = copy.deepcopy(model)
            twin = copy.deepcopy(model)
            rng = torch.get_rng_state().clone()
            cfg = config(head_type="residual", residual_dim=64, cross_layer=True)
            kwargs = dict(device=torch.device("cpu"), batch_reads=8, total=65536, base_sha256=sha256_file(ckpt), config=cfg)
            adapter, report = adapt_output_head(model, source, **kwargs)
            self.assertTrue(report["accepted"])
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertGreater(int(torch.count_nonzero(model.output_head.cross_up.weight)), 0)
            self.assertTrue(all(p.requires_grad and p.grad is None for p in model.parameters()))
            for name, value in original.state_dict().items():
                if not name.startswith("output_head."):
                    self.assertTrue(torch.equal(value, model.state_dict()[name]))
            second, _ = adapt_output_head(twin, source, **kwargs)
            self.assertEqual(adapter, second)
            self.assertEqual(adapter, serialize_head(model.output_head, sha256_file(ckpt)))
            artifact, target = root / "head.json", root / "cross.fqdc"
            common = dict(device=torch.device("cpu"), batch_reads=8, progress=False, verify_cdf=True)
            encode_fastq(source, target, ckpt, **common, head_adaptation_config=cfg, save_head_adapter_path=artifact)
            self.assertEqual(read_container(target).metadata["head_adapter"]["version"], 4)
            repeat = root / "repeat.fqdc"
            encode_fastq(source, repeat, ckpt, **common, head_adapter_path=artifact)
            self.assertEqual(target.read_bytes(), repeat.read_bytes())
            artifact.unlink()
            restored = root / "out.fq"
            decode_fastq(target, restored, ckpt, **common)
            self.assertEqual(raw, restored.read_bytes())
            for i, change in enumerate((lambda a: a.update(source_layer=2), lambda a: a.update(source_layer=True),
                                       lambda a: a.update(source_eps=1e-6), lambda a: a.update(source_normalization="none"),
                                       lambda a: a.update(cross_dim=0), lambda a: a.pop("cross_up_bias_shape"),
                                       lambda a: a.update(cross_down_weight_shape=[16, 9]), lambda a: a.update(version=2))):
                bad = root / f"bad{i}.fqdc"
                rewrite_metadata(target, bad, lambda m: change(m["head_adapter"]))
                with self.assertRaises(ContainerError):
                    read_container(bad)
            kwargs["config"] = config(head_type="residual", cross_layer=True, min_gain_bits_per_quality=1000)
            rejected, _ = adapt_output_head(original, source, **kwargs)
            self.assertIsNone(rejected)
            self.assertNotIsInstance(original.output_head, CrossLayerOutputHead)

    def test_cross_layer_initial_function_and_architecture_guard(self):
        torch.manual_seed(13)
        base = ResidualOutputHead(256, 64)
        torch.manual_seed(13)
        cross = CrossLayerOutputHead(256, 64, 16)
        for name, value in base.state_dict().items():
            self.assertTrue(torch.equal(value, cross.state_dict()[name]))
        h = torch.randn(2, 7, 256)
        self.assertTrue(torch.allclose(base(h), cross(torch.cat((h, torch.randn_like(h)), -1)), atol=1e-7))
        self.assertEqual(serialize_head(cross, "0" * 64)["parameter_bytes"], 139192)
        with tempfile.TemporaryDirectory() as directory:
            model, ckpt = model_and_checkpoint(Path(directory))
            adapter = serialize_head(CrossLayerOutputHead(8, 64, 16), sha256_file(ckpt))
            with self.assertRaisesRegex(ValueError, "at least 4"):
                apply_head_adapter(model, adapter, sha256_file(ckpt))

    def test_cross_layer_cli_and_config_guards(self):
        common = ["codec.encode", "in.fq", "out.fqdc", "base.pt"]
        with mock.patch("sys.argv", common + ["--finetune-head", "--head-type", "residual", "--head-cross-layer"]), \
             mock.patch("codec.encode.encode_fastq") as encode, mock.patch("builtins.print"):
            encode.return_value.to_dict.return_value = {}
            main()
            self.assertTrue(encode.call_args.kwargs["head_adaptation_config"].cross_layer)
        for flags in (["--head-cross-layer"], ["--finetune-head", "--head-cross-layer"],
                      ["--finetune-head", "--head-cross-dim", "32"]):
            with mock.patch("sys.argv", common + flags), self.assertRaises(SystemExit):
                main()
        for values in ({"cross_layer": True}, {"cross_dim": 0}, {"cross_dim": True}):
            with self.assertRaises(ValueError):
                config(**values)

    def test_invalid_config(self):
        for override in ({"steps": True}, {"max_seconds": float("nan")}, {"max_seconds": 0},
                         {"max_reads": 3}, {"learning_rate": 0}, {"anchor_strength": -1},
                         {"head_type": "unknown"}, {"residual_dim": 0}, {"residual_dim": 129},
                         {"residual_dim": True}):
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

    def test_retired_history_adapter_remains_decode_only(self):
        # Build an old-format fixture independently of the production serializer.
        from codec._legacy_quality_history import LegacyHistoryHead, history_feature_schema
        from codec.encode import build_parser
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, ckpt = model_and_checkpoint(root)
            source, raw = source_file(root)
            digest = sha256_file(ckpt)
            head = LegacyHistoryHead(8, 32)
            with torch.no_grad():
                head.up.weight.normal_(std=0.1)
            parameters = (head.weight, head.bias, head.down.weight, head.down.bias, head.up.weight, head.up.bias)
            data = b"".join(p.detach().numpy().astype("<f4").tobytes() for p in parameters)
            adapter = dict(format="direct-quality-output-head", version=3, dtype="little_endian_float32",
                           application="replace_output_head", layout="weight_bias_down_weight_down_bias_up_weight_up_bias",
                           base_checkpoint_sha256=digest, residual_dim=32, activation="gelu_exact",
                           history_features=history_feature_schema(), parameter_bytes=len(data),
                           sha256=hashlib.sha256(data).hexdigest(), data_base64=base64.b64encode(data).decode())
            for name, parameter in zip(("weight_shape", "bias_shape", "down_weight_shape", "down_bias_shape",
                                        "up_weight_shape", "up_bias_shape"), parameters):
                adapter[name] = list(parameter.shape)
            apply_head_adapter(model, adapter, digest)
            with self.assertRaisesRegex(ValueError, "decode-only"):
                serialize_head(model.output_head, digest)
            with self.assertRaisesRegex(ValueError, "cannot be fine-tuned"):
                adapt_output_head(model, source, device=torch.device("cpu"), batch_reads=8,
                                  total=65536, base_sha256=digest, config=config())
            # Test-only injection writes arithmetic bytes with the legacy model;
            # no production flag or artifact can select this head for encoding.
            payload, container = root / "payload.fqdc", root / "legacy.fqdc"
            kwargs = dict(device=torch.device("cpu"), batch_reads=8, progress=False, verify_cdf=True)
            with mock.patch("codec.encode.load_training_checkpoint", return_value=SimpleNamespace(model=model)):
                encode_fastq(source, payload, ckpt, **kwargs)
            rewrite_metadata(payload, container,
                             lambda m: m.update(format_version=3, head_adapter=adapter, probability_profile=None), physical=3)
            restored = root / "out.fq"
            decode_fastq(container, restored, ckpt, **kwargs)
            self.assertEqual(restored.read_bytes(), raw)
            artifact = root / "legacy.json"
            artifact.write_text(json.dumps(dict(format="direct-quality-head-adaptation-artifact", version=1,
                                               base_checkpoint_sha256=digest, adapter=adapter, report={})))
            with self.assertRaisesRegex(ValueError, "retired"):
                encode_fastq(source, root / "forbidden.fqdc", ckpt, **kwargs, head_adapter_path=artifact)
            self.assertFalse((root / "forbidden.fqdc").exists())
            for i, change in enumerate((lambda a: a.pop("history_features"),
                                       lambda a: a["history_features"].update(window=16),
                                       lambda a: a["history_features"].update(version=True),
                                       lambda a: a.update(down_weight_shape=[32, 8]))):
                bad = root / f"bad{i}.fqdc"
                rewrite_metadata(container, bad, lambda m: change(m["head_adapter"]))
                with self.assertRaises(ContainerError):
                    read_container(bad)
        with mock.patch("sys.stderr"), self.assertRaises(SystemExit):
            build_parser().parse_args(["a.fq", "b.fqdc", "base.pt", "--head-history-features"])
        with self.assertRaises(TypeError):
            config(head_type="residual", history_features=True)


if __name__ == "__main__":
    unittest.main()
