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
from codec.encode import encode_fastq, main
from codec.fastq_stream import iter_fastq_batches
from codec.head_adapter import (
    HeadAdaptationConfig, adapt_output_head, apply_head_adapter,
    serialize_head, validate_head_adapter,
)
from codec.model import DirectQualityModelConfig, DirectQualityTransformer, fastq_batch_to_tensors
from codec.running_delta import RunningDeltaPriorConfig


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
                progress=False, verify_cdf=True, online_prior_config=None,
                head_adaptation_config=config(), save_head_adapter_path=artifact)
            self.assertTrue(stats.head_adaptation["accepted"])
            self.assertEqual(stats.quality_and_adapter_bytes,
                stats.range_stream_bytes + stats.head_adaptation["container_adapter_bytes"])
            metadata = read_container(pure).metadata
            self.assertEqual(metadata["format_version"], 3)
            self.assertIsNone(metadata["probability_profile"])
            for name, profile in (("repeat", None), ("mixed", RunningDeltaPriorConfig(cycle_bin_width=2))):
                target = root / (name + ".fqdc")
                replay = encode_fastq(source, target, ckpt, device=torch.device("cpu"), batch_reads=8,
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
                progress=False, online_prior_config=None, head_adaptation_config=config())
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
                    progress=False, online_prior_config=None, head_adaptation_config=config(min_gain_bits_per_quality=1000),
                    save_head_adapter_path=artifact)
                self.assertFalse(stats.head_adaptation["accepted"])
                self.assertEqual(read_container(container).metadata["format_version"], 1)
                decode_fastq(container, out, ckpt, device=torch.device("cpu"), batch_reads=8, progress=False)
                self.assertEqual(raw, out.read_bytes())
                repeated = root / f"repeat{n}.fqdc"
                encode_fastq(source, repeated, ckpt, device=torch.device("cpu"), batch_reads=8,
                             progress=False, online_prior_config=None, head_adapter_path=artifact)
                self.assertEqual(container.read_bytes(), repeated.read_bytes())

    def test_cli_modes_and_invalid_training_options(self):
        common = ["codec.encode", "in.fq", "out.fqdc", "base.pt"]
        for flags, expected in ((["--finetune-head"], None), (["--neural-only"], None),
                                (["--finetune-head", "--adaptive-weights"], RunningDeltaPriorConfig)):
            with mock.patch("sys.argv", common + flags), mock.patch("codec.encode.encode_fastq") as encode:
                encode.return_value.to_dict.return_value = {}
                with mock.patch("builtins.print"):
                    main()
                actual = encode.call_args.kwargs["online_prior_config"]
                if expected is None:
                    self.assertIsNone(actual)
                else:
                    self.assertIsInstance(actual, expected)
        with mock.patch("sys.argv", common + ["--head-steps", "8"]):
            with self.assertRaisesRegex(SystemExit, "require --finetune-head"):
                main()

    def test_invalid_config(self):
        for override in ({"steps": True}, {"max_seconds": float("nan")}, {"max_seconds": 0},
                         {"max_reads": 3}, {"learning_rate": 0}, {"anchor_strength": -1}):
            with self.assertRaises(ValueError):
                config(**override)


if __name__ == "__main__":
    unittest.main()
