import json
import copy
import struct
import tempfile
import unittest
import zlib
from pathlib import Path

import torch

from codec.block_features import encode_block_features
from codec.checkpoint import save_training_checkpoint
from codec.container import (
    BLOCK_FEATURE_CONTAINER_VERSION,
    ContainerError,
    ContainerIntegrityError,
    block_feature_metadata_bytes,
    metadata_payload,
    read_container,
)
from codec.decode import decode_fastq
from codec.encode import _encode_fastq_compat, encode_fastq
from codec.fastq_stream import iter_fastq_batches
from codec.head_adapter import HeadAdaptationConfig
from codec.model import DirectQualityModelConfig, DirectQualityTransformer, feature_schema
from codec.online_prior import DEFAULT_ONLINE_PRIOR_CONFIG


_PREFIX = struct.Struct("<8sHHII")


def fixture_bytes(count):
    records = []
    for index in range(count):
        length = index % 5
        sequence = (b"ACGTN" * 2)[:length]
        quality = bytes(33 + (index * 7 + cycle * 11) % 42 for cycle in range(length))
        ending = b"\r\n" if index % 2 else b"\n"
        records.append(
            b"@read" + str(index).encode() + ending + sequence + ending
            + b"+original" + ending + quality + (ending if index < count - 1 or length == 0 else b"")
        )
    return b"".join(records)


def rewrite_metadata(source, output, mutate):
    data = source.read_bytes()
    magic, version, flags, length, _ = _PREFIX.unpack(data[:_PREFIX.size])
    metadata = json.loads(data[_PREFIX.size:_PREFIX.size + length])
    mutate(metadata)
    encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    output.write_bytes(_PREFIX.pack(magic, version, flags, len(encoded), zlib.crc32(encoded) & 0xFFFFFFFF)
                       + encoded + data[_PREFIX.size + length:])


class BlockCodecTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def checkpoint(self, root, mode="features", block_reads=4):
        torch.manual_seed(601)
        model = DirectQualityTransformer(DirectQualityModelConfig(
            prev_q_embed_dim=4, qmer_embed_dim=2, base_embed_dim=3,
            base_conv_channels=3, base_context_dim=4, d_model=8,
            num_heads=2, num_layers=1, feedforward_dim=16,
            context_length=8, dropout=0.0,
            block_quality_features=True, block_reads=block_reads,
            block_feature_mode=mode, block_conditioning_dim=5,
        ))
        # Exercise actual conditioning rather than only the identity initializer.
        with torch.no_grad():
            model.block_conditioning.encoder[-1].weight.uniform_(-0.1, 0.1)
            model.block_conditioning.encoder[-1].bias.uniform_(-0.1, 0.1)
        path = root / f"{mode}.pt"
        save_training_checkpoint(
            path, model=model, optimizer=None, epoch=1, global_step=1,
            data_split={}, sampler_config={}, sampler_statistics={},
            best_validation_bits_per_quality=1.0, validation_metrics={},
        )
        return path

    def encode(self, source, output, checkpoint, **kwargs):
        return encode_fastq(source, output, checkpoint, device=torch.device("cpu"),
                            batch_reads=4, progress=False, verify_cdf=True,
                            coder_backend="python", **kwargs)

    def test_features_and_constant_roundtrip_boundaries_wire_and_accounting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for mode in ("features", "constant"):
                checkpoint = self.checkpoint(root, mode)
                for count in (0, 1, 4, 5, 9):
                    source = root / f"{mode}{count}.fastq"
                    source.write_bytes(fixture_bytes(count))
                    output = source.with_suffix(".fqdc")
                    compressed_metadata = count == 9
                    statistics = self.encode(source, output, checkpoint, compress_metadata=compressed_metadata,
                                             quantization_version=2)
                    info = read_container(output)
                    self.assertEqual(info.metadata["format_version"], BLOCK_FEATURE_CONTAINER_VERSION)
                    expected_payload = (b"".join(encode_block_features(batch) for batch in iter_fastq_batches(source, batch_reads=4))
                                        if mode == "features" else b"")
                    self.assertEqual(info.read_section("block_features"), expected_payload)
                    self.assertEqual(statistics.block_feature_stream_bytes, len(expected_payload))
                    self.assertEqual(statistics.block_feature_metadata_bytes,
                                     block_feature_metadata_bytes(info.metadata, compressed=compressed_metadata))
                    baseline = copy.deepcopy(info.metadata)
                    baseline.pop("block_features")
                    baseline["sections"] = [section for section in baseline["sections"]
                                             if section["name"] != "block_features"]
                    block_config = baseline["checkpoint"]["model_config"]
                    for name in ("block_quality_features", "block_reads", "block_feature_mode", "block_conditioning_dim"):
                        block_config.pop(name, None)
                    plain_config = DirectQualityModelConfig.from_dict(block_config)
                    baseline["checkpoint"]["model_config"] = plain_config.to_dict()
                    baseline["checkpoint"]["feature_schema"] = feature_schema(plain_config)
                    self.assertEqual(statistics.block_feature_metadata_bytes,
                                     len(metadata_payload(info.metadata, compressed=compressed_metadata))
                                     - len(metadata_payload(baseline, compressed=compressed_metadata)))
                    # In particular, charge the block contract in checkpoint
                    # configuration/schema, not just the stream's own framing.
                    framing_only = copy.deepcopy(info.metadata)
                    framing_only.pop("block_features")
                    framing_only["sections"] = [section for section in framing_only["sections"]
                                                 if section["name"] != "block_features"]
                    self.assertGreater(statistics.block_feature_metadata_bytes,
                                       len(metadata_payload(info.metadata, compressed=compressed_metadata))
                                       - len(metadata_payload(framing_only, compressed=compressed_metadata)))
                    self.assertGreater(statistics.block_feature_metadata_bytes, 0)
                    self.assertEqual(statistics.quality_and_adapter_bytes,
                                     statistics.range_stream_bytes + len(expected_payload) + statistics.block_feature_metadata_bytes)
                    restored = root / f"{mode}{count}.restored.fastq"
                    decode_fastq(output, restored, checkpoint, device=torch.device("cpu"),
                                 batch_reads=4, progress=False, verify_cdf=True)
                    self.assertEqual(restored.read_bytes(), source.read_bytes())
                    if count == 5:
                        repeated = root / f"{mode}.repeat.fqdc"
                        self.encode(source, repeated, checkpoint, quantization_version=2)
                        self.assertEqual(output.read_bytes(), repeated.read_bytes())

    def test_default_256_read_blocks_with_partial_tail_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = self.checkpoint(root, block_reads=256)
            source = root / "source.fastq"
            source.write_bytes(fixture_bytes(257))
            output = root / "source.fqdc"
            statistics = encode_fastq(source, output, checkpoint, device=torch.device("cpu"),
                                      batch_reads=256, progress=False, verify_cdf=True,
                                      quantization_version=2, coder_backend="native")
            info = read_container(output)
            self.assertNotIn("block_reads", info.metadata["checkpoint"]["model_config"])
            self.assertNotIn("block_feature_mode", info.metadata["checkpoint"]["model_config"])
            self.assertEqual(statistics.block_feature_stream_bytes, 32)
            self.assertEqual(info.metadata["block_features"]["block_count"], 2)
            restored = root / "restored.fastq"
            decode_fastq(output, restored, checkpoint, device=torch.device("cpu"),
                         batch_reads=256, progress=False, verify_cdf=True)
            self.assertEqual(restored.read_bytes(), source.read_bytes())

    def test_feature_stream_crc_tampering_is_rejected_before_decode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = self.checkpoint(root)
            source = root / "source.fastq"
            source.write_bytes(fixture_bytes(5))
            container = root / "source.fqdc"
            self.encode(source, container, checkpoint)
            info = read_container(container)
            contents = bytearray(container.read_bytes())
            contents[info.header_bytes + info.section("block_features").offset] ^= 1
            broken = root / "crc.fqdc"
            broken.write_bytes(contents)
            with self.assertRaises(ContainerIntegrityError):
                decode_fastq(broken, root / "bad.fastq", checkpoint, device=torch.device("cpu"),
                             batch_reads=4, progress=False)
            self.assertFalse((root / "bad.fastq").exists())

    def test_schema_mode_count_and_stream_length_metadata_are_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = self.checkpoint(root)
            source = root / "source.fastq"
            source.write_bytes(fixture_bytes(5))
            container = root / "source.fqdc"
            self.encode(source, container, checkpoint)
            mutations = (
                lambda m: m["block_features"].update(mode="constant"),
                lambda m: m["block_features"].update(block_count=1),
                lambda m: m["block_features"].update(stream_bytes=31),
                lambda m: m["block_features"]["schema"].update(version=999),
                lambda m: m["checkpoint"]["model_config"].update(block_reads=8),
                lambda m: m["checkpoint"]["model_config"].update(block_quality_features=False),
                lambda m: m["sections"][-1].update(length=31),
                lambda m: m.update(head_adapter={}),
            )
            for index, mutation in enumerate(mutations):
                broken = root / f"metadata{index}.fqdc"
                rewrite_metadata(container, broken, mutation)
                with self.subTest(index=index), self.assertRaises(ContainerError):
                    read_container(broken)

    def test_block_encoder_rejects_mismatched_batch_and_unsupported_combinations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = self.checkpoint(root)
            source = root / "source.fastq"
            source.write_bytes(fixture_bytes(5))
            with self.assertRaisesRegex(ValueError, "batch-reads"):
                encode_fastq(source, root / "batch.fqdc", checkpoint, device=torch.device("cpu"),
                             batch_reads=2, progress=False)
            for index, kwargs in enumerate((
                {"head_adaptation_config": HeadAdaptationConfig()},
                {"head_adapter_path": root / "unused.json"},
                {"quality_coding": "rank"},
            )):
                with self.subTest(index=index), self.assertRaises(ValueError):
                    self.encode(source, root / f"unsupported{index}.fqdc", checkpoint, **kwargs)
            with self.assertRaises(ValueError):
                _encode_fastq_compat(source, root / "prior.fqdc", checkpoint, device=torch.device("cpu"),
                                     batch_reads=4, online_prior_config=DEFAULT_ONLINE_PRIOR_CONFIG, progress=False)
            self.assertFalse(list(root.glob("*.fqdc")))


if __name__ == "__main__":
    unittest.main()
