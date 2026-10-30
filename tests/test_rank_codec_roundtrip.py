"""Byte-exact integration checks for the fixed, file-wide rank histogram codec."""

import copy
import gzip
import json
import struct
import tempfile
import unittest
import zlib
from pathlib import Path

import torch

from codec.checkpoint import save_training_checkpoint
from codec.container import ContainerError, read_container, sha256_file
from codec.decode import decode_fastq
from codec.encode import _encode_fastq_compat, encode_fastq
from codec.head_adapter import HeadAdaptationConfig, serialize_head
from codec.model import DirectQualityModelConfig, DirectQualityTransformer
from codec.online_prior import OnlinePriorConfig


def checkpoint(root, *, equal_logits=False):
    torch.manual_seed(683)
    model = DirectQualityTransformer(DirectQualityModelConfig(
        d_model=8, num_heads=2, num_layers=1, feedforward_dim=16,
        prev_q_embed_dim=4, qmer_embed_dim=2, base_embed_dim=3,
        base_conv_channels=3, base_context_dim=4, context_length=16,
        dropout=0.0,
    ))
    if equal_logits:
        with torch.no_grad():
            model.output_head.weight.zero_()
            model.output_head.bias.zero_()
    path = root / "base.pt"
    save_training_checkpoint(
        path, model=model, optimizer=None, epoch=1, global_step=1,
        data_split={}, sampler_config={}, sampler_statistics={},
        best_validation_bits_per_quality=1.0, validation_metrics={},
    )
    return model, path


def fastq_fixture():
    """Include Q boundaries, varying lengths, empty reads, mixed endings and no EOF LF."""
    records = []
    quality_counts = [0] * 42
    for index in range(17):
        length = index % 6
        if index == 16:
            length = 5
        qualities = [(index * 7 + position * 11) % 42 for position in range(length)]
        if index == 1:
            qualities[0] = 0
        if index == 2:
            qualities[0] = 41
        for value in qualities:
            quality_counts[value] += 1
        ending = b"\r\n" if index % 2 else b"\n"
        final_ending = b"" if index == 16 else ending
        bases = bytes(b"ACGTN"[(index + position) % 5] for position in range(length))
        records.append(
            b"@read " + str(index).encode() + ending + bases + ending
            + b"+description " + str(index).encode() + ending
            + bytes(33 + value for value in qualities) + final_ending
        )
    return b"".join(records), quality_counts


def artifact(root, checkpoint_path, adapter):
    path = root / "adapter.json"
    path.write_text(json.dumps({
        "format": "direct-quality-head-adaptation-artifact", "version": 1,
        "base_checkpoint_sha256": sha256_file(checkpoint_path),
        "adapter": adapter, "report": {},
    }))
    return path


def rewrite_metadata(source, destination, mutate):
    prefix = struct.Struct("<8sHHII")
    raw = source.read_bytes()
    magic, version, flags, size, _ = prefix.unpack(raw[:prefix.size])
    if flags:
        raise AssertionError("this helper requires uncompressed metadata")
    metadata = json.loads(raw[prefix.size:prefix.size + size])
    mutate(metadata)
    payload = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    destination.write_bytes(prefix.pack(
        magic, version, flags, len(payload), zlib.crc32(payload) & 0xffffffff,
    ) + payload + raw[prefix.size + size:])


class RankCodecRoundTripTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def kwargs(self):
        return dict(device=torch.device("cpu"), batch_reads=8,
                    progress=False, verify_cdf=True)

    def test_rank_roundtrip_ragged_reads_both_quantizers_and_metadata_modes(self):
        raw, quality_counts = fastq_fixture()
        for quantizer, compressed_metadata in ((1, False), (2, False), (2, True)):
            with self.subTest(quantizer=quantizer, compressed_metadata=compressed_metadata):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    _, model_path = checkpoint(root)
                    source, encoded, restored = root / "input.fq.gz", root / "rank.fqdc", root / "out.fq"
                    source.write_bytes(gzip.compress(raw))
                    stats = encode_fastq(
                        source, encoded, model_path, quality_coding="rank",
                        quantization_version=quantizer,
                        compress_metadata=compressed_metadata, **self.kwargs(),
                    )
                    decode_fastq(encoded, restored, model_path, **self.kwargs())
                    self.assertEqual(restored.read_bytes(), raw)
                    metadata = read_container(encoded).metadata
                    self.assertEqual(metadata["format_version"], 5)
                    rank_metadata = metadata["rank_encoding"]
                    self.assertEqual(rank_metadata["format"], "global-quality-rank-v1")
                    self.assertEqual(rank_metadata["order"], "descending_neural_integer_frequency_then_quality")
                    self.assertEqual(len(rank_metadata["counts"]), 42)
                    self.assertEqual(sum(rank_metadata["counts"]), sum(quality_counts))
                    self.assertEqual(stats.quality_coding, "rank")
                    self.assertEqual(stats.rank_histogram, rank_metadata["counts"])
                    self.assertGreater(stats.rank_table_bytes, 0)
                    self.assertEqual(stats.quality_and_adapter_bytes,
                                     stats.range_stream_bytes + stats.rank_table_bytes)

    def test_equal_neural_logits_use_quality_id_tie_order(self):
        raw, expected_counts = fastq_fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, model_path = checkpoint(root, equal_logits=True)
            source, encoded = root / "input.fq", root / "rank.fqdc"
            source.write_bytes(raw)
            encode_fastq(source, encoded, model_path, quality_coding="rank",
                         quantization_version=2, **self.kwargs())
            # With equal logits, floor-quantized neural frequencies are equal;
            # rank is exactly Q id, including the lowest and highest Q values.
            self.assertEqual(read_container(encoded).metadata["rank_encoding"]["counts"], expected_counts)
            restored = root / "out.fq"
            decode_fastq(encoded, restored, model_path, **self.kwargs())
            self.assertEqual(restored.read_bytes(), raw)

    def test_rank_roundtrip_loads_existing_adapter_without_decoder_artifact(self):
        for compressed_metadata in (False, True):
            with self.subTest(compressed_metadata=compressed_metadata), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                model, model_path = checkpoint(root)
                with torch.no_grad():
                    model.output_head.bias.add_(torch.linspace(-0.5, 0.5, 42))
                saved_adapter = serialize_head(model.output_head, sha256_file(model_path))
                artifact_path = artifact(root, model_path, saved_adapter)
                raw, _ = fastq_fixture()
                source, encoded, restored = root / "input.fq", root / "rank.fqdc", root / "out.fq"
                source.write_bytes(raw)
                stats = encode_fastq(
                    source, encoded, model_path, quality_coding="rank",
                    head_adapter_path=artifact_path, quantization_version=2,
                    compress_metadata=compressed_metadata, **self.kwargs(),
                )
                metadata = read_container(encoded).metadata
                self.assertEqual(metadata["head_adapter"], saved_adapter)
                baseline = dict(metadata)
                for member in ("head_adapter", "probability_profile", "rank_encoding"):
                    baseline.pop(member, None)

                def packed_size(value):
                    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                                         ensure_ascii=True, allow_nan=False).encode("utf-8")
                    return len(zlib.compress(payload, level=9) if compressed_metadata else payload)

                # Compressed members share information. Individual charges must
                # still add up to their actual joint contribution to the wire.
                extra_metadata_bytes = packed_size(metadata) - packed_size(baseline)
                self.assertEqual(stats.rank_table_bytes + stats.head_adaptation["container_adapter_bytes"],
                                 extra_metadata_bytes)
                self.assertEqual(stats.quality_and_adapter_bytes,
                                 stats.range_stream_bytes + extra_metadata_bytes)
                artifact_path.unlink()
                decode_fastq(encoded, restored, model_path, **self.kwargs())
                self.assertEqual(restored.read_bytes(), raw)

    def test_rank_empty_file_roundtrip(self):
        for quantizer in (1, 2):
            with self.subTest(quantizer=quantizer), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _, model_path = checkpoint(root)
                source, encoded, restored = root / "empty.fq", root / "rank.fqdc", root / "out.fq"
                source.write_bytes(b"")
                stats = encode_fastq(source, encoded, model_path, quality_coding="rank",
                                     quantization_version=quantizer, **self.kwargs())
                decode_fastq(encoded, restored, model_path, **self.kwargs())
                self.assertEqual(restored.read_bytes(), b"")
                self.assertEqual(read_container(encoded).metadata["rank_encoding"]["counts"], [0] * 42)
                self.assertEqual(stats.quality_symbols, 0)

    def test_default_neural_bytes_and_old_container_version_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, model_path = checkpoint(root)
            source = root / "input.fq"
            source.write_bytes(fastq_fixture()[0])
            default, explicit = root / "default.fqdc", root / "explicit.fqdc"
            encode_fastq(source, default, model_path, **self.kwargs())
            encode_fastq(source, explicit, model_path, quality_coding="neural", **self.kwargs())
            self.assertEqual(default.read_bytes(), explicit.read_bytes())
            metadata = read_container(default).metadata
            self.assertEqual(metadata["format_version"], 1)
            self.assertNotIn("rank_encoding", metadata)

    def test_rank_rejects_legacy_prior_and_read_state_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, model_path = checkpoint(root)
            source = root / "input.fq"
            source.write_bytes(fastq_fixture()[0])
            with self.assertRaisesRegex(ValueError, "rank"):
                _encode_fastq_compat(
                    source, root / "prior.fqdc", model_path, quality_coding="rank",
                    online_prior_config=OnlinePriorConfig(), **self.kwargs(),
                )
            with self.assertRaisesRegex(ValueError, "rank|read.state"):
                encode_fastq(
                    source, root / "state.fqdc", model_path, quality_coding="rank",
                    head_adaptation_config=HeadAdaptationConfig(backbone_layers=1, read_quality_state=True),
                    **self.kwargs(),
                )

    def test_rank_rejects_loaded_read_state_adapter(self):
        from codec.backbone_adapter import serialize_backbone
        from codec.read_state import ReadStateConditioning, serialize_read_state

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, model_path = checkpoint(root)
            candidate = copy.deepcopy(base)
            candidate.read_state_conditioning = ReadStateConditioning(8, [0], [10000, 20000, 30000])
            digest = sha256_file(model_path)
            nested = serialize_backbone(candidate, 1, digest, base_model=base)
            state_adapter = serialize_read_state(candidate, nested, digest)
            artifact_path = artifact(root, model_path, state_adapter)
            source = root / "input.fq"
            source.write_bytes(fastq_fixture()[0])
            with self.assertRaisesRegex(ValueError, "rank|read.state"):
                encode_fastq(source, root / "state.fqdc", model_path, quality_coding="rank",
                             head_adapter_path=artifact_path, **self.kwargs())

    def test_rank_metadata_rejects_invalid_histogram_and_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, model_path = checkpoint(root)
            source, encoded = root / "input.fq", root / "rank.fqdc"
            source.write_bytes(fastq_fixture()[0])
            encode_fastq(source, encoded, model_path, quality_coding="rank", **self.kwargs())
            changes = (
                lambda metadata: metadata["rank_encoding"].update(counts=[1] * 41),
                lambda metadata: metadata["rank_encoding"]["counts"].__setitem__(0, -1),
                lambda metadata: metadata["rank_encoding"]["counts"].__setitem__(0, True),
                lambda metadata: metadata["rank_encoding"]["counts"].__setitem__(0, 999),
                lambda metadata: metadata["rank_encoding"].update(order="descending_float_logits"),
            )
            for index, change in enumerate(changes):
                with self.subTest(index=index):
                    damaged = root / f"damaged{index}.fqdc"
                    rewrite_metadata(encoded, damaged, change)
                    with self.assertRaises(ContainerError):
                        decode_fastq(damaged, root / f"bad{index}.fq", model_path, **self.kwargs())


if __name__ == "__main__":
    unittest.main()
