import gzip
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from codec.checkpoint import LoadedCheckpoint, save_training_checkpoint
from codec.container import (
    CONTAINER_PREFIX_BYTES,
    SECTION_NAMES,
    ContainerError,
    ContainerIntegrityError,
    TruncatedContainerError,
    read_container,
)
from codec.decode import ModelMismatchError, build_parser as build_decode_parser
from codec.decode import decode_fastq
from codec.encode import CodecDeterminismError, _quantize_verified_batch
from codec.encode import build_parser as build_encode_parser
from codec.encode import encode_fastq
from codec.fastq_stream import DEFAULT_BATCH_READS, iter_fastq_batches
from codec.model import DirectQualityModelConfig, DirectQualityTransformer


def tiny_config():
    return DirectQualityModelConfig(
        prev_q_embed_dim=4,
        qmer_embed_dim=2,
        base_embed_dim=3,
        base_conv_channels=3,
        base_context_dim=4,
        d_model=8,
        num_heads=2,
        num_layers=1,
        feedforward_dim=16,
        context_length=8,
        dropout=0.0,
    )


def fastq_bytes(read_count):
    records = []
    bases = b"ACGTN"
    for read_index in range(read_count):
        length = read_index % 4 + 1
        sequence = bytes(
            bases[(read_index + position) % len(bases)] for position in range(length)
        )
        quality_ids = [
            (read_index * 7 + position * 11) % 42 for position in range(length)
        ]
        if read_index == 0:
            quality_ids[0] = 0
        if read_index == 1:
            quality_ids[0] = 41
        quality = bytes(33 + quality_id for quality_id in quality_ids)
        ending = b"\r\n" if read_index % 2 else b"\n"
        quality_ending = b"" if read_index == read_count - 1 else ending
        records.append(
            b"@read "
            + str(read_index).encode("ascii")
            + ending
            + sequence
            + ending
            + b"+description "
            + str(read_index).encode("ascii")
            + ending
            + quality
            + quality_ending
        )
    return b"".join(records)


class MismatchedStepModel(DirectQualityTransformer):
    def forward_step(self, *args, **kwargs):
        logits = super().forward_step(*args, **kwargs).clone()
        logits[:, 0] += 100.0
        return logits


class NeuralCodecRoundTripTest(unittest.TestCase):
    def _checkpoint(self, root, name="model.pt", seed=17):
        torch.manual_seed(seed)
        model = DirectQualityTransformer(tiny_config())
        path = Path(root) / name
        save_training_checkpoint(
            path,
            model=model,
            optimizer=None,
            epoch=1,
            global_step=1,
            data_split={"strategy": "test", "datasets": {}},
            sampler_config={"strategy": "test"},
            sampler_statistics={"total_batches": 1},
            best_validation_bits_per_quality=1.0,
            validation_metrics={"symbol_micro_bits_per_quality": 1.0},
        )
        return path, model

    def _gzip_fastq(self, root, name, contents):
        path = Path(root) / name
        with path.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as handle:
                handle.write(contents)
        return path

    def _round_trip(
        self,
        root,
        read_count,
        *,
        gzip_output=False,
        batch_reads=DEFAULT_BATCH_READS,
    ):
        root = Path(root)
        checkpoint, _ = self._checkpoint(root)
        original = fastq_bytes(read_count)
        source = self._gzip_fastq(root, "input.fq.gz", original)
        container = root / "reads.fqdc"
        output = root / ("restored.fastq.gz" if gzip_output else "restored.fastq")
        encode_stats = encode_fastq(
            source,
            container,
            checkpoint,
            device=torch.device("cpu"),
            batch_reads=batch_reads,
            progress=False,
        )
        decode_stats = decode_fastq(
            container,
            output,
            checkpoint,
            device=torch.device("cpu"),
            batch_reads=batch_reads,
            progress=False,
        )
        if gzip_output:
            with gzip.open(output, "rb") as handle:
                restored = handle.read()
        else:
            restored = output.read_bytes()
        self.assertEqual(restored, original)
        return container, encode_stats, decode_stats

    def test_codec_cli_defaults_to_256_reads(self):
        encode_args = build_encode_parser().parse_args(
            ["input.fq.gz", "output.fqdc", "model.pt"]
        )
        decode_args = build_decode_parser().parse_args(
            ["input.fqdc", "output.fq", "model.pt"]
        )
        self.assertEqual(encode_args.batch_reads, 256)
        self.assertEqual(decode_args.batch_reads, 256)

    def test_existing_64_read_grouping_remains_supported(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            container, encode_stats, decode_stats = self._round_trip(
                temporary_directory, 65, batch_reads=64
            )
            self.assertEqual(encode_stats.batch_count, 2)
            self.assertEqual(decode_stats.read_count, 65)
            self.assertEqual(read_container(container).metadata["batch_reads"], 64)

    def test_63_64_65_and_255_256_257_reads_round_trip(self):
        for read_count in (63, 64, 65, 255, 256, 257):
            with self.subTest(read_count=read_count):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    container, encode_stats, decode_stats = self._round_trip(
                        temporary_directory, read_count
                    )
                    expected_symbols = sum(index % 4 + 1 for index in range(read_count))
                    self.assertEqual(encode_stats.read_count, read_count)
                    self.assertEqual(encode_stats.quality_symbols, expected_symbols)
                    self.assertEqual(
                        encode_stats.batch_count,
                        math.ceil(read_count / DEFAULT_BATCH_READS),
                    )
                    self.assertEqual(decode_stats.quality_symbols, expected_symbols)
                    info = read_container(container)
                    self.assertEqual(
                        tuple(section.name for section in info.sections), SECTION_NAMES
                    )
                    self.assertEqual(info.metadata["read_count"], read_count)
                    self.assertEqual(
                        info.metadata["batch_reads"], DEFAULT_BATCH_READS
                    )
                    self.assertEqual(
                        info.metadata["last_batch_read_count"],
                        ((read_count - 1) % DEFAULT_BATCH_READS) + 1,
                    )
                    self.assertEqual(
                        info.file_size,
                        info.header_bytes + sum(section.length for section in info.sections),
                    )
                    self.assertGreater(encode_stats.header_gzip_bytes, 0)
                    self.assertGreater(encode_stats.base_gzip_bytes, 0)
                    self.assertGreater(encode_stats.plus_gzip_bytes, 0)
                    self.assertGreater(encode_stats.range_stream_bits_per_quality, 0)

    def test_gzip_output_decompresses_to_identical_fastq(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            self._round_trip(temporary_directory, 3, gzip_output=True)

    def test_empty_gzip_fastq_round_trip_has_no_quality_rate(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            container, encode_stats, decode_stats = self._round_trip(
                temporary_directory, 0
            )
            self.assertEqual(encode_stats.read_count, 0)
            self.assertEqual(encode_stats.quality_symbols, 0)
            self.assertIsNone(encode_stats.range_payload_bits_per_quality)
            self.assertIsNone(encode_stats.range_stream_bits_per_quality)
            self.assertEqual(decode_stats.output_uncompressed_bytes, 0)
            self.assertEqual(read_container(container).metadata["batch_count"], 0)

    def test_cycle_major_order_and_full_step_cdfs(self):
        original = (
            b"@r0\nACG\n+\n!\"#\n"
            b"@r1\nT\n+\n$\n"
            b"@r2\nNN\n+\n%&\n"
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = self._gzip_fastq(root, "order.fastq.gz", original)
            _, model = self._checkpoint(root)
            batch = next(iter_fastq_batches(source))
            model.eval()
            items, _ = _quantize_verified_batch(
                model, batch, torch.device("cpu"), total=1 << 16
            )
        self.assertEqual([symbol for symbol, _ in items], [0, 3, 4, 1, 5, 2])
        self.assertTrue(all(len(cdf) == 43 for _, cdf in items))

    def test_q42_is_rejected_without_output(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            checkpoint, _ = self._checkpoint(root)
            source = self._gzip_fastq(root, "bad.fq.gz", b"@r\nA\n+\nK\n")
            output = root / "bad.fqdc"
            with self.assertRaisesRegex(ValueError, "outside Q0-Q41"):
                encode_fastq(
                    source,
                    output,
                    checkpoint,
                    device=torch.device("cpu"),
                    progress=False,
                )
            self.assertFalse(output.exists())

    def test_integer_cdf_mismatch_aborts_before_container_write(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            checkpoint, _ = self._checkpoint(root)
            source = self._gzip_fastq(
                root, "mismatch.fq.gz", b"@r\nAC\n+\n!J\n"
            )
            output = root / "mismatch.fqdc"
            torch.manual_seed(17)
            bad_model = MismatchedStepModel(tiny_config())
            loaded = LoadedCheckpoint(model=bad_model, payload={})
            with mock.patch("codec.encode.load_training_checkpoint", return_value=loaded):
                with self.assertRaisesRegex(CodecDeterminismError, "integer CDF"):
                    encode_fastq(
                        source,
                        output,
                        checkpoint,
                        device=torch.device("cpu"),
                        progress=False,
                    )
            self.assertFalse(output.exists())

    def test_wrong_model_hash_is_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            checkpoint, _ = self._checkpoint(root, "first.pt", seed=1)
            wrong_checkpoint, _ = self._checkpoint(root, "second.pt", seed=2)
            source = self._gzip_fastq(root, "input.fastq.gz", fastq_bytes(2))
            container = root / "reads.fqdc"
            encode_fastq(
                source,
                container,
                checkpoint,
                device=torch.device("cpu"),
                progress=False,
            )
            output = root / "wrong.fastq"
            with self.assertRaisesRegex(ModelMismatchError, "SHA-256"):
                decode_fastq(
                    container,
                    output,
                    wrong_checkpoint,
                    device=torch.device("cpu"),
                    progress=False,
                )
            self.assertFalse(output.exists())

    def test_truncated_corrupt_metadata_and_section_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            checkpoint, _ = self._checkpoint(root)
            source = self._gzip_fastq(root, "input.fq.gz", fastq_bytes(3))
            container = root / "valid.fqdc"
            encode_fastq(
                source,
                container,
                checkpoint,
                device=torch.device("cpu"),
                progress=False,
            )
            valid_bytes = container.read_bytes()

            truncated = root / "truncated.fqdc"
            truncated.write_bytes(valid_bytes[:-1])
            with self.assertRaises(TruncatedContainerError):
                decode_fastq(
                    truncated,
                    root / "truncated.fastq",
                    checkpoint,
                    device=torch.device("cpu"),
                    progress=False,
                )

            bad_metadata_bytes = bytearray(valid_bytes)
            bad_metadata_bytes[CONTAINER_PREFIX_BYTES] ^= 0x01
            bad_metadata = root / "bad_metadata.fqdc"
            bad_metadata.write_bytes(bad_metadata_bytes)
            with self.assertRaises(ContainerIntegrityError):
                decode_fastq(
                    bad_metadata,
                    root / "bad_metadata.fastq",
                    checkpoint,
                    device=torch.device("cpu"),
                    progress=False,
                )

            info = read_container(container)
            quality = info.section("quality_range")
            bad_section_bytes = bytearray(valid_bytes)
            bad_section_bytes[info.header_bytes + quality.offset] ^= 0x80
            bad_section = root / "bad_section.fqdc"
            bad_section.write_bytes(bad_section_bytes)
            with self.assertRaises(ContainerIntegrityError):
                decode_fastq(
                    bad_section,
                    root / "bad_section.fastq",
                    checkpoint,
                    device=torch.device("cpu"),
                    progress=False,
                )

    def test_decoder_batch_size_must_match_container(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            checkpoint, _ = self._checkpoint(root)
            source = self._gzip_fastq(root, "input.fq.gz", fastq_bytes(2))
            container = root / "reads.fqdc"
            encode_fastq(
                source,
                container,
                checkpoint,
                device=torch.device("cpu"),
                batch_reads=2,
                progress=False,
            )
            with self.assertRaisesRegex(ContainerError, "batch_reads"):
                decode_fastq(
                    container,
                    root / "wrong_batch.fastq",
                    checkpoint,
                    device=torch.device("cpu"),
                    batch_reads=DEFAULT_BATCH_READS,
                    progress=False,
                )


if __name__ == "__main__":
    unittest.main()
