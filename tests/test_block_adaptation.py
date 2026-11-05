"""Reservoir sampling must preserve original quality-block membership."""

import hashlib
import random
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from codec.block_adaptation import sample_block_records, sampled_block_ids_per_read
from codec.block_features import decode_block_features, encode_block_features
from codec.fastq_stream import encode_base_ids, iter_fastq_records, make_fastq_batch


def config(**changes):
    values = dict(backbone_max_reads=12, backbone_max_symbols=10000,
                  max_read_length=8, seed=174)
    values.update(changes)
    return SimpleNamespace(**values)


def write_source(root, lengths):
    source = root / "blocks.fastq"
    source.write_bytes(b"".join(
        b"@read" + str(index).encode() + b"\n" + b"A" * length + b"\n+\n"
        + bytes([33 + ((index * 3 + 2) % 42)]) * length + b"\n"
        for index, length in enumerate(lengths)
    ))
    return source


def whole_block_feature(rows):
    batch = make_fastq_batch(
        [encode_base_ids(row.sequence) for row in rows],
        [np.frombuffer(row.quality, dtype=np.uint8) - 33 for row in rows],
        [row.read_index for row in rows], source_name="blocks.fastq",
    )
    return decode_block_features(encode_block_features(batch))


class BlockAdaptationTest(unittest.TestCase):
    def sample(self, source, cfg=None, skip=0, deadline=float("inf"), **changes):
        arguments = dict(block_reads=4, feature_mode="features")
        arguments.update(changes)
        return sample_block_records(source, cfg or config(), skip, deadline, **arguments)

    def assert_block_alignment(self, records, report, features, original):
        ids = sampled_block_ids_per_read(report)
        self.assertEqual(features.dtype, np.float32)
        self.assertEqual(features.shape, (len(records), 16))
        self.assertEqual(len(records), report["sampled_reads"])
        self.assertEqual(len(report["sampled_block_ids"]), report["sampled_blocks"])
        self.assertEqual(sum(len(row.quality) for row in records), report["sampled_symbols"])
        offset = 0
        payload = bytearray()
        for block_id, count, bounds in zip(
            report["sampled_block_ids"], report["sampled_block_read_counts"],
            report["sampled_block_read_ranges"],
        ):
            start = block_id * 4
            stop = min(start + 4, len(original))
            self.assertEqual(bounds, [start, stop])
            self.assertEqual(count, stop - start)
            self.assertEqual([row.read_index for row in records[offset:offset + count]], list(range(start, stop)))
            expected = whole_block_feature(original[start:stop])
            np.testing.assert_array_equal(features[offset:offset + count], np.tile(expected, (count, 1)))
            np.testing.assert_array_equal(ids[offset:offset + count], np.full(count, block_id))
            payload.extend(np.rint(expected * np.float32(255)).astype(np.uint8).tobytes())
            offset += count
        self.assertEqual(report["block_feature_payload_sha256"], hashlib.sha256(payload).hexdigest())
        self.assertEqual(report["read_indices_sha256"], hashlib.sha256(
            np.asarray([row.read_index for row in records], dtype="<i8").tobytes()).hexdigest())
        self.assertEqual(report["sampled_records_sha256"], hashlib.sha256(
            b"".join(row.to_bytes() for row in records)).hexdigest())

    def test_uniform_block_reservoir_is_reproducible_skip_aligned_and_descriptors_lazy(self):
        with tempfile.TemporaryDirectory() as directory:
            source = write_source(Path(directory), [4] * 64)
            original = list(iter_fastq_records(source))
            random_state = random.getstate()
            with mock.patch("codec.block_adaptation.encode_block_features", wraps=encode_block_features) as encode:
                a, report_a, features_a = self.sample(source, skip=5)
            b, report_b, features_b = self.sample(source, skip=5)
            self.assertEqual(random_state, random.getstate())
            self.assertEqual([row.read_index for row in a], [row.read_index for row in b])
            self.assertEqual(report_a, report_b)
            np.testing.assert_array_equal(features_a, features_b)
            self.assertTrue(report_a["sampling_complete"])
            self.assertEqual(report_a["aligned_skip_reads"], 8)
            self.assertEqual(report_a["excluded_warmup_reads"], 8)
            self.assertEqual(report_a["eligible_blocks"], 14)
            self.assertEqual(report_a["sampled_blocks"], 3)
            self.assertEqual(encode.call_count, 3)  # Not all 16 original blocks.
            self.assertTrue(all(row.read_index >= 8 for row in a))
            self.assertGreater(max(row.read_index for row in a), 32)
            self.assert_block_alignment(a, report_a, features_a, original)

    def test_long_read_excludes_entire_block_empty_rows_stay_and_eof_tail_is_genuine(self):
        lengths = [4] * 19
        lengths[2] = 0
        lengths[5] = 9
        lengths[8:12] = [0] * 4
        lengths[17] = 0
        with tempfile.TemporaryDirectory() as directory:
            source = write_source(Path(directory), lengths)
            records, report, features = self.sample(source, config(backbone_max_reads=24))
            original = list(iter_fastq_records(source))
            self.assertEqual(report["total_file_reads"], 19)
            self.assertEqual(report["total_file_symbols"], sum(lengths))
            self.assertEqual(report["total_file_blocks"], 5)
            self.assertEqual(report["excluded_long_blocks"], 1)
            self.assertEqual(report["excluded_long_reads"], 1)
            self.assertEqual(report["excluded_long_block_reads"], 4)
            self.assertEqual(report["excluded_empty_blocks"], 1)
            self.assertEqual(set(report["sampled_block_ids"]), {0, 3, 4})
            self.assertIn(3, report["sampled_block_read_counts"])
            self.assertEqual(len(records), 11)
            self.assertEqual({row.read_index for row in records if not row.quality}, {2, 17})
            self.assert_block_alignment(records, report, features, original)

    def test_symbol_budget_drops_whole_blocks_and_can_still_keep_small_eof_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            source = write_source(Path(directory), [4] * 12 + [1, 1])
            cfg = config(backbone_max_reads=16, backbone_max_symbols=18)
            records, report, features = self.sample(source, cfg)
            self.assertEqual(report["sampled_blocks"], 2)
            self.assertEqual(report["sampled_reads"], 6)
            self.assertEqual(report["sampled_symbols"], 18)
            self.assertEqual(report["excluded_symbol_cap_blocks"], 2)
            self.assertEqual(report["excluded_symbol_cap_reads"], 8)
            self.assertEqual(report["sampled_block_feature_stream_bytes"], 32)
            self.assert_block_alignment(records, report, features, list(iter_fastq_records(source)))

    def test_read_cap_is_capacity_in_whole_blocks_never_a_truncated_read_reservoir(self):
        with tempfile.TemporaryDirectory() as directory:
            source = write_source(Path(directory), [4] * 40)
            records, report, features = self.sample(source, config(backbone_max_reads=10))
            self.assertEqual(report["reservoir_max_blocks"], 2)
            self.assertEqual(report["sampled_blocks"], 2)
            self.assertEqual(len(records), 8)
            self.assert_block_alignment(records, report, features, list(iter_fastq_records(source)))
            empty, small_report, empty_z = self.sample(source, config(backbone_max_reads=3))
            self.assertTrue(small_report["sampling_complete"])
            self.assertEqual(empty, [])
            self.assertEqual(empty_z.shape, (0, 16))
            self.assertEqual(sampled_block_ids_per_read(small_report).shape, (0,))

    def test_constant_control_uses_known_zeros_without_descriptor_computation(self):
        with tempfile.TemporaryDirectory() as directory:
            source = write_source(Path(directory), [4] * 14)
            with mock.patch("codec.block_adaptation.encode_block_features") as encode:
                records, report, features = self.sample(source, feature_mode="constant")
            encode.assert_not_called()
            self.assertGreater(len(records), 0)
            self.assertEqual(np.count_nonzero(features), 0)
            self.assertEqual(report["sampled_block_feature_stream_bytes"], 0)
            self.assertEqual(report["block_feature_payload_sha256"], hashlib.sha256(b"").hexdigest())

    def test_deadline_discards_partial_reservoir_and_keeps_empty_aligned_features(self):
        missing = Path("missing-block-adaptation-source.fastq")
        records, report, features = self.sample(missing, deadline=time.perf_counter() - 1)
        self.assertFalse(report["sampling_complete"])
        self.assertEqual(records, [])
        self.assertEqual(features.shape, (0, 16))
        with tempfile.TemporaryDirectory() as directory:
            source = write_source(Path(directory), [4] * 24)
            # Expire after enough records to have retained a block.
            with mock.patch("codec.block_adaptation.time.perf_counter", side_effect=[0] * 9 + [2]):
                records, report, features = self.sample(source, deadline=1)
            self.assertFalse(report["sampling_complete"])
            self.assertGreater(report["eligible_blocks"], 0)
            self.assertEqual(report["sampled_blocks"], 0)
            self.assertEqual(records, [])
            self.assertEqual(features.shape, (0, 16))

    def test_invalid_contracts_reject_before_source_io(self):
        missing = Path("missing-block-adaptation-source.fastq")
        for changes in ({"block_reads": 0}, {"block_reads": 257}, {"feature_mode": "unknown"}):
            with self.assertRaises(ValueError):
                self.sample(missing, **changes)
        with self.assertRaises(ValueError):
            sampled_block_ids_per_read({"sampled_block_ids": [0], "sampled_block_read_counts": [4], "sampled_reads": 3})


if __name__ == "__main__":
    unittest.main()
