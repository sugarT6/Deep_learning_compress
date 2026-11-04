import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from codec.block_features import (
    BLOCK_FEATURE_BYTES,
    CacheBlockFeatures,
    block_bounds,
    block_feature_schema,
    block_ids_for_read_indices,
    decode_block_features,
    encode_block_features,
)
from codec.fastq_stream import make_fastq_batch
from codec.training_cache import TrainingCacheReader, create_training_cache


def make_batch(rows, start=0):
    qualities = [np.asarray(row, dtype=np.uint8) for row in rows]
    return make_fastq_batch(
        [np.zeros(len(row), dtype=np.uint8) for row in rows],
        qualities,
        range(start, start + len(rows)),
        source_name="source.fastq",
    )


class BlockFeaturesTest(unittest.TestCase):
    def test_exact_wire_order_and_integer_half_up_quantization(self):
        # Every segment receives one Q0 and one Q41.  Mean and std are
        # both Q20.5; low and high fractions are both one half.
        payload = encode_block_features(make_batch([[0] * 4, [41] * 4]))
        self.assertEqual(payload, bytes([128, 128, 128, 128] * 4))
        self.assertEqual(len(payload), BLOCK_FEATURE_BYTES)
        decoded = decode_block_features(payload)
        self.assertEqual(decoded.dtype, np.float32)
        self.assertEqual(decoded.shape, (16,))
        np.testing.assert_array_equal(decoded, np.full(16, np.float32(128) / np.float32(255)))
        self.assertEqual(block_feature_schema()["descriptor_bytes"], 16)
        self.assertEqual(block_feature_schema()["block_reads"], 256)
        self.assertEqual(block_feature_schema(block_reads=64)["block_reads"], 64)
        with self.assertRaises(ValueError):
            block_feature_schema(block_reads=0)

    def test_relative_segments_pool_variable_read_lengths_and_ignore_padding(self):
        # Length4 has one Q per segment, length8 has two.  Segmentation
        # uses each read's length rather than absolute padded columns.
        payload = encode_block_features(make_batch([
            [0, 20, 35, 41],
            [0, 0, 20, 20, 35, 35, 41, 41],
        ]))
        self.assertEqual(payload, bytes([
            0, 0, 255, 0,
            124, 0, 0, 0,
            218, 0, 0, 255,
            255, 0, 0, 255,
        ]))

    def test_empty_segments_and_zero_length_reads_have_explicit_zero_wire(self):
        self.assertEqual(encode_block_features(make_batch([[], []])), bytes(16))
        # floor(s*L/4) places the sole Q in the fourth segment.
        self.assertEqual(
            encode_block_features(make_batch([[], [41]])),
            bytes([0] * 12 + [255, 0, 0, 255]),
        )

    def test_decoder_rejects_wrong_lengths_and_non_byte_inputs(self):
        for payload in (b"", bytes(15), bytes(17)):
            with self.assertRaises(ValueError):
                decode_block_features(payload)
        with self.assertRaises(TypeError):
            decode_block_features([0] * 16)
        np.testing.assert_array_equal(decode_block_features(bytearray(16)), np.zeros(16))

    def test_fixed_block_indices_and_partial_file_tail(self):
        np.testing.assert_array_equal(
            block_ids_for_read_indices([0, 255, 256, 1024, 256]), [0, 0, 1, 4, 1]
        )
        self.assertEqual(block_bounds(4, 1030), (1024, 1030))
        for indices in ([-1], [], [[1]], [1.2]):
            with self.assertRaises((ValueError, IndexError)):
                block_ids_for_read_indices(indices)
        for block_id in (-1, 5, 0.5):
            with self.assertRaises(IndexError):
                block_bounds(block_id, 1030)
        with self.assertRaises(ValueError):
            block_bounds(0, 100, block_reads=0)

    def test_cache_lookup_uses_whole_block_not_requested_microbatch(self):
        class FakeReader:
            metadata = SimpleNamespace(read_count=6)

            def __init__(self):
                self.requests = []

            def read_range(self, start, stop):
                self.requests.append((start, stop))
                return make_batch([[0] * 4 if i < 2 else [41] * 4 for i in range(start, stop)], start)

        reader = FakeReader()
        features = CacheBlockFeatures(reader, block_reads=4, split_start=0, split_stop=4)
        found = features.for_read_indices([0, 3, 0])
        expected = decode_block_features(bytes([128] * 16))
        np.testing.assert_array_equal(found, np.tile(expected, (3, 1)))
        self.assertEqual(reader.requests, [(0, 4)])
        self.assertEqual(features.payload_for_block(0), bytes([128] * 16))
        self.assertEqual(reader.requests, [(0, 4)])
        with self.assertRaises(IndexError):
            features.for_read_indices([4])
        with self.assertRaises(IndexError):
            features.payload_for_block(1)

    def test_cache_large_blocks_are_chunked_without_crossing_split(self):
        class FakeReader:
            metadata = SimpleNamespace(read_count=1030)

            def __init__(self):
                self.requests = []

            def read_range(self, start, stop):
                self.requests.append((start, stop))
                return make_batch([[35] * 4] * (stop - start), start)

        reader = FakeReader()
        features = CacheBlockFeatures(reader, block_reads=512, split_start=512)
        result = features.for_read_indices([1029, 512, 1024])
        self.assertEqual(result.shape, (3, 16))
        self.assertEqual(reader.requests, [(512, 768), (768, 1024), (1024, 1030)])
        np.testing.assert_array_equal(result[0], result[1])
        with self.assertRaises(IndexError):
            features.for_read_indices([511])
        for start, stop in ((1, 512), (0, 513), (512, 512)):
            with self.assertRaises(ValueError):
                CacheBlockFeatures(reader, block_reads=512, split_start=start, split_stop=stop)

    def test_real_training_cache_matches_direct_full_block_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fastq = root / "reads.fastq"
            fastq.write_bytes(b"".join(
                b"@read\nAAAA\n+\n" + bytes([33 + q] * 4) + b"\n"
                for q in (0, 10, 20, 30, 41, 35)
            ))
            cache = root / "reads.h5"
            create_training_cache(fastq, cache)
            with TrainingCacheReader(cache) as reader:
                train = CacheBlockFeatures(reader, block_reads=4, split_stop=4)
                validation = CacheBlockFeatures(reader, block_reads=4, split_start=4)
                self.assertEqual(train.payload_for_block(0), encode_block_features(reader.read_range(0, 4)))
                self.assertEqual(validation.payload_for_block(1), encode_block_features(reader.read_range(4, 6)))
                with self.assertRaises(IndexError):
                    train.payload_for_block(1)
                with self.assertRaises(IndexError):
                    validation.payload_for_block(0)


if __name__ == "__main__":
    unittest.main()
