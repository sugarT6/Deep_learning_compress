import copy
import math
import unittest

import numpy as np

from codec.probability_quantization import logits_to_cdfs
from codec.rank_coding import (
    global_rank_bits,
    make_rank_metadata,
    quality_order,
    ranks_from_cdfs,
    validate_rank_metadata,
)
from codec.range_decoder import RangeDecoder
from codec.range_encoder import RangeEncoder


class RankCodingTest(unittest.TestCase):
    def test_empty_counts_have_deterministic_uniform_positive_frequencies(self):
        counts = [0] * 42
        metadata = make_rank_metadata(counts, 100)
        cdf = validate_rank_metadata(metadata, 100, 0)
        self.assertEqual(cdf.dtype, np.int64)
        np.testing.assert_array_equal(np.diff(cdf), [3] * 16 + [2] * 26)
        self.assertEqual(cdf[-1], 100)
        self.assertEqual(global_rank_bits(counts, cdf), 0.0)

    def test_single_rank_keeps_every_rank_encodable(self):
        counts = [0] * 42
        counts[17] = 1000
        cdf = validate_rank_metadata(make_rank_metadata(counts, 65536), 65536, 1000)
        frequencies = np.diff(cdf)
        self.assertEqual(frequencies[17], 65536 - 41)
        self.assertTrue(np.all(frequencies[np.arange(42) != 17] == 1))
        self.assertAlmostEqual(global_rank_bits(counts, cdf),
                               -1000 * math.log2((65536 - 41) / 65536))

    def test_largest_remainder_ties_prefer_smaller_rank(self):
        counts = [1] * 42
        cdf = validate_rank_metadata(make_rank_metadata(counts, 44), 44, 42)
        np.testing.assert_array_equal(np.diff(cdf), [2, 2] + [1] * 40)
        minimal = validate_rank_metadata(make_rank_metadata(counts, 42), 42, 42)
        np.testing.assert_array_equal(minimal, np.arange(43))

    def test_large_counts_are_apportioned_without_integer_product_overflow(self):
        counts = [(1 << 63) - 1, (1 << 63) - 2] + [0] * 40
        cdf = validate_rank_metadata(make_rank_metadata(counts, 65536), 65536, sum(counts))
        self.assertEqual(cdf[-1], 65536)
        self.assertEqual(cdf[1], 32748)
        self.assertEqual(cdf[2] - cdf[1], 32748)
        self.assertTrue(np.all(np.diff(cdf) > 0))

    def test_quality_order_uses_integer_frequency_then_quality_id(self):
        frequencies = np.ones((1, 42), dtype=np.int64)
        frequencies[0, [3, 7]] = 10
        cdfs = np.pad(np.cumsum(frequencies, axis=1), ((0, 0), (1, 0)))
        expected = [3, 7] + [quality for quality in range(42) if quality not in (3, 7)]
        np.testing.assert_array_equal(quality_order(cdfs)[0], expected)
        np.testing.assert_array_equal(ranks_from_cdfs(cdfs, np.array([7])), [1])
        self.assertEqual(ranks_from_cdfs(cdfs, np.array([7])).dtype, np.uint8)

    def test_quantized_ranks_round_trip_through_static_range_stream(self):
        rng = np.random.default_rng(29)
        for quantization_version in (1, 2):
            with self.subTest(quantization_version=quantization_version):
                logits = rng.normal(size=(500, 42))
                logits[0] = 0
                symbols = rng.integers(0, 42, size=500)
                neural_cdfs = logits_to_cdfs(logits, version=quantization_version)
                ranks = ranks_from_cdfs(neural_cdfs, symbols)
                counts = np.bincount(ranks, minlength=42)
                rank_cdf = validate_rank_metadata(make_rank_metadata(counts, 65536),
                                                  65536, len(symbols))
                encoder = RangeEncoder(backend="python")
                for rank in ranks:
                    encoder.encode(int(rank), rank_cdf)
                decoder = RangeDecoder(encoder.finish())
                decoded_ranks = np.array([decoder.decode(rank_cdf) for _ in ranks])
                decoder.finish()
                recovered = quality_order(neural_cdfs)[np.arange(len(ranks)), decoded_ranks]
                np.testing.assert_array_equal(recovered, symbols)
                self.assertLess(abs(encoder.metadata.payload_bit_count
                                    - global_rank_bits(counts, rank_cdf)), 3)

    def test_empty_neural_rows_produce_empty_rank_vector(self):
        cdfs = np.empty((0, 43), dtype=np.int64)
        self.assertEqual(quality_order(cdfs).shape, (0, 42))
        ranks = ranks_from_cdfs(cdfs, np.empty(0, dtype=np.int64))
        self.assertEqual(ranks.shape, (0,))
        self.assertEqual(ranks.dtype, np.uint8)

    def test_malformed_metadata_is_rejected(self):
        valid = make_rank_metadata([1] * 42, 65536)
        malformed = [None, [], {}, dict(valid, extra=1), dict(valid, format="unknown"),
                     dict(valid, order="raw_logits"), dict(valid, counts=tuple([1] * 42))]
        for bad_count in (True, np.bool_(False), -1, 1.0, "1", None, 1 << 63):
            bad = copy.deepcopy(valid)
            bad["counts"][0] = bad_count
            malformed.append(bad)
        for counts in ([1] * 41, [1] * 43, "x" * 42):
            malformed.append(dict(valid, counts=counts))
        for metadata in malformed:
            with self.subTest(metadata=metadata):
                with self.assertRaises(ValueError):
                    validate_rank_metadata(metadata, 65536, 42)
        for symbol_count in (True, np.bool_(False), -1, 41, 43, 42.0, "42"):
            with self.subTest(symbol_count=symbol_count):
                with self.assertRaises(ValueError):
                    validate_rank_metadata(valid, 65536, symbol_count)
        for total in (True, 41, 65536.0, (1 << 30) + 1):
            with self.subTest(total=total):
                with self.assertRaises(ValueError):
                    make_rank_metadata([1] * 42, total)
                with self.assertRaises(ValueError):
                    validate_rank_metadata(valid, total, 42)

    def test_malformed_cdfs_and_symbols_are_rejected(self):
        valid = np.arange(43, dtype=np.int64)[None, :]
        malformed = [np.arange(43), valid.astype(float), valid.astype(bool),
                     np.ones((1, 43), dtype=np.int64), valid + 1,
                     np.full((1, 43), (1 << 63), dtype=np.uint64)]
        for cdf in malformed:
            with self.subTest(cdf=cdf):
                with self.assertRaises(ValueError):
                    quality_order(cdf)
        for symbols in (np.array([-1]), np.array([42]), np.array([1.0]),
                        np.array([True]), np.array([0, 1]), np.array([[1]])):
            with self.subTest(symbols=symbols):
                with self.assertRaises(ValueError):
                    ranks_from_cdfs(valid, symbols)
        for cdf in (valid, valid[0].astype(float), np.arange(43) + 1):
            with self.subTest(cdf=cdf):
                with self.assertRaises(ValueError):
                    global_rank_bits([1] * 42, cdf)


if __name__ == "__main__":
    unittest.main()
