import math
import unittest

import numpy as np

from codec import (
    RANGE_STREAM_HEADER_BYTES,
    InvalidCDFError,
    InvalidRangeStreamError,
    RangeCodingError,
    TruncatedRangeStreamError,
)
from codec.probability_quantization import (
    TOTAL,
    probabilities_to_cdf,
    quantized_symbol_bits,
)
from codec.range_decoder import RangeDecoder
from codec.range_encoder import RangeEncoder


def round_trip(symbols, cdfs):
    encoder = RangeEncoder()
    for symbol, cdf in zip(symbols, cdfs):
        encoder.encode(symbol, cdf)
    stream = encoder.finish()

    decoder = RangeDecoder(stream)
    decoded = [decoder.decode(cdf) for cdf in cdfs]
    decoder.finish()
    return decoded, stream, encoder.metadata


class RangeCoderTest(unittest.TestCase):
    def test_empty_stream_has_explicit_zero_symbol_termination(self):
        encoder = RangeEncoder()
        stream = encoder.finish()
        self.assertEqual(stream, encoder.finish())
        self.assertEqual(len(stream), RANGE_STREAM_HEADER_BYTES)
        self.assertEqual(encoder.metadata.symbol_count, 0)
        self.assertEqual(encoder.metadata.payload_bit_count, 0)

        decoder = RangeDecoder(stream)
        self.assertTrue(decoder.done)
        decoder.finish()
        with self.assertRaises(RangeCodingError):
            decoder.decode((0, 1))

    def test_q0_q41_and_all_42_quality_ids_round_trip(self):
        cdf = probabilities_to_cdf(np.ones(42))
        symbols = [0, 41] + list(range(42)) + [41, 0]
        decoded, _, metadata = round_trip(symbols, [cdf] * len(symbols))
        self.assertEqual(decoded, symbols)
        self.assertEqual(metadata.symbol_count, len(symbols))

    def test_single_symbol_and_long_repeated_sequence_round_trip(self):
        symbols = [0] * 20000
        decoded, stream, metadata = round_trip(symbols, [(0, 1)] * len(symbols))
        self.assertEqual(decoded, symbols)
        self.assertEqual(metadata.payload_bit_count, 2)
        self.assertEqual(len(stream), RANGE_STREAM_HEADER_BYTES + 1)

    def test_random_symbols_with_random_per_position_cdfs_round_trip(self):
        rng = np.random.default_rng(7719)
        symbols = rng.integers(0, 42, size=1500).tolist()
        cdfs = []
        for _ in symbols:
            frequencies = rng.integers(1, 500, size=42, dtype=np.int64)
            cdfs.append((0,) + tuple(np.cumsum(frequencies).tolist()))

        decoded, _, _ = round_trip(symbols, cdfs)
        self.assertEqual(decoded, symbols)
        self.assertGreater(len(set(cdfs[:50])), 45)

    def test_adaptive_non_neural_quality_context_closes_the_loop(self):
        rng = np.random.default_rng(915)
        symbols = [0, 41] + list(range(42)) + rng.integers(0, 42, size=2000).tolist()

        encoder = RangeEncoder()
        encode_counts = np.ones(42, dtype=np.float64)
        theoretical_bits = 0.0
        for symbol in symbols:
            cdf = probabilities_to_cdf(encode_counts)
            theoretical_bits += quantized_symbol_bits(symbol, cdf)
            encoder.encode(symbol, cdf)
            encode_counts[symbol] += 1.0
        stream = encoder.finish()

        decoder = RangeDecoder(stream)
        decode_counts = np.ones(42, dtype=np.float64)
        decoded = []
        for _ in symbols:
            cdf = probabilities_to_cdf(decode_counts)
            symbol = decoder.decode(cdf)
            decoded.append(symbol)
            decode_counts[symbol] += 1.0
        decoder.finish()

        self.assertEqual(decoded, symbols)
        self.assertEqual(encode_counts.tolist(), decode_counts.tolist())
        payload_gap = encoder.metadata.payload_bit_count - theoretical_bits
        serialized_gap = len(stream) * 8 - theoretical_bits
        statistics = {
            "theoretical_quantized_bits": theoretical_bits,
            "payload_bits": encoder.metadata.payload_bit_count,
            "serialized_bits": len(stream) * 8,
            "payload_minus_theoretical_bits": payload_gap,
            "serialized_minus_theoretical_bits": serialized_gap,
        }
        self.assertTrue(all(math.isfinite(value) for value in statistics.values()))
        self.assertLess(abs(payload_gap), 16.0)
        self.assertGreaterEqual(
            serialized_gap, RANGE_STREAM_HEADER_BYTES * 8 - 16.0
        )
        self.assertLess(
            serialized_gap, RANGE_STREAM_HEADER_BYTES * 8 + 24.0
        )

    def test_finalization_and_decode_count_are_enforced(self):
        cdf = probabilities_to_cdf(np.ones(42))
        encoder = RangeEncoder()
        encoder.encode(7, cdf)
        stream = encoder.finish()
        with self.assertRaises(RangeCodingError):
            encoder.encode(8, cdf)

        decoder = RangeDecoder(stream)
        with self.assertRaises(RangeCodingError):
            decoder.finish()
        self.assertEqual(decoder.decode(cdf), 7)
        decoder.finish()
        with self.assertRaises(RangeCodingError):
            decoder.decode(cdf)

    def test_truncated_corrupt_and_trailing_streams_are_rejected(self):
        cdf = probabilities_to_cdf(np.ones(42))
        encoder = RangeEncoder()
        for symbol in range(42):
            encoder.encode(symbol, cdf)
        stream = encoder.finish()

        for truncated in (stream[:0], stream[:10], stream[:-1]):
            with self.subTest(length=len(truncated)):
                with self.assertRaises(TruncatedRangeStreamError):
                    RangeDecoder(truncated)
        with self.assertRaises(InvalidRangeStreamError):
            RangeDecoder(stream + b"\x00")

        corrupt = bytearray(stream)
        corrupt[-1] ^= 0x80
        with self.assertRaises(InvalidRangeStreamError):
            RangeDecoder(bytes(corrupt))

    def test_invalid_cdfs_and_symbols_are_rejected(self):
        invalid_cdfs = (
            (),
            (0,),
            (1, 2),
            (0, 1, 1),
            (0, 2, 1),
            (0, 1.5, 2),
            (0, (1 << 30) + 1),
        )
        for cdf in invalid_cdfs:
            with self.subTest(cdf=cdf):
                with self.assertRaises(InvalidCDFError):
                    RangeEncoder().encode(0, cdf)

        with self.assertRaises(RangeCodingError):
            RangeEncoder().encode(-1, (0, 1))
        with self.assertRaises(RangeCodingError):
            RangeEncoder().encode(1, (0, 1))

        encoder = RangeEncoder()
        encoder.encode(0, (0, 1))
        decoder = RangeDecoder(encoder.finish())
        with self.assertRaises(InvalidCDFError):
            decoder.decode((0, 0))


if __name__ == "__main__":
    unittest.main()
