import unittest

import numpy as np

from codec.encode_fastpath import fuse_batch_logits, selected_quantized_bits
from codec.online_prior import OnlinePriorConfig, OnlinePriorState
from codec.probability_quantization import TOTAL, logits_to_cdfs, quantized_symbols_bits
from codec.range_encoder import RangeEncoder
from codec.range_decoder import RangeDecoder


class EncodeFastpathTest(unittest.TestCase):
    def test_frozen_context_cache_matches_reference_across_batches(self):
        rng = np.random.default_rng(719)
        for width in (1, 8, 300):
            state = OnlinePriorState(OnlinePriorConfig(cycle_bin_width=width))
            for _ in range(4):
                lengths = rng.integers(1, 151, size=65)
                mask = np.arange(150)[None, :] < lengths[:, None]
                qualities = rng.integers(0, 42, mask.shape)
                qualities[~mask] = 42
                cycles, rows = np.nonzero(mask.T)
                previous = np.full(cycles.size, 42)
                later = cycles > 0
                previous[later] = qualities[rows[later], cycles[later] - 1]
                logits = rng.normal(size=(cycles.size, 42)).astype(np.float32)
                expected = state.fuse_logits(logits, previous, cycles)
                actual = fuse_batch_logits(state, logits, previous, cycles)
                np.testing.assert_array_equal(actual, expected)
                cdfs = logits_to_cdfs(actual)
                np.testing.assert_array_equal(cdfs, logits_to_cdfs(expected))
                symbols = qualities[rows, cycles]
                self.assertEqual(selected_quantized_bits(symbols, cdfs, TOTAL),
                    quantized_symbols_bits(symbols, cdfs))
                before = state.observed_symbols
                fuse_batch_logits(state, logits, previous, cycles)
                self.assertEqual(state.observed_symbols, before)
                state.update_batch(qualities, mask)

    def test_range_local_state_matches_scalar_across_mixed_calls(self):
        rng = np.random.default_rng(20)
        for size in (0, 1, 63, 64, 65, 255, 256, 257, 10000):
            symbols = rng.integers(0, 42, size=size)
            cdfs = logits_to_cdfs(rng.normal(size=(size, 42)) * 4)
            fast, reference = RangeEncoder(), RangeEncoder()
            for start in range(0, size, 64):
                stop = min(start + 64, size)
                fast.encode_prevalidated_batch(symbols[start:stop], cdfs[start:stop], total=TOTAL)
                for i in range(start, stop):
                    reference.encode(int(symbols[i]), cdfs[i])
                # Mix scalar and batch calls, exercising flushed local bit state.
                fast.encode(0, (0, 1, 3))
                reference.encode(0, (0, 1, 3))
            self.assertEqual(fast.finish(), reference.finish())
            decoder = RangeDecoder(fast.finish())
            for start in range(0, size, 64):
                stop = min(start + 64, size)
                self.assertEqual(decoder.decode_prevalidated_batch(cdfs[start:stop], total=TOTAL),
                    symbols[start:stop].tolist())
                self.assertEqual(decoder.decode((0, 1, 3)), 0)
            decoder.finish()
