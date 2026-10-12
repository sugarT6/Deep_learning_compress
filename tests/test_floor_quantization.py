import tempfile
from pathlib import Path
import unittest
from unittest import mock

import numpy as np
import torch

from codec.probability_quantization import floor_frequencies, logits_to_intervals, logits_to_cdfs
from codec.range_encoder import RangeEncoder
from codec.range_decoder import RangeDecoder
from codec.encode import encode_fastq
from codec.decode import decode_fastq
from codec.container import read_container
from test_head_adapter import model_and_checkpoint, source_file, config


class FloorQuantizationTest(unittest.TestCase):
    def test_workspace_matches_original_v2_across_chunks_and_layouts(self):
        rng = np.random.default_rng(12102026)
        original = rng.normal(size=(8195, 42)).astype(np.float32) * 6
        symbols = rng.integers(0, 42, len(original), dtype=np.uint64)
        for logits in (original, original.astype(np.float64),
                       np.asfortranarray(original), original[::-1, ::-1]):
            before = logits.copy()
            logits.flags.writeable = False
            for cap in (42, 65536, 1 << 30):
                # Independent expression from the original v2 protocol, rather
                # than comparing two users of the new workspace helper.
                matrix = np.asarray(logits, dtype=np.float64)
                weights = np.exp(matrix - matrix.max(axis=1, keepdims=True))
                weights /= weights.sum(axis=1, dtype=np.float64, keepdims=True)
                weights *= float(cap - 42)
                expected = np.floor(weights).astype(np.int64) + 1
                self.assertTrue(np.array_equal(floor_frequencies(logits, total=cap), expected))
                prefix = np.cumsum(expected, axis=1, dtype=np.int64)
                upper = prefix[np.arange(len(symbols)), symbols]
                lower = upper - expected[np.arange(len(symbols)), symbols]
                totals = prefix[:, -1]
                for chunk in (1024, 2048, 4096, 8192, 16384):
                    actual = logits_to_intervals(logits, symbols, total=cap, chunk_rows=chunk)
                    for got, wanted in zip(actual, (lower, upper, totals)):
                        self.assertTrue(np.array_equal(got, wanted), (cap, chunk))
            self.assertTrue(np.array_equal(logits, before))

    def test_late_chunk_nonfinite_and_invalid_chunk_sizes(self):
        logits = np.zeros((257, 42), dtype=np.float32)
        logits[-1, -1] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            logits_to_intervals(logits, np.zeros(len(logits), dtype=int), chunk_rows=128)
        for chunk in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                logits_to_intervals(np.zeros((1, 42)), np.array([0]), chunk_rows=chunk)

    def test_integer_intervals_match_full_and_scalar_cdfs(self):
        rng = np.random.default_rng(19)
        logits = rng.normal(size=(213, 42)) * 8
        logits[0] = 0
        logits[1] = -10000
        logits[1, 41] = 10000
        symbols = rng.integers(0, 42, size=len(logits))
        for cap in (42, 43, 65536, 1 << 30):
            cdfs = logits_to_cdfs(logits, total=cap, version=2)
            for chunk in (1, 17, 4096):
                lo, hi, totals = logits_to_intervals(logits, symbols, total=cap, chunk_rows=chunk)
                self.assertTrue(np.array_equal(lo, cdfs[np.arange(len(logits)), symbols]))
                self.assertTrue(np.array_equal(hi, cdfs[np.arange(len(logits)), symbols+1]))
                self.assertTrue(np.array_equal(totals, cdfs[:, -1]))
                self.assertTrue(np.all((totals >= 42) & (totals <= cap)))
            for row, cdf in zip(logits[:3], cdfs[:3]):
                w = np.exp(row-row.max())
                f = 1 + np.floor((w/w.sum()) * (cap-42)).astype(np.int64)
                self.assertTrue(np.array_equal(np.diff(cdf), f))
        self.assertEqual(logits_to_cdfs(np.empty((0, 42)), version=2).shape, (0, 43))
        self.assertEqual(len(logits_to_intervals(np.empty((0, 42)), np.empty(0, dtype=int))[0]), 0)

    def test_direct_encoder_matches_reference_variable_total_stream(self):
        rng = np.random.default_rng(2026)
        scores = rng.normal(size=(2000, 42)) * 3
        y = rng.integers(0, 42, len(scores))
        cdfs = logits_to_cdfs(scores, version=2)
        direct, reference = RangeEncoder(), RangeEncoder()
        lo, hi, totals = logits_to_intervals(scores, y)
        for start in range(0, len(y), 113):
            direct.encode_prevalidated_intervals(lo[start:start+113], hi[start:start+113], totals[start:start+113])
        for symbol, cdf in zip(y, cdfs):
            reference.encode(int(symbol), cdf)
        self.assertEqual(direct.finish(), reference.finish())
        decoder = RangeDecoder(direct.finish())
        self.assertEqual(decoder.decode_prevalidated_batch(cdfs), y.tolist())
        decoder.finish()

    def test_floor_roundtrip_adaptation_and_no_production_full_cdf(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, checkpoint = model_and_checkpoint(root, num_layers=4)
            source, raw = source_file(root)
            kw = dict(device=torch.device("cpu"), batch_reads=8, progress=False)
            artifact = root / "adapter.json"
            target = root / "floor.fqdc"
            stats = encode_fastq(source, target, checkpoint, **kw, quantization_version=2, verify_cdf=True,
                head_adaptation_config=config(head_type="residual", cross_layer=True), save_head_adapter_path=artifact)
            self.assertTrue(stats.head_adaptation["accepted"])
            self.assertEqual(stats.head_adaptation["config"]["quantization_version"], 2)
            self.assertEqual(read_container(target).metadata["probability_quantization"]["version"], 2)
            reused = root / "reused.fqdc"
            with mock.patch("codec.encode.logits_to_cdfs", side_effect=AssertionError("full CDF in production")):
                encode_fastq(source, reused, checkpoint, **kw, quantization_version=2, head_adapter_path=artifact)
            self.assertEqual(target.read_bytes(), reused.read_bytes())
            restored = root / "restored.fq"
            decode_fastq(target, restored, checkpoint, **kw, verify_cdf=True)
            self.assertEqual(restored.read_bytes(), raw)

    def test_invalid_inputs(self):
        for scores in (np.zeros((3, 41)), np.full((3, 42), np.nan), np.full((3, 42), np.inf)):
            with self.assertRaises(ValueError):
                floor_frequencies(scores)
        for symbols in (np.array([42]), np.array([-1]), np.array([1.0])):
            with self.assertRaises(ValueError):
                logits_to_intervals(np.zeros((1, 42)), symbols)
        with self.assertRaises(ValueError):
            logits_to_cdfs(np.zeros((1, 42)), version=3)


if __name__ == "__main__":
    unittest.main()
