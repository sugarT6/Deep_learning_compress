"""Wire compatibility and boundary cases for the optional C++ batch coder."""

from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from codec._native_range import NativeRangeUnavailable, load_native_kernel
from codec._range_common import InvalidCDFError, RangeCodingError
from codec.decode import decode_fastq
from codec.encode import encode_fastq
from codec.probability_quantization import logits_to_cdfs
from codec.range_decoder import RangeDecoder
from codec.range_encoder import RangeEncoder
from test_head_adapter import model_and_checkpoint, source_file


class BackendSelectionTest(unittest.TestCase):
    def test_missing_native_is_explicit_and_auto_can_fall_back(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "not-built.so"
            with mock.patch("codec._native_range.native_library_path", return_value=missing):
                self.assertEqual(RangeEncoder().backend, "python")
                with self.assertRaisesRegex(NativeRangeUnavailable, "codec.build_native"):
                    RangeEncoder(backend="native")
        with mock.patch("codec._native_range.load_native_kernel", side_effect=AssertionError):
            self.assertEqual(RangeEncoder(backend="python").backend, "python")
        with self.assertRaises(RangeCodingError):
            RangeEncoder(backend="invalid")


class NativeRangeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            load_native_kernel()
        except NativeRangeUnavailable as exc:
            raise unittest.SkipTest(str(exc))

    def test_mixed_scalar_cdf_and_interval_calls_match_reference(self):
        rng = np.random.default_rng(20261012)
        for cap in (42, 65536, 1 << 30):
            with self.subTest(cap=cap):
                symbols = rng.integers(0, 42, 2049)
                cdfs = logits_to_cdfs(rng.normal(size=(2049, 42)) * 8, total=cap, version=2)
                native, reference = RangeEncoder(backend="native"), RangeEncoder(backend="python")
                for start in range(0, len(symbols), 113):
                    stop = min(start + 113, len(symbols))
                    y, cdf = symbols[start:stop], cdfs[start:stop]
                    indices = np.arange(len(y))
                    lo, hi, totals = cdf[indices, y], cdf[indices, y + 1], cdf[:, -1]
                    native.encode_prevalidated_intervals(lo, hi, totals)
                    reference.encode_prevalidated_intervals(lo, hi, totals)
                    # Public scalar updates must see the exact native state.
                    native.encode(1, (0, 1, 3, 4))
                    reference.encode(1, (0, 1, 3, 4))
                    fixed = logits_to_cdfs(rng.normal(size=(len(y), 42)), total=cap)
                    native.encode_prevalidated_batch(y, fixed, total=cap)
                    for symbol, row in zip(y, fixed):
                        reference.encode(int(symbol), row)
                self.assertEqual(native.finish(), reference.finish())
                self.assertEqual(native.metadata, reference.metadata)
                self.assertEqual(native.finish(), native.finish())
                with self.assertRaises(RangeCodingError):
                    native.encode_prevalidated_intervals(lo, hi, totals)
                with self.assertRaises(RangeCodingError):
                    native.encode_prevalidated_batch(y, fixed, total=cap)

    def test_long_underflow_flush_and_smallest_intervals(self):
        native, reference = RangeEncoder(backend="native"), RangeEncoder(backend="python")
        # Every middle-half symbol defers a bit across the batch boundary.
        lower = np.full(20000, 1, dtype=np.int64)
        upper, totals = lower * 3, lower * 4
        for encoder in (native, reference):
            encoder.encode_prevalidated_intervals(lower, upper, totals)
            self.assertEqual(encoder._pending_underflow_bits, 20000)
            encoder.encode_prevalidated_intervals(
                np.array([0]), np.array([1]), np.array([2]))
            encoder.encode_prevalidated_intervals(
                np.zeros(1024, dtype=np.int64), np.ones(1024, dtype=np.int64),
                np.full(1024, 1 << 30, dtype=np.int64))
        self.assertEqual(native.finish(), reference.finish())
        decoder = RangeDecoder(native.finish())
        for _ in range(20000):
            self.assertEqual(decoder.decode((0, 1, 3, 4)), 1)
        self.assertEqual(decoder.decode((0, 1, 2)), 0)
        for _ in range(1024):
            self.assertEqual(decoder.decode((0, 1, 1 << 30)), 0)
        decoder.finish()

    def test_readonly_strided_and_byte_swapped_arrays(self):
        rng = np.random.default_rng(71)
        symbols = rng.integers(0, 42, 513)
        cdfs = logits_to_cdfs(rng.normal(size=(513, 42)))
        expected = RangeEncoder(backend="python")
        expected.encode_prevalidated_batch(symbols, cdfs, total=65536)
        for matrix in (np.asfortranarray(cdfs), cdfs.astype(">i8"), cdfs[::-1][::-1]):
            matrix.setflags(write=False)
            native = RangeEncoder(backend="native")
            native.encode_prevalidated_batch(symbols, matrix, total=65536)
            self.assertEqual(native.finish(), expected.finish())
        rows = np.arange(len(symbols))
        arrays = (cdfs[rows, symbols], cdfs[rows, symbols + 1], cdfs[:, -1])
        native, reference = RangeEncoder(backend="native"), RangeEncoder(backend="python")
        for encoder in (native, reference):
            encoder.encode_prevalidated_intervals(*(a[::-2] for a in arrays))
        self.assertEqual(native.finish(), reference.finish())

    def test_empty_and_rejected_batches_leave_state_usable(self):
        native = RangeEncoder(backend="native")
        native.encode_prevalidated_intervals(*(np.empty(0, dtype=np.int64) for _ in range(3)))
        native.encode_prevalidated_batch(np.empty(0, dtype=np.int64),
                                        np.empty((0, 43), dtype=np.int64), total=65536)
        self.assertEqual(native.finish(), RangeEncoder(backend="python").finish())
        native = RangeEncoder(backend="native")
        for arrays in (([0, -1], [1, 1], [2, 2]), ([0], [0], [2]),
                       ([0], [1], [(1 << 30) + 1])):
            with self.assertRaises(InvalidCDFError):
                native.encode_prevalidated_intervals(*(np.array(a) for a in arrays))
        for arrays in ((np.array([0]), np.array([1, 2]), np.array([2])),
                       (np.array([0.0]), np.array([1]), np.array([2]))):
            with self.assertRaises(RangeCodingError):
                native.encode_prevalidated_intervals(*arrays)
        with self.assertRaises(RangeCodingError):
            native.encode_prevalidated_batch(np.array([2]), np.array([[0, 1, 2]]), total=2)
        with self.assertRaises(RangeCodingError):
            native.encode_prevalidated_batch(np.array([0, 0]), np.array([[0, 1]]), total=1)
        with self.assertRaises(InvalidCDFError):
            native.encode_prevalidated_batch(np.array([0]), np.array([[0, 1]]), total=0)
        self.assertEqual(native.symbol_count, 0)
        native.encode(0, (0, 1, 2))
        reference = RangeEncoder(backend="python")
        reference.encode(0, (0, 1, 2))
        self.assertEqual(native.finish(), reference.finish())

    def test_complete_containers_match_and_existing_decoder_restores_fastq(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, checkpoint = model_and_checkpoint(root)
            source, original = source_file(root, count=19)
            for version in (1, 2):
                targets = []
                for backend in ("python", "native"):
                    target = root / f"{version}-{backend}.fqdc"
                    stats = encode_fastq(source, target, checkpoint,
                        device=torch.device("cpu"), batch_reads=8, progress=False,
                        quantization_version=version, coder_backend=backend)
                    self.assertEqual(stats.coder_backend, backend)
                    targets.append(target)
                self.assertEqual(targets[0].read_bytes(), targets[1].read_bytes())
                restored = root / f"{version}.fastq"
                decode_fastq(targets[1], restored, checkpoint,
                             device=torch.device("cpu"), batch_reads=8, progress=False)
                self.assertEqual(restored.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
