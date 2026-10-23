from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from codec.prediction_report import PredictionReport


class PredictionReportTest(unittest.TestCase):
    def test_accuracy_uses_entire_stream_but_preview_stops_at_one_thousand(self):
        report = PredictionReport()
        first = np.zeros((600, 42), dtype=np.float32)
        first[:, 17] = 1.0
        second = np.zeros((800, 42), dtype=np.float32)
        second[:, 3] = 2.0
        report.observe(first, np.full(600, 17, dtype=np.uint8))
        report.observe(second, np.full(800, 2, dtype=np.uint8))
        self.assertEqual(report.total, 1400)
        self.assertEqual(report.correct, 600)
        self.assertAlmostEqual(report.accuracy, 600 / 1400)
        self.assertEqual(report.positions_recorded, 1000)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "prediction.txt"
            report.write(target)
            lines = target.read_text().splitlines()
        self.assertEqual(len(lines), 1001)
        self.assertEqual(
            lines[0], "prediction_accuracy=42.857143% correct=600 total=1400"
        )
        self.assertTrue(lines[600].endswith(" yes"))
        self.assertFalse(lines[601].endswith(" yes"))
        self.assertEqual(lines[1].split()[0], "17")
        self.assertEqual(lines[601].split()[0], "3")

    def test_softmax_top_five_ties_input_order_and_readonly_arrays(self):
        logits = np.zeros((3, 42), dtype=np.float64)
        logits[1] = np.linspace(-20.0, 21.0, 42)
        logits[2] = -10000.0
        logits[2, 41] = 10000.0
        symbols = np.array([0, 40, 41], dtype=np.int64)
        before_logits, before_symbols = logits.copy(), symbols.copy()
        logits.flags.writeable = False
        symbols.flags.writeable = False
        report = PredictionReport()
        report.observe(logits, symbols)
        self.assertEqual(report.correct, 2)
        self.assertTrue(np.array_equal(before_logits, logits))
        self.assertTrue(np.array_equal(before_symbols, symbols))
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "prediction.txt"
            report.write(target)
            lines = target.read_text().splitlines()[1:]
        first = lines[0].split()
        self.assertEqual(first[::2][:5], ["0", "1", "2", "3", "4"])
        self.assertTrue(lines[0].endswith(" yes"))
        second = lines[1].split()
        self.assertEqual(second[::2], ["41", "40", "39", "38", "37"])
        expected = np.exp(before_logits[1] - before_logits[1].max())
        expected /= expected.sum()
        self.assertTrue(
            np.allclose([float(value) for value in second[1::2]], expected[-1:-6:-1])
        )
        self.assertTrue(lines[2].startswith("41 1 "))
        self.assertTrue(lines[2].endswith(" yes"))

    def test_extreme_finite_scores_and_underflow_preserve_top_five_order(self):
        scores = np.full((1, 42), -np.finfo(np.float64).max)
        scores[0, 41] = np.finfo(np.float64).max
        scores[0, 4] = -1e20
        scores[0, 3] = -1e30
        report = PredictionReport()
        with np.errstate(all="raise"):
            report.observe(scores, np.array([41]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.txt"
            report.write(path)
            tokens = path.read_text().splitlines()[1].split()
        self.assertEqual(tokens[::2][:5], ["41", "4", "3", "0", "1"])

    def test_empty_accuracy_and_zero_preview_are_supported(self):
        report = PredictionReport(max_positions=0)
        report.observe(np.empty((0, 42)), np.empty(0, dtype=np.uint8))
        self.assertIsNone(report.accuracy)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.txt"
            report.write(path)
            self.assertEqual(
                path.read_text(), "prediction_accuracy=N/A correct=0 total=0\n"
            )
        report.observe(np.zeros((1, 42)), np.array([0]))
        self.assertEqual(report.correct, 1)
        self.assertEqual(report.positions_recorded, 0)

    def test_invalid_observations_leave_previous_counts_unchanged(self):
        invalid = [
            (np.zeros((2, 41)), np.zeros(2, dtype=int)),
            (np.zeros((42,)), np.array([0])),
            (np.full((1, 42), np.nan), np.array([0])),
            (np.full((1, 42), np.inf), np.array([0])),
            (np.ones((1, 42), dtype=complex), np.array([0])),
            (np.full((1, 42), "0"), np.array([0])),
            (np.zeros((1, 42)), np.array([0, 1])),
            (np.zeros((1, 42)), np.array([0.0])),
            (np.zeros((1, 42)), np.array([True])),
            (np.zeros((1, 42)), np.array([-1])),
            (np.zeros((1, 42)), np.array([42])),
        ]
        report = PredictionReport()
        report.observe(np.zeros((1, 42)), np.array([0]))
        for scores, symbols in invalid:
            with self.subTest(shape=scores.shape, dtype=scores.dtype, symbols=symbols):
                with self.assertRaises(ValueError):
                    report.observe(scores, symbols)
                self.assertEqual((report.total, report.correct, report.positions_recorded), (1, 1, 1))
        for value in (-1, 1001, 1.5, True, np.bool_(False), "1000"):
            with self.subTest(max_positions=value):
                with self.assertRaises(ValueError):
                    PredictionReport(value)

    def test_write_refuses_existing_file_and_cleans_failed_new_file(self):
        report = PredictionReport()
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "existing.txt"
            target.write_text("keep me")
            with self.assertRaises(FileExistsError):
                report.write(target)
            self.assertEqual(target.read_text(), "keep me")

            failing_target = Path(directory) / "partial.txt"
            original_open = Path.open

            def failing_open(path, *args, **kwargs):
                real_stream = original_open(path, *args, **kwargs)
                manager = mock.MagicMock()
                manager.__enter__.return_value.write.side_effect = OSError("disk full")
                manager.__exit__.side_effect = lambda *args: real_stream.close()
                return manager

            with mock.patch.object(Path, "open", failing_open):
                with self.assertRaisesRegex(OSError, "disk full"):
                    report.write(failing_target)
            self.assertFalse(failing_target.exists())


if __name__ == "__main__":
    unittest.main()
