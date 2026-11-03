from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from codec.prediction_report import PredictionReport


class PredictionReportTest(unittest.TestCase):
    def _lines(self, report):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "prediction.txt"
            report.write(target)
            return target.read_text(encoding="utf-8").splitlines()

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
        self.assertEqual(len(lines), 1008)
        self.assertEqual(
            lines[0], "prediction_accuracy=42.857143% correct=600 total=1400"
        )
        self.assertEqual(lines[1:8], [
            "rank1 42.857143%", "rank2 0.000000%", "rank3 0.000000%",
            "rank4 57.142857%", "rank5 0.000000%", "rank6 0.000000%",
            "rank其它 0.000000%",
        ])
        self.assertTrue(lines[607].endswith(" yes"))
        self.assertFalse(lines[608].endswith(" yes"))
        self.assertEqual(lines[8].split()[0], "17")
        self.assertEqual(lines[608].split()[0], "3")

    def test_rank_categories_are_individual_not_cumulative(self):
        report = PredictionReport()
        # Equal scores make Q id + 1 the rank. Include both rank 7 and
        # rank 42 in the final bucket and distinct counts for ranks 4--6.
        symbols = np.repeat(np.array([0, 1, 2, 3, 4, 5, 6, 41]), [28, 21, 17, 13, 9, 7, 3, 2])
        report.observe(np.zeros((100, 42)), symbols)
        lines = self._lines(report)
        self.assertEqual(lines[0], "prediction_accuracy=28.000000% correct=28 total=100")
        self.assertEqual(lines[1:8], [
            "rank1 28.000000%", "rank2 21.000000%", "rank3 17.000000%",
            "rank4 13.000000%", "rank5 9.000000%", "rank6 7.000000%",
            "rank其它 5.000000%",
        ])

    def test_rank_ties_choose_smaller_quality_id_before_lower_scores(self):
        report = PredictionReport()
        scores = np.zeros((7, 42))
        scores[:, [5, 6, 20]] = 7.0
        report.observe(scores, np.array([5, 6, 20, 0, 1, 2, 3]))
        self.assertEqual(self._lines(report)[1:8], [
            "rank1 14.285714%", "rank2 14.285714%", "rank3 14.285714%",
            "rank4 14.285714%", "rank5 14.285714%", "rank6 14.285714%",
            "rank其它 14.285714%",
        ])

    def test_all_rank_categories_include_batches_after_preview_is_full(self):
        report = PredictionReport()
        report.observe(np.zeros((1000, 42)), np.zeros(1000, dtype=np.int64))
        report.observe(np.zeros((40, 42)), np.ones(40, dtype=np.int64))
        report.observe(np.zeros((60, 42)), np.full(60, 2, dtype=np.int64))
        report.observe(np.zeros((80, 42)), np.full(80, 3, dtype=np.int64))
        report.observe(np.zeros((100, 42)), np.full(100, 4, dtype=np.int64))
        report.observe(np.zeros((120, 42)), np.full(120, 5, dtype=np.int64))
        report.observe(np.zeros((150, 42)), np.full(150, 6, dtype=np.int64))
        report.observe(np.zeros((450, 42)), np.full(450, 41, dtype=np.int64))
        lines = self._lines(report)
        self.assertEqual((report.total, report.correct, report.positions_recorded), (2000, 1000, 1000))
        self.assertEqual(len(lines), 1008)
        self.assertEqual(lines[1:8], [
            "rank1 50.000000%", "rank2 2.000000%", "rank3 3.000000%",
            "rank4 4.000000%", "rank5 5.000000%", "rank6 6.000000%",
            "rank其它 30.000000%",
        ])

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
            lines = target.read_text().splitlines()[8:]
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
            tokens = path.read_text().splitlines()[8].split()
        self.assertEqual(tokens[::2][:5], ["41", "4", "3", "0", "1"])

    def test_underflow_does_not_turn_distinct_scores_into_rank_ties(self):
        report = PredictionReport(max_positions=0)
        scores = np.full((7, 42), -np.finfo(np.float64).max)
        scores[:, 41] = np.finfo(np.float64).max
        scores[:, 4] = -1e20
        scores[:, 3] = -1e30
        # Rank uses original logits, even though every nonmaximum softmax
        # probability underflows: Q41, Q4, Q3, Q0, Q1, Q2, Q5 (rank 7).
        report.observe(scores, np.array([41, 4, 3, 0, 1, 2, 5]))
        self.assertEqual(self._lines(report)[1:8], [
            "rank1 14.285714%", "rank2 14.285714%", "rank3 14.285714%",
            "rank4 14.285714%", "rank5 14.285714%", "rank6 14.285714%",
            "rank其它 14.285714%",
        ])

    def test_empty_accuracy_and_zero_preview_are_supported(self):
        report = PredictionReport(max_positions=0)
        report.observe(np.empty((0, 42)), np.empty(0, dtype=np.uint8))
        self.assertIsNone(report.accuracy)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.txt"
            report.write(path)
            self.assertEqual(
                path.read_text(), "prediction_accuracy=N/A correct=0 total=0\n"
                "rank1 N/A\nrank2 N/A\nrank3 N/A\nrank4 N/A\n"
                "rank5 N/A\nrank6 N/A\nrank其它 N/A\n"
            )
        report.observe(np.zeros((1, 42)), np.array([0]))
        self.assertEqual(report.correct, 1)
        self.assertEqual(report.positions_recorded, 0)
        self.assertEqual(self._lines(report)[1:8], [
            "rank1 100.000000%", "rank2 0.000000%", "rank3 0.000000%",
            "rank4 0.000000%", "rank5 0.000000%", "rank6 0.000000%",
            "rank其它 0.000000%",
        ])

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
        before = self._lines(report)
        for scores, symbols in invalid:
            with self.subTest(shape=scores.shape, dtype=scores.dtype, symbols=symbols):
                with self.assertRaises(ValueError):
                    report.observe(scores, symbols)
                self.assertEqual((report.total, report.correct, report.positions_recorded), (1, 1, 1))
                self.assertEqual(self._lines(report), before)
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
