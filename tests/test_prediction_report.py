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

    def _observe(self, report, scores, symbols, positions=None):
        if positions is None:
            positions = np.ones(len(symbols), dtype=np.int64)
        report.observe(scores, symbols, positions=positions)

    def _probability_logits(self, probabilities):
        scores = np.full((len(probabilities), 42), -1000.0)
        for row, probability in enumerate(probabilities):
            if probability == 0:
                scores[row, 0], scores[row, 1] = -10000.0, 10000.0
            elif probability == 1:
                scores[row, 0], scores[row, 1] = 10000.0, -10000.0
            else:
                scores[row, 0] = np.log(probability)
                scores[row, 1] = np.log(1 - probability)
        return scores

    def test_accuracy_and_probability_statistics_use_entire_stream_without_preview(self):
        report = PredictionReport()
        first = np.zeros((600, 42), dtype=np.float32)
        first[:, 17] = 1.0
        second = np.zeros((800, 42), dtype=np.float32)
        second[:, 3] = 2.0
        self._observe(report, first, np.full(600, 17, dtype=np.uint8))
        self._observe(report, second, np.full(800, 2, dtype=np.uint8), np.full(800, 2))
        self.assertEqual((report.total, report.correct), (1400, 600))
        self.assertAlmostEqual(report.accuracy, 600 / 1400)
        self.assertEqual(report.probability_counts, (1400,) + (0,) * 9)
        self.assertEqual(report.low_probability_count, 1400)
        self.assertEqual(report.position_totals, (600, 800))
        self.assertEqual(report.low_probability_position_counts, (600, 800))
        lines = self._lines(report)
        self.assertEqual(len(lines), 22)
        self.assertEqual(lines[:8], [
            "prediction_accuracy=42.857143% correct=600 total=1400",
            "rank1 42.857143%", "rank2 0.000000%", "rank3 0.000000%",
            "rank4 57.142857%", "rank5 0.000000%", "rank6 0.000000%",
            "rank其它 0.000000%",
        ])
        self.assertEqual(lines[8], "[0.0,0.1) 100.000000%")
        self.assertEqual(lines[18], "low_probability=[0.0,0.3] count=1400 total=1400 proportion=100.000000%")
        self.assertEqual(lines[-2:], [
            "position1 42.857143% low=600 total=600 rate=100.000000%",
            "position2 57.142857% low=800 total=800 rate=100.000000%",
        ])

    def test_rank_categories_are_individual_not_cumulative(self):
        report = PredictionReport()
        symbols = np.repeat(np.array([0, 1, 2, 3, 4, 5, 6, 41]), [28, 21, 17, 13, 9, 7, 3, 2])
        self._observe(report, np.zeros((100, 42)), symbols)
        self.assertEqual(self._lines(report)[:8], [
            "prediction_accuracy=28.000000% correct=28 total=100",
            "rank1 28.000000%", "rank2 21.000000%", "rank3 17.000000%",
            "rank4 13.000000%", "rank5 9.000000%", "rank6 7.000000%",
            "rank其它 5.000000%",
        ])

    def test_rank_ties_choose_smaller_quality_id_before_lower_scores(self):
        report = PredictionReport()
        scores = np.zeros((7, 42))
        scores[:, [5, 6, 20]] = 7.0
        self._observe(report, scores, np.array([5, 6, 20, 0, 1, 2, 3]))
        self.assertEqual(self._lines(report)[1:8], [
            "rank1 14.285714%", "rank2 14.285714%", "rank3 14.285714%",
            "rank4 14.285714%", "rank5 14.285714%", "rank6 14.285714%",
            "rank其它 14.285714%",
        ])

    def test_all_rank_categories_include_batches_after_first_thousand(self):
        report = PredictionReport()
        for count, quality in ((1000, 0), (40, 1), (60, 2), (80, 3), (100, 4), (120, 5), (150, 6), (450, 41)):
            self._observe(report, np.zeros((count, 42)), np.full(count, quality, dtype=np.int64))
        self.assertEqual((report.total, report.correct), (2000, 1000))
        self.assertEqual(report.probability_counts, (2000,) + (0,) * 9)
        self.assertEqual(report.low_probability_count, 2000)
        self.assertEqual(self._lines(report)[1:8], [
            "rank1 50.000000%", "rank2 2.000000%", "rank3 3.000000%",
            "rank4 4.000000%", "rank5 5.000000%", "rank6 6.000000%",
            "rank其它 30.000000%",
        ])

    def test_probability_bins_use_true_quality_instead_of_prediction_maximum(self):
        report = PredictionReport()
        self._observe(report, self._probability_logits([0.15, 0.85, 0]), np.zeros(3, dtype=int))
        # All p_max values are >= 0.85, but two actual characters have low
        # probability, including one whose probability underflows.
        self.assertEqual(report.probability_counts, (1, 1, 0, 0, 0, 0, 0, 0, 1, 0))
        self.assertEqual(report.low_probability_count, 2)
        self.assertEqual(report.correct, 1)

    def test_all_ten_bins_include_zero_one_and_exact_point_three_boundary(self):
        probabilities = [0, 0.05, 0.15, 0.25, 0.3, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1]
        report = PredictionReport()
        self._observe(report, self._probability_logits(probabilities), np.zeros(len(probabilities), dtype=int))
        self.assertEqual(report.probability_counts, (2, 1, 1, 2, 1, 1, 1, 1, 1, 2))
        self.assertEqual(sum(report.probability_counts), 13)
        self.assertEqual(report.low_probability_count, 5)
        lines = self._lines(report)
        self.assertEqual(lines[8], "[0.0,0.1) 15.384615%")
        self.assertEqual(lines[11], "[0.3,0.4) 15.384615%")
        self.assertEqual(lines[17], "[0.9,1.0] 15.384615%")

    def test_low_probability_boundary_and_position_denominators(self):
        scores = self._probability_logits([0.3, 0.2999, 0.3001, 0.9])
        weights = np.exp(scores[0] - scores[0].max())
        self.assertEqual(weights[0] / weights.sum(), 0.3)
        report = PredictionReport()
        self._observe(report, scores, np.zeros(4, dtype=int), np.array([1, 2, 2, 3]))
        self.assertEqual(report.position_totals, (1, 2, 1))
        self.assertEqual(report.low_probability_position_counts, (1, 1, 0))
        self.assertEqual(report.low_probability_count, 2)
        self.assertEqual(self._lines(report)[18:], [
            "low_probability=[0.0,0.3] count=2 total=4 proportion=50.000000%",
            "low_probability_positions (1-based; share=position_low/all_low; rate=position_low/position_total)",
            "position1 50.000000% low=1 total=1 rate=100.000000%",
            "position2 50.000000% low=1 total=2 rate=50.000000%",
            "position3 0.000000% low=0 total=1 rate=0.000000%",
        ])

    def test_large_observation_and_later_batch_add_every_probability_and_position(self):
        report = PredictionReport()
        self._observe(report, self._probability_logits([0.15] * 9000), np.zeros(9000, dtype=int), np.ones(9000, dtype=int))
        self._observe(report, self._probability_logits([0.85] * 1100), np.zeros(1100, dtype=int), np.full(1100, 3))
        self.assertEqual(report.probability_counts, (0, 9000, 0, 0, 0, 0, 0, 0, 1100, 0))
        self.assertEqual(report.position_totals, (9000, 0, 1100))
        self.assertEqual(report.low_probability_position_counts, (9000, 0, 0))
        self.assertEqual(report.total, 10100)

    def test_softmax_and_readonly_arrays_are_not_modified(self):
        logits = np.zeros((3, 42), dtype=np.float64)
        logits[1] = np.linspace(-20.0, 21.0, 42)
        logits[2] = -10000.0
        logits[2, 41] = 10000.0
        symbols = np.array([0, 40, 41], dtype=np.int64)
        positions = np.array([3, 1, 2], dtype=np.int64)
        original = [array.copy() for array in (logits, symbols, positions)]
        for array in (logits, symbols, positions):
            array.flags.writeable = False
        report = PredictionReport()
        self._observe(report, logits, symbols, positions)
        for before, after in zip(original, (logits, symbols, positions)):
            np.testing.assert_array_equal(before, after)
        self.assertEqual(report.correct, 2)
        self.assertEqual(report.probability_counts, (1, 0, 1, 0, 0, 0, 0, 0, 0, 1))

    def test_extreme_finite_scores_do_not_change_raw_logit_rank_order(self):
        report = PredictionReport()
        scores = np.full((7, 42), -np.finfo(np.float64).max)
        scores[:, 41] = np.finfo(np.float64).max
        scores[:, 4] = -1e20
        scores[:, 3] = -1e30
        with np.errstate(all="raise"):
            self._observe(report, scores, np.array([41, 4, 3, 0, 1, 2, 5]))
        self.assertEqual(report.probability_counts, (6,) + (0,) * 8 + (1,))
        self.assertEqual(self._lines(report)[1:8], [
            "rank1 14.285714%", "rank2 14.285714%", "rank3 14.285714%",
            "rank4 14.285714%", "rank5 14.285714%", "rank6 14.285714%",
            "rank其它 14.285714%",
        ])

    def test_empty_report_has_na_bins_and_no_position_rows(self):
        report = PredictionReport()
        self._observe(report, np.empty((0, 42)), np.empty(0, dtype=np.uint8), np.empty(0, dtype=np.int64))
        self.assertIsNone(report.accuracy)
        self.assertEqual(report.probability_counts, (0,) * 10)
        self.assertEqual(report.position_totals, ())
        self.assertEqual(report.low_probability_count, 0)
        self.assertEqual(report.low_probability_position_counts, ())
        lines = self._lines(report)
        self.assertEqual(len(lines), 20)
        self.assertEqual(lines[:8], [
            "prediction_accuracy=N/A correct=0 total=0", "rank1 N/A", "rank2 N/A", "rank3 N/A",
            "rank4 N/A", "rank5 N/A", "rank6 N/A", "rank其它 N/A",
        ])
        self.assertTrue(all(line.endswith(" N/A") for line in lines[8:18]))
        self.assertEqual(lines[18], "low_probability=[0.0,0.3] count=0 total=0 proportion=N/A")

    def test_no_low_probability_has_na_share_but_defined_zero_rate(self):
        report = PredictionReport()
        self._observe(report, self._probability_logits([0.85, 0.95]), np.zeros(2, dtype=int), np.array([1, 2]))
        self.assertEqual(report.low_probability_count, 0)
        self.assertEqual(self._lines(report)[-2:], [
            "position1 N/A low=0 total=1 rate=0.000000%",
            "position2 N/A low=0 total=1 rate=0.000000%",
        ])

    def test_invalid_observations_leave_all_statistics_unchanged(self):
        invalid = [
            (np.zeros((2, 41)), np.zeros(2, dtype=int), np.ones(2, dtype=int)),
            (np.zeros((42,)), np.array([0]), np.array([1])),
            (np.full((1, 42), np.nan), np.array([0]), np.array([1])),
            (np.full((1, 42), np.inf), np.array([0]), np.array([1])),
            (np.ones((1, 42), dtype=complex), np.array([0]), np.array([1])),
            (np.full((1, 42), "0"), np.array([0]), np.array([1])),
            (np.zeros((1, 42)), np.array([0, 1]), np.array([1])),
            (np.zeros((1, 42)), np.array([0.0]), np.array([1])),
            (np.zeros((1, 42)), np.array([True]), np.array([1])),
            (np.zeros((1, 42)), np.array([-1]), np.array([1])),
            (np.zeros((1, 42)), np.array([42]), np.array([1])),
            (np.zeros((1, 42)), np.array([0]), np.array([1, 2])),
            (np.zeros((1, 42)), np.array([0]), np.array([0])),
            (np.zeros((1, 42)), np.array([0]), np.array([-1])),
            (np.zeros((1, 42)), np.array([0]), np.array([1.0])),
            (np.zeros((1, 42)), np.array([0]), np.array([True])),
        ]
        report = PredictionReport()
        self._observe(report, np.zeros((1, 42)), np.array([0]), np.array([2]))
        before = self._lines(report)
        for scores, symbols, positions in invalid:
            with self.subTest(shape=scores.shape, dtype=scores.dtype, symbols=symbols, positions=positions):
                with self.assertRaises(ValueError):
                    self._observe(report, scores, symbols, positions)
                self.assertEqual((report.total, report.correct), (1, 1))
                self.assertEqual(self._lines(report), before)

    def test_write_refuses_existing_file_and_cleans_failed_new_file(self):
        report = PredictionReport()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "existing.txt"
            target.write_text("keep me")
            with self.assertRaises(FileExistsError):
                report.write(target)
            self.assertEqual(target.read_text(), "keep me")
            failing_target = root / "partial.txt"
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
