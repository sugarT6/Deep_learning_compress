import unittest

from codec.evaluate import DatasetMetrics, build_evaluation_summary


class DirectQualityEvaluationTest(unittest.TestCase):
    def test_summary_reports_macro_micro_family_and_worst_dataset(self):
        metrics = [
            DatasetMetrics("A", "Illumina", "validation", False, 2, 10, 20.0, 2.0),
            DatasetMetrics("B", "Illumina", "validation", False, 2, 30, 120.0, 4.0),
            DatasetMetrics("C", "MGI/BGI", "validation", False, 2, 10, 60.0, 6.0),
            DatasetMetrics("D", "Illumina", "unseen_dataset", True, 2, 20, 100.0, 5.0),
        ]

        summary = build_evaluation_summary(metrics)

        validation = summary["validation"]
        self.assertAlmostEqual(validation["dataset_macro_bits_per_quality"], 4.0)
        self.assertAlmostEqual(validation["symbol_micro_bits_per_quality"], 4.0)
        self.assertAlmostEqual(
            validation["families"]["Illumina"]["symbol_micro_bits_per_quality"],
            3.5,
        )
        self.assertAlmostEqual(
            validation["platform_family_macro_bits_per_quality"], 4.75
        )
        self.assertEqual(validation["worst_dataset"]["accession"], "C")
        self.assertEqual(summary["unseen_instrument"]["dataset_count"], 1)
        self.assertEqual(
            summary["unseen_instrument"]["worst_dataset"]["accession"], "D"
        )


if __name__ == "__main__":
    unittest.main()
