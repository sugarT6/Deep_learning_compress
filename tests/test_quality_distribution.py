import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from query_quality_distribution import (
    count_quality_ids,
    format_quality_distribution,
)


class QualityDistributionTest(unittest.TestCase):
    def test_counts_multiple_h5_files_in_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = root / "first.h5"
            second = root / "second.h5"
            with h5py.File(first, "w") as handle:
                handle.create_dataset("observed", data=np.array([2, 2, 5], dtype=np.uint8))
            with h5py.File(second, "w") as handle:
                handle.create_dataset("observed", data=np.array([5, 94], dtype=np.uint8))

            counts = count_quality_ids([first, second], chunk_rows=2)

            self.assertEqual(int(counts.sum()), 5)
            self.assertEqual(int(counts[2]), 2)
            self.assertEqual(int(counts[5]), 2)
            self.assertEqual(int(counts[94]), 1)

    def test_formats_only_present_ids_five_per_line(self) -> None:
        counts = np.zeros(95, dtype=np.int64)
        counts[:6] = 1

        output = format_quality_distribution(counts)

        self.assertEqual(
            output,
            "Q0 16.67%\tQ1 16.67%\tQ2 16.67%\tQ3 16.67%\tQ4 16.67%\n"
            "Q5 16.67%",
        )

    def test_rejects_out_of_range_quality_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "bad.h5"
            with h5py.File(path, "w") as handle:
                handle.create_dataset("observed", data=np.array([95], dtype=np.uint8))

            with self.assertRaisesRegex(ValueError, "outside"):
                count_quality_ids([path])


if __name__ == "__main__":
    unittest.main()
