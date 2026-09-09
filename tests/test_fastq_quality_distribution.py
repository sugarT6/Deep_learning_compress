import gzip
import tempfile
import unittest
from pathlib import Path

import numpy as np
from openpyxl import Workbook

from query_fastq_quality_distribution import (
    DatasetMetadata,
    count_fastq_quality_ids,
    format_dataset_report,
    load_dataset_metadata,
    resolve_fastq_path,
    select_datasets,
)


class FastqQualityDistributionTest(unittest.TestCase):
    def test_loads_platform_group_and_selects_only_human_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "details.xlsx"
            workbook = Workbook()
            worksheet = workbook.active
            worksheet.title = "sequencing_platform_details"
            worksheet.append(
                [
                    "Accession",
                    "Platform",
                    "Instrument_Model",
                    "Platform_Group",
                    "Species",
                ]
            )
            worksheet.append(
                ["SRR1", "ILLUMINA", "NovaSeq 6000", "Illumina NovaSeq", "Homo sapiens"]
            )
            worksheet.append(
                ["SRR2", "ILLUMINA", "HiSeq 2000", "", "Mus musculus"]
            )
            workbook.save(path)
            workbook.close()

            records = load_dataset_metadata(path)
            selected = select_datasets(records)

            self.assertEqual(
                selected,
                [DatasetMetadata("SRR1", "Illumina NovaSeq", "Homo sapiens")],
            )
            self.assertEqual(select_datasets(records, ["SRR1_1.fastq.gz"]), selected)

    def test_explicit_selection_accepts_multiple_species_in_input_order(self) -> None:
        records = [
            DatasetMetadata("SRR1", "Platform 1", "Homo sapiens"),
            DatasetMetadata("SRR2", "Platform 2", "Mus musculus"),
            DatasetMetadata("SRR3", "Platform 3", "Arabidopsis thaliana"),
        ]

        selected = select_datasets(records, ["SRR3", "SRR2.fastq.gz"])

        self.assertEqual(selected, [records[2], records[1]])

    def test_resolves_one_accession_fastq(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            expected = root / "SRR123_2.head2M.fastq.gz"
            expected.touch()
            (root / "SRR12.head2M.fastq.gz").touch()

            self.assertEqual(resolve_fastq_path(root, "SRR123"), expected)

    def test_counts_complete_quality_strings_from_gzip_fastq(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "SRR1.fastq.gz"
            with gzip.open(path, "wb") as handle:
                handle.write(b"@read1\nACG\n+\n!\"#\n")
                handle.write(b"@read2\nTT\n+\nJJ\n")

            counts = count_fastq_quality_ids(path)

            self.assertEqual(int(counts.sum()), 5)
            self.assertEqual(int(counts[0]), 1)
            self.assertEqual(int(counts[1]), 1)
            self.assertEqual(int(counts[2]), 1)
            self.assertEqual(int(counts[41]), 2)

    def test_stops_after_requested_number_of_quality_lines(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "SRR1.fastq.gz"
            with gzip.open(path, "wb") as handle:
                handle.write(b"@read1\nA\n+\n!\n")
                handle.write(b"@read2\nA\n+\nJ\n")

            counts = count_fastq_quality_ids(path, max_reads=1)

            self.assertEqual(int(counts.sum()), 1)
            self.assertEqual(int(counts[0]), 1)
            self.assertEqual(int(counts[41]), 0)

    def test_report_has_platform_accession_and_six_entries_per_line(self) -> None:
        counts = np.zeros(95, dtype=np.int64)
        counts[:7] = 1
        metadata = DatasetMetadata("SRR1", "Illumina NovaSeq 6000", "Homo sapiens")

        report = format_dataset_report(metadata, counts)

        self.assertEqual(
            report,
            "Illumina NovaSeq 6000\tSRR1\n"
            "Q0 14.2857%\tQ1 14.2857%\tQ2 14.2857%\tQ3 14.2857%\t"
            "Q4 14.2857%\tQ5 14.2857%\n"
            "Q6 14.2857%",
        )

    def test_rejects_sequence_quality_length_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "bad.fastq"
            path.write_bytes(b"@read1\nACG\n+\n!!\n")

            with self.assertRaisesRegex(ValueError, "lengths differ"):
                count_fastq_quality_ids(path)


if __name__ == "__main__":
    unittest.main()
