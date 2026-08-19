import tempfile
import unittest
from pathlib import Path

from dataset_registry import (
    dataset_metadata,
    expand_dataset_selection,
    resolve_dataset_files,
)


class DatasetRegistryTest(unittest.TestCase):
    def test_groups_expand_in_stable_order(self) -> None:
        self.assertEqual(
            expand_dataset_selection("illumina"),
            ("novaseq", "nextseq2000"),
        )
        self.assertEqual(
            expand_dataset_selection("bgi_mgi"),
            ("dnbseq_t7", "mgiseq2000"),
        )
        self.assertEqual(
            expand_dataset_selection("mixed"),
            ("novaseq", "nextseq2000", "dnbseq_t7", "mgiseq2000"),
        )

    def test_aliases_and_duplicates_are_normalized(self) -> None:
        self.assertEqual(
            expand_dataset_selection("NovaSeq,nextseq-2000,illumina"),
            ("novaseq", "nextseq2000"),
        )
        self.assertEqual(
            expand_dataset_selection("DNBSEQ-T7,MGISEQ-2000"),
            ("dnbseq_t7", "mgiseq2000"),
        )

    def test_unknown_dataset_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown dataset/group"):
            expand_dataset_selection("unknown")

    def test_resolves_explicit_h5_fastq_pairs_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            items = resolve_dataset_files(
                "novaseq",
                root,
                require_h5=False,
                require_fastq=False,
            )
            self.assertEqual(len(items), 2)
            self.assertEqual(
                items[0].h5_path,
                root / "h5/subset_HG001_1.fq.gz.qual_model.h5",
            )
            self.assertEqual(
                items[0].fastq_path,
                root / "fq/NovaSeq/subset_HG001_1.fq.gz",
            )
            metadata = dataset_metadata(items)
            self.assertEqual(metadata["dataset_names"], ["novaseq"])
            self.assertEqual(
                metadata["dataset_platform_by_file"][items[0].h5_path.name],
                "Illumina",
            )


if __name__ == "__main__":
    unittest.main()
