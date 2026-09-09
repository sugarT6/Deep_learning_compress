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
        self.assertEqual(
            expand_dataset_selection("matrix_novaseq6000_all"),
            ("matrix_novaseq6000_train", "matrix_novaseq6000_holdout"),
        )
        self.assertEqual(len(expand_dataset_selection("matrix_mixed_train")), 5)
        self.assertEqual(len(expand_dataset_selection("matrix_all")), 10)

    def test_aliases_and_duplicates_are_normalized(self) -> None:
        self.assertEqual(
            expand_dataset_selection("NovaSeq,nextseq-2000,illumina"),
            ("novaseq", "nextseq2000"),
        )
        self.assertEqual(
            expand_dataset_selection("DNBSEQ-T7,MGISEQ-2000"),
            ("dnbseq_t7", "mgiseq2000"),
        )
        self.assertEqual(
            expand_dataset_selection("novaseq-train"),
            ("novaseq_hg",),
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
            self.assertEqual(len(items), 4)
            self.assertEqual(
                items[0].h5_path,
                root / "h5/subset_HG001_1.fq.gz.qual_model.h5",
            )
            self.assertEqual(
                items[0].fastq_path,
                root / "fq/NovaSeq/subset_HG001_1.fq.gz",
            )
            self.assertEqual(
                items[2].h5_path,
                root / "h5/subset_HG003_1.fq.gz.qual_model.h5",
            )
            self.assertEqual(
                items[2].fastq_path,
                root / "fq/NovaSeq/subset_HG003_1.fq.gz",
            )
            self.assertEqual(
                items[3].h5_path,
                root / "h5/subset_NA12891.novaseq_1.fq.gz.qual_model.h5",
            )
            self.assertEqual(
                items[3].fastq_path,
                root / "fq/subset_NA12891.novaseq_1.fq.gz",
            )
            metadata = dataset_metadata(items)
            self.assertEqual(metadata["dataset_names"], ["novaseq"])
            self.assertEqual(
                metadata["dataset_platform_by_file"][items[0].h5_path.name],
                "Illumina",
            )

    def test_novaseq_hg_training_subset_excludes_na12891(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            items = resolve_dataset_files(
                "novaseq_hg",
                root,
                require_h5=False,
                require_fastq=False,
            )
            self.assertEqual(
                [item.h5_path.name for item in items],
                [
                    "subset_HG001_1.fq.gz.qual_model.h5",
                    "subset_HG002_1.fq.gz.qual_model.h5",
                    "subset_HG003_1.fq.gz.qual_model.h5",
                ],
            )

    def test_nextseq2000_includes_new_srr15731080(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            items = resolve_dataset_files(
                "nextseq2000",
                Path(temporary_directory),
                require_h5=False,
                require_fastq=False,
            )
            self.assertEqual(len(items), 3)
            self.assertEqual(
                items[-1].h5_path.name,
                "subset_SRR15731080_1.fq.gz.qual_model.h5",
            )

    def test_matrix_novaseq6000_fixed_train_and_all_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            train_items = resolve_dataset_files(
                "matrix_novaseq6000_train",
                root,
                require_h5=False,
                require_fastq=False,
            )
            all_items = resolve_dataset_files(
                "matrix_novaseq6000_all",
                root,
                require_h5=False,
                require_fastq=False,
            )
            self.assertEqual(
                [item.h5_path.name for item in train_items],
                [
                    "subset_ERR10746686_1.500k.fastq.gz.qual_model.h5",
                    "subset_ERR11454184_1.500k.fastq.gz.qual_model.h5",
                    "subset_ERR16748054_1.500k.fastq.gz.qual_model.h5",
                ],
            )
            self.assertEqual(len(all_items), 6)
            self.assertEqual(
                all_items[-1].h5_path.name,
                "subset_ERR3989434_1.500k.fastq.gz.qual_model.h5",
            )
            self.assertEqual(
                all_items[0].h5_path,
                root
                / "matrix/NovaSeq_6000/"
                "subset_ERR10746686_1.500k.fastq.gz.qual_model.h5",
            )
            self.assertEqual(
                all_items[0].fastq_path,
                root
                / "matrix/sub_NovaSeq_6000/"
                "subset_ERR10746686_1.500k.fastq.gz",
            )

    def test_every_matrix_group_resolves_three_train_and_six_all_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            platforms = (
                "dnbseq_t7",
                "nextseq2000",
                "novaseq6000",
                "novaseq_x_plus",
                "mgiseq_g400",
            )
            for platform in platforms:
                train_items = resolve_dataset_files(
                    f"matrix_{platform}_train",
                    root,
                    require_h5=False,
                    require_fastq=False,
                )
                all_items = resolve_dataset_files(
                    f"matrix_{platform}_all",
                    root,
                    require_h5=False,
                    require_fastq=False,
                )
                self.assertEqual(len(train_items), 3, platform)
                self.assertEqual(len(all_items), 6, platform)
                self.assertEqual(all_items[:3], train_items, platform)


if __name__ == "__main__":
    unittest.main()
