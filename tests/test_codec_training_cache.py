import gzip
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from codec.fastq_stream import iter_fastq_batches
from codec.training_cache import (
    CACHE_FORMAT,
    CACHE_SCHEMA_VERSION,
    BalancedTrainingCacheSampler,
    TrainingCacheReader,
    create_training_cache,
    default_cache_path,
    inspect_training_cache,
)


def make_variable_fastq(read_count):
    records = []
    sequences = (b"A", b"CG", b"TNA", b"acgt")
    qualities = (b"!", b"#J", b"$%&", b"'()J")
    for index in range(read_count):
        sequence = sequences[index % len(sequences)]
        quality = qualities[index % len(qualities)]
        records.append(
            b"@read" + str(index).encode("ascii") + b"\n"
            + sequence + b"\n+description\n" + quality + b"\n"
        )
    return b"".join(records)


class TrainingCacheTest(unittest.TestCase):
    def test_writes_required_schema_and_metadata(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fastq_path = root / "SRRTEST_1.head2M.fastq"
            fastq_path.write_bytes(make_variable_fastq(5))
            cache_path = root / "cache" / "SRRTEST.direct_quality.h5"

            metadata = create_training_cache(
                fastq_path, cache_path, flush_symbols=3
            )

            self.assertEqual(metadata.source_fastq_basename, fastq_path.name)
            self.assertEqual(metadata.source_size, fastq_path.stat().st_size)
            self.assertEqual(metadata.read_count, 5)
            self.assertEqual(metadata.symbol_count, 11)
            self.assertEqual(metadata.maximum_read_length, 4)
            self.assertEqual(metadata.minimum_quality_id, 0)
            self.assertEqual(metadata.maximum_quality_id, 41)
            self.assertEqual(metadata.phred_offset, 33)
            self.assertEqual(len(metadata.source_sha256), 64)

            with h5py.File(cache_path, "r") as handle:
                self.assertEqual(handle.attrs["format"], CACHE_FORMAT)
                self.assertEqual(
                    int(handle.attrs["schema_version"]), CACHE_SCHEMA_VERSION
                )
                self.assertTrue(bool(handle.attrs["derived_without_seqarc"]))
                self.assertEqual(
                    set(handle.keys()),
                    {"base_values", "quality_values", "read_offsets"},
                )
                np.testing.assert_array_equal(
                    handle["read_offsets"][:], [0, 1, 3, 6, 10, 11]
                )

            inspected = inspect_training_cache(
                cache_path, source_path=fastq_path, deep=True
            )
            self.assertEqual(inspected, metadata)
            self.assertEqual(
                default_cache_path(fastq_path, root).name,
                "SRRTEST.direct_quality.h5",
            )

    def test_parser_and_cache_reader_batches_are_elementwise_identical(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fastq_path = root / "reads.fq.gz"
            with gzip.open(fastq_path, "wb") as handle:
                handle.write(make_variable_fastq(270))
            cache_path = root / "reads.direct_quality.h5"
            create_training_cache(fastq_path, cache_path, flush_symbols=7)

            direct_batches = list(iter_fastq_batches(fastq_path))
            with TrainingCacheReader(
                cache_path, source_path=fastq_path
            ) as reader:
                cache_batches = list(reader.iter_batches())

            self.assertEqual(len(direct_batches), len(cache_batches))
            self.assertEqual(
                [batch.read_count for batch in cache_batches], [256, 14]
            )
            for direct, cached in zip(direct_batches, cache_batches):
                np.testing.assert_array_equal(cached.bases, direct.bases)
                np.testing.assert_array_equal(cached.qualities, direct.qualities)
                np.testing.assert_array_equal(cached.lengths, direct.lengths)
                np.testing.assert_array_equal(cached.active_mask, direct.active_mask)
                np.testing.assert_array_equal(cached.read_indices, direct.read_indices)
                self.assertEqual(cached.read_count, direct.read_count)
                self.assertEqual(cached.source_name, direct.source_name)
                self.assertEqual(cached.raw_records, ())

    def test_random_read_indices_preserve_requested_order(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fastq_path = root / "reads.fq"
            fastq_path.write_bytes(make_variable_fastq(8))
            cache_path = root / "reads.h5"
            create_training_cache(fastq_path, cache_path)

            with TrainingCacheReader(cache_path) as reader:
                batch = reader.read_indices([6, 1, 6, 0])

            np.testing.assert_array_equal(batch.read_indices, [6, 1, 6, 0])
            np.testing.assert_array_equal(batch.lengths, [3, 2, 3, 1])

    def test_schema_version_is_validated(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fastq_path = root / "reads.fq"
            fastq_path.write_bytes(make_variable_fastq(1))
            cache_path = root / "reads.h5"
            create_training_cache(fastq_path, cache_path)
            with h5py.File(cache_path, "r+") as handle:
                handle.attrs["schema_version"] = CACHE_SCHEMA_VERSION + 1

            with self.assertRaisesRegex(ValueError, "schema version"):
                inspect_training_cache(cache_path)

    def test_source_size_and_fingerprint_are_validated(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fastq_path = root / "reads.fq"
            original = make_variable_fastq(2)
            fastq_path.write_bytes(original)
            cache_path = root / "reads.h5"
            create_training_cache(fastq_path, cache_path)

            replacement = original.replace(b"@read0", b"@other", 1)
            self.assertEqual(len(replacement), len(original))
            fastq_path.write_bytes(replacement)
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                inspect_training_cache(cache_path, source_path=fastq_path)

            fastq_path.write_bytes(replacement + b"x")
            with self.assertRaisesRegex(ValueError, "source size mismatch"):
                inspect_training_cache(cache_path, source_path=fastq_path)

    def test_invalid_fastq_does_not_leave_output_cache(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fastq_path = root / "bad.fq"
            fastq_path.write_bytes(b"@r\nA\n+\nK\n")
            cache_path = root / "bad.h5"
            with self.assertRaisesRegex(ValueError, "outside Q0-Q41"):
                create_training_cache(fastq_path, cache_path)
            self.assertFalse(cache_path.exists())

    def test_empty_fastq_is_rejected_but_zero_length_read_is_cached(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            empty_path = root / "empty.fq"
            empty_path.write_bytes(b"")
            with self.assertRaisesRegex(ValueError, "contains no records"):
                create_training_cache(empty_path, root / "empty.h5")

            zero_path = root / "zero.fq"
            zero_path.write_bytes(b"@r\n\n+\n\n")
            zero_cache = root / "zero.h5"
            metadata = create_training_cache(zero_path, zero_cache)
            self.assertEqual(metadata.read_count, 1)
            self.assertEqual(metadata.symbol_count, 0)
            self.assertEqual(metadata.minimum_quality_id, -1)
            self.assertEqual(metadata.maximum_quality_id, -1)
            with TrainingCacheReader(zero_cache) as reader:
                batch = reader.read_range(0, 1)
            self.assertEqual(batch.bases.shape, (1, 0))

    def test_two_level_sampler_returns_one_family_and_one_file_batch(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            cache_paths = []
            for index in range(3):
                fastq_path = root / f"reads{index}.fq"
                fastq_path.write_bytes(make_variable_fastq(index + 2))
                cache_path = root / f"reads{index}.h5"
                create_training_cache(fastq_path, cache_path)
                cache_paths.append(cache_path)

            with BalancedTrainingCacheSampler(
                {"Illumina": cache_paths[:2], "MGI": cache_paths[2:]}, seed=7
            ) as sampler:
                for _ in range(10):
                    sampled = sampler.sample_batch(batch_reads=2)
                    self.assertIn(sampled.platform_family, ("Illumina", "MGI"))
                    self.assertIn(sampled.cache_path, cache_paths)
                    self.assertLessEqual(sampled.batch.read_count, 2)
                    self.assertEqual(
                        sampled.batch.source_name,
                        inspect_training_cache(
                            sampled.cache_path
                        ).source_fastq_basename,
                    )


if __name__ == "__main__":
    unittest.main()
