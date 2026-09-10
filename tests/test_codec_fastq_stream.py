import gzip
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np

from codec.fastq_stream import (
    BASE_A_ID,
    BASE_C_ID,
    BASE_G_ID,
    BASE_N_ID,
    BASE_OTHER_ID,
    BASE_PAD_ID,
    BASE_T_ID,
    DEFAULT_BATCH_READS,
    MAX_BATCH_READS,
    QUALITY_PAD_ID,
    iter_cycle_major_positions,
    iter_fastq_batches,
    iter_fastq_records,
    write_raw_fastq_batches,
)


def fastq_records(count, *, line_ending=b"\n"):
    records = []
    for index in range(count):
        records.append(
            b"@read" + str(index).encode("ascii") + line_ending
            + b"AC" + line_ending
            + b"+description " + str(index).encode("ascii") + line_ending
            + b"!J" + line_ending
        )
    return b"".join(records)


class FastqStreamTest(unittest.TestCase):
    def _write(self, root, name, data):
        path = Path(root) / name
        if name.endswith(".gz"):
            with gzip.open(path, "wb") as handle:
                handle.write(data)
        else:
            path.write_bytes(data)
        return path

    def test_supported_plain_and_gzip_suffixes(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            for name in ("reads.fastq", "reads.fq", "reads.fastq.gz", "reads.fq.gz"):
                with self.subTest(name=name):
                    path = self._write(
                        temporary_directory, name, b"@r\nA\n+\n!\n"
                    )
                    batches = list(iter_fastq_batches(path))
                    self.assertEqual(len(batches), 1)
                    self.assertEqual(batches[0].read_count, 1)

    def test_batch_boundaries_and_read_order(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            for read_count, expected_batch_sizes in (
                (1, [1]),
                (17, [17]),
                (64, [64]),
                (65, [65]),
                (255, [255]),
                (256, [256]),
                (257, [256, 1]),
                (513, [256, 256, 1]),
            ):
                with self.subTest(read_count=read_count):
                    path = self._write(
                        temporary_directory,
                        f"reads_{read_count}.fq",
                        fastq_records(read_count),
                    )
                    batches = list(iter_fastq_batches(path))
                    self.assertEqual(
                        [batch.read_count for batch in batches], expected_batch_sizes
                    )
                    observed_order = np.concatenate(
                        [batch.read_indices for batch in batches]
                    )
                    np.testing.assert_array_equal(
                        observed_order, np.arange(read_count, dtype=np.int64)
                    )
            self.assertEqual(DEFAULT_BATCH_READS, 256)
            self.assertEqual(MAX_BATCH_READS, 256)

    def test_variable_lengths_padding_mask_and_cycle_major_order(self):
        data = (
            b"@r0\nACG\n+\n!J#\n"
            b"@r1\nN\n+label\n$\n"
            b"@r2\naXtN\n+\n%&'(\n"
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = self._write(temporary_directory, "variable.fastq", data)
            batch = next(iter_fastq_batches(path))

        np.testing.assert_array_equal(batch.lengths, [3, 1, 4])
        np.testing.assert_array_equal(
            batch.active_mask,
            [
                [True, True, True, False],
                [True, False, False, False],
                [True, True, True, True],
            ],
        )
        np.testing.assert_array_equal(
            batch.bases,
            [
                [BASE_A_ID, BASE_C_ID, BASE_G_ID, BASE_PAD_ID],
                [BASE_N_ID, BASE_PAD_ID, BASE_PAD_ID, BASE_PAD_ID],
                [BASE_A_ID, BASE_OTHER_ID, BASE_T_ID, BASE_N_ID],
            ],
        )
        self.assertEqual(int(batch.qualities[0, 3]), QUALITY_PAD_ID)
        self.assertEqual(
            list(iter_cycle_major_positions(batch)),
            [(0, 0), (1, 0), (2, 0), (0, 1), (2, 1), (0, 2), (2, 2), (2, 3)],
        )

    def test_q0_q41_and_q42_rejection(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            good = self._write(
                temporary_directory, "boundary.fq", b"@r\nAC\n+\n!J\n"
            )
            batch = next(iter_fastq_batches(good))
            np.testing.assert_array_equal(batch.qualities, [[0, 41]])

            bad = self._write(
                temporary_directory, "outside.fq", b"@r\nA\n+\nK\n"
            )
            with self.assertRaisesRegex(ValueError, "outside Q0-Q41"):
                list(iter_fastq_batches(bad))

    def test_exact_raw_round_trip_for_lf_crlf_plus_description_and_no_final_newline(self):
        data = (
            b"@lf\nAC\n+same description\n!J\n"
            b"@crlf\r\nT\r\n+another description\r\n#"
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = self._write(temporary_directory, "mixed.fastq", data)
            batches = list(iter_fastq_batches(path, batch_reads=1))

        self.assertEqual(batches[0].raw_records[0].plus, b"+same description")
        self.assertEqual(batches[1].raw_records[0].line_endings[-1], b"")
        reconstructed = io.BytesIO()
        write_raw_fastq_batches(batches, reconstructed)
        self.assertEqual(reconstructed.getvalue(), data)

    def test_empty_read_with_line_ending_is_supported(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = self._write(
                temporary_directory, "empty.fq", b"@empty\n\n+description\n\n"
            )
            batch = next(iter_fastq_batches(path))
        self.assertEqual(batch.max_read_length, 0)
        self.assertEqual(batch.bases.shape, (1, 0))
        self.assertEqual(batch.active_mask.shape, (1, 0))
        self.assertEqual(batch.to_fastq_bytes(), b"@empty\n\n+description\n\n")

    def test_structural_validation_errors(self):
        cases = (
            ("mismatch", b"@r\nAC\n+\n!\n", "lengths differ"),
            ("header", b"r\nA\n+\n!\n", "header does not start"),
            ("plus", b"@r\nA\nminus\n!\n", "plus line does not start"),
            ("truncated_sequence", b"@r\n", "truncated FASTQ"),
            ("truncated_plus", b"@r\nA\n", "truncated FASTQ"),
            ("truncated_quality", b"@r\nA\n+\n", "truncated FASTQ"),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            for name, data, message in cases:
                with self.subTest(name=name):
                    path = self._write(temporary_directory, f"{name}.fq", data)
                    with self.assertRaisesRegex(ValueError, message):
                        list(iter_fastq_records(path))

    def test_rejects_unsupported_suffix_and_batch_size(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = self._write(
                temporary_directory, "reads.txt", b"@r\nA\n+\n!\n"
            )
            with self.assertRaisesRegex(ValueError, "unsupported FASTQ suffix"):
                list(iter_fastq_batches(path))
            valid = self._write(
                temporary_directory, "reads.fq", b"@r\nA\n+\n!\n"
            )
            with self.assertRaisesRegex(ValueError, "batch_reads"):
                list(iter_fastq_batches(valid, batch_reads=257))

    def test_corrupt_gzip_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "corrupt.fq.gz"
            path.write_bytes(b"not a gzip stream")
            with self.assertRaises((OSError, EOFError)):
                list(iter_fastq_batches(path))


if __name__ == "__main__":
    unittest.main()
