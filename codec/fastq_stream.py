"""Strict streaming FASTQ parsing and the shared model batch contract."""

from __future__ import annotations

import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable, Iterator, Optional, Sequence, Tuple, Union

import numpy as np


DEFAULT_BATCH_READS = 64
SUPPORTED_FASTQ_SUFFIXES = (".fastq", ".fq", ".fastq.gz", ".fq.gz")

PHRED_OFFSET = 33
QUALITY_ALPHABET_SIZE = 42
QUALITY_PAD_ID = QUALITY_ALPHABET_SIZE

BASE_A_ID = 0
BASE_C_ID = 1
BASE_G_ID = 2
BASE_T_ID = 3
BASE_N_ID = 4
BASE_OTHER_ID = 5
BASE_PAD_ID = 6
BASE_ALPHABET_SIZE = BASE_PAD_ID + 1

_BASE_LOOKUP = np.full(256, BASE_OTHER_ID, dtype=np.uint8)
for _character, _base_id in (
    (b"A", BASE_A_ID),
    (b"C", BASE_C_ID),
    (b"G", BASE_G_ID),
    (b"T", BASE_T_ID),
    (b"N", BASE_N_ID),
):
    _BASE_LOOKUP[_character[0]] = _base_id
    _BASE_LOOKUP[_character.lower()[0]] = _base_id


@dataclass(frozen=True)
class RawFastqRecord:
    """One validated FASTQ record with exact original line terminators."""

    header: bytes
    sequence: bytes
    plus: bytes
    quality: bytes
    line_endings: Tuple[bytes, bytes, bytes, bytes]
    read_index: int

    def to_bytes(self) -> bytes:
        fields = (self.header, self.sequence, self.plus, self.quality)
        return b"".join(
            field + ending for field, ending in zip(fields, self.line_endings)
        )


@dataclass(frozen=True)
class FastqBatch:
    """Shared parser/cache representation consumed by the direct model.

    ``bases`` and ``qualities`` are read-major padded uint8 arrays.  Padding is
    described only by ``active_mask`` and must never enter the loss or entropy
    coder.  ``read_indices`` preserves source read order.  Direct FASTQ batches
    additionally carry ``raw_records`` so the original uncompressed FASTQ bytes
    can be reconstructed; training-cache batches intentionally do not.
    """

    bases: np.ndarray
    qualities: np.ndarray
    lengths: np.ndarray
    active_mask: np.ndarray
    read_count: int
    read_indices: np.ndarray
    source_name: str
    raw_records: Tuple[RawFastqRecord, ...] = ()

    @property
    def max_read_length(self) -> int:
        return int(self.bases.shape[1])

    def to_fastq_bytes(self) -> bytes:
        if len(self.raw_records) != self.read_count:
            raise ValueError("batch does not contain raw FASTQ records")
        return b"".join(record.to_bytes() for record in self.raw_records)


def _validate_fastq_path(path: Path) -> None:
    if not any(path.name.endswith(suffix) for suffix in SUPPORTED_FASTQ_SUFFIXES):
        supported = ", ".join(SUPPORTED_FASTQ_SUFFIXES)
        raise ValueError(f"{path}: unsupported FASTQ suffix; expected one of {supported}")


def _open_fastq(path: Path) -> BinaryIO:
    _validate_fastq_path(path)
    if path.name.endswith(".gz"):
        return gzip.open(path, "rb")
    return path.open("rb")


def _split_line_ending(line: bytes) -> Tuple[bytes, bytes]:
    if line.endswith(b"\r\n"):
        return line[:-2], b"\r\n"
    if line.endswith(b"\n"):
        return line[:-1], b"\n"
    return line, b""


def encode_base_ids(sequence: bytes) -> np.ndarray:
    """Map FASTQ bases to stable ids while retaining raw bytes separately."""

    return _BASE_LOOKUP[np.frombuffer(sequence, dtype=np.uint8)]


def encode_quality_ids(quality: bytes, path: Path, record_number: int) -> np.ndarray:
    """Map Phred+33 bytes to Q0..Q41, rejecting every value outside support."""

    if not quality:
        return np.empty(0, dtype=np.uint8)
    ascii_values = np.frombuffer(quality, dtype=np.uint8).astype(np.int16)
    quality_ids = ascii_values - PHRED_OFFSET
    minimum = int(quality_ids.min())
    maximum = int(quality_ids.max())
    if minimum < 0 or maximum >= QUALITY_ALPHABET_SIZE:
        raise ValueError(
            f"{path}: record {record_number} quality id outside Q0-Q41 "
            f"(observed Q{minimum}..Q{maximum})"
        )
    return quality_ids.astype(np.uint8, copy=False)


def _iter_fastq_records_with_ids(
    path: Path,
) -> Iterator[Tuple[RawFastqRecord, np.ndarray, np.ndarray]]:
    with _open_fastq(path) as handle:
        read_index = 0
        while True:
            raw_header = handle.readline()
            if raw_header == b"":
                return
            raw_sequence = handle.readline()
            raw_plus = handle.readline()
            raw_quality = handle.readline()
            record_number = read_index + 1
            if raw_sequence == b"" or raw_plus == b"" or raw_quality == b"":
                raise ValueError(f"{path}: truncated FASTQ record {record_number}")

            header, header_ending = _split_line_ending(raw_header)
            sequence, sequence_ending = _split_line_ending(raw_sequence)
            plus, plus_ending = _split_line_ending(raw_plus)
            quality, quality_ending = _split_line_ending(raw_quality)

            if not header.startswith(b"@"):
                raise ValueError(
                    f"{path}: record {record_number} header does not start with @"
                )
            if not plus.startswith(b"+"):
                raise ValueError(
                    f"{path}: record {record_number} plus line does not start with +"
                )
            if len(sequence) != len(quality):
                raise ValueError(
                    f"{path}: record {record_number} sequence/quality lengths differ "
                    f"({len(sequence)} != {len(quality)})"
                )

            base_ids = encode_base_ids(sequence)
            quality_ids = encode_quality_ids(quality, path, record_number)
            record = RawFastqRecord(
                header=header,
                sequence=sequence,
                plus=plus,
                quality=quality,
                line_endings=(
                    header_ending,
                    sequence_ending,
                    plus_ending,
                    quality_ending,
                ),
                read_index=read_index,
            )
            yield record, base_ids, quality_ids
            read_index += 1


def iter_fastq_records(path: Union[str, Path]) -> Iterator[RawFastqRecord]:
    """Yield strict four-line FASTQ records without loading the file at once.

    LF and CRLF are accepted independently on every line.  A missing line
    ending is accepted only for the final quality line; the original ending of
    every field is retained for byte-exact reconstruction of decompressed data.
    """

    for record, _, _ in _iter_fastq_records_with_ids(Path(path)):
        yield record


def _id_row(values: np.ndarray, maximum: int, name: str) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError(f"{name} rows must be one-dimensional")
    if values.size:
        minimum_value = int(values.min())
        maximum_value = int(values.max())
        if minimum_value < 0 or maximum_value > maximum:
            raise ValueError(
                f"{name} row contains id outside [0, {maximum}]: "
                f"{minimum_value}..{maximum_value}"
            )
    return values.astype(np.uint8, copy=False)


def make_fastq_batch(
    base_rows: Sequence[np.ndarray],
    quality_rows: Sequence[np.ndarray],
    read_indices: Sequence[int],
    *,
    source_name: str,
    raw_records: Sequence[RawFastqRecord] = (),
) -> FastqBatch:
    """Pad encoded read rows according to the single shared batch contract."""

    read_count = len(base_rows)
    if read_count == 0:
        raise ValueError("a FastqBatch must contain at least one read")
    if read_count > DEFAULT_BATCH_READS:
        raise ValueError(
            f"a FastqBatch may contain at most {DEFAULT_BATCH_READS} reads"
        )
    if len(quality_rows) != read_count or len(read_indices) != read_count:
        raise ValueError("base, quality, and read-index counts must match")
    if raw_records and len(raw_records) != read_count:
        raise ValueError("raw record count must match read count")

    checked_bases = [
        _id_row(row, BASE_OTHER_ID, "base") for row in base_rows
    ]
    checked_qualities = [
        _id_row(row, QUALITY_ALPHABET_SIZE - 1, "quality")
        for row in quality_rows
    ]
    lengths = np.asarray([row.size for row in checked_bases], dtype=np.int64)
    quality_lengths = np.asarray(
        [row.size for row in checked_qualities], dtype=np.int64
    )
    if not np.array_equal(lengths, quality_lengths):
        raise ValueError("encoded base and quality row lengths must match")

    max_read_length = int(lengths.max()) if read_count else 0
    bases = np.full(
        (read_count, max_read_length), BASE_PAD_ID, dtype=np.uint8
    )
    qualities = np.full(
        (read_count, max_read_length), QUALITY_PAD_ID, dtype=np.uint8
    )
    for row_index, (base_row, quality_row) in enumerate(
        zip(checked_bases, checked_qualities)
    ):
        length = int(lengths[row_index])
        bases[row_index, :length] = base_row
        qualities[row_index, :length] = quality_row

    active_mask = (
        np.arange(max_read_length, dtype=np.int64)[None, :] < lengths[:, None]
    )
    ordered_indices = np.asarray(read_indices, dtype=np.int64)
    if ordered_indices.ndim != 1:
        raise ValueError("read_indices must be one-dimensional")

    return FastqBatch(
        bases=bases,
        qualities=qualities,
        lengths=lengths,
        active_mask=active_mask,
        read_count=read_count,
        read_indices=ordered_indices,
        source_name=source_name,
        raw_records=tuple(raw_records),
    )


def _batch_from_parsed_records(
    parsed_records: Sequence[Tuple[RawFastqRecord, np.ndarray, np.ndarray]],
    path: Path,
) -> FastqBatch:
    records = [parsed[0] for parsed in parsed_records]
    return make_fastq_batch(
        [parsed[1] for parsed in parsed_records],
        [parsed[2] for parsed in parsed_records],
        [record.read_index for record in records],
        source_name=path.name,
        raw_records=records,
    )


def iter_fastq_batches(
    path: Union[str, Path], *, batch_reads: int = DEFAULT_BATCH_READS
) -> Iterator[FastqBatch]:
    """Stream validated FASTQ records in batches of at most 64 reads."""

    if batch_reads <= 0 or batch_reads > DEFAULT_BATCH_READS:
        raise ValueError(f"batch_reads must be in [1, {DEFAULT_BATCH_READS}]")
    path = Path(path)
    pending = []
    for parsed_record in _iter_fastq_records_with_ids(path):
        pending.append(parsed_record)
        if len(pending) == batch_reads:
            yield _batch_from_parsed_records(pending, path)
            pending = []
    if pending:
        yield _batch_from_parsed_records(pending, path)


def iter_cycle_major_positions(batch: FastqBatch) -> Iterator[Tuple[int, int]]:
    """Yield active ``(row, cycle)`` positions in codec symbol order."""

    for cycle in range(batch.max_read_length):
        for row in range(batch.read_count):
            if batch.active_mask[row, cycle]:
                yield row, cycle


def write_raw_fastq_batches(
    batches: Iterable[FastqBatch], output: BinaryIO
) -> None:
    """Write exact decompressed FASTQ bytes retained by direct-parser batches."""

    for batch in batches:
        if len(batch.raw_records) != batch.read_count:
            raise ValueError("batch does not contain raw FASTQ records")
        for record in batch.raw_records:
            output.write(record.to_bytes())
