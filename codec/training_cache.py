#!/usr/bin/env python3
"""Build and read training-only direct-quality HDF5 caches.

These caches accelerate model training.  They are never inputs to the actual
compressor or decompressor and contain no SeqArc probabilities, q_hat,
residuals, quality-distribution prior, header, or plus fields.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Mapping, Optional, Sequence, Tuple, Union

import h5py
import numpy as np

from .fastq_stream import (
    BASE_OTHER_ID,
    DEFAULT_BATCH_READS,
    MAX_BATCH_READS,
    PHRED_OFFSET,
    QUALITY_ALPHABET_SIZE,
    FastqBatch,
    iter_fastq_batches,
    make_fastq_batch,
)


CACHE_FORMAT = "direct-quality-training-cache"
CACHE_SCHEMA_VERSION = 1
CACHE_FILENAME_SUFFIX = ".direct_quality.h5"
DEFAULT_FLUSH_SYMBOLS = 1_048_576
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "2nd" / "training_cache"

_REQUIRED_DATASETS = ("base_values", "quality_values", "read_offsets")
_REQUIRED_ATTRIBUTES = (
    "format",
    "schema_version",
    "source_fastq_basename",
    "source_size",
    "source_fingerprint_algorithm",
    "source_sha256",
    "read_count",
    "maximum_read_length",
    "minimum_quality_id",
    "maximum_quality_id",
    "phred_offset",
)


@dataclass(frozen=True)
class TrainingCacheMetadata:
    cache_path: Path
    source_fastq_basename: str
    source_size: int
    source_sha256: str
    read_count: int
    symbol_count: int
    maximum_read_length: int
    minimum_quality_id: int
    maximum_quality_id: int
    phred_offset: int


@dataclass(frozen=True)
class SampledCacheBatch:
    platform_family: str
    cache_path: Path
    batch: FastqBatch


def sha256_file(path: Union[str, Path], *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Return a SHA-256 fingerprint of the exact source file bytes."""

    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _fastq_stem(path: Path) -> str:
    name = path.name
    for suffix in (".fastq.gz", ".fq.gz", ".fastq", ".fq"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    raise ValueError(f"{path}: unsupported FASTQ filename")


def infer_accession(path: Union[str, Path]) -> str:
    """Infer the accession used for the default cache filename."""

    stem = _fastq_stem(Path(path))
    for marker in (".head2M", ".block"):
        stem = stem.split(marker, 1)[0]
    if stem.endswith("_1") or stem.endswith("_2"):
        stem = stem[:-2]
    if not stem:
        raise ValueError(f"{path}: cannot infer accession")
    return stem


def default_cache_path(
    fastq_path: Union[str, Path], output_dir: Union[str, Path] = DEFAULT_CACHE_DIR
) -> Path:
    return Path(output_dir) / f"{infer_accession(fastq_path)}{CACHE_FILENAME_SUFFIX}"


def _append_dataset(dataset: h5py.Dataset, values: np.ndarray) -> None:
    if not values.size:
        return
    old_size = int(dataset.shape[0])
    new_size = old_size + int(values.size)
    dataset.resize((new_size,))
    dataset[old_size:new_size] = values


def _flush_value_rows(
    base_dataset: h5py.Dataset,
    quality_dataset: h5py.Dataset,
    pending_bases: list,
    pending_qualities: list,
) -> None:
    if not pending_bases:
        return
    bases = np.concatenate(pending_bases) if pending_bases else np.empty(0, np.uint8)
    qualities = (
        np.concatenate(pending_qualities)
        if pending_qualities
        else np.empty(0, np.uint8)
    )
    _append_dataset(base_dataset, bases)
    _append_dataset(quality_dataset, qualities)
    pending_bases.clear()
    pending_qualities.clear()


def _source_signature(path: Path) -> Tuple[int, int, int, int]:
    """Return source identity fields that are unaffected by access-time updates."""

    status = path.stat()
    return status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns


def create_training_cache(
    fastq_path: Union[str, Path],
    output_path: Union[str, Path],
    *,
    force: bool = False,
    flush_symbols: int = DEFAULT_FLUSH_SYMBOLS,
) -> TrainingCacheMetadata:
    """Fully scan one FASTQ and atomically create a validated training cache."""

    fastq_path = Path(fastq_path)
    output_path = Path(output_path)
    if flush_symbols <= 0:
        raise ValueError("flush_symbols must be positive")
    if not fastq_path.is_file():
        raise FileNotFoundError(fastq_path)
    if output_path.exists() and not force:
        raise FileExistsError(f"{output_path} already exists; pass force=True to replace it")
    if fastq_path.resolve() == output_path.resolve():
        raise ValueError("training cache output must differ from the source FASTQ")

    source_stat = fastq_path.stat()
    source_signature = _source_signature(fastq_path)
    source_sha256 = sha256_file(fastq_path)
    if _source_signature(fastq_path) != source_signature:
        raise RuntimeError(f"{fastq_path}: source changed while fingerprinting")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=str(output_path.parent)
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)

    read_count = 0
    symbol_count = 0
    maximum_read_length = 0
    minimum_quality_id = QUALITY_ALPHABET_SIZE
    maximum_quality_id = -1
    current_offset = 0
    pending_symbol_count = 0
    pending_bases = []
    pending_qualities = []
    pending_offsets = []
    offset_flush_reads = 65_536

    try:
        with h5py.File(temporary_path, "w") as output:
            value_chunk = min(max(flush_symbols, 1), DEFAULT_FLUSH_SYMBOLS)
            base_dataset = output.create_dataset(
                "base_values",
                shape=(0,),
                maxshape=(None,),
                chunks=(value_chunk,),
                dtype=np.uint8,
            )
            quality_dataset = output.create_dataset(
                "quality_values",
                shape=(0,),
                maxshape=(None,),
                chunks=(value_chunk,),
                dtype=np.uint8,
            )
            offsets_dataset = output.create_dataset(
                "read_offsets",
                data=np.asarray([0], dtype=np.int64),
                maxshape=(None,),
                chunks=(offset_flush_reads,),
                dtype=np.int64,
            )

            for batch in iter_fastq_batches(fastq_path):
                active_bases = batch.bases[batch.active_mask]
                active_qualities = batch.qualities[batch.active_mask]
                pending_bases.append(active_bases)
                pending_qualities.append(active_qualities)
                pending_symbol_count += int(active_bases.size)

                for length in batch.lengths:
                    current_offset += int(length)
                    pending_offsets.append(current_offset)
                read_count += batch.read_count
                symbol_count += int(active_bases.size)
                maximum_read_length = max(maximum_read_length, batch.max_read_length)
                if active_qualities.size:
                    minimum_quality_id = min(
                        minimum_quality_id, int(active_qualities.min())
                    )
                    maximum_quality_id = max(
                        maximum_quality_id, int(active_qualities.max())
                    )

                if pending_symbol_count >= flush_symbols:
                    _flush_value_rows(
                        base_dataset,
                        quality_dataset,
                        pending_bases,
                        pending_qualities,
                    )
                    pending_symbol_count = 0
                if len(pending_offsets) >= offset_flush_reads:
                    _append_dataset(
                        offsets_dataset,
                        np.asarray(pending_offsets, dtype=np.int64),
                    )
                    pending_offsets.clear()

            _flush_value_rows(
                base_dataset, quality_dataset, pending_bases, pending_qualities
            )
            _append_dataset(
                offsets_dataset, np.asarray(pending_offsets, dtype=np.int64)
            )

            if read_count == 0:
                raise ValueError(f"{fastq_path}: FASTQ contains no records")
            if int(base_dataset.shape[0]) != symbol_count:
                raise RuntimeError("internal base cache length mismatch")
            if int(quality_dataset.shape[0]) != symbol_count:
                raise RuntimeError("internal quality cache length mismatch")
            if int(offsets_dataset.shape[0]) != read_count + 1:
                raise RuntimeError("internal read-offset cache length mismatch")

            output.attrs["format"] = CACHE_FORMAT
            output.attrs["schema_version"] = CACHE_SCHEMA_VERSION
            output.attrs["source_fastq_basename"] = fastq_path.name
            output.attrs["source_size"] = source_stat.st_size
            output.attrs["source_mtime_ns"] = source_stat.st_mtime_ns
            output.attrs["source_fingerprint_algorithm"] = "sha256"
            output.attrs["source_sha256"] = source_sha256
            output.attrs["read_count"] = read_count
            output.attrs["maximum_read_length"] = maximum_read_length
            output.attrs["minimum_quality_id"] = (
                minimum_quality_id if maximum_quality_id >= 0 else -1
            )
            output.attrs["maximum_quality_id"] = maximum_quality_id
            output.attrs["phred_offset"] = PHRED_OFFSET
            output.attrs["quality_alphabet_size"] = QUALITY_ALPHABET_SIZE
            output.attrs["base_encoding"] = "A=0,C=1,G=2,T=3,N=4,other=5"
            output.attrs["derived_without_seqarc"] = True

        if _source_signature(fastq_path) != source_signature:
            raise RuntimeError(f"{fastq_path}: source changed while building cache")
        os.replace(str(temporary_path), str(output_path))
    except Exception:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise

    return inspect_training_cache(output_path, deep=True)


def _attribute_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _validate_open_cache(handle: h5py.File, cache_path: Path, *, deep: bool) -> None:
    missing_datasets = [name for name in _REQUIRED_DATASETS if name not in handle]
    if missing_datasets:
        raise ValueError(
            f"{cache_path}: missing cache datasets: {', '.join(missing_datasets)}"
        )
    missing_attributes = [
        name for name in _REQUIRED_ATTRIBUTES if name not in handle.attrs
    ]
    if missing_attributes:
        raise ValueError(
            f"{cache_path}: missing cache attributes: {', '.join(missing_attributes)}"
        )
    if _attribute_text(handle.attrs["format"]) != CACHE_FORMAT:
        raise ValueError(f"{cache_path}: unsupported cache format")
    if int(handle.attrs["schema_version"]) != CACHE_SCHEMA_VERSION:
        raise ValueError(
            f"{cache_path}: unsupported cache schema version "
            f"{handle.attrs['schema_version']!r}"
        )
    if int(handle.attrs["phred_offset"]) != PHRED_OFFSET:
        raise ValueError(f"{cache_path}: unsupported Phred offset")
    if _attribute_text(handle.attrs["source_fingerprint_algorithm"]) != "sha256":
        raise ValueError(f"{cache_path}: unsupported source fingerprint algorithm")

    bases = handle["base_values"]
    qualities = handle["quality_values"]
    offsets = handle["read_offsets"]
    if bases.ndim != 1 or qualities.ndim != 1 or offsets.ndim != 1:
        raise ValueError(f"{cache_path}: cache datasets must be one-dimensional")
    if bases.dtype != np.dtype(np.uint8) or qualities.dtype != np.dtype(np.uint8):
        raise ValueError(f"{cache_path}: base/quality datasets must use uint8")
    if offsets.dtype != np.dtype(np.int64):
        raise ValueError(f"{cache_path}: read_offsets must use int64")
    if bases.shape != qualities.shape:
        raise ValueError(f"{cache_path}: base/quality dataset lengths differ")

    read_count = int(handle.attrs["read_count"])
    if read_count <= 0 or int(offsets.shape[0]) != read_count + 1:
        raise ValueError(f"{cache_path}: read count and offsets length disagree")
    if int(offsets[0]) != 0 or int(offsets[-1]) != int(bases.shape[0]):
        raise ValueError(f"{cache_path}: invalid first or final read offset")

    minimum_quality_id = int(handle.attrs["minimum_quality_id"])
    maximum_quality_id = int(handle.attrs["maximum_quality_id"])
    if minimum_quality_id < -1 or maximum_quality_id >= QUALITY_ALPHABET_SIZE:
        raise ValueError(f"{cache_path}: invalid quality-id metadata")
    if maximum_quality_id < 0 and minimum_quality_id != -1:
        raise ValueError(f"{cache_path}: invalid empty-quality metadata")
    if maximum_quality_id >= 0 and (
        minimum_quality_id < 0 or minimum_quality_id > maximum_quality_id
    ):
        raise ValueError(f"{cache_path}: invalid quality-id metadata")
    source_sha256 = _attribute_text(handle.attrs["source_sha256"])
    if len(source_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in source_sha256
    ):
        raise ValueError(f"{cache_path}: invalid source SHA-256 fingerprint")

    if deep:
        validation_chunk = 1_048_576
        previous_offset = 0
        observed_maximum_length = 0
        for start in range(1, int(offsets.shape[0]), validation_chunk):
            chunk = offsets[start : start + validation_chunk]
            differences = np.diff(
                np.concatenate(
                    (np.asarray([previous_offset], dtype=np.int64), chunk)
                )
            )
            if np.any(differences < 0):
                raise ValueError(f"{cache_path}: read_offsets are not nondecreasing")
            observed_maximum_length = max(
                observed_maximum_length, int(differences.max())
            )
            previous_offset = int(chunk[-1])

        for start in range(0, int(bases.shape[0]), validation_chunk):
            if int(bases[start : start + validation_chunk].max()) > BASE_OTHER_ID:
                raise ValueError(f"{cache_path}: base id outside supported range")

        observed_minimum = QUALITY_ALPHABET_SIZE
        observed_maximum = -1
        for start in range(0, int(qualities.shape[0]), validation_chunk):
            chunk = qualities[start : start + validation_chunk]
            observed_minimum = min(observed_minimum, int(chunk.min()))
            observed_maximum = max(observed_maximum, int(chunk.max()))
        if observed_maximum >= QUALITY_ALPHABET_SIZE:
            raise ValueError(f"{cache_path}: quality id outside Q0-Q41")
        if observed_maximum < 0:
            observed_minimum = -1
        if (
            observed_minimum != minimum_quality_id
            or observed_maximum != maximum_quality_id
        ):
            raise ValueError(f"{cache_path}: quality-id metadata does not match data")
        if observed_maximum_length != int(handle.attrs["maximum_read_length"]):
            raise ValueError(f"{cache_path}: maximum read length metadata mismatch")


def _metadata_from_open_cache(
    handle: h5py.File, cache_path: Path
) -> TrainingCacheMetadata:
    return TrainingCacheMetadata(
        cache_path=cache_path,
        source_fastq_basename=_attribute_text(
            handle.attrs["source_fastq_basename"]
        ),
        source_size=int(handle.attrs["source_size"]),
        source_sha256=_attribute_text(handle.attrs["source_sha256"]),
        read_count=int(handle.attrs["read_count"]),
        symbol_count=int(handle["base_values"].shape[0]),
        maximum_read_length=int(handle.attrs["maximum_read_length"]),
        minimum_quality_id=int(handle.attrs["minimum_quality_id"]),
        maximum_quality_id=int(handle.attrs["maximum_quality_id"]),
        phred_offset=int(handle.attrs["phred_offset"]),
    )


def _validate_source(metadata: TrainingCacheMetadata, source_path: Path) -> None:
    if source_path.name != metadata.source_fastq_basename:
        raise ValueError(
            f"{metadata.cache_path}: source basename mismatch "
            f"({source_path.name!r} != {metadata.source_fastq_basename!r})"
        )
    if source_path.stat().st_size != metadata.source_size:
        raise ValueError(f"{metadata.cache_path}: source size mismatch")
    observed_sha256 = sha256_file(source_path)
    if observed_sha256 != metadata.source_sha256:
        raise ValueError(f"{metadata.cache_path}: source SHA-256 fingerprint mismatch")


def inspect_training_cache(
    cache_path: Union[str, Path],
    *,
    source_path: Optional[Union[str, Path]] = None,
    deep: bool = False,
) -> TrainingCacheMetadata:
    """Validate cache structure and optionally its exact source fingerprint."""

    cache_path = Path(cache_path)
    with h5py.File(cache_path, "r") as handle:
        _validate_open_cache(handle, cache_path, deep=deep)
        metadata = _metadata_from_open_cache(handle, cache_path)
    if source_path is not None:
        _validate_source(metadata, Path(source_path))
    return metadata


class TrainingCacheReader:
    """Random-access reader that returns the shared ``FastqBatch`` contract."""

    def __init__(
        self,
        cache_path: Union[str, Path],
        *,
        source_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.cache_path = Path(cache_path)
        self._handle = h5py.File(self.cache_path, "r")
        try:
            _validate_open_cache(self._handle, self.cache_path, deep=False)
            self.metadata = _metadata_from_open_cache(
                self._handle, self.cache_path
            )
            if source_path is not None:
                _validate_source(self.metadata, Path(source_path))
        except Exception:
            self._handle.close()
            raise

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> "TrainingCacheReader":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    def _require_open(self) -> h5py.File:
        if self._handle is None:
            raise RuntimeError("training cache reader is closed")
        return self._handle

    def read_indices(self, read_indices: Sequence[int]) -> FastqBatch:
        """Read up to 256 reads by zero-based index in the requested order."""

        handle = self._require_open()
        requested = np.asarray(read_indices, dtype=np.int64)
        if requested.ndim != 1 or requested.size == 0:
            raise ValueError("read_indices must be a nonempty one-dimensional sequence")
        if requested.size > MAX_BATCH_READS:
            raise ValueError(
                f"a cache batch may contain at most {MAX_BATCH_READS} reads"
            )
        if int(requested.min()) < 0 or int(requested.max()) >= self.metadata.read_count:
            raise IndexError("cache read index out of range")
        if requested.size > 1 and np.all(requested[1:] == requested[:-1] + 1):
            return self.read_range(int(requested[0]), int(requested[-1]) + 1)

        offsets = handle["read_offsets"]
        base_values = handle["base_values"]
        quality_values = handle["quality_values"]
        base_rows = []
        quality_rows = []
        for read_index in requested:
            start = int(offsets[int(read_index)])
            stop = int(offsets[int(read_index) + 1])
            if stop < start:
                raise ValueError(f"{self.cache_path}: decreasing read offset")
            base_rows.append(base_values[start:stop])
            quality_rows.append(quality_values[start:stop])

        return make_fastq_batch(
            base_rows,
            quality_rows,
            requested,
            source_name=self.metadata.source_fastq_basename,
        )

    def read_range(self, start: int, stop: int) -> FastqBatch:
        """Read a contiguous half-open read range containing at most 256 reads."""

        if start < 0 or stop <= start or stop > self.metadata.read_count:
            raise IndexError("invalid cache read range")
        if stop - start > MAX_BATCH_READS:
            raise ValueError(
                f"a cache batch may contain at most {MAX_BATCH_READS} reads"
            )
        handle = self._require_open()
        offsets = handle["read_offsets"][start : stop + 1]
        first_offset = int(offsets[0])
        final_offset = int(offsets[-1])
        if np.any(offsets[1:] < offsets[:-1]):
            raise ValueError(f"{self.cache_path}: decreasing read offset")
        flat_bases = handle["base_values"][first_offset:final_offset]
        flat_qualities = handle["quality_values"][first_offset:final_offset]
        relative_offsets = offsets - first_offset
        base_rows = [
            flat_bases[int(relative_offsets[row]) : int(relative_offsets[row + 1])]
            for row in range(stop - start)
        ]
        quality_rows = [
            flat_qualities[
                int(relative_offsets[row]) : int(relative_offsets[row + 1])
            ]
            for row in range(stop - start)
        ]
        return make_fastq_batch(
            base_rows,
            quality_rows,
            np.arange(start, stop, dtype=np.int64),
            source_name=self.metadata.source_fastq_basename,
        )

    def iter_batches(
        self, *, batch_reads: int = DEFAULT_BATCH_READS
    ) -> Iterator[FastqBatch]:
        if batch_reads <= 0 or batch_reads > MAX_BATCH_READS:
            raise ValueError(f"batch_reads must be in [1, {MAX_BATCH_READS}]")
        for start in range(0, self.metadata.read_count, batch_reads):
            stop = min(start + batch_reads, self.metadata.read_count)
            yield self.read_range(start, stop)


class BalancedTrainingCacheSampler:
    """Two-level uniform family/file sampler producing same-file batches."""

    def __init__(
        self,
        cache_paths_by_family: Mapping[str, Sequence[Union[str, Path]]],
        *,
        seed: Optional[int] = None,
    ) -> None:
        if not cache_paths_by_family:
            raise ValueError("at least one platform family is required")
        self._families = tuple(cache_paths_by_family)
        self._paths: Dict[str, Tuple[Path, ...]] = {}
        self._readers: Dict[Path, TrainingCacheReader] = {}
        for family in self._families:
            paths = tuple(Path(path) for path in cache_paths_by_family[family])
            if not paths:
                raise ValueError(f"platform family {family!r} has no cache files")
            self._paths[family] = paths
        self._rng = np.random.default_rng(seed)

    def close(self) -> None:
        for reader in self._readers.values():
            reader.close()
        self._readers.clear()

    def __enter__(self) -> "BalancedTrainingCacheSampler":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    def sample_batch(
        self, *, batch_reads: int = DEFAULT_BATCH_READS
    ) -> SampledCacheBatch:
        if batch_reads <= 0 or batch_reads > MAX_BATCH_READS:
            raise ValueError(f"batch_reads must be in [1, {MAX_BATCH_READS}]")
        family = self._families[int(self._rng.integers(len(self._families)))]
        paths = self._paths[family]
        cache_path = paths[int(self._rng.integers(len(paths)))]
        reader = self._readers.get(cache_path)
        if reader is None:
            reader = TrainingCacheReader(cache_path)
            self._readers[cache_path] = reader

        count = min(batch_reads, reader.metadata.read_count)
        maximum_start = reader.metadata.read_count - count
        start = int(self._rng.integers(maximum_start + 1))
        return SampledCacheBatch(
            platform_family=family,
            cache_path=cache_path,
            batch=reader.read_range(start, start + count),
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fully scan FASTQ files and create training-only direct-quality HDF5 "
            "caches; these caches are not used by the actual codec."
        )
    )
    parser.add_argument("fastq", nargs="+", type=Path, help="input FASTQ files")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"cache directory; default: {DEFAULT_CACHE_DIR}",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--flush-symbols", type=int, default=DEFAULT_FLUSH_SYMBOLS)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    for fastq_path in args.fastq:
        output_path = default_cache_path(fastq_path, args.output_dir)
        metadata = create_training_cache(
            fastq_path,
            output_path,
            force=args.force,
            flush_symbols=args.flush_symbols,
        )
        print(
            f"{output_path}: {metadata.read_count} reads, "
            f"{metadata.symbol_count} quality symbols, "
            f"Q{metadata.minimum_quality_id}..Q{metadata.maximum_quality_id}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
