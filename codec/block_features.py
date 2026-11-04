"""Deterministic, explicitly transmitted quality-block descriptions.

A description occupies exactly 16 bytes: four relative read-position segments,
each with mean Q, population standard deviation, P(Q < 20), and P(Q >= 35).
Statistics are pooled across the original contiguous reads in one block.  The
model input is decoded from these bytes, never from unquantized statistics.

The encoder may inspect the whole quality block to build this description, but
must transmit it before encoding that block.  A decoder must use the received
description; it cannot recompute one from qualities it has not decoded yet.
"""

from __future__ import annotations

from math import isqrt
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from .fastq_stream import MAX_BATCH_READS, QUALITY_ALPHABET_SIZE, FastqBatch


DEFAULT_BLOCK_READS = 256
BLOCK_FEATURE_SEGMENTS = 4
BLOCK_FEATURE_DIM = 16
BLOCK_FEATURE_BYTES = 16
BLOCK_FEATURE_SCHEMA_VERSION = 1
_MAX_Q = QUALITY_ALPHABET_SIZE - 1
_LOW_THRESHOLD = 20
_HIGH_THRESHOLD = 35


def block_feature_schema(block_reads: int = DEFAULT_BLOCK_READS) -> Dict[str, Any]:
    """Return the JSON-serializable wire and normalization contract."""

    block_reads = _positive_integer(block_reads, "block_reads")
    return {
        "name": "relative-quality-quartiles",
        "version": BLOCK_FEATURE_SCHEMA_VERSION,
        "descriptor_bytes": BLOCK_FEATURE_BYTES,
        "feature_dimension": BLOCK_FEATURE_DIM,
        "block_reads": block_reads,
        "block_boundaries": "file-relative read index // block_reads; keep final partial block",
        "segments": BLOCK_FEATURE_SEGMENTS,
        "segment_boundaries": "floor(segment * read_length / 4)",
        "statistics_per_segment": [
            "mean_q",
            "population_std_q",
            "fraction_q_lt_20",
            "fraction_q_ge_35",
        ],
        "q_statistic_normalization": _MAX_Q,
        "quantization": "nearest uint8 of normalized statistic * 255; half up",
        "decoded_feature": "float32(byte) / float32(255)",
        "empty_segment": [0, 0, 0, 0],
        "uses_full_block_quality": True,
        "uses_header_or_platform": False,
    }


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _indices_array(read_indices: Sequence[int]) -> np.ndarray:
    values = np.asarray(read_indices)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("read_indices must be a nonempty one-dimensional sequence")
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError("read_indices must contain integers")
    if np.any(values < 0):
        raise IndexError("read indices must be nonnegative")
    if np.any(values > np.iinfo(np.int64).max):
        raise IndexError("read index exceeds int64 range")
    return values.astype(np.int64, copy=False)


def block_ids_for_read_indices(
    read_indices: Sequence[int], block_reads: int = DEFAULT_BLOCK_READS
) -> np.ndarray:
    """Map zero-based source read indices to fixed, file-relative block IDs."""

    block_reads = _positive_integer(block_reads, "block_reads")
    return _indices_array(read_indices) // block_reads


def block_bounds(
    block_id: int, read_count: int, block_reads: int = DEFAULT_BLOCK_READS
) -> Tuple[int, int]:
    """Return the original file's half-open read range for one block."""

    block_reads = _positive_integer(block_reads, "block_reads")
    read_count = _positive_integer(read_count, "read_count")
    if isinstance(block_id, (bool, np.bool_)) or not isinstance(block_id, (int, np.integer)):
        raise IndexError("block_id must be a nonnegative integer")
    block_id = int(block_id)
    start = block_id * block_reads
    if block_id < 0 or start >= read_count:
        raise IndexError("block_id outside source file")
    return start, min(start + block_reads, read_count)


def _new_statistics() -> np.ndarray:
    # Each row is count, sum(Q), sum(Q**2), low count, high count.
    return np.zeros((BLOCK_FEATURE_SEGMENTS, 5), dtype=np.int64)


def _accumulate_statistics(statistics: np.ndarray, batch: FastqBatch) -> None:
    qualities = np.asarray(batch.qualities, dtype=np.int64)
    positions = np.arange(qualities.shape[1], dtype=np.int64)[None, :]
    lengths = np.asarray(batch.lengths, dtype=np.int64)[:, None]
    for segment in range(BLOCK_FEATURE_SEGMENTS):
        start = lengths * segment // BLOCK_FEATURE_SEGMENTS
        stop = lengths * (segment + 1) // BLOCK_FEATURE_SEGMENTS
        mask = batch.active_mask & (positions >= start) & (positions < stop)
        values = qualities[mask]
        if values.size == 0:
            continue
        if np.any(values < 0) or np.any(values > _MAX_Q):
            raise ValueError(f"active quality id outside Q0-Q{_MAX_Q}")
        statistics[segment] += (
            int(values.size),
            int(values.sum(dtype=np.int64)),
            int(np.square(values).sum(dtype=np.int64)),
            int(np.count_nonzero(values < _LOW_THRESHOLD)),
            int(np.count_nonzero(values >= _HIGH_THRESHOLD)),
        )


def _round_ratio(numerator: int, denominator: int) -> int:
    return (2 * numerator + denominator) // (2 * denominator)


def _quantize_std(count: int, total: int, squares: int) -> int:
    # variance = (count*squares - total**2) / count**2.  Integer square
    # roots and an exact midpoint comparison avoid floating-point wire drift.
    variance_numerator = count * squares - total * total
    scaled_square = 255 * 255 * variance_numerator
    denominator = _MAX_Q * count
    lower = isqrt(scaled_square) // denominator
    midpoint = (2 * lower + 1) * denominator
    return lower + int(4 * scaled_square >= midpoint * midpoint)


def _encode_statistics(statistics: np.ndarray) -> bytes:
    values = []
    for row in statistics:
        count, total, squares, low, high = map(int, row)
        if count == 0:
            values.extend((0, 0, 0, 0))
            continue
        values.extend((
            _round_ratio(255 * total, _MAX_Q * count),
            _quantize_std(count, total, squares),
            _round_ratio(255 * low, count),
            _round_ratio(255 * high, count),
        ))
    return bytes(values)


def encode_block_features(batch: FastqBatch) -> bytes:
    """Describe every active Q in ``batch`` using the fixed 16-byte format.

    The caller supplies a whole original block, rather than a sampled training
    microbatch.  ``CacheBlockFeatures`` handles that distinction for training.
    """

    statistics = _new_statistics()
    _accumulate_statistics(statistics, batch)
    return _encode_statistics(statistics)


def decode_block_features(payload: bytes) -> np.ndarray:
    """Return exactly the float32[16] condition available to the decoder."""

    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError("block feature payload must be bytes-like")
    payload = bytes(payload)
    if len(payload) != BLOCK_FEATURE_BYTES:
        raise ValueError(f"block feature payload must contain {BLOCK_FEATURE_BYTES} bytes")
    return np.frombuffer(payload, dtype=np.uint8).astype(np.float32) / np.float32(255)


class CacheBlockFeatures:
    """Lazy, split-isolated descriptions of original contiguous cache blocks.

    ``reader`` is an open ``TrainingCacheReader``.  Its public ``read_range``
    method is used in chunks of at most 256 reads, so larger future block sizes
    do not change that shared batch contract.  Only 16-byte descriptions are
    retained; reads and unquantized Q statistics are not retained.
    """

    def __init__(
        self,
        reader: Any,
        block_reads: int = DEFAULT_BLOCK_READS,
        split_start: int = 0,
        split_stop: Optional[int] = None,
    ) -> None:
        self.reader = reader
        self.block_reads = _positive_integer(block_reads, "block_reads")
        self.read_count = int(reader.metadata.read_count)
        self.split_start = int(split_start)
        self.split_stop = self.read_count if split_stop is None else int(split_stop)
        if not 0 <= self.split_start < self.split_stop <= self.read_count:
            raise ValueError("invalid block feature split range")
        if self.split_start % self.block_reads:
            raise ValueError("block feature split_start must be a fixed block boundary")
        if self.split_stop != self.read_count and self.split_stop % self.block_reads:
            raise ValueError("block feature split_stop must be a fixed block boundary")
        self._payloads: Dict[int, bytes] = {}

    def payload_for_block(self, block_id: int) -> bytes:
        """Read and cache a whole block, rejecting access beyond this split."""

        start, stop = block_bounds(block_id, self.read_count, self.block_reads)
        if start < self.split_start or stop > self.split_stop:
            raise IndexError("quality block lies outside block feature split")
        block_id = int(block_id)
        if block_id not in self._payloads:
            statistics = _new_statistics()
            for chunk_start in range(start, stop, MAX_BATCH_READS):
                batch = self.reader.read_range(chunk_start, min(chunk_start + MAX_BATCH_READS, stop))
                _accumulate_statistics(statistics, batch)
            self._payloads[block_id] = _encode_statistics(statistics)
        return self._payloads[block_id]

    def for_read_indices(self, read_indices: Sequence[int]) -> np.ndarray:
        """Return float32[B,16] in requested read order, duplicates included."""

        indices = _indices_array(read_indices)
        if np.any(indices < self.split_start) or np.any(indices >= self.split_stop):
            raise IndexError("read index outside block feature split")
        block_ids = block_ids_for_read_indices(indices, self.block_reads)
        unique, inverse = np.unique(block_ids, return_inverse=True)
        descriptions = np.stack([
            decode_block_features(self.payload_for_block(int(block_id)))
            for block_id in unique
        ])
        return descriptions[inverse]
