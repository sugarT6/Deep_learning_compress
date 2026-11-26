"""Shared constants, validation, and framing for the version-1 range stream."""

from __future__ import annotations

import operator
import struct
import zlib
from dataclasses import dataclass
from typing import Iterable, Tuple


RANGE_CODER_VERSION = 1
STREAM_MAGIC = b"QRC1"
STATE_BITS = 32
FULL_RANGE = 1 << STATE_BITS
MAX_CODE = FULL_RANGE - 1
HALF_RANGE = FULL_RANGE >> 1
QUARTER_RANGE = HALF_RANGE >> 1
THREE_QUARTER_RANGE = QUARTER_RANGE * 3
MAX_CDF_TOTAL = QUARTER_RANGE

# Little-endian, matching the future codec container contract:
# magic, symbol count, valid arithmetic payload bits, CRC32(prefix + payload).
_HEADER_PREFIX = struct.Struct("<4sQQ")
_HEADER = struct.Struct("<4sQQI")
RANGE_STREAM_HEADER_BYTES = _HEADER.size


class RangeCodingError(ValueError):
    """Base class for deterministic range-stream errors."""


class InvalidCDFError(RangeCodingError):
    """Raised when a CDF cannot define nonempty integer symbol intervals."""


class InvalidRangeStreamError(RangeCodingError):
    """Raised for malformed, corrupt, or semantically invalid stream bytes."""


class TruncatedRangeStreamError(InvalidRangeStreamError):
    """Raised when fewer bytes than the stream header advertises are available."""


@dataclass(frozen=True)
class RangeStreamMetadata:
    symbol_count: int
    payload_bit_count: int
    payload_byte_count: int
    total_byte_count: int


def normalize_cdf(cdf: Iterable[int]) -> Tuple[int, ...]:
    """Return a validated strictly increasing integer CDF beginning at zero."""

    try:
        raw_values = tuple(cdf)
    except TypeError as exc:
        raise InvalidCDFError("cdf must be an iterable of integers") from exc
    if len(raw_values) < 2:
        raise InvalidCDFError("cdf must contain at least two entries")

    values = []
    for value in raw_values:
        if isinstance(value, bool):
            raise InvalidCDFError("cdf entries must be integers, not booleans")
        try:
            values.append(operator.index(value))
        except TypeError as exc:
            raise InvalidCDFError("cdf entries must be integers") from exc
    if values[0] != 0:
        raise InvalidCDFError("cdf[0] must be zero")
    if any(right <= left for left, right in zip(values, values[1:])):
        raise InvalidCDFError("cdf entries must be strictly increasing")
    if values[-1] > MAX_CDF_TOTAL:
        raise InvalidCDFError(
            f"cdf total must not exceed {MAX_CDF_TOTAL} for {STATE_BITS}-bit state"
        )
    return tuple(values)


def build_range_stream(
    symbol_count: int, payload_bit_count: int, payload: bytes
) -> bytes:
    if symbol_count < 0 or symbol_count >= 1 << 64:
        raise RangeCodingError("symbol count must fit an unsigned 64-bit integer")
    if payload_bit_count < 0 or payload_bit_count >= 1 << 64:
        raise RangeCodingError("payload bit count must fit an unsigned 64-bit integer")
    if len(payload) != (payload_bit_count + 7) // 8:
        raise RangeCodingError("payload byte length does not match payload bit count")
    prefix = _HEADER_PREFIX.pack(STREAM_MAGIC, symbol_count, payload_bit_count)
    checksum = zlib.crc32(prefix)
    checksum = zlib.crc32(payload, checksum) & 0xFFFFFFFF
    return _HEADER.pack(STREAM_MAGIC, symbol_count, payload_bit_count, checksum) + payload


def parse_range_stream(data: bytes) -> tuple[RangeStreamMetadata, bytes]:
    try:
        stream = memoryview(data).tobytes()
    except TypeError as exc:
        raise InvalidRangeStreamError("range stream must be bytes-like") from exc
    if len(stream) < _HEADER.size:
        raise TruncatedRangeStreamError(
            f"range stream is truncated before the {_HEADER.size}-byte header"
        )

    magic, symbol_count, payload_bit_count, stored_checksum = _HEADER.unpack_from(stream)
    if magic != STREAM_MAGIC:
        raise InvalidRangeStreamError("invalid range-stream magic/version")
    payload_byte_count = (payload_bit_count + 7) // 8
    expected_size = _HEADER.size + payload_byte_count
    if len(stream) < expected_size:
        raise TruncatedRangeStreamError(
            f"range stream advertises {expected_size} bytes but only {len(stream)} remain"
        )
    if len(stream) > expected_size:
        raise InvalidRangeStreamError("range stream has trailing bytes")
    payload = stream[_HEADER.size:]

    prefix = _HEADER_PREFIX.pack(magic, symbol_count, payload_bit_count)
    observed_checksum = zlib.crc32(prefix)
    observed_checksum = zlib.crc32(payload, observed_checksum) & 0xFFFFFFFF
    if observed_checksum != stored_checksum:
        raise InvalidRangeStreamError("range-stream checksum mismatch")
    if payload_bit_count % 8 and payload:
        unused_bits = 8 - (payload_bit_count % 8)
        if payload[-1] & ((1 << unused_bits) - 1):
            raise InvalidRangeStreamError("nonzero padding bits in range stream")
    if symbol_count == 0 and payload_bit_count != 0:
        raise InvalidRangeStreamError("empty symbol stream must have an empty payload")
    if symbol_count > 0 and payload_bit_count == 0:
        raise InvalidRangeStreamError("nonempty symbol stream must have a payload")

    metadata = RangeStreamMetadata(
        symbol_count=symbol_count,
        payload_bit_count=payload_bit_count,
        payload_byte_count=payload_byte_count,
        total_byte_count=expected_size,
    )
    return metadata, payload


__all__ = [
    "FULL_RANGE",
    "HALF_RANGE",
    "InvalidCDFError",
    "InvalidRangeStreamError",
    "MAX_CDF_TOTAL",
    "MAX_CODE",
    "QUARTER_RANGE",
    "RANGE_CODER_VERSION",
    "RANGE_STREAM_HEADER_BYTES",
    "RangeCodingError",
    "RangeStreamMetadata",
    "STATE_BITS",
    "THREE_QUARTER_RANGE",
    "TruncatedRangeStreamError",
    "build_range_stream",
    "normalize_cdf",
    "parse_range_stream",
]
