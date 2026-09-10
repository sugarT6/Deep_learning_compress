"""Decoder for the version-1 single-stream integer range format.

The header's symbol count is the sole successful termination condition.  A
caller must decode exactly that many symbols and then call :meth:`finish`.
The arithmetic decoder reads zero guard bits after the meaningful finalized
payload, a standard finite-message convention; the exact meaningful bit count,
physical byte length, zero padding, and CRC32 are validated before decoding.
"""

from __future__ import annotations

from bisect import bisect_right
from typing import Iterable

from ._range_common import (
    HALF_RANGE,
    MAX_CODE,
    QUARTER_RANGE,
    STATE_BITS,
    THREE_QUARTER_RANGE,
    RangeCodingError,
    RangeStreamMetadata,
    normalize_cdf,
    parse_range_stream,
)


class _BitReader:
    def __init__(self, payload: bytes, bit_count: int) -> None:
        self._payload = payload
        self._bit_count = bit_count
        self._position = 0

    def read_with_zero_extension(self) -> int:
        if self._position >= self._bit_count:
            self._position += 1
            return 0
        byte = self._payload[self._position // 8]
        bit = (byte >> (7 - self._position % 8)) & 1
        self._position += 1
        return bit


class RangeDecoder:
    """Decode a framed stream with a synchronized CDF for every symbol."""

    def __init__(self, data: bytes) -> None:
        self._metadata, payload = parse_range_stream(data)
        self._reader = _BitReader(payload, self._metadata.payload_bit_count)
        self._low = 0
        self._high = MAX_CODE
        self._code = 0
        self._decoded_count = 0
        if self._metadata.symbol_count:
            for _ in range(STATE_BITS):
                self._code = (
                    self._code << 1
                ) | self._reader.read_with_zero_extension()

    @property
    def metadata(self) -> RangeStreamMetadata:
        return self._metadata

    @property
    def decoded_count(self) -> int:
        return self._decoded_count

    @property
    def done(self) -> bool:
        return self._decoded_count == self._metadata.symbol_count

    def decode(self, cdf: Iterable[int]) -> int:
        """Decode one symbol using exactly the encoder's CDF for this position."""

        if self.done:
            raise RangeCodingError("range stream has no undecoded symbols")
        normalized_cdf = normalize_cdf(cdf)
        total = normalized_cdf[-1]
        interval = self._high - self._low + 1
        scaled_value = ((self._code - self._low + 1) * total - 1) // interval
        symbol = bisect_right(normalized_cdf, scaled_value) - 1
        if symbol < 0 or symbol >= len(normalized_cdf) - 1:
            raise RangeCodingError("decoded cumulative value is outside the CDF")

        symbol_low = normalized_cdf[symbol]
        symbol_high = normalized_cdf[symbol + 1]
        new_low = self._low + (interval * symbol_low) // total
        new_high = self._low + (interval * symbol_high) // total - 1
        if new_low > new_high or not (new_low <= self._code <= new_high):
            raise RangeCodingError("range decoder entered an invalid interval")
        self._low, self._high = new_low, new_high

        while True:
            if self._high < HALF_RANGE:
                pass
            elif self._low >= HALF_RANGE:
                self._low -= HALF_RANGE
                self._high -= HALF_RANGE
                self._code -= HALF_RANGE
            elif self._low >= QUARTER_RANGE and self._high < THREE_QUARTER_RANGE:
                self._low -= QUARTER_RANGE
                self._high -= QUARTER_RANGE
                self._code -= QUARTER_RANGE
            else:
                break
            self._low <<= 1
            self._high = (self._high << 1) | 1
            self._code = (
                self._code << 1
            ) | self._reader.read_with_zero_extension()

        self._decoded_count += 1
        return symbol

    def finish(self) -> None:
        """Require that the header-declared symbol count was decoded exactly."""

        if not self.done:
            raise RangeCodingError(
                f"decoded {self._decoded_count} of {self._metadata.symbol_count} symbols"
            )


__all__ = ["RangeDecoder"]
