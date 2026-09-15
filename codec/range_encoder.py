"""Single-stream, integer-only arithmetic range encoder.

The coder maintains an inclusive 32-bit interval ``[low, high]``.  Every
symbol update uses only integer multiply, divide, add, and subtract with that
position's CDF.  E1/E2/E3 renormalization emits bits most-significant first.
Finalization emits one disambiguating bit plus deferred underflow bits, then
zero-pads only the final byte.  The stream header records the symbol count and
the number of meaningful payload bits; CRC32 detects truncation/corruption.
"""

from __future__ import annotations

import operator
from typing import Iterable, Optional, Sequence

from ._range_common import (
    HALF_RANGE,
    MAX_CODE,
    QUARTER_RANGE,
    THREE_QUARTER_RANGE,
    InvalidCDFError,
    RangeCodingError,
    RangeStreamMetadata,
    build_range_stream,
    normalize_cdf,
)


class _BitWriter:
    def __init__(self) -> None:
        self._bytes = bytearray()
        self._current_byte = 0
        self._bits_in_current_byte = 0
        self.bit_count = 0

    def write(self, bit: int) -> None:
        self._current_byte = (self._current_byte << 1) | bit
        self._bits_in_current_byte += 1
        self.bit_count += 1
        if self._bits_in_current_byte == 8:
            self._bytes.append(self._current_byte)
            self._current_byte = 0
            self._bits_in_current_byte = 0

    def finish(self) -> bytes:
        if self._bits_in_current_byte:
            self._current_byte <<= 8 - self._bits_in_current_byte
            self._bytes.append(self._current_byte)
            self._current_byte = 0
            self._bits_in_current_byte = 0
        return bytes(self._bytes)


class RangeEncoder:
    """Encode symbols sequentially with a potentially different CDF each time."""

    def __init__(self) -> None:
        self._low = 0
        self._high = MAX_CODE
        self._pending_underflow_bits = 0
        self._writer = _BitWriter()
        self._symbol_count = 0
        self._finished_stream: Optional[bytes] = None
        self._metadata: Optional[RangeStreamMetadata] = None

    @property
    def symbol_count(self) -> int:
        return self._symbol_count

    @property
    def metadata(self) -> RangeStreamMetadata:
        if self._metadata is None:
            raise RangeCodingError("range encoder has not been finalized")
        return self._metadata

    def _emit_with_underflow(self, bit: int) -> None:
        self._writer.write(bit)
        opposite = 1 - bit
        for _ in range(self._pending_underflow_bits):
            self._writer.write(opposite)
        self._pending_underflow_bits = 0

    def encode(self, symbol: int, cdf: Iterable[int]) -> None:
        """Encode one integer symbol using the supplied integer CDF."""

        if self._finished_stream is not None:
            raise RangeCodingError("cannot encode after finalization")
        if isinstance(symbol, bool):
            raise RangeCodingError("symbol must be an integer")
        try:
            normalized_symbol = operator.index(symbol)
        except TypeError as exc:
            raise RangeCodingError("symbol must be an integer") from exc
        normalized_cdf = normalize_cdf(cdf)
        symbol_limit = len(normalized_cdf) - 1
        if normalized_symbol < 0 or normalized_symbol >= symbol_limit:
            raise RangeCodingError(
                f"symbol {normalized_symbol} is outside [0, {symbol_limit})"
            )

        self._encode_interval(
            normalized_cdf[normalized_symbol],
            normalized_cdf[normalized_symbol + 1],
            normalized_cdf[-1],
        )

    def encode_prevalidated_batch(
        self,
        symbols: Sequence[int],
        cdfs: Sequence[Sequence[int]],
        *,
        total: int,
    ) -> None:
        """Encode codec-internal CDFs without repeating public validation.

        The probability quantizer owns the shape, integer, monotonicity, and
        fixed-total checks.  This method is deliberately only used after that
        batch-level validation; callers supplying arbitrary CDFs must use
        :meth:`encode`.
        """

        if self._finished_stream is not None:
            raise RangeCodingError("cannot encode after finalization")
        # Keep the hot arithmetic/bit state in locals across the whole batch.
        # The scalar public path remains the reference implementation.
        low, high = self._low, self._high
        pending = self._pending_underflow_bits
        writer = self._writer
        byte, used, bit_count = writer._current_byte, writer._bits_in_current_byte, writer.bit_count
        append_byte = writer._bytes.append
        for index in range(len(symbols)):
            symbol = int(symbols[index])
            cdf = cdfs[index]
            interval = high - low + 1
            new_high = low + (interval * int(cdf[symbol + 1])) // total - 1
            low += (interval * int(cdf[symbol])) // total
            high = new_high
            if low > high:
                raise InvalidCDFError("cdf resolution collapsed the current range interval")
            while True:
                if high < HALF_RANGE:
                    bit = 0
                elif low >= HALF_RANGE:
                    bit = 1
                    low -= HALF_RANGE
                    high -= HALF_RANGE
                elif low >= QUARTER_RANGE and high < THREE_QUARTER_RANGE:
                    pending += 1
                    low = (low - QUARTER_RANGE) << 1
                    high = ((high - QUARTER_RANGE) << 1) | 1
                    continue
                else:
                    break
                byte = (byte << 1) | bit
                used += 1
                bit_count += 1
                if used == 8:
                    append_byte(byte)
                    byte, used = 0, 0
                # Append deferred identical bits in byte-sized chunks.
                bit_count += pending
                while pending:
                    take = min(pending, 8 - used)
                    byte = (byte << take) | (((1 << take) - 1) if bit == 0 else 0)
                    used += take
                    pending -= take
                    if used == 8:
                        append_byte(byte)
                        byte, used = 0, 0
                low <<= 1
                high = (high << 1) | 1
        self._low, self._high = low, high
        self._pending_underflow_bits = pending
        writer._current_byte, writer._bits_in_current_byte = byte, used
        writer.bit_count = bit_count
        self._symbol_count += len(symbols)

    def _encode_interval(self, symbol_low: int, symbol_high: int, total: int) -> None:
        """Update the arithmetic state from one already validated interval."""

        interval = self._high - self._low + 1
        new_low = self._low + (interval * symbol_low) // total
        new_high = self._low + (interval * symbol_high) // total - 1
        if new_low > new_high:
            raise InvalidCDFError("cdf resolution collapsed the current range interval")
        self._low, self._high = new_low, new_high

        while True:
            if self._high < HALF_RANGE:
                self._emit_with_underflow(0)
            elif self._low >= HALF_RANGE:
                self._emit_with_underflow(1)
                self._low -= HALF_RANGE
                self._high -= HALF_RANGE
            elif self._low >= QUARTER_RANGE and self._high < THREE_QUARTER_RANGE:
                self._pending_underflow_bits += 1
                self._low -= QUARTER_RANGE
                self._high -= QUARTER_RANGE
            else:
                break
            self._low <<= 1
            self._high = (self._high << 1) | 1

        self._symbol_count += 1

    def finish(self) -> bytes:
        """Finalize once and return a self-framed deterministic byte stream."""

        if self._finished_stream is not None:
            return self._finished_stream
        if self._symbol_count == 0:
            payload = b""
        else:
            self._pending_underflow_bits += 1
            self._emit_with_underflow(0 if self._low < QUARTER_RANGE else 1)
            payload = self._writer.finish()

        self._finished_stream = build_range_stream(
            self._symbol_count, self._writer.bit_count, payload
        )
        self._metadata = RangeStreamMetadata(
            symbol_count=self._symbol_count,
            payload_bit_count=self._writer.bit_count,
            payload_byte_count=len(payload),
            total_byte_count=len(self._finished_stream),
        )
        return self._finished_stream


__all__ = ["RangeEncoder"]
