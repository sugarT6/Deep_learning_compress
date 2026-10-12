"""ctypes bridge for the optional, single-threaded QRC1 C++ batch kernel."""

from __future__ import annotations

import ctypes
from functools import lru_cache
import hashlib
import operator
from pathlib import Path
import platform

import numpy as np

from ._range_common import InvalidCDFError, MAX_CDF_TOTAL, RangeCodingError


class NativeRangeUnavailable(RangeCodingError):
    """The kernel has not been built for this source revision/platform."""


def native_source_path() -> Path:
    return Path(__file__).with_name("native") / "range_encoder.cpp"


def native_library_path() -> Path:
    source = native_source_path()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    return source.parent / "_build" / f"libfqrc-{platform.machine()}-{digest}.so"


class _State(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in
                ("low", "high", "pending", "byte", "used", "bit_count")]


_I64 = ctypes.POINTER(ctypes.c_int64)
_U8 = ctypes.POINTER(ctypes.c_uint8)
_SIZE_OUT = ctypes.POINTER(ctypes.c_size_t)


def _integers(values, ndim):
    array = np.asarray(values)
    if array.ndim != ndim or array.dtype.kind not in "iu":
        raise RangeCodingError(f"native coder requires a {ndim}D integer array")
    # Usually a zero-copy view: codec quantizers already return native int64.
    array = np.require(array, dtype=np.int64, requirements=["C", "A"])
    return array


class _NativeKernel:
    def __init__(self, path):
        self.library = ctypes.CDLL(str(path))
        self.library.fqrc_abi_version.argtypes = []
        self.library.fqrc_abi_version.restype = ctypes.c_uint
        if self.library.fqrc_abi_version() != 1:
            raise ValueError("native range coder ABI mismatch")
        self.intervals = self.library.fqrc_encode_intervals
        self.intervals.argtypes = [ctypes.POINTER(_State), _I64, _I64, _I64,
                                  ctypes.c_size_t, _U8, ctypes.c_size_t, _SIZE_OUT]
        self.intervals.restype = ctypes.c_int
        self.cdfs = self.library.fqrc_encode_cdfs
        self.cdfs.argtypes = [ctypes.POINTER(_State), _I64, _I64, ctypes.c_size_t,
                             ctypes.c_size_t, ctypes.c_int64, _U8,
                             ctypes.c_size_t, _SIZE_OUT]
        self.cdfs.restype = ctypes.c_int

    def _encode(self, encoder, function, arrays, parameters, count):
        writer = encoder._writer
        state = _State(encoder._low, encoder._high, encoder._pending_underflow_bits,
                       writer._current_byte, writer._bits_in_current_byte, writer.bit_count)
        # A valid 32-bit update needs at most 32 renormalization shifts. Old
        # pending/partial bits may also flush; final partial bits stay in state.
        capacity = 4 * count + (state.pending + state.used + 7) // 8 + 1
        output = bytearray(capacity)
        buffer = (ctypes.c_uint8 * capacity).from_buffer(output)
        written = ctypes.c_size_t()
        status = function(ctypes.byref(state),
                          *(a.ctypes.data_as(_I64) for a in arrays),
                          *parameters, buffer, capacity, ctypes.byref(written))
        if status == 1:
            raise InvalidCDFError("invalid or collapsed native range interval")
        if status == 3:
            raise RangeCodingError("symbol is outside the CDF alphabet")
        if status != 0:
            raise RangeCodingError(f"native range coder failed (status {status})")
        writer._bytes.extend(memoryview(output)[:written.value])
        encoder._low, encoder._high = state.low, state.high
        encoder._pending_underflow_bits = state.pending
        writer._current_byte, writer._bits_in_current_byte = state.byte, state.used
        writer.bit_count = state.bit_count
        encoder._symbol_count += count

    def encode_intervals(self, encoder, lower, upper, totals):
        arrays = [_integers(a, 1) for a in (lower, upper, totals)]
        if not len(arrays[0]) == len(arrays[1]) == len(arrays[2]):
            raise RangeCodingError("interval array lengths differ")
        self._encode(encoder, self.intervals, arrays, [len(arrays[0])], len(arrays[0]))

    def encode_cdfs(self, encoder, symbols, cdfs, total):
        symbols, cdfs = _integers(symbols, 1), _integers(cdfs, 2)
        if cdfs.shape[0] != len(symbols) or cdfs.shape[1] < 2:
            raise RangeCodingError("symbol/CDF array shapes differ")
        total = operator.index(total)
        if not 1 <= total <= MAX_CDF_TOTAL:
            raise InvalidCDFError("invalid CDF total")
        self._encode(encoder, self.cdfs, [symbols, cdfs],
                     [len(symbols), cdfs.shape[1], total], len(symbols))


@lru_cache(maxsize=4)
def _load(path):
    return _NativeKernel(path)


def load_native_kernel():
    path = native_library_path()
    try:
        return _load(str(path))
    except (OSError, AttributeError, ValueError) as exc:
        raise NativeRangeUnavailable(
            f"native range coder unavailable ({path}); run: python -m codec.build_native"
        ) from exc
