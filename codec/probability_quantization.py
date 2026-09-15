"""Deterministic Q0--Q41 probability quantization.

Version 1 uses a fixed default total of ``2**16``.  Every quality id first
receives frequency one.  The remaining mass is apportioned with the largest
remainder method: take the floor of every normalized quota, then give the
remaining units to descending fractional remainders.  Exact ties are resolved
by ascending quality id.  The calculation is performed with finite IEEE-754
float64 values on CPU; this rule is part of the quantizer contract.
"""

from __future__ import annotations

import math
import operator
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np


QUALITY_ALPHABET_SIZE = 42
QUANTIZATION_VERSION = 1
TOTAL = 1 << 16
MAX_TOTAL = 1 << 30


class ProbabilityQuantizationError(ValueError):
    """Raised when probabilities, logits, frequencies, or TOTAL are invalid."""


def _validate_total(total: int) -> int:
    if isinstance(total, (bool, np.bool_)):
        raise ProbabilityQuantizationError("TOTAL must be an integer")
    try:
        normalized_total = operator.index(total)
    except TypeError as exc:
        raise ProbabilityQuantizationError("TOTAL must be an integer") from exc
    if normalized_total < QUALITY_ALPHABET_SIZE:
        raise ProbabilityQuantizationError(
            f"TOTAL must be at least {QUALITY_ALPHABET_SIZE} so every symbol "
            "can receive frequency one"
        )
    if normalized_total > MAX_TOTAL:
        raise ProbabilityQuantizationError(
            f"TOTAL must not exceed {MAX_TOTAL} for the version-1 range coder"
        )
    return normalized_total


def validate_total(total: int) -> int:
    """Return a validated version-1 quantization total."""

    return _validate_total(total)


def _as_quality_vector(values: Sequence[float], *, name: str) -> np.ndarray:
    try:
        vector = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ProbabilityQuantizationError(
            f"{name} must contain numeric values"
        ) from exc
    if vector.shape != (QUALITY_ALPHABET_SIZE,):
        raise ProbabilityQuantizationError(
            f"{name} must have shape ({QUALITY_ALPHABET_SIZE},), got {vector.shape}"
        )
    if not np.isfinite(vector).all():
        raise ProbabilityQuantizationError(f"{name} must not contain NaN or Inf")
    return vector


def _largest_remainder_frequencies(
    normalized_probabilities: np.ndarray, total: int
) -> Tuple[int, ...]:
    remaining = total - QUALITY_ALPHABET_SIZE
    if remaining == 0:
        return (1,) * QUALITY_ALPHABET_SIZE

    quotas = normalized_probabilities * float(remaining)
    floors = np.floor(quotas).astype(np.int64)
    fractions = quotas - floors
    units_left = remaining - int(floors.sum())

    if units_left > 0:
        order = sorted(
            range(QUALITY_ALPHABET_SIZE),
            key=lambda quality_id: (-float(fractions[quality_id]), quality_id),
        )
        for quality_id in order[:units_left]:
            floors[quality_id] += 1
    elif units_left < 0:  # Defensive correction for an extreme float64 sum error.
        order = sorted(
            (
                quality_id
                for quality_id in range(QUALITY_ALPHABET_SIZE)
                if floors[quality_id] > 0
            ),
            key=lambda quality_id: (float(fractions[quality_id]), -quality_id),
        )
        if -units_left > len(order):
            raise ProbabilityQuantizationError(
                "float64 apportionment produced an invalid remainder"
            )
        for quality_id in order[: -units_left]:
            floors[quality_id] -= 1

    frequencies = tuple(int(value) + 1 for value in floors)
    if min(frequencies) < 1 or sum(frequencies) != total:
        raise ProbabilityQuantizationError(
            "internal quantization error: frequencies violate the fixed-total contract"
        )
    return frequencies


def probabilities_to_frequencies(
    probabilities: Sequence[float], *, total: int = TOTAL
) -> Tuple[int, ...]:
    """Quantize one length-42 probability/weight vector into integer frequencies."""

    normalized_total = _validate_total(total)
    vector = _as_quality_vector(probabilities, name="probabilities")
    if np.any(vector < 0.0):
        raise ProbabilityQuantizationError("probabilities must be nonnegative")
    maximum = float(vector.max())
    if maximum == 0.0:
        raise ProbabilityQuantizationError(
            "probabilities must contain at least one positive value"
        )
    scaled = vector / maximum
    denominator = float(scaled.sum(dtype=np.float64))
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise ProbabilityQuantizationError("probabilities have an invalid sum")
    normalized = scaled / denominator
    return _largest_remainder_frequencies(normalized, normalized_total)


def logits_to_frequencies(
    logits: Sequence[float], *, total: int = TOTAL
) -> Tuple[int, ...]:
    """Apply stable float64 softmax and quantize one length-42 logit vector."""

    normalized_total = _validate_total(total)
    vector = _as_quality_vector(logits, name="logits")
    shifted = vector - float(vector.max())
    weights = np.exp(shifted)
    denominator = float(weights.sum(dtype=np.float64))
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise ProbabilityQuantizationError("logits produced an invalid softmax sum")
    normalized = weights / denominator
    return _largest_remainder_frequencies(normalized, normalized_total)


def frequencies_to_cdf(
    frequencies: Iterable[int], *, total: Optional[int] = None
) -> Tuple[int, ...]:
    """Convert exactly 42 positive integer frequencies to a 43-entry CDF."""

    values = tuple(frequencies)
    if len(values) != QUALITY_ALPHABET_SIZE:
        raise ProbabilityQuantizationError(
            f"frequencies must contain {QUALITY_ALPHABET_SIZE} values"
        )
    normalized = []
    for value in values:
        if isinstance(value, (bool, np.bool_)):
            raise ProbabilityQuantizationError("frequencies must be integers")
        try:
            integer = operator.index(value)
        except TypeError as exc:
            raise ProbabilityQuantizationError(
                "frequencies must be integers"
            ) from exc
        if integer < 1:
            raise ProbabilityQuantizationError("every frequency must be at least one")
        normalized.append(integer)

    observed_total = sum(normalized)
    if total is not None and observed_total != _validate_total(total):
        raise ProbabilityQuantizationError(
            f"frequency sum {observed_total} does not equal TOTAL {total}"
        )
    _validate_total(observed_total)

    cdf = [0]
    cumulative = 0
    for frequency in normalized:
        cumulative += frequency
        cdf.append(cumulative)
    return tuple(cdf)


def probabilities_to_cdf(
    probabilities: Sequence[float], *, total: int = TOTAL
) -> Tuple[int, ...]:
    """Quantize probabilities and return the canonical 43-entry CDF."""

    return frequencies_to_cdf(
        probabilities_to_frequencies(probabilities, total=total), total=total
    )


def logits_to_cdf(
    logits: Sequence[float], *, total: int = TOTAL
) -> Tuple[int, ...]:
    """Quantize logits and return the canonical 43-entry CDF."""

    return frequencies_to_cdf(logits_to_frequencies(logits, total=total), total=total)


def logits_to_cdfs(
    logits: Sequence[Sequence[float]], *, total: int = TOTAL
) -> np.ndarray:
    """Batch-quantize ``[N, 42]`` logits into an ``int64 [N, 43]`` CDF array.

    This is the production equivalent of calling :func:`logits_to_cdf` for
    every row.  Float64 softmax, largest-remainder allocation, stable quality-id
    tie breaking, and the minimum frequency of one are applied along each row.
    """

    normalized_total = _validate_total(total)
    try:
        matrix = np.asarray(logits, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ProbabilityQuantizationError(
            "logits must contain numeric values"
        ) from exc
    if matrix.ndim != 2 or matrix.shape[1] != QUALITY_ALPHABET_SIZE:
        raise ProbabilityQuantizationError(
            "logits must have shape [N, "
            f"{QUALITY_ALPHABET_SIZE}], got {matrix.shape}"
        )
    if not np.isfinite(matrix).all():
        raise ProbabilityQuantizationError("logits must not contain NaN or Inf")

    row_count = matrix.shape[0]
    cdfs = np.zeros(
        (row_count, QUALITY_ALPHABET_SIZE + 1), dtype=np.int64
    )
    if row_count == 0:
        return cdfs
    if normalized_total == QUALITY_ALPHABET_SIZE:
        cdfs[:, 1:] = np.arange(
            1, QUALITY_ALPHABET_SIZE + 1, dtype=np.int64
        )
        return cdfs

    shifted = matrix - matrix.max(axis=1, keepdims=True)
    weights = np.exp(shifted)
    denominators = weights.sum(axis=1, dtype=np.float64, keepdims=True)
    if not np.isfinite(denominators).all() or np.any(denominators <= 0.0):
        raise ProbabilityQuantizationError("logits produced an invalid softmax sum")

    remaining = normalized_total - QUALITY_ALPHABET_SIZE
    quotas = (weights / denominators) * float(remaining)
    floors = np.floor(quotas).astype(np.int64)
    fractions = quotas - floors
    units_left = remaining - floors.sum(axis=1, dtype=np.int64)

    positive_rows = np.flatnonzero(units_left > 0)
    if positive_rows.size:
        positive_fractions = fractions[positive_rows]
        order = np.argsort(-positive_fractions, axis=1, kind="stable")
        ranks = np.empty_like(order)
        row_indices = np.arange(positive_rows.size, dtype=np.int64)[:, None]
        ranks[row_indices, order] = np.arange(
            QUALITY_ALPHABET_SIZE, dtype=np.int64
        )[None, :]
        floors[positive_rows] += ranks < units_left[positive_rows, None]

    # This branch is only a defensive correction for an extreme float64 sum
    # error.  Keep the scalar tie rule exact instead of complicating the common
    # vectorized path.
    for row in np.flatnonzero(units_left < 0):
        eligible = np.flatnonzero(floors[row] > 0).tolist()
        order = sorted(
            eligible,
            key=lambda quality_id: (
                float(fractions[row, quality_id]),
                -quality_id,
            ),
        )
        correction = -int(units_left[row])
        if correction > len(order):
            raise ProbabilityQuantizationError(
                "float64 apportionment produced an invalid remainder"
            )
        floors[row, order[:correction]] -= 1

    frequencies = floors + 1
    if np.any(frequencies < 1) or np.any(
        frequencies.sum(axis=1, dtype=np.int64) != normalized_total
    ):
        raise ProbabilityQuantizationError(
            "internal quantization error: frequencies violate the fixed-total contract"
        )
    cdfs[:, 1:] = np.cumsum(frequencies, axis=1, dtype=np.int64)
    return cdfs


def logits_symbols_bits(
    logits: Sequence[Sequence[float]], symbols: Sequence[int]
) -> float:
    """Return stable float64 softmax cross-entropy bits for selected symbols.

    Unlike :func:`quantized_symbols_bits`, this diagnostic does not construct
    integer frequencies or CDFs.  It is used for the optional neural-only
    baseline without duplicating the production largest-remainder quantizer.
    """

    try:
        matrix = np.asarray(logits, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ProbabilityQuantizationError(
            "logits must contain numeric values"
        ) from exc
    symbol_array = np.asarray(symbols)
    if matrix.ndim != 2 or matrix.shape[1] != QUALITY_ALPHABET_SIZE:
        raise ProbabilityQuantizationError(
            "logits must have shape [N, "
            f"{QUALITY_ALPHABET_SIZE}], got {matrix.shape}"
        )
    if symbol_array.shape != (matrix.shape[0],):
        raise ProbabilityQuantizationError("symbols must have shape [N]")
    if symbol_array.dtype.kind not in "iu":
        raise ProbabilityQuantizationError("symbols must be integers")
    if not np.isfinite(matrix).all():
        raise ProbabilityQuantizationError("logits must not contain NaN or Inf")
    if np.any(symbol_array < 0) or np.any(symbol_array >= QUALITY_ALPHABET_SIZE):
        raise ProbabilityQuantizationError("symbols must be in Q0..Q41")
    if matrix.shape[0] == 0:
        return 0.0

    maxima = matrix.max(axis=1)
    log_normalizers = maxima + np.log(
        np.exp(matrix - maxima[:, None]).sum(axis=1, dtype=np.float64)
    )
    selected = matrix[
        np.arange(matrix.shape[0]), symbol_array.astype(np.int64, copy=False)
    ]
    return float(((log_normalizers - selected) / math.log(2.0)).sum(dtype=np.float64))


def quantized_symbols_bits(
    symbols: Sequence[int], cdfs: Sequence[Sequence[int]], *, total: int = TOTAL
) -> float:
    """Return the vectorized theoretical bit sum for precomputed quality CDFs."""

    normalized_total = _validate_total(total)
    symbol_array = np.asarray(symbols)
    cdf_array = np.asarray(cdfs)
    if symbol_array.ndim != 1:
        raise ProbabilityQuantizationError("symbols must have shape [N]")
    if cdf_array.shape != (
        symbol_array.size,
        QUALITY_ALPHABET_SIZE + 1,
    ):
        raise ProbabilityQuantizationError(
            f"cdfs must have shape [N, {QUALITY_ALPHABET_SIZE + 1}]"
        )
    if symbol_array.dtype.kind not in "iu" or cdf_array.dtype.kind not in "iu":
        raise ProbabilityQuantizationError("symbols and cdfs must be integers")
    if np.any(symbol_array < 0) or np.any(symbol_array >= QUALITY_ALPHABET_SIZE):
        raise ProbabilityQuantizationError("symbols must be in Q0..Q41")
    if cdf_array.size:
        if np.any(cdf_array[:, 0] != 0) or np.any(
            cdf_array[:, -1] != normalized_total
        ):
            raise ProbabilityQuantizationError(
                "every cdf must start at zero and end at TOTAL"
            )
        frequencies = np.diff(cdf_array, axis=1)
        if np.any(frequencies <= 0):
            raise ProbabilityQuantizationError(
                "every cdf frequency must be at least one"
            )
        selected = frequencies[
            np.arange(symbol_array.size), symbol_array.astype(np.int64, copy=False)
        ]
        return float(
            -np.log2(selected.astype(np.float64) / normalized_total).sum(
                dtype=np.float64
            )
        )
    return 0.0


def quantized_symbol_bits(symbol: int, cdf: Sequence[int]) -> float:
    """Return ``-log2(freq[symbol] / total)`` for a validated quality CDF."""

    values = tuple(cdf)
    if len(values) != QUALITY_ALPHABET_SIZE + 1:
        raise ProbabilityQuantizationError(
            f"cdf must contain {QUALITY_ALPHABET_SIZE + 1} values"
        )
    frequencies = tuple(values[index + 1] - values[index] for index in range(42))
    canonical = frequencies_to_cdf(frequencies, total=values[-1])
    if canonical != values:
        raise ProbabilityQuantizationError("cdf must start at zero and be cumulative")
    if isinstance(symbol, (bool, np.bool_)):
        raise ProbabilityQuantizationError("symbol must be an integer quality id")
    try:
        quality_id = operator.index(symbol)
    except TypeError as exc:
        raise ProbabilityQuantizationError(
            "symbol must be an integer quality id"
        ) from exc
    if quality_id < 0 or quality_id >= QUALITY_ALPHABET_SIZE:
        raise ProbabilityQuantizationError("symbol must be in Q0..Q41")
    return -math.log2(frequencies[quality_id] / values[-1])


__all__ = [
    "MAX_TOTAL",
    "ProbabilityQuantizationError",
    "QUALITY_ALPHABET_SIZE",
    "QUANTIZATION_VERSION",
    "TOTAL",
    "frequencies_to_cdf",
    "logits_symbols_bits",
    "logits_to_cdf",
    "logits_to_cdfs",
    "logits_to_frequencies",
    "probabilities_to_cdf",
    "probabilities_to_frequencies",
    "quantized_symbol_bits",
    "quantized_symbols_bits",
    "validate_total",
]
