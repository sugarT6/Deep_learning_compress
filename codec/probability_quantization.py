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
    "logits_to_cdf",
    "logits_to_frequencies",
    "probabilities_to_cdf",
    "probabilities_to_frequencies",
    "quantized_symbol_bits",
    "validate_total",
]
