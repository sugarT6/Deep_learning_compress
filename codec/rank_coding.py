"""Static rank coding ordered by neural integer frequency, then quality id.

``global-quality-rank-v1`` stores all 42 whole-file counts. Its fixed CDF
reserves one unit per rank, then apportions the remaining units by exact
integer largest remainder; equal remainders prefer smaller ranks. Empty
histograms use uniform weights. No learned model or probability correction
is involved. Ranking uses integer frequencies, not unquantized model logits.
"""

from __future__ import annotations

import operator
from typing import Any, Dict, Sequence

import numpy as np

from .probability_quantization import QUALITY_ALPHABET_SIZE, validate_total


RANK_FORMAT = "global-quality-rank-v1"
RANK_ORDER = "descending_neural_integer_frequency_then_quality"
_MAX_COUNT = (1 << 63) - 1


def _counts(values: Sequence[int]) -> list:
    if isinstance(values, (str, bytes)):
        raise ValueError("rank counts must contain 42 integers")
    try:
        entries = list(values)
    except TypeError as exc:
        raise ValueError("rank counts must contain 42 integers") from exc
    if len(entries) != QUALITY_ALPHABET_SIZE:
        raise ValueError("rank counts must contain 42 integers")
    result = []
    for entry in entries:
        if isinstance(entry, (bool, np.bool_)):
            raise ValueError("rank counts must be nonnegative integers")
        try:
            value = operator.index(entry)
        except TypeError as exc:
            raise ValueError("rank counts must be nonnegative integers") from exc
        if not 0 <= value <= _MAX_COUNT:
            raise ValueError("rank count outside the supported int64 range")
        result.append(int(value))
    return result


def _frequency_rows(cdfs: np.ndarray) -> np.ndarray:
    rows = np.asarray(cdfs)
    if rows.ndim != 2 or rows.shape[1] != QUALITY_ALPHABET_SIZE + 1:
        raise ValueError("neural CDFs must have shape [N, 43]")
    if rows.dtype.kind not in "iu":
        raise ValueError("CDFs must contain integers")
    if rows.size and (np.any(rows < 0) or np.any(rows > _MAX_COUNT)):
        raise ValueError("CDF values outside the supported int64 range")
    rows = rows.astype(np.int64, copy=False)
    frequencies = np.diff(rows, axis=1)
    if np.any(rows[:, 0] != 0) or np.any(frequencies <= 0):
        raise ValueError("CDFs must start at zero and increase strictly")
    return frequencies


def quality_order(cdfs: np.ndarray) -> np.ndarray:
    """Return quality ids ordered by descending integer frequency per row."""
    return np.argsort(-_frequency_rows(cdfs), axis=1, kind="stable")


def ranks_from_cdfs(cdfs: np.ndarray, symbols: np.ndarray) -> np.ndarray:
    """Map true Q0--Q41 symbols to zero-based ranks in cycle-major row order."""
    order = quality_order(cdfs)
    observed = np.asarray(symbols)
    if observed.shape != (len(order),) or observed.dtype.kind not in "iu":
        raise ValueError("quality symbols must be an integer vector matching CDFs")
    if np.any(observed < 0) or np.any(observed >= QUALITY_ALPHABET_SIZE):
        raise ValueError("quality symbols must be in Q0--Q41")
    return np.argmax(order == observed[:, None], axis=1).astype(np.uint8)


def make_rank_metadata(counts: Sequence[int], total: int) -> Dict[str, Any]:
    """Build canonical metadata; ``total`` comes from probability_quantization."""
    validate_total(total)
    return {"format": RANK_FORMAT, "order": RANK_ORDER, "counts": _counts(counts)}


def validate_rank_metadata(metadata: Dict[str, Any], total: int,
                           symbol_count: int) -> np.ndarray:
    """Validate the wire contract and reconstruct its exact fixed int64 CDF."""
    total = validate_total(total)
    if not isinstance(metadata, dict) or set(metadata) != {"format", "order", "counts"}:
        raise ValueError("rank metadata must contain exactly format, order and counts")
    if metadata["format"] != RANK_FORMAT or metadata["order"] != RANK_ORDER:
        raise ValueError("unsupported rank format or ordering")
    if not isinstance(metadata["counts"], list):
        raise ValueError("rank metadata counts must be a list")
    counts = _counts(metadata["counts"])
    if isinstance(symbol_count, (bool, np.bool_)):
        raise ValueError("symbol count must be a nonnegative integer")
    try:
        symbol_count = operator.index(symbol_count)
    except TypeError as exc:
        raise ValueError("symbol count must be a nonnegative integer") from exc
    if symbol_count < 0 or sum(counts) != symbol_count:
        raise ValueError("rank counts do not match the quality symbol count")
    weights = counts if symbol_count else [1] * QUALITY_ALPHABET_SIZE
    denominator = sum(weights)
    allocations = [divmod(weight * (total - QUALITY_ALPHABET_SIZE), denominator)
                   for weight in weights]
    frequencies = [1 + quotient for quotient, _ in allocations]
    remaining = total - sum(frequencies)
    order = sorted(range(QUALITY_ALPHABET_SIZE),
                   key=lambda rank: (-allocations[rank][1], rank))
    for rank in order[:remaining]:
        frequencies[rank] += 1
    return np.asarray([0] + list(np.cumsum(frequencies, dtype=np.int64)), dtype=np.int64)


def global_rank_bits(counts: Sequence[int], cdf: np.ndarray) -> float:
    """Return the whole-file ideal bit count under the quantized rank CDF."""
    observed = np.asarray(_counts(counts), dtype=np.float64)
    values = np.asarray(cdf)
    if values.shape != (QUALITY_ALPHABET_SIZE + 1,):
        raise ValueError("rank CDF must have shape [43]")
    frequencies = _frequency_rows(values[None, :])[0]
    total = validate_total(int(values[-1]))
    return float(-np.dot(observed, np.log2(frequencies.astype(np.float64) / total)))
