"""Whole-file ranks, true-Q probabilities, and low-probability read positions.

Statistics use the final adapted model's unquantized 42-class softmax, before
coding transformations. Positions are one-based within each read; the encoder
supplies only valid positions, excluding padding.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np


QUALITY_ALPHABET_SIZE = 42
MAX_REPORTED_RANK = 6
PROBABILITY_CHUNK_ROWS = 8192
LOW_PROBABILITY_THRESHOLD = 0.3
_PROBABILITY_EDGES = np.arange(1, 10, dtype=np.float64) / 10


def _percentage(count: int, total: int) -> str:
    return f"{100.0 * (count / total):.6f}%" if total else "N/A"


class PredictionReport:
    """Accumulate whole-file statistics with memory bounded by read length.

    Rank and probability-bin fractions use all Q symbols as their denominator.
    The probability is assigned to the true Q, including top-1 errors. Bins are
    left-closed/right-open, with 1 included in the last bin. Low probability is
    independently defined as p_true <= 0.3. Position ``share`` divides its low
    count by all low Q; position ``rate`` divides by all Q at that position.
    No logits or per-symbol lists are retained.
    """

    def __init__(self) -> None:
        self._correct = 0
        self._total = 0
        self._rank_counts = np.zeros(MAX_REPORTED_RANK + 1, dtype=np.int64)
        self._probability_counts = np.zeros(10, dtype=np.int64)
        self._position_totals = np.zeros(0, dtype=np.int64)
        self._position_low_counts = np.zeros(0, dtype=np.int64)

    @property
    def correct(self) -> int:
        return self._correct

    @property
    def total(self) -> int:
        return self._total

    @property
    def correct_symbols(self) -> int:
        return self._correct

    @property
    def total_symbols(self) -> int:
        return self._total

    @property
    def accuracy(self) -> Optional[float]:
        return self._correct / self._total if self._total else None

    @property
    def probability_counts(self) -> Tuple[int, ...]:
        return tuple(int(value) for value in self._probability_counts)

    @property
    def low_probability_count(self) -> int:
        return int(self._position_low_counts.sum())

    @property
    def position_totals(self) -> Tuple[int, ...]:
        return tuple(int(value) for value in self._position_totals)

    @property
    def low_probability_position_counts(self) -> Tuple[int, ...]:
        return tuple(int(value) for value in self._position_low_counts)

    def observe(
        self, logits: np.ndarray, symbols: np.ndarray, *, positions: np.ndarray,
    ) -> None:
        """Observe [N,42] scores, [N] true Q and [N] one-based read positions.

        Validate every input before updating counters. Float64 stable softmax
        runs in bounded chunks without changing inputs or rerunning the model.
        """
        scores = np.asarray(logits)
        observed = np.asarray(symbols)
        read_positions = np.asarray(positions)
        if scores.ndim != 2 or scores.shape[1] != QUALITY_ALPHABET_SIZE:
            raise ValueError("logits must have shape [N, 42]")
        if scores.dtype.kind not in "fiu":
            raise ValueError("logits must contain real numeric values")
        if not np.isfinite(scores).all():
            raise ValueError("logits must contain only finite values")
        if observed.shape != (scores.shape[0],):
            raise ValueError("symbols must have shape [N], matching logits")
        if observed.dtype.kind not in "iu":
            raise ValueError("symbols must contain integer quality values")
        if np.any(observed < 0) or np.any(observed >= QUALITY_ALPHABET_SIZE):
            raise ValueError("symbols must be quality values between 0 and 41")
        if read_positions.shape != observed.shape or read_positions.dtype.kind not in "iu":
            raise ValueError("positions must contain [N] integer one-based read positions")
        if np.any(read_positions < 1):
            raise ValueError("positions must be positive one-based read positions")
        if not len(observed):
            return
        maximum_position = int(read_positions.max())
        if maximum_position > np.iinfo(np.intp).max:
            raise ValueError("read position exceeds the supported index range")
        position_indices = read_positions.astype(np.intp, copy=False) - 1
        position_totals = np.bincount(position_indices, minlength=maximum_position)
        position_low_counts = np.zeros(maximum_position, dtype=np.int64)
        rank_counts = np.zeros(MAX_REPORTED_RANK + 1, dtype=np.int64)
        probability_counts = np.zeros(10, dtype=np.int64)
        quality_ids = np.arange(QUALITY_ALPHABET_SIZE)[None, :]
        for begin in range(0, len(observed), PROBABILITY_CHUNK_ROWS):
            end = min(begin + PROBABILITY_CHUNK_ROWS, len(observed))
            chunk = scores[begin:end]
            targets = observed[begin:end]
            rows = np.arange(len(targets))
            true_scores = chunk[rows, targets][:, None]
            ranks = np.count_nonzero(chunk > true_scores, axis=1)
            ranks += np.count_nonzero(
                (chunk == true_scores) & (quality_ids < targets[:, None]), axis=1,
            )
            rank_counts += np.bincount(
                np.minimum(ranks, MAX_REPORTED_RANK), minlength=MAX_REPORTED_RANK + 1,
            )
            weights = np.array(chunk, dtype=np.float64, copy=True)
            with np.errstate(over="ignore", under="ignore"):
                weights -= weights.max(axis=1, keepdims=True)
                np.exp(weights, out=weights)
            true_probabilities = weights[rows, targets] / weights.sum(axis=1)
            bins = np.searchsorted(_PROBABILITY_EDGES, true_probabilities, side="right")
            probability_counts += np.bincount(bins, minlength=10)
            low = true_probabilities <= LOW_PROBABILITY_THRESHOLD
            position_low_counts += np.bincount(
                position_indices[begin:end][low], minlength=maximum_position,
            )

        width = max(maximum_position, len(self._position_totals))
        accumulated_totals = np.zeros(width, dtype=np.int64)
        accumulated_low = np.zeros(width, dtype=np.int64)
        accumulated_totals[:len(self._position_totals)] = self._position_totals
        accumulated_low[:len(self._position_low_counts)] = self._position_low_counts
        accumulated_totals[:maximum_position] += position_totals
        accumulated_low[:maximum_position] += position_low_counts
        self._correct += int(rank_counts[0])
        self._total += len(observed)
        self._rank_counts += rank_counts
        self._probability_counts += probability_counts
        self._position_totals = accumulated_totals
        self._position_low_counts = accumulated_low

    def write(self, path: Union[str, Path]) -> None:
        """Write the complete report, refusing to overwrite an existing file."""
        target = Path(path)
        created = False
        try:
            with target.open("x", encoding="utf-8", newline="\n") as stream:
                created = True
                stream.write(
                    f"prediction_accuracy={_percentage(self._correct, self._total)} "
                    f"correct={self._correct} total={self._total}\n"
                )
                labels = tuple(f"rank{rank}" for rank in range(1, MAX_REPORTED_RANK + 1)) + ("rank其它",)
                for label, count in zip(labels, self._rank_counts):
                    stream.write(f"{label} {_percentage(int(count), self._total)}\n")
                for index, count in enumerate(self._probability_counts):
                    closing = "]" if index == 9 else ")"
                    label = f"[{index / 10:.1f},{(index + 1) / 10:.1f}{closing}"
                    stream.write(f"{label} {_percentage(int(count), self._total)}\n")
                low_count = self.low_probability_count
                stream.write(
                    f"low_probability=[0.0,0.3] count={low_count} total={self._total} "
                    f"proportion={_percentage(low_count, self._total)}\n"
                )
                stream.write(
                    "low_probability_positions "
                    "(1-based; share=position_low/all_low; rate=position_low/position_total)\n"
                )
                for index, (low, total) in enumerate(zip(self._position_low_counts, self._position_totals), 1):
                    stream.write(
                        f"position{index} {_percentage(int(low), low_count)} "
                        f"low={int(low)} total={int(total)} rate={_percentage(int(low), int(total))}\n"
                    )
        except BaseException:
            if created:
                try:
                    target.unlink()
                except OSError:
                    pass
            raise
