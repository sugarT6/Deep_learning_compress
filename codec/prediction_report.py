"""Bounded diagnostic text reports for the quality model's predictions.

The first line reports top-1 accuracy over every observed quality symbol.  The
remaining lines show the five largest *unquantized* softmax probabilities for
at most the first 1,000 positions, in the order supplied by the encoder.  These
sampled lines are a preview, rather than a representative accuracy estimate.
Quality values are numeric Phred values Q0--Q41, not FASTQ ASCII characters.
"""

from __future__ import annotations

import operator
from pathlib import Path
from typing import List, Optional, Union

import numpy as np


QUALITY_ALPHABET_SIZE = 42
MAX_REPORT_POSITIONS = 1000


class PredictionReport:
    """Accumulate whole-stream accuracy and a bounded top-five preview.

    ``observe`` must receive the final adapted model's logits in actual coding
    order, before probability quantization.  Only the preview is limited by
    ``max_positions``; ``correct``, ``total``, and ``accuracy`` always include
    every observed symbol.  ``accuracy`` is a fraction, or ``None`` for an
    empty stream.  Exact score ties choose the smaller quality value first.
    Neither logits nor symbols are modified or retained.
    """

    def __init__(self, max_positions: int = MAX_REPORT_POSITIONS) -> None:
        if isinstance(max_positions, (bool, np.bool_)):
            raise ValueError("max_positions must be an integer between 0 and 1000")
        try:
            normalized_max = operator.index(max_positions)
        except TypeError as exc:
            raise ValueError(
                "max_positions must be an integer between 0 and 1000"
            ) from exc
        if not 0 <= normalized_max <= MAX_REPORT_POSITIONS:
            raise ValueError("max_positions must be an integer between 0 and 1000")
        self._max_positions = normalized_max
        self._correct = 0
        self._total = 0
        self._preview_lines: List[str] = []

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
    def positions_recorded(self) -> int:
        return len(self._preview_lines)

    def observe(self, logits: np.ndarray, symbols: np.ndarray) -> None:
        """Observe ``[N, 42]`` logits and ``[N]`` integer true qualities.

        All values are validated before any accumulated counts change.  The
        preview uses float64 stable softmax; later batches need only argmax.
        """

        scores = np.asarray(logits)
        observed = np.asarray(symbols)
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
        if not len(observed):
            return

        predicted = scores.argmax(axis=1)
        correct = predicted == observed
        preview_count = min(
            self._max_positions - len(self._preview_lines), len(observed)
        )
        new_lines: List[str] = []
        if preview_count:
            preview_scores = np.asarray(scores[:preview_count], dtype=np.float64)
            # Subtracting extreme finite values may yield -Inf; exp(-Inf)=0 is
            # the correct limiting probability and the row maximum stays 0.
            with np.errstate(over="ignore", under="ignore"):
                weights = np.exp(
                    preview_scores - preview_scores.max(axis=1, keepdims=True)
                )
            probabilities = weights / weights.sum(axis=1, keepdims=True)
            # Sorting the original scores preserves top-five ordering even
            # when tiny probabilities underflow to zero in float64 softmax.
            order = np.argsort(-preview_scores, axis=1, kind="stable")[:, :5]
            for row_index, quality_ids in enumerate(order):
                line = " ".join(
                    f"{int(quality)} {probabilities[row_index, quality]:.12g}"
                    for quality in quality_ids
                )
                if correct[row_index]:
                    line += " yes"
                new_lines.append(line)

        self._correct += int(np.count_nonzero(correct))
        self._total += len(observed)
        self._preview_lines.extend(new_lines)

    def write(self, path: Union[str, Path]) -> None:
        """Write the report, refusing to overwrite an existing file.

        A failed write removes only the partial file created by this call.
        The caller is responsible for creating the parent directory.
        """

        target = Path(path)
        created = False
        try:
            with target.open("x", encoding="utf-8", newline="\n") as stream:
                created = True
                value = f"{100.0 * self.accuracy:.6f}%" if self._total else "N/A"
                stream.write(
                    f"prediction_accuracy={value} "
                    f"correct={self._correct} total={self._total}\n"
                )
                for line in self._preview_lines:
                    stream.write(line + "\n")
        except BaseException:
            if created:
                try:
                    target.unlink()
                except OSError:
                    pass
            raise
