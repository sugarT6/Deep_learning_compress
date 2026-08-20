#!/usr/bin/env python3
"""Print the observed quality-id distribution of explicit HDF5 files."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np

QUALITY_ALPHABET_SIZE = 95
DEFAULT_CHUNK_ROWS = 1_000_000


def count_quality_ids(
    paths: Iterable[Path],
    *,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
) -> np.ndarray:
    """Count ``/observed`` quality ids without loading whole files into memory."""

    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")

    counts = np.zeros(QUALITY_ALPHABET_SIZE, dtype=np.int64)
    for path in paths:
        with h5py.File(path, "r") as handle:
            if "/observed" not in handle:
                raise ValueError(f"{path}: missing required dataset /observed")
            observed = handle["/observed"]
            if observed.ndim != 1:
                raise ValueError(f"{path}: /observed must be one-dimensional")

            for start in range(0, int(observed.shape[0]), chunk_rows):
                values = np.asarray(observed[start : start + chunk_rows])
                if not np.issubdtype(values.dtype, np.integer):
                    raise ValueError(f"{path}: /observed must contain integer ids")
                if values.size and (
                    int(values.min()) < 0
                    or int(values.max()) >= QUALITY_ALPHABET_SIZE
                ):
                    raise ValueError(
                        f"{path}: observed quality id outside [0, "
                        f"{QUALITY_ALPHABET_SIZE - 1}]"
                    )
                counts += np.bincount(
                    values.astype(np.int64, copy=False),
                    minlength=QUALITY_ALPHABET_SIZE,
                )
    return counts


def format_quality_distribution(
    counts: np.ndarray,
    *,
    entries_per_line: int = 5,
) -> str:
    """Format nonzero ``Q<id> percentage`` entries in id order."""

    if entries_per_line <= 0:
        raise ValueError("entries_per_line must be positive")
    if counts.ndim != 1 or np.any(counts < 0):
        raise ValueError("counts must be a one-dimensional nonnegative array")

    total = int(counts.sum())
    if total == 0:
        raise ValueError("selected dataset contains no quality values")

    entries = [
        f"Q{quality_id} {100.0 * int(count) / total:.2f}%"
        for quality_id, count in enumerate(counts)
        if count
    ]
    lines = [
        "\t".join(entries[start : start + entries_per_line])
        for start in range(0, len(entries), entries_per_line)
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Print the aggregate distribution of true quality ids from one or "
            "more quality-model HDF5 files."
        )
    )
    parser.add_argument(
        "h5_files",
        nargs="+",
        type=Path,
        help="one or more quality-model HDF5 file paths; distributions are merged",
    )
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=DEFAULT_CHUNK_ROWS,
        help="number of /observed rows read at a time; default: 1000000",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        counts = count_quality_ids(
            args.h5_files,
            chunk_rows=args.chunk_rows,
        )
        print(format_quality_distribution(counts))
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
