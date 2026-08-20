#!/usr/bin/env python3
"""Print the observed quality-id distribution of registered datasets."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np

from dataset_registry import resolve_dataset_files


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
    """Format nonzero ``quality_id proportion`` entries in id order."""

    if entries_per_line <= 0:
        raise ValueError("entries_per_line must be positive")
    if counts.ndim != 1 or np.any(counts < 0):
        raise ValueError("counts must be a one-dimensional nonnegative array")

    total = int(counts.sum())
    if total == 0:
        raise ValueError("selected dataset contains no quality values")

    entries = [
        f"{quality_id} {int(count) / total:.8f}"
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
            "more registered datasets. Proportions are decimals in [0, 1]."
        )
    )
    parser.add_argument(
        "--datasets",
        required=True,
        help=(
            "comma-separated registered datasets/groups, for example novaseq, "
            "dnbseq_t7, illumina, or mixed"
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="root containing registered h5/ data; default: data",
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
        files = resolve_dataset_files(
            args.datasets,
            args.data_root,
            require_h5=True,
        )
        counts = count_quality_ids(
            (item.h5_path for item in files),
            chunk_rows=args.chunk_rows,
        )
        print(format_quality_distribution(counts))
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
