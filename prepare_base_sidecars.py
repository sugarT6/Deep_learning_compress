#!/usr/bin/env python3
"""Build random-access full-read base sidecars aligned with quality-model H5 files."""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path
from typing import BinaryIO, Iterator

import h5py
import numpy as np

from dataset_registry import resolve_dataset_files

from sequence_residual_transformer_model import (
    BASE_A_TOKEN,
    BASE_C_TOKEN,
    BASE_G_TOKEN,
    BASE_N_TOKEN,
    BASE_OTHER_TOKEN,
    BASE_PAD_TOKEN,
    BASE_T_TOKEN,
    base_sidecar_path_for_h5,
    discover_h5_files,
    inspect_base_sidecar,
)


BASE_LOOKUP = np.full(256, BASE_OTHER_TOKEN, dtype=np.uint8)
for character, token in (
    (b"A", BASE_A_TOKEN),
    (b"C", BASE_C_TOKEN),
    (b"G", BASE_G_TOKEN),
    (b"T", BASE_T_TOKEN),
    (b"N", BASE_N_TOKEN),
):
    BASE_LOOKUP[character[0]] = token
    BASE_LOOKUP[character.lower()[0]] = token


def _strip_line_ending(line: bytes) -> bytes:
    if line.endswith(b"\n"):
        line = line[:-1]
    if line.endswith(b"\r"):
        line = line[:-1]
    return line


def _open_fastq(path: Path) -> BinaryIO:
    if path.name.endswith(".gz"):
        return gzip.open(path, "rb")
    return path.open("rb")


def iter_fastq_records(path: Path) -> Iterator[tuple[bytes, bytes, bytes]]:
    """Yield strict four-line FASTQ records as header, sequence, quality."""

    with _open_fastq(path) as handle:
        record_index = 0
        while True:
            header = handle.readline()
            if not header:
                return
            sequence = handle.readline()
            plus = handle.readline()
            quality = handle.readline()
            if not sequence or not plus or not quality:
                raise ValueError(f"{path}: truncated FASTQ record {record_index}")

            header = _strip_line_ending(header)
            sequence = _strip_line_ending(sequence)
            plus = _strip_line_ending(plus)
            quality = _strip_line_ending(quality)
            if not header.startswith(b"@"):
                raise ValueError(f"{path}: record {record_index} header does not start with @")
            if not plus.startswith(b"+"):
                raise ValueError(f"{path}: record {record_index} separator does not start with +")
            if len(sequence) != len(quality):
                raise ValueError(
                    f"{path}: record {record_index} sequence/quality length mismatch "
                    f"({len(sequence)} != {len(quality)})"
                )
            yield header, sequence, quality
            record_index += 1


def fastq_path_for_h5(h5_path: Path, fastq_dir: Path) -> Path:
    suffix = ".qual_model.h5"
    if not h5_path.name.endswith(suffix):
        raise ValueError(f"{h5_path}: expected filename ending in {suffix}")
    return fastq_dir / h5_path.name[: -len(suffix)]


def encode_bases(sequence: bytes) -> np.ndarray:
    ascii_values = np.frombuffer(sequence, dtype=np.uint8)
    return BASE_LOOKUP[ascii_values]


def create_base_sidecar(
    h5_path: Path,
    fastq_path: Path,
    output_path: Path,
    force: bool = False,
    flush_bases: int = 1_048_576,
) -> None:
    """Create one sidecar while validating every FASTQ/H5 body position."""

    if flush_bases <= 0:
        raise ValueError("flush_bases must be positive")
    if output_path.exists() and not force:
        raise FileExistsError(f"{output_path} already exists; pass --force to replace it")
    if not fastq_path.is_file():
        raise FileNotFoundError(fastq_path)

    with h5py.File(h5_path, "r") as quality_handle:
        for name in ("/observed", "/read_offsets"):
            if name not in quality_handle:
                raise ValueError(f"{h5_path}: missing required dataset {name}")
        observed = np.asarray(quality_handle["/observed"][:], dtype=np.int16)
        body_offsets = np.asarray(quality_handle["/read_offsets"][:], dtype=np.int64)

    body_lengths = np.diff(body_offsets)
    if body_offsets.ndim != 1 or body_offsets.size < 2:
        raise ValueError(f"{h5_path}: invalid /read_offsets")
    if int(body_offsets[0]) != 0 or int(body_offsets[-1]) != int(observed.size):
        raise ValueError(f"{h5_path}: /read_offsets does not match /observed")
    if np.any(body_lengths < 0):
        raise ValueError(f"{h5_path}: /read_offsets must be non-decreasing")

    read_count = int(body_lengths.size)
    base_offsets = np.zeros(read_count + 1, dtype=np.int64)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    if temporary_path.exists():
        temporary_path.unlink()

    try:
        with h5py.File(temporary_path, "w") as output:
            output.attrs["format"] = "fastq_full_base_sidecar_v1"
            output.attrs["source_h5_name"] = h5_path.name
            output.attrs["source_fastq_name"] = fastq_path.name
            output.attrs["base_encoding"] = "A=0,C=1,G=2,T=3,N=4,other=5,pad=6"
            output.attrs["complete_base_read_available_before_quality"] = True
            output.attrs["body_length_is_quality_side_information"] = True

            base_dataset = output.create_dataset(
                "/base_ids",
                shape=(0,),
                maxshape=(None,),
                chunks=(flush_bases,),
                dtype=np.uint8,
            )
            pending: list[np.ndarray] = []
            pending_count = 0
            written_count = 0
            record_count = 0

            def flush_pending() -> None:
                nonlocal pending_count, written_count
                if not pending:
                    return
                values = np.concatenate(pending)
                new_count = written_count + int(values.size)
                base_dataset.resize((new_count,))
                base_dataset[written_count:new_count] = values
                written_count = new_count
                pending.clear()
                pending_count = 0

            for record_count, (_, sequence, quality) in enumerate(
                iter_fastq_records(fastq_path),
                start=1,
            ):
                read_index = record_count - 1
                if read_index >= read_count:
                    raise ValueError(
                        f"{fastq_path}: contains more than {read_count} H5-aligned reads"
                    )

                body_start = int(body_offsets[read_index])
                body_stop = int(body_offsets[read_index + 1])
                body_length = body_stop - body_start
                if len(sequence) < body_length:
                    raise ValueError(
                        f"{fastq_path}: read {read_index} has {len(sequence)} bases but "
                        f"{body_length} body qualities"
                    )

                quality_ascii = np.frombuffer(quality, dtype=np.uint8).astype(np.int16)
                quality_ids = quality_ascii - 33
                if np.any(quality_ids < 0) or np.any(quality_ids > 94):
                    raise ValueError(f"{fastq_path}: read {read_index} quality id out of [0, 94]")
                if not np.array_equal(quality_ids[:body_length], observed[body_start:body_stop]):
                    raise ValueError(
                        f"{fastq_path}: read {read_index} quality prefix does not match H5"
                    )
                if body_length < len(quality) and quality[body_length:] != b"#" * (
                    len(quality) - body_length
                ):
                    raise ValueError(
                        f"{fastq_path}: read {read_index} H5 suffix is not entirely Q2 '#'"
                    )

                encoded = encode_bases(sequence)
                pending.append(encoded)
                pending_count += int(encoded.size)
                base_offsets[record_count] = base_offsets[record_count - 1] + int(encoded.size)
                if pending_count >= flush_bases:
                    flush_pending()

            if record_count != read_count:
                raise ValueError(
                    f"{fastq_path}: contains {record_count} reads, expected {read_count}"
                )
            flush_pending()
            if written_count != int(base_offsets[-1]):
                raise RuntimeError("internal base sidecar length mismatch")

            output.create_dataset("/base_read_offsets", data=base_offsets, dtype=np.int64)
            output.create_dataset("/body_lengths", data=body_lengths, dtype=np.int64)

        temporary_path.replace(output_path)
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise

    inspect_base_sidecar(h5_path, output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create full-read base HDF5 sidecars and validate exact FASTQ/H5 "
            "read/quality-prefix alignment."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        default=[],
        help="explicit quality-model H5 files/directories; default: h5",
    )
    parser.add_argument(
        "--datasets",
        default="",
        help="registered datasets/groups such as illumina, bgi_mgi, or mixed",
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--fastq-dir", type=Path, default=Path("fq"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.datasets:
        if args.inputs:
            raise SystemExit("--datasets cannot be combined with positional inputs")
        try:
            registered_files = resolve_dataset_files(
                args.datasets,
                args.data_root,
                require_h5=True,
                require_fastq=True,
            )
        except (ValueError, FileNotFoundError) as exc:
            raise SystemExit(str(exc)) from exc
        pairs = [(item.h5_path, item.fastq_path) for item in registered_files]
        output_dir = args.output_dir or (args.data_root / "base_sidecars")
    else:
        files = discover_h5_files(args.inputs or [Path("h5")])
        pairs = [
            (h5_path, fastq_path_for_h5(h5_path, args.fastq_dir))
            for h5_path in files
        ]
        output_dir = args.output_dir or Path("base_sidecars")

    for h5_path, fastq_path in pairs:
        output_path = base_sidecar_path_for_h5(h5_path, output_dir)
        create_base_sidecar(
            h5_path=h5_path,
            fastq_path=fastq_path,
            output_path=output_path,
            force=args.force,
        )
        info = inspect_base_sidecar(h5_path, output_path)
        print(
            f"wrote {output_path} reads={info.read_count} bases={info.base_count} "
            f"raw_len={info.min_read_len}..{info.max_read_len}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
