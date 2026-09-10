#!/usr/bin/env python3
"""Compress a gzip FASTQ into the version-1 direct-quality container."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is optional
    tqdm = None

from ._range_common import RANGE_CODER_VERSION
from .checkpoint import CHECKPOINT_SCHEMA_VERSION, load_training_checkpoint
from .container import (
    CONTAINER_FORMAT,
    CONTAINER_VERSION,
    ContainerError,
    SectionSource,
    SideStreamWriter,
    sha256_file,
    write_container,
)
from .fastq_stream import (
    DEFAULT_BATCH_READS,
    PHRED_OFFSET,
    QUALITY_ALPHABET_SIZE,
    FastqBatch,
)
from .fastq_stream import iter_fastq_batches
from .model import DirectQualityTransformer, fastq_batch_to_tensors, feature_schema
from .probability_quantization import (
    QUANTIZATION_VERSION,
    TOTAL,
    logits_to_cdf,
    quantized_symbol_bits,
    validate_total,
)
from .range_encoder import RangeEncoder


SUPPORTED_INPUT_SUFFIXES = (".fastq.gz", ".fq.gz")
GZIP_COMPRESSLEVEL = 6


class CodecDeterminismError(ContainerError):
    """Raised when full and step inference quantize to different CDFs."""


@dataclass(frozen=True)
class EncodeStatistics:
    input_bytes: int
    output_bytes: int
    output_to_input_ratio: float
    read_count: int
    batch_count: int
    quality_symbols: int
    quantized_theoretical_bits: float
    range_payload_bits: int
    range_payload_bits_per_quality: Optional[float]
    range_stream_bytes: int
    range_stream_bits_per_quality: Optional[float]
    header_gzip_bytes: int
    base_gzip_bytes: int
    plus_gzip_bytes: int
    container_header_bytes: int
    encode_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return device


def _validate_paths(input_path: Path, output_path: Path, checkpoint_path: Path) -> None:
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if not any(input_path.name.endswith(suffix) for suffix in SUPPORTED_INPUT_SUFFIXES):
        raise ValueError(
            f"{input_path}: input must end in .fastq.gz or .fq.gz"
        )
    if output_path.exists():
        raise FileExistsError(f"{output_path}: output already exists")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)


def _quantize_verified_batch(
    model: DirectQualityTransformer,
    batch: FastqBatch,
    device: torch.device,
    *,
    total: int,
) -> Tuple[List[Tuple[int, Tuple[int, ...]]], float]:
    """Return cycle-major symbols/CDFs after full-vs-step integer verification."""

    tensors = fastq_batch_to_tensors(batch, device)
    items: List[Tuple[int, Tuple[int, ...]]] = []
    theoretical_bits = 0.0
    with torch.inference_mode():
        full_logits = model.forward_full(**tensors)
        for cycle in range(batch.max_read_length):
            step_logits = model.forward_step(
                tensors["bases"],
                tensors["qualities"][:, :cycle],
                tensors["lengths"],
                tensors["active_mask"],
            )
            active_rows = [
                row for row in range(batch.read_count) if batch.active_mask[row, cycle]
            ]
            for row in active_rows:
                full_cdf = logits_to_cdf(
                    full_logits[row, cycle].detach().cpu().numpy(), total=total
                )
                step_cdf = logits_to_cdf(
                    step_logits[row].detach().cpu().numpy(), total=total
                )
                if full_cdf != step_cdf:
                    read_index = int(batch.read_indices[row])
                    raise CodecDeterminismError(
                        "forward_full/forward_step integer CDF mismatch at "
                        f"read {read_index}, cycle {cycle}"
                    )
                symbol = int(batch.qualities[row, cycle])
                items.append((symbol, full_cdf))
                theoretical_bits += quantized_symbol_bits(symbol, full_cdf)
    return items, theoretical_bits


def encode_fastq(
    input_path: Path,
    output_path: Path,
    checkpoint_path: Path,
    *,
    device: torch.device,
    batch_reads: int = DEFAULT_BATCH_READS,
    quantization_total: int = TOTAL,
    progress: bool = True,
) -> EncodeStatistics:
    """Stream a gzip FASTQ through the neural model into one atomic container."""

    started = time.perf_counter()
    input_path = Path(input_path)
    output_path = Path(output_path)
    checkpoint_path = Path(checkpoint_path)
    _validate_paths(input_path, output_path, checkpoint_path)
    if batch_reads <= 0 or batch_reads > DEFAULT_BATCH_READS:
        raise ValueError(f"batch_reads must be in [1, {DEFAULT_BATCH_READS}]")
    quantization_total = validate_total(quantization_total)

    checkpoint_sha256 = sha256_file(checkpoint_path)
    loaded = load_training_checkpoint(checkpoint_path, device=device)
    model = loaded.model
    model.eval()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    source_digest = hashlib.sha256()
    source_uncompressed_size = 0
    read_count = 0
    batch_count = 0
    quality_symbols = 0
    quantized_theoretical_bits = 0.0
    range_encoder = RangeEncoder()
    progress_bar = None
    if progress and tqdm is not None:
        progress_bar = tqdm(desc="encode FASTQ", unit="read", leave=True)

    try:
        with tempfile.TemporaryDirectory(
            prefix=f".{output_path.name}.sections.", dir=str(output_path.parent)
        ) as temporary_directory:
            temporary_root = Path(temporary_directory)
            header_path = temporary_root / "header.gz"
            base_path = temporary_root / "base.gz"
            plus_path = temporary_root / "plus.gz"
            quality_path = temporary_root / "quality.range"

            with SideStreamWriter(
                header_path, "header", compresslevel=GZIP_COMPRESSLEVEL
            ) as header_writer, SideStreamWriter(
                base_path, "base", compresslevel=GZIP_COMPRESSLEVEL
            ) as base_writer, SideStreamWriter(
                plus_path, "plus", compresslevel=GZIP_COMPRESSLEVEL
            ) as plus_writer:
                for batch in iter_fastq_batches(input_path, batch_reads=batch_reads):
                    if len(batch.raw_records) != batch.read_count:
                        raise ContainerError("direct FASTQ batch is missing raw records")
                    for record in batch.raw_records:
                        raw_record = record.to_bytes()
                        source_digest.update(raw_record)
                        source_uncompressed_size += len(raw_record)
                        header_writer.write_record(
                            record.header, record.line_endings[0]
                        )
                        base_writer.write_record(
                            record.sequence, record.line_endings[1]
                        )
                        plus_writer.write_record(
                            record.plus,
                            record.line_endings[2],
                            quality_line_ending=record.line_endings[3],
                        )

                    items, batch_theoretical_bits = _quantize_verified_batch(
                        model,
                        batch,
                        device,
                        total=quantization_total,
                    )
                    for symbol, cdf in items:
                        range_encoder.encode(symbol, cdf)
                    read_count += batch.read_count
                    batch_count += 1
                    quality_symbols += len(items)
                    quantized_theoretical_bits += batch_theoretical_bits
                    if progress_bar is not None:
                        progress_bar.update(batch.read_count)

            range_stream = range_encoder.finish()
            quality_path.write_bytes(range_stream)
            if range_encoder.symbol_count != quality_symbols:
                raise ContainerError("range symbol count does not match FASTQ qualities")

            metadata = {
                "format": CONTAINER_FORMAT,
                "format_version": CONTAINER_VERSION,
                "source_basename": input_path.name,
                "source_compressed_size": input_path.stat().st_size,
                "source_uncompressed_size": source_uncompressed_size,
                "source_uncompressed_sha256": source_digest.hexdigest(),
                "read_count": read_count,
                "batch_reads": batch_reads,
                "batch_count": batch_count,
                "last_batch_read_count": (
                    0 if read_count == 0 else ((read_count - 1) % batch_reads) + 1
                ),
                "quality_symbol_count": quality_symbols,
                "gzip_side_streams": {
                    "compresslevel": GZIP_COMPRESSLEVEL,
                    "record_schema_version": 1,
                    "quality_line_ending_location": "plus_gzip",
                },
                "probability_quantization": {
                    "version": QUANTIZATION_VERSION,
                    "total": quantization_total,
                    "quality_alphabet_size": QUALITY_ALPHABET_SIZE,
                    "phred_offset": PHRED_OFFSET,
                    "float_contract": "finite_cpu_float64",
                },
                "range_coder": {
                    "version": RANGE_CODER_VERSION,
                    "stream_count": 1,
                    "payload_bit_count": range_encoder.metadata.payload_bit_count,
                },
                "checkpoint": {
                    "sha256": checkpoint_sha256,
                    "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
                    "model_config": model.config.to_dict(),
                    "feature_schema": feature_schema(),
                },
                "inference_runtime": {
                    "torch_version": torch.__version__,
                    "device_type": device.type,
                    "dtype": "float32",
                },
            }
            container_info = write_container(
                output_path,
                metadata,
                (
                    SectionSource(
                        "header_gzip", header_path, header_writer.uncompressed_length
                    ),
                    SectionSource(
                        "base_gzip", base_path, base_writer.uncompressed_length
                    ),
                    SectionSource(
                        "plus_gzip", plus_path, plus_writer.uncompressed_length
                    ),
                    SectionSource("quality_range", quality_path, len(range_stream)),
                ),
            )
    finally:
        if progress_bar is not None:
            progress_bar.close()

    input_bytes = input_path.stat().st_size
    output_bytes = container_info.file_size
    range_stream_bytes = container_info.section("quality_range").length
    return EncodeStatistics(
        input_bytes=input_bytes,
        output_bytes=output_bytes,
        output_to_input_ratio=(output_bytes / input_bytes if input_bytes else math.inf),
        read_count=read_count,
        batch_count=batch_count,
        quality_symbols=quality_symbols,
        quantized_theoretical_bits=quantized_theoretical_bits,
        range_payload_bits=range_encoder.metadata.payload_bit_count,
        range_payload_bits_per_quality=(
            range_encoder.metadata.payload_bit_count / quality_symbols
            if quality_symbols
            else None
        ),
        range_stream_bytes=range_stream_bytes,
        range_stream_bits_per_quality=(
            range_stream_bytes * 8 / quality_symbols
            if quality_symbols
            else None
        ),
        header_gzip_bytes=container_info.section("header_gzip").length,
        base_gzip_bytes=container_info.section("base_gzip").length,
        plus_gzip_bytes=container_info.section("plus_gzip").length,
        container_header_bytes=container_info.header_bytes,
        encode_seconds=time.perf_counter() - started,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compress .fastq.gz/.fq.gz with the no-SeqArc neural codec"
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-reads", type=int, default=DEFAULT_BATCH_READS)
    parser.add_argument("--quantization-total", type=int, default=TOTAL)
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        statistics = encode_fastq(
            args.input,
            args.output,
            args.checkpoint,
            device=choose_device(args.device),
            batch_reads=args.batch_reads,
            quantization_total=args.quantization_total,
            progress=not args.no_progress,
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(statistics.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CodecDeterminismError",
    "EncodeStatistics",
    "choose_device",
    "encode_fastq",
]
