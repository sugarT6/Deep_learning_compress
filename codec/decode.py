#!/usr/bin/env python3
"""Decode a version-1 neural quality container back to byte-exact FASTQ."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import tempfile
import time
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Sequence, Tuple

import numpy as np
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
    ContainerIntegrityError,
    SideRecord,
    SideStreamReader,
    read_container,
    sha256_file,
)
from .encode import (
    CodecDeterminismError,
    _add_timing,
    _synchronize_device,
    choose_device,
)
from .fastq_stream import (
    BASE_PAD_ID,
    DEFAULT_BATCH_READS,
    MAX_BATCH_READS,
    PHRED_OFFSET,
    QUALITY_ALPHABET_SIZE,
    QUALITY_PAD_ID,
    encode_base_ids,
)
from .model import DirectQualityModelConfig, DirectQualityTransformer, feature_schema
from .probability_quantization import (
    QUANTIZATION_VERSION,
    logits_to_cdf,
    validate_total,
)
from .range_decoder import RangeDecoder


SUPPORTED_OUTPUT_SUFFIXES = (".fastq", ".fq", ".fastq.gz", ".fq.gz")


class ModelMismatchError(ContainerError):
    """Raised before decoding when the supplied checkpoint is not the encoded one."""


@dataclass(frozen=True)
class DecodeStatistics:
    container_bytes: int
    output_bytes: int
    output_uncompressed_bytes: int
    read_count: int
    quality_symbols: int
    decode_seconds: float
    reconstructed_sha256: str
    timing_seconds: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _validate_output_path(output_path: Path) -> None:
    if output_path.exists():
        raise FileExistsError(f"{output_path}: output already exists")
    if not any(output_path.name.endswith(suffix) for suffix in SUPPORTED_OUTPUT_SUFFIXES):
        raise ValueError(
            f"{output_path}: output must end in .fastq, .fq, .fastq.gz, or .fq.gz"
        )


def _validate_codec_metadata(metadata: Dict[str, Any], *, batch_reads: int) -> int:
    if metadata["format"] != CONTAINER_FORMAT:
        raise ContainerError("unsupported container format")
    if int(metadata["format_version"]) != CONTAINER_VERSION:
        raise ContainerError("unsupported container version")
    stored_batch_reads = int(metadata["batch_reads"])
    if batch_reads != stored_batch_reads:
        raise ContainerError(
            f"decoder batch_reads={batch_reads} does not match container value "
            f"{stored_batch_reads}"
        )
    gzip_streams = metadata["gzip_side_streams"]
    if int(gzip_streams["record_schema_version"]) != 1:
        raise ContainerError("unsupported gzip side-stream record schema")
    quantization = metadata["probability_quantization"]
    if int(quantization["version"]) != QUANTIZATION_VERSION:
        raise ContainerError("unsupported probability quantization version")
    if int(quantization["quality_alphabet_size"]) != QUALITY_ALPHABET_SIZE:
        raise ContainerError("container quality alphabet is not Q0-Q41")
    if int(quantization["phred_offset"]) != PHRED_OFFSET:
        raise ContainerError("container does not use Phred+33")
    if quantization.get("float_contract") != "finite_cpu_float64":
        raise ContainerError("unsupported probability float contract")
    total = validate_total(quantization["total"])
    range_coder = metadata["range_coder"]
    if int(range_coder["version"]) != RANGE_CODER_VERSION:
        raise ContainerError("unsupported range coder version")
    if int(range_coder["stream_count"]) != 1:
        raise ContainerError("version 1 requires one quality range stream")
    if metadata["inference_runtime"]["dtype"] != "float32":
        raise ContainerError("unsupported model inference dtype")
    return total


def _validate_checkpoint(
    checkpoint_path: Path,
    metadata: Dict[str, Any],
    device: torch.device,
) -> DirectQualityTransformer:
    checkpoint_metadata = metadata["checkpoint"]
    observed_hash = sha256_file(checkpoint_path)
    if observed_hash != checkpoint_metadata["sha256"]:
        raise ModelMismatchError(
            "checkpoint SHA-256 does not match the model required by the container"
        )
    loaded = load_training_checkpoint(checkpoint_path, device=device)
    if int(checkpoint_metadata["checkpoint_schema_version"]) != CHECKPOINT_SCHEMA_VERSION:
        raise ModelMismatchError("checkpoint schema version does not match container")
    stored_config = DirectQualityModelConfig.from_dict(
        checkpoint_metadata["model_config"]
    ).to_dict()
    if loaded.model.config.to_dict() != stored_config:
        raise ModelMismatchError("checkpoint model configuration does not match container")
    if feature_schema() != checkpoint_metadata["feature_schema"]:
        raise ModelMismatchError("checkpoint feature schema does not match container")
    loaded.model.eval()
    return loaded.model


def _batch_tensors(
    base_records: Sequence[SideRecord], device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    base_rows = [encode_base_ids(record.field) for record in base_records]
    lengths_array = np.asarray([row.size for row in base_rows], dtype=np.int64)
    maximum_length = int(lengths_array.max()) if len(base_rows) else 0
    bases_array = np.full(
        (len(base_rows), maximum_length), BASE_PAD_ID, dtype=np.uint8
    )
    for row_index, row in enumerate(base_rows):
        bases_array[row_index, : row.size] = row
    active_array = (
        np.arange(maximum_length, dtype=np.int64)[None, :]
        < lengths_array[:, None]
    )
    return (
        torch.as_tensor(bases_array, dtype=torch.long, device=device),
        torch.as_tensor(lengths_array, dtype=torch.long, device=device),
        torch.as_tensor(active_array, dtype=torch.bool, device=device),
    )


def _decode_quality_batch(
    model: DirectQualityTransformer,
    range_decoder: RangeDecoder,
    base_records: Sequence[SideRecord],
    *,
    device: torch.device,
    quantization_total: int,
    first_read_index: int,
    timings: Dict[str, float],
) -> Tuple[np.ndarray, int]:
    stage_started = time.perf_counter()
    bases, lengths, active_mask = _batch_tensors(base_records, device)
    _synchronize_device(device)
    _add_timing(timings, "tensor_transfer", stage_started)
    read_count, maximum_length = bases.shape
    decoded = torch.full(
        (read_count, maximum_length),
        QUALITY_PAD_ID,
        dtype=torch.long,
        device=device,
    )
    step_cdfs: List[Tuple[int, int, Tuple[int, ...]]] = []
    decoded_symbols = 0

    with torch.inference_mode():
        for cycle in range(maximum_length):
            _synchronize_device(device)
            stage_started = time.perf_counter()
            step_logits = model.forward_step(
                bases,
                decoded[:, :cycle],
                lengths,
                active_mask,
            )
            _synchronize_device(device)
            _add_timing(timings, "model_forward_step", stage_started)
            stage_started = time.perf_counter()
            cycle_cdfs = []
            for row in range(read_count):
                if not bool(active_mask[row, cycle].item()):
                    continue
                cdf = logits_to_cdf(
                    step_logits[row].detach().cpu().numpy(),
                    total=quantization_total,
                )
                cycle_cdfs.append((row, cdf))
            _add_timing(timings, "cdf_quantization_and_transfer", stage_started)

            stage_started = time.perf_counter()
            for row, cdf in cycle_cdfs:
                symbol = range_decoder.decode(cdf)
                decoded[row, cycle] = symbol
                step_cdfs.append((row, cycle, cdf))
                decoded_symbols += 1
            _synchronize_device(device)
            _add_timing(timings, "range_decode_and_symbol_update", stage_started)

        _synchronize_device(device)
        stage_started = time.perf_counter()
        full_logits = model.forward_full(bases, decoded, lengths, active_mask)
        _synchronize_device(device)
        _add_timing(timings, "model_forward_full_verification", stage_started)
        stage_started = time.perf_counter()
        for row, cycle, step_cdf in step_cdfs:
            full_cdf = logits_to_cdf(
                full_logits[row, cycle].detach().cpu().numpy(),
                total=quantization_total,
            )
            if full_cdf != step_cdf:
                raise CodecDeterminismError(
                    "decoder forward_step/forward_full integer CDF mismatch at "
                    f"read {first_read_index + row}, cycle {cycle}"
                )
        _add_timing(timings, "cdf_verification_and_transfer", stage_started)
    stage_started = time.perf_counter()
    decoded_array = decoded.cpu().numpy()
    _add_timing(timings, "decoded_tensor_to_cpu", stage_started)
    return decoded_array, decoded_symbols


def _validate_side_records(
    header: SideRecord,
    base: SideRecord,
    plus: SideRecord,
    *,
    read_index: int,
    read_count: int,
) -> None:
    if not header.field.startswith(b"@"):
        raise ContainerIntegrityError(f"decoded read {read_index} header lacks @")
    if not plus.field.startswith(b"+"):
        raise ContainerIntegrityError(f"decoded read {read_index} plus line lacks +")
    if not header.line_ending or not base.line_ending or not plus.line_ending:
        raise ContainerIntegrityError(
            f"decoded read {read_index} has a missing non-quality line ending"
        )
    if plus.quality_line_ending is None:
        raise ContainerIntegrityError("plus side stream lacks quality line ending")
    if not plus.quality_line_ending and read_index != read_count - 1:
        raise ContainerIntegrityError(
            "only the final FASTQ quality line may omit its line ending"
        )


def _record_bytes(
    header: SideRecord,
    base: SideRecord,
    plus: SideRecord,
    quality_ids: np.ndarray,
) -> bytes:
    quality = bytes((quality_ids.astype(np.uint16) + PHRED_OFFSET).tolist())
    return b"".join(
        (
            header.field,
            header.line_ending,
            base.field,
            base.line_ending,
            plus.field,
            plus.line_ending,
            quality,
            plus.quality_line_ending or b"",
        )
    )


def _decode_to_handle(
    container_info: Any,
    output: BinaryIO,
    model: DirectQualityTransformer,
    range_decoder: RangeDecoder,
    *,
    device: torch.device,
    batch_reads: int,
    quantization_total: int,
    progress_bar: Any,
    timings: Dict[str, float],
) -> Tuple[int, int, str]:
    metadata = container_info.metadata
    read_count = int(metadata["read_count"])
    reconstructed_digest = hashlib.sha256()
    reconstructed_size = 0
    decoded_symbols = 0

    with ExitStack() as stack:
        header_compressed = stack.enter_context(
            container_info.open_section("header_gzip")
        )
        base_compressed = stack.enter_context(container_info.open_section("base_gzip"))
        plus_compressed = stack.enter_context(container_info.open_section("plus_gzip"))
        header_reader = SideStreamReader(
            header_compressed,
            "header",
            expected_uncompressed_length=container_info.section(
                "header_gzip"
            ).uncompressed_length,
        )
        base_reader = SideStreamReader(
            base_compressed,
            "base",
            expected_uncompressed_length=container_info.section(
                "base_gzip"
            ).uncompressed_length,
        )
        plus_reader = SideStreamReader(
            plus_compressed,
            "plus",
            expected_uncompressed_length=container_info.section(
                "plus_gzip"
            ).uncompressed_length,
        )
        stack.callback(header_reader.close)
        stack.callback(base_reader.close)
        stack.callback(plus_reader.close)

        for batch_start in range(0, read_count, batch_reads):
            current_batch_reads = min(batch_reads, read_count - batch_start)
            headers = []
            bases = []
            pluses = []
            stage_started = time.perf_counter()
            for batch_row in range(current_batch_reads):
                read_index = batch_start + batch_row
                header = header_reader.read_record()
                base = base_reader.read_record()
                plus = plus_reader.read_record()
                _validate_side_records(
                    header,
                    base,
                    plus,
                    read_index=read_index,
                    read_count=read_count,
                )
                headers.append(header)
                bases.append(base)
                pluses.append(plus)
            _add_timing(timings, "side_stream_read", stage_started)

            decoded, batch_symbols = _decode_quality_batch(
                model,
                range_decoder,
                bases,
                device=device,
                quantization_total=quantization_total,
                first_read_index=batch_start,
                timings=timings,
            )
            decoded_symbols += batch_symbols
            stage_started = time.perf_counter()
            for row, (header, base, plus) in enumerate(
                zip(headers, bases, pluses)
            ):
                length = len(base.field)
                record = _record_bytes(header, base, plus, decoded[row, :length])
                output.write(record)
                reconstructed_digest.update(record)
                reconstructed_size += len(record)
            _add_timing(timings, "fastq_output_write", stage_started)
            if progress_bar is not None:
                progress_bar.update(batch_symbols)

        stage_started = time.perf_counter()
        header_reader.finish()
        base_reader.finish()
        plus_reader.finish()
        _add_timing(timings, "side_stream_read", stage_started)

    stage_started = time.perf_counter()
    range_decoder.finish()
    _add_timing(timings, "range_decode_and_symbol_update", stage_started)
    return reconstructed_size, decoded_symbols, reconstructed_digest.hexdigest()


def decode_fastq(
    container_path: Path,
    output_path: Path,
    checkpoint_path: Path,
    *,
    device: torch.device,
    batch_reads: int = DEFAULT_BATCH_READS,
    progress: bool = True,
) -> DecodeStatistics:
    """Validate and atomically reconstruct the original uncompressed FASTQ bytes."""

    started = time.perf_counter()
    container_path = Path(container_path)
    output_path = Path(output_path)
    checkpoint_path = Path(checkpoint_path)
    if not container_path.is_file():
        raise FileNotFoundError(container_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    _validate_output_path(output_path)
    if batch_reads <= 0 or batch_reads > MAX_BATCH_READS:
        raise ValueError(f"batch_reads must be in [1, {MAX_BATCH_READS}]")

    timings = {
        "container_validation": 0.0,
        "checkpoint_hash_and_load": 0.0,
        "quality_stream_read_and_init": 0.0,
        "side_stream_read": 0.0,
        "tensor_transfer": 0.0,
        "model_forward_step": 0.0,
        "cdf_quantization_and_transfer": 0.0,
        "range_decode_and_symbol_update": 0.0,
        "model_forward_full_verification": 0.0,
        "cdf_verification_and_transfer": 0.0,
        "decoded_tensor_to_cpu": 0.0,
        "fastq_output_write": 0.0,
        "output_finalize": 0.0,
    }
    stage_started = time.perf_counter()
    container_info = read_container(container_path, verify_checksums=True)
    quantization_total = _validate_codec_metadata(
        container_info.metadata, batch_reads=batch_reads
    )
    _add_timing(timings, "container_validation", stage_started)
    stage_started = time.perf_counter()
    model = _validate_checkpoint(checkpoint_path, container_info.metadata, device)
    _synchronize_device(device)
    _add_timing(timings, "checkpoint_hash_and_load", stage_started)
    stage_started = time.perf_counter()
    quality_range_bytes = container_info.read_section("quality_range")
    range_decoder = RangeDecoder(quality_range_bytes)
    expected_symbols = int(container_info.metadata["quality_symbol_count"])
    if range_decoder.metadata.symbol_count != expected_symbols:
        raise ContainerIntegrityError(
            "quality range-stream symbol count does not match container metadata"
        )
    if (
        range_decoder.metadata.payload_bit_count
        != int(container_info.metadata["range_coder"]["payload_bit_count"])
    ):
        raise ContainerIntegrityError(
            "quality range payload length does not match container metadata"
        )
    _add_timing(timings, "quality_stream_read_and_init", stage_started)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=str(output_path.parent)
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    progress_bar = None
    if progress and tqdm is not None:
        progress_bar = tqdm(
            total=expected_symbols,
            desc="decode quality",
            unit="Q",
            leave=True,
        )
    try:
        with temporary_path.open("wb") as raw_output:
            if output_path.name.endswith(".gz"):
                with gzip.GzipFile(
                    filename="",
                    mode="wb",
                    compresslevel=6,
                    fileobj=raw_output,
                    mtime=0,
                ) as decoded_output:
                    reconstructed_size, decoded_symbols, reconstructed_hash = (
                        _decode_to_handle(
                            container_info,
                            decoded_output,
                            model,
                            range_decoder,
                            device=device,
                            batch_reads=batch_reads,
                            quantization_total=quantization_total,
                            progress_bar=progress_bar,
                            timings=timings,
                        )
                    )
            else:
                reconstructed_size, decoded_symbols, reconstructed_hash = (
                    _decode_to_handle(
                        container_info,
                        raw_output,
                        model,
                        range_decoder,
                        device=device,
                        batch_reads=batch_reads,
                        quantization_total=quantization_total,
                        progress_bar=progress_bar,
                        timings=timings,
                    )
                )
            stage_started = time.perf_counter()
            raw_output.flush()
            os.fsync(raw_output.fileno())
            _add_timing(timings, "output_finalize", stage_started)

        if decoded_symbols != expected_symbols:
            raise ContainerIntegrityError(
                "decoded quality symbol count does not match container metadata"
            )
        if reconstructed_size != int(
            container_info.metadata["source_uncompressed_size"]
        ):
            raise ContainerIntegrityError("reconstructed FASTQ size mismatch")
        if reconstructed_hash != container_info.metadata["source_uncompressed_sha256"]:
            raise ContainerIntegrityError("reconstructed FASTQ SHA-256 mismatch")
        stage_started = time.perf_counter()
        os.replace(str(temporary_path), str(output_path))
        _add_timing(timings, "output_finalize", stage_started)
    except Exception:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise
    finally:
        if progress_bar is not None:
            progress_bar.close()

    decode_seconds = time.perf_counter() - started
    accounted_seconds = sum(timings.values())
    timings["unattributed"] = max(0.0, decode_seconds - accounted_seconds)
    timings["total"] = decode_seconds
    return DecodeStatistics(
        container_bytes=container_info.file_size,
        output_bytes=output_path.stat().st_size,
        output_uncompressed_bytes=reconstructed_size,
        read_count=int(container_info.metadata["read_count"]),
        quality_symbols=decoded_symbols,
        decode_seconds=decode_seconds,
        reconstructed_sha256=reconstructed_hash,
        timing_seconds=timings,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Decode a no-SeqArc neural container to FASTQ or gzip FASTQ"
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-reads", type=int, default=DEFAULT_BATCH_READS)
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        statistics = decode_fastq(
            args.input,
            args.output,
            args.checkpoint,
            device=choose_device(args.device),
            batch_reads=args.batch_reads,
            progress=not args.no_progress,
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(statistics.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["DecodeStatistics", "ModelMismatchError", "decode_fastq"]
