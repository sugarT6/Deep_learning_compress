#!/usr/bin/env python3
"""Pure-neural FASTQ encoding, with optional transmitted output-head adaptation.

Private historical profile branches are retained for decoder regression fixtures.
Neither the public encoder nor its CLI can select those branches.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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
    ADAPTED_CONTAINER_VERSION,
    LEGACY_CONTAINER_VERSION,
    ContainerError,
    SectionSource,
    SideStreamWriter,
    sha256_file,
    write_container,
)
from .fastq_stream import (
    DEFAULT_BATCH_READS,
    MAX_BATCH_READS,
    PHRED_OFFSET,
    QUALITY_ALPHABET_SIZE,
    SUPPORTED_FASTQ_SUFFIXES,
    FastqBatch,
)
from .fastq_stream import iter_fastq_batches
from .model import DirectQualityTransformer, fastq_batch_to_tensors, feature_schema
from .online_prior import (
    BOS_QUALITY_ID,
    DEFAULT_ONLINE_PRIOR_CONFIG,
    OnlinePriorConfig,
    OnlinePriorState,
)
from .encode_fastpath import fuse_batch_logits, selected_quantized_bits
from .adaptive_prior import AdaptivePriorState
from .mixture_prior import (
    MixturePriorConfig, MixturePriorState, make_prior_state,
    fuse_profile_positions,
)
from .probability_quantization import (
    QUANTIZATION_VERSION,
    TOTAL,
    logits_symbols_bits,
    logits_to_cdfs,
    validate_total,
)
from .range_encoder import RangeEncoder
from .head_adapter import HeadAdaptationConfig, adapt_output_head, apply_head_adapter


SUPPORTED_INPUT_SUFFIXES = SUPPORTED_FASTQ_SUFFIXES
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
    quantized_theoretical_bits_per_quality: Optional[float]
    neural_only_theoretical_bits: Optional[float]
    neural_only_theoretical_bits_per_quality: Optional[float]
    range_payload_bits: int
    range_payload_bits_per_quality: Optional[float]
    range_stream_bytes: int
    range_stream_bits_per_quality: Optional[float]
    header_gzip_bytes: int
    base_gzip_bytes: int
    plus_gzip_bytes: int
    container_header_bytes: int
    encode_seconds: float
    quality_model_prediction_seconds: float
    quality_entropy_coding_seconds: float
    encoding_stage_seconds: Dict[str, float]
    online_adaptation: Optional[Dict[str, Any]]
    head_adaptation: Optional[Dict[str, Any]]
    quality_and_adapter_bytes: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return device


def _synchronize_device(device: torch.device) -> None:
    """Finish queued CUDA work so wall-clock stage timings are meaningful."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _add_timing(timings: Optional[Dict[str, float]], name: str, started: float) -> None:
    if timings is not None:
        timings[name] = timings.get(name, 0.0) + (time.perf_counter() - started)


def _validate_paths(input_path: Path, output_path: Path, checkpoint_path: Path) -> None:
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if not any(input_path.name.endswith(suffix) for suffix in SUPPORTED_INPUT_SUFFIXES):
        raise ValueError(
            f"{input_path}: input must end in .fastq, .fq, .fastq.gz, or .fq.gz"
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
    online_prior: Optional[OnlinePriorState | MixturePriorState] = None,
    timings: Optional[Dict[str, float]] = None,
    verify_cdf: bool = True,
    report_neural_only_bits: bool = True,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Return cycle-major symbols/CDFs, optionally cross-checking step inference."""

    stage_started = time.perf_counter()
    tensors = fastq_batch_to_tensors(batch, device)
    _synchronize_device(device)
    _add_timing(timings, "tensor_transfer", stage_started)
    with torch.inference_mode():
        _synchronize_device(device)
        stage_started = time.perf_counter()
        full_logits = model.forward_full(**tensors)
        _synchronize_device(device)
        _add_timing(timings, "model_forward_full", stage_started)

        stage_started = time.perf_counter()
        full_logits_cpu = full_logits.detach().cpu().numpy()
        cycles, rows = np.nonzero(batch.active_mask.T)
        symbols = np.asarray(batch.qualities[rows, cycles], dtype=np.int64)
        active_logits = full_logits_cpu[rows, cycles]
        _add_timing(timings, "logits_transfer_and_context", stage_started)
        stage_started = time.perf_counter()
        neural_theoretical_bits = (
            logits_symbols_bits(active_logits, symbols) if report_neural_only_bits else 0.0
        )
        _add_timing(timings, "neural_only_diagnostic", stage_started)
        stage_started = time.perf_counter()
        if online_prior is None:
            fused_logits = active_logits
        elif isinstance(online_prior, AdaptivePriorState):
            fused_logits = online_prior.fuse_positions(active_logits, batch.qualities, rows, cycles, capture=True)
        elif isinstance(online_prior, MixturePriorState):
            fused_logits = online_prior.fuse_positions(active_logits, batch.qualities, rows, cycles)
        else:
            previous = np.full(symbols.shape, BOS_QUALITY_ID, dtype=np.int64)
            later = cycles > 0
            previous[later] = batch.qualities[rows[later], cycles[later] - 1]
            fused_logits = fuse_batch_logits(online_prior, active_logits, previous, cycles)
        _add_timing(timings, "prior_fusion", stage_started)
        stage_started = time.perf_counter()
        cdfs = logits_to_cdfs(fused_logits, total=total)
        _add_timing(timings, "cdf_quantization", stage_started)
        stage_started = time.perf_counter()
        theoretical_bits = selected_quantized_bits(symbols, cdfs, total)
        _add_timing(timings, "quantized_bits", stage_started)

        if verify_cdf:
            offset = 0
            for cycle in range(batch.max_read_length):
                _synchronize_device(device)
                stage_started = time.perf_counter()
                step_logits = model.forward_step(
                    tensors["bases"],
                    tensors["qualities"][:, :cycle],
                    tensors["lengths"],
                    tensors["active_mask"],
                )
                _synchronize_device(device)
                _add_timing(
                    timings, "model_forward_step_verification", stage_started
                )
                stage_started = time.perf_counter()
                active_rows = np.flatnonzero(batch.active_mask[:, cycle])
                step_scores = step_logits.detach().cpu().numpy()[active_rows]
                if online_prior is not None:
                    step_scores = fuse_profile_positions(online_prior,
                        step_scores, batch.qualities, active_rows,
                        np.full(active_rows.size, cycle, dtype=np.int64))
                step_cdfs = logits_to_cdfs(step_scores, total=total)
                stop = offset + active_rows.size
                full_cycle_cdfs = cdfs[offset:stop]
                if not np.array_equal(full_cycle_cdfs, step_cdfs):
                    mismatch = np.argwhere(full_cycle_cdfs != step_cdfs)[0]
                    row = int(active_rows[int(mismatch[0])])
                    read_index = int(batch.read_indices[row])
                    raise CodecDeterminismError(
                        "forward_full/forward_step integer CDF mismatch at "
                        f"read {read_index}, cycle {cycle}"
                    )
                offset = stop
                _add_timing(timings, "cdf_quantization_and_transfer", stage_started)
            if offset != symbols.size:
                raise ContainerError("CDF verification did not cover every quality")
    return symbols, cdfs, theoretical_bits, neural_theoretical_bits


def _encode_fastq_compat(
    input_path: Path,
    output_path: Path,
    checkpoint_path: Path,
    *,
    device: torch.device,
    batch_reads: int = DEFAULT_BATCH_READS,
    quantization_total: int = TOTAL,
    progress: bool = True,
    verify_cdf: bool = False,
    online_prior_config: Optional[OnlinePriorConfig | MixturePriorConfig] = DEFAULT_ONLINE_PRIOR_CONFIG,
    report_neural_only_bits: bool = False,
    head_adaptation_config: Optional[HeadAdaptationConfig] = None,
    head_adapter_path: Optional[Path] = None,
    save_head_adapter_path: Optional[Path] = None,
) -> EncodeStatistics:
    """Shared engine; expert profiles are for historical decoder fixtures only."""

    started = time.perf_counter()
    input_path = Path(input_path)
    output_path = Path(output_path)
    checkpoint_path = Path(checkpoint_path)
    _validate_paths(input_path, output_path, checkpoint_path)
    if head_adaptation_config is not None and not isinstance(head_adaptation_config, HeadAdaptationConfig):
        raise ValueError("invalid head_adaptation_config")
    if head_adaptation_config is not None and head_adapter_path is not None:
        raise ValueError("cannot train and load an adapter together")
    if save_head_adapter_path is not None:
        save_head_adapter_path = Path(save_head_adapter_path)
        if head_adaptation_config is None:
            raise ValueError("saving an adapter requires head adaptation")
        if save_head_adapter_path.exists() or save_head_adapter_path.resolve() in {
            input_path.resolve(), output_path.resolve(), checkpoint_path.resolve()
        }:
            raise FileExistsError("adapter export path must be new and distinct")
    if batch_reads <= 0 or batch_reads > MAX_BATCH_READS:
        raise ValueError(f"batch_reads must be in [1, {MAX_BATCH_READS}]")
    quantization_total = validate_total(quantization_total)
    if online_prior_config is not None and not isinstance(
        online_prior_config, (OnlinePriorConfig, MixturePriorConfig)
    ):
        raise ValueError("online_prior_config must be a supported prior config or None")

    timings = {
        "checkpoint_hash_and_load": 0.0,
        "fastq_parse": 0.0,
        "side_stream_record_write": 0.0,
        "tensor_transfer": 0.0,
        "model_forward_full": 0.0,
        "model_forward_step_verification": 0.0,
        "cdf_quantization_and_transfer": 0.0,
        "range_encode": 0.0,
        "range_finalize_and_stage": 0.0,
        "container_write": 0.0,
    }
    stage_started = time.perf_counter()
    checkpoint_sha256 = sha256_file(checkpoint_path)
    loaded = load_training_checkpoint(checkpoint_path, device=device)
    model = loaded.model
    model.eval()
    _synchronize_device(device)
    _add_timing(timings, "checkpoint_hash_and_load", stage_started)
    adapter = None
    adaptation_report = None
    if head_adaptation_config is not None or head_adapter_path is not None:
        stage_started = time.perf_counter()
        if head_adaptation_config is not None:
            print("Adapting output head on a bounded prefix (experts disabled)...", file=sys.stderr)
            adapter, adaptation_report = adapt_output_head(model, input_path, device=device,
                batch_reads=batch_reads, total=quantization_total, base_sha256=checkpoint_sha256,
                config=head_adaptation_config)
            if save_head_adapter_path is not None:
                artifact = {"format": "direct-quality-head-adaptation-artifact", "version": 1,
                            "base_checkpoint_sha256": checkpoint_sha256, "adapter": adapter,
                            "report": adaptation_report, "source": str(input_path.resolve())}
                save_head_adapter_path.parent.mkdir(parents=True, exist_ok=True)
                with save_head_adapter_path.open("x", encoding="utf-8") as handle:
                    json.dump(artifact, handle, sort_keys=True, allow_nan=False)
        else:
            artifact = json.loads(Path(head_adapter_path).read_text())
            if (not isinstance(artifact, dict) or artifact.get("format") != "direct-quality-head-adaptation-artifact"
                    or type(artifact.get("version")) is not int or artifact["version"] != 1
                    or artifact.get("base_checkpoint_sha256") != checkpoint_sha256
                    or "adapter" not in artifact or not isinstance(artifact.get("report"), dict)):
                raise ValueError("invalid adapter artifact or base checkpoint mismatch")
            adapter = artifact["adapter"]
            if adapter is not None:
                apply_head_adapter(model, adapter, checkpoint_sha256)
            adaptation_report = {"accepted": adapter is not None, "reason": "loaded_artifact",
                                 "artifact_source": artifact.get("source"), "original_report": artifact["report"]}
        _synchronize_device(device)
        _add_timing(timings, "head_adaptation", stage_started)
        adaptation_report["total_stage_seconds"] = timings["head_adaptation"]
        adaptation_report["parameter_bytes"] = adapter["parameter_bytes"] if adapter else 0
        # Actual added canonical JSON member bytes (base64 and framing included).
        adaptation_report["container_adapter_bytes"] = (len(json.dumps({"head_adapter": adapter},
            sort_keys=True, separators=(",", ":")).encode()) - 1 if adapter else 0)
        if adapter is not None and online_prior_config is None:
            adaptation_report["container_adapter_bytes"] += len('"probability_profile":null,')
        print(f"Head adaptation: {adaptation_report['reason']}; "
              f"accepted={adaptation_report['accepted']}; {timings['head_adaptation']:.3f}s", file=sys.stderr)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    source_digest = hashlib.sha256()
    source_uncompressed_size = 0
    read_count = 0
    batch_count = 0
    quality_symbols = 0
    quantized_theoretical_bits = 0.0
    neural_only_theoretical_bits = 0.0
    range_encoder = RangeEncoder()
    online_prior = make_prior_state(online_prior_config) if online_prior_config is not None else None
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
                batch_iterator = iter_fastq_batches(
                    input_path, batch_reads=batch_reads
                )
                while True:
                    stage_started = time.perf_counter()
                    try:
                        batch = next(batch_iterator)
                    except StopIteration:
                        _add_timing(timings, "fastq_parse", stage_started)
                        break
                    _add_timing(timings, "fastq_parse", stage_started)
                    if len(batch.raw_records) != batch.read_count:
                        raise ContainerError(
                            "direct FASTQ batch is missing raw records"
                        )
                    stage_started = time.perf_counter()
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
                    _add_timing(timings, "side_stream_record_write", stage_started)

                    (
                        symbols,
                        cdfs,
                        batch_theoretical_bits,
                        batch_neural_only_bits,
                    ) = _quantize_verified_batch(
                        model,
                        batch,
                        device,
                        total=quantization_total,
                        online_prior=online_prior,
                        timings=timings,
                        verify_cdf=verify_cdf,
                        report_neural_only_bits=report_neural_only_bits,
                    )
                    stage_started = time.perf_counter()
                    range_encoder.encode_prevalidated_batch(
                        symbols, cdfs, total=quantization_total
                    )
                    _add_timing(timings, "range_encode", stage_started)
                    if online_prior is not None:
                        stage_started = time.perf_counter()
                        if isinstance(online_prior, AdaptivePriorState):
                            online_prior.observe_symbols(symbols)
                        _add_timing(timings, "adaptive_weight_feedback", stage_started)
                        stage_started = time.perf_counter()
                        online_prior.update_batch(batch.qualities, batch.active_mask)
                        _add_timing(timings, "prior_update", stage_started)
                    read_count += batch.read_count
                    batch_count += 1
                    quality_symbols += int(symbols.size)
                    quantized_theoretical_bits += batch_theoretical_bits
                    neural_only_theoretical_bits += batch_neural_only_bits
                    if progress_bar is not None:
                        progress_bar.update(batch.read_count)

            stage_started = time.perf_counter()
            range_stream = range_encoder.finish()
            quality_path.write_bytes(range_stream)
            _add_timing(timings, "range_finalize_and_stage", stage_started)
            if range_encoder.symbol_count != quality_symbols:
                raise ContainerError(
                    "range symbol count does not match FASTQ qualities"
                )

            metadata = {
                "format": CONTAINER_FORMAT,
                "format_version": (
                    CONTAINER_VERSION
                    if online_prior_config is not None
                    else LEGACY_CONTAINER_VERSION
                ),
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
            if online_prior_config is not None:
                metadata["probability_profile"] = (
                    online_prior_config.to_profile_metadata()
                )
            if adapter is not None:
                metadata["format_version"] = ADAPTED_CONTAINER_VERSION
                metadata["head_adapter"] = adapter
                metadata.setdefault("probability_profile", None)
            stage_started = time.perf_counter()
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
            _add_timing(timings, "container_write", stage_started)
    finally:
        if progress_bar is not None:
            progress_bar.close()

    input_bytes = input_path.stat().st_size
    output_bytes = container_info.file_size
    range_stream_bytes = container_info.section("quality_range").length
    encode_seconds = time.perf_counter() - started
    quality_model_prediction_seconds = timings["model_forward_full"]
    quality_entropy_coding_seconds = (
        timings["cdf_quantization_and_transfer"]
        + timings["range_encode"]
        + timings["range_finalize_and_stage"]
        + sum(timings.get(name, 0.0) for name in (
            "logits_transfer_and_context", "neural_only_diagnostic", "prior_fusion",
            "cdf_quantization", "quantized_bits", "prior_update",
            "adaptive_weight_feedback",
        ))
    )
    return EncodeStatistics(
        input_bytes=input_bytes,
        output_bytes=output_bytes,
        output_to_input_ratio=(output_bytes / input_bytes if input_bytes else math.inf),
        read_count=read_count,
        batch_count=batch_count,
        quality_symbols=quality_symbols,
        quantized_theoretical_bits=quantized_theoretical_bits,
        quantized_theoretical_bits_per_quality=(
            quantized_theoretical_bits / quality_symbols
            if quality_symbols
            else None
        ),
        neural_only_theoretical_bits=(
            neural_only_theoretical_bits if report_neural_only_bits else None
        ),
        neural_only_theoretical_bits_per_quality=(
            neural_only_theoretical_bits / quality_symbols
            if quality_symbols and report_neural_only_bits
            else None
        ),
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
        encode_seconds=encode_seconds,
        quality_model_prediction_seconds=quality_model_prediction_seconds,
        quality_entropy_coding_seconds=quality_entropy_coding_seconds,
        encoding_stage_seconds=dict(timings),
        online_adaptation=(online_prior.diagnostics() if isinstance(online_prior, AdaptivePriorState) else None),
        head_adaptation=adaptation_report,
        quality_and_adapter_bytes=range_stream_bytes + (
            adaptation_report["container_adapter_bytes"] if adaptation_report else 0),
    )


def encode_fastq(
    input_path: Path, output_path: Path, checkpoint_path: Path, *, device: torch.device,
    batch_reads: int = DEFAULT_BATCH_READS, quantization_total: int = TOTAL,
    progress: bool = True, verify_cdf: bool = False,
    report_neural_only_bits: bool = False,
    head_adaptation_config: Optional[HeadAdaptationConfig] = None,
    head_adapter_path: Optional[Path] = None,
    save_head_adapter_path: Optional[Path] = None,
) -> EncodeStatistics:
    """Encode only neural probabilities; legacy experts cannot be selected here."""
    return _encode_fastq_compat(
        input_path, output_path, checkpoint_path, device=device, batch_reads=batch_reads,
        quantization_total=quantization_total, progress=progress, verify_cdf=verify_cdf,
        online_prior_config=None, report_neural_only_bits=report_neural_only_bits,
        head_adaptation_config=head_adaptation_config, head_adapter_path=head_adapter_path,
        save_head_adapter_path=save_head_adapter_path,
    )


class _NeuralEncoderParser(argparse.ArgumentParser):
    def parse_known_args(self, args=None, namespace=None):
        supplied = list(sys.argv[1:] if args is None else args)
        for token in supplied:
            option = token.split("=", 1)[0]
            if option in ("--adaptive-weights", "--probability-profile") or option.startswith("--prior-"):
                self.error(f"{option} has been retired: encoding is pure neural. "
                           "Old expert containers remain decodable; historical v3 batch commands "
                           "must not be reused for new neural runs.")
        return super().parse_known_args(supplied, namespace)


def build_parser() -> argparse.ArgumentParser:
    parser = _NeuralEncoderParser(
        description="Compress plain/gzip FASTQ with pure neural probabilities and optional head fitting",
        allow_abbrev=False,
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-reads", type=int, default=DEFAULT_BATCH_READS)
    parser.add_argument("--quantization-total", type=int, default=TOTAL)
    parser.add_argument(
        "--verify-cdf",
        action="store_true",
        help=(
            "slow debug check: compare forward_full and forward_step integer "
            "CDFs for every active quality"
        ),
    )
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--neural-only", action="store_true", help="compatibility alias; encoding is always neural-only")
    adaptation_options = parser.add_mutually_exclusive_group()
    adaptation_options.add_argument("--finetune-head", action="store_true",
        help="adapt frozen-feature output head before pure-neural encoding")
    adaptation_options.add_argument("--head-adapter", type=Path, help="load exported adapter, without retraining")
    parser.add_argument("--save-head-adapter", type=Path, help="export accepted head or explicit fallback for paired tests")
    parser.add_argument("--head-max-reads", type=int, default=4096)
    parser.add_argument("--head-max-symbols", type=int, default=600000)
    parser.add_argument("--head-max-read-length", type=int, default=2048)
    parser.add_argument("--head-steps", type=int, default=50)
    parser.add_argument("--head-symbols-per-step", type=int, default=8192)
    parser.add_argument("--head-learning-rate", type=float, default=0.001)
    parser.add_argument("--head-anchor-strength", type=float, default=0.01)
    parser.add_argument("--head-max-seconds", type=float, default=20.0)
    parser.add_argument("--head-min-gain", type=float, default=0.0)
    parser.add_argument("--head-seed", type=int, default=20260921)
    parser.add_argument("--head-type", choices=("linear", "residual"), default="linear",
        help="fitted output head: linear baseline or linear + small GELU residual")
    parser.add_argument("--head-residual-dim", type=int, default=32,
        help="residual hidden width (1..128); requires --head-type residual")
    parser.add_argument(
        "--report-neural-only-bits", action="store_true",
        help="extra diagnostic softmax pass; disabled by default for encoding speed",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        head_config = None
        if args.head_residual_dim != 32 and args.head_type != "residual":
            raise ValueError("--head-residual-dim requires --head-type residual")
        if args.finetune_head:
            head_config = HeadAdaptationConfig(max_reads=args.head_max_reads, max_symbols=args.head_max_symbols,
                max_read_length=args.head_max_read_length, steps=args.head_steps,
                symbols_per_step=args.head_symbols_per_step, learning_rate=args.head_learning_rate,
                anchor_strength=args.head_anchor_strength, max_seconds=args.head_max_seconds,
                min_gain_bits_per_quality=args.head_min_gain, seed=args.head_seed,
                head_type=args.head_type, residual_dim=args.head_residual_dim)
        elif any(getattr(args, name) != build_parser().get_default(name) for name in (
            "head_max_reads", "head_max_symbols", "head_max_read_length", "head_steps", "head_symbols_per_step",
            "head_learning_rate", "head_anchor_strength", "head_max_seconds", "head_min_gain", "head_seed",
            "head_type", "head_residual_dim")):
            raise ValueError("--head-* training options require --finetune-head")
        statistics = encode_fastq(
            args.input,
            args.output,
            args.checkpoint,
            device=choose_device(args.device),
            batch_reads=args.batch_reads,
            quantization_total=args.quantization_total,
            progress=not args.no_progress,
            verify_cdf=args.verify_cdf,
            report_neural_only_bits=args.report_neural_only_bits,
            head_adaptation_config=head_config,
            head_adapter_path=args.head_adapter,
            save_head_adapter_path=args.save_head_adapter,
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
