"""Reduce suffix-LoRA mantissa precision within the existing FP32 v8 format.

Only predictor parameters are rounded. Quality symbols are still range-coded
losslessly using the exact rounded predictor reconstructed by the decoder.
"""
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
from pathlib import Path
from typing import Optional, Sequence
import zlib

import numpy as np

from .checkpoint import load_training_checkpoint
from .container import sha256_file
from .head_adapter import apply_head_adapter, adapter_container_bytes
from .suffix_lora_adapter import validate_suffix_lora, _canonical_parameters, _merged_digest


def round_float32_mantissa(values, retained_bits: int):
    """Round to nearest, ties to even; preserve the full FP32 exponent range."""
    if type(retained_bits) is not int or not 2 <= retained_bits <= 23:
        raise ValueError("mantissa bits must be an integer from 2 to 23")
    with np.errstate(over="ignore", invalid="ignore"):
        array = np.asarray(values).astype("<f4", copy=True)
    if not np.isfinite(array).all():
        raise ValueError("adapter parameters must be finite float32 values")
    if retained_bits == 23:
        return array
    dropped = 23 - retained_bits
    bits = array.view("<u4")
    increment = np.uint32((1 << (dropped - 1)) - 1) + ((bits >> dropped) & 1)
    mask = np.uint32(0xFFFFFFFF ^ ((1 << dropped) - 1))
    result = ((bits + increment) & mask).view("<f4")
    if not np.isfinite(result).all():
        raise ValueError("mantissa rounding overflowed float32")
    return result


def compact_suffix_adapter(metadata, base_model, base_sha256, *,
                           factor_mantissa_bits=8, head_mantissa_bits=23):
    """Return a validated v8 adapter, preserving rank, layout and shared base."""
    if not isinstance(metadata, dict) or metadata.get("version") != 8:
        raise ValueError("parameter compaction requires a suffix LoRA v8 adapter")
    values = validate_suffix_lora(metadata, base_model.config.d_model, base_sha256)
    # Verify the old canonical merge against the actual provided model first.
    apply_head_adapter(copy.deepcopy(base_model), metadata, base_sha256)
    result = copy.deepcopy(metadata)
    rounded = round_float32_mantissa(values, factor_mantissa_bits)
    raw = rounded.astype("<f4", copy=False).tobytes()
    planes = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 4).T.copy().tobytes()
    result["data_base64"] = base64.b64encode(zlib.compress(planes, 6)).decode("ascii")
    result["sha256"] = hashlib.sha256(raw).hexdigest()
    merged = _canonical_parameters(base_model, result["matrices"], rounded, result["rank"])
    result["merged_sha256"] = _merged_digest(merged)
    head = result["head"]
    head_raw = base64.b64decode(head["data_base64"], validate=True)
    head_values = np.frombuffer(head_raw, dtype="<f4")
    head_raw = round_float32_mantissa(head_values, head_mantissa_bits).astype("<f4", copy=False).tobytes()
    head["data_base64"] = base64.b64encode(head_raw).decode("ascii")
    head["sha256"] = hashlib.sha256(head_raw).hexdigest()
    validate_suffix_lora(result, base_model.config.d_model, base_sha256)
    apply_head_adapter(copy.deepcopy(base_model), result, base_sha256)
    return result


def compact_artifact(adapter_path, checkpoint_path, output_path, *,
                     factor_mantissa_bits=8, head_mantissa_bits=23):
    adapter_path, checkpoint_path, output_path = map(Path, (adapter_path, checkpoint_path, output_path))
    if output_path.exists() or output_path.resolve() in {adapter_path.resolve(), checkpoint_path.resolve()}:
        raise FileExistsError("choose a new adapter output path distinct from all inputs")
    artifact = json.loads(adapter_path.read_text())
    digest = sha256_file(checkpoint_path)
    if (not isinstance(artifact, dict) or artifact.get("format") != "direct-quality-head-adaptation-artifact"
            or type(artifact.get("version")) is not int or artifact["version"] != 1
            or artifact.get("base_checkpoint_sha256") != digest
            or not isinstance(artifact.get("adapter"), dict)):
        raise ValueError("adapter artifact or base checkpoint SHA-256 mismatch")
    base = load_training_checkpoint(checkpoint_path).model.eval().requires_grad_(False)
    metadata = compact_suffix_adapter(artifact["adapter"], base, digest,
        factor_mantissa_bits=factor_mantissa_bits, head_mantissa_bits=head_mantissa_bits)
    report = dict(reason="parameter_compaction", factor_mantissa_bits=factor_mantissa_bits,
                  head_mantissa_bits=head_mantissa_bits, original_adapter_sha256=sha256_file(adapter_path),
                  original_container_adapter_bytes=adapter_container_bytes(artifact["adapter"]),
                  container_adapter_bytes=adapter_container_bytes(metadata),
                  original_report=artifact.get("report", {}))
    result = dict(format="direct-quality-head-adaptation-artifact", version=1,
                  base_checkpoint_sha256=digest, adapter=metadata, report=report,
                  source=artifact.get("source"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return dict(output=str(output_path.resolve()), **{k: v for k, v in report.items() if k != "original_report"})


def main(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description="Compact a saved v8 LoRA adapter by rounding predictor parameters")
    parser.add_argument("adapter", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--factor-mantissa-bits", type=int, default=8)
    parser.add_argument("--head-mantissa-bits", type=int, default=23)
    args = parser.parse_args(argv)
    try:
        result = compact_artifact(args.adapter, args.checkpoint, args.output,
            factor_mantissa_bits=args.factor_mantissa_bits, head_mantissa_bits=args.head_mantissa_bits)
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
