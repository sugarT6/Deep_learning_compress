"""Bounded prefix adaptation of a frozen causal model's FP32 output head.

No online expert is invoked here. The decoder loads exact transmitted weights;
it never repeats optimization. Deadline checks are cooperative, not real-time.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .fastq_stream import encode_base_ids, iter_fastq_records, make_fastq_batch
from .model import fastq_batch_to_tensors
from .probability_quantization import logits_to_cdfs
from .encode_fastpath import selected_quantized_bits


ADAPTER_FORMAT = "direct-quality-output-head"


@dataclass(frozen=True)
class HeadAdaptationConfig:
    max_reads: int = 4096
    max_symbols: int = 600000
    max_read_length: int = 2048
    steps: int = 50
    symbols_per_step: int = 8192
    learning_rate: float = 0.001
    anchor_strength: float = 0.01
    max_seconds: float = 20.0
    min_gain_bits_per_quality: float = 0.0
    seed: int = 20260921

    def __post_init__(self):
        for name in ("max_reads", "max_symbols", "max_read_length", "steps", "symbols_per_step", "seed"):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name == "seed" else 1):
                raise ValueError(f"invalid head adaptation {name}")
        if self.max_reads < 4:
            raise ValueError("head adaptation needs max_reads >= 4")
        for name in ("learning_rate", "anchor_strength", "max_seconds", "min_gain_bits_per_quality"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"invalid head adaptation {name}")
        if self.learning_rate <= 0 or self.max_seconds <= 0 or self.anchor_strength < 0 or self.min_gain_bits_per_quality < 0:
            raise ValueError("invalid head adaptation rate, time, anchor or gain")


def serialize_head(head, base_sha256):
    arrays = [p.detach().cpu().numpy().astype("<f4", copy=False) for p in (head.weight, head.bias)]
    if not all(np.isfinite(a).all() for a in arrays):
        raise ValueError("adapter parameters must be finite")
    raw = b"".join(a.tobytes(order="C") for a in arrays)
    return {
        "format": ADAPTER_FORMAT, "version": 1, "dtype": "little_endian_float32",
        "layout": "weight_row_major_then_bias", "application": "replace_output_head",
        "base_checkpoint_sha256": base_sha256,
        "weight_shape": list(arrays[0].shape), "bias_shape": list(arrays[1].shape),
        "parameter_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
        "data_base64": base64.b64encode(raw).decode("ascii"),
    }


def validate_head_adapter(metadata, d_model, base_sha256):
    """Validate bounded plain data before allocating model tensors; no pickle."""
    keys = {"format", "version", "dtype", "layout", "application", "base_checkpoint_sha256",
            "weight_shape", "bias_shape", "parameter_bytes", "sha256", "data_base64"}
    if not isinstance(metadata, dict) or set(metadata) != keys:
        raise ValueError("adapter fields missing or unknown")
    if (metadata["format"] != ADAPTER_FORMAT or type(metadata["version"]) is not int
            or metadata["version"] != 1 or metadata["dtype"] != "little_endian_float32"
            or metadata["layout"] != "weight_row_major_then_bias"
            or metadata["application"] != "replace_output_head"):
        raise ValueError("unsupported adapter format or rules")
    if type(d_model) is not int or not 1 <= d_model <= 65536:
        raise ValueError("invalid adapter hidden dimension")
    for name, expected in (("weight_shape", [42, d_model]), ("bias_shape", [42])):
        shape = metadata[name]
        if not isinstance(shape, list) or any(type(v) is not int for v in shape) or shape != expected:
            raise ValueError("adapter shape mismatch")
    if metadata["base_checkpoint_sha256"] != base_sha256:
        raise ValueError("adapter base checkpoint SHA-256 mismatch")
    size = (42 * d_model + 42) * 4
    encoded = metadata["data_base64"]
    if (type(metadata["parameter_bytes"]) is not int or metadata["parameter_bytes"] != size
            or not isinstance(encoded, str) or len(encoded) != 4 * ((size + 2) // 3)):
        raise ValueError("adapter parameter length mismatch")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, UnicodeError) as exc:
        raise ValueError("invalid adapter base64") from exc
    if len(raw) != size or hashlib.sha256(raw).hexdigest() != metadata["sha256"]:
        raise ValueError("adapter length or SHA-256 mismatch")
    values = np.frombuffer(raw, dtype="<f4")
    if not np.isfinite(values).all():
        raise ValueError("adapter parameters must be finite")
    return values


def apply_head_adapter(model, metadata, base_sha256):
    values = validate_head_adapter(metadata, model.config.d_model, base_sha256)
    split = 42 * model.config.d_model
    with torch.no_grad():
        model.output_head.weight.copy_(torch.from_numpy(values[:split].copy().reshape(42, -1)))
        model.output_head.bias.copy_(torch.from_numpy(values[split:].copy()))


def adapt_output_head(model, source, *, device, batch_reads, total, base_sha256, config):
    """Return optional adapter + report; mutate only the accepted model head.

    Prefix split is by whole reads (first 75% train, last 25% validation).
    Admission tests neural quantized bits, not hybrid or whole-file savings.
    """
    started = time.perf_counter()
    deadline = started + config.max_seconds
    work_deadline = started + config.max_seconds * 0.70
    report = {"config": asdict(config), "accepted": False, "reason": "insufficient_prefix",
              "steps_completed": 0, "prefix_reads": 0, "prefix_symbols": 0,
              "train_reads": 0, "validation_reads": 0, "train_symbols": 0,
              "validation_symbols": 0, "parameter_bytes": 0, "metadata_bytes": 0,
              "before_validation_bits_per_quality": None,
              "after_validation_bits_per_quality": None,
              "selection_metric": "neural_only_quantized_bits_per_quality",
              "prefix_sha256": None, "sample_stop": "max_reads"}
    stages = {}

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def finish(adapter=None):
        sync()
        report["seconds"] = time.perf_counter() - started
        report["budget_exceeded"] = report["seconds"] > config.max_seconds
        report["stage_seconds"] = stages
        return adapter, report

    records = []
    digest = hashlib.sha256()
    iterator = iter_fastq_records(Path(source))
    try:
        for _ in range(config.max_reads):
            if time.perf_counter() >= work_deadline:
                report["sample_stop"] = "time_budget"
                break
            record = next(iterator, None)
            if record is None:
                report["sample_stop"] = "eof"
                break
            length = len(record.quality)
            if length > config.max_read_length or report["prefix_symbols"] + length > config.max_symbols:
                report["sample_stop"] = "length_or_symbol_cap"
                break
            records.append(record)
            digest.update(record.to_bytes())
            report["prefix_symbols"] += length
    finally:
        iterator.close()
    stages["prefix_read"] = time.perf_counter() - started
    report["prefix_sha256"] = digest.hexdigest()
    report["prefix_reads"] = len(records)
    if len(records) < 4:
        return finish()
    split = len(records) * 3 // 4
    report.update(train_reads=split, validation_reads=len(records) - split,
                  train_read_range=[0, split], validation_read_range=[split, len(records)])
    original_training = model.training
    original_requires_grad = [p.requires_grad for p in model.parameters()]
    model.eval()
    model.requires_grad_(False)
    try:
        feature_started = time.perf_counter()
        cached = []
        # Preallocate active-only features; avoid an additional full-size cat copy.
        for name, subset in (("train", records[:split]), ("validation", records[split:])):
            n = sum(len(r.quality) for r in subset)
            report[name + "_symbols"] = n
            if not n:
                return finish()
            features = torch.empty((n, model.config.d_model), dtype=torch.float32, device=device)
            labels = torch.empty(n, dtype=torch.long, device=device)
            offset = 0
            for begin in range(0, len(subset), batch_reads):
                if time.perf_counter() >= work_deadline:
                    report["reason"] = "feature_time_budget"
                    return finish()
                rows = subset[begin:begin + batch_reads]
                batch = make_fastq_batch([encode_base_ids(r.sequence) for r in rows],
                    [np.frombuffer(r.quality, dtype=np.uint8) - 33 for r in rows],
                    [r.read_index for r in rows], source_name=Path(source).name)
                tensors = fastq_batch_to_tensors(batch, device)
                with torch.no_grad():
                    h = model.forward_features(**tensors)
                    active = tensors["active_mask"]
                    count = int(batch.active_mask.sum())
                    features[offset:offset + count].copy_(h[active])
                    labels[offset:offset + count].copy_(tensors["qualities"][active])
                    offset += count
                sync()
            cached.append((features, labels))
        stages["features"] = time.perf_counter() - feature_started
        (train_h, train_y), (val_h, val_y) = cached
        head = copy.deepcopy(model.output_head).requires_grad_(True)
        anchor = [p.detach().clone() for p in head.parameters()]
        optimizer = torch.optim.Adam(head.parameters(), lr=config.learning_rate)
        generator = torch.Generator(device=device).manual_seed(config.seed)
        train_started = time.perf_counter()
        for _ in range(config.steps):
            if time.perf_counter() >= work_deadline:
                break
            indices = torch.randint(train_y.numel(), (min(config.symbols_per_step, train_y.numel()),),
                                    generator=generator, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(head(train_h[indices]), train_y[indices])
            loss = loss + config.anchor_strength * sum(
                (p - a).square().mean() for p, a in zip(head.parameters(), anchor))
            if not torch.isfinite(loss):
                report["reason"] = "nonfinite_training"
                return finish()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            sync()
            report["steps_completed"] += 1
        stages["optimization"] = time.perf_counter() - train_started
        if not report["steps_completed"]:
            report["reason"] = "training_time_budget"
            return finish()
        if not all(torch.isfinite(p).all() for p in head.parameters()):
            report["reason"] = "nonfinite_training"
            return finish()
        serialization_started = time.perf_counter()
        adapter = serialize_head(head, base_sha256)
        # Validate/restore the exact wire representation before admission scoring.
        values = validate_head_adapter(adapter, model.config.d_model, base_sha256)
        wire_head = copy.deepcopy(head).requires_grad_(False)
        with torch.no_grad():
            wire_head.weight.copy_(torch.from_numpy(values[:42 * model.config.d_model].copy().reshape(42, -1)))
            wire_head.bias.copy_(torch.from_numpy(values[42 * model.config.d_model:].copy()))
        report["candidate_parameter_bytes"] = adapter["parameter_bytes"]
        report["candidate_metadata_bytes"] = len(json.dumps(adapter, sort_keys=True, separators=(",", ":")).encode())
        stages["serialization"] = time.perf_counter() - serialization_started
        validation_started = time.perf_counter()

        def score(candidate):
            bits = 0.0
            with torch.no_grad():
                for begin in range(0, val_y.numel(), 4096):
                    if time.perf_counter() >= deadline:
                        return None
                    logits = candidate(val_h[begin:begin + 4096]).cpu().numpy()
                    targets = val_y[begin:begin + 4096].cpu().numpy()
                    cdfs = logits_to_cdfs(logits, total=total)
                    bits += selected_quantized_bits(targets, cdfs, total)
            return bits / val_y.numel()

        before, after = score(model.output_head), score(wire_head)
        stages["validation"] = time.perf_counter() - validation_started
        report["before_validation_bits_per_quality"] = before
        report["after_validation_bits_per_quality"] = after
        if before is None or after is None or time.perf_counter() >= deadline:
            report["reason"] = "validation_time_budget"
        elif before - after <= config.min_gain_bits_per_quality:
            report["reason"] = "no_validation_gain"
        else:
            apply_head_adapter(model, adapter, base_sha256)
            sync()
            report.update(accepted=True, reason="validation_gain", parameter_bytes=adapter["parameter_bytes"],
                          metadata_bytes=report["candidate_metadata_bytes"])
            return finish(adapter)
        return finish()
    finally:
        for p, flag in zip(model.parameters(), original_requires_grad):
            p.requires_grad_(flag)
        model.train(original_training)


__all__ = ["HeadAdaptationConfig", "adapt_output_head", "apply_head_adapter",
           "serialize_head", "validate_head_adapter"]
