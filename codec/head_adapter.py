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
from .model import CrossLayerOutputHead, ResidualOutputHead, fastq_batch_to_tensors
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
    head_type: str = "linear"
    residual_dim: int = 32
    cross_layer: bool = False
    cross_dim: int = 16
    lora: bool = False
    lora_rank: int = 4
    lora_steps: int = 1000
    lora_learning_rate: float = 0.0003

    def __post_init__(self):
        if type(self.lora) is not bool or (self.lora and not self.cross_layer):
            raise ValueError("lora requires cross_layer and a boolean flag")
        if type(self.lora_rank) is not int or not 1 <= self.lora_rank <= 16:
            raise ValueError("lora_rank must be in [1, 16]")
        if type(self.lora_steps) is not int or self.lora_steps < 1:
            raise ValueError("lora_steps must be positive")
        if (isinstance(self.lora_learning_rate, bool) or not isinstance(self.lora_learning_rate, (float, int))
                or not math.isfinite(self.lora_learning_rate) or self.lora_learning_rate <= 0):
            raise ValueError("invalid lora_learning_rate")
        if type(self.cross_layer) is not bool or (self.cross_layer and self.head_type != "residual"):
            raise ValueError("cross_layer requires a residual head and a boolean flag")
        if type(self.cross_dim) is not int or not 1 <= self.cross_dim <= 128:
            raise ValueError("cross_dim must be in [1, 128]")
        if self.head_type not in ("linear", "residual"):
            raise ValueError("head_type must be linear or residual")
        if type(self.residual_dim) is not int or not 1 <= self.residual_dim <= 128:
            raise ValueError("residual_dim must be in [1, 128]")
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


def _head_parameters(head):
    parameters = [head.weight, head.bias]
    if isinstance(head, ResidualOutputHead):
        parameters.extend((head.down.weight, head.down.bias, head.up.weight, head.up.bias))
    if isinstance(head, CrossLayerOutputHead):
        parameters.extend((head.cross_down.weight, head.cross_down.bias, head.cross_up.weight, head.cross_up.bias))
    return parameters


def _new_head(d_model, residual_dim, *, device, dtype, seed=0, cross_dim=None):
    # Initialization runs on CPU with an isolated RNG, including during decode.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        if cross_dim is not None:
            head = CrossLayerOutputHead(d_model, residual_dim, cross_dim)
        else:
            head = (ResidualOutputHead(d_model, residual_dim) if residual_dim is not None
                    else torch.nn.Linear(d_model, 42))
    return head.to(device=device, dtype=dtype)


def serialize_head(head, base_sha256):
    if getattr(head, "decode_only", False):
        raise ValueError("Q-history adapters are retired and decode-only")
    arrays = [p.detach().cpu().numpy().astype("<f4", copy=False) for p in _head_parameters(head)]
    if not all(np.isfinite(a).all() for a in arrays):
        raise ValueError("adapter parameters must be finite")
    raw = b"".join(a.tobytes(order="C") for a in arrays)
    metadata = {
        "format": ADAPTER_FORMAT, "version": 1, "dtype": "little_endian_float32",
        "layout": "weight_row_major_then_bias", "application": "replace_output_head",
        "base_checkpoint_sha256": base_sha256,
        "weight_shape": list(arrays[0].shape), "bias_shape": list(arrays[1].shape),
        "parameter_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
        "data_base64": base64.b64encode(raw).decode("ascii"),
    }
    if isinstance(head, ResidualOutputHead):
        metadata.update(version=2, residual_dim=head.down.out_features, activation="gelu_exact",
                        layout="weight_bias_down_weight_down_bias_up_weight_up_bias")
        for name, array in zip(("down_weight_shape", "down_bias_shape", "up_weight_shape", "up_bias_shape"), arrays[2:]):
            metadata[name] = list(array.shape)
    if isinstance(head, CrossLayerOutputHead):
        metadata.update(version=4, cross_dim=head.cross_down.out_features, source_layer=3,
                        source_normalization="layer_norm_no_affine", source_eps=1e-5,
                        layout=metadata["layout"] + "_cross_down_weight_cross_down_bias_cross_up_weight_cross_up_bias")
        for name, array in zip(("cross_down_weight_shape", "cross_down_bias_shape", "cross_up_weight_shape", "cross_up_bias_shape"), arrays[6:]):
            metadata[name] = list(array.shape)
    return metadata


def validate_head_adapter(metadata, d_model, base_sha256):
    """Validate bounded plain data before allocating model tensors; no pickle."""
    if isinstance(metadata, dict) and type(metadata.get("version")) is int and metadata["version"] == 6:
        from .lora_adapter import validate_lora_adapter
        return validate_lora_adapter(metadata, d_model, base_sha256)
    keys = {"format", "version", "dtype", "layout", "application", "base_checkpoint_sha256",
            "weight_shape", "bias_shape", "parameter_bytes", "sha256", "data_base64"}
    if not isinstance(metadata, dict):
        raise ValueError("adapter fields missing or unknown")
    version = metadata.get("version")
    if type(version) is not int or version not in (1, 2, 3, 4):
        raise ValueError("unsupported adapter version")
    if version >= 2:
        keys.update(("residual_dim", "activation", "down_weight_shape", "down_bias_shape", "up_weight_shape", "up_bias_shape"))
    if version == 3:
        keys.add("history_features")
    if version == 4:
        keys.update(("cross_dim", "source_layer", "source_normalization", "source_eps",
                     "cross_down_weight_shape", "cross_down_bias_shape", "cross_up_weight_shape", "cross_up_bias_shape"))
    if set(metadata) != keys:
        raise ValueError("adapter fields missing or unknown")
    if version == 3:
        from ._legacy_quality_history import validate_history_feature_schema
        validate_history_feature_schema(metadata["history_features"])
    layout = ("weight_row_major_then_bias" if version == 1 else
              "weight_bias_down_weight_down_bias_up_weight_up_bias")
    if version == 4:
        layout += "_cross_down_weight_cross_down_bias_cross_up_weight_cross_up_bias"
    if (metadata["format"] != ADAPTER_FORMAT or metadata["dtype"] != "little_endian_float32"
            or metadata["layout"] != layout
            or metadata["application"] != "replace_output_head"):
        raise ValueError("unsupported adapter format or rules")
    if type(d_model) is not int or not 1 <= d_model <= 65536:
        raise ValueError("invalid adapter hidden dimension")
    shapes = [("weight_shape", [42, d_model]), ("bias_shape", [42])]
    if version >= 2:
        width = metadata["residual_dim"]
        if type(width) is not int or not 1 <= width <= 128 or metadata["activation"] != "gelu_exact":
            raise ValueError("invalid residual head width or activation")
        input_dim = d_model + (8 if version == 3 else 0)
        shapes.extend((("down_weight_shape", [width, input_dim]), ("down_bias_shape", [width]),
                       ("up_weight_shape", [42, width]), ("up_bias_shape", [42])))
    if version == 4:
        width = metadata["cross_dim"]
        if (type(width) is not int or not 1 <= width <= 128
                or type(metadata["source_layer"]) is not int or metadata["source_layer"] != 3
                or metadata["source_normalization"] != "layer_norm_no_affine"
                or type(metadata["source_eps"]) is not float or metadata["source_eps"] != 1e-5):
            raise ValueError("invalid cross-layer head protocol")
        shapes.extend((("cross_down_weight_shape", [width, d_model]), ("cross_down_bias_shape", [width]),
                       ("cross_up_weight_shape", [42, width]), ("cross_up_bias_shape", [42])))
    for name, expected in shapes:
        shape = metadata[name]
        if not isinstance(shape, list) or any(type(v) is not int for v in shape) or shape != expected:
            raise ValueError("adapter shape mismatch")
    if metadata["base_checkpoint_sha256"] != base_sha256:
        raise ValueError("adapter base checkpoint SHA-256 mismatch")
    size = sum(math.prod(shape) for _, shape in shapes) * 4
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


def _restore_head(metadata, d_model, base_sha256, *, device, dtype):
    values = validate_head_adapter(metadata, d_model, base_sha256)
    if metadata["version"] == 3:
        from ._legacy_quality_history import LegacyHistoryHead
        with torch.random.fork_rng(devices=[]):
            head = LegacyHistoryHead(d_model, metadata["residual_dim"])
        head = head.to(device=device, dtype=dtype)
    else:
        head = _new_head(d_model, metadata.get("residual_dim"), device=device, dtype=dtype,
                         cross_dim=metadata.get("cross_dim"))
    offset = 0
    with torch.no_grad():
        for parameter in _head_parameters(head):
            stop = offset + parameter.numel()
            parameter.copy_(torch.from_numpy(values[offset:stop].copy().reshape(tuple(parameter.shape))))
            offset = stop
    return head


def apply_head_adapter(model, metadata, base_sha256):
    if isinstance(metadata, dict) and metadata.get("version") == 6:
        from .lora_adapter import apply_lora_adapter
        return apply_lora_adapter(model, metadata, base_sha256)
    if isinstance(metadata, dict) and metadata.get("version") == 4 and model.config.num_layers < 4:
        raise ValueError("cross-layer adapter requires at least 4 Transformer layers")
    original = model.output_head
    head = _restore_head(metadata, model.config.d_model, base_sha256,
                         device=original.weight.device, dtype=original.weight.dtype)
    head.train(original.training)
    head.requires_grad_(original.weight.requires_grad)
    head.bias.requires_grad_(original.bias.requires_grad)
    if hasattr(model, "_lora_original_qkv"):
        with torch.no_grad():
            model.transformer.layers[3].self_attn.in_proj_weight.copy_(model._lora_original_qkv)
        del model._lora_original_qkv
    model.output_head = head


def adapt_output_head(model, source, *, device, batch_reads, total, base_sha256, config):
    """Return optional adapter + report; mutate only the accepted model head.

    Prefix split is by whole reads (first 75% train, last 25% validation).
    Admission tests neural quantized bits, not hybrid or whole-file savings.
    """
    if config.lora:
        from .lora_adapter import adapt_last_layer
        return adapt_last_layer(model, source, device=device, batch_reads=batch_reads,
                               total=total, base_sha256=base_sha256, config=config)
    if getattr(model.output_head, "decode_only", False):
        raise ValueError("Q-history adapters are retired and cannot be fine-tuned")
    if config.cross_layer and model.config.num_layers < 4:
        raise ValueError("cross-layer adaptation requires at least 4 Transformer layers")
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
    original_requires_grad = {name: p.requires_grad for name, p in model.named_parameters()}
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
            features = torch.empty((n, model.config.d_model * (2 if config.cross_layer else 1)), dtype=torch.float32, device=device)
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
                    h = (model.forward_cross_layer_features(**tensors) if config.cross_layer
                         else model.forward_features(**tensors))
                    active = tensors["active_mask"]
                    count = int(batch.active_mask.sum())
                    features[offset:offset + count].copy_(h[active])
                    labels[offset:offset + count].copy_(tensors["qualities"][active])
                    offset += count
                sync()
            cached.append((features, labels))
        stages["features"] = time.perf_counter() - feature_started
        (train_h, train_y), (val_h, val_y) = cached
        if config.head_type == "residual":
            head = _new_head(model.config.d_model, config.residual_dim, device=device,
                             dtype=model.output_head.weight.dtype, seed=config.seed,
                             cross_dim=config.cross_dim if config.cross_layer else None)
            with torch.no_grad():
                head.weight.copy_(model.output_head.weight)
                head.bias.copy_(model.output_head.bias)
        else:
            head = copy.deepcopy(model.output_head)
        head.requires_grad_(True)
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
        wire_head = _restore_head(adapter, model.config.d_model, base_sha256,
                                  device=device, dtype=head.weight.dtype).requires_grad_(False)
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
                    inputs = val_h[begin:begin + 4096]
                    if config.cross_layer and not isinstance(candidate, CrossLayerOutputHead):
                        inputs = inputs[:, :model.config.d_model]
                    logits = candidate(inputs).cpu().numpy()
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
        for name, p in model.named_parameters():
            p.requires_grad_(original_requires_grad.get(name, original_requires_grad["output_head.weight"]))
        model.train(original_training)


__all__ = ["HeadAdaptationConfig", "adapt_output_head", "apply_head_adapter",
           "serialize_head", "validate_head_adapter"]
