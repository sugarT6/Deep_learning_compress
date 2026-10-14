"""Dense adaptation of a Transformer suffix and its output head.

The frozen prefix remains unchanged. Complete reads are sampled across the
input, and inference uses the exact transmitted FP32 replacement parameters.
No optimizer state, pickle, or training is needed by the decoder.
"""

import base64
import copy
from dataclasses import asdict, replace
import hashlib
import json
import math
import random
import sys
import time
import zlib

import numpy as np
import torch
from torch.nn import functional as F

from .fastq_stream import encode_base_ids, iter_fastq_records, make_fastq_batch
from .model import fastq_batch_to_tensors
from .probability_quantization import logits_to_cdfs
from .encode_fastpath import selected_quantized_bits
from .training_control import TrainingControl, TrainingControlConfig


PROTOCOL = "transformer_suffix_and_final_norm_xor32_byteplanes_zlib_v1"
MAX_RAW_BYTES = 32 << 20


def parameter_shapes(dim, ff, depth, layers):
    block = (
        ("self_attn.in_proj_weight", [3 * dim, dim]),
        ("self_attn.in_proj_bias", [3 * dim]),
        ("self_attn.out_proj.weight", [dim, dim]),
        ("self_attn.out_proj.bias", [dim]),
        ("linear1.weight", [ff, dim]), ("linear1.bias", [ff]),
        ("linear2.weight", [dim, ff]), ("linear2.bias", [dim]),
        ("norm1.weight", [dim]), ("norm1.bias", [dim]),
        ("norm2.weight", [dim]), ("norm2.bias", [dim]),
    )
    return [dict(name=f"transformer.layers.{i}.{name}", shape=shape)
            for i in layers for name, shape in block] + [
                dict(name="transformer.norm.weight", shape=[dim]),
                dict(name="transformer.norm.bias", shape=[dim])]


def serialize_backbone(model, layer_count, digest, *, base_model):
    from .head_adapter import ADAPTER_FORMAT, serialize_head
    cfg = model.config
    if type(layer_count) is not int or not 1 <= layer_count <= min(4, cfg.num_layers):
        raise ValueError("invalid number of backbone layers")
    if model.config.to_dict() != base_model.config.to_dict():
        raise ValueError("base and adapted model configurations must match")
    layers = list(range(cfg.num_layers - layer_count, cfg.num_layers))
    shapes = parameter_shapes(cfg.d_model, cfg.feedforward_dim, cfg.num_layers, layers)
    parameters = dict(model.named_parameters())
    raw = b"".join(parameters[item["name"]].detach().cpu().numpy().astype("<f4").tobytes()
                   for item in shapes)
    if not np.isfinite(np.frombuffer(raw, dtype="<f4")).all():
        raise ValueError("backbone parameters must be finite")
    base_raw = _base_bytes(base_model, shapes)
    delta = (np.frombuffer(raw, dtype="<u4") ^ np.frombuffer(base_raw, dtype="<u4")).astype("<u4")
    planes = delta.view(np.uint8).reshape(-1, 4).T.copy().tobytes()
    head = serialize_head(model.output_head, digest)
    meta = dict(format=ADAPTER_FORMAT, version=7, application="replace_head_and_transformer_suffix",
                protocol=PROTOCOL, base_checkpoint_sha256=digest, head=head,
                num_layers=cfg.num_layers, feedforward_dim=cfg.feedforward_dim,
                layers=layers, tensors=shapes, dtype="little_endian_float32",
                compression="xor32_byteplanes_zlib", raw_bytes=len(raw),
                parameter_bytes=len(raw) + head["parameter_bytes"],
                sha256=hashlib.sha256(raw).hexdigest(),
                delta_sha256=hashlib.sha256(planes).hexdigest(),
                data_base64=base64.b64encode(zlib.compress(planes, level=6)).decode("ascii"))
    validate_backbone_adapter(meta, cfg.d_model, digest)
    return meta


def validate_backbone_adapter(meta, dim, digest):
    from .head_adapter import ADAPTER_FORMAT, validate_head_adapter
    keys = {"format", "version", "application", "protocol", "base_checkpoint_sha256", "head",
            "num_layers", "feedforward_dim", "layers", "tensors", "dtype", "compression",
            "raw_bytes", "parameter_bytes", "sha256", "delta_sha256", "data_base64"}
    if (not isinstance(meta, dict) or set(meta) != keys or meta["format"] != ADAPTER_FORMAT
            or type(meta["version"]) is not int or meta["version"] != 7
            or meta["application"] != "replace_head_and_transformer_suffix"
            or meta["protocol"] != PROTOCOL or meta["base_checkpoint_sha256"] != digest
            or meta["dtype"] != "little_endian_float32"
            or meta["compression"] != "xor32_byteplanes_zlib"):
        raise ValueError("invalid backbone adapter protocol")
    depth, ff, layers = meta["num_layers"], meta["feedforward_dim"], meta["layers"]
    if (type(dim) is not int or not 1 <= dim <= 65536
            or type(depth) is not int or not 1 <= depth <= 64
            or type(ff) is not int or not 1 <= ff <= 65536
            or not isinstance(layers, list) or not 1 <= len(layers) <= min(4, depth)
            or any(type(i) is not int for i in layers)
            or layers != list(range(depth - len(layers), depth))):
        raise ValueError("invalid backbone suffix dimensions")
    if not isinstance(meta["head"], dict) or type(meta["head"].get("version")) is not int or meta["head"]["version"] not in (1, 2, 4):
        raise ValueError("backbone adapter requires a plain, residual, or cross-layer head")
    if meta["head"]["version"] == 4 and depth < 4:
        raise ValueError("cross-layer head requires four or more layers")
    validate_head_adapter(meta["head"], dim, digest)
    shapes = parameter_shapes(dim, ff, depth, layers)
    # Canonical JSON distinguishes bool/int and float/int in tensor descriptors.
    if json.dumps(meta["tensors"], sort_keys=True) != json.dumps(shapes, sort_keys=True):
        raise ValueError("backbone tensor layout mismatch")
    size = 4 * sum(math.prod(item["shape"]) for item in shapes)
    if (size > MAX_RAW_BYTES or type(meta["raw_bytes"]) is not int or meta["raw_bytes"] != size
            or type(meta["parameter_bytes"]) is not int
            or meta["parameter_bytes"] != size + meta["head"]["parameter_bytes"]
            or not isinstance(meta["data_base64"], str)
            or len(meta["data_base64"]) > 4 * ((size + (1 << 16) + 2) // 3)):
        raise ValueError("invalid backbone payload length")
    try:
        compressed = base64.b64decode(meta["data_base64"], validate=True)
        stream = zlib.decompressobj()
        planes = stream.decompress(compressed, size + 1)
    except (ValueError, UnicodeError, zlib.error) as exc:
        raise ValueError("invalid compressed backbone payload") from exc
    if (len(planes) != size or not stream.eof or stream.unused_data or stream.unconsumed_tail
            or hashlib.sha256(planes).hexdigest() != meta["delta_sha256"]):
        raise ValueError("backbone payload length or SHA-256 mismatch")
    if (not isinstance(meta["sha256"], str) or len(meta["sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in meta["sha256"])):
        raise ValueError("invalid reconstructed backbone SHA-256")
    return np.frombuffer(np.frombuffer(planes, dtype=np.uint8).reshape(4, -1).T.copy().tobytes(), dtype="<u4")


def _base_bytes(model, shapes):
    """Find pristine checkpoint tensors even when an adapter is already active."""
    parameters = dict(model.named_parameters())
    originals = getattr(model, "_backbone_original_parameters", {})
    raw = []
    for item in shapes:
        name = item["name"]
        value = originals.get(name, parameters[name])
        if name == "transformer.layers.3.self_attn.in_proj_weight" and hasattr(model, "_lora_original_qkv"):
            value = model._lora_original_qkv
        raw.append(value.detach().cpu().numpy().astype("<f4").tobytes())
    return b"".join(raw)


def restore_backbone(model):
    originals = getattr(model, "_backbone_original_parameters", None)
    if originals is not None:
        parameters = dict(model.named_parameters())
        with torch.no_grad():
            for name, value in originals.items():
                parameters[name].copy_(value)
        del model._backbone_original_parameters


def apply_backbone_adapter(model, meta, digest):
    from .head_adapter import _restore_head
    delta = validate_backbone_adapter(meta, model.config.d_model, digest)
    if (meta["num_layers"] != model.config.num_layers
            or meta["feedforward_dim"] != model.config.feedforward_dim):
        raise ValueError("backbone adapter model configuration mismatch")
    values = (np.frombuffer(_base_bytes(model, meta["tensors"]), dtype="<u4") ^ delta).astype("<u4").view("<f4")
    if not np.isfinite(values).all():
        raise ValueError("reconstructed backbone parameters must be finite")
    if hashlib.sha256(values.tobytes()).hexdigest() != meta["sha256"]:
        raise ValueError("reconstructed backbone SHA-256 mismatch")
    original = model.output_head
    head = _restore_head(meta["head"], model.config.d_model, digest,
                         device=original.weight.device, dtype=original.weight.dtype)
    head.train(original.training)
    head.requires_grad_(original.weight.requires_grad)
    restore_backbone(model)
    if hasattr(model, "_lora_original_qkv"):
        with torch.no_grad():
            model.transformer.layers[3].self_attn.in_proj_weight.copy_(model._lora_original_qkv)
        del model._lora_original_qkv
    parameters = dict(model.named_parameters())
    model._backbone_original_parameters = {
        item["name"]: parameters[item["name"]].detach().cpu().clone() for item in meta["tensors"]}
    offset = 0
    with torch.no_grad():
        for item in meta["tensors"]:
            count = math.prod(item["shape"])
            parameters[item["name"]].copy_(torch.from_numpy(values[offset:offset + count].copy().reshape(item["shape"])))
            offset += count
    model.output_head = head


def sample_records(source, config, skip_reads, deadline):
    """Uniform reservoir of complete nonempty reads, without Header features."""
    rng = random.Random(config.seed)
    records, eligible = [], 0
    report = dict(total_file_reads=0, total_file_symbols=0, eligible_symbols=0,
                  excluded_long_reads=0, excluded_warmup_reads=skip_reads)
    iterator = iter_fastq_records(source)
    try:
        for record in iterator:
            if time.perf_counter() >= deadline:
                return [], {**report, "sampling_complete": False}
            report["total_file_reads"] += 1
            length = len(record.quality)
            report["total_file_symbols"] += length
            if record.read_index < skip_reads or not length:
                continue
            if length > config.max_read_length:
                report["excluded_long_reads"] += 1
                continue
            eligible += 1
            report["eligible_symbols"] += length
            if len(records) < config.backbone_max_reads:
                records.append(record)
            else:
                index = rng.randrange(eligible)
                if index < len(records):
                    records[index] = record
    finally:
        iterator.close()
    rng.shuffle(records)
    selected, symbols = [], 0
    for record in records:
        if symbols + len(record.quality) > config.backbone_max_symbols:
            break
        selected.append(record)
        symbols += len(record.quality)
    report.update(sampling_complete=True, sampled_reads=len(selected), sampled_symbols=symbols,
                  read_indices_sha256=hashlib.sha256(
                      np.asarray([r.read_index for r in selected], dtype="<i8").tobytes()).hexdigest())
    return selected, report


def _tensors(records, device, source):
    batch = make_fastq_batch([encode_base_ids(r.sequence) for r in records],
        [np.frombuffer(r.quality, dtype=np.uint8) - 33 for r in records],
        [r.read_index for r in records], source_name=str(source))
    return fastq_batch_to_tensors(batch, device)


@torch.no_grad()
def score_model(model, records, *, device, batch_reads, source, total, version, deadline, quantized):
    was_training = model.training
    model.eval()
    try:
        bits, symbols = 0., 0
        for begin in range(0, len(records), batch_reads):
            if time.perf_counter() >= deadline:
                return None
            tensors = _tensors(records[begin:begin + batch_reads], device, source)
            active = tensors["active_mask"]
            logits, labels = model.forward_full(**tensors)[active], tensors["qualities"][active]
            if not torch.isfinite(logits).all():
                return float("inf")
            count = labels.numel()
            if not count:
                continue
            if quantized:
                cdfs = logits_to_cdfs(logits.cpu().numpy(), total=total, version=version)
                bits += selected_quantized_bits(labels.cpu().numpy(), cdfs, cdfs[:, -1])
            else:
                bits += float(F.cross_entropy(logits, labels, reduction="sum")) / math.log(2.)
            symbols += count
        return bits / symbols if symbols else None
    finally:
        model.train(was_training)


def adapt_backbone(model, source, *, device, batch_reads, total, base_sha256, config):
    from .head_adapter import (
        adapt_output_head, adapter_container_bytes, apply_head_adapter, _new_head,
    )
    if config.backbone_layers > model.config.num_layers:
        raise ValueError("cannot adapt more backbone layers than the model contains")
    if hasattr(model, "_backbone_original_parameters") or hasattr(model, "_lora_original_qkv"):
        raise ValueError("backbone fitting requires a fresh base model")
    started = time.perf_counter()
    deadline = started + config.max_seconds if config.max_seconds else float("inf")
    warm, warm_report = adapt_output_head(model, source, device=device, batch_reads=batch_reads,
        total=total, base_sha256=base_sha256, config=replace(config, backbone_layers=0))
    report = dict(warm_report)
    report.update(config=asdict(config), warmup_report=warm_report,
                  backbone_accepted=False, backbone_steps_completed=0, backbone_selected_step=0,
                  backbone_validation_history=[], backbone_stage_seconds={},
                  backbone_validation_scope="random_read_holdout_after_warmup_prefix")

    def finish(adapter, reason):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        report.update(reason=reason, accepted=adapter is not None,
                      seconds=time.perf_counter() - started)
        report["budget_exceeded"] = bool(config.max_seconds and report["seconds"] > config.max_seconds)
        return adapter, report

    sampling_started = time.perf_counter()
    records, sample = sample_records(source, config, warm_report["prefix_reads"], deadline)
    report["backbone_sampling"] = sample
    report["backbone_stage_seconds"]["sampling"] = time.perf_counter() - sampling_started
    if not sample["sampling_complete"] or len(records) < 4:
        return finish(warm, "backbone_sampling_budget_or_insufficient_reads")
    split = len(records) * 3 // 4
    train, validation = records[:split], records[split:]
    report.update(backbone_train_reads=len(train), backbone_validation_reads=len(validation))
    candidate = copy.deepcopy(model).eval().requires_grad_(False)
    if warm is None:
        old_head = candidate.output_head
        candidate.output_head = _new_head(model.config.d_model,
            config.residual_dim if config.head_type == "residual" else None,
            cross_dim=config.cross_dim if config.cross_layer else None,
            device=device, dtype=old_head.weight.dtype, seed=config.seed)
        with torch.no_grad():
            candidate.output_head.weight.copy_(old_head.weight)
            candidate.output_head.bias.copy_(old_head.bias)
    first = model.config.num_layers - config.backbone_layers
    for layer in candidate.transformer.layers[first:]:
        layer.requires_grad_(True)
    candidate.transformer.norm.requires_grad_(True)
    candidate.output_head.requires_grad_(True)
    backbone = [p for name, p in candidate.named_parameters() if p.requires_grad and not name.startswith("output_head.")]
    head = list(candidate.output_head.parameters())
    trainable = {name: p for name, p in candidate.named_parameters() if p.requires_grad}
    anchors = {name: p.detach().clone() for name, p in trainable.items()}
    optimizer = torch.optim.Adam([
        {"params": backbone, "lr": config.backbone_learning_rate},
        {"params": head, "lr": config.learning_rate},
    ])
    common = dict(device=device, batch_reads=config.backbone_batch_reads, source=source,
                  total=total, version=config.quantization_version, deadline=deadline)
    best_loss = score_model(candidate, validation, quantized=False, **common)
    if best_loss is None:
        return finish(warm, "backbone_validation_budget")
    if not math.isfinite(best_loss):
        return finish(warm, "backbone_nonfinite_validation")
    best = {name: p.detach().cpu().clone() for name, p in trainable.items()}
    control = TrainingControl(optimizer, TrainingControlConfig(
        lr_scheduler="plateau", lr_patience=2,
        min_lr=min(config.learning_rate, config.backbone_learning_rate) * .01,
        early_stopping_patience=config.backbone_patience), best_bits=best_loss)
    report["backbone_validation_history"].append(dict(step=0, bits_per_quality=best_loss))
    rng = random.Random(config.seed)
    optimization_started = time.perf_counter()
    for step in range(1, config.backbone_steps + 1):
        if time.perf_counter() >= deadline:
            break
        rows = rng.sample(train, min(config.backbone_batch_reads, len(train)))
        tensors = _tensors(rows, device, source)
        optimizer.zero_grad(set_to_none=True)
        active = tensors["active_mask"]
        loss = F.cross_entropy(candidate.forward_full(**tensors)[active], tensors["qualities"][active])
        loss = loss + config.anchor_strength * sum(
            (p - anchors[name]).square().mean() for name, p in trainable.items())
        if not torch.isfinite(loss):
            return finish(warm, "backbone_nonfinite_training")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(trainable.values()), 1.)
        optimizer.step()
        report["backbone_steps_completed"] = step
        if step % config.backbone_eval_every == 0 or step == config.backbone_steps:
            value = score_model(candidate, validation, quantized=False, **common)
            if value is None:
                break
            if not math.isfinite(value):
                return finish(warm, "backbone_nonfinite_validation")
            if value < best_loss:
                best_loss = value
                best = {name: p.detach().cpu().clone() for name, p in trainable.items()}
                report["backbone_selected_step"] = step
            stop = control.step(value)
            report["backbone_validation_history"].append(dict(
                step=step, bits_per_quality=value, backbone_lr=optimizer.param_groups[0]["lr"],
                head_lr=optimizer.param_groups[1]["lr"]))
            print(f"Backbone step={step} val_bits={value:.6f} best={best_loss:.6f}", file=sys.stderr)
            if stop:
                report["backbone_stopped_early"] = True
                break
    report["backbone_stage_seconds"]["optimization"] = time.perf_counter() - optimization_started
    if report["backbone_selected_step"] == 0:
        return finish(warm, "backbone_no_heldout_gain")
    with torch.no_grad():
        for name, parameter in trainable.items():
            parameter.copy_(best[name])
    serialization_started = time.perf_counter()
    adapter = serialize_backbone(candidate, config.backbone_layers, base_sha256, base_model=model)
    wire = copy.deepcopy(model)
    apply_backbone_adapter(wire, adapter, base_sha256)
    wire.eval()
    report["backbone_stage_seconds"]["serialization"] = time.perf_counter() - serialization_started
    validation_started = time.perf_counter()
    before = score_model(model, validation, quantized=True, **common)
    after = score_model(wire, validation, quantized=True, **common)
    report["backbone_stage_seconds"]["wire_validation"] = time.perf_counter() - validation_started
    report.update(candidate_parameter_bytes=adapter["parameter_bytes"],
                  candidate_container_adapter_bytes=adapter_container_bytes(adapter),
                  candidate_backbone_raw_bytes=adapter["raw_bytes"],
                  backbone_before_validation_bits_per_quality=before,
                  backbone_after_validation_bits_per_quality=after)
    if before is None or after is None:
        return finish(warm, "backbone_wire_validation_budget")
    if not math.isfinite(before) or not math.isfinite(after):
        return finish(warm, "backbone_nonfinite_wire_validation")
    saving = (before - after) * sample["eligible_symbols"] / 8.
    cost = adapter_container_bytes(adapter) - adapter_container_bytes(warm)
    report.update(estimated_quality_saving_bytes=saving, incremental_adapter_bytes=cost,
                  estimated_net_saving_bytes=saving - cost)
    if before - after <= config.min_gain_bits_per_quality:
        return finish(warm, "backbone_no_quantized_gain")
    if saving <= cost:
        return finish(warm, "backbone_weight_cost_exceeds_estimated_saving")
    apply_head_adapter(model, adapter, base_sha256)
    report["backbone_accepted"] = True
    return finish(adapter, "backbone_estimated_net_gain")
