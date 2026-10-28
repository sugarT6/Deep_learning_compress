"""Paired per-file adaptation probes; these do not produce codec checkpoints.

The two branches share the already warmed model and the exact read sampler.
Only held-out float BPQ chooses a trained state.  Final scores use that state
and, for LoRA, the same canonical merge that a decoder would receive.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import time
from typing import Mapping

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.utils import parametrize

from .backbone_adapter import _tensors
from .encode_fastpath import selected_quantized_bits
from .head_adapter import apply_head_adapter, adapter_container_bytes
from .probability_quantization import logits_to_cdfs, validate_total
from .suffix_lora_adapter import install_suffix_lora, serialize_suffix_lora
from .training_control import TrainingControl, TrainingControlConfig


DIAGNOSTIC_FORMAT = "direct-quality-adaptation-diagnostic"
DIAGNOSTIC_VERSION = 1
_LOG_TWO = math.log(2.0)


def _positive_integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _finite_rate(value, name, *, allow_zero=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0
            or (not allow_zero and value == 0)):
        raise ValueError(f"invalid {name}")


def _synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def score_records(model, records, *, device, batch_reads, source,
                  total=65536, version=2, quantized=False):
    """Score every active Q once, optionally quantizing the same forward pass.

    Counts include empty reads, but padding and empty reads never enter either
    sum.  No score is defined for an input with no quality symbols.
    """
    _positive_integer(batch_reads, "batch_reads")
    if not len(records):
        raise ValueError("cannot score an empty collection of reads")
    if type(quantized) is not bool:
        raise ValueError("quantized must be a boolean")
    if type(version) is not int or version not in (1, 2):
        raise ValueError("unsupported quantization version")
    total = validate_total(total)
    was_training = model.training
    model.eval()
    float_bits, quantized_bits, symbols = 0.0, 0.0, 0
    try:
        for begin in range(0, len(records), batch_reads):
            rows = records[begin:begin + batch_reads]
            # An all-empty batch need not enter attention with a zero length.
            if not any(len(record.quality) for record in rows):
                continue
            tensors = _tensors(rows, device, source)
            active = tensors["active_mask"]
            logits = model.forward_full(**tensors)[active]
            labels = tensors["qualities"][active]
            if not labels.numel():
                continue
            if not torch.isfinite(logits).all():
                raise ValueError("nonfinite diagnostic scoring logits")
            value = float(F.cross_entropy(logits, labels, reduction="sum")) / _LOG_TWO
            if not math.isfinite(value):
                raise ValueError("nonfinite diagnostic float code length")
            float_bits += value
            symbols += labels.numel()
            if quantized:
                cdfs = logits_to_cdfs(logits.detach().cpu().numpy(), total=total,
                                     version=version)
                value = float(selected_quantized_bits(
                    labels.detach().cpu().numpy(), cdfs, cdfs[:, -1]))
                if not math.isfinite(value):
                    raise ValueError("nonfinite diagnostic quantized code length")
                quantized_bits += value
        if not symbols:
            raise ValueError("cannot score reads without quality symbols")
        return {
            "reads": len(records), "quality_symbols": int(symbols),
            "float_bits": float_bits, "float_bpq": float_bits / symbols,
            "quantized_bits": quantized_bits if quantized else None,
            "quantized_bpq": quantized_bits / symbols if quantized else None,
        }
    finally:
        model.train(was_training)


def prepare_branch_model(warm_model, *, mode, device, lora_layers=2,
                         lora_rank=8, seed=20260921):
    """Return an independent branch without modifying the shared warm model."""
    if mode not in ("lora", "full"):
        raise ValueError("mode must be 'lora' or 'full'")
    if any(parametrize.is_parametrized(module) for module in warm_model.modules()):
        raise ValueError("the shared warm model must not have parametrizations")
    model = copy.deepcopy(warm_model).to(device)
    model.eval().requires_grad_(False)
    if mode == "lora":
        install_suffix_lora(model, lora_layers, lora_rank, seed)
        model.transformer.norm.requires_grad_(True)
        model.output_head.requires_grad_(True)
    else:
        model.requires_grad_(True)
    return model


def _head_spec(head):
    if hasattr(head, "cross_down"):
        return {"type": "cross_layer", "residual_dim": head.down.out_features,
                "cross_dim": head.cross_down.out_features}
    if hasattr(head, "down"):
        return {"type": "residual", "residual_dim": head.down.out_features}
    return {"type": "linear"}


def _write_json_new(path, payload):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _save_checkpoint_new(path, payload):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    os.close(descriptor)
    try:
        torch.save(payload, temporary_name)
        os.replace(temporary_name, str(path))
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def fit_branch(warm_model, train_records, validation_records, train_probe_records,
               *, mode, source, device, output_dir, base_sha256, initial_metrics,
               steps=10000, eval_every=200, batch_reads=16, eval_batch_reads=256,
               learning_rate=.0003, head_learning_rate=.001, anchor_strength=.01,
               seed=20260921, lora_layers=2, lora_rank=8, patience=6,
               total=65536, quantization_version=2, progress=True):
    """Fit a paired diagnostic branch and save its best held-out trained state.

    The returned model checkpoint is deliberately not a training/codec
    checkpoint.  Scores and parameter sizes here do not describe an encoded
    file or account for a deployable full-model distribution format.
    """
    for name, value in (("steps", steps), ("eval_every", eval_every),
                        ("batch_reads", batch_reads), ("eval_batch_reads", eval_batch_reads)):
        _positive_integer(value, name)
    for name, value in (("learning_rate", learning_rate),
                        ("head_learning_rate", head_learning_rate)):
        _finite_rate(value, name)
    _finite_rate(anchor_strength, "anchor_strength", allow_zero=True)
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(patience) is not int or patience < 0:
        raise ValueError("patience must be a nonnegative integer")
    if not all(len(records) for records in (
            train_records, validation_records, train_probe_records)):
        raise ValueError("training, validation and training probe reads must be nonempty")
    if any(not len(record.quality) for record in train_records):
        raise ValueError("diagnostic training reads must have quality symbols")
    train_by_index = {record.read_index: record for record in train_records}
    validation_indices = {record.read_index for record in validation_records}
    probe_indices = {record.read_index for record in train_probe_records}
    if (len(train_by_index) != len(train_records)
            or len(validation_indices) != len(validation_records)
            or len(probe_indices) != len(train_probe_records)):
        raise ValueError("diagnostic read collections must not contain duplicate indices")
    if set(train_by_index) & validation_indices:
        raise ValueError("diagnostic training and validation reads must not overlap")
    if not probe_indices <= set(train_by_index):
        raise ValueError("the training probe must be a subset of training reads")
    for record in train_probe_records:
        original = train_by_index[record.read_index]
        if record.sequence != original.sequence or record.quality != original.quality:
            raise ValueError("training probe content does not match its training read")
    total = validate_total(total)
    if type(quantization_version) is not int or quantization_version not in (1, 2):
        raise ValueError("unsupported quantization version")
    if not isinstance(initial_metrics, Mapping):
        raise ValueError("initial_metrics must be a mapping")
    if (not isinstance(base_sha256, str) or len(base_sha256) != 64
            or any(c not in "0123456789abcdef" for c in base_sha256)):
        raise ValueError("base_sha256 must be a SHA-256 hex digest")
    if mode not in ("lora", "full"):
        raise ValueError("mode must be 'lora' or 'full'")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / f"{mode}.history.jsonl"
    checkpoint_path = output_dir / f"{mode}.best.pt"
    adapter_path = output_dir / f"{mode}.adapter.json"
    for path in (history_path, checkpoint_path, adapter_path):
        if path.exists():
            raise FileExistsError(path)
    device = torch.device(device)
    candidate = prepare_branch_model(warm_model, mode=mode, device=device,
        lora_layers=lora_layers, lora_rank=lora_rank, seed=seed)
    trainable = {name: parameter for name, parameter in candidate.named_parameters()
                 if parameter.requires_grad}
    body = [parameter for name, parameter in trainable.items()
            if not name.startswith("output_head.")]
    head = list(candidate.output_head.parameters())
    optimizer = torch.optim.Adam([
        {"params": body, "lr": learning_rate},
        {"params": head, "lr": head_learning_rate},
    ])
    anchors = ({name: parameter.detach().clone() for name, parameter in trainable.items()}
               if anchor_strength else {})
    score_kwargs = dict(device=device, batch_reads=eval_batch_reads, source=source,
                        total=total, version=quantization_version)
    initial = copy.deepcopy(dict(initial_metrics))
    for key, records in (("train_probe", train_probe_records),
                         ("validation", validation_records)):
        if key not in initial:
            initial[key] = score_records(candidate, records, **score_kwargs)
        if not math.isfinite(initial[key]["float_bpq"]):
            raise ValueError("nonfinite initial diagnostic BPQ")
    control = TrainingControl(optimizer, TrainingControlConfig(
        lr_scheduler="plateau", lr_factor=.5, lr_patience=2,
        min_lr=min(learning_rate, head_learning_rate) * .01,
        min_delta=1e-4, early_stopping_patience=patience),
        best_bits=initial["validation"]["float_bpq"])
    settings = {
        "steps": steps, "eval_every": eval_every, "batch_reads": batch_reads,
        "eval_batch_reads": eval_batch_reads, "learning_rate": learning_rate,
        "head_learning_rate": head_learning_rate, "anchor_strength": anchor_strength,
        "seed": seed, "lora_layers": lora_layers if mode == "lora" else None,
        "lora_rank": lora_rank if mode == "lora" else None, "patience": patience,
        "total": total, "quantization_version": quantization_version,
        "dropout_enabled": False, "selection_metric": "validation_float_bpq",
        "step_zero_is_candidate": False, "training_control": control.config.to_dict(),
        "anchor_definition": "coefficient_times_sum_of_parameter_tensor_mean_squared_deltas",
    }
    rng = random.Random(seed)
    draw_digest = hashlib.sha256()
    best_loss, best_parameters, selected_step = float("inf"), None, None
    steps_completed, stopped_early = 0, False
    started = time.perf_counter()
    last_evaluation = None

    with history_path.open("x", encoding="utf-8") as history:
        def record_evaluation(step, probe_score, validation_score,
                              batch_float_bpq=None, anchor_penalty=None):
            nonlocal last_evaluation
            used_lr = optimizer.param_groups[0]["lr"]
            used_head_lr = optimizer.param_groups[1]["lr"]
            stop = control.step(validation_score["float_bpq"]) if step else False
            _synchronize(device)
            row = {
                "step": step, "train_probe": probe_score, "validation": validation_score,
                "last_batch_float_bpq": batch_float_bpq,
                "last_batch_cross_entropy_nats": (
                    batch_float_bpq * _LOG_TWO if batch_float_bpq is not None else None),
                "last_batch_anchor_penalty": anchor_penalty,
                "last_batch_anchor_penalty_nats": anchor_penalty,
                "learning_rate": used_lr, "head_learning_rate": used_head_lr,
                "next_learning_rate": optimizer.param_groups[0]["lr"],
                "next_head_learning_rate": optimizer.param_groups[1]["lr"],
                "elapsed_seconds": time.perf_counter() - started,
                "batch_read_indices_sha256": draw_digest.hexdigest(),
                "stopped_early": stop,
            }
            history.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            history.flush()
            last_evaluation = row
            if progress:
                print(f"{mode} step={step} train_probe_bpq={probe_score['float_bpq']:.6f} "
                      f"val_bpq={validation_score['float_bpq']:.6f} "
                      f"best_bpq={best_loss if step else validation_score['float_bpq']:.6f} "
                      f"lr={used_lr:.6g} next_lr={optimizer.param_groups[0]['lr']:.6g}",
                      file=sys.stderr, flush=True)
            return stop

        record_evaluation(0, initial["train_probe"], initial["validation"])
        for step in range(1, steps + 1):
            rows = rng.sample(train_records, min(batch_reads, len(train_records)))
            tensors = _tensors(rows, device, source)
            active = tensors["active_mask"]
            if not bool(active.any()):
                raise ValueError("diagnostic training batch has no quality symbols")
            optimizer.zero_grad(set_to_none=True)
            logits = candidate.forward_full(**tensors)[active]
            if not torch.isfinite(logits).all():
                raise ValueError("nonfinite diagnostic training logits")
            ce = F.cross_entropy(logits, tensors["qualities"][active])
            penalty = ce.new_zeros(())
            if anchor_strength:
                penalty = anchor_strength * sum(
                    (parameter - anchors[name]).square().mean()
                    for name, parameter in trainable.items())
            loss = ce + penalty
            if not torch.isfinite(loss):
                raise ValueError("nonfinite diagnostic training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(trainable.values()), 1.,
                                           error_if_nonfinite=True)
            optimizer.step()
            steps_completed = step
            draw_digest.update(np.asarray(
                [record.read_index for record in rows], dtype="<i8").tobytes())
            if step % eval_every != 0 and step != steps:
                continue
            probe_score = score_records(candidate, train_probe_records, **score_kwargs)
            validation_score = score_records(candidate, validation_records, **score_kwargs)
            value = validation_score["float_bpq"]
            if value < best_loss:
                best_loss, selected_step = value, step
                best_parameters = {name: parameter.detach().cpu().clone()
                                   for name, parameter in trainable.items()}
            stop = record_evaluation(step, probe_score, validation_score,
                                     float(ce.detach()) / _LOG_TWO,
                                     float(penalty.detach()))
            if stop:
                stopped_early = True
                break

    if best_parameters is None:
        raise ValueError("no trained diagnostic state was evaluated")
    with torch.no_grad():
        for name, parameter in trainable.items():
            parameter.copy_(best_parameters[name])
    optimization_and_evaluation_seconds = time.perf_counter() - started

    adapter = None
    if mode == "lora":
        adapter = serialize_suffix_lora(candidate, lora_layers, lora_rank,
                                        base_sha256, base_model=warm_model)
        selected = copy.deepcopy(warm_model).to(device)
        apply_head_adapter(selected, adapter, base_sha256)
        wire_parameterization = "canonical_lora"
    else:
        selected = candidate
        wire_parameterization = "direct_full"
    selected.eval()
    final_started = time.perf_counter()
    selected_metrics = {
        name: score_records(selected, records, quantized=True, **score_kwargs)
        for name, records in (("train", train_records),
                              ("validation", validation_records),
                              ("train_probe", train_probe_records))
    }
    _synchronize(device)
    final_scoring_seconds = time.perf_counter() - final_started
    payload = {
        "format": DIAGNOSTIC_FORMAT, "version": DIAGNOSTIC_VERSION,
        "mode": mode, "model_config": selected.config.to_dict(),
        "model_state_dict": {name: value.detach().cpu().clone()
                             for name, value in selected.state_dict().items()},
        "head_spec": _head_spec(selected.output_head),
        "base_checkpoint_sha256": base_sha256,
        "selected_step": selected_step, "wire_parameterization": wire_parameterization,
    }
    _save_checkpoint_new(checkpoint_path, payload)
    if adapter is not None:
        _write_json_new(adapter_path, {
            "format": "direct-quality-head-adaptation-artifact", "version": 1,
            "base_checkpoint_sha256": base_sha256, "adapter": adapter,
            "report": {"diagnostic": True, "mode": mode,
                       "selected_step": selected_step, "settings": settings},
            "source": str(Path(source).resolve()),
        })
    return {
        "mode": mode, "steps_completed": steps_completed, "selected_step": selected_step,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable.values()),
        "backbone_trainable_parameters": sum(parameter.numel() for parameter in body),
        "head_trainable_parameters": sum(parameter.numel() for parameter in head),
        "initial_metrics": initial, "selected_metrics": selected_metrics,
        "selected_validation_float_bpq_before_wire": best_loss,
        "last_evaluation": last_evaluation,
        "batch_read_indices_sha256": draw_digest.hexdigest(),
        "batch_read_indices_hash_encoding": "concatenated_step_sample_order_little_endian_int64",
        "history_path": str(history_path.resolve()),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "adapter_path": str(adapter_path.resolve()) if adapter is not None else None,
        "canonical_container_adapter_bytes": adapter_container_bytes(adapter) if adapter is not None else None,
        "wire_parameterization": wire_parameterization, "settings": settings,
        "stopped_early": stopped_early,
        "optimization_and_evaluation_seconds": optimization_and_evaluation_seconds,
        "final_scoring_seconds": final_scoring_seconds,
        "diagnostic_only": True,
    }
