"""Bounded last-block Q/V LoRA, with canonical wire-weight reconstruction."""
from dataclasses import asdict, replace
import base64
import copy
import hashlib
import itertools
import json
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.func import functional_call

from .fastq_stream import encode_base_ids, iter_fastq_records, make_fastq_batch
from .model import fastq_batch_to_tensors
from .probability_quantization import logits_to_cdfs
from .encode_fastpath import selected_quantized_bits


PROTOCOL = "block4_qv_rank_major_f64_merge_v1"


class QVLoRA(nn.Module):
    def __init__(self, dim, rank, seed):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.qa = nn.Parameter(torch.randn(rank, dim) * 0.02)
            self.qb = nn.Parameter(torch.zeros(dim, rank))
            self.va = nn.Parameter(torch.randn(rank, dim) * 0.02)
            self.vb = nn.Parameter(torch.zeros(dim, rank))

    def merged(self, base):
        dim = base.shape[1]
        return torch.cat((base[:dim] + self.qb @ self.qa, base[dim:2*dim],
                          base[2*dim:] + self.vb @ self.va), dim=0)


def validate_lora_adapter(meta, dim, digest):
    from .head_adapter import ADAPTER_FORMAT, validate_head_adapter
    keys = {"format", "version", "application", "protocol", "rank", "head",
            "base_checkpoint_sha256", "parameter_bytes", "data_base64", "sha256"}
    if (set(meta) != keys or meta["format"] != ADAPTER_FORMAT
            or type(meta["version"]) is not int or meta["version"] != 6
            or meta["application"] != "replace_head_and_block4_qv"
            or meta["protocol"] != PROTOCOL or meta["base_checkpoint_sha256"] != digest):
        raise ValueError("invalid LoRA adapter protocol")
    rank = meta["rank"]
    if type(rank) is not int or not 1 <= rank <= 16:
        raise ValueError("invalid LoRA rank")
    # Restrict nested schemas before recursion; never permit recursive envelopes.
    if not isinstance(meta["head"], dict) or meta["head"].get("version") != 4:
        raise ValueError("LoRA requires an embedded cross-layer head v4")
    validate_head_adapter(meta["head"], dim, digest)
    size = 4 * dim * rank * 4
    encoded = meta["data_base64"]
    if (type(meta["parameter_bytes"]) is not int
            or meta["parameter_bytes"] != meta["head"]["parameter_bytes"] + size
            or not isinstance(encoded, str) or len(encoded) != 4 * ((size + 2) // 3)):
        raise ValueError("invalid LoRA parameter length")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, UnicodeError) as exc:
        raise ValueError("invalid LoRA base64") from exc
    if len(raw) != size or hashlib.sha256(raw).hexdigest() != meta["sha256"]:
        raise ValueError("invalid LoRA SHA-256")
    values = np.frombuffer(raw, dtype="<f4")
    if not np.isfinite(values).all():
        raise ValueError("LoRA parameters must be finite")
    return values


def serialize_lora(head, lora, digest):
    from .head_adapter import ADAPTER_FORMAT, serialize_head
    nested = serialize_head(head, digest)
    raw = b"".join(p.detach().cpu().numpy().astype("<f4").tobytes() for p in lora.parameters())
    meta = dict(format=ADAPTER_FORMAT, version=6, application="replace_head_and_block4_qv",
        protocol=PROTOCOL, rank=lora.qa.shape[0], head=nested, base_checkpoint_sha256=digest,
        parameter_bytes=nested["parameter_bytes"] + len(raw),
        data_base64=base64.b64encode(raw).decode("ascii"), sha256=hashlib.sha256(raw).hexdigest())
    validate_lora_adapter(meta, head.in_features, digest)
    return meta


def canonical_weight(base, values, rank):
    """No GPU/BLAS matrix product: fixed rank-order FP64 accumulation, then FP32."""
    base = base.detach().cpu().numpy().astype(np.float64)
    dim = base.shape[1]
    parts = np.split(values, 4)
    for row, a, b in ((0, parts[0], parts[1]), (2*dim, parts[2], parts[3])):
        a, b = a.reshape(rank, dim).astype(np.float64), b.reshape(dim, rank).astype(np.float64)
        delta = np.zeros((dim, dim), dtype=np.float64)
        for k in range(rank):
            delta += b[:, k:k+1] * a[k:k+1, :]
        base[row:row+dim] += delta
    with np.errstate(over="ignore", invalid="ignore"):
        merged = base.astype(np.float32)
    if not np.isfinite(merged).all():
        raise ValueError("merged LoRA weight must be finite")
    return torch.from_numpy(merged)


def apply_lora_adapter(model, meta, digest):
    from .head_adapter import _restore_head
    if model.config.num_layers != 4:
        raise ValueError("LoRA adapter requires exactly four Transformer layers")
    values = validate_lora_adapter(meta, model.config.d_model, digest)
    original = model.output_head
    head = _restore_head(meta["head"], model.config.d_model, digest,
                         device=original.weight.device, dtype=original.weight.dtype)
    head.train(original.training)
    head.requires_grad_(original.weight.requires_grad)
    head.bias.requires_grad_(original.bias.requires_grad)
    weight = model.transformer.layers[3].self_attn.in_proj_weight
    # Preserve base tensor for idempotent artifact application (not a checkpoint parameter).
    base = getattr(model, "_lora_original_qkv", weight.detach().cpu().clone())
    merged = canonical_weight(base, values, meta["rank"])
    model._lora_original_qkv = base
    with torch.no_grad():
        weight.copy_(merged)
    model.output_head = head


def adapt_last_layer(model, source, *, device, batch_reads, total, base_sha256, config):
    from .head_adapter import adapt_output_head, _restore_head
    if model.config.num_layers != 4 or hasattr(model, "_lora_original_qkv"):
        raise ValueError("LoRA fitting requires a fresh four-layer base model")
    started = time.perf_counter()
    deadline, work_deadline = started + config.max_seconds, started + 0.70 * config.max_seconds
    warm, warm_report = adapt_output_head(model, source, device=device, batch_reads=batch_reads,
        total=total, base_sha256=base_sha256, config=replace(config, lora=False))
    report = dict(warm_report)
    report.update(config=asdict(config), warmup_report=warm_report,
                  lora_steps_completed=0, lora_selected_step=0, lora_accepted=False,
                  lora_validation_history=[], lora_stage_seconds={})

    def finish(adapter, reason):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        report.update(reason=reason, seconds=time.perf_counter()-started)
        report["budget_exceeded"] = report["seconds"] > config.max_seconds
        return adapter, report

    if warm is None:
        return finish(None, "lora_warmup_" + warm_report["reason"])
    if time.perf_counter() >= work_deadline:
        return finish(warm, "lora_no_remaining_training_budget")
    # Preserve complete sequences; sampled positions alone cannot train attention.
    cache_started = time.perf_counter()
    iterator = iter_fastq_records(source)
    try:
        records = list(itertools.islice(iterator, warm_report["prefix_reads"]))
    finally:
        iterator.close()
    prefix_digest = hashlib.sha256()
    for record in records:
        prefix_digest.update(record.to_bytes())
    if (len(records) != warm_report["prefix_reads"]
            or prefix_digest.hexdigest() != warm_report["prefix_sha256"]):
        raise ValueError("prefix changed between head warm-up and LoRA caching")
    split = warm_report["train_reads"]
    original_training = model.training
    model.eval()
    caches = []
    try:
        for subset in (records[:split], records[split:]):
            chunks = []
            for begin in range(0, len(subset), 32):
                if time.perf_counter() >= work_deadline:
                    return finish(warm, "lora_cache_time_budget")
                rows = subset[begin:begin+32]
                batch = make_fastq_batch([encode_base_ids(r.sequence) for r in rows],
                    [np.frombuffer(r.quality, dtype=np.uint8)-33 for r in rows],
                    [r.read_index for r in rows], source_name=str(source))
                tensors = fastq_batch_to_tensors(batch, device)
                captured = []
                handle = model.transformer.layers[2].register_forward_hook(lambda m, a, o: captured.append(o))
                try:
                    with torch.no_grad():
                        model.forward_features(**tensors)
                finally:
                    handle.remove()
                chunks.append((captured[0].detach(), tensors["qualities"], tensors["active_mask"]))
            caches.append(chunks)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        report["lora_stage_seconds"]["cache"] = time.perf_counter()-cache_started
        train, validation = caches
        layer = copy.deepcopy(model.transformer.layers[3]).eval().requires_grad_(False)
        norm = copy.deepcopy(model.transformer.norm).eval().requires_grad_(False)
        head = copy.deepcopy(model.output_head).eval().requires_grad_(True)
        lora = QVLoRA(model.config.d_model, config.lora_rank, config.seed).to(device)
        params = list(head.parameters()) + list(lora.parameters())
        anchors = [p.detach().clone() for p in params]
        optimizer = torch.optim.Adam(params, lr=config.lora_learning_rate)
        generator = torch.Generator().manual_seed(config.seed)

        def logits(chunk, candidate_head=head, weight=None):
            h, y, active = chunk
            if h.shape[1] == 0:
                return h.new_empty((h.shape[0], 0, 42))
            mask = model._attention_mask(h.shape[1], h.device)
            if weight is None:
                weight = lora.merged(layer.self_attn.in_proj_weight)
            final = functional_call(layer, {"self_attn.in_proj_weight": weight},
                                    (h,), {"src_mask": mask})
            return candidate_head(torch.cat((norm(final), F.layer_norm(h,
                (model.config.d_model,), eps=1e-5)), dim=-1))

        def validation_ce():
            loss, count = 0.0, 0
            with torch.no_grad():
                for chunk in validation:
                    if time.perf_counter() >= work_deadline:
                        return None
                    _, y, active = chunk
                    if not active.any():
                        continue
                    loss += float(F.cross_entropy(logits(chunk)[active], y[active], reduction="sum"))
                    count += int(active.sum())
            return loss / count

        best_loss = validation_ce()
        if best_loss is None:
            return finish(warm, "lora_initial_validation_budget")
        best = [p.detach().clone() for p in params]
        report["lora_validation_history"].append(dict(step=0, cross_entropy=best_loss))
        optimization_started = time.perf_counter()
        for step in range(1, config.lora_steps+1):
            if time.perf_counter() >= work_deadline:
                break
            chunk = train[int(torch.randint(len(train), (), generator=generator))]
            _, y, active = chunk
            if not active.any():
                continue
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(logits(chunk)[active], y[active])
            loss += config.anchor_strength * sum((p-a).square().mean() for p, a in zip(params, anchors))
            if not torch.isfinite(loss):
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            report["lora_steps_completed"] += 1
            if step % 20 == 0 or step == config.lora_steps:
                value = validation_ce()
                if value is None:
                    break
                report["lora_validation_history"].append(dict(step=step, cross_entropy=value))
                if value < best_loss:
                    best_loss = value
                    best = [p.detach().clone() for p in params]
                    report["lora_selected_step"] = step
        report["lora_stage_seconds"]["optimization_and_selection"] = time.perf_counter()-optimization_started
        if report["lora_selected_step"] == 0:
            return finish(warm, "lora_no_selected_improvement")
        with torch.no_grad():
            for p, value in zip(params, best):
                p.copy_(value)
        adapter = serialize_lora(head, lora, base_sha256)
        report.update(candidate_parameter_bytes=adapter["parameter_bytes"],
                      candidate_metadata_bytes=len(json.dumps(adapter, sort_keys=True, separators=(",", ":")).encode()))
        values = validate_lora_adapter(adapter, model.config.d_model, base_sha256)
        wire_weight = canonical_weight(layer.self_attn.in_proj_weight, values, config.lora_rank).to(device)
        wire_head = _restore_head(adapter["head"], model.config.d_model, base_sha256,
                                  device=device, dtype=head.weight.dtype).eval()
        validation_started = time.perf_counter()
        bits, count = 0.0, 0
        with torch.no_grad():
            for chunk in validation:
                if time.perf_counter() >= deadline:
                    return finish(warm, "lora_wire_validation_budget")
                _, y, active = chunk
                scores = logits(chunk, wire_head, wire_weight)[active].cpu().numpy()
                targets = y[active].cpu().numpy()
                bits += selected_quantized_bits(targets, logits_to_cdfs(scores, total=total), total)
                count += targets.size
        report["lora_stage_seconds"]["wire_validation"] = time.perf_counter()-validation_started
        after = bits/count
        before = warm_report["after_validation_bits_per_quality"]
        report.update(lora_before_validation_bits_per_quality=before,
                      lora_candidate_validation_bits_per_quality=after)
        if time.perf_counter() >= deadline:
            return finish(warm, "lora_wire_validation_budget")
        if before-after <= config.min_gain_bits_per_quality:
            return finish(warm, "lora_no_quantized_gain")
        apply_lora_adapter(model, adapter, base_sha256)
        report.update(lora_accepted=True, after_validation_bits_per_quality=after,
                      parameter_bytes=adapter["parameter_bytes"],
                      metadata_bytes=len(json.dumps(adapter, sort_keys=True, separators=(",", ":")).encode()))
        return finish(adapter, "lora_validation_gain")
    finally:
        model.train(original_training)
