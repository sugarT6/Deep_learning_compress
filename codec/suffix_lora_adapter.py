"""Multi-block Q/K/V/O + FFN LoRA with deterministic wire reconstruction."""

import base64
import hashlib
import json
import zlib

import numpy as np
import torch
from torch import nn
from torch.nn.utils import parametrize

from .backbone_adapter import MAX_RAW_BYTES, _base_bytes, restore_backbone


PROTOCOL = "transformer_suffix_qkvo_ffn_rank_major_f64_merge_v1"
SWIGLU_PROTOCOL = "transformer_suffix_qkvo_swiglu_rank_major_f64_merge_v1"


def matrix_layout(dim, ff, layers, ffn_type="gelu"):
    if ffn_type not in ("gelu", "swiglu"):
        raise ValueError("unsupported suffix LoRA FFN type")
    matrices = (("self_attn.in_proj_weight", [3 * dim, dim], 3),
                ("self_attn.out_proj.weight", [dim, dim], 1),
                ("linear1.weight", [ff, dim], 1),
                ("linear2.weight", [dim, ff], 1))
    if ffn_type == "swiglu":
        matrices += (("linear_gate.weight", [ff, dim], 1),)
    return [dict(name=f"transformer.layers.{i}.{name}", shape=shape, parts=parts)
            for i in layers for name, shape, parts in matrices]


class LowRankWeight(nn.Module):
    """Independent rank-r updates for each Q/K/V slice; scaling is exactly 1."""

    def __init__(self, shape, parts, rank, seed):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.a = nn.Parameter(torch.randn(parts, rank, shape[1]) * .02)
            self.b = nn.Parameter(torch.zeros(parts, shape[0] // parts, rank))

    def forward(self, weight):
        return weight + torch.bmm(self.b, self.a).flatten(0, 1)


def install_suffix_lora(model, layer_count, rank, seed):
    """Training-only parametrizations; original matrices remain frozen."""
    cfg = model.config
    if (type(layer_count) is not int or not 1 <= layer_count <= min(4, cfg.num_layers)
            or type(rank) is not int or not 1 <= rank <= 16):
        raise ValueError("invalid suffix LoRA layers or rank")
    layers = list(range(cfg.num_layers - layer_count, cfg.num_layers))
    layout = matrix_layout(cfg.d_model, cfg.feedforward_dim, layers, getattr(cfg, "ffn_type", "gelu"))
    for index, item in enumerate(layout):
        module_name, name = item["name"].rsplit(".", 1)
        module = model.get_submodule(module_name)
        if parametrize.is_parametrized(module, name):
            raise ValueError("suffix LoRA requires unparametrized matrices")
        weight = getattr(module, name)
        weight.requires_grad_(False)
        factors = LowRankWeight(item["shape"], item["parts"], rank, seed + index).to(
            device=weight.device, dtype=weight.dtype)
        parametrize.register_parametrization(module, name, factors)
    return layout


def _state_shapes(layout, dim):
    return [dict(name=item["name"], shape=item["shape"]) for item in layout] + [
        dict(name="transformer.norm.weight", shape=[dim]),
        dict(name="transformer.norm.bias", shape=[dim])]


def _canonical_parameters(model, layout, values, rank):
    """Merge on CPU in fixed FP64 rank order, round once, then validate FP32."""
    dim = model.config.d_model
    shapes = _state_shapes(layout, dim)
    base = np.frombuffer(_base_bytes(model, shapes), dtype="<f4")
    result, offset, base_offset = {}, 0, 0
    for item in layout:
        rows, cols = item["shape"]
        parts, part_rows = item["parts"], rows // item["parts"]
        a_size, b_size = parts * rank * cols, parts * part_rows * rank
        a = values[offset:offset + a_size].reshape(parts, rank, cols).astype(np.float64)
        offset += a_size
        b = values[offset:offset + b_size].reshape(parts, part_rows, rank).astype(np.float64)
        offset += b_size
        weight = base[base_offset:base_offset + rows * cols].reshape(parts, part_rows, cols).astype(np.float64)
        base_offset += rows * cols
        for part in range(parts):
            delta = np.zeros((part_rows, cols), dtype=np.float64)
            for k in range(rank):
                delta += b[part, :, k:k + 1] * a[part, k:k + 1, :]
            weight[part] += delta
        with np.errstate(over="ignore", invalid="ignore"):
            result[item["name"]] = weight.reshape(rows, cols).astype("<f4")
    for name in ("transformer.norm.weight", "transformer.norm.bias"):
        result[name] = values[offset:offset + dim].copy()
        offset += dim
    if offset != len(values) or any(not np.isfinite(v).all() for v in result.values()):
        raise ValueError("merged suffix LoRA parameters must be finite and complete")
    return result


def _merged_digest(parameters):
    return hashlib.sha256(b"".join(v.tobytes() for v in parameters.values())).hexdigest()


def serialize_suffix_lora(model, layer_count, rank, digest, *, base_model):
    from .head_adapter import ADAPTER_FORMAT, serialize_head
    cfg = model.config
    if (type(layer_count) is not int or not 1 <= layer_count <= min(4, cfg.num_layers)
            or type(rank) is not int or not 1 <= rank <= 16):
        raise ValueError("invalid suffix LoRA layers or rank")
    if cfg.to_dict() != base_model.config.to_dict():
        raise ValueError("base and adapted model configurations must match")
    layers = list(range(cfg.num_layers - layer_count, cfg.num_layers))
    ffn_type = getattr(cfg, "ffn_type", "gelu")
    layout = matrix_layout(cfg.d_model, cfg.feedforward_dim, layers, ffn_type)
    arrays = []
    for item in layout:
        module_name, name = item["name"].rsplit(".", 1)
        factors = model.get_submodule(module_name).parametrizations[name][0]
        arrays.extend((factors.a, factors.b))
    arrays.extend((model.transformer.norm.weight, model.transformer.norm.bias))
    raw = b"".join(p.detach().cpu().numpy().astype("<f4").tobytes() for p in arrays)
    values = np.frombuffer(raw, dtype="<f4")
    if not np.isfinite(values).all():
        raise ValueError("suffix LoRA parameters must be finite")
    merged = _canonical_parameters(base_model, layout, values, rank)
    head = serialize_head(model.output_head, digest)
    planes = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 4).T.copy().tobytes()
    meta = dict(format=ADAPTER_FORMAT, version=8, application="replace_head_and_suffix_lora",
                protocol=SWIGLU_PROTOCOL if ffn_type == "swiglu" else PROTOCOL,
                base_checkpoint_sha256=digest, head=head,
                num_layers=cfg.num_layers, feedforward_dim=cfg.feedforward_dim,
                layers=layers, rank=rank, scaling=1, matrices=layout,
                dtype="little_endian_float32", compression="byteplanes_zlib",
                raw_bytes=len(raw), parameter_bytes=len(raw) + head["parameter_bytes"],
                sha256=hashlib.sha256(raw).hexdigest(), merged_sha256=_merged_digest(merged),
                data_base64=base64.b64encode(zlib.compress(planes, level=6)).decode("ascii"))
    if ffn_type == "swiglu":
        meta["ffn_type"] = ffn_type
    validate_suffix_lora(meta, cfg.d_model, digest)
    return meta


def validate_suffix_lora(meta, dim, digest):
    from .head_adapter import ADAPTER_FORMAT, validate_head_adapter
    keys = {"format", "version", "application", "protocol", "base_checkpoint_sha256", "head",
            "num_layers", "feedforward_dim", "layers", "rank", "scaling", "matrices", "dtype",
            "compression", "raw_bytes", "parameter_bytes", "sha256", "merged_sha256", "data_base64"}
    swiglu = isinstance(meta, dict) and meta.get("protocol") == SWIGLU_PROTOCOL
    if swiglu:
        keys = keys | {"ffn_type"}
    if (not isinstance(meta, dict) or set(meta) != keys or meta["format"] != ADAPTER_FORMAT
            or type(meta["version"]) is not int or meta["version"] != 8
            or meta["application"] != "replace_head_and_suffix_lora"
            or meta["protocol"] != (SWIGLU_PROTOCOL if swiglu else PROTOCOL)
            or (swiglu and meta["ffn_type"] != "swiglu")
            or meta["base_checkpoint_sha256"] != digest
            or meta["dtype"] != "little_endian_float32" or meta["compression"] != "byteplanes_zlib"
            or type(meta["scaling"]) is not int or meta["scaling"] != 1):
        raise ValueError("invalid suffix LoRA protocol")
    depth, ff, layers, rank = meta["num_layers"], meta["feedforward_dim"], meta["layers"], meta["rank"]
    if (type(dim) is not int or not 1 <= dim <= 65536
            or type(depth) is not int or not 1 <= depth <= 64
            or type(ff) is not int or not 1 <= ff <= 65536
            or type(rank) is not int or not 1 <= rank <= 16
            or not isinstance(layers, list) or not 1 <= len(layers) <= min(4, depth)
            or any(type(i) is not int for i in layers)
            or layers != list(range(depth - len(layers), depth))):
        raise ValueError("invalid suffix LoRA dimensions")
    head = meta["head"]
    if (not isinstance(head, dict) or type(head.get("version")) is not int
            or head["version"] not in (1, 2, 4) or (head["version"] == 4 and depth < 4)):
        raise ValueError("suffix LoRA requires a nonrecursive head compatible with model depth")
    validate_head_adapter(head, dim, digest)
    layout = matrix_layout(dim, ff, layers, "swiglu" if swiglu else "gelu")
    if json.dumps(meta["matrices"], sort_keys=True) != json.dumps(layout, sort_keys=True):
        raise ValueError("suffix LoRA matrix layout mismatch")
    size = 4 * (sum(rank * (i["shape"][0] + i["parts"] * i["shape"][1])
                    for i in layout) + 2 * dim)
    if (size > MAX_RAW_BYTES or type(meta["raw_bytes"]) is not int or meta["raw_bytes"] != size
            or type(meta["parameter_bytes"]) is not int
            or meta["parameter_bytes"] != size + head["parameter_bytes"]
            or not isinstance(meta["data_base64"], str)
            or len(meta["data_base64"]) > 4 * ((size + (1 << 16) + 2) // 3)):
        raise ValueError("invalid suffix LoRA payload length")
    try:
        stream = zlib.decompressobj()
        planes = stream.decompress(base64.b64decode(meta["data_base64"], validate=True), size + 1)
    except (ValueError, UnicodeError, zlib.error) as exc:
        raise ValueError("invalid compressed suffix LoRA payload") from exc
    if len(planes) != size or not stream.eof or stream.unused_data or stream.unconsumed_tail:
        raise ValueError("suffix LoRA payload length mismatch")
    raw = np.frombuffer(planes, dtype=np.uint8).reshape(4, -1).T.copy().tobytes()
    if hashlib.sha256(raw).hexdigest() != meta["sha256"]:
        raise ValueError("suffix LoRA SHA-256 mismatch")
    if (not isinstance(meta["merged_sha256"], str) or len(meta["merged_sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in meta["merged_sha256"])):
        raise ValueError("invalid merged suffix LoRA SHA-256")
    values = np.frombuffer(raw, dtype="<f4")
    if not np.isfinite(values).all():
        raise ValueError("suffix LoRA parameters must be finite")
    return values


def apply_suffix_lora(model, meta, digest):
    from .head_adapter import _restore_head
    values = validate_suffix_lora(meta, model.config.d_model, digest)
    if (meta["num_layers"] != model.config.num_layers
            or meta["feedforward_dim"] != model.config.feedforward_dim
            or meta.get("ffn_type", "gelu") != getattr(model.config, "ffn_type", "gelu")):
        raise ValueError("suffix LoRA model configuration mismatch")
    merged = _canonical_parameters(model, meta["matrices"], values, meta["rank"])
    if _merged_digest(merged) != meta["merged_sha256"]:
        raise ValueError("merged suffix LoRA SHA-256 mismatch")
    original = model.output_head
    head = _restore_head(meta["head"], model.config.d_model, digest,
                         device=original.weight.device, dtype=original.weight.dtype)
    head.train(original.training)
    head.requires_grad_(original.weight.requires_grad)
    head.bias.requires_grad_(original.bias.requires_grad)
    from .read_state import clear_conditioning
    clear_conditioning(model)
    restore_backbone(model)
    if hasattr(model, "_lora_original_qkv"):
        with torch.no_grad():
            model.transformer.layers[3].self_attn.in_proj_weight.copy_(model._lora_original_qkv)
        del model._lora_original_qkv
    parameters = dict(model.named_parameters())
    model._backbone_original_parameters = {
        name: parameters[name].detach().cpu().clone() for name in merged}
    with torch.no_grad():
        for name, value in merged.items():
            parameters[name].copy_(torch.from_numpy(value.copy()))
    model.output_head = head
