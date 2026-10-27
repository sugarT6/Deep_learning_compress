"""File-specific recurrent quality predictors and the strict adapter-v10 wire.

Public input embeddings/CNN remain external and SHA-bound. Every GRU, norm and
head parameter is transmitted. Explicit FP32 recurrence is shared by full and
incremental inference; cuDNN's sequence implementation is never used here.
"""
from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import zlib

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


PROTOCOL = "public_features_f16_f32_gru_rzn_cycle_contiguous_v1"
MAX_RAW_BYTES = 32 << 20


def feature_dim(config):
    from .model import POSITION_FEATURE_DIM, QUALITY_PREFIX_FEATURE_DIM, QMER_KS
    return (config.prev_q_embed_dim + len(QMER_KS) * config.qmer_embed_dim
            + config.base_context_dim + POSITION_FEATURE_DIM
            + (QUALITY_PREFIX_FEATURE_DIM if config.quality_prefix_features else 0))


def tensor_layout(inputs, width, layers):
    if (type(inputs) is not int or not 1 <= inputs <= 4096
            or type(width) is not int or not 1 <= width <= 512
            or type(layers) is not int or not 1 <= layers <= 4):
        raise ValueError("invalid recurrent quality dimensions")
    result = []
    for layer in range(layers):
        result.extend(((f"gru.weight_ih_l{layer}", (3 * width, inputs if layer == 0 else width)),
                       (f"gru.weight_hh_l{layer}", (3 * width, width)),
                       (f"gru.bias_ih_l{layer}", (3 * width,)),
                       (f"gru.bias_hh_l{layer}", (3 * width,))))
    result.extend((("norm.weight", (width,)), ("norm.bias", (width,)),
                   ("head.weight", (42, width)), ("head.bias", (42,))))
    size = 4 * sum(math.prod(shape) for _, shape in result)
    if size > MAX_RAW_BYTES:
        raise ValueError("recurrent quality parameters exceed size limit")
    return result


def _unpack(encoded, size, digest):
    if (not isinstance(encoded, str)
            or len(encoded) > 4 * ((size + 65536 + 2) // 3)):
        raise ValueError("recurrent quality payload length exceeds limit")
    try:
        compressed = base64.b64decode(encoded, validate=True)
        stream = zlib.decompressobj()
        planes = stream.decompress(compressed, size + 1)
    except (ValueError, UnicodeError, zlib.error) as exc:
        raise ValueError("invalid recurrent quality payload") from exc
    if (len(planes) != size or not stream.eof
            or stream.unused_data or stream.unconsumed_tail):
        raise ValueError("recurrent quality payload size or stream mismatch")
    raw = np.frombuffer(planes, dtype=np.uint8).reshape(4, -1).T.copy().tobytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("recurrent quality parameter SHA-256 mismatch")
    values = np.frombuffer(raw, dtype="<f4")
    if not np.isfinite(values).all():
        raise ValueError("recurrent quality parameters must be finite")
    return values


def validate_quality_gru(meta, base_sha256):
    from .head_adapter import ADAPTER_FORMAT
    keys = {"format", "version", "application", "protocol", "base_checkpoint_sha256",
            "feature_dim", "hidden_dim", "recurrent_layers", "dtype", "compression",
            "parameter_bytes", "sha256", "data_base64"}
    if (not isinstance(meta, dict) or set(meta) != keys
            or meta["format"] != ADAPTER_FORMAT or type(meta["version"]) is not int
            or meta["version"] != 10 or meta["application"] != "replace_quality_predictor"
            or meta["protocol"] != PROTOCOL or meta["dtype"] != "little_endian_float32"
            or meta["compression"] != "byteplanes_zlib"):
        raise ValueError("invalid recurrent quality adapter protocol")
    if meta["base_checkpoint_sha256"] != base_sha256:
        raise ValueError("recurrent quality base checkpoint SHA-256 mismatch")
    layout = tensor_layout(meta["feature_dim"], meta["hidden_dim"], meta["recurrent_layers"])
    size = 4 * sum(math.prod(shape) for _, shape in layout)
    if type(meta["parameter_bytes"]) is not int or meta["parameter_bytes"] != size:
        raise ValueError("recurrent quality parameter length mismatch")
    return _unpack(meta["data_base64"], size, meta["sha256"])


@dataclass
class QualityGRUState:
    owner: object
    bases: torch.Tensor
    lengths: torch.Tensor
    active_mask: torch.Tensor
    base_features: torch.Tensor
    position_features: torch.Tensor
    hidden: list
    prefix: torch.Tensor
    cycle: int = 0


class QualityGRUPredictor(nn.Module):
    def __init__(self, inputs, width, layers):
        super().__init__()
        tensor_layout(inputs, width, layers)
        # GRU is a parameter layout only. Both codec ends use advance(), below.
        self.gru = nn.GRU(inputs, width, num_layers=layers, batch_first=True, dropout=0)
        self.norm = nn.LayerNorm(width, eps=1e-5)
        self.head = nn.Linear(width, 42)

    def initial_hidden(self, batch):
        return [self.head.weight.new_zeros((batch, self.gru.hidden_size))
                for _ in range(self.gru.num_layers)]

    def advance(self, x, hidden):
        x = x.contiguous()
        updated = []
        for layer in range(self.gru.num_layers):
            old = hidden[layer].contiguous()
            gi = F.linear(x, getattr(self.gru, f"weight_ih_l{layer}"),
                          getattr(self.gru, f"bias_ih_l{layer}"))
            gh = F.linear(old, getattr(self.gru, f"weight_hh_l{layer}"),
                          getattr(self.gru, f"bias_hh_l{layer}"))
            ir, iz, inn = gi.chunk(3, -1)
            hr, hz, hn = gh.chunk(3, -1)
            reset = torch.sigmoid(ir + hr)
            update = torch.sigmoid(iz + hz)
            new = torch.tanh(inn + reset * hn)
            x = ((1 - update) * new + update * old).contiguous()
            updated.append(x)
        return self.head(self.norm(x)), updated

    def forward(self, features):
        if features.shape[1] == 0:
            return self.head.weight.new_empty((features.shape[0], 0, 42))
        features = features.to(torch.float16).to(torch.float32)
        hidden = self.initial_hidden(features.shape[0])
        result = []
        for cycle in range(features.shape[1]):
            logits, hidden = self.advance(features[:, cycle], hidden)
            result.append(logits)
        return torch.stack(result, 1)

    def begin_batch(self, model, bases, lengths, active_mask):
        if bases.shape[1] == 0:
            raise ValueError("an empty recurrent batch needs no state")
        return QualityGRUState(self, bases.clone(), lengths.clone(), active_mask.clone(),
            model.base_encoder(bases, active_mask), model._position_features(lengths, active_mask),
            self.initial_hidden(bases.shape[0]), bases.new_empty((bases.shape[0], 0)))

    def step(self, model, bases, prefix, lengths, active_mask, state):
        if (not isinstance(state, QualityGRUState) or state.owner is not self
                or state.cycle != prefix.shape[1]
                or not torch.equal(state.bases, bases)
                or not torch.equal(state.lengths, lengths)
                or not torch.equal(state.active_mask, active_mask)
                or (state.cycle and not torch.equal(state.prefix, prefix[:, :-1]))):
            raise ValueError("recurrent state requires the same batch and consecutive decoded prefix")
        qualities = torch.full_like(bases, 42)
        qualities[:, :state.cycle] = prefix
        features, _ = model._quality_input_features(bases, qualities, lengths, active_mask,
            base_features=state.base_features, position_features=state.position_features)
        x = features[:, state.cycle].to(torch.float16).to(torch.float32)
        logits, hidden = self.advance(x, state.hidden)
        state.hidden = hidden
        state.prefix = prefix.clone()
        state.cycle += 1
        return logits


def serialize_quality_gru(predictor, digest):
    from .head_adapter import ADAPTER_FORMAT
    inputs, width, layers = predictor.gru.input_size, predictor.gru.hidden_size, predictor.gru.num_layers
    layout = tensor_layout(inputs, width, layers)
    parameters = predictor.state_dict()
    raw = b"".join(parameters[name].detach().cpu().numpy().astype("<f4").tobytes()
                   for name, _ in layout)
    planes = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 4).T.copy().tobytes()
    meta = dict(format=ADAPTER_FORMAT, version=10, application="replace_quality_predictor",
                protocol=PROTOCOL, base_checkpoint_sha256=digest, feature_dim=inputs,
                hidden_dim=width, recurrent_layers=layers, dtype="little_endian_float32",
                compression="byteplanes_zlib", parameter_bytes=len(raw),
                sha256=hashlib.sha256(raw).hexdigest(),
                data_base64=base64.b64encode(zlib.compress(planes, 9)).decode("ascii"))
    validate_quality_gru(meta, digest)
    return meta


def apply_quality_gru(model, meta, digest):
    values = validate_quality_gru(meta, digest)
    if meta["feature_dim"] != feature_dim(model.config):
        raise ValueError("recurrent quality feature dimension does not match checkpoint")
    if model.output_head.weight.dtype != torch.float32:
        raise ValueError("recurrent quality inference requires FP32 public parameters")
    # Do not perturb the caller's RNG when allocating storage for wire values.
    with torch.random.fork_rng(devices=[]):
        predictor = QualityGRUPredictor(meta["feature_dim"], meta["hidden_dim"], meta["recurrent_layers"])
    parameters = predictor.state_dict()
    offset = 0
    for name, shape in tensor_layout(meta["feature_dim"], meta["hidden_dim"], meta["recurrent_layers"]):
        stop = offset + math.prod(shape)
        parameters[name].copy_(torch.from_numpy(values[offset:stop].copy().reshape(shape)))
        offset = stop
    predictor.to(device=model.output_head.weight.device).eval().requires_grad_(False)
    from .read_state import clear_conditioning
    clear_conditioning(model)
    model.quality_gru = predictor


def import_research_artifact(source, checkpoint, output):
    """Convert a checked research packet, without fitting or rounding weights."""
    from .checkpoint import load_training_checkpoint
    from .container import sha256_file
    source, checkpoint, output = map(Path, (source, checkpoint, output))
    if output.exists() or output.resolve() in {source.resolve(), checkpoint.resolve()}:
        raise FileExistsError("choose a new recurrent adapter output distinct from inputs")
    if source.stat().st_size > 48 << 20:
        raise ValueError("research recurrent packet exceeds size limit")
    meta = json.loads(source.read_text())
    digest = sha256_file(checkpoint)
    if (not isinstance(meta, dict) or meta.get("format") != "research-private-quality-gru-v1"
            or meta.get("mode") != "absolute" or meta.get("feature_rounding") != "FP16->FP32"
            or meta.get("public_feature_checkpoint_sha256") != digest):
        raise ValueError("invalid research recurrent predictor or checkpoint SHA-256")
    base = load_training_checkpoint(checkpoint).model
    layout = tensor_layout(feature_dim(base.config), meta.get("width"), meta.get("layers"))
    specs = meta.get("specs")
    if (not isinstance(specs, list) or len(specs) != len(layout)
            or any(not isinstance(s, dict) or set(s) != {"name", "shape"}
                   or not isinstance(s["name"], str) or not isinstance(s["shape"], list)
                   or any(type(v) is not int for v in s["shape"]) for s in specs)
            or len({s["name"] for s in specs}) != len(layout)
            or {s["name"]: s["shape"] for s in specs} != {n: list(shape) for n, shape in layout}
            ):
        raise ValueError("research recurrent tensor layout mismatch")
    size = 4 * sum(math.prod(shape) for _, shape in layout)
    values = _unpack(meta.get("data_base64"), size, meta.get("raw_sha256"))
    with torch.random.fork_rng(devices=[]):
        predictor = QualityGRUPredictor(feature_dim(base.config), meta["width"], meta["layers"])
    offset = 0
    for item in specs:
        count = math.prod(item["shape"])
        predictor.state_dict()[item["name"]].copy_(torch.from_numpy(
            values[offset:offset + count].copy().reshape(item["shape"])))
        offset += count
    adapter = serialize_quality_gru(predictor, digest)
    # Applying also verifies the actual feature contract before exporting.
    apply_quality_gru(base, adapter, digest)
    artifact = dict(format="direct-quality-head-adaptation-artifact", version=1,
                    base_checkpoint_sha256=digest, adapter=adapter, source=None,
                    report=dict(reason="imported_frozen_quality_gru", research_packet_sha256=sha256_file(source),
                                parameter_bytes=size, weights_changed=False))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(artifact, handle, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return dict(output=str(output.resolve()), adapter_version=10, parameter_bytes=size,
                base_checkpoint_sha256=digest, weights_changed=False)


def main():
    parser = argparse.ArgumentParser(description="Import a frozen research GRU as a codec adapter")
    parser.add_argument("source", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        report = import_research_artifact(args.source, args.checkpoint, args.output)
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(report, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
