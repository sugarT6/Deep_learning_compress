"""Transmitted read-mean labels and state-conditioned Transformer blocks.

Only the encoder computes a state from a complete quality string. The decoder
reads the one- or two-bit labels before predicting Q; it never infers future qualities.
"""

import base64
import hashlib
import json

import numpy as np
import torch
from torch import nn


STATE_PROTOCOL = "mean_milli_four_class_film_suffix_v1"
LABEL_PROTOCOL = "packed_2bit_msb_first_v1"
TWO_STATE_PROTOCOL = "mean_milli_two_class_film_suffix_v1"
TWO_LABEL_PROTOCOL = "packed_1bit_msb_first_v1"


def bits_per_label(class_count):
    if type(class_count) is not int or class_count not in (2, 4):
        raise ValueError("read-state class_count must be 2 or 4")
    return 1 if class_count == 2 else 2


def validate_thresholds(thresholds, class_count=None):
    if class_count is None:
        class_count = len(thresholds) + 1 if isinstance(thresholds, list) else 0
    bits_per_label(class_count)
    if (not isinstance(thresholds, list) or len(thresholds) != class_count - 1
            or any(type(v) is not int or not 0 <= v <= 41000 for v in thresholds)
            or thresholds != sorted(thresholds)):
        raise ValueError("read-state thresholds must match class_count and contain sorted integer milli-Q values")


def mean_milli(qualities, lengths):
    """Integer means, ignoring padding; empty reads have canonical state zero."""
    q = np.asarray(qualities, dtype=np.int64)
    n = np.asarray(lengths, dtype=np.int64)
    active = np.arange(q.shape[1])[None, :] < n[:, None]
    return np.where(active, q, 0).sum(axis=1) * 1000 // np.maximum(n, 1)


def fit_thresholds(records, class_count=4):
    bits_per_label(class_count)
    means = sorted(sum(v - 33 for v in r.quality) * 1000 // len(r.quality)
                   for r in records if r.quality)
    if not means:
        raise ValueError("read-state fitting needs nonempty training reads")
    # Nearest-rank quantiles use only the adaptation training split. Repeated
    # thresholds are valid for constant/small-alphabet files; classes may be empty.
    return [means[(len(means) * k + class_count - 1) // class_count - 1]
            for k in range(1, class_count)]


def classify_reads(qualities, lengths, thresholds):
    validate_thresholds(thresholds)
    result = np.searchsorted(thresholds, mean_milli(qualities, lengths), side="right")
    return np.where(np.asarray(lengths) > 0, result, 0).astype(np.int64)


class ReadStateConditioning(nn.Module):
    """Per-state affine conditioning at the input of every adapted block."""

    def __init__(self, dim, layers, thresholds):
        super().__init__()
        validate_thresholds(thresholds)
        self.layers = tuple(layers)
        self.thresholds = list(thresholds)
        self.class_count = len(thresholds) + 1
        self.scale = nn.Parameter(torch.zeros(len(layers), self.class_count, dim))
        self.shift = nn.Parameter(torch.zeros(len(layers), self.class_count, dim))

    def forward(self, hidden, states, layer_index):
        index = self.layers.index(layer_index)
        return hidden * (1.0 + self.scale[index, states].unsqueeze(1)) + self.shift[index, states].unsqueeze(1)


def clear_conditioning(model):
    if hasattr(model, "read_state_conditioning"):
        del model.read_state_conditioning


def conditioned_tensors(model, tensors):
    """Training/encoder helper; never used to derive decoder inputs."""
    if not hasattr(model, "read_state_conditioning"):
        return tensors
    q, lengths = tensors["qualities"], tensors["lengths"]
    active = tensors["active_mask"]
    means = torch.div(q.masked_fill(~active, 0).sum(1) * 1000,
                      lengths.clamp(min=1), rounding_mode="floor")
    thresholds = torch.tensor(model.read_state_conditioning.thresholds, device=q.device)
    states = (means[:, None] >= thresholds[None, :]).sum(1)
    states = states.masked_fill(lengths == 0, 0)
    return dict(tensors, read_states=states)


def serialize_read_state(model, backbone, digest):
    from .head_adapter import ADAPTER_FORMAT
    state = model.read_state_conditioning
    raw = b"".join(p.detach().cpu().numpy().astype("<f4").tobytes()
                   for p in (state.scale, state.shift))
    meta = dict(format=ADAPTER_FORMAT, version=9,
                application="replace_suffix_and_read_state_conditioning",
                protocol=(TWO_STATE_PROTOCOL if state.class_count == 2 else STATE_PROTOCOL),
                base_checkpoint_sha256=digest,
                backbone=backbone, num_layers=model.config.num_layers,
                feedforward_dim=model.config.feedforward_dim, layers=list(state.layers),
                thresholds_milli=state.thresholds, class_count=state.class_count,
                dtype="little_endian_float32", layout="scale_then_shift_layer_state_hidden",
                raw_bytes=len(raw), parameter_bytes=backbone["parameter_bytes"] + len(raw),
                sha256=hashlib.sha256(raw).hexdigest(),
                data_base64=base64.b64encode(raw).decode("ascii"))
    validate_read_state_adapter(meta, model.config.d_model, digest)
    return meta


def validate_read_state_adapter(meta, dim, digest):
    from .head_adapter import ADAPTER_FORMAT, validate_head_adapter
    keys = {"format", "version", "application", "protocol", "base_checkpoint_sha256",
            "backbone", "num_layers", "feedforward_dim", "layers", "thresholds_milli",
            "class_count", "dtype", "layout", "raw_bytes", "parameter_bytes", "sha256", "data_base64"}
    if (not isinstance(meta, dict) or set(meta) != keys or meta["format"] != ADAPTER_FORMAT
            or type(meta["version"]) is not int or meta["version"] != 9
            or meta["application"] != "replace_suffix_and_read_state_conditioning"
            or meta["base_checkpoint_sha256"] != digest
            or type(meta["class_count"]) is not int or meta["class_count"] not in (2, 4)
            or meta["protocol"] != (TWO_STATE_PROTOCOL if meta["class_count"] == 2 else STATE_PROTOCOL)
            or meta["dtype"] != "little_endian_float32"
            or meta["layout"] != "scale_then_shift_layer_state_hidden"):
        raise ValueError("invalid read-state adapter protocol")
    nested = meta["backbone"]
    if (not isinstance(nested, dict) or type(nested.get("version")) is not int
            or nested["version"] not in (7, 8)):
        raise ValueError("read-state adapter requires a nonrecursive dense/LoRA suffix")
    validate_head_adapter(nested, dim, digest)
    for field in ("num_layers", "feedforward_dim", "layers"):
        if json.dumps(meta[field]) != json.dumps(nested[field]):
            raise ValueError("read-state and backbone architecture mismatch")
    validate_thresholds(meta["thresholds_milli"], meta["class_count"])
    size = len(meta["layers"]) * meta["class_count"] * dim * 2 * 4
    encoded = meta["data_base64"]
    if (type(meta["raw_bytes"]) is not int or meta["raw_bytes"] != size
            or type(meta["parameter_bytes"]) is not int
            or meta["parameter_bytes"] != size + nested["parameter_bytes"]
            or not isinstance(encoded, str) or len(encoded) != 4 * ((size + 2) // 3)):
        raise ValueError("read-state parameter length mismatch")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, UnicodeError) as exc:
        raise ValueError("invalid read-state base64") from exc
    if len(raw) != size or hashlib.sha256(raw).hexdigest() != meta["sha256"]:
        raise ValueError("read-state parameter SHA-256 mismatch")
    values = np.frombuffer(raw, dtype="<f4")
    if not np.isfinite(values).all():
        raise ValueError("read-state parameters must be finite")
    return values


def apply_read_state_adapter(model, meta, digest):
    from .head_adapter import apply_head_adapter
    values = validate_read_state_adapter(meta, model.config.d_model, digest)
    if (meta["num_layers"] != model.config.num_layers
            or meta["feedforward_dim"] != model.config.feedforward_dim):
        raise ValueError("read-state adapter model configuration mismatch")
    state = ReadStateConditioning(model.config.d_model, meta["layers"], meta["thresholds_milli"])
    offset = 0
    with torch.no_grad():
        for parameter in (state.scale, state.shift):
            stop = offset + parameter.numel()
            parameter.copy_(torch.from_numpy(values[offset:stop].copy().reshape(parameter.shape)))
            offset = stop
    state.to(device=model.output_head.weight.device, dtype=model.output_head.weight.dtype)
    state.train(model.training)
    state.requires_grad_(model.output_head.weight.requires_grad)
    # Nested application validates/merges before changing the live model, and
    # clears a previous conditioning module when switching adapter types.
    apply_head_adapter(model, meta["backbone"], digest)
    model.read_state_conditioning = state


class StateWriter:
    """Continuous fixed-width packing across batches, with zero low-bit padding."""

    def __init__(self, handle, class_count=4):
        self.label_bits = bits_per_label(class_count)
        self.class_count = class_count
        self.handle, self.count, self.pending, self.bits = handle, 0, 0, 0
        self.class_counts = np.zeros(class_count, dtype=np.int64)
        self.finished = False

    def write(self, states):
        if self.finished:
            raise ValueError("cannot write a finished read-state stream")
        states = np.asarray(states)
        if (states.ndim != 1 or states.dtype.kind not in "iu"
                or np.any((states < 0) | (states >= self.class_count))):
            raise ValueError("invalid read-state labels")
        out = bytearray()
        for state in states:
            self.pending = (self.pending << self.label_bits) | int(state)
            self.bits += self.label_bits
            if self.bits == 8:
                out.append(self.pending)
                self.pending = self.bits = 0
        self.handle.write(out)
        self.count += len(states)
        self.class_counts += np.bincount(states.astype(np.int64), minlength=self.class_count)

    def finish(self):
        if self.bits:
            self.handle.write(bytes([self.pending << (8 - self.bits)]))
            self.pending = self.bits = 0
        self.finished = True


class StateReader:
    def __init__(self, handle, count, class_count=4):
        self.label_bits = bits_per_label(class_count)
        self.mask = class_count - 1
        if type(count) is not int or count < 0:
            raise ValueError("read-state label count must be a nonnegative integer")
        self.handle, self.remaining, self.pending, self.bits = handle, count, 0, 0

    def read(self, count):
        if type(count) is not int or count > self.remaining or count < 0:
            raise ValueError("read-state label count mismatch")
        out = np.empty(count, dtype=np.int64)
        for i in range(count):
            if self.bits == 0:
                value = self.handle.read(1)
                if not value:
                    raise ValueError("truncated read-state stream")
                self.pending, self.bits = value[0], 8
            self.bits -= self.label_bits
            out[i] = (self.pending >> self.bits) & self.mask
        self.remaining -= count
        return out

    def finish(self):
        if self.remaining or (self.bits and self.pending & ((1 << self.bits) - 1)) or self.handle.read(1):
            raise ValueError("read-state stream has missing labels, nonzero padding, or trailing bytes")


def label_metadata(adapter, read_count):
    class_count = adapter["class_count"]
    label_bits = bits_per_label(class_count)
    validate_thresholds(adapter["thresholds_milli"], class_count)
    if adapter.get("protocol") != (TWO_STATE_PROTOCOL if class_count == 2 else STATE_PROTOCOL):
        raise ValueError("read-state adapter protocol and class_count mismatch")
    if type(read_count) is not int or read_count < 0:
        raise ValueError("read-state label count must be a nonnegative integer")
    return dict(version=1, protocol=(TWO_LABEL_PROTOCOL if class_count == 2 else LABEL_PROTOCOL),
                class_count=class_count, bits_per_read=label_bits,
                label_count=read_count, valid_bits=label_bits * read_count,
                thresholds_milli=adapter["thresholds_milli"], empty_read_state=0)


def validate_label_metadata(metadata):
    state = metadata.get("read_state")
    adapter = metadata.get("head_adapter")
    if not isinstance(adapter, dict) or type(adapter.get("version")) is not int or adapter["version"] != 9:
        raise ValueError("read-state container requires adapter v9")
    expected = label_metadata(adapter, metadata["read_count"])
    if json.dumps(state, sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise ValueError("invalid read-state label metadata")


def state_container_overhead(info):
    """All additional label bytes and JSON framing, alongside adapter cost."""
    if "read_state" not in info.metadata:
        return 0, 0
    section = next(s for s in info.metadata["sections"] if s["name"] == "read_state")
    member = len(json.dumps({"read_state": info.metadata["read_state"]}, sort_keys=True, separators=(",", ":")).encode()) - 1
    descriptor = len(json.dumps(section, sort_keys=True, separators=(",", ":")).encode()) + 1
    return info.section("read_state").length, member + descriptor
