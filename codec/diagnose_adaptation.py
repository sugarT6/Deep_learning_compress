"""Compare existing suffix LoRA with full fine-tuning on one fixed read split.

This diagnostic starts from a pretrained checkpoint. It scores probability
models without encoding the complete input or claiming an archive byte size.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

from .adaptation_diagnostic import fit_branch, score_records
from .backbone_adapter import sample_records
from .checkpoint import load_training_checkpoint
from .container import sha256_file
from .evaluate import choose_device
from .head_adapter import HeadAdaptationConfig, _new_head, adapt_output_head
from .probability_quantization import validate_total
from .train import _write_json_atomic


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "codec/output/adaptation_diagnostic_20261021"


@dataclass(frozen=True)
class DiagnosticConfig:
    steps: int = 10000
    eval_every: int = 200
    batch_reads: int = 16
    eval_batch_reads: int = 256
    lora_lr: float = 0.0003
    full_lr: float = 0.0001
    head_lr: float = 0.001
    anchor_strength: float = 0.01
    seed: int = 20260921
    sample_reads: int = 32000
    sample_symbols: int = 4800000
    train_probe_reads: int = 2048
    warmup_reads: int = 8000
    warmup_symbols: int = 1200000
    warmup_steps: int = 2000
    warmup_symbols_per_step: int = 8192
    warmup_anchor_strength: float = 0.01
    max_read_length: int = 2048
    residual_dim: int = 64
    cross_dim: int = 16
    lora_layers: int = 2
    lora_rank: int = 8
    patience: int = 6
    quantization_total: int = 65536

    def __post_init__(self):
        for name in (
            "steps", "eval_every", "batch_reads", "eval_batch_reads", "sample_reads",
            "sample_symbols", "train_probe_reads", "warmup_reads", "warmup_symbols",
            "warmup_steps", "warmup_symbols_per_step", "max_read_length",
            "residual_dim", "cross_dim", "lora_layers", "lora_rank", "quantization_total",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("seed", "patience"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in ("lora_lr", "full_lr", "head_lr", "anchor_strength", "warmup_anchor_strength"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0
                    or (name.endswith("lr") and value == 0)):
                raise ValueError(f"invalid {name}")
        if max(self.batch_reads, self.eval_batch_reads) > 256:
            raise ValueError("read batch sizes must not exceed 256")
        if self.sample_reads < 4 or self.warmup_reads < 4:
            raise ValueError("sampling and warm-up need at least four reads")
        if self.lora_layers > 4 or self.lora_rank > 16:
            raise ValueError("LoRA supports at most four suffix layers and rank sixteen")
        if max(self.residual_dim, self.cross_dim) > 128:
            raise ValueError("output-head dimensions must not exceed 128")
        validate_total(self.quantization_total)

    def head_config(self) -> HeadAdaptationConfig:
        return HeadAdaptationConfig(
            max_reads=self.warmup_reads, max_symbols=self.warmup_symbols,
            max_read_length=self.max_read_length, steps=self.warmup_steps,
            symbols_per_step=self.warmup_symbols_per_step, learning_rate=self.head_lr,
            anchor_strength=self.warmup_anchor_strength, max_seconds=0,
            seed=self.seed, head_type="residual", residual_dim=self.residual_dim,
            cross_layer=True, cross_dim=self.cross_dim, quantization_version=2,
            backbone_max_reads=self.sample_reads, backbone_max_symbols=self.sample_symbols,
        )


def _indices_digest(records) -> str:
    return hashlib.sha256(np.asarray([r.read_index for r in records], dtype="<i8").tobytes()).hexdigest()


def _content_digest(records) -> str:
    """Hash only numerical model inputs and targets, with explicit boundaries."""
    digest = hashlib.sha256()
    for record in records:
        digest.update(np.asarray([record.read_index, len(record.sequence)], dtype="<i8").tobytes())
        digest.update(record.sequence)
        digest.update(record.quality)
    return digest.hexdigest()


def _model_digest(model) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def run_diagnostic(
    source: Path,
    checkpoint: Path,
    *,
    output_dir: Path,
    mode: str = "both",
    device: torch.device = torch.device("cpu"),
    config: DiagnosticConfig = DiagnosticConfig(),
    progress: bool = True,
    sample_cache: Optional[Path] = None,
):
    if mode not in ("both", "lora", "full"):
        raise ValueError("mode must be both, lora or full")
    device = torch.device(device)
    source, checkpoint, output_dir = Path(source).resolve(), Path(checkpoint).resolve(), Path(output_dir)
    if not source.is_file():
        raise FileNotFoundError(source)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"{output_dir}: choose a new empty diagnostic directory")
    cached_records, cache_manifest = None, None
    if sample_cache is not None:
        from .adaptation_sample import load_sample
        cached_records, cache_manifest = load_sample(source, sample_cache, config)
    loaded = load_training_checkpoint(checkpoint, device=device)
    model = loaded.model.eval()
    if config.lora_layers > model.config.num_layers:
        raise ValueError("LoRA suffix exceeds the pretrained model depth")
    head_config = config.head_config()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_sha = sha256_file(checkpoint)
    stat = source.stat()
    source_identity = {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    base_metrics = None
    if cached_records is not None:
        base_metrics = score_records(model, cached_records[len(cached_records) * 3 // 4:],
            device=device, batch_reads=config.eval_batch_reads, source=source,
            total=config.quantization_total, version=2, quantized=True)
        _write_json_atomic(output_dir / "base_validation_metrics.json", base_metrics)
    if progress:
        print(f"Diagnostic directory: {output_dir.resolve()}", file=sys.stderr, flush=True)
        print("Stage 1/3: fit the shared output head from the pretrained model", file=sys.stderr, flush=True)
    warm_adapter, warmup = adapt_output_head(
        model, source, device=device, batch_reads=config.eval_batch_reads,
        total=config.quantization_total, base_sha256=checkpoint_sha, config=head_config,
    )
    if warm_adapter is None:
        # Match the existing suffix adaptation path when prefix fitting does
        # not pass its gain test: both branches still get the same zero-start
        # residual/cross head rather than unequal architectural capacity.
        original = model.output_head
        model.output_head = _new_head(
            model.config.d_model, config.residual_dim, cross_dim=config.cross_dim,
            device=device, dtype=original.weight.dtype, seed=config.seed,
        )
        with torch.no_grad():
            model.output_head.weight.copy_(original.weight)
            model.output_head.bias.copy_(original.bias)
    model.eval()
    if progress:
        print(f"Shared head: {warmup['reason']}; prefix_reads={warmup['prefix_reads']}",
              file=sys.stderr, flush=True)
        print("Stage 2/3: " + ("reuse the frozen reservoir sample" if cached_records is not None
              else "scan the FASTQ once for the fixed reservoir sample"), file=sys.stderr, flush=True)
    if cached_records is None:
        records, sample = sample_records(source, head_config, warmup["prefix_reads"], float("inf"))
    else:
        if warmup["prefix_reads"] != cache_manifest["prefix_reads"]:
            raise ValueError("warm-up prefix does not match frozen sample exclusion")
        records, sample = cached_records, cache_manifest["sample"]
    after = source.stat()
    if source_identity != {"size_bytes": after.st_size, "mtime_ns": after.st_mtime_ns}:
        raise ValueError("source file changed while preparing diagnostic reads")
    if not sample["sampling_complete"] or len(records) < 4:
        raise ValueError("diagnostic sampling needs at least four eligible nonempty reads")
    split = len(records) * 3 // 4
    train, validation = records[:split], records[split:]
    # The probe is a fixed, representative subset of training reads; its seed
    # is independent of the minibatch generator and never advances that RNG.
    probe_positions = sorted(random.Random(config.seed + 1).sample(
        range(len(train)), min(config.train_probe_reads, len(train))))
    probe = [train[index] for index in probe_positions]
    manifest = {
        "format": "direct-quality-adaptation-diagnostic", "version": 1,
        "source": str(source), "source_identity": source_identity,
        "base_checkpoint": str(checkpoint), "base_checkpoint_sha256": checkpoint_sha,
        "model_config": model.config.to_dict(), "config": asdict(config), "device": str(device),
        "sample": sample, "warmup": warmup,
        "warm_model_sha256": _model_digest(model),
        "split": {
            "rule": "same_uniform_reservoir_shuffled_75_percent_train_25_percent_validation",
            "train_reads": len(train), "validation_reads": len(validation),
            "train_probe_reads": len(probe),
            "train_quality_symbols": sum(len(r.quality) for r in train),
            "validation_quality_symbols": sum(len(r.quality) for r in validation),
            "train_indices_sha256": _indices_digest(train),
            "validation_indices_sha256": _indices_digest(validation),
            "train_probe_indices_sha256": _indices_digest(probe),
            "sample_content_sha256": _content_digest(records),
            "indices_file": "split_indices.npz",
        },
        "scope": "probability_diagnostic_only_no_complete_file_compression",
        "selection": "best_positive_step_by_float_validation_bpq",
        "joint_regularization": "anchor_strength_times_sum_of_per_tensor_mean_squared_parameter_displacements",
        "comparison_limits": [
            "full_and_lora_use_different_default_backbone_learning_rates",
            "the_same_anchor_coefficient_is_not_the_same_prior_under_different_parameterizations",
            "full_parameter_state_storage_cost_is_not_included_in_bpq",
            "a_negative_result_does_not_establish_an_entropy_or_capacity_limit",
        ],
    }
    if cache_manifest is not None:
        manifest["sample_cache"] = {"path": str(Path(sample_cache).resolve()),
            "manifest_sha256": sha256_file(Path(sample_cache) / "manifest.json"),
            "records_sha256": cache_manifest["records_sha256"]}
        manifest["base_validation_metrics"] = base_metrics
    np.savez_compressed(
        output_dir / "split_indices.npz",
        train=np.asarray([r.read_index for r in train], dtype=np.int64),
        validation=np.asarray([r.read_index for r in validation], dtype=np.int64),
        train_probe=np.asarray([r.read_index for r in probe], dtype=np.int64),
    )
    _write_json_atomic(output_dir / "manifest.json", manifest)
    common = dict(device=device, batch_reads=config.eval_batch_reads, source=source,
                  total=config.quantization_total, version=2, quantized=True)
    if progress:
        print(f"Fixed split: train={len(train)} validation={len(validation)} probe={len(probe)}; "
              "scoring the shared head-only starting point", file=sys.stderr, flush=True)
    initial = {"train": score_records(model, train, **common),
               "validation": score_records(model, validation, **common),
               "train_probe": score_records(model, probe, **common)}
    _write_json_atomic(output_dir / "initial_metrics.json", initial)
    results = {}
    for branch in (("lora", "full") if mode == "both" else (mode,)):
        if progress:
            print(f"Stage 3/3: {branch} adaptation; target_steps={config.steps}", file=sys.stderr, flush=True)
        results[branch] = fit_branch(
            model, train, validation, probe,
            mode=branch, source=source, device=device, output_dir=output_dir,
            base_sha256=checkpoint_sha, initial_metrics=initial,
            steps=config.steps, eval_every=config.eval_every, batch_reads=config.batch_reads,
            eval_batch_reads=config.eval_batch_reads,
            learning_rate=config.lora_lr if branch == "lora" else config.full_lr,
            head_learning_rate=config.head_lr, anchor_strength=config.anchor_strength,
            seed=config.seed, lora_layers=config.lora_layers, lora_rank=config.lora_rank,
            patience=config.patience, total=config.quantization_total,
            quantization_version=2, progress=progress,
        )
        _write_json_atomic(output_dir / f"{branch}.summary.json", results[branch])
        report = {"format": manifest["format"], "version": 1, "output_dir": str(output_dir.resolve()),
                  "manifest": "manifest.json", "common_initial": initial, "modes": results,
                  "scope": manifest["scope"]}
        _write_json_atomic(output_dir / "summary.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--mode", choices=("both", "lora", "full"), default="both")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-cache", type=Path,
                        help="reuse a frozen codec.adaptation_sample reservoir instead of rescanning FASTQ")
    for name, field in DiagnosticConfig.__dataclass_fields__.items():
        parser.add_argument("--" + name.replace("_", "-"), type=type(field.default), default=field.default)
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = DiagnosticConfig(**{name: getattr(args, name) for name in DiagnosticConfig.__dataclass_fields__})
        device = choose_device(args.device)
        if not args.input.is_file() or not args.checkpoint.is_file():
            raise FileNotFoundError("input FASTQ and pretrained checkpoint must both exist")
        args.output_root.mkdir(parents=True, exist_ok=True)
        output_dir = Path(tempfile.mkdtemp(prefix=f"{args.input.name.split('.')[0]}_{args.mode}_",
                                           dir=args.output_root))
        report = run_diagnostic(args.input, args.checkpoint, output_dir=output_dir,
                                mode=args.mode, device=device, config=config, progress=not args.no_progress,
                                sample_cache=args.sample_cache)
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
