"""Evaluate parameter-group reversions of a completed full-adaptation probe.

Every candidate starts from the same saved full model. No training or complete
file compression is performed, and no modified model replaces an existing
checkpoint. Scores measure dependence of this full solution on each group;
they are not independently additive contributions.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Optional, Sequence
import sys

import numpy as np
import torch

from .adaptation_diagnostic import DIAGNOSTIC_FORMAT, score_records
from .checkpoint import load_training_checkpoint
from .container import sha256_file
from .diagnose_adaptation import DiagnosticConfig, _content_digest
from .evaluate import choose_device
from .fastq_stream import iter_fastq_records
from .model import CrossLayerOutputHead, DirectQualityModelConfig
from .train import _write_json_atomic

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "output/adaptation_ablation_20261021"
SCORE_TOLERANCE_BPQ = 2e-5


def parameter_groups(model, early_layers=2):
    if type(early_layers) is not int or not 1 <= early_layers <= model.config.num_layers:
        raise ValueError("early_layers must be between one and the model depth")
    names = list(model.state_dict())
    early_prefixes = tuple(f"transformer.layers.{index}." for index in range(early_layers))
    input_prefixes = ("prev_q_embedding.", "qmer_embeddings.",
                      "base_encoder.", "input_projection.")
    return {"early_layers": [name for name in names if name.startswith(early_prefixes)],
            "input_features": [name for name in names if name.startswith(input_prefixes)]}


def build_candidate(full_model, base_model, *, revert_early_layers=False,
                    revert_input_features=False, early_layers=2):
    if type(revert_early_layers) is not bool or type(revert_input_features) is not bool:
        raise ValueError("reversion flags must be booleans")
    if full_model.config != base_model.config:
        raise ValueError("full and base model configurations differ")
    groups = parameter_groups(full_model, early_layers)
    selected = ((groups["early_layers"] if revert_early_layers else [])
                + (groups["input_features"] if revert_input_features else []))
    candidate = copy.deepcopy(full_model).eval().requires_grad_(False)
    candidate_state, base_state = candidate.state_dict(), base_model.state_dict()
    with torch.no_grad():
        for name in selected:
            if name not in base_state or base_state[name].shape != candidate_state[name].shape:
                raise ValueError(f"base model is missing a compatible tensor: {name}")
            candidate_state[name].copy_(base_state[name])
    return candidate


def _read_mapping(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON mapping")
    return value


def _check_source_identity(source, expected):
    stat = source.stat()
    if {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns} != expected:
        raise ValueError("source identity does not match the completed diagnostic")


def _load_split(path, split, warmup_reads):
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"train", "validation", "train_probe"}:
            raise ValueError("unexpected diagnostic split fields")
        indices = {name: archive[name].copy() for name in archive.files}
    for name, values in indices.items():
        if values.ndim != 1 or values.dtype.kind not in ("i", "u") or not len(values):
            raise ValueError("split indices must be nonempty one-dimensional integer arrays")
        values = values.astype(np.int64)
        if np.any(values < warmup_reads) or len(np.unique(values)) != len(values):
            raise ValueError("split indices overlap warm-up reads or contain duplicates")
        if len(values) != split[name + "_reads"]:
            raise ValueError("split read counts do not match manifest")
        digest = hashlib.sha256(values.astype("<i8").tobytes()).hexdigest()
        if digest != split[name + "_indices_sha256"]:
            raise ValueError("split index SHA-256 does not match manifest")
        indices[name] = values
    train, validation = set(indices["train"]), set(indices["validation"])
    if train & validation or not set(indices["train_probe"]) <= train:
        raise ValueError("training/validation overlap or invalid training probe")
    return indices


def _collect_records(source, indices, *, total_reads, progress):
    wanted = set(indices["train"]) | set(indices["validation"])
    found = {}
    iterator = iter_fastq_records(source)
    bar = (tqdm(total=total_reads, desc="recover fixed read split", unit="read", file=sys.stderr)
           if progress and tqdm is not None else None)
    count = 0
    try:
        for record in iterator:
            count += 1
            if record.read_index in wanted:
                found[record.read_index] = record
            if bar is not None:
                bar.update(1)
    finally:
        iterator.close()
        if bar is not None:
            bar.close()
    if count != total_reads or set(found) != wanted:
        raise ValueError("source read count or selected indices no longer match the diagnostic")
    return {name: [found[int(index)] for index in values] for name, values in indices.items()}


def _load_full_model(diagnostic_dir, manifest, base_model, full_summary):
    payload = torch.load(diagnostic_dir / "full.best.pt", map_location="cpu")
    if (not isinstance(payload, dict) or payload.get("format") != DIAGNOSTIC_FORMAT
            or payload.get("version") != 1 or payload.get("mode") != "full"
            or payload.get("wire_parameterization") != "direct_full"
            or payload.get("base_checkpoint_sha256") != manifest["base_checkpoint_sha256"]
            or payload.get("selected_step") != full_summary["selected_step"]):
        raise ValueError("full diagnostic checkpoint identity does not match")
    config = DirectQualityModelConfig.from_dict(payload["model_config"])
    if config != base_model.config:
        raise ValueError("full diagnostic model configuration does not match base")
    expected_head = {"type": "cross_layer", "residual_dim": manifest["config"]["residual_dim"],
                     "cross_dim": manifest["config"]["cross_dim"]}
    if payload.get("head_spec") != expected_head:
        raise ValueError("full diagnostic head does not match manifest")
    model = copy.deepcopy(base_model)
    model.output_head = CrossLayerOutputHead(config.d_model, expected_head["residual_dim"],
                                             expected_head["cross_dim"])
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if any(not torch.isfinite(tensor).all() for tensor in model.state_dict().values()):
        raise ValueError("full diagnostic checkpoint contains nonfinite tensors")
    return model.eval().requires_grad_(False)


def run_ablation(diagnostic_dir, *, output_dir, device=torch.device("cpu"),
                 eval_batch_reads=256, early_layers=2, progress=True):
    diagnostic_dir, output_dir = Path(diagnostic_dir).resolve(), Path(output_dir).resolve()
    device = torch.device(device)
    if type(eval_batch_reads) is not int or not 1 <= eval_batch_reads <= 256:
        raise ValueError("eval_batch_reads must be in [1, 256]")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("choose a new empty ablation output directory")
    manifest = _read_mapping(diagnostic_dir / "manifest.json")
    if manifest.get("format") != DIAGNOSTIC_FORMAT or manifest.get("version") != 1:
        raise ValueError("unsupported diagnostic manifest")
    config = DiagnosticConfig(**manifest["config"])
    full_summary = _read_mapping(diagnostic_dir / "full.summary.json")
    if full_summary.get("mode") != "full":
        raise ValueError("this evaluation requires a completed full diagnostic")
    source = Path(manifest["source"])
    base_path = Path(manifest["base_checkpoint"])
    _check_source_identity(source, manifest["source_identity"])
    if sha256_file(base_path) != manifest["base_checkpoint_sha256"]:
        raise ValueError("base checkpoint SHA-256 does not match diagnostic")
    base = load_training_checkpoint(base_path).model.eval().requires_grad_(False)
    if base.config != DirectQualityModelConfig.from_dict(manifest["model_config"]):
        raise ValueError("base configuration does not match diagnostic manifest")
    full = _load_full_model(diagnostic_dir, manifest, base, full_summary).to(device)
    groups = parameter_groups(full, early_layers)
    split_path = Path(manifest["split"]["indices_file"])
    if split_path.name != str(split_path):
        raise ValueError("split file must be a local diagnostic filename")
    indices = _load_split(diagnostic_dir / split_path, manifest["split"],
                          manifest["warmup"]["prefix_reads"])
    if progress:
        print(f"Ablation output: {output_dir}", file=sys.stderr, flush=True)
        print("Load the original fixed split; no sampling or training", file=sys.stderr, flush=True)
    records = _collect_records(source, indices,
        total_reads=manifest["sample"]["total_file_reads"], progress=progress)
    _check_source_identity(source, manifest["source_identity"])
    if _content_digest(records["train"] + records["validation"]) != manifest["split"]["sample_content_sha256"]:
        raise ValueError("sample content SHA-256 does not match diagnostic")
    for name in ("train", "validation"):
        if sum(len(record.quality) for record in records[name]) != manifest["split"][name + "_quality_symbols"]:
            raise ValueError("sample quality symbol counts do not match manifest")
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance = {
        "diagnostic_dir": str(diagnostic_dir), "base_checkpoint": str(base_path),
        "base_checkpoint_sha256": manifest["base_checkpoint_sha256"],
        "full_checkpoint_sha256": sha256_file(diagnostic_dir / "full.best.pt"),
        "selected_step": full_summary["selected_step"], "source": str(source),
        "sample_content_sha256": manifest["split"]["sample_content_sha256"],
        "split": manifest["split"], "early_layers": early_layers,
        "eval_batch_reads": eval_batch_reads, "device": str(device),
        "quantization_version": 2, "quantization_total": config.quantization_total,
    }
    report = {"format": "direct-quality-adaptation-ablation", "version": 1,
              "provenance": provenance, "cases": {},
              "interpretation": "dependence_of_this_full_solution_on_jointly_adapted_groups_not_additive_contributions"}
    score_kwargs = dict(device=device, batch_reads=eval_batch_reads, source=source,
                        total=config.quantization_total, version=2, quantized=True)
    candidates = (("full", False, False), ("revert_early_layers", True, False),
                  ("revert_input_features", False, True), ("revert_both", True, True))
    for name, early, inputs in candidates:
        if progress:
            print(f"Evaluate {name}", file=sys.stderr, flush=True)
        candidate = build_candidate(full, base, revert_early_layers=early,
                                    revert_input_features=inputs, early_layers=early_layers)
        result = {subset: score_records(candidate, records[subset], **score_kwargs)
                  for subset in ("train_probe", "validation")}
        del candidate
        result["reverted_parameters"] = ((groups["early_layers"] if early else [])
                                          + (groups["input_features"] if inputs else []))
        if name == "full":
            comparison = {}
            for subset in ("train_probe", "validation"):
                expected = full_summary["selected_metrics"][subset]
                observed = result[subset]
                if (observed["reads"] != expected["reads"]
                        or observed["quality_symbols"] != expected["quality_symbols"]):
                    raise ValueError("full baseline score read/symbol counts do not match")
                comparison[subset] = {metric: observed[metric] - expected[metric]
                                      for metric in ("float_bpq", "quantized_bpq")}
                if any(abs(delta) > SCORE_TOLERANCE_BPQ for delta in comparison[subset].values()):
                    raise ValueError("full baseline scores do not reproduce the completed diagnostic")
            report["initial_full_score_comparison"] = comparison
            report["score_reproduction_tolerance_bpq"] = SCORE_TOLERANCE_BPQ
        reference = report["cases"].get("full", result)["validation"]
        result["validation_delta_from_full"] = result["validation"]["float_bpq"] - reference["float_bpq"]
        result["validation_quantized_delta_from_full"] = result["validation"]["quantized_bpq"] - reference["quantized_bpq"]
        report["cases"][name] = result
        _write_json_atomic(output_dir / f"{name}.json", result)
        _write_json_atomic(output_dir / "summary.json", report)
        if progress:
            print(f"{name}: validation_bpq={result['validation']['float_bpq']:.6f} "
                  f"quantized_bpq={result['validation']['quantized_bpq']:.6f} "
                  f"delta={result['validation_quantized_delta_from_full']:+.6f}", file=sys.stderr, flush=True)
    return report


def main(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("diagnostic_dir", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-batch-reads", type=int, default=256)
    parser.add_argument("--early-layers", type=int, default=2)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args(argv)
    try:
        device = choose_device(args.device)
        args.output_root.mkdir(parents=True, exist_ok=True)
        output_dir = Path(tempfile.mkdtemp(prefix="full_reversion_", dir=args.output_root))
        report = run_ablation(args.diagnostic_dir, output_dir=output_dir, device=device,
            eval_batch_reads=args.eval_batch_reads, early_layers=args.early_layers,
            progress=not args.no_progress)
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
