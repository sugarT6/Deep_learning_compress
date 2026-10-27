"""Locate code-length losses using saved models and a fresh, fixed read sample.

No fitting, arithmetic coding, or FASTQ Header features are used. The completed
Full diagnostic supplies provenance and the read indices to exclude. The suffix
adapter must document the same reservoir so that neither fitting nor model
selection reads enter this new sample.
"""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import tempfile
from typing import Optional, Sequence

import numpy as np
import torch

from .ablate_adaptation import (
    _check_source_identity, _load_full_model, _load_split, _read_mapping,
)
from .adaptation_diagnostic import DIAGNOSTIC_FORMAT
from .backbone_adapter import _tensors
from .checkpoint import load_training_checkpoint
from .container import sha256_file
from .diagnose_adaptation import DiagnosticConfig, _content_digest
from .evaluate import choose_device
from .fastq_stream import iter_fastq_records
from .head_adapter import adapter_container_bytes, apply_head_adapter
from .model import DirectQualityModelConfig
from .probability_quantization import logits_to_intervals, validate_total
from .train import _write_json_atomic

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "output/code_length_diagnostic"
MODEL_NAMES = ("base", "lora", "full")
TRANSITIONS = ("first", "unchanged", "drop_1_to_4", "drop_5_plus",
               "rise_1_to_4", "rise_5_plus")


@dataclass(frozen=True)
class CodeLengthConfig:
    reads: int = 8192
    blocks: int = 8
    batch_reads: int = 256
    seed: int = 20261027
    bootstrap_samples: int = 1000
    max_symbols: int = 8000000

    def __post_init__(self):
        for name in ("reads", "blocks", "batch_reads", "bootstrap_samples", "max_symbols"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.reads < self.blocks or self.blocks > 65535:
            raise ValueError("reads must cover all blocks, and blocks must be <=65535")
        if self.batch_reads > 256:
            raise ValueError("batch_reads must be <=256")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")


def _indices_sha(values):
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def _load_inputs(diagnostic_dir, adapter_path):
    manifest = _read_mapping(diagnostic_dir / "manifest.json")
    if manifest.get("format") != DIAGNOSTIC_FORMAT or manifest.get("version") != 1:
        raise ValueError("unsupported completed diagnostic manifest")
    old_config = DiagnosticConfig(**manifest["config"])
    source, checkpoint = Path(manifest["source"]), Path(manifest["base_checkpoint"])
    _check_source_identity(source, manifest["source_identity"])
    digest = sha256_file(checkpoint)
    if digest != manifest["base_checkpoint_sha256"]:
        raise ValueError("base checkpoint SHA-256 does not match diagnostic")
    split_file = Path(manifest["split"]["indices_file"])
    if split_file.name != str(split_file):
        raise ValueError("split indices must be a local diagnostic filename")
    prefix_reads = manifest["warmup"]["prefix_reads"]
    split = _load_split(diagnostic_dir / split_file, manifest["split"], prefix_reads)
    reservoir = np.concatenate((split["train"], split["validation"]))
    if _indices_sha(reservoir) != manifest["sample"]["read_indices_sha256"]:
        raise ValueError("diagnostic reservoir does not match its saved split")
    total_reads = manifest["sample"]["total_file_reads"]
    if type(total_reads) is not int or total_reads < 1 or np.any(reservoir >= total_reads):
        raise ValueError("invalid total reads or out-of-range exclusion indices")
    artifact = _read_mapping(adapter_path)
    if (artifact.get("format") != "direct-quality-head-adaptation-artifact"
            or artifact.get("version") != 1
            or artifact.get("base_checkpoint_sha256") != digest):
        raise ValueError("adapter artifact/base checkpoint mismatch")
    if Path(artifact.get("source", "")).resolve() != source.resolve():
        raise ValueError("adapter source does not match the diagnostic FASTQ")
    adapter, report = artifact.get("adapter"), artifact.get("report", {})
    if not isinstance(adapter, dict) or adapter.get("version") != 8:
        raise ValueError("this diagnostic requires a saved suffix LoRA v8 adapter without read labels")
    if (not report.get("backbone_accepted")
            or report.get("backbone_sampling") != manifest["sample"]
            or report.get("warmup_report", {}).get("prefix_reads") != prefix_reads
            or report.get("backbone_train_reads") != len(split["train"])
            or report.get("backbone_validation_reads") != len(split["validation"])
            or report.get("config", {}).get("quantization_version") != 2
            or report.get("config", {}).get("backbone_layers") != len(adapter.get("layers", []))
            or report.get("backbone_lora_rank") != adapter.get("rank")):
        raise ValueError("LoRA and Full must document the same reservoir and warm-up prefix")
    base = load_training_checkpoint(checkpoint).model.eval().requires_grad_(False)
    if base.config != DirectQualityModelConfig.from_dict(manifest["model_config"]):
        raise ValueError("base configuration does not match the diagnostic")
    lora = copy.deepcopy(base)
    apply_head_adapter(lora, adapter, digest)
    summary = _read_mapping(diagnostic_dir / "full.summary.json")
    if summary.get("mode") != "full":
        raise ValueError("this diagnostic requires a completed Full branch")
    full = _load_full_model(diagnostic_dir, manifest, base, summary)
    provenance = {
        "diagnostic_dir": str(diagnostic_dir), "source": str(source.resolve()),
        "source_identity": manifest["source_identity"],
        "base_checkpoint": str(checkpoint.resolve()), "base_checkpoint_sha256": digest,
        "full_checkpoint_sha256": sha256_file(diagnostic_dir / "full.best.pt"),
        "full_selected_step": summary["selected_step"],
        "adapter_path": str(adapter_path), "adapter_sha256": sha256_file(adapter_path),
        "lora_layers": len(adapter["layers"]), "lora_rank": adapter["rank"],
        "lora_selected_step": report["backbone_selected_step"],
        "lora_container_adapter_bytes": adapter_container_bytes(adapter),
        "full_diagnostic_checkpoint_bytes": (diagnostic_dir / "full.best.pt").stat().st_size,
        "full_checkpoint_bytes_are_not_a_minimal_deployable_adapter": True,
        "excluded_prefix_reads": prefix_reads, "excluded_reservoir_reads": len(reservoir),
        "excluded_train_indices_sha256": manifest["split"]["train_indices_sha256"],
        "excluded_validation_indices_sha256": manifest["split"]["validation_indices_sha256"],
        "quantization_version": 2, "quantization_total": old_config.quantization_total,
        "models": {
            "base": "unadapted shared model, original output head",
            "lora": "saved suffix LoRA and its jointly trained output head",
            "full": "saved Full diagnostic model",
        },
        "warm_head_note": "The historical prefix-only warm head was not saved; it is not reconstructed or retrained.",
    }
    return manifest, old_config, reservoir, {"base": base, "lora": lora, "full": full}, provenance


def sample_fresh_records(source, *, total_reads, prefix_reads, excluded_indices,
                         max_read_length, config, progress=True):
    """Uniform reservoirs within equal read-index blocks; only retain new reads."""
    if type(total_reads) is not int or total_reads < config.blocks:
        raise ValueError("the source must contain at least one read per block")
    excluded = set(int(index) for index in excluded_indices)
    quotas = [config.reads // config.blocks + (i < config.reads % config.blocks)
              for i in range(config.blocks)]
    pools = [[] for _ in quotas]
    eligible = [0] * config.blocks
    rng = random.Random(config.seed)
    counts = dict(total_reads=0, total_symbols=0, excluded_prefix=0,
                  excluded_reservoir=0, excluded_empty=0, excluded_long=0)
    bar = (tqdm(total=total_reads, desc="sample fresh reads", unit="read", file=sys.stderr)
           if progress and tqdm is not None else None)
    iterator = iter_fastq_records(source)
    try:
        for record in iterator:
            if record.read_index != counts["total_reads"]:
                raise ValueError("FASTQ read indices are not contiguous")
            counts["total_reads"] += 1
            counts["total_symbols"] += len(record.quality)
            if bar is not None:
                bar.update(1)
            if record.read_index >= total_reads:
                raise ValueError("FASTQ has more reads than the saved diagnostic")
            if record.read_index < prefix_reads:
                counts["excluded_prefix"] += 1
                continue
            if record.read_index in excluded:
                counts["excluded_reservoir"] += 1
                continue
            if not record.quality:
                counts["excluded_empty"] += 1
                continue
            if len(record.quality) > max_read_length:
                counts["excluded_long"] += 1
                continue
            block = record.read_index * config.blocks // total_reads
            eligible[block] += 1
            if len(pools[block]) < quotas[block]:
                pools[block].append(record)
            else:
                slot = rng.randrange(eligible[block])
                if slot < quotas[block]:
                    pools[block][slot] = record
    finally:
        iterator.close()
        if bar is not None:
            bar.close()
    if counts["total_reads"] != total_reads:
        raise ValueError("FASTQ read count does not match the saved diagnostic")
    if any(len(pool) != quota for pool, quota in zip(pools, quotas)):
        raise ValueError("a file block has too few unused eligible reads; reduce --reads/--blocks")
    records = sorted((r for pool in pools for r in pool), key=lambda r: r.read_index)
    if sum(len(r.quality) for r in records) > config.max_symbols:
        raise ValueError("selected reads exceed --max-symbols; reduce --reads or raise the explicit limit")
    counts.update(block_eligible_reads=eligible, block_sample_reads=quotas,
                  sampled_reads=len(records), sampled_symbols=sum(len(r.quality) for r in records),
                  read_indices_sha256=_indices_sha([r.read_index for r in records]),
                  sample_content_sha256=_content_digest(records),
                  rule="uniform_without_replacement_within_equal_read_index_blocks")
    return records, counts


def symbol_metadata(records, *, total_reads, blocks):
    lengths = np.asarray([len(r.quality) for r in records], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(lengths)))
    qualities = np.concatenate([np.frombuffer(r.quality, dtype=np.uint8) - 33 for r in records])
    delta = np.zeros(len(qualities), dtype=np.int16)
    cycles = np.concatenate([np.arange(1, n + 1, dtype=np.uint16) for n in lengths])
    for begin, end in zip(offsets[:-1], offsets[1:]):
        delta[begin + 1:end] = np.diff(qualities[begin:end].astype(np.int16))
    transition = np.ones(len(qualities), dtype=np.uint8)
    transition[(delta < 0) & (delta > -5)] = 2
    transition[delta <= -5] = 3
    transition[(delta > 0) & (delta < 5)] = 4
    transition[delta >= 5] = 5
    transition[cycles == 1] = 0
    indices = np.asarray([r.read_index for r in records], dtype=np.int64)
    block_ids = indices * blocks // total_reads + 1
    return dict(read_offsets=offsets, sample_read_indices=indices,
                sample_block_ids=block_ids, read_index=np.repeat(indices, lengths),
                cycle=cycles, quality=qualities, quality_delta=delta,
                transition=transition, file_block=np.repeat(block_ids, lengths))


def symbol_code_lengths(logits, labels, *, total=65536):
    """Float64 true-Q log loss and exact floor-v2 selected-interval log loss."""
    total = validate_total(total)
    matrix = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels)
    if (matrix.ndim != 2 or matrix.shape[1] != 42 or labels.shape != (len(matrix),)
            or labels.dtype.kind not in "iu" or np.any(labels < 0) or np.any(labels > 41)
            or not np.isfinite(matrix).all()):
        raise ValueError("invalid logits or quality labels for symbol scoring")
    if not len(matrix):
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    shifted = matrix - matrix.max(axis=1, keepdims=True)
    float_bits = (np.log(np.exp(shifted).sum(axis=1, dtype=np.float64))
                  - shifted[np.arange(len(matrix)), labels]) / math.log(2.)
    low, high, totals = logits_to_intervals(matrix, labels, total=total)
    quantized_bits = -np.log2((high - low).astype(np.float64) / totals)
    return float_bits, quantized_bits


@torch.no_grad()
def score_symbols(model, records, *, source, device, batch_reads, total, progress=True, name="model"):
    size = sum(len(r.quality) for r in records)
    floating, quantized = np.empty(size, dtype=np.float64), np.empty(size, dtype=np.float64)
    was_training = model.training
    model.eval()
    offset = 0
    iterator = range(0, len(records), batch_reads)
    bar = (tqdm(iterator, desc=f"score {name}", unit="batch", file=sys.stderr)
           if progress and tqdm is not None else iterator)
    try:
        for begin in bar:
            rows = records[begin:begin + batch_reads]
            tensors = _tensors(rows, device, source)
            active = tensors["active_mask"]
            logits = model.forward_full(**tensors)[active].detach().cpu().numpy()
            labels = tensors["qualities"][active].cpu().numpy()
            expected = np.concatenate([np.frombuffer(r.quality, dtype=np.uint8) - 33 for r in rows])
            if not np.array_equal(labels, expected):
                raise ValueError("active tensor positions do not match read-major sample labels")
            f, q = symbol_code_lengths(logits, labels, total=total)
            floating[offset:offset + len(labels)] = f
            quantized[offset:offset + len(labels)] = q
            offset += len(labels)
    finally:
        if hasattr(bar, "close"):
            bar.close()
        model.train(was_training)
    if offset != size:
        raise ValueError("not every sampled symbol was scored")
    return floating, quantized


def paired_bootstrap(gain_per_read, lengths, block_ids, *, samples, seed):
    """Resample whole reads, paired between models and stratified by file block."""
    rng = np.random.default_rng(seed)
    draws = []
    for start in range(0, samples, 64):
        count = min(64, samples - start)
        numerator, denominator = np.zeros(count), np.zeros(count)
        for block in np.unique(block_ids):
            rows = np.flatnonzero(block_ids == block)
            selected = rows[rng.integers(0, len(rows), size=(count, len(rows)))]
            numerator += gain_per_read[selected].sum(axis=1)
            denominator += lengths[selected].sum(axis=1)
        draws.extend(numerator / denominator)
    return [float(x) for x in np.quantile(draws, [.025, .975])]


def _write_csv(path, rows):
    with Path(path).open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_scores(metadata, scores, *, config):
    offsets = metadata["read_offsets"]
    lengths = np.diff(offsets)
    reference = scores["lora"][1]
    models, comparisons, per_read = {}, {}, {}
    for name, (f, q) in scores.items():
        models[name] = dict(reads=len(lengths), quality_symbols=len(q),
                            float_bits=float(f.sum()), float_bpq=float(f.mean()),
                            quantized_bits=float(q.sum()), quantized_bpq=float(q.mean()),
                            quantization_penalty_bpq=float((q - f).mean()))
        read_f, read_q = np.add.reduceat(f, offsets[:-1]), np.add.reduceat(q, offsets[:-1])
        per_read[name] = (read_f, read_q)
        gain = reference - q
        read_gain = np.add.reduceat(gain, offsets[:-1])
        comparisons[name] = dict(
            gain_bits_vs_lora=float(gain.sum()), gain_bpq_vs_lora=float(gain.mean()),
            gain_bpq_95pct_read_bootstrap=paired_bootstrap(
                read_gain, lengths, metadata["sample_block_ids"],
                samples=config.bootstrap_samples, seed=config.seed + 1),
            fraction_reads_better_than_lora=float((read_gain > 0).mean()),
            positive_gain_bits=float(gain[gain > 0].sum()),
            negative_gain_bits=float(gain[gain < 0].sum()))
    grouped = []
    for dimension, values in (("cycle", metadata["cycle"]), ("quality", metadata["quality"]),
                              ("transition", metadata["transition"]), ("file_block", metadata["file_block"])):
        groups, inverse = np.unique(values, return_inverse=True)
        counts = np.bincount(inverse)
        ref_bits = np.bincount(inverse, weights=reference)
        for name, (f, q) in scores.items():
            sum_f, sum_q = np.bincount(inverse, weights=f), np.bincount(inverse, weights=q)
            total_q = float(q.sum())
            for i, group in enumerate(groups):
                grouped.append(dict(dimension=dimension,
                    group=TRANSITIONS[int(group)] if dimension == "transition" else int(group),
                    model=name, symbols=int(counts[i]), float_bits=float(sum_f[i]),
                    float_bpq=float(sum_f[i] / counts[i]), quantized_bits=float(sum_q[i]),
                    quantized_bpq=float(sum_q[i] / counts[i]),
                    fraction_of_model_sample_bits=float(sum_q[i] / total_q),
                    gain_bits_vs_lora=float(ref_bits[i] - sum_q[i]),
                    gain_bpq_vs_lora=float((ref_bits[i] - sum_q[i]) / counts[i])))
    reads = []
    for i, index in enumerate(metadata["sample_read_indices"]):
        begin, end = offsets[i:i + 2]
        row = dict(read_index=int(index), file_block=int(metadata["sample_block_ids"][i]),
                   length=int(lengths[i]), mean_q=float(metadata["quality"][begin:end].mean()),
                   drops_5_plus=int((metadata["quality_delta"][begin + 1:end] <= -5).sum()))
        for name, (f, q) in per_read.items():
            row.update({f"{name}_float_bits": float(f[i]), f"{name}_quantized_bits": float(q[i]),
                        f"{name}_quantized_bpq": float(q[i] / lengths[i]),
                        f"{name}_gain_bits_vs_lora": float(per_read["lora"][1][i] - q[i])})
        reads.append(row)
    return models, comparisons, grouped, reads


def _markdown_report(report, grouped):
    lines = ["# Fixed-sample code-length diagnostic", "",
             "No training or full-file compression was performed. Positive gain means fewer bits than saved LoRA.",
             "All sizes below are probability code lengths on the sample, excluding adapter/model storage.", "",
             f"Sample: {report['sampling']['sampled_reads']} reads / {report['sampling']['sampled_symbols']} Q; "
             f"{report['config']['blocks']} file blocks. Read indices are zero-based; cycles and blocks are one-based.", "",
             "| Model | Float bit/Q | Floor bit/Q | Gain vs LoRA bit/Q | Paired read bootstrap 95% |",
             "|---|---:|---:|---:|---|"]
    for name, metric in report["models"].items():
        comparison = report["comparisons"][name]
        low, high = comparison["gain_bpq_95pct_read_bootstrap"]
        lines.append(f"| {name} | {metric['float_bpq']:.9f} | {metric['quantized_bpq']:.9f} | "
                     f"{comparison['gain_bpq_vs_lora']:+.9f} | [{low:+.9f}, {high:+.9f}] |")
    for dimension in ("cycle", "quality", "transition", "file_block"):
        hard = [r for r in grouped if r["dimension"] == dimension and r["model"] == "lora"]
        hard.sort(key=lambda r: r["quantized_bits"], reverse=True)
        lines += ["", f"## {dimension}: largest LoRA code-length contributions", "",
                  "| Group | Q count | LoRA bit/Q | Fraction of sample bits |",
                  "|---|---:|---:|---:|"]
        for row in hard[:12]:
            lines.append(f"| {row['group']} | {row['symbols']} | {row['quantized_bpq']:.6f} | "
                         f"{row['fraction_of_model_sample_bits']:.3%} |")
        rows = [r for r in grouped if r["dimension"] == dimension and r["model"] == "full"]
        rows.sort(key=lambda r: r["gain_bits_vs_lora"], reverse=True)
        lines += ["", f"## {dimension}: Full gain relative to LoRA (largest total gains first)", "",
                  "| Group | Q count | Full bit/Q | Gain bit/Q | Total gain bits |",
                  "|---|---:|---:|---:|---:|"]
        for row in rows[:12]:
            lines.append(f"| {row['group']} | {row['symbols']} | {row['quantized_bpq']:.6f} | "
                         f"{row['gain_bpq_vs_lora']:+.6f} | {row['gain_bits_vs_lora']:+.3f} |")
    lines += ["", "## Interpretation limits", ""]
    lines += [f"- {limit}" for limit in report["limits"]]
    return "\n".join(lines) + "\n"


def run_code_length_diagnostic(diagnostic_dir, adapter_path, *, output_dir,
                               device="cpu", config=CodeLengthConfig(), progress=True):
    diagnostic_dir, adapter_path = Path(diagnostic_dir).resolve(), Path(adapter_path).resolve()
    output_dir, device = Path(output_dir).resolve(), torch.device(device)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("choose a new empty code-length output directory")
    manifest, old_config, excluded, models, provenance = _load_inputs(diagnostic_dir, adapter_path)
    source = Path(manifest["source"])
    if progress:
        print(f"Code-length output: {output_dir}", file=sys.stderr, flush=True)
        print("Saved models only; exclude both adaptation and selection reads. No fitting.",
              file=sys.stderr, flush=True)
    records, sampling = sample_fresh_records(source,
        total_reads=manifest["sample"]["total_file_reads"],
        prefix_reads=manifest["warmup"]["prefix_reads"], excluded_indices=excluded,
        max_read_length=old_config.max_read_length, config=config, progress=progress)
    _check_source_identity(source, manifest["source_identity"])
    if sampling["total_symbols"] != manifest["sample"]["total_file_symbols"]:
        raise ValueError("FASTQ symbol count does not match the diagnostic")
    metadata = symbol_metadata(records, total_reads=sampling["total_reads"], blocks=config.blocks)
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance.update(config=asdict(config), sampling=sampling, device=str(device))
    _write_json_atomic(output_dir / "manifest.json", provenance)
    np.savez_compressed(output_dir / "sample_indices.npz",
                        read_indices=metadata["sample_read_indices"], block_ids=metadata["sample_block_ids"])
    scores = {}
    for name in MODEL_NAMES:
        model = models.pop(name).to(device).eval().requires_grad_(False)
        scores[name] = score_symbols(model, records, source=source, device=device,
            batch_reads=config.batch_reads, total=old_config.quantization_total, progress=progress, name=name)
        del model
    metrics, comparisons, grouped, read_rows = summarize_scores(metadata, scores, config=config)
    report = dict(format="direct-quality-code-length-diagnostic", version=1,
                  config=asdict(config), provenance=provenance, sampling=sampling,
                  models=metrics, comparisons=comparisons, limits=[
        "The prefix-only warm head was not archived. 'base' is the unadapted shared model, not a refitted warm head.",
        "This is a fresh read sample from a repeatedly studied file, not an unseen-file generalization test.",
        "Each block contributes a fixed read quota. Global BPQ is the sampled-symbol mean, not a population-weighted file estimate.",
        "Bootstrap resamples whole reads within file blocks. It excludes training-seed uncertainty and may understate cross-read dependence.",
        "Quality/transition/mean-Q groups use true labels only for analysis, never as model inputs. Each Q is predicted causally.",
        "Float loss uses CPU float64 softmax; floor loss uses true positive integer frequencies and actual row totals.",
        "No arithmetic-coder framing, adapter cost or Full-model storage is included in BPQ. No complete-file byte saving is claimed.",
        "Comparisons of saved jointly trained models do not isolate causal module contributions or establish an entropy limit.",
    ])
    arrays = dict(metadata)
    for name, (f, q) in scores.items():
        arrays[f"{name}_float_bits"], arrays[f"{name}_quantized_bits"] = f, q
    np.savez_compressed(output_dir / "per_symbol.npz", **arrays)
    _write_csv(output_dir / "grouped_code_lengths.csv", grouped)
    _write_csv(output_dir / "per_read.csv", read_rows)
    _write_json_atomic(output_dir / "summary.json", report)
    (output_dir / "summary.md").write_text(_markdown_report(report, grouped), encoding="utf-8")
    if progress:
        for name, values in metrics.items():
            print(f"{name}: floor_bpq={values['quantized_bpq']:.9f} "
                  f"gain_vs_lora={comparisons[name]['gain_bpq_vs_lora']:+.9f}",
                  file=sys.stderr, flush=True)
        print(f"Report: {output_dir / 'summary.md'}", file=sys.stderr, flush=True)
    return report


def main(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("diagnostic_dir", type=Path)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda")
    for name, field in CodeLengthConfig.__dataclass_fields__.items():
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=field.default)
    args = parser.parse_args(argv)
    try:
        config = CodeLengthConfig(**{name: getattr(args, name) for name in CodeLengthConfig.__dataclass_fields__})
        device = choose_device(args.device)
        args.output_root.mkdir(parents=True, exist_ok=True)
        output = Path(tempfile.mkdtemp(prefix="code_lengths_", dir=args.output_root))
        run_code_length_diagnostic(args.diagnostic_dir, args.adapter, output_dir=output,
                                   device=device, config=config)
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit(str(exc)) from exc
    print(str(output.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
