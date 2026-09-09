#!/usr/bin/env python3
"""Evaluate ideal direct-quality code length from training caches."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch

from .checkpoint import load_training_checkpoint
from .datasets import DATASETS, TRAIN_DATASETS, DirectQualityDataset
from .model import (
    DirectQualityTransformer,
    fastq_batch_to_tensors,
    theoretical_bits,
)
from .training_cache import DEFAULT_CACHE_DIR, TrainingCacheReader, sha256_file


EVALUATION_CATEGORIES = ("train", "validation", "unseen_dataset")


@dataclass(frozen=True)
class DatasetMetrics:
    accession: str
    platform_family: str
    category: str
    unseen_instrument: bool
    reads: int
    symbols: int
    total_bits: float
    bits_per_quality: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return device


def evaluate_reader_range(
    model: DirectQualityTransformer,
    reader: TrainingCacheReader,
    dataset: DirectQualityDataset,
    *,
    category: str,
    start: int,
    stop: int,
    batch_reads: int,
    max_reads: int,
    device: torch.device,
) -> DatasetMetrics:
    """Evaluate a contiguous split without using it for checkpoint selection."""

    if category not in EVALUATION_CATEGORIES:
        raise ValueError(f"unknown evaluation category {category!r}")
    if batch_reads <= 0 or batch_reads > 64:
        raise ValueError("batch_reads must be in [1, 64]")
    if max_reads < 0:
        raise ValueError("max_reads must be nonnegative")
    if start < 0 or stop <= start or stop > reader.metadata.read_count:
        raise ValueError("invalid evaluation read range")
    if max_reads:
        stop = min(stop, start + max_reads)

    was_training = model.training
    model.eval()
    total_bits = 0.0
    total_symbols = 0
    evaluated_reads = 0
    try:
        with torch.inference_mode():
            for batch_start in range(start, stop, batch_reads):
                batch_stop = min(batch_start + batch_reads, stop)
                batch = reader.read_range(batch_start, batch_stop)
                tensors = fastq_batch_to_tensors(batch, device)
                logits = model.forward_full(**tensors)
                bits, symbols = theoretical_bits(
                    logits, tensors["qualities"], tensors["active_mask"]
                )
                total_bits += bits
                total_symbols += symbols
                evaluated_reads += batch.read_count
    finally:
        model.train(was_training)
    if total_symbols == 0:
        raise ValueError(f"{dataset.accession}: evaluation split has no quality symbols")
    return DatasetMetrics(
        accession=dataset.accession,
        platform_family=dataset.platform_family,
        category=category,
        unseen_instrument=dataset.unseen_instrument,
        reads=evaluated_reads,
        symbols=total_symbols,
        total_bits=total_bits,
        bits_per_quality=total_bits / total_symbols,
    )


def summarize_metric_group(metrics: Sequence[DatasetMetrics]) -> Dict[str, Any]:
    if not metrics:
        return {
            "dataset_count": 0,
            "dataset_macro_bits_per_quality": None,
            "symbol_micro_bits_per_quality": None,
            "platform_family_macro_bits_per_quality": None,
            "families": {},
            "worst_dataset": None,
        }
    total_bits = sum(metric.total_bits for metric in metrics)
    total_symbols = sum(metric.symbols for metric in metrics)
    families = {}
    for family in sorted({metric.platform_family for metric in metrics}):
        members = [metric for metric in metrics if metric.platform_family == family]
        family_bits = sum(metric.total_bits for metric in members)
        family_symbols = sum(metric.symbols for metric in members)
        families[family] = {
            "dataset_count": len(members),
            "dataset_macro_bits_per_quality": sum(
                metric.bits_per_quality for metric in members
            )
            / len(members),
            "symbol_micro_bits_per_quality": family_bits / family_symbols,
            "symbols": family_symbols,
        }
    worst = max(metrics, key=lambda metric: metric.bits_per_quality)
    return {
        "dataset_count": len(metrics),
        "dataset_macro_bits_per_quality": sum(
            metric.bits_per_quality for metric in metrics
        )
        / len(metrics),
        "symbol_micro_bits_per_quality": total_bits / total_symbols,
        "platform_family_macro_bits_per_quality": sum(
            family["symbol_micro_bits_per_quality"] for family in families.values()
        )
        / len(families),
        "symbols": total_symbols,
        "families": families,
        "worst_dataset": {
            "accession": worst.accession,
            "bits_per_quality": worst.bits_per_quality,
        },
    }


def build_evaluation_summary(metrics: Sequence[DatasetMetrics]) -> Dict[str, Any]:
    summary = {
        category: summarize_metric_group(
            [metric for metric in metrics if metric.category == category]
        )
        for category in EVALUATION_CATEGORIES
    }
    summary["unseen_instrument"] = summarize_metric_group(
        [
            metric
            for metric in metrics
            if metric.category == "unseen_dataset" and metric.unseen_instrument
        ]
    )
    summary["all_reported_splits"] = summarize_metric_group(metrics)
    return summary


def format_evaluation_report(
    metrics: Sequence[DatasetMetrics], summary: Mapping[str, Any]
) -> str:
    lines = [
        "category\tfamily\taccession\treads\tsymbols\tbits/Q\tunseen_instrument"
    ]
    for metric in metrics:
        lines.append(
            f"{metric.category}\t{metric.platform_family}\t{metric.accession}\t"
            f"{metric.reads}\t{metric.symbols}\t{metric.bits_per_quality:.8f}\t"
            f"{str(metric.unseen_instrument).lower()}"
        )
    lines.append("")
    for category in (*EVALUATION_CATEGORIES, "unseen_instrument"):
        group = summary[category]
        if not group["dataset_count"]:
            continue
        lines.append(
            f"{category}: dataset_macro={group['dataset_macro_bits_per_quality']:.8f} "
            f"symbol_micro={group['symbol_micro_bits_per_quality']:.8f} "
            f"family_macro={group['platform_family_macro_bits_per_quality']:.8f} "
            f"worst={group['worst_dataset']['accession']} "
            f"({group['worst_dataset']['bits_per_quality']:.8f} bits/Q)"
        )
        for family, family_metrics in group["families"].items():
            lines.append(
                f"  {family}: dataset_macro="
                f"{family_metrics['dataset_macro_bits_per_quality']:.8f} "
                f"symbol_micro={family_metrics['symbol_micro_bits_per_quality']:.8f}"
            )
    return "\n".join(lines)


def _checkpoint_split_entry(
    split_manifest: Mapping[str, Any], accession: str
) -> Mapping[str, Any]:
    entries = split_manifest.get("datasets", {})
    if accession not in entries:
        raise ValueError(f"checkpoint has no split entry for {accession}")
    return entries[accession]


def evaluate_fixed_split(
    checkpoint_path: Path,
    *,
    cache_dir: Path,
    categories: Sequence[str],
    batch_reads: int,
    max_reads_per_split: int,
    device: torch.device,
) -> Tuple[List[DatasetMetrics], Dict[str, Any]]:
    unknown = sorted(set(categories) - set(EVALUATION_CATEGORIES))
    if unknown:
        raise ValueError(f"unknown evaluation categories: {', '.join(unknown)}")
    loaded = load_training_checkpoint(checkpoint_path, device=device)
    model = loaded.model
    split_manifest = loaded.payload["data_split"]

    needed = []
    for dataset in DATASETS:
        if dataset.training_source and not ({"train", "validation"} & set(categories)):
            continue
        if not dataset.training_source and "unseen_dataset" not in categories:
            continue
        needed.append(dataset)
    missing = [
        str(dataset.cache_path(cache_dir))
        for dataset in needed
        if not dataset.cache_path(cache_dir).is_file()
    ]
    if missing:
        raise FileNotFoundError("missing evaluation caches: " + ", ".join(missing))

    metrics = []
    for dataset in needed:
        cache_path = dataset.cache_path(cache_dir)
        with TrainingCacheReader(cache_path) as reader:
            if dataset.training_source:
                entry = _checkpoint_split_entry(split_manifest, dataset.accession)
                if int(entry["read_count"]) != reader.metadata.read_count:
                    raise ValueError(f"{dataset.accession}: checkpoint/cache read count mismatch")
                if entry["source_sha256"] != reader.metadata.source_sha256:
                    raise ValueError(f"{dataset.accession}: checkpoint/cache source mismatch")
                if "train" in categories:
                    metrics.append(
                        evaluate_reader_range(
                            model,
                            reader,
                            dataset,
                            category="train",
                            start=int(entry["train_range"][0]),
                            stop=int(entry["train_range"][1]),
                            batch_reads=batch_reads,
                            max_reads=max_reads_per_split,
                            device=device,
                        )
                    )
                if "validation" in categories:
                    metrics.append(
                        evaluate_reader_range(
                            model,
                            reader,
                            dataset,
                            category="validation",
                            start=int(entry["validation_range"][0]),
                            stop=int(entry["validation_range"][1]),
                            batch_reads=batch_reads,
                            max_reads=max_reads_per_split,
                            device=device,
                        )
                    )
            else:
                metrics.append(
                    evaluate_reader_range(
                        model,
                        reader,
                        dataset,
                        category="unseen_dataset",
                        start=0,
                        stop=reader.metadata.read_count,
                        batch_reads=batch_reads,
                        max_reads=max_reads_per_split,
                        device=device,
                    )
                )
    return metrics, build_evaluation_summary(metrics)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Report theoretical bits/Q for the fixed 10-train/9-unseen split."
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--categories",
        default=",".join(EVALUATION_CATEGORIES),
        help="comma-separated train,validation,unseen_dataset; default: all",
    )
    parser.add_argument("--batch-reads", type=int, default=64)
    parser.add_argument(
        "--max-reads-per-split",
        type=int,
        default=0,
        help="per-file split limit; 0 evaluates every read",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output-json",
        type=Path,
        help="default: <checkpoint-directory>/evaluation_19.json",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    categories = tuple(
        value.strip() for value in args.categories.split(",") if value.strip()
    )
    if not categories:
        raise SystemExit("--categories must not be empty")
    if args.batch_reads <= 0 or args.batch_reads > 64:
        raise SystemExit("--batch-reads must be in [1, 64]")
    if args.max_reads_per_split < 0:
        raise SystemExit("--max-reads-per-split must be nonnegative")
    try:
        device = choose_device(args.device)
        metrics, summary = evaluate_fixed_split(
            args.checkpoint,
            cache_dir=args.cache_dir,
            categories=categories,
            batch_reads=args.batch_reads,
            max_reads_per_split=args.max_reads_per_split,
            device=device,
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    report = format_evaluation_report(metrics, summary)
    print(report)
    output_path = args.output_json or args.checkpoint.parent / "evaluation_19.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "checkpoint_sha256": sha256_file(args.checkpoint),
                "metrics": [metric.to_dict() for metric in metrics],
                "summary": summary,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
