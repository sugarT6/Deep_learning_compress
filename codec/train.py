#!/usr/bin/env python3
"""Train the minimal no-SeqArc direct-quality model from stage-A caches."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import torch

from .checkpoint import save_training_checkpoint
from .datasets import (
    TRAIN_DATASETS,
    UNSEEN_DATASETS,
    DirectQualityDataset,
    datasets_by_family,
)
from .evaluate import DatasetMetrics, choose_device, evaluate_reader_range
from .model import (
    DirectQualityModelConfig,
    DirectQualityTransformer,
    fastq_batch_to_tensors,
    feature_schema,
    masked_cross_entropy,
)
from .fastq_stream import FastqBatch
from .training_cache import (
    CACHE_FORMAT,
    CACHE_SCHEMA_VERSION,
    DEFAULT_CACHE_DIR,
    TrainingCacheReader,
)


DEFAULT_OUTPUT_DIR = Path("runs/direct_quality_no_seqarc_qmer234_baseconv357_b64")


@dataclass(frozen=True)
class TrainingSample:
    dataset: DirectQualityDataset
    batch: FastqBatch


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        temporary_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(str(temporary_path), str(path))
    except Exception:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise


class BalancedTrainingSampler:
    """Uniform family-then-file sampler restricted to each file's train split."""

    def __init__(
        self,
        datasets: Sequence[DirectQualityDataset],
        cache_dir: Path,
        *,
        train_fraction: float,
        batch_reads: int,
        seed: int,
    ) -> None:
        if not 0.0 < train_fraction < 1.0:
            raise ValueError("train_fraction must be between 0 and 1")
        if batch_reads <= 0 or batch_reads > 64:
            raise ValueError("batch_reads must be in [1, 64]")
        if not datasets:
            raise ValueError("at least one training dataset is required")
        if any(not dataset.training_source for dataset in datasets):
            raise ValueError("training sampler may not include unseen datasets")

        grouped = datasets_by_family(datasets)
        if not grouped:
            raise ValueError("training datasets have no recognized platform families")
        self.datasets = tuple(datasets)
        self.cache_dir = Path(cache_dir)
        self.train_fraction = float(train_fraction)
        self.batch_reads = int(batch_reads)
        self.seed = int(seed)
        self._rng = np.random.default_rng(seed)
        self._by_family = grouped
        self._families = tuple(grouped)
        self._readers = {}
        self._train_stops = {}
        self._sample_counts = {dataset.accession: 0 for dataset in datasets}
        self._family_counts = {family: 0 for family in self._families}
        self._total_samples = 0

        missing = [
            str(dataset.cache_path(self.cache_dir))
            for dataset in datasets
            if not dataset.cache_path(self.cache_dir).is_file()
        ]
        if missing:
            raise FileNotFoundError("missing training caches: " + ", ".join(missing))
        try:
            for dataset in datasets:
                reader = TrainingCacheReader(dataset.cache_path(self.cache_dir))
                if reader.metadata.read_count < 2:
                    reader.close()
                    raise ValueError(
                        f"{dataset.accession}: cache needs at least two reads for train/validation"
                    )
                train_stop = int(math.floor(reader.metadata.read_count * train_fraction))
                train_stop = min(max(train_stop, 1), reader.metadata.read_count - 1)
                self._readers[dataset.accession] = reader
                self._train_stops[dataset.accession] = train_stop
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        for reader in self._readers.values():
            reader.close()
        self._readers.clear()

    def __enter__(self) -> "BalancedTrainingSampler":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    def reader_for(self, accession: str) -> TrainingCacheReader:
        if accession not in self._readers:
            raise KeyError(accession)
        return self._readers[accession]

    def train_stop_for(self, accession: str) -> int:
        return self._train_stops[accession]

    def sample_batch(self) -> TrainingSample:
        family = self._families[int(self._rng.integers(len(self._families)))]
        candidates = self._by_family[family]
        dataset = candidates[int(self._rng.integers(len(candidates)))]
        reader = self._readers[dataset.accession]
        train_stop = self._train_stops[dataset.accession]
        count = min(self.batch_reads, train_stop)
        maximum_start = train_stop - count
        start = int(self._rng.integers(maximum_start + 1))

        self._sample_counts[dataset.accession] += 1
        self._family_counts[family] += 1
        self._total_samples += 1
        return TrainingSample(
            dataset=dataset,
            batch=reader.read_range(start, start + count),
        )

    def config_dict(self) -> Dict[str, Any]:
        return {
            "strategy": "uniform_platform_family_then_uniform_file",
            "batch_source": "one_training_cache_file",
            "read_sampling": "uniform_contiguous_start_within_training_prefix",
            "batch_reads": self.batch_reads,
            "seed": self.seed,
            "families": {
                family: [dataset.accession for dataset in datasets]
                for family, datasets in self._by_family.items()
            },
        }

    def statistics_dict(self) -> Dict[str, Any]:
        denominator = max(self._total_samples, 1)
        return {
            "total_batches": self._total_samples,
            "datasets": {
                accession: {
                    "count": count,
                    "proportion": count / denominator,
                }
                for accession, count in self._sample_counts.items()
            },
            "families": {
                family: {
                    "count": count,
                    "proportion": count / denominator,
                }
                for family, count in self._family_counts.items()
            },
        }

    def data_split_manifest(
        self, unseen_datasets: Sequence[DirectQualityDataset]
    ) -> Dict[str, Any]:
        entries = {}
        for dataset in self.datasets:
            reader = self._readers[dataset.accession]
            train_stop = self._train_stops[dataset.accession]
            entries[dataset.accession] = {
                "platform_family": dataset.platform_family,
                "cache_filename": reader.cache_path.name,
                "source_fastq_basename": reader.metadata.source_fastq_basename,
                "source_sha256": reader.metadata.source_sha256,
                "read_count": reader.metadata.read_count,
                "train_range": [0, train_stop],
                "validation_range": [train_stop, reader.metadata.read_count],
            }
        return {
            "strategy": "contiguous_prefix_train_suffix_validation",
            "train_fraction": self.train_fraction,
            "datasets": entries,
            "unseen_datasets": [dataset.accession for dataset in unseen_datasets],
            "unseen_instrument_datasets": [
                dataset.accession for dataset in unseen_datasets if dataset.unseen_instrument
            ],
            "checkpoint_selection_uses": "validation_only",
            "unseen_used_for_checkpoint_selection": False,
        }


def evaluate_validation(
    model: DirectQualityTransformer,
    sampler: BalancedTrainingSampler,
    *,
    max_reads_per_file: int,
    device: torch.device,
) -> Tuple[Sequence[DatasetMetrics], Dict[str, Any]]:
    metrics = []
    for dataset in sampler.datasets:
        reader = sampler.reader_for(dataset.accession)
        metrics.append(
            evaluate_reader_range(
                model,
                reader,
                dataset,
                category="validation",
                start=sampler.train_stop_for(dataset.accession),
                stop=reader.metadata.read_count,
                batch_reads=sampler.batch_reads,
                max_reads=max_reads_per_file,
                device=device,
            )
        )
    total_bits = sum(metric.total_bits for metric in metrics)
    total_symbols = sum(metric.symbols for metric in metrics)
    return metrics, {
        "symbol_micro_bits_per_quality": total_bits / total_symbols,
        "dataset_macro_bits_per_quality": sum(
            metric.bits_per_quality for metric in metrics
        )
        / len(metrics),
        "per_dataset": {
            metric.accession: metric.to_dict() for metric in metrics
        },
    }


def run_training(
    *,
    cache_dir: Path,
    output_dir: Path,
    model_config: DirectQualityModelConfig,
    epochs: int,
    steps_per_epoch: int,
    batch_reads: int,
    train_fraction: float,
    validation_max_reads_per_file: int,
    learning_rate: float,
    weight_decay: float,
    grad_clip: float,
    device: torch.device,
    seed: int,
    train_datasets: Sequence[DirectQualityDataset] = TRAIN_DATASETS,
    unseen_datasets: Sequence[DirectQualityDataset] = UNSEEN_DATASETS,
    progress: bool = True,
) -> Path:
    """Train and select checkpoints using only train-source validation reads."""

    if epochs <= 0 or steps_per_epoch <= 0:
        raise ValueError("epochs and steps_per_epoch must be positive")
    if validation_max_reads_per_file < 0:
        raise ValueError("validation_max_reads_per_file must be nonnegative")
    if learning_rate <= 0 or weight_decay < 0 or grad_clip <= 0:
        raise ValueError("invalid optimizer configuration")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"{output_dir}: output directory is not empty; choose a new run directory"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    model = DirectQualityTransformer(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    best_validation_bits = float("inf")
    global_step = 0
    best_path = output_dir / "best.pt"

    with BalancedTrainingSampler(
        train_datasets,
        cache_dir,
        train_fraction=train_fraction,
        batch_reads=batch_reads,
        seed=seed,
    ) as sampler:
        split_manifest = sampler.data_split_manifest(unseen_datasets)
        run_config = {
            "model_config": model_config.to_dict(),
            "feature_schema": feature_schema(),
            "cache_schema": {
                "format": CACHE_FORMAT,
                "schema_version": CACHE_SCHEMA_VERSION,
            },
            "data_split": split_manifest,
            "sampler_config": sampler.config_dict(),
            "optimizer": {
                "name": "AdamW",
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "grad_clip": grad_clip,
            },
            "epochs": epochs,
            "steps_per_epoch": steps_per_epoch,
            "validation_max_reads_per_file": validation_max_reads_per_file,
            "device": str(device),
            "seed": seed,
        }
        _write_json_atomic(output_dir / "run_config.json", run_config)

        for epoch in range(1, epochs + 1):
            model.train()
            epoch_nats = 0.0
            epoch_symbols = 0
            for step in range(1, steps_per_epoch + 1):
                sample = sampler.sample_batch()
                tensors = fastq_batch_to_tensors(sample.batch, device)
                optimizer.zero_grad(set_to_none=True)
                logits = model.forward_full(**tensors)
                loss = masked_cross_entropy(
                    logits, tensors["qualities"], tensors["active_mask"]
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

                symbols = int(tensors["active_mask"].sum().item())
                epoch_nats += float(loss.item()) * symbols
                epoch_symbols += symbols
                global_step += 1
                if progress:
                    print(
                        f"epoch={epoch} step={step}/{steps_per_epoch} "
                        f"dataset={sample.dataset.accession} "
                        f"loss_bits/Q={loss.item() / math.log(2.0):.6f}",
                        flush=True,
                    )

            _, validation_summary = evaluate_validation(
                model,
                sampler,
                max_reads_per_file=validation_max_reads_per_file,
                device=device,
            )
            validation_bits = validation_summary["symbol_micro_bits_per_quality"]
            sampling_statistics = sampler.statistics_dict()
            _write_json_atomic(
                output_dir / "sampling_statistics.json", sampling_statistics
            )
            epoch_record = {
                "epoch": epoch,
                "global_step": global_step,
                "train_bits_per_quality": epoch_nats
                / (epoch_symbols * math.log(2.0)),
                "validation": validation_summary,
                "sampling_statistics": sampling_statistics,
            }
            with (output_dir / "training_log.jsonl").open(
                "a", encoding="utf-8"
            ) as handle:
                handle.write(json.dumps(epoch_record, sort_keys=True) + "\n")

            improved = validation_bits < best_validation_bits
            if improved:
                best_validation_bits = validation_bits
            checkpoint_arguments = {
                "model": model,
                "optimizer": optimizer,
                "epoch": epoch,
                "global_step": global_step,
                "data_split": split_manifest,
                "sampler_config": sampler.config_dict(),
                "sampler_statistics": sampling_statistics,
                "best_validation_bits_per_quality": best_validation_bits,
                "validation_metrics": validation_summary,
            }
            save_training_checkpoint(output_dir / "last.pt", **checkpoint_arguments)
            if improved:
                save_training_checkpoint(best_path, **checkpoint_arguments)
            if progress:
                print(
                    f"epoch={epoch} train_bits/Q="
                    f"{epoch_record['train_bits_per_quality']:.6f} "
                    f"validation_bits/Q={validation_bits:.6f} "
                    f"best={best_validation_bits:.6f}",
                    flush=True,
                )
    return best_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train the 42-class direct-quality model from the fixed ten training "
            "caches. Unseen datasets are never used for checkpoint selection."
        )
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--steps-per-epoch", type=int, default=2000)
    parser.add_argument("--batch-reads", type=int, default=64)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--validation-max-reads-per-file", type=int, default=5000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--feedforward-dim", type=int, default=512)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        device = choose_device(args.device)
        config = DirectQualityModelConfig(
            d_model=args.d_model,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            feedforward_dim=args.feedforward_dim,
            context_length=args.context_length,
            dropout=args.dropout,
        )
        best_path = run_training(
            cache_dir=args.cache_dir,
            output_dir=args.output_dir,
            model_config=config,
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            batch_reads=args.batch_reads,
            train_fraction=args.train_fraction,
            validation_max_reads_per_file=args.validation_max_reads_per_file,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            device=device,
            seed=args.seed,
            progress=not args.no_progress,
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(f"best checkpoint: {best_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
