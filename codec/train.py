#!/usr/bin/env python3
"""Train the minimal no-SeqArc direct-quality model from stage-A caches."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is optional
    tqdm = None

from .checkpoint import load_training_checkpoint, save_training_checkpoint
from .datasets import (
    TRAIN_DATASETS,
    UNSEEN_DATASETS,
    DirectQualityDataset,
    datasets_by_family,
)
from .evaluate import (
    DatasetMetrics, choose_device, evaluate_reader_range, summarize_metric_group,
)
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


DEFAULT_OUTPUT_DIR = Path("codec/runs/direct_quality_dataset_balanced_b256")
SELECTION_METRIC = "dataset_macro_bits_per_quality"
SAMPLER_VERSION = 2


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


def _resume_contract(config: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: value for key, value in config.items()
        if key not in ("epochs", "resume_from")
    }


class BalancedTrainingSampler:
    """Exact dataset schedules with persistent shuffled, nonoverlapping blocks."""

    def __init__(
        self,
        datasets: Sequence[DirectQualityDataset],
        cache_dir: Path,
        *,
        train_fraction: float,
        batch_reads: int,
        seed: int,
        steps_per_epoch: int = 2000,
    ) -> None:
        if not 0.0 < train_fraction < 1.0:
            raise ValueError("train_fraction must be between 0 and 1")
        if batch_reads <= 0 or batch_reads > 256:
            raise ValueError("batch_reads must be in [1, 256]")
        if steps_per_epoch <= 0:
            raise ValueError("steps_per_epoch must be positive")
        if not datasets:
            raise ValueError("at least one training dataset is required")
        if any(not dataset.training_source for dataset in datasets):
            raise ValueError("training sampler may not include unseen datasets")

        grouped = datasets_by_family(datasets)
        if sum(len(members) for members in grouped.values()) != len(datasets):
            raise ValueError("training datasets have no recognized platform families")
        self.datasets = tuple(datasets)
        self.cache_dir = Path(cache_dir)
        self.train_fraction = float(train_fraction)
        self.batch_reads = int(batch_reads)
        self.seed = int(seed)
        self.steps_per_epoch = int(steps_per_epoch)
        if len({dataset.accession for dataset in datasets}) != len(datasets):
            raise ValueError("training dataset accessions must be unique")
        self._rng = np.random.default_rng(seed)
        self._schedule = []
        self._schedule_cursor = 0
        self._epoch = 0
        self._extra_cursor = 0
        self._block_orders = {}
        self._block_cursors = {}
        self._block_rounds = {}
        self._read_counts = {dataset.accession: 0 for dataset in datasets}
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
                self._block_rounds[dataset.accession] = 0
                self._block_cursors[dataset.accession] = 0
                self._block_orders[dataset.accession] = self._block_order(dataset, 0)
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

    def _block_order(self, dataset: DirectQualityDataset, round_index: int):
        index = self.datasets.index(dataset)
        rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, index, round_index])
        )
        blocks = (
            self._train_stops[dataset.accession] + self.batch_reads - 1
        ) // self.batch_reads
        return rng.permutation(blocks).tolist()

    def _next_epoch(self) -> None:
        count = len(self.datasets)
        quotient, remainder = divmod(self.steps_per_epoch, count)
        schedule = list(range(count)) * quotient
        schedule.extend((self._extra_cursor + i) % count for i in range(remainder))
        self._extra_cursor = (self._extra_cursor + remainder) % count
        self._rng.shuffle(schedule)
        self._schedule = schedule
        self._schedule_cursor = 0
        self._epoch += 1

    def sample_batch(self) -> TrainingSample:
        if self._schedule_cursor == len(self._schedule):
            self._next_epoch()
        dataset = self.datasets[self._schedule[self._schedule_cursor]]
        accession = dataset.accession
        family = dataset.platform_family
        reader = self._readers[dataset.accession]
        train_stop = self._train_stops[dataset.accession]
        if self._block_cursors[accession] == len(self._block_orders[accession]):
            self._block_rounds[accession] += 1
            self._block_orders[accession] = self._block_order(
                dataset, self._block_rounds[accession]
            )
            self._block_cursors[accession] = 0
        block = self._block_orders[accession][self._block_cursors[accession]]
        start = block * self.batch_reads
        stop = min(start + self.batch_reads, train_stop)
        batch = reader.read_range(start, stop)
        self._block_cursors[accession] += 1
        self._schedule_cursor += 1
        self._read_counts[accession] += stop - start

        self._sample_counts[dataset.accession] += 1
        self._family_counts[family] += 1
        self._total_samples += 1
        return TrainingSample(
            dataset=dataset,
            batch=batch,
        )

    def config_dict(self) -> Dict[str, Any]:
        return {
            "strategy": "strict_dataset_balanced_shuffled_blocks",
            "version": SAMPLER_VERSION,
            "batch_source": "one_training_cache_file",
            "read_sampling": "nonoverlapping_blocks_without_replacement_keep_tail",
            "steps_per_epoch": self.steps_per_epoch,
            "remainder_policy": "rotate_across_epochs",
            "dataset_order": [d.accession for d in self.datasets],
            "train_stops": dict(self._train_stops),
            "batch_reads": self.batch_reads,
            "seed": self.seed,
            "families": {
                family: [dataset.accession for dataset in datasets]
                for family, datasets in self._by_family.items()
            },
        }

    def state_dict(self) -> Dict[str, Any]:
        return copy.deepcopy({
            "config": self.config_dict(),
            "rng_state": self._rng.bit_generator.state,
            "schedule": self._schedule,
            "schedule_cursor": self._schedule_cursor,
            "epoch": self._epoch,
            "extra_cursor": self._extra_cursor,
            "block_orders": self._block_orders,
            "block_cursors": self._block_cursors,
            "block_rounds": self._block_rounds,
            "sample_counts": self._sample_counts,
            "family_counts": self._family_counts,
            "read_counts": self._read_counts,
            "total_samples": self._total_samples,
        })

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if set(state) != set(self.state_dict()) or state["config"] != self.config_dict():
            raise ValueError("sampler state/config mismatch")
        state = copy.deepcopy(dict(state))
        for dataset in self.datasets:
            accession = dataset.accession
            if state["block_rounds"][accession] < 0:
                raise ValueError("invalid sampler block round")
            expected = self._block_order(dataset, state["block_rounds"][accession])
            if state["block_orders"][accession] != expected:
                raise ValueError("invalid sampler block permutation")
            if not 0 <= state["block_cursors"][accession] <= len(expected):
                raise ValueError("invalid sampler block cursor")
            if state["sample_counts"][accession] != (
                state["block_rounds"][accession] * len(expected)
                + state["block_cursors"][accession]
            ):
                raise ValueError("sampler selection count does not match block cursor")
        if not 0 <= state["schedule_cursor"] <= len(state["schedule"]):
            raise ValueError("invalid sampler schedule cursor")
        if any(i not in range(len(self.datasets)) for i in state["schedule"]):
            raise ValueError("invalid sampler dataset schedule")
        if state["total_samples"] != sum(state["sample_counts"].values()):
            raise ValueError("sampler total count mismatch")
        self._rng.bit_generator.state = state.pop("rng_state")
        state.pop("config")
        for key, value in state.items():
            setattr(self, "_" + key, value)

    def statistics_dict(self) -> Dict[str, Any]:
        denominator = max(self._total_samples, 1)
        unique_reads = {
            accession: min(self._read_counts[accession], stop)
            for accession, stop in self._train_stops.items()
        }
        return {
            "total_batches": self._total_samples,
            "total_reads": sum(self._read_counts.values()),
            "nominal_read_slots": self._total_samples * self.batch_reads,
            "unique_reads": sum(unique_reads.values()),
            "train_read_count": sum(self._train_stops.values()),
            "datasets": {
                accession: {
                    "count": count,
                    "proportion": count / denominator,
                    "reads": self._read_counts[accession],
                    "unique_reads": unique_reads[accession],
                    "coverage_fraction": (
                        unique_reads[accession] / self._train_stops[accession]
                    ),
                    "block_round": self._block_rounds[accession],
                    "block_cursor": self._block_cursors[accession],
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
    return metrics, {
        **summarize_metric_group(metrics),
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
    resume: Optional[Path] = None,
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
    start_epoch = 1

    with BalancedTrainingSampler(
        train_datasets,
        cache_dir,
        train_fraction=train_fraction,
        batch_reads=batch_reads,
        seed=seed,
        steps_per_epoch=steps_per_epoch,
    ) as sampler:
        split_manifest = sampler.data_split_manifest(unseen_datasets)
        validation_counts = {}
        for dataset in train_datasets:
            available = (
                sampler.reader_for(dataset.accession).metadata.read_count
                - sampler.train_stop_for(dataset.accession)
            )
            validation_counts[dataset.accession] = (
                min(validation_max_reads_per_file, available)
                if validation_max_reads_per_file else available
            )
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
            "validation_read_rule": "fixed_start_of_validation_suffix_contiguous",
            "validation_read_counts": validation_counts,
            "selection_metric": SELECTION_METRIC,
            "device": str(device),
            "seed": seed,
        }
        if resume is not None:
            loaded = load_training_checkpoint(resume, device=device)
            payload = loaded.payload
            saved_training = payload.get("training_state")
            if not payload.get("sampler_state") or not saved_training:
                raise ValueError("checkpoint lacks resumable training/sampler state")
            previous_config = saved_training["run_config"]
            if saved_training.get("version") != 1:
                raise ValueError("unsupported training state version")
            if _resume_contract(previous_config) != _resume_contract(run_config):
                raise ValueError("resume run configuration does not match checkpoint")
            if payload.get("selection_metric") != SELECTION_METRIC:
                raise ValueError("resume checkpoint selection metric mismatch")
            start_epoch = payload["epoch"] + 1
            if epochs < start_epoch:
                raise ValueError("epochs must exceed the restored completed epoch")
            sampler.load_state_dict(payload["sampler_state"])
            if (
                sampler._epoch != payload["epoch"]
                or sampler._schedule_cursor != steps_per_epoch
                or sampler._total_samples != payload["global_step"]
            ):
                raise ValueError("resume requires a completed epoch checkpoint")
            model.load_state_dict(loaded.model.state_dict())
            optimizer.load_state_dict(payload["optimizer_state_dict"])
            global_step = payload["global_step"]
            best_validation_bits = payload["best_validation_bits_per_quality"]
            # Preserve the prior best in the new, empty output directory.
            best_source = Path(resume).parent / "best.pt"
            best_payload = load_training_checkpoint(best_source).payload
            best_config = (best_payload.get("training_state") or {}).get(
                "run_config", {}
            )
            if (
                best_payload.get("selection_metric") != SELECTION_METRIC
                or best_payload["validation_metrics"][SELECTION_METRIC] != best_validation_bits
                or _resume_contract(best_config) != _resume_contract(previous_config)
            ):
                raise ValueError("resume requires the matching best.pt beside its checkpoint")
            shutil.copyfile(best_source, best_path)
            random.setstate(saved_training["python_rng"])
            np.random.set_state(saved_training["numpy_rng"])
            torch.set_rng_state(saved_training["torch_rng"].cpu())
            if device.type == "cuda":
                torch.cuda.set_rng_state_all([s.cpu() for s in saved_training["cuda_rng"]])
            run_config["resume_from"] = str(Path(resume).resolve())
        _write_json_atomic(output_dir / "run_config.json", run_config)

        for epoch in range(start_epoch, epochs + 1):
            model.train()
            epoch_nats = 0.0
            epoch_symbols = 0
            step_iter = range(1, steps_per_epoch + 1)
            if progress and tqdm is not None:
                step_iter = tqdm(
                    step_iter,
                    total=steps_per_epoch,
                    desc=f"epoch {epoch}/{epochs}",
                    unit="batch",
                    leave=True,
                )

            for step in step_iter:
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
                if (
                    progress
                    and tqdm is not None
                    and (step % 10 == 0 or step == steps_per_epoch)
                ):
                    step_iter.set_postfix(
                        train_bits=(
                            f"{epoch_nats / (epoch_symbols * math.log(2.0)):.4f}"
                        )
                    )

            _, validation_summary = evaluate_validation(
                model,
                sampler,
                max_reads_per_file=validation_max_reads_per_file,
                device=device,
            )
            validation_bits = validation_summary[SELECTION_METRIC]
            if not math.isfinite(validation_bits):
                raise ValueError("validation dataset macro must be finite")
            sampling_statistics = sampler.statistics_dict()
            train_loss = epoch_nats / epoch_symbols
            train_bits = train_loss / math.log(2.0)
            epoch_record = {
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": train_loss,
                "train_bits_per_quality": train_bits,
                "validation": validation_summary,
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
                "selection_metric": SELECTION_METRIC,
                "sampler_state": sampler.state_dict(),
                "training_state": {
                    "version": 1,
                    "run_config": run_config,
                    "python_rng": random.getstate(),
                    "numpy_rng": np.random.get_state(),
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                },
            }
            save_training_checkpoint(output_dir / "last.pt", **checkpoint_arguments)
            if improved:
                save_training_checkpoint(best_path, **checkpoint_arguments)
            if progress:
                print(
                    f"epoch={epoch} loss={train_loss:.4f} "
                    f"train_bits={train_bits:.4f} "
                    f"val_dataset_macro={validation_bits:.4f} "
                    f"val_micro={validation_summary['symbol_micro_bits_per_quality']:.4f} "
                    f"val_family_macro={validation_summary['platform_family_macro_bits_per_quality']:.4f} "
                    f"val_worst={validation_summary['worst_dataset']['bits_per_quality']:.4f} "
                    f"best_val_bits={best_validation_bits:.4f}",
                    flush=True,
                )
        _write_json_atomic(
            output_dir / "sampling_statistics.json", sampler.statistics_dict()
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
    parser.add_argument("--batch-reads", type=int, default=256)
    parser.add_argument(
        "--resume", type=Path,
        help="epoch checkpoint; use a NEW empty output dir; epochs is total target",
    )
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
            resume=args.resume,
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(f"best checkpoint: {best_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
