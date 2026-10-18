#!/usr/bin/env python3
"""Train the minimal no-SeqArc direct-quality model from stage-A caches."""

from __future__ import annotations

import argparse
import copy
import hashlib
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
    EXPANDED_FAMILY_WEIGHTS,
    EXPANDED_TRAIN_DATASETS,
    EXPANDED_UNSEEN_DATASETS,
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
from .training_control import TrainingControl, TrainingControlConfig
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
        if key not in ("epochs", "resume_from", "training_control")
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
        family_weights: Optional[Mapping[str, int]] = None,
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
        if family_weights is not None:
            if set(family_weights) != set(grouped) or any(
                type(weight) is not int or weight <= 0
                for weight in family_weights.values()
            ):
                raise ValueError("family weights must be positive integers for every training family")
        self.family_weights = dict(family_weights) if family_weights is not None else None
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
        if self.family_weights is not None:
            # A persistent integer-weight cycle fixes family exposure even when
            # steps_per_epoch is not divisible by the sum of the weights.
            cycle = [
                family for family in self._families
                for _ in range(self.family_weights[family])
            ]
            quotient, remainder = divmod(self.steps_per_epoch, len(cycle))
            offset = (self._epoch * self.steps_per_epoch) % len(cycle)
            extras = [cycle[(offset + i) % len(cycle)] for i in range(remainder)]
            schedule = []
            for family in self._families:
                members = [
                    i for i, dataset in enumerate(self.datasets)
                    if dataset.platform_family == family
                ]
                slots = quotient * self.family_weights[family] + extras.count(family)
                per_file, extra = divmod(slots, len(members))
                cursor = self._family_counts[family] % len(members)
                schedule.extend(members * per_file)
                schedule.extend(members[(cursor + i) % len(members)] for i in range(extra))
            self._rng.shuffle(schedule)
            self._schedule = schedule
            self._schedule_cursor = 0
            self._epoch += 1
            return
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
        config = {
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
        if self.family_weights is not None:
            config.update({
                "strategy": "fixed_family_weights_equal_files_shuffled_blocks",
                "version": 3,
                "family_weights": dict(self.family_weights),
                "remainder_policy": "persistent_family_cycle_and_within_family_rotation",
            })
        return config

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

    def corpus_statistics_dict(self) -> Dict[str, Any]:
        entries = {}
        for dataset in self.datasets:
            reader = self.reader_for(dataset.accession)
            stop = self.train_stop_for(dataset.accession)
            entries[dataset.accession] = {
                "train_reads": stop,
                "validation_reads": reader.metadata.read_count - stop,
                "train_quality_tokens": reader.count_symbols(0, stop),
                "validation_quality_tokens": reader.count_symbols(stop, reader.metadata.read_count),
            }
        return {
            "training_files": len(self.datasets),
            **{key: sum(entry[key] for entry in entries.values()) for key in (
                "train_reads", "validation_reads", "train_quality_tokens", "validation_quality_tokens",
            )},
            "datasets": entries,
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
    init_from: Optional[Path] = None,
    family_weights: Optional[Mapping[str, int]] = None,
    control_config: Optional[TrainingControlConfig] = None,
) -> Path:
    """Train and select checkpoints using only train-source validation reads."""

    if epochs <= 0 or steps_per_epoch <= 0:
        raise ValueError("epochs and steps_per_epoch must be positive")
    if resume is not None and init_from is not None:
        raise ValueError("resume and init_from are mutually exclusive")
    if {d.accession for d in train_datasets} & {d.accession for d in unseen_datasets}:
        raise ValueError("training and unseen datasets must not overlap")
    if validation_max_reads_per_file < 0:
        raise ValueError("validation_max_reads_per_file must be nonnegative")
    if (not all(math.isfinite(value) for value in (learning_rate, weight_decay, grad_clip))
            or learning_rate <= 0 or weight_decay < 0 or grad_clip <= 0):
        raise ValueError("invalid optimizer configuration")
    control_config = control_config or TrainingControlConfig()
    if control_config.lr_scheduler == "plateau" and control_config.min_lr > learning_rate:
        raise ValueError("min_lr must not exceed learning_rate")

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
    training_quality_tokens = 0
    training_tokens_by_dataset = {dataset.accession: 0 for dataset in train_datasets}
    best_path = output_dir / "best.pt"
    start_epoch = 1
    restored_control_state = None
    stopped_early = False

    with BalancedTrainingSampler(
        train_datasets,
        cache_dir,
        train_fraction=train_fraction,
        batch_reads=batch_reads,
        seed=seed,
        steps_per_epoch=steps_per_epoch,
        family_weights=family_weights,
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
            "training_control": control_config.to_dict(),
        }
        if init_from is not None:
            # Loading constructs a temporary CPU model. Preserve RNG draws for
            # this run's dropout, independently of how its weights were loaded.
            with torch.random.fork_rng(devices=[]):
                loaded = load_training_checkpoint(init_from)
            if loaded.model.config != model_config:
                raise ValueError("init-from model configuration does not match requested model")
            model.load_state_dict(loaded.model.state_dict(), strict=True)
            digest = hashlib.sha256()
            with Path(init_from).open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            run_config["initialization"] = {
                "method": "weights_only",
                "checkpoint": str(Path(init_from).resolve()),
                "sha256": digest.hexdigest(),
                "source_global_step": loaded.payload.get("global_step"),
            }
        if resume is not None:
            loaded = load_training_checkpoint(resume, device=device)
            payload = loaded.payload
            saved_training = payload.get("training_state")
            if not payload.get("sampler_state") or not saved_training:
                raise ValueError("checkpoint lacks resumable training/sampler state")
            previous_config = saved_training["run_config"]
            if "initialization" in previous_config:
                run_config["initialization"] = copy.deepcopy(previous_config["initialization"])
            if saved_training.get("version") not in (1, 2):
                raise ValueError("unsupported training state version")
            if _resume_contract(previous_config) != _resume_contract(run_config):
                raise ValueError("resume run configuration does not match checkpoint")
            if saved_training["version"] == 2:
                if previous_config.get("training_control") != control_config.to_dict():
                    raise ValueError("resume training controls do not match checkpoint")
                restored_control_state = saved_training.get("control_state")
                if restored_control_state is None:
                    raise ValueError("checkpoint lacks training control state")
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
            # Historical checkpoints did not count tokens; a partial count
            # must not be presented as their cumulative training total.
            training_quality_tokens = saved_training.get("training_quality_tokens")
            training_tokens_by_dataset = saved_training.get("training_tokens_by_dataset")
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
        control = TrainingControl(optimizer, control_config, best_bits=best_validation_bits)
        if restored_control_state is not None:
            control.load_state_dict(restored_control_state)
        _write_json_atomic(output_dir / "run_config.json", run_config)
        corpus = sampler.corpus_statistics_dict()
        _write_json_atomic(output_dir / "training_corpus.json", corpus)
        if progress:
            print(
                f"training_files={len(train_datasets)} train_reads={corpus['train_reads']} "
                f"train_quality_tokens={corpus['train_quality_tokens']} "
                f"sampler={sampler.config_dict()['strategy']} "
                f"start_epoch={start_epoch} target_epoch={epochs}",
                flush=True,
            )

        for epoch in range(start_epoch, epochs + 1):
            model.train()
            epoch_learning_rate = optimizer.param_groups[0]["lr"]
            epoch_nats = 0.0
            epoch_symbols = 0
            epoch_tokens_by_dataset = {dataset.accession: 0 for dataset in train_datasets}
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
                epoch_tokens_by_dataset[sample.dataset.accession] += symbols
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
            stopped_early = control.step(validation_bits)
            sampling_statistics = sampler.statistics_dict()
            if training_quality_tokens is not None:
                training_quality_tokens += epoch_symbols
            if training_tokens_by_dataset is not None:
                for accession, tokens in epoch_tokens_by_dataset.items():
                    training_tokens_by_dataset[accession] += tokens
            train_loss = epoch_nats / epoch_symbols
            train_bits = train_loss / math.log(2.0)
            epoch_record = {
                "epoch": epoch,
                "global_step": global_step,
                "epoch_training_quality_tokens": epoch_symbols,
                "training_quality_tokens": training_quality_tokens,
                "epoch_tokens_by_dataset": epoch_tokens_by_dataset,
                "training_tokens_by_dataset": training_tokens_by_dataset,
                "train_loss": train_loss,
                "train_bits_per_quality": train_bits,
                "validation": validation_summary,
                "learning_rate": epoch_learning_rate,
                "next_learning_rate": optimizer.param_groups[0]["lr"],
                "early_stopping_bad_epochs": control.bad_epochs,
                "stopped_early": stopped_early,
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
                    "version": 2,
                    "training_quality_tokens": training_quality_tokens,
                    "training_tokens_by_dataset": training_tokens_by_dataset,
                    "control_state": control.state_dict(),
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
                    f"lr={epoch_learning_rate:.6g} next_lr={optimizer.param_groups[0]['lr']:.6g}",
                    flush=True,
                )
            if stopped_early:
                if progress:
                    print(f"early stopping after epoch {epoch}", flush=True)
                break
        _write_json_atomic(
            output_dir / "sampling_statistics.json", sampler.statistics_dict()
        )
        _write_json_atomic(output_dir / "training_summary.json", {
            "start_epoch": start_epoch, "completed_epoch": epoch,
            "requested_epochs": epochs, "global_step": global_step,
            "training_quality_tokens": training_quality_tokens,
            "training_tokens_by_dataset": training_tokens_by_dataset,
            "training_corpus_quality_tokens": corpus["train_quality_tokens"],
            "stopped_early": stopped_early,
            "best_validation_bits_per_quality": best_validation_bits,
            "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "final_learning_rate": optimizer.param_groups[0]["lr"],
        })
    return best_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train the 42-class direct-quality model from an explicit cache split. "
            "Unseen datasets are never used for checkpoint selection."
        )
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--training-set", choices=("original10", "expanded20"), default="original10",
        help="expanded20 adds ten audited human sources with family weights 3:5:1:1",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--steps-per-epoch", type=int, default=2000)
    parser.add_argument("--batch-reads", type=int, default=256)
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument(
        "--resume", type=Path,
        help="epoch checkpoint; use a NEW empty output dir; epochs is total target",
    )
    initialization.add_argument(
        "--init-from", type=Path,
        help="initialize weights only; reset optimizer, sampler, controls and step count for a new data split",
    )
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--validation-max-reads-per-file", type=int, default=5000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lr-scheduler", choices=("none", "plateau"), default="none")
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--lr-patience", type=int, default=2,
                        help="plateau epochs tolerated before reducing LR on the next non-improvement")
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--early-stopping-patience", type=int, default=0,
                        help="stop after this many epochs without a min-delta improvement; 0 disables")
    parser.add_argument("--min-delta", type=float, default=1e-4,
                        help="absolute macro BPQ improvement for scheduling/stopping, not best.pt selection")
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
            init_from=args.init_from,
            train_datasets=EXPANDED_TRAIN_DATASETS if args.training_set == "expanded20" else TRAIN_DATASETS,
            unseen_datasets=EXPANDED_UNSEEN_DATASETS if args.training_set == "expanded20" else UNSEEN_DATASETS,
            family_weights=EXPANDED_FAMILY_WEIGHTS if args.training_set == "expanded20" else None,
            control_config=TrainingControlConfig(
                lr_scheduler=args.lr_scheduler, lr_factor=args.lr_factor,
                lr_patience=args.lr_patience, min_lr=args.min_lr,
                early_stopping_patience=args.early_stopping_patience,
                min_delta=args.min_delta,
            ),
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(f"best checkpoint: {best_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
