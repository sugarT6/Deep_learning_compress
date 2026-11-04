"""Initialization and paid-description validation for block-conditioned training.

These helpers deliberately do not import ``train``.  The legacy training and
evaluation paths keep their original sampler, checkpoint, and metric contracts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch

from .block_features import BLOCK_FEATURE_BYTES, BLOCK_FEATURE_DIM, CacheBlockFeatures
from .datasets import DirectQualityDataset
from .evaluate import DatasetMetrics, EVALUATION_CATEGORIES, summarize_metric_group
from .fastq_stream import MAX_BATCH_READS
from .model import (
    DirectQualityModelConfig,
    DirectQualityTransformer,
    fastq_batch_to_tensors,
    theoretical_bits,
)
from .probability_quantization import logits_to_cdfs
from .training_cache import TrainingCacheReader


BLOCK_SELECTION_METRIC = "dataset_macro_quantized_Q_plus_block_side_bits_per_quality"
_BLOCK_CONFIG_FIELDS = (
    "block_quality_features", "block_reads", "block_feature_mode", "block_conditioning_dim",
)
_CONDITIONING_PREFIX = "block_conditioning."


def transfer_block_initialization(
    model: DirectQualityTransformer,
    loaded_model: DirectQualityTransformer,
) -> None:
    """Copy an identical block model or add only identity-start block conditioning.

    Ordinary parent models must agree in every non-block configuration field and
    tensor.  The new conditioning module retains its freshly initialized hidden
    projection and zero affine output; unrestricted ``strict=False`` is avoided.
    """

    if not model.config.block_quality_features:
        raise ValueError("block initialization requires a block-conditioned target")
    if loaded_model.config.block_quality_features:
        if loaded_model.config != model.config:
            raise ValueError("block init-from requires identical model configurations")
        model.load_state_dict(loaded_model.state_dict(), strict=True)
        return

    source_config = asdict(loaded_model.config)
    target_config = asdict(model.config)
    for name in _BLOCK_CONFIG_FIELDS:
        source_config.pop(name)
        target_config.pop(name)
    if source_config != target_config:
        raise ValueError("init-from may only add block quality conditioning")

    source = loaded_model.state_dict()
    target = model.state_dict()
    new_keys = {key for key in target if key.startswith(_CONDITIONING_PREFIX)}
    if (not new_keys or set(source) != set(target) - new_keys
            or any(key.startswith(_CONDITIONING_PREFIX) for key in source)):
        raise ValueError("init-from model tensor keys do not match the ordinary backbone")
    for name, value in source.items():
        if value.shape != target[name].shape or value.dtype != target[name].dtype:
            raise ValueError(f"init-from model tensor shape/dtype mismatch: {name}")
    affine = model.block_conditioning.encoder[-1]
    if torch.count_nonzero(affine.weight).item() or torch.count_nonzero(affine.bias).item():
        raise ValueError("new block conditioner must have zero-initialized affine output")
    merged = dict(target)
    merged.update(source)
    model.load_state_dict(merged, strict=True)


def _dataset_splits(manifest: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    if not isinstance(manifest, Mapping) or not isinstance(manifest.get("datasets"), Mapping):
        raise ValueError(f"{name} lacks a dataset split manifest")
    return manifest["datasets"]


def _train_bounds(entry: Mapping[str, Any], accession: str) -> tuple:
    bounds = entry.get("train_range")
    if (not isinstance(bounds, (list, tuple)) or len(bounds) != 2
            or any(type(value) is not int for value in bounds)
            or not 0 <= bounds[0] < bounds[1] <= entry["read_count"]):
        raise ValueError(f"{accession}: invalid parent/current training range")
    return tuple(bounds)


def validate_parent_split(
    parent_payload: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
) -> None:
    """Reject cache changes or parent training Q entering the new validation split."""

    parent = _dataset_splits(parent_payload.get("data_split"), "init-from checkpoint")
    current = _dataset_splits(split_manifest, "new run")
    for accession, old in parent.items():
        if accession not in current:
            raise ValueError(f"{accession}: parent training source is absent from the new split")
        new = current[accession]
        if not isinstance(old, Mapping) or not isinstance(new, Mapping):
            raise ValueError(f"{accession}: invalid dataset split entry")
        for entry in (old, new):
            if (not isinstance(entry.get("source_sha256"), str)
                    or not entry["source_sha256"]
                    or type(entry.get("read_count")) is not int
                    or entry["read_count"] < 2):
                raise ValueError(f"{accession}: split lacks source identity/read count")
        if (old["source_sha256"] != new["source_sha256"]
                or old["read_count"] != new["read_count"]):
            raise ValueError(f"{accession}: init-from source SHA/read count mismatch")
        old_start, old_stop = _train_bounds(old, accession)
        new_start, new_stop = _train_bounds(new, accession)
        validation = new.get("validation_range")
        if validation != [new_stop, new["read_count"]]:
            raise ValueError(f"{accession}: new validation must follow the training prefix")
        if old_start < new_start or old_stop > new_stop:
            raise ValueError(f"{accession}: parent training reads would enter new validation")


def initialize_block_model(
    model: DirectQualityTransformer,
    parent_payload: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
    *,
    loaded_model: Optional[DirectQualityTransformer] = None,
) -> None:
    """Validate parent data provenance before transferring its model weights."""

    validate_parent_split(parent_payload, split_manifest)
    if loaded_model is None:
        config = DirectQualityModelConfig.from_dict(parent_payload["model_config"])
        # Constructing a temporary loader must not change the new run's RNG.
        with torch.random.fork_rng(devices=[]):
            loaded_model = DirectQualityTransformer(config)
        loaded_model.load_state_dict(parent_payload["model_state_dict"], strict=True)
    elif (DirectQualityModelConfig.from_dict(parent_payload["model_config"])
          != loaded_model.config):
        raise ValueError("loaded parent model does not match checkpoint configuration")
    transfer_block_initialization(model, loaded_model)


@dataclass(frozen=True)
class BlockDatasetMetrics(DatasetMetrics):
    quantized_bits: float
    block_count: int
    block_side_bits: int
    quantized_Q_plus_block_side_bits_per_quality: float

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["quantized_bits_per_quality"] = self.quantized_bits / self.symbols
        return result


def evaluate_block_reader_range(
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
) -> BlockDatasetMetrics:
    """Score whole fixed blocks, charging every descriptor once, including tails.

    A positive read limit rounds *up* to finish the last original block.  The
    returned read/symbol counts report that actual scope.  Pure float Q metrics
    retain the legacy meaning; checkpoint selection uses floor-v2 Q plus side bits.
    """

    if not model.config.block_quality_features:
        raise ValueError("block evaluation requires a block-conditioned model")
    if category not in EVALUATION_CATEGORIES:
        raise ValueError(f"unknown evaluation category {category!r}")
    if not 0 < batch_reads <= MAX_BATCH_READS:
        raise ValueError(f"batch_reads must be in [1, {MAX_BATCH_READS}]")
    if max_reads < 0:
        raise ValueError("max_reads must be nonnegative")
    count = int(reader.metadata.read_count)
    block_reads = model.config.block_reads
    if (not 0 <= start < stop <= count or start % block_reads
            or (stop != count and stop % block_reads)):
        raise ValueError("block evaluation range must contain complete original blocks")
    if max_reads:
        requested_stop = start + max_reads
        stop = min(stop, ((requested_stop + block_reads - 1) // block_reads) * block_reads)
    features = CacheBlockFeatures(reader, block_reads, split_start=start, split_stop=stop)
    was_training = model.training
    model.eval()
    total_bits = 0.0
    quantized_bits = 0.0
    total_symbols = 0
    evaluated_reads = 0
    blocks = 0
    try:
        with torch.inference_mode():
            for block_start in range(start, stop, block_reads):
                block_stop = min(block_start + block_reads, stop)
                blocks += 1
                for query_start in range(block_start, block_stop, batch_reads):
                    query_stop = min(query_start + batch_reads, block_stop)
                    batch = reader.read_range(query_start, query_stop)
                    evaluated_reads += batch.read_count
                    tensors = fastq_batch_to_tensors(batch, device)
                    active = tensors["active_mask"]
                    if not torch.any(active):
                        continue
                    condition = (
                        np.zeros((batch.read_count, BLOCK_FEATURE_DIM), dtype=np.float32)
                        if model.config.block_feature_mode == "constant"
                        else features.for_read_indices(batch.read_indices)
                    )
                    logits = model.forward_full(
                        **tensors,
                        block_features=torch.as_tensor(condition, dtype=torch.float32, device=device),
                    )
                    bits, symbols = theoretical_bits(logits, tensors["qualities"], active)
                    total_bits += bits
                    total_symbols += symbols
                    cdf = logits_to_cdfs(
                        logits[active].detach().cpu().numpy(), total=65536, version=2,
                    )
                    target = tensors["qualities"][active].cpu().numpy()
                    row = np.arange(len(target))
                    frequency = cdf[row, target + 1] - cdf[row, target]
                    quantized_bits += float(-np.log2(frequency / cdf[:, -1]).sum())
    finally:
        model.train(was_training)
    if total_symbols == 0:
        raise ValueError(f"{dataset.accession}: evaluation split has no quality symbols")
    side_bits = blocks * BLOCK_FEATURE_BYTES * 8 if model.config.block_feature_mode == "features" else 0
    return BlockDatasetMetrics(
        accession=dataset.accession,
        platform_family=dataset.platform_family,
        category=category,
        unseen_instrument=dataset.unseen_instrument,
        reads=evaluated_reads,
        symbols=total_symbols,
        total_bits=total_bits,
        bits_per_quality=total_bits / total_symbols,
        quantized_bits=quantized_bits,
        block_count=blocks,
        block_side_bits=side_bits,
        quantized_Q_plus_block_side_bits_per_quality=(quantized_bits + side_bits) / total_symbols,
    )


def summarize_block_metrics(metrics: Sequence[BlockDatasetMetrics]) -> Dict[str, Any]:
    """Keep legacy pure-float summaries and add paid quantized selection metrics."""

    summary = summarize_metric_group(metrics)
    summary["per_dataset"] = {metric.accession: metric.to_dict() for metric in metrics}
    summary["block_count"] = sum(metric.block_count for metric in metrics)
    summary["block_side_bits"] = sum(metric.block_side_bits for metric in metrics)
    summary["quantized_Q_bits"] = sum(metric.quantized_bits for metric in metrics)
    summary[BLOCK_SELECTION_METRIC] = (
        sum(metric.quantized_Q_plus_block_side_bits_per_quality for metric in metrics) / len(metrics)
        if metrics else None
    )
    summary["dataset_macro_quantized_Q_bits_per_quality"] = (
        sum(metric.quantized_bits / metric.symbols for metric in metrics) / len(metrics)
        if metrics else None
    )
    symbols = sum(metric.symbols for metric in metrics)
    summary["symbol_micro_quantized_Q_plus_block_side_bits_per_quality"] = (
        (summary["quantized_Q_bits"] + summary["block_side_bits"]) / symbols if symbols else None
    )
    return summary
