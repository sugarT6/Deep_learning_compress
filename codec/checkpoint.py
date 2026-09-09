"""Versioned checkpoint save/load contract for the direct-quality model."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import torch

from .model import DirectQualityModelConfig, DirectQualityTransformer, feature_schema
from .training_cache import CACHE_FORMAT, CACHE_SCHEMA_VERSION


CHECKPOINT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class LoadedCheckpoint:
    model: DirectQualityTransformer
    payload: Dict[str, Any]


def cache_schema() -> Dict[str, Any]:
    return {"format": CACHE_FORMAT, "schema_version": CACHE_SCHEMA_VERSION}


def save_training_checkpoint(
    path: Union[str, Path],
    *,
    model: DirectQualityTransformer,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    global_step: int,
    data_split: Mapping[str, Any],
    sampler_config: Mapping[str, Any],
    sampler_statistics: Mapping[str, Any],
    best_validation_bits_per_quality: float,
    validation_metrics: Mapping[str, Any],
) -> None:
    """Atomically save everything needed to identify a training run."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_config": model.config.to_dict(),
        "feature_schema": feature_schema(),
        "cache_schema": cache_schema(),
        "data_split": dict(data_split),
        "sampler_config": dict(sampler_config),
        "sampler_statistics": dict(sampler_statistics),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_validation_bits_per_quality": float(
            best_validation_bits_per_quality
        ),
        "validation_metrics": dict(validation_metrics),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": (
            optimizer.state_dict() if optimizer is not None else None
        ),
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(payload, temporary_path)
        os.replace(str(temporary_path), str(path))
    except Exception:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise


def load_training_checkpoint(
    path: Union[str, Path],
    *,
    device: Union[str, torch.device] = "cpu",
) -> LoadedCheckpoint:
    """Load a checkpoint and reject incompatible model/feature/cache schemas."""

    path = Path(path)
    payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: checkpoint payload must be a mapping")
    required = {
        "checkpoint_schema_version",
        "model_config",
        "feature_schema",
        "cache_schema",
        "data_split",
        "sampler_config",
        "sampler_statistics",
        "model_state_dict",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"{path}: missing checkpoint fields: {', '.join(missing)}")
    if int(payload["checkpoint_schema_version"]) != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"{path}: unsupported checkpoint schema version")
    if payload["feature_schema"] != feature_schema():
        raise ValueError(f"{path}: feature schema does not match this implementation")
    if payload["cache_schema"] != cache_schema():
        raise ValueError(f"{path}: training-cache schema does not match")

    config = DirectQualityModelConfig.from_dict(payload["model_config"])
    model = DirectQualityTransformer(config)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device)
    return LoadedCheckpoint(model=model, payload=payload)
