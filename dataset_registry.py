#!/usr/bin/env python3
"""Explicit dataset catalog for multi-platform FASTQ quality experiments.

This module owns only dataset names, groups, metadata, and file resolution.
HDF5 inspection, read splitting, batching, and sampling remain in
``sequence_residual_transformer_model.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_DATA_ROOT = Path("data")


@dataclass(frozen=True)
class DatasetFile:
    dataset: str
    platform: str
    h5_path: Path
    fastq_path: Path


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    platform: str
    h5_names: tuple[str, ...]
    fastq_relative_paths: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.h5_names) != len(self.fastq_relative_paths):
            raise ValueError(f"{self.name}: H5 and FASTQ file counts differ")


DATASETS: dict[str, DatasetSpec] = {
    "novaseq": DatasetSpec(
        name="novaseq",
        platform="Illumina",
        h5_names=(
            "subset_HG001_1.fq.gz.qual_model.h5",
            "subset_HG002_1.fq.gz.qual_model.h5",
        ),
        fastq_relative_paths=(
            "fq/NovaSeq/subset_HG001_1.fq.gz",
            "fq/NovaSeq/subset_HG002_1.fq.gz",
        ),
    ),
    "nextseq2000": DatasetSpec(
        name="nextseq2000",
        platform="Illumina",
        h5_names=(
            "subset_SRR15731087_1.fq.gz.qual_model.h5",
            "subset_SRR22228918_1.fq.gz.qual_model.h5",
        ),
        fastq_relative_paths=(
            "fq/NextSeq-2000/subset_SRR15731087_1.fq.gz",
            "fq/NextSeq-2000/subset_SRR22228918_1.fq.gz",
        ),
    ),
    "dnbseq_t7": DatasetSpec(
        name="dnbseq_t7",
        platform="BGI/MGI",
        h5_names=(
            "subset_E200029822_1_450.fq.gz.qual_model.h5",
            "subset_E200029822_1_451.fq.gz.qual_model.h5",
        ),
        fastq_relative_paths=(
            "fq/DNBSEQ-T7/subset_E200029822_1_450.fq.gz",
            "fq/DNBSEQ-T7/subset_E200029822_1_451.fq.gz",
        ),
    ),
    "mgiseq2000": DatasetSpec(
        name="mgiseq2000",
        platform="BGI/MGI",
        h5_names=(
            "subset_MGISEQ2000_NA24385_L03_1.fq.gz.qual_model.h5",
            "subset_MGISEQ2000_NA24385_L04_1.fq.gz.qual_model.h5",
        ),
        fastq_relative_paths=(
            "fq/MGISEQ-2000/subset_MGISEQ2000_NA24385_L03_1.fq.gz",
            "fq/MGISEQ-2000/subset_MGISEQ2000_NA24385_L04_1.fq.gz",
        ),
    ),
}


DATASET_GROUPS: dict[str, tuple[str, ...]] = {
    "illumina": ("novaseq", "nextseq2000"),
    "bgi_mgi": ("dnbseq_t7", "mgiseq2000"),
    "mixed": ("novaseq", "nextseq2000", "dnbseq_t7", "mgiseq2000"),
}


_ALIASES = {
    "nextseq_2000": "nextseq2000",
    "dnbseqt7": "dnbseq_t7",
    "dnbseq_t_7": "dnbseq_t7",
    "mgiseq_2000": "mgiseq2000",
    "bgi": "bgi_mgi",
    "mgi": "bgi_mgi",
    "all": "mixed",
}


def _normalize_name(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_").replace("/", "_")
    return _ALIASES.get(normalized, normalized)


def expand_dataset_selection(selection: str) -> tuple[str, ...]:
    """Expand comma-separated dataset/group names into unique dataset names."""

    if not selection.strip():
        raise ValueError("dataset selection must not be empty")

    expanded: list[str] = []
    for raw_name in selection.split(","):
        name = _normalize_name(raw_name)
        if not name:
            raise ValueError("dataset selection contains an empty name")
        if name in DATASET_GROUPS:
            candidates = DATASET_GROUPS[name]
        elif name in DATASETS:
            candidates = (name,)
        else:
            valid = sorted((*DATASETS.keys(), *DATASET_GROUPS.keys()))
            raise ValueError(f"unknown dataset/group {raw_name.strip()!r}; choose from {valid}")
        for candidate in candidates:
            if candidate not in expanded:
                expanded.append(candidate)
    return tuple(expanded)


def resolve_dataset_files(
    selection: str,
    data_root: Path = DEFAULT_DATA_ROOT,
    *,
    require_h5: bool = True,
    require_fastq: bool = False,
) -> list[DatasetFile]:
    """Resolve a selection to ordered H5/FASTQ pairs and validate paths."""

    files: list[DatasetFile] = []
    seen_h5_names: set[str] = set()
    for dataset_name in expand_dataset_selection(selection):
        spec = DATASETS[dataset_name]
        for h5_name, fastq_relative_path in zip(
            spec.h5_names,
            spec.fastq_relative_paths,
        ):
            if h5_name in seen_h5_names:
                raise ValueError(f"duplicate H5 basename in selection: {h5_name}")
            seen_h5_names.add(h5_name)
            item = DatasetFile(
                dataset=dataset_name,
                platform=spec.platform,
                h5_path=data_root / "h5" / h5_name,
                fastq_path=data_root / fastq_relative_path,
            )
            if require_h5 and not item.h5_path.is_file():
                raise FileNotFoundError(item.h5_path)
            if require_fastq and not item.fastq_path.is_file():
                raise FileNotFoundError(item.fastq_path)
            files.append(item)
    return files


def dataset_metadata(files: list[DatasetFile]) -> dict[str, object]:
    """Return checkpoint-friendly metadata for a resolved dataset selection."""

    dataset_names: list[str] = []
    for item in files:
        if item.dataset not in dataset_names:
            dataset_names.append(item.dataset)
    return {
        "dataset_names": dataset_names,
        "dataset_by_file": {item.h5_path.name: item.dataset for item in files},
        "dataset_platform_by_file": {
            item.h5_path.name: item.platform for item in files
        },
        "fastq_by_file": {
            item.h5_path.name: str(item.fastq_path) for item in files
        },
    }
