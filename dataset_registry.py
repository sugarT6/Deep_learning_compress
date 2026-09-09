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
    h5_relative_paths: tuple[str, ...]
    fastq_relative_paths: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.h5_relative_paths) != len(self.fastq_relative_paths):
            raise ValueError(f"{self.name}: H5 and FASTQ file counts differ")


DATASETS: dict[str, DatasetSpec] = {
    "novaseq": DatasetSpec(
        name="novaseq",
        platform="Illumina",
        h5_relative_paths=(
            "h5/subset_HG001_1.fq.gz.qual_model.h5",
            "h5/subset_HG002_1.fq.gz.qual_model.h5",
            "h5/subset_HG003_1.fq.gz.qual_model.h5",
            "h5/subset_NA12891.novaseq_1.fq.gz.qual_model.h5",
        ),
        fastq_relative_paths=(
            "fq/NovaSeq/subset_HG001_1.fq.gz",
            "fq/NovaSeq/subset_HG002_1.fq.gz",
            "fq/NovaSeq/subset_HG003_1.fq.gz",
            "fq/subset_NA12891.novaseq_1.fq.gz",
        ),
    ),
    "novaseq_hg": DatasetSpec(
        name="novaseq_hg",
        platform="Illumina",
        h5_relative_paths=(
            "h5/subset_HG001_1.fq.gz.qual_model.h5",
            "h5/subset_HG002_1.fq.gz.qual_model.h5",
            "h5/subset_HG003_1.fq.gz.qual_model.h5",
        ),
        fastq_relative_paths=(
            "fq/NovaSeq/subset_HG001_1.fq.gz",
            "fq/NovaSeq/subset_HG002_1.fq.gz",
            "fq/NovaSeq/subset_HG003_1.fq.gz",
        ),
    ),
    "nextseq2000": DatasetSpec(
        name="nextseq2000",
        platform="Illumina",
        h5_relative_paths=(
            "h5/subset_SRR15731087_1.fq.gz.qual_model.h5",
            "h5/subset_SRR22228918_1.fq.gz.qual_model.h5",
            "h5/subset_SRR15731080_1.fq.gz.qual_model.h5",
        ),
        fastq_relative_paths=(
            "fq/NextSeq-2000/subset_SRR15731087_1.fq.gz",
            "fq/NextSeq-2000/subset_SRR22228918_1.fq.gz",
            "fq/NextSeq-2000/subset_SRR15731080_1.fq.gz",
        ),
    ),
    "dnbseq_t7": DatasetSpec(
        name="dnbseq_t7",
        platform="BGI/MGI",
        h5_relative_paths=(
            "h5/subset_E200029822_1_450.fq.gz.qual_model.h5",
            "h5/subset_E200029822_1_451.fq.gz.qual_model.h5",
        ),
        fastq_relative_paths=(
            "fq/DNBSEQ-T7/subset_E200029822_1_450.fq.gz",
            "fq/DNBSEQ-T7/subset_E200029822_1_451.fq.gz",
        ),
    ),
    "mgiseq2000": DatasetSpec(
        name="mgiseq2000",
        platform="BGI/MGI",
        h5_relative_paths=(
            "h5/subset_MGISEQ2000_NA24385_L03_1.fq.gz.qual_model.h5",
            "h5/subset_MGISEQ2000_NA24385_L04_1.fq.gz.qual_model.h5",
        ),
        fastq_relative_paths=(
            "fq/MGISEQ-2000/subset_MGISEQ2000_NA24385_L03_1.fq.gz",
            "fq/MGISEQ-2000/subset_MGISEQ2000_NA24385_L04_1.fq.gz",
        ),
    ),
    "matrix_dnbseq_t7_train": DatasetSpec(
        name="matrix_dnbseq_t7_train",
        platform="DNBSEQ-T7",
        h5_relative_paths=(
            "matrix/DNBSEQ-T7/subset_ERR15766980_1.500k.fastq.gz.qual_model.h5",
            "matrix/DNBSEQ-T7/subset_SRR30041373_1.500k.fastq.gz.qual_model.h5",
            "matrix/DNBSEQ-T7/subset_SRR30917699_1.500k.fastq.gz.qual_model.h5",
        ),
        fastq_relative_paths=(
            "matrix/sub_DNBSEQ-T7/subset_ERR15766980_1.500k.fastq.gz",
            "matrix/sub_DNBSEQ-T7/subset_SRR30041373_1.500k.fastq.gz",
            "matrix/sub_DNBSEQ-T7/subset_SRR30917699_1.500k.fastq.gz",
        ),
    ),
    "matrix_dnbseq_t7_holdout": DatasetSpec(
        name="matrix_dnbseq_t7_holdout",
        platform="DNBSEQ-T7",
        h5_relative_paths=(
            "matrix/DNBSEQ-T7/subset_SRR31204453_1.500k.fastq.gz.qual_model.h5",
            "matrix/DNBSEQ-T7/subset_SRR32293491_1.500k.fastq.gz.qual_model.h5",
            "matrix/DNBSEQ-T7/subset_ERR15801889_1.500k.fastq.gz.qual_model.h5",
        ),
        fastq_relative_paths=(
            "matrix/sub_DNBSEQ-T7/subset_SRR31204453_1.500k.fastq.gz",
            "matrix/sub_DNBSEQ-T7/subset_SRR32293491_1.500k.fastq.gz",
            "matrix/sub_DNBSEQ-T7/subset_ERR15801889_1.500k.fastq.gz",
        ),
    ),
    "matrix_nextseq2000_train": DatasetSpec(
        name="matrix_nextseq2000_train",
        platform="NextSeq 2000",
        h5_relative_paths=(
            "matrix/NextSeq_2000/subset_ERR12916630_1.500k.fastq.gz.qual_model.h5",
            "matrix/NextSeq_2000/subset_SRR38425641_1.500k.fastq.gz.qual_model.h5",
            "matrix/NextSeq_2000/subset_ERR15158516_1.500k.fastq.gz.qual_model.h5",
        ),
        fastq_relative_paths=(
            "matrix/sub_NextSeq_2000/subset_ERR12916630_1.500k.fastq.gz",
            "matrix/sub_NextSeq_2000/subset_SRR38425641_1.500k.fastq.gz",
            "matrix/sub_NextSeq_2000/subset_ERR15158516_1.500k.fastq.gz",
        ),
    ),
    "matrix_nextseq2000_holdout": DatasetSpec(
        name="matrix_nextseq2000_holdout",
        platform="NextSeq 2000",
        h5_relative_paths=(
            "matrix/NextSeq_2000/subset_SRR27385067_1.500k.fastq.gz.qual_model.h5",
            "matrix/NextSeq_2000/subset_SRR29436245_1.500k.fastq.gz.qual_model.h5",
            "matrix/NextSeq_2000/subset_ERR16794678_1.500k.fastq.gz.qual_model.h5",
        ),
        fastq_relative_paths=(
            "matrix/sub_NextSeq_2000/subset_SRR27385067_1.500k.fastq.gz",
            "matrix/sub_NextSeq_2000/subset_SRR29436245_1.500k.fastq.gz",
            "matrix/sub_NextSeq_2000/subset_ERR16794678_1.500k.fastq.gz",
        ),
    ),
    "matrix_novaseq6000_train": DatasetSpec(
        name="matrix_novaseq6000_train",
        platform="NovaSeq 6000",
        h5_relative_paths=(
            "matrix/NovaSeq_6000/subset_ERR10746686_1.500k.fastq.gz.qual_model.h5",
            "matrix/NovaSeq_6000/subset_ERR11454184_1.500k.fastq.gz.qual_model.h5",
            "matrix/NovaSeq_6000/subset_ERR16748054_1.500k.fastq.gz.qual_model.h5",
        ),
        fastq_relative_paths=(
            "matrix/sub_NovaSeq_6000/subset_ERR10746686_1.500k.fastq.gz",
            "matrix/sub_NovaSeq_6000/subset_ERR11454184_1.500k.fastq.gz",
            "matrix/sub_NovaSeq_6000/subset_ERR16748054_1.500k.fastq.gz",
        ),
    ),
    "matrix_novaseq6000_holdout": DatasetSpec(
        name="matrix_novaseq6000_holdout",
        platform="NovaSeq 6000",
        h5_relative_paths=(
            "matrix/NovaSeq_6000/subset_ERR16822574_1.500k.fastq.gz.qual_model.h5",
            "matrix/NovaSeq_6000/subset_SRR11411692_1.500k.fastq.gz.qual_model.h5",
            "matrix/NovaSeq_6000/subset_ERR3989434_1.500k.fastq.gz.qual_model.h5",
        ),
        fastq_relative_paths=(
            "matrix/sub_NovaSeq_6000/subset_ERR16822574_1.500k.fastq.gz",
            "matrix/sub_NovaSeq_6000/subset_SRR11411692_1.500k.fastq.gz",
            "matrix/sub_NovaSeq_6000/subset_ERR3989434_1.500k.fastq.gz",
        ),
    ),
    "matrix_novaseq_x_plus_train": DatasetSpec(
        name="matrix_novaseq_x_plus_train",
        platform="NovaSeq X Plus",
        h5_relative_paths=(
            "matrix/NovaSeq_X_Plus/subset_DRR917081_1.500k.fastq.gz.qual_model.h5",
            "matrix/NovaSeq_X_Plus/subset_SRR36865461_1.500k.fastq.gz.qual_model.h5",
            "matrix/NovaSeq_X_Plus/subset_SRR29727531_1.500k.fastq.gz.qual_model.h5",
        ),
        fastq_relative_paths=(
            "matrix/sub_NovaSeq_X_Plus/subset_DRR917081_1.500k.fastq.gz",
            "matrix/sub_NovaSeq_X_Plus/subset_SRR36865461_1.500k.fastq.gz",
            "matrix/sub_NovaSeq_X_Plus/subset_SRR29727531_1.500k.fastq.gz",
        ),
    ),
    "matrix_novaseq_x_plus_holdout": DatasetSpec(
        name="matrix_novaseq_x_plus_holdout",
        platform="NovaSeq X Plus",
        h5_relative_paths=(
            "matrix/NovaSeq_X_Plus/subset_SRR36237671_1.500k.fastq.gz.qual_model.h5",
            "matrix/NovaSeq_X_Plus/subset_SRR30693652_1.500k.fastq.gz.qual_model.h5",
            "matrix/NovaSeq_X_Plus/subset_SRR37380935_1.500k.fastq.gz.qual_model.h5",
        ),
        fastq_relative_paths=(
            "matrix/sub_NovaSeq_X_Plus/subset_SRR36237671_1.500k.fastq.gz",
            "matrix/sub_NovaSeq_X_Plus/subset_SRR30693652_1.500k.fastq.gz",
            "matrix/sub_NovaSeq_X_Plus/subset_SRR37380935_1.500k.fastq.gz",
        ),
    ),
    "matrix_mgiseq_g400_train": DatasetSpec(
        name="matrix_mgiseq_g400_train",
        platform="MGISEQ-2000/DNBSEQ-G400",
        h5_relative_paths=(
            "matrix/MGISEQ-2000_DNBSEQ-G400/subset_ERR13433159_1.500k.fastq.gz.qual_model.h5",
            "matrix/MGISEQ-2000_DNBSEQ-G400/subset_SRR13142212.500k.fastq.gz.qual_model.h5",
            "matrix/MGISEQ-2000_DNBSEQ-G400/subset_SRR31071044_1.500k.fastq.gz.qual_model.h5",
        ),
        fastq_relative_paths=(
            "matrix/sub_MGISEQ-2000_DNBSEQ-G400/subset_ERR13433159_1.500k.fastq.gz",
            "matrix/sub_MGISEQ-2000_DNBSEQ-G400/subset_SRR13142212.500k.fastq.gz",
            "matrix/sub_MGISEQ-2000_DNBSEQ-G400/subset_SRR31071044_1.500k.fastq.gz",
        ),
    ),
    "matrix_mgiseq_g400_holdout": DatasetSpec(
        name="matrix_mgiseq_g400_holdout",
        platform="MGISEQ-2000/DNBSEQ-G400",
        h5_relative_paths=(
            "matrix/MGISEQ-2000_DNBSEQ-G400/subset_SRR33170689_1.500k.fastq.gz.qual_model.h5",
            "matrix/MGISEQ-2000_DNBSEQ-G400/subset_SRR31781315_1.500k.fastq.gz.qual_model.h5",
            "matrix/MGISEQ-2000_DNBSEQ-G400/subset_SRR32083931_1.500k.fastq.gz.qual_model.h5",
        ),
        fastq_relative_paths=(
            "matrix/sub_MGISEQ-2000_DNBSEQ-G400/subset_SRR33170689_1.500k.fastq.gz",
            "matrix/sub_MGISEQ-2000_DNBSEQ-G400/subset_SRR31781315_1.500k.fastq.gz",
            "matrix/sub_MGISEQ-2000_DNBSEQ-G400/subset_SRR32083931_1.500k.fastq.gz",
        ),
    ),
}


DATASET_GROUPS: dict[str, tuple[str, ...]] = {
    "illumina": ("novaseq", "nextseq2000"),
    "bgi_mgi": ("dnbseq_t7", "mgiseq2000"),
    "mixed": ("novaseq", "nextseq2000", "dnbseq_t7", "mgiseq2000"),
    "matrix_dnbseq_t7_all": (
        "matrix_dnbseq_t7_train",
        "matrix_dnbseq_t7_holdout",
    ),
    "matrix_nextseq2000_all": (
        "matrix_nextseq2000_train",
        "matrix_nextseq2000_holdout",
    ),
    "matrix_novaseq6000_all": (
        "matrix_novaseq6000_train",
        "matrix_novaseq6000_holdout",
    ),
    "matrix_novaseq_x_plus_all": (
        "matrix_novaseq_x_plus_train",
        "matrix_novaseq_x_plus_holdout",
    ),
    "matrix_mgiseq_g400_all": (
        "matrix_mgiseq_g400_train",
        "matrix_mgiseq_g400_holdout",
    ),
    "matrix_mixed_train": (
        "matrix_dnbseq_t7_train",
        "matrix_nextseq2000_train",
        "matrix_novaseq6000_train",
        "matrix_novaseq_x_plus_train",
        "matrix_mgiseq_g400_train",
    ),
    "matrix_all": (
        "matrix_dnbseq_t7_train",
        "matrix_dnbseq_t7_holdout",
        "matrix_nextseq2000_train",
        "matrix_nextseq2000_holdout",
        "matrix_novaseq6000_train",
        "matrix_novaseq6000_holdout",
        "matrix_novaseq_x_plus_train",
        "matrix_novaseq_x_plus_holdout",
        "matrix_mgiseq_g400_train",
        "matrix_mgiseq_g400_holdout",
    ),
}


_ALIASES = {
    "novaseq_train": "novaseq_hg",
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
        for h5_relative_path, fastq_relative_path in zip(
            spec.h5_relative_paths,
            spec.fastq_relative_paths,
        ):
            h5_name = Path(h5_relative_path).name
            if h5_name in seen_h5_names:
                raise ValueError(f"duplicate H5 basename in selection: {h5_name}")
            seen_h5_names.add(h5_name)
            item = DatasetFile(
                dataset=dataset_name,
                platform=spec.platform,
                h5_path=data_root / h5_relative_path,
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
