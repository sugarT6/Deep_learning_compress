"""Fixed 19-dataset split for the first direct-quality experiment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

from .training_cache import CACHE_FILENAME_SUFFIX, DEFAULT_CACHE_DIR


@dataclass(frozen=True)
class DirectQualityDataset:
    accession: str
    platform_family: str
    training_source: bool
    unseen_instrument: bool = False

    def cache_path(self, cache_dir: Path = DEFAULT_CACHE_DIR) -> Path:
        return cache_dir / f"{self.accession}{CACHE_FILENAME_SUFFIX}"


DATASETS: Tuple[DirectQualityDataset, ...] = (
    DirectQualityDataset("ERR2755197", "MGI/BGI", True),
    DirectQualityDataset("CNR0847458", "MGI/BGI", True),
    DirectQualityDataset("MGI_Q4", "MGI/BGI", True),
    DirectQualityDataset("CNR0066422", "MGI/BGI", False),
    DirectQualityDataset("CNR0847462", "MGI/BGI", False),
    DirectQualityDataset("CNR1261866", "MGI/BGI", False),
    DirectQualityDataset("ERR966765", "Illumina", True),
    DirectQualityDataset("SRR5604291", "Illumina", True),
    DirectQualityDataset("SRR622457", "Illumina", True),
    DirectQualityDataset("SRR10377488", "Illumina", True),
    DirectQualityDataset("SRR6691666", "Illumina", True),
    DirectQualityDataset("SRR3066199", "Illumina", False),
    DirectQualityDataset("SRR13114615", "Illumina", False),
    DirectQualityDataset("SRR10965088", "Illumina", False, True),
    DirectQualityDataset("SRR29287266", "Illumina", False, True),
    DirectQualityDataset("SRR5181541", "ABI SOLiD", True),
    DirectQualityDataset("SRR835803", "ABI SOLiD", False),
    DirectQualityDataset("SRR5867380", "Ion Torrent", True),
    DirectQualityDataset("SRR1238539", "Ion Torrent", False),
)

DATASET_BY_ACCESSION: Dict[str, DirectQualityDataset] = {
    dataset.accession: dataset for dataset in DATASETS
}
TRAIN_DATASETS = tuple(dataset for dataset in DATASETS if dataset.training_source)
UNSEEN_DATASETS = tuple(dataset for dataset in DATASETS if not dataset.training_source)
UNSEEN_INSTRUMENT_DATASETS = tuple(
    dataset for dataset in UNSEEN_DATASETS if dataset.unseen_instrument
)
PLATFORM_FAMILIES = ("MGI/BGI", "Illumina", "ABI SOLiD", "Ion Torrent")


def datasets_by_family(
    datasets: Iterable[DirectQualityDataset],
) -> Dict[str, Tuple[DirectQualityDataset, ...]]:
    grouped = {}
    for family in PLATFORM_FAMILIES:
        members = tuple(
            dataset for dataset in datasets if dataset.platform_family == family
        )
        if members:
            grouped[family] = members
    return grouped


def validate_fixed_split() -> None:
    if len(DATASETS) != 19 or len(TRAIN_DATASETS) != 10 or len(UNSEEN_DATASETS) != 9:
        raise RuntimeError("direct-quality split must contain 10 train and 9 unseen datasets")
    if len(DATASET_BY_ACCESSION) != len(DATASETS):
        raise RuntimeError("direct-quality dataset accessions must be unique")
    unknown_families = {
        dataset.platform_family for dataset in DATASETS
    } - set(PLATFORM_FAMILIES)
    if unknown_families:
        raise RuntimeError(f"unknown platform families: {sorted(unknown_families)}")


validate_fixed_split()
