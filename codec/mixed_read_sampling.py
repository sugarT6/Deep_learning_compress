"""Read-level, source-stratified batches for quality-model sampling comparisons.

The original sampler remains untouched. Platform weights describe *read slots*
here, whereas the original sampler applies them to single-source batches. Read
permutations and cursors persist across logging epochs. Only seed, round and
cursor are checkpointed; neither permutations nor optional RAM caches are saved.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .datasets import DirectQualityDataset
from .fastq_stream import FastqBatch, make_fastq_batch
from .train import BalancedTrainingSampler


@dataclass(frozen=True)
class MixedTrainingSample:
    batch: FastqBatch
    row_dataset_indices: np.ndarray
    row_read_indices: np.ndarray
    source_read_counts: dict[str, int]


class MixedReadTrainingSampler(BalancedTrainingSampler):
    """Mix all sources while retaining the old platform/file probabilities.

    Each file contributes a persistent random permutation of its training read
    indices. Exhausting one permutation starts another; a batch crossing that
    boundary may contain an index from both rounds. No validation read is used.
    ``preload_training_values`` copies only training prefixes into RAM, without
    creating files. Expanded20 needs about 9.8 GB for bases, Q and read offsets,
    plus about 288 MB for the active int64 read permutations.
    """

    STATE_VERSION = 1
    PERMUTATION_SEED_TAG = 0x4D495845

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
        preload_training_values: bool = False,
        feature_block_reads: int = 0,
    ) -> None:
        if feature_block_reads:
            raise ValueError("mixed-read sampling does not support block features")
        if type(preload_training_values) is not bool:
            raise ValueError("preload_training_values must be a boolean")
        self._preloaded_values: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._read_orders: dict[str, np.ndarray] = {}
        self.preloaded_training_bytes = 0
        super().__init__(datasets, cache_dir, train_fraction=train_fraction,
                         batch_reads=batch_reads, seed=seed,
                         steps_per_epoch=steps_per_epoch,
                         family_weights=family_weights, feature_block_reads=0)
        try:
            # With no platform weights the old sampler weights all files equally.
            self._quota_weights = (dict(self.family_weights)
                                   if self.family_weights is not None else
                                   {family: len(members)
                                    for family, members in self._by_family.items()})
            self._family_cycle = tuple(
                family for family in self._families
                for _ in range(self._quota_weights[family]))
            minimum_family_slots = self.batch_reads // len(self._family_cycle)
            if any(minimum_family_slots * self._quota_weights[family] < len(members)
                   for family, members in self._by_family.items()):
                raise ValueError("batch_reads is too small to include every source "
                                 "with these platform weights")
            self.source_probabilities = {
                dataset.accession: (
                    self._quota_weights[dataset.platform_family]
                    / len(self._family_cycle)
                    / len(self._by_family[dataset.platform_family]))
                for dataset in self.datasets}
            self._dataset_indices = {d.accession: i for i, d in enumerate(self.datasets)}
            self._read_rounds = {d.accession: 0 for d in self.datasets}
            self._read_cursors = {d.accession: 0 for d in self.datasets}
            for dataset in self.datasets:
                self._read_orders[dataset.accession] = self._read_order(dataset.accession, 0)
            if preload_training_values:
                self.preload_training_values()
        except Exception:
            self.close()
            raise

    def _read_order(self, accession: str, round_index: int) -> np.ndarray:
        rng = np.random.default_rng(np.random.SeedSequence([
            self.seed, self._dataset_indices[accession], round_index,
            self.PERMUTATION_SEED_TAG]))
        return rng.permutation(self._train_stops[accession])

    def preload_training_values(self) -> int:
        """Load training prefixes once into RAM and return their byte count."""
        for dataset in self.datasets:
            accession = dataset.accession
            if accession in self._preloaded_values:
                continue
            # Reuse the audited reader's HDF5 handle. The original reader remains
            # open for metadata, validation and corpus-statistics operations.
            handle = self.reader_for(accession)._require_open()
            offsets = np.asarray(handle["read_offsets"][:self._train_stops[accession] + 1],
                                 dtype=np.int64)
            end = int(offsets[-1])
            bases = np.asarray(handle["base_values"][:end], dtype=np.uint8)
            qualities = np.asarray(handle["quality_values"][:end], dtype=np.uint8)
            for values in (offsets, bases, qualities):
                values.flags.writeable = False
            self._preloaded_values[accession] = (offsets, bases, qualities)
            self.preloaded_training_bytes += offsets.nbytes + bases.nbytes + qualities.nbytes
        return self.preloaded_training_bytes

    def close(self) -> None:
        super().close()
        self._preloaded_values.clear()
        self._read_orders.clear()
        self.preloaded_training_bytes = 0

    def _source_quotas(self) -> dict[str, int]:
        """Rotate integer remainders across batches rather than biasing files."""
        quotient, remainder = divmod(self.batch_reads, len(self._family_cycle))
        start = (self._total_samples * self.batch_reads) % len(self._family_cycle)
        extras = [self._family_cycle[(start + i) % len(self._family_cycle)]
                  for i in range(remainder)]
        quotas = {}
        for family in self._families:
            members = self._by_family[family]
            family_slots = quotient * self._quota_weights[family] + extras.count(family)
            per_file, extra = divmod(family_slots, len(members))
            cursor = self._family_counts[family] % len(members)
            for dataset in members:
                quotas[dataset.accession] = per_file
            for i in range(extra):
                quotas[members[(cursor + i) % len(members)].accession] += 1
        return quotas

    def _take_read_indices(self, accession: str, count: int) -> np.ndarray:
        chunks = []
        remaining = count
        while remaining:
            order = self._read_orders[accession]
            cursor = self._read_cursors[accession]
            if cursor == len(order):
                self._read_rounds[accession] += 1
                order = self._read_order(accession, self._read_rounds[accession])
                self._read_orders[accession] = order
                cursor = 0
            stop = min(cursor + remaining, len(order))
            chunks.append(order[cursor:stop])
            remaining -= stop - cursor
            self._read_cursors[accession] = stop
        return np.concatenate(chunks)

    def _read_rows(self, accession: str, indices: np.ndarray):
        cached = self._preloaded_values.get(accession)
        if cached is not None:
            offsets, bases, qualities = cached
            return ([bases[int(offsets[index]):int(offsets[index + 1])] for index in indices],
                    [qualities[int(offsets[index]):int(offsets[index + 1])] for index in indices])
        batch = self.reader_for(accession).read_indices(indices)
        return ([batch.bases[row, :length] for row, length in enumerate(batch.lengths)],
                [batch.qualities[row, :length] for row, length in enumerate(batch.lengths)])

    def sample_batch(self) -> MixedTrainingSample:
        if not self._epoch or self._schedule_cursor == self.steps_per_epoch:
            self._epoch += 1
            self._schedule_cursor = 0
        quotas = self._source_quotas()
        bases, qualities, source_rows, read_rows = [], [], [], []
        for dataset in self.datasets:
            accession = dataset.accession
            count = quotas[accession]
            indices = self._take_read_indices(accession, count)
            base_rows, quality_rows = self._read_rows(accession, indices)
            bases.extend(base_rows)
            qualities.extend(quality_rows)
            source_rows.extend([self._dataset_indices[accession]] * count)
            read_rows.extend(indices.tolist())
            self._read_counts[accession] += count
            self._sample_counts[accession] += 1
            self._family_counts[dataset.platform_family] += count
        permutation = self._rng.permutation(self.batch_reads)
        row_dataset_indices = np.asarray(source_rows, dtype=np.int64)[permutation]
        row_read_indices = np.asarray(read_rows, dtype=np.int64)[permutation]
        batch = make_fastq_batch([bases[i] for i in permutation],
                                 [qualities[i] for i in permutation],
                                 row_read_indices, source_name="mixed_training_sources")
        self._schedule_cursor += 1
        self._total_samples += 1
        return MixedTrainingSample(batch=batch, row_dataset_indices=row_dataset_indices,
                                   row_read_indices=row_read_indices,
                                   source_read_counts=quotas)

    def config_dict(self) -> dict[str, Any]:
        return {
            "strategy": "mixed_reads_fixed_family_weights_equal_files",
            "version": self.STATE_VERSION,
            "batch_source": "all_training_cache_files",
            "quota_unit": "reads",
            "read_sampling": "persistent_file_read_permutations_without_replacement",
            "remainder_policy": "persistent_family_read_cycle_and_within_family_rotation",
            "steps_per_epoch": self.steps_per_epoch,
            "dataset_order": [d.accession for d in self.datasets],
            "train_stops": dict(self._train_stops),
            "batch_reads": self.batch_reads,
            "seed": self.seed,
            "families": {family: [d.accession for d in members]
                         for family, members in self._by_family.items()},
            "family_weights": dict(self._quota_weights),
            "source_probabilities": dict(self.source_probabilities),
            "row_order": "independent_rng_shuffled_with_paired_bases_and_qualities",
        }

    def state_dict(self) -> dict[str, Any]:
        return copy.deepcopy({
            "version": self.STATE_VERSION,
            "config": self.config_dict(),
            "rng_state": self._rng.bit_generator.state,
            "epoch": self._epoch,
            "schedule_cursor": self._schedule_cursor,
            "total_samples": self._total_samples,
            "read_rounds": self._read_rounds,
            "read_cursors": self._read_cursors,
            "read_counts": self._read_counts,
            "source_batch_counts": self._sample_counts,
            "family_read_counts": self._family_counts,
        })

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if set(state) != set(self.state_dict()):
            raise ValueError("mixed sampler state keys mismatch")
        state = copy.deepcopy(dict(state))
        if state["version"] != self.STATE_VERSION or state["config"] != self.config_dict():
            raise ValueError("mixed sampler state/config mismatch")
        total = state["total_samples"]
        if type(total) is not int or total < 0:
            raise ValueError("invalid mixed sampler total batch count")
        epoch = (total - 1) // self.steps_per_epoch + 1 if total else 0
        cursor = (total - 1) % self.steps_per_epoch + 1 if total else 0
        if state["epoch"] != epoch or state["schedule_cursor"] != cursor:
            raise ValueError("mixed sampler epoch/cursor mismatch")
        accessions = set(self._train_stops)
        for key in ("read_rounds", "read_cursors", "read_counts", "source_batch_counts"):
            if set(state[key]) != accessions:
                raise ValueError("mixed sampler source set mismatch")
            if any(type(value) is not int or value < 0 for value in state[key].values()):
                raise ValueError("invalid mixed sampler source count")
        if set(state["family_read_counts"]) != set(self._families):
            raise ValueError("mixed sampler platform set mismatch")
        full_cycles, remainder = divmod(total * self.batch_reads, len(self._family_cycle))
        for family, members in self._by_family.items():
            family_reads = (full_cycles * self._quota_weights[family]
                            + self._family_cycle[:remainder].count(family))
            if state["family_read_counts"][family] != family_reads:
                raise ValueError("mixed sampler platform quota mismatch")
            per_file, extra = divmod(family_reads, len(members))
            for i, dataset in enumerate(members):
                accession = dataset.accession
                expected_reads = per_file + (i < extra)
                if state["read_counts"][accession] != expected_reads:
                    raise ValueError("mixed sampler file quota mismatch")
                if state["source_batch_counts"][accession] != total:
                    raise ValueError("mixed sampler source batch count mismatch")
                read_cursor = state["read_cursors"][accession]
                read_round = state["read_rounds"][accession]
                size = self._train_stops[accession]
                if not 0 <= read_cursor <= size or read_round * size + read_cursor != expected_reads:
                    raise ValueError("mixed sampler read round/cursor mismatch")
                if expected_reads and read_cursor == 0:
                    raise ValueError("mixed sampler noncanonical read cursor")
        # Validate the saved RNG before changing this sampler's live state.
        check_rng = np.random.default_rng()
        check_rng.bit_generator.state = state["rng_state"]
        orders = {accession: self._read_order(accession, state["read_rounds"][accession])
                  for accession in accessions}
        self._rng.bit_generator.state = state["rng_state"]
        self._epoch = epoch
        self._schedule_cursor = cursor
        self._total_samples = total
        self._read_orders = orders
        self._read_rounds = state["read_rounds"]
        self._read_cursors = state["read_cursors"]
        self._read_counts = state["read_counts"]
        self._sample_counts = state["source_batch_counts"]
        self._family_counts = state["family_read_counts"]

    def statistics_dict(self) -> dict[str, Any]:
        total_reads = sum(self._read_counts.values())
        denominator = max(total_reads, 1)
        unique = {accession: min(count, self._train_stops[accession])
                  for accession, count in self._read_counts.items()}
        return {
            "count_unit": "reads",
            "total_batches": self._total_samples,
            "total_reads": total_reads,
            "nominal_read_slots": self._total_samples * self.batch_reads,
            "unique_reads": sum(unique.values()),
            "train_read_count": sum(self._train_stops.values()),
            "preloaded_training_bytes": self.preloaded_training_bytes,
            "datasets": {accession: {
                "count": count, "reads": count, "proportion": count / denominator,
                "target_read_proportion": self.source_probabilities[accession],
                "batches": self._sample_counts[accession],
                "unique_reads": unique[accession],
                "coverage_fraction": unique[accession] / self._train_stops[accession],
                "read_round": self._read_rounds[accession],
                "read_cursor": self._read_cursors[accession],
            } for accession, count in self._read_counts.items()},
            "families": {family: {
                "count": count, "reads": count, "proportion": count / denominator,
                "target_read_proportion": self._quota_weights[family] / len(self._family_cycle),
            } for family, count in self._family_counts.items()},
        }
