"""Whole-original-block sampling for bounded block-conditioned adaptation.

The scan retains a uniform reservoir of eligible original FASTQ blocks.  Only
retained blocks receive descriptors, and each descriptor uses that entire block
before its reads enter any adaptation minibatch.  Hashes record provenance;
headers and source identities never become model features.
"""

from __future__ import annotations

import hashlib
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np

from .block_features import BLOCK_FEATURE_BYTES, BLOCK_FEATURE_DIM, decode_block_features, encode_block_features
from .fastq_stream import MAX_BATCH_READS, RawFastqRecord, encode_base_ids, iter_fastq_records, make_fastq_batch


def _integer(value: Any, name: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def sampled_block_ids_per_read(report: Mapping[str, Any]) -> np.ndarray:
    """Expand compact returned block provenance into IDs aligned with records/z."""

    blocks = report["sampled_block_ids"]
    counts = report["sampled_block_read_counts"]
    if len(blocks) != len(counts):
        raise ValueError("sampled block IDs/counts have different lengths")
    for value in blocks:
        _integer(value, "sampled block ID", 0)
    for count in counts:
        _integer(count, "sampled block read count", 1)
    result = np.repeat(np.asarray(blocks, dtype=np.int64), counts)
    if result.size != report["sampled_reads"]:
        raise ValueError("sampled block counts do not match sampled_reads")
    return result


def sample_block_records(
    source: Path,
    config: Any,
    skip_reads: int,
    deadline: float,
    *,
    block_reads: int,
    feature_mode: str,
) -> Tuple[List[RawFastqRecord], Dict[str, Any], np.ndarray]:
    """Return whole-block records, compact provenance, and aligned wire features.

    Warmup exclusion rounds up to a file-relative block boundary.  Reservoir
    capacity is ``backbone_max_reads // block_reads`` including a possible EOF
    tail.  After deterministic block shuffling, the symbol budget drops whole
    blocks which do not fit and continues considering the remaining reservoir.
    Uniformity describes the reservoir *before* that explicit symbol filtering.
    A deadline failure discards all partial samples and returns ``[0, 16]`` z.
    """

    block_reads = _integer(block_reads, "block_reads", 1)
    if block_reads > MAX_BATCH_READS:
        raise ValueError(f"block_reads must be <= {MAX_BATCH_READS}")
    skip_reads = _integer(skip_reads, "skip_reads", 0)
    max_reads = _integer(config.backbone_max_reads, "backbone_max_reads", 1)
    max_symbols = _integer(config.backbone_max_symbols, "backbone_max_symbols", 1)
    max_length = _integer(config.max_read_length, "max_read_length", 1)
    seed = _integer(config.seed, "seed", 0)
    if feature_mode not in ("features", "constant"):
        raise ValueError("feature_mode must be features or constant")
    source = Path(source)
    aligned_skip = ((skip_reads + block_reads - 1) // block_reads) * block_reads
    capacity = max_reads // block_reads
    rng = random.Random(seed)
    empty_hash = hashlib.sha256(b"").hexdigest()
    report: Dict[str, Any] = dict(
        sampling_strategy="uniform_reservoir_of_whole_original_blocks",
        descriptor_source="same_original_codec_block",
        block_reads=block_reads, block_feature_mode=feature_mode,
        requested_skip_reads=skip_reads, aligned_skip_reads=aligned_skip,
        reservoir_max_blocks=capacity,
        total_file_reads=0, total_file_symbols=0, total_file_blocks=0,
        eligible_blocks=0, eligible_reads=0, eligible_symbols=0,
        excluded_warmup_blocks=0, excluded_warmup_reads=0,
        excluded_long_blocks=0, excluded_long_reads=0, excluded_long_block_reads=0,
        excluded_empty_blocks=0, excluded_empty_reads=0,
        excluded_symbol_cap_blocks=0, excluded_symbol_cap_reads=0,
        excluded_symbol_cap_symbols=0,
        sampled_blocks=0, sampled_reads=0, sampled_symbols=0,
        sampled_block_ids=[], sampled_block_read_counts=[], sampled_block_read_ranges=[],
        sampled_block_feature_stream_bytes=0,
        read_indices_sha256=empty_hash, block_ids_sha256=empty_hash,
        sampled_records_sha256=empty_hash, block_feature_payload_sha256=empty_hash,
        sampling_complete=False,
    )
    empty_features = np.empty((0, BLOCK_FEATURE_DIM), dtype=np.float32)
    reservoir = []

    def deadline_failure():
        # Selection output remains empty even if a deadline hits while building
        # the descriptors after the source scan has already finished.
        return [], report, empty_features

    def consider(rows: List[RawFastqRecord]) -> None:
        block_id = rows[0].read_index // block_reads
        report["total_file_blocks"] += 1
        if rows[0].read_index < aligned_skip:
            report["excluded_warmup_blocks"] += 1
            report["excluded_warmup_reads"] += len(rows)
            return
        lengths = [len(row.quality) for row in rows]
        too_long = sum(length > max_length for length in lengths)
        if too_long:
            report["excluded_long_blocks"] += 1
            report["excluded_long_reads"] += too_long
            report["excluded_long_block_reads"] += len(rows)
            return
        symbols = sum(lengths)
        if not symbols:
            report["excluded_empty_blocks"] += 1
            report["excluded_empty_reads"] += len(rows)
            return
        report["eligible_blocks"] += 1
        report["eligible_reads"] += len(rows)
        report["eligible_symbols"] += symbols
        item = (block_id, rows, symbols)
        if len(reservoir) < capacity:
            reservoir.append(item)
        elif capacity:
            index = rng.randrange(report["eligible_blocks"])
            if index < capacity:
                reservoir[index] = item

    if time.perf_counter() >= deadline:
        return deadline_failure()
    iterator = iter_fastq_records(source)
    current: List[RawFastqRecord] = []
    try:
        while True:
            if time.perf_counter() >= deadline:
                return deadline_failure()
            record = next(iterator, None)
            if record is None:
                if current:
                    consider(current)  # Only a genuine EOF creates a short block.
                break
            if record.read_index != report["total_file_reads"]:
                raise ValueError("block adaptation requires consecutive original source read indices")
            report["total_file_reads"] += 1
            report["total_file_symbols"] += len(record.quality)
            current.append(record)
            if len(current) == block_reads:
                consider(current)
                current = []
    finally:
        iterator.close()

    report["reservoir_blocks"] = len(reservoir)
    report["excluded_reservoir_blocks"] = report["eligible_blocks"] - len(reservoir)
    rng.shuffle(reservoir)
    selected = []
    symbols = 0
    for item in reservoir:
        if symbols + item[2] > max_symbols:
            report["excluded_symbol_cap_blocks"] += 1
            report["excluded_symbol_cap_reads"] += len(item[1])
            report["excluded_symbol_cap_symbols"] += item[2]
            continue
        selected.append(item)
        symbols += item[2]

    records: List[RawFastqRecord] = []
    conditions = []
    selected_ids, selected_counts, selected_ranges = [], [], []
    record_digest = hashlib.sha256()
    payload_digest = hashlib.sha256()
    for block_id, rows, _ in selected:
        if time.perf_counter() >= deadline:
            return deadline_failure()
        if feature_mode == "features":
            batch = make_fastq_batch(
                [encode_base_ids(row.sequence) for row in rows],
                [np.frombuffer(row.quality, dtype=np.uint8) - 33 for row in rows],
                [row.read_index for row in rows], source_name=source.name,
            )
            payload = encode_block_features(batch)
        else:
            payload = bytes(BLOCK_FEATURE_BYTES)
        condition = decode_block_features(payload)
        records.extend(rows)
        conditions.append(np.broadcast_to(condition, (len(rows), BLOCK_FEATURE_DIM)).copy())
        selected_ids.append(block_id)
        selected_counts.append(len(rows))
        selected_ranges.append([rows[0].read_index, rows[-1].read_index + 1])
        if feature_mode == "features":
            payload_digest.update(payload)
        for row in rows:
            record_digest.update(row.to_bytes())
    if time.perf_counter() >= deadline:
        return deadline_failure()
    features = np.concatenate(conditions) if conditions else empty_features
    report.update(
        sampling_complete=True, sampled_blocks=len(selected),
        sampled_reads=len(records), sampled_symbols=symbols,
        sampled_block_ids=selected_ids, sampled_block_read_counts=selected_counts,
        sampled_block_read_ranges=selected_ranges,
        sampled_block_feature_stream_bytes=(len(selected) * BLOCK_FEATURE_BYTES
                                           if feature_mode == "features" else 0),
        read_indices_sha256=hashlib.sha256(
            np.asarray([row.read_index for row in records], dtype="<i8").tobytes()).hexdigest(),
        block_ids_sha256=hashlib.sha256(np.asarray(selected_ids, dtype="<i8").tobytes()).hexdigest(),
        sampled_records_sha256=record_digest.hexdigest(),
        block_feature_payload_sha256=payload_digest.hexdigest(),
    )
    return records, report, features
