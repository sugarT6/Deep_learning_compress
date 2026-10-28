"""Freeze a reusable, header-free reservoir for adaptation diagnostics.

Preparing a cache scans FASTQ for sampling only; it never scores a model.
Warm-up still reads the short original prefix so its existing wire hash and
admission semantics remain unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .backbone_adapter import sample_records
from .container import sha256_file
from .fastq_stream import RawFastqRecord, iter_fastq_records, encode_base_ids, encode_quality_ids


SAMPLING_FIELDS = ("seed", "sample_reads", "sample_symbols", "warmup_reads",
                   "warmup_symbols", "max_read_length")


def source_identity(source):
    stat = Path(source).stat()
    return {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def content_digest(records):
    digest = hashlib.sha256()
    for record in records:
        digest.update(np.asarray([record.read_index, len(record.sequence)], dtype="<i8").tobytes())
        digest.update(record.sequence)
        digest.update(record.quality)
    return digest.hexdigest()


def prefix_records(source, config):
    records, symbols = [], 0
    iterator = iter_fastq_records(source)
    try:
        for _ in range(config.warmup_reads):
            record = next(iterator, None)
            if record is None:
                break
            length = len(record.quality)
            if length > config.max_read_length or symbols + length > config.warmup_symbols:
                break
            records.append(record)
            symbols += length
    finally:
        iterator.close()
    return records


def _validate_records(records, sample, config, prefix_count):
    indices = [r.read_index for r in records]
    if (not sample.get("sampling_complete") or len(records) < 4
            or len(records) > config.sample_reads or len(set(indices)) != len(indices)
            or any(type(i) is not int or i < prefix_count
                   or i >= sample["total_file_reads"] for i in indices)):
        raise ValueError("invalid or overlapping diagnostic reservoir indices")
    symbols = 0
    for record in records:
        length = len(record.quality)
        if not 0 < length <= config.max_read_length or len(record.sequence) != length:
            raise ValueError("invalid diagnostic read length")
        encode_base_ids(record.sequence)
        encode_quality_ids(record.quality, Path("cached.fastq"), record.read_index + 1)
        symbols += length
    if (symbols > config.sample_symbols or sample["sampled_reads"] != len(records)
            or sample["sampled_symbols"] != symbols
            or sample["excluded_warmup_reads"] != prefix_count
            or sample["read_indices_sha256"] != hashlib.sha256(
                np.asarray(indices, dtype="<i8").tobytes()).hexdigest()):
        raise ValueError("diagnostic reservoir counts or indices hash mismatch")


def save_sample(source, directory, config, records, sample, *, provenance=None):
    """Save freshly sampled or independently audited historical numeric reads."""
    source, directory = Path(source).resolve(), Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError("choose a new empty sample cache directory")
    identity = source_identity(source)
    prefix = prefix_records(source, config)
    _validate_records(records, sample, config, len(prefix))
    directory.mkdir(parents=True, exist_ok=True)
    offsets = np.r_[0, np.cumsum([len(r.quality) for r in records], dtype=np.int64)]
    arrays = directory / "records.npz"
    np.savez_compressed(arrays, offsets=offsets.astype("<i8"),
        indices=np.asarray([r.read_index for r in records], dtype="<i8"),
        sequence=np.frombuffer(b"".join(r.sequence for r in records), dtype=np.uint8),
        quality=np.frombuffer(b"".join(r.quality for r in records), dtype=np.uint8))
    if identity != source_identity(source):
        raise ValueError("source changed while saving sample cache")
    manifest = dict(format="direct-quality-adaptation-sample", version=1,
        source=str(source), source_identity=identity,
        sampling_config={k: getattr(config, k) for k in SAMPLING_FIELDS},
        prefix_reads=len(prefix), prefix_content_sha256=content_digest(prefix),
        sample=sample, sample_content_sha256=content_digest(records),
        records_sha256=sha256_file(arrays), provenance=provenance,
        scope="development_reservoir_only_no_complete_file_model_scoring")
    with (directory / "manifest.json").open("x") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return manifest


def load_sample(source, directory, config):
    source, directory = Path(source).resolve(), Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    if (manifest.get("format") != "direct-quality-adaptation-sample"
            or manifest.get("version") != 1 or manifest["source"] != str(source)
            or manifest["source_identity"] != source_identity(source)
            or manifest["sampling_config"] != {k: getattr(config, k) for k in SAMPLING_FIELDS}):
        raise ValueError("sample cache source identity or sampling configuration mismatch")
    path = directory / "records.npz"
    if sha256_file(path) != manifest["records_sha256"]:
        raise ValueError("sample cache file SHA-256 mismatch")
    with np.load(path, allow_pickle=False) as data:
        offsets, indices, sequence, quality = (data[k] for k in
            ("offsets", "indices", "sequence", "quality"))
        if (offsets.dtype.kind != "i" or indices.dtype.kind != "i"
                or sequence.dtype != np.uint8 or quality.dtype != np.uint8
                or any(a.ndim != 1 for a in (offsets, indices, sequence, quality))
                or len(offsets) != len(indices) + 1 or offsets[0] != 0
                or np.any(np.diff(offsets) <= 0)
                or offsets[-1] != len(sequence) or len(sequence) != len(quality)):
            raise ValueError("invalid sample cache arrays")
        records = [RawFastqRecord(b"@cached", sequence[a:b].tobytes(), b"+",
            quality[a:b].tobytes(), (b"\n",) * 4, int(index))
            for a, b, index in zip(offsets[:-1], offsets[1:], indices)]
    prefix = prefix_records(source, config)
    if (len(prefix) != manifest["prefix_reads"]
            or content_digest(prefix) != manifest["prefix_content_sha256"]
            or content_digest(records) != manifest["sample_content_sha256"]):
        raise ValueError("sample cache numerical content mismatch")
    _validate_records(records, manifest["sample"], config, len(prefix))
    return records, manifest


def prepare_sample(source, directory, config):
    source = Path(source).resolve()
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError("choose a new empty sample cache directory")
    identity = source_identity(source)
    prefix = prefix_records(source, config)
    records, sample = sample_records(source, config.head_config(), len(prefix), float("inf"))
    if identity != source_identity(source):
        raise ValueError("source changed during reservoir sampling")
    return save_sample(source, directory, config, records, sample,
                       provenance={"method": "single_uniform_reservoir_scan"})


def main():
    from .diagnose_adaptation import DiagnosticConfig
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    for name in SAMPLING_FIELDS:
        parser.add_argument("--" + name.replace("_", "-"), type=int,
                            default=getattr(DiagnosticConfig(), name))
    args = parser.parse_args()
    config = DiagnosticConfig(**{k: getattr(args, k) for k in SAMPLING_FIELDS})
    print(json.dumps(prepare_sample(args.input, args.output, config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
