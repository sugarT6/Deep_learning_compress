"""Single-thread CPU floor benchmark on fixed real logits; no full-file encode.

The reference quantizer is loaded from a local Git revision, without modifying
the checkout. Model prediction, validation and range coding are outside the
quantizer timers. Reuse the saved logits with --logits-cache for another trial.
"""

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time

import numpy as np
import torch

from .checkpoint import load_training_checkpoint
from .container import sha256_file
from .fastq_stream import iter_fastq_records, make_fastq_batch, encode_base_ids
from .head_adapter import apply_head_adapter
from .model import fastq_batch_to_tensors
from .probability_quantization import DEFAULT_FLOOR_CHUNK_ROWS, logits_to_intervals
from .range_encoder import RangeEncoder


def _reference_quantizer(revision):
    repo = Path(__file__).resolve().parents[1]
    source = subprocess.check_output(
        ["git", "show", f"{revision}:codec/probability_quantization.py"], cwd=repo
    )
    namespace = {"__name__": "floor_benchmark_reference"}
    exec(compile(source, f"{revision}:probability_quantization.py", "exec"), namespace)
    return namespace["logits_to_intervals"]


def _real_batches(args, checkpoint_sha):
    artifact = json.loads(args.adapter.read_text())
    if artifact.get("base_checkpoint_sha256") != checkpoint_sha:
        raise ValueError("adapter artifact checkpoint SHA-256 mismatch")
    model = load_training_checkpoint(args.checkpoint, device="cpu").model.eval()
    if artifact["adapter"] is not None:
        apply_head_adapter(model, artifact["adapter"], checkpoint_sha)
    iterator = iter_fastq_records(args.source)
    try:
        records = list(itertools.islice(iterator, args.skip_reads, args.skip_reads + args.reads))
    finally:
        iterator.close()
    if not records:
        raise ValueError("no reads in the requested source range")
    batches = []
    with torch.inference_mode():
        for begin in range(0, len(records), 256):
            rows = records[begin:begin + 256]
            batch = make_fastq_batch(
                [encode_base_ids(r.sequence) for r in rows],
                [np.frombuffer(r.quality, dtype=np.uint8) - 33 for r in rows],
                [r.read_index for r in rows], source_name=args.source.name,
            )
            logits = model.forward_full(**fastq_batch_to_tensors(batch, "cpu")).numpy()
            cycles, row_ids = np.nonzero(batch.active_mask.T)
            batches.append((logits[row_ids, cycles], batch.qualities[row_ids, cycles].astype(np.int64)))
            print(f"Prepared {min(begin + 256, len(records))}/{len(records)} real reads", flush=True)
    return batches


def _quantize(batches, function, chunk):
    return [function(logits, symbols, chunk_rows=chunk) for logits, symbols in batches]


def _assert_equal(actual, reference):
    for result, expected in zip(actual, reference):
        if any(not np.array_equal(a, b) for a, b in zip(result, expected)):
            raise AssertionError("integer interval mismatch against reference")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("adapter", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reads", type=int, default=1024)
    parser.add_argument("--skip-reads", type=int, default=8000)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--chunks", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192, 16384, 32768, 65536])
    parser.add_argument("--reference-ref", default="99c8956")
    parser.add_argument("--logits-cache", type=Path)
    args = parser.parse_args()
    if args.reads < 1 or args.skip_reads < 0 or args.repeats < 2 or any(c < 1 for c in args.chunks):
        parser.error("reads/chunks must be positive, skip nonnegative, repeats at least two")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    reference_fn = _reference_quantizer(args.reference_ref)
    checkpoint_sha = sha256_file(args.checkpoint)
    if args.logits_cache is None:
        batches = _real_batches(args, checkpoint_sha)
        cache = args.output / "real_logits.npz"
        np.savez(cache, **{f"{name}_{i}": value for i, batch in enumerate(batches)
                          for name, value in zip(("logits", "symbols"), batch)})
    else:
        cache = args.logits_cache
        with np.load(cache, allow_pickle=False) as saved:
            batches = [(saved[f"logits_{i}"], saved[f"symbols_{i}"])
                       for i in range(len(saved.files) // 2)]
    report = {
        "source": str(args.source.resolve()), "checkpoint_sha256": checkpoint_sha,
        "adapter": str(args.adapter.resolve()), "adapter_sha256": sha256_file(args.adapter),
        "skip_reads": args.skip_reads,
        "requested_reads": args.reads, "quality_symbols": sum(len(y) for _, y in batches),
        "logits_cache": str(cache.resolve()), "logits_cache_sha256": sha256_file(cache),
        "reference_ref": args.reference_ref, "default_chunk_rows": DEFAULT_FLOOR_CHUNK_ROWS,
        "quantizer_source_sha256": sha256_file(Path(__file__).with_name("probability_quantization.py")),
        "batch_positions": [len(y) for _, y in batches],
        "hostname": platform.node(), "architecture": platform.machine(),
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "torch_version": torch.__version__, "cpu_threads": torch.get_num_threads(),
        "thread_environment": {k: os.environ.get(k) for k in
                               ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "trials": [], "summary": {},
    }
    expected = _quantize(batches, reference_fn, 4096)
    cases = [("reference_4096", reference_fn, 4096)] + [
        (f"optimized_{chunk}", logits_to_intervals, chunk) for chunk in args.chunks
    ]
    # Warm each case and validate it before measuring. Alternate forward and
    # reverse orders to reduce systematic warm-cache/order bias.
    for _, function, chunk in cases:
        _assert_equal(_quantize(batches, function, chunk), expected)
    for trial in range(args.repeats):
        order = cases if trial % 2 == 0 else list(reversed(cases))
        for name, function, chunk in order:
            started = time.perf_counter()
            actual = _quantize(batches, function, chunk)
            seconds = time.perf_counter() - started
            _assert_equal(actual, expected)
            report["trials"].append({"trial": trial, "case": name, "seconds": seconds})
        print(f"Timing round {trial + 1}/{args.repeats} completed", flush=True)
    reference_stream = None
    for name, function, chunk in cases:
        seconds = [t["seconds"] for t in report["trials"] if t["case"] == name]
        encoder = RangeEncoder()
        for intervals in _quantize(batches, function, chunk):
            encoder.encode_prevalidated_intervals(*intervals)
        stream = encoder.finish()
        if reference_stream is None:
            reference_stream = stream
        elif stream != reference_stream:
            raise AssertionError("encoded bytes differ from the reference")
        report["summary"][name] = {
            "median_seconds": statistics.median(seconds), "min_seconds": min(seconds),
            "max_seconds": max(seconds), "chunk_rows": chunk,
            "range_stream_bytes": len(stream), "range_stream_sha256": hashlib.sha256(stream).hexdigest(),
        }
    report["integer_intervals_equal"] = True
    report["range_stream_bytes_equal"] = True
    report["recommended_chunk_rows"] = min(
        args.chunks, key=lambda c: report["summary"][f"optimized_{c}"]["median_seconds"]
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
