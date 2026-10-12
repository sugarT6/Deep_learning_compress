"""Compare Python/C++ on identical cached real logits, outside quantizer timers."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import time

import numpy as np

from ._native_range import load_native_kernel, native_library_path
from .probability_quantization import logits_to_intervals
from .range_encoder import RangeEncoder


def _run(backend, batches, copies):
    encoder = RangeEncoder(backend=backend)
    started = time.perf_counter()
    for _ in range(copies):
        for intervals in batches:
            encoder.encode_prevalidated_intervals(*intervals)
    range_seconds = time.perf_counter() - started
    started = time.perf_counter()
    stream = encoder.finish()
    finalize_seconds = time.perf_counter() - started
    return stream, range_seconds, finalize_seconds, encoder.metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logits_cache", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--copies", type=int, default=4,
                        help="replay cached batches into one continuous arithmetic stream")
    args = parser.parse_args()
    if args.repeats < 2 or args.copies < 1:
        parser.error("repeats must be at least two; copies must be positive")
    load_native_kernel()
    with np.load(args.logits_cache, allow_pickle=False) as saved:
        batches = [logits_to_intervals(saved[f"logits_{i}"], saved[f"symbols_{i}"])
                   for i in range(len(saved.files) // 2)]
    if not batches or not sum(len(b[0]) for b in batches):
        parser.error("logits cache is empty")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "hostname": platform.node(), "architecture": platform.machine(),
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "logits_cache": str(args.logits_cache.resolve()),
        "logits_cache_sha256": hashlib.sha256(args.logits_cache.read_bytes()).hexdigest(),
        "native_library": str(native_library_path()),
        "native_library_sha256": hashlib.sha256(native_library_path().read_bytes()).hexdigest(),
        "thread_environment": {key: os.environ.get(key) for key in
                               ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "cached_batch_positions": [len(b[0]) for b in batches],
        "copies": args.copies, "quality_symbols": args.copies * sum(len(b[0]) for b in batches),
        "quantization_in_timer": False, "trials": [], "summary": {},
    }
    expected = _run("python", batches, args.copies)[0]
    if _run("native", batches, args.copies)[0] != expected:
        raise AssertionError("native/reference warmup stream mismatch")
    for trial in range(args.repeats):
        for backend in (("python", "native") if trial % 2 == 0 else ("native", "python")):
            stream, seconds, finalize, metadata = _run(backend, batches, args.copies)
            if stream != expected:
                raise AssertionError(f"{backend} stream mismatch")
            report["trials"].append({"trial": trial, "backend": backend,
                                     "range_seconds": seconds, "finalize_seconds": finalize})
        print(f"Timing round {trial + 1}/{args.repeats} completed", flush=True)
    for backend in ("python", "native"):
        seconds = [t["range_seconds"] for t in report["trials"] if t["backend"] == backend]
        report["summary"][backend] = {
            "median_range_seconds": statistics.median(seconds),
            "min_range_seconds": min(seconds), "max_range_seconds": max(seconds),
        }
    report["coder_speedup"] = (report["summary"]["python"]["median_range_seconds"] /
                               report["summary"]["native"]["median_range_seconds"])
    report["range_stream_bytes_equal"] = True
    report["range_stream_bytes"] = len(expected)
    report["range_payload_bits"] = metadata.payload_bit_count
    report["range_stream_sha256"] = hashlib.sha256(expected).hexdigest()
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"summary": report["summary"], "coder_speedup": report["coder_speedup"],
                      "range_stream_bytes_equal": True}, indent=2))


if __name__ == "__main__":
    main()
