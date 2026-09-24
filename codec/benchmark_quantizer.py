"""Read-only source, CPU real-logit quantizer/range benchmark; no full-file encode."""
import argparse
import gzip
import itertools
import json
from pathlib import Path
import statistics
import tempfile
import time

import numpy as np
import torch

from .checkpoint import load_training_checkpoint
from .container import sha256_file
from .head_adapter import apply_head_adapter
from .fastq_stream import iter_fastq_records, make_fastq_batch, encode_base_ids
from .model import fastq_batch_to_tensors
from .probability_quantization import logits_to_cdfs, logits_to_intervals
from .range_encoder import RangeEncoder
from .encode import encode_fastq
from .decode import decode_fastq


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("adapter", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    model = load_training_checkpoint(args.checkpoint, device=torch.device("cpu")).model.eval()
    artifact = json.loads(args.adapter.read_text())
    apply_head_adapter(model, artifact["adapter"], sha256_file(args.checkpoint))
    iterator = iter_fastq_records(args.source)
    try:
        records = list(itertools.islice(iterator, 1024))
    finally:
        iterator.close()
    batches = []
    with torch.inference_mode():
        for start in range(0, len(records), 256):
            rows = records[start:start+256]
            batch = make_fastq_batch([encode_base_ids(r.sequence) for r in rows],
                [np.frombuffer(r.quality, dtype=np.uint8)-33 for r in rows],
                [r.read_index for r in rows], source_name=args.source.name)
            logits = model.forward_full(**fastq_batch_to_tensors(batch, "cpu")).numpy()
            cycles, row_ids = np.nonzero(batch.active_mask.T)
            batches.append((logits[row_ids, cycles], batch.qualities[row_ids, cycles].astype(np.int64)))
    del model
    report = dict(reads=len(records), quality_symbols=sum(len(y) for _, y in batches),
                  device="cpu", reused_adapter=str(args.adapter), trials=[])
    for mode in ("v1", "floor_cdf", "floor_direct", "floor_direct", "floor_cdf", "v1"):
        encoder = RangeEncoder()
        quant_seconds = encode_seconds = bits = 0.0
        for logits, symbols in batches:
            start = time.perf_counter()
            if mode == "floor_direct":
                lo, hi, totals = logits_to_intervals(logits, symbols)
            else:
                cdfs = logits_to_cdfs(logits, version=1 if mode == "v1" else 2)
                indices = np.arange(len(symbols))
                lo, hi, totals = cdfs[indices, symbols], cdfs[indices, symbols+1], cdfs[:, -1]
            quant_seconds += time.perf_counter()-start
            start = time.perf_counter()
            if mode == "v1":
                encoder.encode_prevalidated_batch(symbols, cdfs, total=65536)
            else:
                encoder.encode_prevalidated_intervals(lo, hi, totals)
            encode_seconds += time.perf_counter()-start
            bits += float(-np.log2((hi-lo).astype(np.float64)/totals).sum())
        stream = encoder.finish()
        item = dict(mode=mode, quantization_seconds=quant_seconds, range_seconds=encode_seconds,
                    bits_per_quality=bits/report["quality_symbols"], range_stream_bytes=len(stream))
        report["trials"].append(item)
        print(json.dumps(item), flush=True)
    report["summary"] = {}
    for mode in ("v1", "floor_cdf", "floor_direct"):
        trials = [t for t in report["trials"] if t["mode"] == mode]
        report["summary"][mode] = {k: statistics.mean(t[k] for t in trials) for k in
            ("quantization_seconds", "range_seconds", "bits_per_quality", "range_stream_bytes")}
    with tempfile.TemporaryDirectory(prefix="floor_roundtrip_") as temporary:
        temp = Path(temporary)
        raw = b"".join(r.to_bytes() for r in records[:9])
        source, target, restored = temp/"sample.fq.gz", temp/"sample.fqdc", temp/"restored.fq"
        source.write_bytes(gzip.compress(raw))
        kw = dict(device=torch.device("cpu"), batch_reads=8, progress=False, verify_cdf=True)
        encode_fastq(source, target, args.checkpoint, **kw, head_adapter_path=args.adapter, quantization_version=2)
        decode_fastq(target, restored, args.checkpoint, **kw)
        assert restored.read_bytes() == raw
        report["roundtrip"] = dict(reads=9, byte_exact=True, interval_and_full_step_cdf_verified=True)
    (args.output/"report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
