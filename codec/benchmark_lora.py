"""Compare baseline, time-budgeted head-only, and last-block LoRA on 8000 reads."""
import argparse
import gzip
import itertools
import json
from pathlib import Path
import statistics
import tempfile

import numpy as np
import torch

from .checkpoint import load_training_checkpoint
from .container import sha256_file
from .decode import decode_fastq
from .encode import encode_fastq
from .encode_fastpath import selected_quantized_bits
from .fastq_stream import encode_base_ids, iter_fastq_records, make_fastq_batch
from .head_adapter import HeadAdaptationConfig, adapt_output_head
from .model import fastq_batch_to_tensors
from .probability_quantization import logits_to_cdfs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path, help="new result directory")
    parser.add_argument("--lora-steps", type=int, default=1000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    device = torch.device("cuda")
    digest = sha256_file(args.checkpoint)
    report = dict(device=torch.cuda.get_device_name(), torch_version=torch.__version__,
                  base_sha256=digest, source=str(args.source.resolve()), trials=[])
    iterator = iter_fastq_records(args.source)
    try:
        records = list(itertools.islice(iterator, 12000))
    finally:
        iterator.close()
    if len(records) != 12000:
        raise ValueError("Benchmark requires at least 12000 reads")

    def save():
        (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))

    lora_artifact = None
    for trial, label in enumerate(("baseline", "head_budget", "lora", "lora", "head_budget", "baseline")):
        model = load_training_checkpoint(args.checkpoint, device=device).model.eval()
        cfg = HeadAdaptationConfig(max_reads=8000, max_symbols=1200000,
            steps=100000 if label == "head_budget" else 1000, max_seconds=20,
            head_type="residual", residual_dim=64, cross_layer=True, cross_dim=16,
            lora=label == "lora", lora_rank=4, lora_steps=args.lora_steps)
        adapter, fit = adapt_output_head(model, args.source, device=device, batch_reads=256,
            total=65536, base_sha256=digest, config=cfg)
        result = dict(mode=label, fit=fit)
        report["trials"].append(result)
        save()
        print(json.dumps(dict(mode=label, seconds=fit["seconds"], steps=fit["steps_completed"],
            lora_steps=fit.get("lora_steps_completed"), selected=fit.get("lora_selected_step"),
            reason=fit["reason"], validation=fit["after_validation_bits_per_quality"])), flush=True)
        if adapter is None:
            raise RuntimeError("Candidate not accepted; see partial report")
        path = args.output / f"{trial}_{label}.adapter.json"
        path.write_text(json.dumps(dict(format="direct-quality-head-adaptation-artifact", version=1,
            base_checkpoint_sha256=digest, adapter=adapter, report=fit, source=str(args.source.resolve())), sort_keys=True))
        result["artifact"] = str(path.resolve())
        result["adapter_version"] = adapter["version"]
        result["container_adapter_bytes"] = len(json.dumps({"head_adapter": adapter}, sort_keys=True,
            separators=(",", ":")).encode()) - 1 + len('"probability_profile":null,')
        if label == "lora" and lora_artifact is None:
            lora_artifact = path
        bits, symbols = 0.0, 0
        for begin in range(8000, 12000, 256):
            rows = records[begin:min(begin+256, 12000)]
            batch = make_fastq_batch([encode_base_ids(r.sequence) for r in rows],
                [np.frombuffer(r.quality, dtype=np.uint8)-33 for r in rows],
                [r.read_index for r in rows], source_name=args.source.name)
            tensors = fastq_batch_to_tensors(batch, device)
            with torch.inference_mode():
                logits = model.forward_full(**tensors)[tensors["active_mask"]].cpu().numpy()
                y = tensors["qualities"][tensors["active_mask"]].cpu().numpy()
                bits += selected_quantized_bits(y, logits_to_cdfs(logits, total=65536), 65536)
                symbols += y.size
        result["later_bits_per_quality"] = bits/symbols
        result["later_symbols"] = symbols
        print(json.dumps(dict(mode=label, later_bits_per_quality=bits/symbols)), flush=True)
        save()
        del model
    report["summary"] = {}
    for label in ("baseline", "head_budget", "lora"):
        trials = [t for t in report["trials"] if t["mode"] == label]
        report["summary"][label] = dict(
            mean_seconds=statistics.mean(t["fit"]["seconds"] for t in trials),
            mean_later_bits_per_quality=statistics.mean(t["later_bits_per_quality"] for t in trials),
            container_adapter_bytes=trials[0]["container_adapter_bytes"])
    save()
    with tempfile.TemporaryDirectory(prefix="lora_roundtrip_") as temp:
        temp = Path(temp)
        raw = b"".join(r.to_bytes() for r in records[:257])
        sample, target, restored = temp/"sample.fq.gz", temp/"sample.fqdc", temp/"restored.fq"
        sample.write_bytes(gzip.compress(raw))
        kw = dict(device=device, batch_reads=256, progress=False, verify_cdf=True)
        stats = encode_fastq(sample, target, args.checkpoint, **kw, head_adapter_path=lora_artifact)
        decode_fastq(target, restored, args.checkpoint, **kw)
        assert restored.read_bytes() == raw
        report["roundtrip"] = dict(reads=257, byte_exact=True, full_step_integer_cdf_verified=True,
            container_adapter_bytes=stats.head_adaptation["container_adapter_bytes"])
    assert sha256_file(args.checkpoint) == digest
    save()
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
