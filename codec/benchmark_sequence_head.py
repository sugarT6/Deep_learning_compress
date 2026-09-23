"""Small paired adaptation benchmark; does not encode the full input file."""
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
from .head_adapter import HeadAdaptationConfig, adapt_output_head, apply_head_adapter
from .model import causal_quality_windows, fastq_batch_to_tensors
from .probability_quantization import logits_to_cdfs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path, help="new, non-existing result directory")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    device = torch.device("cuda")
    digest = sha256_file(args.checkpoint)
    report = dict(device=torch.cuda.get_device_name(), torch_version=torch.__version__,
                  base_sha256=digest, source=str(args.source.resolve()), trials=[])

    def save():
        (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))

    adapters, artifacts, first_steps = {}, {}, {}
    # ABBA controls some warm-up and load drift. Both share the same seed/split.
    for sequence in (False, True, True, False):
        label = "cross16_sequence" if sequence else "cross16"
        model = load_training_checkpoint(args.checkpoint, device=device).model.eval()
        cfg = HeadAdaptationConfig(max_reads=8000, max_symbols=1200000, steps=1000,
            max_seconds=20, head_type="residual", residual_dim=64, cross_layer=True,
            cross_dim=16, sequence_branch=sequence)
        adapter, fit = adapt_output_head(model, args.source, device=device, batch_reads=256,
            total=65536, base_sha256=digest, config=cfg)
        report["trials"].append(dict(mode=label, fit=fit))
        save()
        print(json.dumps(dict(mode=label, seconds=fit["seconds"], steps=fit["steps_completed"],
            accepted=fit["accepted"], validation=fit["after_validation_bits_per_quality"])), flush=True)
        if adapter is None:
            raise RuntimeError("Candidate not accepted; see partial report")
        if label not in adapters:
            adapters[label] = adapter
            first_steps[label] = fit["steps_completed"]
            artifacts[label] = args.output / (label + ".adapter.json")
            artifacts[label].write_text(json.dumps(dict(format="direct-quality-head-adaptation-artifact",
                version=1, base_checkpoint_sha256=digest, adapter=adapter, report=fit,
                source=str(args.source.resolve())), sort_keys=True))
        elif fit["steps_completed"] == first_steps[label] == 1000:
            assert adapters[label] == adapter, "Fixed-step repeated fit changed tensors"
        del model

    report["summary"] = {}
    for label, adapter in adapters.items():
        fits = [t["fit"] for t in report["trials"] if t["mode"] == label]
        report["summary"][label] = dict(
            mean_adaptation_seconds=statistics.mean(f["seconds"] for f in fits),
            mean_optimization_seconds=statistics.mean(f["stage_seconds"]["optimization"] for f in fits),
            parameter_bytes=adapter["parameter_bytes"],
            adapter_metadata_bytes=len(json.dumps(adapter, sort_keys=True, separators=(",", ":")).encode()))
    records_iter = iter_fastq_records(args.source)
    try:
        records = list(itertools.islice(records_iter, 12000))
    finally:
        records_iter.close()
    if len(records) < 12000:
        raise ValueError("Benchmark requires at least 12000 reads")
    model = load_training_checkpoint(args.checkpoint, device=device).model.eval()
    heads = {}
    for label, adapter in adapters.items():
        apply_head_adapter(model, adapter, digest)
        heads[label] = model.output_head
    bits, symbols = dict.fromkeys(heads, 0.0), 0
    for begin in range(8000, 12000, 256):
        rows = records[begin:min(begin + 256, 12000)]
        batch = make_fastq_batch([encode_base_ids(r.sequence) for r in rows],
            [np.frombuffer(r.quality, dtype=np.uint8) - 33 for r in rows],
            [r.read_index for r in rows], source_name=args.source.name)
        tensors = fastq_batch_to_tensors(batch, device)
        with torch.inference_mode():
            h = model.forward_cross_layer_features(**tensors)
            sequence_h = torch.cat((h, causal_quality_windows(tensors["qualities"]).to(h.dtype)), dim=-1)
            active = tensors["active_mask"]
            labels = tensors["qualities"][active].cpu().numpy()
            for label, head in heads.items():
                inputs = sequence_h if label == "cross16_sequence" else h
                logits = head(inputs[active]).cpu().numpy()
                bits[label] += selected_quantized_bits(labels, logits_to_cdfs(logits, total=65536), 65536)
            symbols += labels.size
    report["later_region"] = dict(read_range_zero_based_half_open=[8000, 12000], symbols=symbols,
        bits_per_quality={k: v / symbols for k, v in bits.items()})
    save()
    print(json.dumps(report["later_region"]), flush=True)
    del model, heads
    with tempfile.TemporaryDirectory(prefix="sequence_roundtrip_") as temp:
        temp = Path(temp)
        raw = b"".join(r.to_bytes() for r in records[:257])
        sample, target, restored = temp / "sample.fq.gz", temp / "sample.fqdc", temp / "restored.fq"
        sample.write_bytes(gzip.compress(raw))
        kw = dict(device=device, batch_reads=256, progress=False, verify_cdf=True)
        stats = encode_fastq(sample, target, args.checkpoint, **kw,
                            head_adapter_path=artifacts["cross16_sequence"])
        decode_fastq(target, restored, args.checkpoint, **kw)
        assert restored.read_bytes() == raw
        report["roundtrip"] = dict(reads=257, byte_exact=True, full_step_integer_cdf_verified=True,
            container_adapter_bytes=stats.head_adaptation["container_adapter_bytes"])
    assert sha256_file(args.checkpoint) == digest
    save()
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
