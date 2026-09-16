"""Bounded train-validation-only fusion study; no training or range encoding."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .checkpoint import load_training_checkpoint
from .datasets import TRAIN_DATASETS
from .encode_fastpath import fuse_batch_logits
from .evaluate import DatasetMetrics, choose_device, summarize_metric_group
from .mixture_prior import MixturePriorConfig, MixturePriorState, history_contexts
from .model import fastq_batch_to_tensors
from .online_prior import OnlinePriorConfig, OnlinePriorState
from .probability_quantization import TOTAL, logits_to_cdfs
from .training_cache import DEFAULT_CACHE_DIR, TrainingCacheReader, sha256_file


def candidate_configs():
    result = {"neural_only": None, "current": OnlinePriorConfig(),
        "online_hierarchical": None, "online_enriched": None}
    for mode in ("hierarchical", "enriched"):
        for alpha in (0.25, 0.5, 0.75):
            result[f"mix_{mode}_a{alpha}_t1"] = MixturePriorConfig(alpha=alpha, context_mode=mode)
    for temperature in (0.85, 1.15):
        result[f"mix_enriched_a0.5_t{temperature}"] = MixturePriorConfig(temperature=temperature)
    return result


def candidate_scores(logits, qualities, rows, cycles, base, enriched, configs):
    previous, _, _ = history_contexts(qualities, rows, cycles)
    hierarchical = base.probabilities(previous, cycles)
    rich = enriched.probabilities(qualities, rows, cycles)
    scores = {"neural_only": logits,
        "current": fuse_batch_logits(base, logits, previous, cycles),
        "online_hierarchical": np.log(hierarchical), "online_enriched": np.log(rich)}
    for name, config in configs.items():
        if not isinstance(config, MixturePriorConfig):
            continue
        z = np.asarray(logits, dtype=np.float64) / config.temperature
        if not base.observed_symbols:
            scores[name] = z
        else:
            p = np.exp(z - z.max(axis=1, keepdims=True))
            p /= p.sum(axis=1, keepdims=True)
            online = rich if config.context_mode == "enriched" else hierarchical
            scores[name] = np.log((1 - config.alpha) * p + config.alpha * online)
    return scores


def validate_manifest(payload, datasets):
    expected = {d.accession for d in TRAIN_DATASETS}
    if {d.accession for d in datasets} != expected or any(not d.training_source for d in datasets):
        raise ValueError("fusion selection requires exactly the fixed ten training datasets")
    entries = payload["data_split"]["datasets"]
    if set(entries) != expected:
        raise ValueError("checkpoint split must contain exactly the ten training datasets")
    for entry in entries.values():
        train_start, train_stop = entry["train_range"]
        val_start, val_stop = entry["validation_range"]
        if not 0 <= train_start < train_stop <= val_start < val_stop <= entry["read_count"]:
            raise ValueError("invalid or overlapping training/validation ranges")
    return entries


def evaluate_fusion(checkpoint, cache_dir, *, batch_reads=256, warmup_reads=1024,
                    score_reads=4096, device=torch.device("cpu"), offset=0):
    if not 1 <= batch_reads <= 256 or warmup_reads < 0 or score_reads <= 0 or offset < 0:
        raise ValueError("invalid bounded evaluation counts")
    if warmup_reads % batch_reads:
        raise ValueError("warmup must end on a completed codec batch")
    loaded = load_training_checkpoint(checkpoint, device=device)
    entries = validate_manifest(loaded.payload, TRAIN_DATASETS)
    model = loaded.model.eval()
    configs = candidate_configs()
    metrics = {name: [] for name in configs}
    breakdown = {name: {} for name in configs}
    ranges = {}
    forward_calls = 0
    with torch.inference_mode():
        for dataset in TRAIN_DATASETS:
            entry = entries[dataset.accession]
            with TrainingCacheReader(dataset.cache_path(cache_dir)) as reader:
                if reader.metadata.source_sha256 != entry["source_sha256"] or reader.metadata.read_count != entry["read_count"]:
                    raise ValueError("checkpoint/cache fingerprint mismatch")
                start = int(entry["validation_range"][0]) + offset
                score_start = start + warmup_reads
                stop = score_start + score_reads
                if stop > entry["validation_range"][1]:
                    raise ValueError(f"{dataset.accession}: requested window exceeds validation suffix")
                ranges[dataset.accession] = {"warmup": [start, score_start], "score": [score_start, stop],
                    "source_sha256": entry["source_sha256"]}
                base = OnlinePriorState()
                enriched = MixturePriorState(MixturePriorConfig())
                totals = {name: 0.0 for name in configs}
                symbols_count = 0
                for batch_start in range(start, stop, batch_reads):
                    batch = reader.read_range(batch_start, min(batch_start + batch_reads, stop))
                    if batch_start >= score_start:
                        logits = model.forward_full(**fastq_batch_to_tensors(batch, device)).cpu().numpy()
                        forward_calls += 1
                        cycles, rows = np.nonzero(batch.active_mask.T)
                        symbols = batch.qualities[rows, cycles].astype(np.int64)
                        scores = candidate_scores(logits[rows, cycles], batch.qualities, rows, cycles, base, enriched, configs)
                        previous, _, _ = history_contexts(batch.qualities, rows, cycles)
                        support = base.prev_q_counts[previous].sum(axis=1)
                        groups = {"q": symbols, "cycle_bin": cycles // 8, "prev_q": previous,
                            "support_bin": np.searchsorted([1, 16, 256, 4096], support, side="right")}
                        for name, values in scores.items():
                            cdfs = logits_to_cdfs(values)
                            ix = np.arange(symbols.size)
                            frequencies = cdfs[ix, symbols + 1] - cdfs[ix, symbols]
                            bits = -np.log2(frequencies.astype(np.float64) / TOTAL)
                            totals[name] += float(bits.sum(dtype=np.float64))
                            target = breakdown[name].setdefault(dataset.accession, {})
                            for group, ids in groups.items():
                                counts = np.bincount(ids)
                                sums = np.bincount(ids, weights=bits)
                                table = target.setdefault(group, {})
                                for key in np.flatnonzero(counts):
                                    record = table.setdefault(str(key), {"symbols": 0, "bits": 0.0})
                                    record["symbols"] += int(counts[key])
                                    record["bits"] += float(sums[key])
                        symbols_count += symbols.size
                    # No candidate can see the current batch before every score is measured.
                    base.update_batch(batch.qualities, batch.active_mask)
                    enriched.update_batch(batch.qualities, batch.active_mask)
                for name in configs:
                    metrics[name].append(DatasetMetrics(dataset.accession, dataset.platform_family,
                        "validation", False, score_reads, symbols_count, totals[name], totals[name] / symbols_count))
            print(f"validation {dataset.accession}: {score_reads} scored reads", flush=True)
    summaries = {name: summarize_metric_group(values) for name, values in metrics.items()}
    deployable = [name for name, config in configs.items() if config is not None or name == "neural_only"]
    winner = min(deployable, key=lambda name: summaries[name]["dataset_macro_bits_per_quality"])
    return {"schema_version": 1, "checkpoint_sha256": sha256_file(checkpoint),
        "selection_scope": "ten_training_validation_only", "selection_metric": "dataset_macro_bits_per_quality",
        "selection_status": "provisional_window_result_not_unseen_evidence",
        "batch_reads": batch_reads, "warmup_reads": warmup_reads, "score_reads": score_reads,
        "validation_offset": offset, "reset_rule": "zero_state_at_window_start_warmup_unscored",
        "ranges": ranges, "model_forward_calls": forward_calls, "quantization_total": TOTAL,
        "summaries": summaries, "per_dataset": {name: [m.to_dict() for m in values] for name, values in metrics.items()},
        "breakdown": breakdown, "candidate_profiles": {name: (c.to_profile_metadata() if c else None) for name, c in configs.items() if name in deployable},
        "best_deployable_candidate": winner,
        "selected_profile": configs[winner].to_profile_metadata() if configs[winner] else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-reads", type=int, default=256)
    parser.add_argument("--warmup-reads", type=int, default=1024)
    parser.add_argument("--score-reads", type=int, default=4096)
    parser.add_argument("--validation-offset", type=int, default=0)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("output directory must be new")
    report = evaluate_fusion(args.checkpoint, args.cache_dir, batch_reads=args.batch_reads,
        warmup_reads=args.warmup_reads, score_reads=args.score_reads,
        device=choose_device(args.device), offset=args.validation_offset)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "fusion_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (args.output_dir / "selected_profile.json").write_text(json.dumps(report["selected_profile"], indent=2) + "\n")
    for name, summary in report["summaries"].items():
        print(f"{name}: macro={summary['dataset_macro_bits_per_quality']:.8f}")
    print("provisional best:", report["best_deployable_candidate"])


if __name__ == "__main__":
    main()
