"""Matched-data ordinary vs first-order adapter-aware continuation experiments.

Both modes load shared weights only, use the same file-balanced batches, and
retain the ordinary codec model architecture. The adaptation branch learns a
temporary head/all-layer LoRA on support reads, then transfers query gradients
to the shared model without differentiating through the inner optimizer.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import sys
import tempfile
import time
from typing import Optional, Sequence

import numpy as np
import torch

from .checkpoint import load_training_checkpoint, save_training_checkpoint
from .container import sha256_file
from .datasets import EXPANDED_FAMILY_WEIGHTS, EXPANDED_TRAIN_DATASETS, EXPANDED_UNSEEN_DATASETS
from .evaluate import choose_device
from .fastq_stream import FastqBatch, make_fastq_batch
from .head_adapter import _new_head
from .model import fastq_batch_to_tensors, masked_cross_entropy
from .suffix_lora_adapter import install_suffix_lora
from .train import BalancedTrainingSampler, _write_json_atomic

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INITIAL = ROOT / "codec/runs/direct_quality_4layer_expanded20_200k_S1pPAq/best.pt"
DEFAULT_OUTPUT_ROOT = ROOT / "codec/runs/adaptation_training_ab_20261027"
FORMAT = "direct-quality-adaptation-ab-training"
SELECTION_METRIC = "short_adaptation_macro_bits_per_quality"


@dataclass(frozen=True)
class ABConfig:
    epochs: int = 5
    steps_per_epoch: int = 2000
    batch_reads: int = 256
    episode_blocks: int = 8
    inner_batch_reads: int = 16
    validation_episodes_per_file: int = 4
    learning_rate: float = 1e-5
    min_learning_rate: float = 1e-6
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    inner_learning_rate: float = 0.0003
    inner_head_learning_rate: float = 0.001
    anchor_strength: float = 0.01
    train_fraction: float = 0.9
    lora_rank: int = 4
    residual_dim: int = 64
    cross_dim: int = 16
    seed: int = 20261027
    adapter_seed: int = 20260921

    def __post_init__(self):
        for name in ("epochs", "steps_per_epoch", "batch_reads", "episode_blocks", "inner_batch_reads",
                     "validation_episodes_per_file", "lora_rank", "residual_dim", "cross_dim"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 2 <= self.batch_reads <= 256 or self.inner_batch_reads > 256:
            raise ValueError("batch_reads must be 2..256 and inner_batch_reads <=256")
        if self.episode_blocks > self.batch_reads:
            raise ValueError("episode_blocks must not exceed batch_reads")
        if self.lora_rank > 16 or max(self.residual_dim, self.cross_dim) > 128:
            raise ValueError("adapter rank/dimensions exceed the codec limits")
        for name in ("seed", "adapter_seed"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in ("learning_rate", "min_learning_rate", "grad_clip", "inner_learning_rate",
                     "inner_head_learning_rate", "weight_decay", "anchor_strength", "train_fraction"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not 0 < self.min_learning_rate <= self.learning_rate or self.grad_clip <= 0:
            raise ValueError("invalid shared learning rate, floor or gradient clip")
        if min(self.inner_learning_rate, self.inner_head_learning_rate) <= 0:
            raise ValueError("inner learning rates must be positive")
        if self.weight_decay < 0 or self.anchor_strength < 0 or not 0 < self.train_fraction < 1:
            raise ValueError("invalid regularization or train fraction")


def subset_batch(batch, rows):
    rows = np.asarray(rows, dtype=np.int64)
    if rows.ndim != 1 or not len(rows):
        raise ValueError("episode subset must be nonempty")
    return FastqBatch(bases=batch.bases[rows].copy(), qualities=batch.qualities[rows].copy(),
        lengths=batch.lengths[rows].copy(), active_mask=batch.active_mask[rows].copy(),
        read_count=len(rows), read_indices=batch.read_indices[rows].copy(), source_name=batch.source_name)


def split_episode(batch, *, seed, step):
    if batch.read_count < 2 or len(np.unique(batch.read_indices)) != batch.read_count:
        raise ValueError("each episode needs at least two distinct reads")
    rng = np.random.default_rng(np.random.SeedSequence([seed, step]))
    order = rng.permutation(batch.read_count)
    middle = batch.read_count // 2
    return subset_batch(batch, order[:middle]), subset_batch(batch, order[middle:])


def spread_episode(sample, sampler, config, step):
    """Use the balanced file schedule, then draw chunks across its training range.

    The sampler's contiguous batch is only a scheduling anchor. Its read count
    determines this episode's size (including tails); actual model inputs come
    from separated strata. Exposure hashes describe the actual input reads.
    """
    count = sample.batch.read_count
    accession = sample.dataset.accession
    reader, stop = sampler.reader_for(accession), sampler.train_stop_for(accession)
    blocks = min(config.episode_blocks, count)
    rng = np.random.default_rng(np.random.SeedSequence([config.seed, step, 441]))
    bases, qualities, indices = [], [], []
    for block in range(blocks):
        quota = count // blocks + int(block < count % blocks)
        left, right = block * stop // blocks, (block + 1) * stop // blocks
        if right - left < quota:
            raise ValueError("training stratum is too small for the requested episode")
        begin = int(rng.integers(left, right - quota + 1))
        part = reader.read_range(begin, begin + quota)
        for row, length in enumerate(part.lengths):
            bases.append(part.bases[row, :length])
            qualities.append(part.qualities[row, :length])
            indices.append(part.read_indices[row])
    return make_fastq_batch(bases, qualities, indices, source_name=sample.batch.source_name)


def sampling_description(sampler, config):
    return dict(strategy="fixed_family_schedule_spread_file_episodes",
                read_sampling="one_random_contiguous_chunk_per_training_stratum; repeats_allowed_across_episodes",
                episode_blocks=config.episode_blocks, anchor_schedule=sampler.config_dict())


def sampling_statistics(sampler):
    raw = sampler.statistics_dict()
    return dict(total_batches=raw["total_batches"], total_reads=raw["total_reads"],
        train_read_count=raw["train_read_count"], unique_reads=None,
        note="Read counts are exposures, not unique coverage; model inputs are spread chunks, not scheduling anchors.",
        families=raw["families"], datasets={name: {k: entry[k] for k in ("count", "proportion", "reads")}
                                            for name, entry in raw["datasets"].items()})


def batch_loss(model, batch, device):
    tensors = fastq_batch_to_tensors(batch, device)
    count = int(tensors["active_mask"].sum().item())
    if not count:
        raise ValueError("an episode subset has no quality symbols")
    loss = masked_cross_entropy(model.forward_full(**tensors), tensors["qualities"], tensors["active_mask"])
    if not torch.isfinite(loss):
        raise ValueError("nonfinite episode cross entropy")
    return loss, count


def prepare_adapter(shared, config):
    if shared.config.num_layers < 4 or type(shared.output_head) is not torch.nn.Linear:
        raise ValueError("adapter-aware continuation requires >=4 layers and the ordinary shared linear head")
    candidate = copy.deepcopy(shared).eval().requires_grad_(False)
    original = candidate.output_head
    candidate.output_head = _new_head(shared.config.d_model, config.residual_dim,
        cross_dim=config.cross_dim, device=original.weight.device, dtype=original.weight.dtype,
        seed=config.adapter_seed)
    with torch.no_grad():
        candidate.output_head.weight.copy_(original.weight)
        candidate.output_head.bias.copy_(original.bias)
    install_suffix_lora(candidate, shared.config.num_layers, config.lora_rank, config.adapter_seed)
    candidate.transformer.norm.requires_grad_(True)
    candidate.output_head.requires_grad_(True)
    return candidate


def fit_support(shared, support, config, device):
    """One pass over support reads, resetting all temporary adaptation state."""
    candidate = prepare_adapter(shared, config)
    trainable = {name: p for name, p in candidate.named_parameters() if p.requires_grad}
    anchors = {name: p.detach().clone() for name, p in trainable.items()} if config.anchor_strength else {}
    optimizer = torch.optim.Adam([
        {"params": [p for name, p in trainable.items() if not name.startswith("output_head.")],
         "lr": config.inner_learning_rate},
        {"params": list(candidate.output_head.parameters()), "lr": config.inner_head_learning_rate},
    ])
    symbols, steps = 0, 0
    with torch.enable_grad():
        for begin in range(0, support.read_count, config.inner_batch_reads):
            batch = subset_batch(support, np.arange(begin, min(begin + config.inner_batch_reads, support.read_count)))
            optimizer.zero_grad(set_to_none=True)
            ce, count = batch_loss(candidate, batch, device)
            loss = ce
            if anchors:
                loss = loss + config.anchor_strength * sum(
                    (p - anchors[name]).square().mean() for name, p in trainable.items())
            if not torch.isfinite(loss):
                raise ValueError("nonfinite support loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(trainable.values()), config.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            symbols += count
            steps += 1
    candidate.zero_grad(set_to_none=True)
    return candidate, {"inner_steps": steps, "inner_backward_symbols": symbols}


def shared_parameter_map(shared, adapted):
    parameters = dict(adapted.named_parameters())
    result = {}
    for name, original in shared.named_parameters():
        adapted_name = name
        if adapted_name not in parameters:
            module, weight = name.rsplit(".", 1)
            adapted_name = f"{module}.parametrizations.{weight}.original"
        parameter = parameters.get(adapted_name)
        if parameter is None or parameter.shape != original.shape:
            raise ValueError(f"cannot map query gradient to shared parameter: {name}")
        result[name] = parameter
    return result


def query_gradients(shared, adapted, query, device):
    """FO approximation: freeze fitted file variables; map query gradients to theta.

    Adapted norm and linear-head values map back to their shared initialization
    with the first-order identity approximation. Original matrices underlying
    LoRA map to their shared names. Extra head branches and A/B factors never
    become shared parameters, and support optimizer graphs are not retained.
    """
    shared.zero_grad(set_to_none=True)
    adapted.zero_grad(set_to_none=True)
    adapted.eval().requires_grad_(False)
    mapping = shared_parameter_map(shared, adapted)
    for parameter in mapping.values():
        parameter.requires_grad_(True)
    loss, count = batch_loss(adapted, query, device)
    loss.backward()
    for name, parameter in shared.named_parameters():
        grad = mapping[name].grad
        parameter.grad = None if grad is None else grad.detach().clone()
    return float(loss.detach()), count


def outer_update(shared, optimizer, batch, *, mode, config, device, step):
    support, query = split_episode(batch, seed=config.seed, step=step)
    optimizer.zero_grad(set_to_none=True)
    shared.eval()  # Same no-dropout training regime in both branches.
    if mode == "ordinary":
        loss, count = batch_loss(shared, batch, device)
        loss.backward()
        value = float(loss.detach())
        stats = {"inner_steps": 0, "inner_backward_symbols": 0}
    elif mode == "adaptation":
        candidate, stats = fit_support(shared, support, config, device)
        value, count = query_gradients(shared, candidate, query, device)
        del candidate
    else:
        raise ValueError("mode must be ordinary or adaptation")
    torch.nn.utils.clip_grad_norm_(shared.parameters(), config.grad_clip, error_if_nonfinite=True)
    optimizer.step()
    stats.update(objective_nats=value, outer_backward_symbols=count,
                 support_symbols=int(support.active_mask.sum()), query_symbols=int(query.active_mask.sum()))
    digest = hashlib.sha256(batch.source_name.encode() + b"\0")
    for part in (support, query):
        digest.update(np.asarray(part.read_indices, dtype="<i8").tobytes())
    stats["episode_sha256"] = digest.hexdigest()
    return stats


def validation_episodes(sampler, config):
    episodes, indices = {}, {}
    required = config.batch_reads * config.validation_episodes_per_file
    for index, dataset in enumerate(sampler.datasets):
        accession = dataset.accession
        reader, start = sampler.reader_for(accession), sampler.train_stop_for(accession)
        available = reader.metadata.read_count - start
        if start < 2 or start % config.batch_reads == 1:
            raise ValueError(f"{accession}: training split has a one-read tail; choose another --batch-reads")
        if available < required:
            raise ValueError(f"{accession}: validation suffix needs {required} distinct reads")
        rng = np.random.default_rng(np.random.SeedSequence([config.seed, index, 993]))
        chosen = rng.choice(available, size=required, replace=False) + start
        episodes[accession], indices[accession] = [], []
        for episode in range(config.validation_episodes_per_file):
            batch = reader.read_indices(chosen[episode * config.batch_reads:(episode + 1) * config.batch_reads])
            support, query = split_episode(batch, seed=config.seed + 1, step=index * config.validation_episodes_per_file + episode)
            episodes[accession].append((support, query))
            indices[accession].append(dict(support=support.read_indices.tolist(), query=query.read_indices.tolist()))
    return episodes, indices


def evaluate_proxy(shared, episodes, config, device, *, progress=False):
    result = {}
    items = episodes.items()
    if progress and tqdm:
        items = tqdm(items, desc="validate short adaptation", unit="file", file=sys.stderr)
    for accession, pairs in items:
        cold_nats, adapted_nats, symbols = 0., 0., 0
        for support, query in pairs:
            with torch.no_grad():
                cold_loss, count = batch_loss(shared, query, device)
            candidate, _ = fit_support(shared, support, config, device)
            with torch.no_grad():
                loss, actual_count = batch_loss(candidate, query, device)
            if count != actual_count:
                raise ValueError("validation query symbol count changed")
            cold_nats += float(cold_loss) * count
            adapted_nats += float(loss) * count
            symbols += count
            del candidate
        result[accession] = dict(query_symbols=symbols, unadapted_bpq=cold_nats / symbols / math.log(2.),
                                 short_adapted_bpq=adapted_nats / symbols / math.log(2.))
    return {SELECTION_METRIC: float(np.mean([v["short_adapted_bpq"] for v in result.values()])),
            "unadapted_macro_bits_per_quality": float(np.mean([v["unadapted_bpq"] for v in result.values()])),
            "per_dataset": result}


def learning_rate_at(step, config):
    fraction = (step - 1) / max(config.epochs * config.steps_per_epoch - 1, 1)
    return config.min_learning_rate + .5 * (config.learning_rate - config.min_learning_rate) * (1 + math.cos(math.pi * fraction))


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_continuation(mode, *, output_dir, cache_dir, device="cpu", config=ABConfig(),
                     init_from=None, resume=None, datasets=EXPANDED_TRAIN_DATASETS,
                     unseen_datasets=EXPANDED_UNSEEN_DATASETS, family_weights=EXPANDED_FAMILY_WEIGHTS,
                     progress=True):
    if mode not in ("ordinary", "adaptation") or (init_from is None) == (resume is None):
        raise ValueError("choose ordinary/adaptation and exactly one of init_from/resume")
    if {d.accession for d in datasets} & {d.accession for d in unseen_datasets}:
        raise ValueError("training and unseen files must not overlap")
    output_dir, cache_dir, device = Path(output_dir).resolve(), Path(cache_dir).resolve(), torch.device(device)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("choose a new empty A/B run directory")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    source = Path(init_from if init_from is not None else resume).resolve()
    loaded = load_training_checkpoint(source)
    shared = loaded.model.to(device).eval().requires_grad_(True)
    if shared.config.num_layers < 4 or type(shared.output_head) is not torch.nn.Linear:
        raise ValueError("A/B training requires the shared >=4-layer linear-head architecture")
    optimizer = torch.optim.AdamW(shared.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    state = loaded.payload.get("training_state") if resume is not None else None
    if resume is not None and (not isinstance(state, dict) or state.get("format") != FORMAT or state.get("version") != 1):
        raise ValueError("--resume requires this A/B trainer's last.pt; use --init-from for old pretraining weights")
    initialization = (state["contract"]["initialization"] if state else
        dict(checkpoint=str(source), sha256=sha256_file(source), method="weights_only_reset_optimizer",
             source_global_step=loaded.payload.get("global_step")))
    with BalancedTrainingSampler(datasets, cache_dir, train_fraction=config.train_fraction,
            batch_reads=config.batch_reads, seed=config.seed, steps_per_epoch=config.steps_per_epoch,
            family_weights=family_weights) as sampler:
        episodes, val_indices = validation_episodes(sampler, config)
        split = sampler.data_split_manifest(unseen_datasets)
        contract = dict(mode=mode, config=asdict(config), model_config=shared.config.to_dict(),
                        initialization=initialization, sampler_config=sampling_description(sampler, config),
                        data_split=split, device_type=device.type,
                        validation_indices_sha256=hashlib.sha256(json.dumps(val_indices, sort_keys=True).encode()).hexdigest(),
                        training_dropout_enabled=False, selection_metric=SELECTION_METRIC)
        step, start_epoch, best = 0, 1, float("inf")
        counters = dict(data_quality_tokens=0, support_quality_tokens=0, query_quality_tokens=0,
                        inner_backward_symbols=0, outer_backward_symbols=0, inner_steps=0,
                        training_seconds=0., validation_seconds=0.,
                        initial_validation_seconds=0.,
                        data_footprint_sha256=hashlib.sha256(b"").hexdigest(),
                        tokens_by_dataset={d.accession: 0 for d in datasets})
        output_dir.mkdir(parents=True, exist_ok=True)
        if state:
            if state["contract"] != contract or loaded.payload.get("selection_metric") != SELECTION_METRIC:
                raise ValueError("resume contract differs (mode, budget, data, seed or training recipe)")
            if loaded.payload["global_step"] != loaded.payload["epoch"] * config.steps_per_epoch:
                raise ValueError("resume requires a completed epoch checkpoint")
            sampler.load_state_dict(loaded.payload["sampler_state"])
            if (sampler._epoch != loaded.payload["epoch"]
                    or sampler._total_samples != loaded.payload["global_step"]
                    or sampler._schedule_cursor != config.steps_per_epoch):
                raise ValueError("resume sampler position differs from checkpoint")
            optimizer.load_state_dict(loaded.payload["optimizer_state_dict"])
            step, start_epoch = loaded.payload["global_step"], loaded.payload["epoch"] + 1
            if start_epoch > config.epochs:
                raise ValueError("this fixed-budget run is already complete")
            best = loaded.payload["best_validation_bits_per_quality"]
            best_path = source.parent / "best.pt"
            best_payload = load_training_checkpoint(best_path).payload
            if (best_payload.get("training_state", {}).get("contract") != contract
                    or best_payload["validation_metrics"][SELECTION_METRIC] != best):
                raise ValueError("resume needs the matching best.pt beside last.pt")
            shutil.copyfile(best_path, output_dir / "best.pt")
            counters = copy.deepcopy(state["counters"])
            initial = state["initial_validation"]
            random.setstate(state["python_rng"])
            np.random.set_state(state["numpy_rng"])
            torch.set_rng_state(state["torch_rng"].cpu())
            if device.type == "cuda":
                torch.cuda.set_rng_state_all([v.cpu() for v in state["cuda_rng"]])
        else:
            if progress:
                print("Evaluate common initial short-adaptation proxy (training-source validation only)", file=sys.stderr, flush=True)
            synchronize(device)
            started = time.perf_counter()
            initial = evaluate_proxy(shared, episodes, config, device, progress=progress)
            synchronize(device)
            counters["initial_validation_seconds"] = time.perf_counter() - started
        _write_json_atomic(output_dir / "run_config.json", contract)
        _write_json_atomic(output_dir / "training_corpus.json", sampler.corpus_statistics_dict())
        _write_json_atomic(output_dir / "validation_indices.json", val_indices)
        _write_json_atomic(output_dir / "initial_validation.json", initial)
        if progress:
            print(f"A/B mode={mode} output={output_dir} additional_outer_updates={config.epochs * config.steps_per_epoch} "
                  f"max_inner_steps_per_episode={math.ceil((config.batch_reads // 2) / config.inner_batch_reads)}", file=sys.stderr, flush=True)
        for epoch in range(start_epoch, config.epochs + 1):
            synchronize(device)
            started = time.perf_counter()
            objective_nats, objective_symbols = 0., 0
            steps = range(config.steps_per_epoch)
            bar = tqdm(steps, desc=f"{mode} {epoch}/{config.epochs}", unit="episode", file=sys.stderr) if progress and tqdm else steps
            for _ in bar:
                sample = sampler.sample_batch()
                step += 1
                batch = spread_episode(sample, sampler, config, step)
                lr = learning_rate_at(step, config)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                stats = outer_update(shared, optimizer, batch, mode=mode, config=config, device=device, step=step)
                tokens = stats["support_symbols"] + stats["query_symbols"]
                counters["data_quality_tokens"] += tokens
                counters["support_quality_tokens"] += stats["support_symbols"]
                counters["query_quality_tokens"] += stats["query_symbols"]
                for key in ("inner_backward_symbols", "outer_backward_symbols", "inner_steps"):
                    counters[key] += stats[key]
                counters["tokens_by_dataset"][sample.dataset.accession] += tokens
                counters["data_footprint_sha256"] = hashlib.sha256(
                    bytes.fromhex(counters["data_footprint_sha256"]) + bytes.fromhex(stats["episode_sha256"])).hexdigest()
                objective_nats += stats["objective_nats"] * stats["outer_backward_symbols"]
                objective_symbols += stats["outer_backward_symbols"]
                if progress and tqdm and step % 20 == 0:
                    bar.set_postfix(outer_bpq=f"{objective_nats / objective_symbols / math.log(2.):.4f}", lr=f"{lr:.2g}")
            synchronize(device)
            counters["training_seconds"] += time.perf_counter() - started
            started = time.perf_counter()
            validation = evaluate_proxy(shared, episodes, config, device, progress=progress)
            synchronize(device)
            counters["validation_seconds"] += time.perf_counter() - started
            metric = validation[SELECTION_METRIC]
            if not math.isfinite(metric):
                raise ValueError("nonfinite adaptation validation proxy")
            improved = metric < best
            best = min(best, metric)
            row = dict(epoch=epoch, global_step=step, learning_rate=lr,
                       objective_bpq=objective_nats / objective_symbols / math.log(2.),
                       objective="all_batch_ce" if mode == "ordinary" else "adapted_query_ce_first_order",
                       validation=validation, counters=copy.deepcopy(counters))
            with (output_dir / "training_log.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            arguments = dict(model=shared, optimizer=optimizer, epoch=epoch, global_step=step,
                data_split=split, sampler_config=contract["sampler_config"], sampler_statistics=sampling_statistics(sampler),
                best_validation_bits_per_quality=best, validation_metrics=validation, selection_metric=SELECTION_METRIC,
                sampler_state=sampler.state_dict(), training_state=dict(format=FORMAT, version=1,
                    contract=contract, counters=copy.deepcopy(counters), initial_validation=initial,
                    python_rng=random.getstate(), numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else []))
            save_training_checkpoint(output_dir / "last.pt", **arguments)
            if improved:
                save_training_checkpoint(output_dir / "best.pt", **arguments)
            _write_json_atomic(output_dir / "training_summary.json", dict(
                mode=mode, completed_epoch=epoch, global_step=step, counters=counters,
                best_short_adaptation_macro_bpq=best, initialization=initialization,
                primary_budget_matched_checkpoint="last.pt", proxy_selected_checkpoint="best.pt",
                warning="Short float adaptation proxy is not 10k file adaptation or actual compressed bytes; A/B match data exposure, not wall time."))
            if progress:
                print(f"epoch={epoch} shared_step={step} short_adapted_macro={metric:.6f} "
                      f"cold_macro={validation['unadapted_macro_bits_per_quality']:.6f} "
                      f"Q_seen={counters['data_quality_tokens']} train_s={counters['training_seconds']:.1f}",
                      file=sys.stderr, flush=True)
        _write_json_atomic(output_dir / "sampling_statistics.json", sampling_statistics(sampler))
    return output_dir / "last.pt"


def main(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("ordinary", "adaptation"))
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--init-from", type=Path)
    initialization.add_argument("--resume", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/2nd/training_cache")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    for name, field in ABConfig.__dataclass_fields__.items():
        parser.add_argument("--" + name.replace("_", "-"), type=type(field.default), default=field.default)
    args = parser.parse_args(argv)
    try:
        config = ABConfig(**{name: getattr(args, name) for name in ABConfig.__dataclass_fields__})
        initial = args.init_from if args.init_from is not None else (None if args.resume else DEFAULT_INITIAL)
        if args.dry_run:
            print(json.dumps(dict(mode=args.mode, config=asdict(config), init_from=str(initial) if initial else None,
                resume=str(args.resume) if args.resume else None, training_accessions=[d.accession for d in EXPANDED_TRAIN_DATASETS],
                family_weights=EXPANDED_FAMILY_WEIGHTS, output_root=str(args.output_root.resolve()),
                primary_comparison="last.pt after equal additional shared updates and matched data exposure; not equal time"), indent=2))
            return 0
        device = choose_device(args.device)
        args.output_root.mkdir(parents=True, exist_ok=True)
        output_dir = Path(tempfile.mkdtemp(prefix=args.mode + "_", dir=args.output_root))
        last = run_continuation(args.mode, output_dir=output_dir, cache_dir=args.cache_dir, device=device,
                                config=config, init_from=initial, resume=args.resume)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        raise SystemExit(str(exc)) from exc
    print(f"budget-matched checkpoint: {last}")
    print(f"short-proxy best checkpoint: {last.parent / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
