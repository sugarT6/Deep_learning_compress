"""Four-layer Transformer continuation: original vs quality-profile sampling."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import sys
import tempfile
import time

import numpy as np
import torch
from torch.nn import functional as F

from .checkpoint import load_training_checkpoint, save_training_checkpoint
from .container import sha256_file
from .datasets import EXPANDED_FAMILY_WEIGHTS, EXPANDED_TRAIN_DATASETS, EXPANDED_UNSEEN_DATASETS
from .evaluate import choose_device
from .model import fastq_batch_to_tensors
from .quality_sampling import QualityMixtureSampler, load_profile, prepare_profile
from .train import SELECTION_METRIC, _write_json_atomic, evaluate_validation

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INITIAL = ROOT / "codec/runs/direct_quality_4layer_expanded20_200k_S1pPAq/best.pt"
DEFAULT_PROFILE = ROOT / "codec/output/quality_sampling_ab_20261028/quality_modes.json"
DEFAULT_OUTPUT_ROOT = ROOT / "codec/runs/quality_sampling_ab_20261028"
FORMAT = "direct-quality-sampling-ab-training-v1"


@dataclass(frozen=True)
class SamplingConfig:
    updates: int = 10000
    quality_tokens_per_update: int = 32768
    eval_every: int = 2000
    batch_reads: int = 256
    schedule_batches: int = 2000
    validation_reads_per_file: int = 1024
    train_fraction: float = .9
    learning_rate: float = 1e-5
    min_learning_rate: float = 1e-6
    weight_decay: float = 1e-5
    grad_clip: float = 1.
    seed: int = 20261028

    def __post_init__(self):
        for name in ("updates", "quality_tokens_per_update", "eval_every", "batch_reads",
                     "schedule_batches", "validation_reads_per_file"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(name + " must be a positive integer")
        if self.batch_reads > 256 or self.updates % self.eval_every:
            raise ValueError("batch_reads must be <=256 and updates divisible by eval_every")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        for name in ("train_fraction", "learning_rate", "min_learning_rate", "weight_decay", "grad_clip"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError(name + " must be finite")
        if (not 0 < self.train_fraction < 1 or not 0 < self.min_learning_rate <= self.learning_rate
                or self.weight_decay < 0 or self.grad_clip <= 0):
            raise ValueError("invalid split or optimizer settings")


class TokenStream:
    """Carry partial source batches across updates; score each target once.

    Loss targets are consumed in read-major order. Full preceding Q context
    stays intact; loss masking never changes the model's active/padding mask.
    """

    def __init__(self, sampler):
        self.sampler, self.pending, self.offset = sampler, None, 0
        self.tokens = 0
        self.tokens_by_dataset = {d.accession: 0 for d in sampler.datasets}
        self.forward_symbols = 0
        self.footprint = hashlib.sha256(b"").hexdigest()

    def take(self, wanted):
        if wanted <= 0:
            raise ValueError("token quota must be positive")
        while self.pending is None:
            sample = self.sampler.sample_batch()
            if sample.batch.active_mask.any():
                self.pending, self.offset = sample, 0
        sample, offset = self.pending, self.offset
        batch = sample.batch
        flat = np.flatnonzero(batch.active_mask.reshape(-1))
        count = min(wanted, len(flat) - offset)
        selected = flat[offset:offset + count]
        width = batch.max_read_length
        first, last = int(selected[0] // width), int(selected[-1] // width) + 1
        part = replace(batch, bases=batch.bases[first:last], qualities=batch.qualities[first:last],
                       lengths=batch.lengths[first:last], active_mask=batch.active_mask[first:last],
                       read_count=last - first, read_indices=batch.read_indices[first:last])
        selected = selected - first * width
        name = sample.dataset.accession
        descriptor = np.asarray([batch.read_indices[0], batch.read_indices[-1] + 1,
                                 offset, count], dtype="<i8").tobytes()
        self.footprint = hashlib.sha256(bytes.fromhex(self.footprint) + name.encode() + b"\0" + descriptor
            + part.qualities.reshape(-1)[selected].tobytes()
            + part.bases.reshape(-1)[selected].tobytes()).hexdigest()
        self.tokens += count
        self.tokens_by_dataset[name] += count
        self.forward_symbols += int(part.active_mask.sum())
        self.offset += count
        if self.offset == len(flat):
            self.pending, self.offset = None, 0
        return part, selected, count

    def state_dict(self):
        pending = None
        if self.pending is not None:
            batch = self.pending.batch
            pending = dict(accession=self.pending.dataset.accession,
                           start=int(batch.read_indices[0]), stop=int(batch.read_indices[-1]) + 1,
                           offset=self.offset)
        return dict(pending=pending, tokens=self.tokens, tokens_by_dataset=self.tokens_by_dataset.copy(),
                    forward_symbols=self.forward_symbols, loss_target_footprint_sha256=self.footprint)

    def load_state_dict(self, state):
        if set(state) != set(self.state_dict()) or set(state["tokens_by_dataset"]) != set(self.tokens_by_dataset):
            raise ValueError("token stream state mismatch")
        if (type(state["tokens"]) is not int or state["tokens"] < 0
                or any(type(v) is not int or v < 0 for v in state["tokens_by_dataset"].values())
                or sum(state["tokens_by_dataset"].values()) != state["tokens"]
                or state["forward_symbols"] < state["tokens"]):
            raise ValueError("invalid token stream counts")
        if len(bytes.fromhex(state["loss_target_footprint_sha256"])) != 32:
            raise ValueError("invalid target footprint")
        self.pending, self.offset = None, 0
        pending = state["pending"]
        if pending is not None:
            from .train import TrainingSample
            by_name = {d.accession: d for d in self.sampler.datasets}
            name = pending["accession"]
            if not 0 <= pending["start"] < pending["stop"] <= self.sampler.train_stop_for(name):
                raise ValueError("pending token batch crosses training boundary")
            batch = self.sampler.reader_for(name).read_range(pending["start"], pending["stop"])
            if not 0 <= pending["offset"] < int(batch.active_mask.sum()):
                raise ValueError("invalid pending target offset")
            self.pending, self.offset = TrainingSample(by_name[name], batch), pending["offset"]
        self.tokens = state["tokens"]
        self.tokens_by_dataset = state["tokens_by_dataset"].copy()
        self.forward_symbols = state["forward_symbols"]
        self.footprint = state["loss_target_footprint_sha256"]


def update_on_tokens(model, optimizer, stream, config, device):
    model.train()  # retain checkpoint dropout configuration in both arms
    optimizer.zero_grad(set_to_none=True)
    remaining, nats = config.quality_tokens_per_update, 0.
    while remaining:
        batch, indices, count = stream.take(remaining)
        tensors = fastq_batch_to_tensors(batch, device)
        selected = torch.as_tensor(indices, dtype=torch.long, device=device)
        logits = model.forward_full(**tensors).reshape(-1, 42).index_select(0, selected)
        target = tensors["qualities"].reshape(-1).index_select(0, selected)
        loss_sum = F.cross_entropy(logits, target, reduction="sum")
        if not torch.isfinite(loss_sum):
            raise ValueError("nonfinite quality cross entropy")
        (loss_sum / config.quality_tokens_per_update).backward()
        nats += float(loss_sum.detach().item())
        remaining -= count
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip, error_if_nonfinite=True)
    optimizer.step()
    return nats


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_sampling(mode, *, output_dir, cache_dir, profile_path, config=SamplingConfig(), device="cpu",
                 init_from=None, resume=None, datasets=EXPANDED_TRAIN_DATASETS,
                 unseen_datasets=EXPANDED_UNSEEN_DATASETS, family_weights=EXPANDED_FAMILY_WEIGHTS,
                 progress=True):
    if mode not in ("original", "quality") or (init_from is None) == (resume is None):
        raise ValueError("choose original/quality and exactly one of init_from/resume")
    if {d.accession for d in datasets} & {d.accession for d in unseen_datasets}:
        raise ValueError("training and unseen manifests overlap")
    output_dir, cache_dir, device = Path(output_dir).resolve(), Path(cache_dir).resolve(), torch.device(device)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("choose a new empty sampling run directory")
    source = Path(init_from if init_from is not None else resume).resolve()
    loaded = load_training_checkpoint(source)
    model = loaded.model.to(device)
    if model.config.num_layers != 4 or type(model.output_head) is not torch.nn.Linear:
        raise ValueError("sampling comparison requires the original four-layer linear-head Transformer")
    state = loaded.payload.get("training_state") if resume is not None else None
    if resume is not None and (not isinstance(state, dict) or state.get("format") != FORMAT):
        raise ValueError("resume requires this sampling trainer's last.pt")
    initialization = (state["contract"]["initialization"] if state else
        dict(sha256=sha256_file(source), source_global_step=loaded.payload.get("global_step"),
             method="weights_only_reset_optimizer"))
    raw_profile = json.loads(Path(profile_path).read_text())
    kwargs = dict(train_fraction=config.train_fraction, batch_reads=config.batch_reads,
                  seed=config.seed, steps_per_epoch=config.schedule_batches)
    # Identical quota allocator in both arms: only these probabilities differ.
    kwargs.update(probabilities=raw_profile[mode + "_probabilities"], profile_sha256=raw_profile["sha256"])
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    with QualityMixtureSampler(datasets, cache_dir, **kwargs) as sampler:
        profile = load_profile(profile_path, sampler)
        if profile["family_weights"] != family_weights:
            raise ValueError("original family weights differ from prepared profile")
        for dataset in datasets:
            if not sampler.reader_for(dataset.accession).count_symbols(0, sampler.train_stop_for(dataset.accession)):
                raise ValueError("training source has no Q: " + dataset.accession)
        split = sampler.data_split_manifest(unseen_datasets)
        contract = dict(format=FORMAT, version=1, mode=mode, config=asdict(config),
                        model_config=model.config.to_dict(), initialization=initialization,
                        profile_sha256=profile["sha256"], sampler_config=sampler.config_dict(),
                        data_split=split, device_type=device.type, training_dropout_enabled=True,
                        selection_metric=SELECTION_METRIC)
        stream = TokenStream(sampler)
        step, best = 0, float("inf")
        timers = dict(training_seconds=0., validation_seconds=0., initial_validation_seconds=0.)
        if state:
            if state["contract"] != contract or loaded.payload.get("selection_metric") != SELECTION_METRIC:
                raise ValueError("resume contract differs (mode, profile, budget, data or optimizer)")
            step, best = loaded.payload["global_step"], loaded.payload["best_validation_bits_per_quality"]
            if step % config.eval_every or step >= config.updates:
                raise ValueError("resume requires an unfinished evaluation-boundary checkpoint")
            sampler.load_state_dict(loaded.payload["sampler_state"])
            stream.load_state_dict(state["stream"])
            if stream.tokens != step * config.quality_tokens_per_update:
                raise ValueError("resume token budget does not match update count")
            optimizer.load_state_dict(loaded.payload["optimizer_state_dict"])
            best_source = source.parent / "best.pt"
            old_best = load_training_checkpoint(best_source).payload
            if (old_best.get("training_state", {}).get("contract") != contract
                    or old_best["validation_metrics"][SELECTION_METRIC] != best):
                raise ValueError("matching best.pt must be beside resume last.pt")
            initial, timers = state["initial_validation"], state["timers"].copy()
            random.setstate(state["python_rng"])
            np.random.set_state(state["numpy_rng"])
            torch.set_rng_state(state["torch_rng"].cpu())
            if device.type == "cuda":
                torch.cuda.set_rng_state_all([v.cpu() for v in state["cuda_rng"]])
        output_dir.mkdir(parents=True, exist_ok=True)
        if state:
            shutil.copyfile(best_source, output_dir / "best.pt")
        else:
            if progress:
                print("Evaluate common initial checkpoint on training-source validation suffixes", file=sys.stderr, flush=True)
            synchronize(device)
            started = time.perf_counter()
            _, initial = evaluate_validation(model, sampler, max_reads_per_file=config.validation_reads_per_file, device=device)
            synchronize(device)
            timers["initial_validation_seconds"] = time.perf_counter() - started
        _write_json_atomic(output_dir / "run_config.json", contract)
        _write_json_atomic(output_dir / "quality_modes.json", profile)
        _write_json_atomic(output_dir / "initial_validation.json", initial)
        _write_json_atomic(output_dir / "training_corpus.json", sampler.corpus_statistics_dict())
        if progress:
            print(f"Sampling mode={mode} output={output_dir} updates={config.updates} "
                  f"exact_loss_Q={config.updates * config.quality_tokens_per_update}", file=sys.stderr, flush=True)
        while step < config.updates:
            synchronize(device)
            started, objective = time.perf_counter(), 0.
            steps = range(step, step + config.eval_every)
            bar = tqdm(steps, desc=f"{mode} {step // config.eval_every + 1}", unit="update", file=sys.stderr) if progress and tqdm else steps
            for _ in bar:
                fraction = step / max(config.updates - 1, 1)
                lr = config.min_learning_rate + .5 * (config.learning_rate - config.min_learning_rate) * (1 + math.cos(math.pi * fraction))
                for group in optimizer.param_groups:
                    group["lr"] = lr
                objective += update_on_tokens(model, optimizer, stream, config, device)
                step += 1
                if progress and tqdm and step % 20 == 0:
                    bar.set_postfix(Q=stream.tokens, lr=f"{lr:.2g}")
            synchronize(device)
            timers["training_seconds"] += time.perf_counter() - started
            started = time.perf_counter()
            _, validation = evaluate_validation(model, sampler, max_reads_per_file=config.validation_reads_per_file, device=device)
            synchronize(device)
            timers["validation_seconds"] += time.perf_counter() - started
            metric = validation[SELECTION_METRIC]
            if not math.isfinite(metric):
                raise ValueError("nonfinite validation macro BPQ")
            improved = metric < best
            best = min(best, metric)
            row = dict(global_step=step, learning_rate=lr,
                       training_bpq=objective / (config.eval_every * config.quality_tokens_per_update) / math.log(2),
                       validation=validation, stream=stream.state_dict(), timers=timers.copy())
            with (output_dir / "training_log.jsonl").open("a") as handle:
                handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            training_state = dict(format=FORMAT, contract=contract, stream=stream.state_dict(), timers=timers.copy(),
                initial_validation=initial, python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else [])
            args = dict(model=model, optimizer=optimizer, epoch=step // config.eval_every, global_step=step,
                data_split=split, sampler_config=contract["sampler_config"], sampler_statistics=sampler.statistics_dict(),
                best_validation_bits_per_quality=best, validation_metrics=validation, selection_metric=SELECTION_METRIC,
                sampler_state=sampler.state_dict(), training_state=training_state)
            if improved:
                save_training_checkpoint(output_dir / "best.pt", **args)
            save_training_checkpoint(output_dir / "last.pt", **args)
            summary = dict(mode=mode, contract=contract, global_step=step, stream=stream.state_dict(), timers=timers.copy(),
                           initial_validation=initial, last_validation=validation, best_validation_macro_bpq=best,
                           primary_checkpoint="last.pt", secondary_checkpoint="best.pt",
                           scope="Exact loss-Q/update budget; not equal wall time or total forward context. Validation is unadapted source float BPQ, not target file compression.")
            _write_json_atomic(output_dir / "training_summary.json", summary)
            _write_json_atomic(output_dir / "sampling_statistics.json", sampler.statistics_dict())
            if progress:
                print(f"step={step} Q={stream.tokens} val_macro={metric:.6f} "
                      f"train_s={timers['training_seconds']:.1f}", file=sys.stderr, flush=True)
    return output_dir / "last.pt"


def compare_runs(original, quality):
    a, b = [json.loads((Path(path) / "training_summary.json").read_text()) for path in (original, quality)]
    if a["mode"] != "original" or b["mode"] != "quality":
        raise ValueError("compare requires original run then quality run")
    ca, cb = [dict(row["contract"]) for row in (a, b)]
    for value in (ca, cb):
        value.pop("mode"); value.pop("sampler_config")
    if ca != cb:
        raise ValueError("run contracts do not match")
    config = ca["config"]
    for row in (a, b):
        if (row["global_step"] != config["updates"] or
                row["stream"]["tokens"] != config["updates"] * config["quality_tokens_per_update"]):
            raise ValueError("both runs must finish the complete matched Q/update budget")
    per_dataset = {name: dict(original_bpq=metrics["bits_per_quality"],
        quality_bpq=b["last_validation"]["per_dataset"][name]["bits_per_quality"],
        quality_gain_bpq=metrics["bits_per_quality"] - b["last_validation"]["per_dataset"][name]["bits_per_quality"])
        for name, metrics in a["last_validation"]["per_dataset"].items()}
    return dict(original_macro_bpq=a["last_validation"][SELECTION_METRIC],
                quality_macro_bpq=b["last_validation"][SELECTION_METRIC],
                quality_gain_bpq=a["last_validation"][SELECTION_METRIC] - b["last_validation"][SELECTION_METRIC],
                per_dataset=per_dataset,
                note="Source float validation only; final comparison still needs identical file adaptation on predeclared development files.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="profile training prefixes, without training")
    prepare.add_argument("--cache-dir", type=Path, default=ROOT / "data/2nd/training_cache")
    prepare.add_argument("--profile-out", type=Path, default=DEFAULT_PROFILE)
    prepare.add_argument("--profile-reads", type=int, default=4096)
    prepare.add_argument("--strata", type=int, default=16)
    prepare.add_argument("--clusters", type=int, default=4)
    prepare.add_argument("--mode-fraction", type=float, default=.5)
    prepare.add_argument("--train-fraction", type=float, default=.9)
    for mode in ("original", "quality"):
        train = sub.add_parser(mode)
        initialization = train.add_mutually_exclusive_group()
        initialization.add_argument("--init-from", type=Path)
        initialization.add_argument("--resume", type=Path)
        train.add_argument("--cache-dir", type=Path, default=ROOT / "data/2nd/training_cache")
        train.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
        train.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
        train.add_argument("--device", default="cuda")
        train.add_argument("--dry-run", action="store_true")
        for name, field in SamplingConfig.__dataclass_fields__.items():
            train.add_argument("--" + name.replace("_", "-"), type=type(field.default), default=field.default)
    compare = sub.add_parser("compare")
    compare.add_argument("original", type=Path)
    compare.add_argument("quality", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare_profile(args.profile_out, datasets=EXPANDED_TRAIN_DATASETS, cache_dir=args.cache_dir,
                family_weights=EXPANDED_FAMILY_WEIGHTS, profile_reads=args.profile_reads, strata=args.strata,
                clusters=args.clusters, mode_fraction=args.mode_fraction, train_fraction=args.train_fraction, progress=True)
            print(json.dumps(dict(profile=str(args.profile_out.resolve()), groups=result["groups"],
                                  quality_probabilities=result["quality_probabilities"]), indent=2))
            return 0
        if args.command == "compare":
            print(json.dumps(compare_runs(args.original, args.quality), indent=2))
            return 0
        config = SamplingConfig(**{name: getattr(args, name) for name in SamplingConfig.__dataclass_fields__})
        initial = args.init_from if args.init_from is not None else (None if args.resume else DEFAULT_INITIAL)
        if args.dry_run:
            print(json.dumps(dict(mode=args.command, config=asdict(config), init_from=str(initial) if initial else None,
                resume=str(args.resume) if args.resume else None, profile=str(args.profile),
                quality_token_budget=config.updates * config.quality_tokens_per_update,
                training_accessions=[d.accession for d in EXPANDED_TRAIN_DATASETS],
                output_root=str(args.output_root), primary_comparison="last.pt with exactly equal loss-Q and updates"), indent=2))
            return 0
        device = choose_device(args.device)
        args.output_root.mkdir(parents=True, exist_ok=True)
        output = Path(tempfile.mkdtemp(prefix=args.command + "_", dir=args.output_root))
        path = run_sampling(args.command, output_dir=output, cache_dir=args.cache_dir, profile_path=args.profile,
                            config=config, device=device, init_from=initial, resume=args.resume)
        print("budget-matched checkpoint:", path)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        raise SystemExit(str(exc)) from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
