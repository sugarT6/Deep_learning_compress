"""Large Transformer DDP training that preserves the audited sampler's *global* batch schedule.

Only training forward passes use autocast. Master parameters, AdamW state,
validation and the ordinary codec-compatible checkpoints remain float32.
The legacy trainer and model remain unchanged. FlashAttention is a training
backend; checkpoint tensors retain the ordinary codec state-dict contract.
"""

from contextlib import contextmanager, nullcontext
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys

import numpy as np
import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel

from .checkpoint import (CHECKPOINT_SCHEMA_VERSION, cache_schema,
                         save_training_checkpoint as save_checkpoint)
from .datasets import (EXPANDED_FAMILY_WEIGHTS, EXPANDED_TRAIN_DATASETS,
                       EXPANDED_UNSEEN_DATASETS)
from .evaluate import DatasetMetrics, evaluate_reader_range, summarize_metric_group
from .model import DirectQualityModelConfig, fastq_batch_to_tensors, feature_schema
from .training_cache import DEFAULT_CACHE_DIR
from .training_control import TrainingControl, TrainingControlConfig
from .large_transformer import LargeQualityTransformer
from .train import BalancedTrainingSampler, _write_json_atomic as _legacy_write_json

CHECKPOINT_FORMAT = "large_transformer_direct_quality_training"
TRAINING_KIND = "large_transformer_ddp"
SELECTION_METRIC = "dataset_macro_bits_per_quality"
DEFAULT_CONTROL = TrainingControlConfig(lr_scheduler="plateau")


def write_json(path, payload):
    _legacy_write_json(Path(path), payload)


def resume_contract(run_config):
    return {key: value for key, value in run_config.items()
            if key not in ("epochs", "resume_from")}


def _sampler(cache_dir, train_datasets, train_fraction, batch_reads, seed,
             steps_per_epoch, family_weights):
    return BalancedTrainingSampler(
        train_datasets, Path(cache_dir), train_fraction=train_fraction,
        batch_reads=batch_reads, seed=seed, steps_per_epoch=steps_per_epoch,
        family_weights=family_weights, feature_block_reads=0)


def inspect_sampler(sampler, model_config, unseen_datasets, validation_max_reads_per_file):
    """Check explicit caches and split boundaries without truncating long reads.

    context_length is an attention window, not a maximum read length. A read
    longer than that window must still be trained in its entirety.
    """
    if validation_max_reads_per_file < 0:
        raise ValueError("validation_max_reads_per_file must be nonnegative")
    entries = {}
    for dataset in sampler.datasets:
        reader = sampler.reader_for(dataset.accession)
        metadata = reader.metadata
        train_stop = sampler.train_stop_for(dataset.accession)
        for start, stop in ((0, min(sampler.batch_reads, train_stop)),
                            (train_stop, min(train_stop + sampler.batch_reads, metadata.read_count))):
            batch = reader.read_range(start, stop)
            if not batch.active_mask.any() or np.any(batch.qualities[batch.active_mask] > 41):
                raise ValueError(f"{dataset.accession}: invalid quality symbols at split boundary")
        available = metadata.read_count - train_stop
        entries[dataset.accession] = {
            "cache_path": str(reader.cache_path.resolve()),
            "maximum_read_length": metadata.maximum_read_length,
            "minimum_quality_id": metadata.minimum_quality_id,
            "maximum_quality_id": metadata.maximum_quality_id,
            "validation_reads_evaluated": min(available, validation_max_reads_per_file)
                if validation_max_reads_per_file else available,
        }
    return {
        "format": CHECKPOINT_FORMAT, "model_config": model_config.to_dict(),
        "feature_schema": feature_schema(model_config), "cache_schema": cache_schema(),
        "data_split": sampler.data_split_manifest(unseen_datasets),
        "sampler_config": sampler.config_dict(), "corpus": sampler.corpus_statistics_dict(),
        "datasets": entries, "block_features": False,
        "preflight": "metadata and first training/validation cache batches; not a full cache rescan",
        "read_length_rule": "never_truncate; context_length_is_the_causal_attention_window",
    }


def preflight(*, model_config, cache_dir=DEFAULT_CACHE_DIR, train_fraction=.9,
              batch_reads=256, seed=20260909, steps_per_epoch=2000,
              validation_max_reads_per_file=5000,
              train_datasets=EXPANDED_TRAIN_DATASETS,
              unseen_datasets=EXPANDED_UNSEEN_DATASETS,
              family_weights=EXPANDED_FAMILY_WEIGHTS):
    with _sampler(cache_dir, train_datasets, train_fraction, batch_reads, seed,
                  steps_per_epoch, family_weights) as sampler:
        return inspect_sampler(sampler, model_config, unseen_datasets,
                               validation_max_reads_per_file)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


DISTRIBUTED_STATE_VERSION = 1


@contextmanager
def distributed_context(device, *, backend=None):
    """Use a caller's process group or create one from torchrun's environment."""
    device = torch.device(device)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    expected_backend = backend or ("nccl" if device.type == "cuda" else "gloo")
    owned = not dist.is_initialized()
    if owned:
        if int(os.environ.get("WORLD_SIZE", 1)) < 2:
            raise ValueError("distributed training requires torchrun with at least two ranks")
        dist.init_process_group(expected_backend, timeout=timedelta(minutes=60))
    try:
        if dist.get_world_size() < 2:
            raise ValueError("distributed training requires at least two ranks")
        if str(dist.get_backend()) != expected_backend:
            raise ValueError("existing distributed backend differs from requested backend")
        yield device, dist.get_rank(), dist.get_world_size()
    finally:
        if owned and dist.is_initialized():
            dist.destroy_process_group()


def _primary_action(action, rank):
    """Propagate rank-zero I/O errors before other ranks enter the next phase."""
    status = [None]
    if rank == 0:
        try:
            action()
        except Exception as exc:
            status[0] = f"{type(exc).__name__}: {exc}"
    dist.broadcast_object_list(status, src=0)
    if status[0] is not None:
        raise RuntimeError(status[0])


def _rng_state(device):
    return {
        "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def _restore_rng(state, device):
    if set(state) != {"python_rng", "numpy_rng", "torch_rng", "cuda_rng"}:
        raise ValueError("invalid distributed rank RNG state")
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"].cpu())
    if device.type == "cuda":
        if state["cuda_rng"] is None:
            raise ValueError("resume checkpoint lacks this rank's CUDA RNG state")
        torch.cuda.set_rng_state(state["cuda_rng"].cpu(), device)
    elif state["cuda_rng"] is not None:
        raise ValueError("resume CUDA RNG state cannot be restored on CPU")


def _load_payload(path, model_config):
    """Read without allocating a second 0.5B model alongside the live model."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if (not isinstance(payload, dict)
            or payload.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION
            or payload.get("training_state", {}).get("kind") != TRAINING_KIND):
        raise ValueError("resume requires a large Transformer DDP training checkpoint")
    if (payload.get("model_config") != model_config.to_dict()
            or payload.get("feature_schema") != feature_schema(model_config)
            or payload.get("cache_schema") != cache_schema()):
        raise ValueError("resume model, feature or cache schema differs")
    return payload


def rank_read_range(read_count, rank, world_size):
    """Contiguous disjoint shards include every global-batch read exactly once."""
    if read_count < 0 or world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("invalid read shard arguments")
    return read_count * rank // world_size, read_count * (rank + 1) // world_size


def distributed_train_step(model, optimizer, tensors, *, microbatch_reads,
                           grad_clip, precision):
    """One global sampler update, including uneven and empty rank shards.

    Every rank receives the same global batch from its identical sampler. DDP
    averages gradients; multiplying local summed CE by world_size/global_Q
    therefore yields exactly the global per-Q mean, irrespective of read lengths.
    Ranks execute the same number of forwards/backwards. Empty slots use a
    zero-weight real read so the final DDP collective always has a full graph.
    """
    rank, world_size = dist.get_rank(), dist.get_world_size()
    read_count = tensors["bases"].shape[0]
    symbols = int(tensors["active_mask"].sum().item())
    if not symbols:
        raise ValueError("training batch contains no quality symbols")
    start, stop = rank_read_range(read_count, rank, world_size)
    largest_shard = (read_count + world_size - 1) // world_size
    micro = microbatch_reads or largest_shard
    if micro <= 0:
        raise ValueError("microbatch_reads must be nonnegative")
    slots = (largest_shard + micro - 1) // micro
    dummy_row = int(torch.nonzero(tensors["active_mask"].any(dim=1), as_tuple=False)[0].item())
    optimizer.zero_grad(set_to_none=True)
    local_nats = torch.zeros((), device=tensors["bases"].device, dtype=torch.float64)
    for slot in range(slots):
        begin, end = start + slot * micro, min(start + (slot + 1) * micro, stop)
        real = begin < end
        chunk = {key: value[begin:end] for key, value in tensors.items()} if real else None
        if real:
            real = bool(chunk["active_mask"].any().item())
        if not real:
            chunk = {key: value[dummy_row:dummy_row + 1] for key, value in tensors.items()}
        sync = nullcontext() if slot == slots - 1 else model.no_sync()
        with sync:
            with torch.autocast(device_type=tensors["bases"].device.type,
                                dtype=torch.bfloat16, enabled=precision == "bf16"):
                logits = model(**chunk)
                # CE is explicitly float32 even when the backbone uses bf16.
                nats = F.cross_entropy(logits[chunk["active_mask"]].float(),
                                       chunk["qualities"][chunk["active_mask"]],
                                       reduction="sum")
                loss = nats * (world_size / symbols if real else 0.)
            loss.backward()
        if real:
            local_nats += nats.detach().double()
    dist.all_reduce(local_nats, op=dist.ReduceOp.SUM)
    if not math.isfinite(local_nats.item()):
        raise ValueError("nonfinite distributed training loss")
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip,
                                         error_if_nonfinite=True)
    optimizer.step()
    return float(local_nats.item()), symbols, float(norm.item())


def evaluate_distributed_validation(model, sampler, *, max_reads_per_file,
                                    microbatch_reads, device):
    """Fixed validation prefix per source, FP32, partitioned without overlap."""
    rank, world_size = dist.get_rank(), dist.get_world_size()
    metrics = []
    for dataset in sampler.datasets:
        reader = sampler.reader_for(dataset.accession)
        start = sampler.train_stop_for(dataset.accession)
        count = reader.metadata.read_count - start
        if max_reads_per_file:
            count = min(count, max_reads_per_file)
        first, last = rank_read_range(count, rank, world_size)
        totals = torch.zeros(3, dtype=torch.float64, device=device)
        if first < last:
            metric = evaluate_reader_range(
                model, reader, dataset, category="validation", start=start + first,
                stop=start + last, batch_reads=microbatch_reads,
                max_reads=0, device=device)
            totals[:] = torch.tensor([metric.total_bits, metric.symbols, metric.reads],
                                     dtype=torch.float64, device=device)
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        bits, symbols, reads = totals.tolist()
        if symbols <= 0:
            raise ValueError(f"{dataset.accession}: validation contains no Q symbols")
        metrics.append(DatasetMetrics(
            accession=dataset.accession, platform_family=dataset.platform_family,
            category="validation", unseen_instrument=dataset.unseen_instrument,
            reads=int(reads), symbols=int(symbols), total_bits=bits,
            bits_per_quality=bits / symbols))
    return {**summarize_metric_group(metrics),
            "per_dataset": {metric.accession: metric.to_dict() for metric in metrics}}


def _sampler_fingerprint(sampler):
    return hashlib.sha256(json.dumps(sampler.state_dict(), sort_keys=True).encode()).hexdigest()


def _prime_reducer(ddp, optimizer, sampler, device, precision):
    """Finish DDP's one-time bucket rebuild before the reproducible step stream.

    DDP rebuilds communication buckets after its first backward. A restored
    process otherwise starts with different buckets than an uninterrupted run;
    changed reduction sizes can alter FP32 summation order. Two zero-gradient
    backwards initialize and rebuild those buckets without an optimizer update,
    sampler draw or change to the continuation's RNG stream.
    """
    rng = _rng_state(device)
    dataset = sampler.datasets[0]
    reader = sampler.reader_for(dataset.accession)
    batch = reader.read_range(0, min(sampler.batch_reads,
                                     sampler.train_stop_for(dataset.accession)))
    row = int(np.flatnonzero(batch.active_mask.any(axis=1))[0])
    tensors = fastq_batch_to_tensors(reader.read_range(row, row + 1), device)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=precision == "bf16"):
            logits = ddp(**tensors)
            loss = logits.float().sum() * 0.
        loss.backward()
    optimizer.zero_grad(set_to_none=True)
    _restore_rng(rng, device)


def run_distributed_training(*, model_config, output_dir, cache_dir=DEFAULT_CACHE_DIR,
                             init_mode="random", resume=None,
                             epochs=100, steps_per_epoch=2000, batch_reads=256,
                             train_fraction=.9, validation_max_reads_per_file=5000,
                             learning_rate=2e-4, weight_decay=1e-5, grad_clip=1.,
                             device=torch.device("cuda"), seed=20260909, progress=True,
                             control_config=DEFAULT_CONTROL, microbatch_reads=16,
                             precision="bf16", activation_checkpointing=True,
                             warmup_steps=2000, validation_microbatch_reads=16,
                             attention_backend="flash",
                             attention_query_chunk_size=16,
                             distributed_backend=None,
                             train_datasets=EXPANDED_TRAIN_DATASETS,
                             unseen_datasets=EXPANDED_UNSEEN_DATASETS,
                             family_weights=EXPANDED_FAMILY_WEIGHTS):
    """Train under torchrun; resume only from completed distributed epochs."""
    if epochs <= 0 or steps_per_epoch <= 0:
        raise ValueError("epochs and steps_per_epoch must be positive")
    if init_mode != "random":
        raise ValueError("large Transformer supports random initialization only")
    if type(microbatch_reads) is not int or not 0 <= microbatch_reads <= batch_reads:
        raise ValueError("microbatch_reads must be between 0 and batch_reads")
    if type(validation_microbatch_reads) is not int or not 1 <= validation_microbatch_reads <= 256:
        raise ValueError("validation_microbatch_reads must be between 1 and 256")
    if type(warmup_steps) is not int or warmup_steps < 0:
        raise ValueError("warmup_steps must be nonnegative")
    if precision not in ("float32", "bf16"):
        raise ValueError("distributed precision must be float32 or bf16")
    if type(activation_checkpointing) is not bool:
        raise ValueError("activation_checkpointing must be boolean")
    if attention_backend not in ("flash", "flash2", "math"):
        raise ValueError("attention_backend must be flash, flash2 or math")
    if type(attention_query_chunk_size) is not int or attention_query_chunk_size <= 0:
        raise ValueError("attention_query_chunk_size must be a positive integer")
    if attention_backend in ("flash", "flash2") and precision != "bf16":
        raise ValueError("FlashAttention training requires bf16 autocast")
    if (not all(math.isfinite(x) for x in (learning_rate, weight_decay, grad_clip))
            or learning_rate <= 0 or weight_decay < 0 or grad_clip <= 0):
        raise ValueError("invalid optimizer configuration")
    if control_config.min_lr > learning_rate and control_config.lr_scheduler == "plateau":
        raise ValueError("min_lr must not exceed learning_rate")
    if {d.accession for d in train_datasets} & {d.accession for d in unseen_datasets}:
        raise ValueError("training and unseen datasets overlap")
    torch.set_num_threads(1)
    output_dir = Path(output_dir)
    with distributed_context(device, backend=distributed_backend) as (device, rank, world_size):
        if precision == "bf16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
            raise ValueError("bf16 training requires a CUDA GPU with bf16 support")

        def check_output():
            if output_dir.exists() and any(output_dir.iterdir()):
                raise FileExistsError(f"{output_dir}: choose a new empty output directory")
        _primary_action(check_output, rank)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(seed)
        with _sampler(cache_dir, train_datasets, train_fraction, batch_reads, seed,
                      steps_per_epoch, family_weights) as sampler:
            report = inspect_sampler(sampler, model_config, unseen_datasets,
                                     validation_max_reads_per_file)
            model = LargeQualityTransformer(model_config, attention_backend=attention_backend,
                attention_query_chunk_size=attention_query_chunk_size).to(device)
            if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
                raise ValueError("large Transformer master parameters must be float32")
            model.set_activation_checkpointing(activation_checkpointing)
            attention_metadata = model.training_attention_metadata()
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                          weight_decay=weight_decay)
            run_config = {
                "format": CHECKPOINT_FORMAT, "model_config": model_config.to_dict(),
                "feature_schema": feature_schema(model_config), "cache_schema": cache_schema(),
                "data_split": report["data_split"], "sampler_config": sampler.config_dict(),
                "optimizer": {"name": "AdamW", "learning_rate": learning_rate,
                              "weight_decay": weight_decay, "grad_clip": grad_clip},
                "initialization": {"mode": init_mode,
                                   "transformer_layers": "independent_pytorch_default"},
                "epochs": epochs, "steps_per_epoch": steps_per_epoch,
                "validation_max_reads_per_file": validation_max_reads_per_file,
                "validation_microbatch_reads": validation_microbatch_reads,
                "validation_read_rule": "fixed_start_of_validation_suffix_contiguous",
                "validation_read_counts": {key: row["validation_reads_evaluated"]
                                           for key, row in report["datasets"].items()},
                "selection_metric": SELECTION_METRIC, "device": device.type,
                "seed": seed, "attention_backend": attention_backend,
                "attention_implementation": attention_metadata,
                "torch_cpu_threads": 1, "training_control": control_config.to_dict(),
                "microbatch_reads": microbatch_reads, "warmup_steps": warmup_steps,
                "precision": {"training_autocast": precision, "parameters": "float32",
                              "optimizer_state": "float32", "validation": "float32",
                              "saved_model": "float32", "loss": "float32"},
                "activation_checkpointing": activation_checkpointing,
                "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda,
                    "device_name": (torch.cuda.get_device_name(device)
                                    if device.type == "cuda" else "cpu"),
                    "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                    "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                    "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                    "cudnn_benchmark": torch.backends.cudnn.benchmark,
                    "cudnn_deterministic": torch.backends.cudnn.deterministic},
                "distributed": {"world_size": world_size, "backend": str(dist.get_backend()),
                    "sampler": "identical_replicated_global_batch_then_contiguous_read_shards",
                    "loss": "global_active_quality_weighted_mean",
                    "empty_shards": "zero_weight_real_read_same_number_of_microbatch_slots",
                    "reducer_initialization": "two_zero_gradient_backwards_preserve_rng",
                    "rng": "independent_rank_streams_saved_per_rank"},
                "block_features": False,
            }
            # Catch accidentally inconsistent torchrun arguments before DDP.
            configs = [None] * world_size
            dist.all_gather_object(configs, run_config)
            if any(config != run_config for config in configs):
                raise ValueError("distributed ranks have different training configurations")
            best_bits, global_step, start_epoch = float("inf"), 0, 1
            total_tokens = 0
            tokens_by_dataset = {dataset.accession: 0 for dataset in train_datasets}
            saved_control, saved_rng, best_source = None, None, None
            if resume is not None:
                payload = _load_payload(resume, model_config)
                state = payload.get("training_state", {})
                if (state.get("version") != DISTRIBUTED_STATE_VERSION
                        or state.get("kind") != TRAINING_KIND
                        or resume_contract(state.get("run_config", {})) != resume_contract(run_config)):
                    raise ValueError("resume distributed training configuration differs from saved run")
                if payload.get("selection_metric") != SELECTION_METRIC:
                    raise ValueError("resume selection metric mismatch")
                if payload.get("optimizer_state_dict") is None:
                    raise ValueError("resume checkpoint lacks optimizer state")
                start_epoch = payload["epoch"] + 1
                if epochs < start_epoch:
                    raise ValueError("epochs must exceed the already completed epoch")
                sampler.load_state_dict(payload["sampler_state"])
                if (sampler._epoch != payload["epoch"]
                        or sampler._schedule_cursor != steps_per_epoch
                        or sampler._total_samples != payload["global_step"]):
                    raise ValueError("resume requires a completed epoch checkpoint")
                fingerprints = state.get("sampler_fingerprints", [])
                if (len(fingerprints) != world_size
                        or any(value != _sampler_fingerprint(sampler) for value in fingerprints)):
                    raise ValueError("resume per-rank sampler state differs")
                rank_states = state.get("rank_rng_states", [])
                if len(rank_states) != world_size:
                    raise ValueError("resume requires RNG state for every saved rank")
                if any(value.is_floating_point() and value.dtype != torch.float32
                       for value in payload["model_state_dict"].values()):
                    raise ValueError("resume master model weights must be float32")
                model.load_state_dict(payload["model_state_dict"], strict=True)
                optimizer.load_state_dict(payload["optimizer_state_dict"])
                global_step = payload["global_step"]
                total_tokens = state["training_quality_tokens"]
                tokens_by_dataset = dict(state["training_tokens_by_dataset"])
                if (set(tokens_by_dataset) != {dataset.accession for dataset in train_datasets}
                        or sum(tokens_by_dataset.values()) != total_tokens):
                    raise ValueError("resume training token counters are inconsistent")
                best_bits = payload["best_validation_bits_per_quality"]
                saved_control, saved_rng = state["control_state"], rank_states[rank]
                best_source = Path(resume).parent / "best.pt"
                # Release the last checkpoint before reading another large optimizer.
                del payload, state, rank_states
                previous_best = _load_payload(best_source, model_config)
                if (previous_best.get("selection_metric") != SELECTION_METRIC
                        or previous_best["validation_metrics"][SELECTION_METRIC] != best_bits
                        or resume_contract(previous_best["training_state"]["run_config"])
                        != resume_contract(run_config)):
                    raise ValueError("resume requires its matching best.pt beside the checkpoint")
                del previous_best
                run_config["resume_from"] = str(Path(resume).resolve())
            control = TrainingControl(optimizer, control_config, best_bits=best_bits)
            if saved_control is not None:
                control.load_state_dict(saved_control)
            ddp = DistributedDataParallel(
                model, device_ids=[device.index] if device.type == "cuda" else None,
                broadcast_buffers=False, gradient_as_bucket_view=True)
            # Model initialization is shared; dropout streams thereafter are rank-specific.
            if saved_rng is None:
                random.seed(seed + rank)
                np.random.seed(seed + rank)
                torch.manual_seed(seed + rank)
                if device.type == "cuda":
                    torch.cuda.manual_seed(seed + rank)
            else:
                _restore_rng(saved_rng, device)
            _prime_reducer(ddp, optimizer, sampler, device, precision)
            best_path = output_dir / "best.pt"

            def initialize_output():
                output_dir.mkdir(parents=True, exist_ok=True)
                if best_source is not None:
                    shutil.copyfile(best_source, best_path)
                write_json(output_dir / "run_config.json", run_config)
                write_json(output_dir / "preflight.json", report)
                write_json(output_dir / "training_corpus.json", report["corpus"])
            _primary_action(initialize_output, rank)
            parameters = sum(p.numel() for p in model.parameters())
            if progress and rank == 0:
                print(f"world_size={world_size} global_batch_reads={batch_reads} "
                      f"local_microbatch_reads={microbatch_reads} parameters={parameters} "
                      f"precision={precision} start_epoch={start_epoch} target_epoch={epochs}",
                      file=sys.stderr, flush=True)
            stopped_early = False
            for epoch in range(start_epoch, epochs + 1):
                ddp.train()
                epoch_nats, epoch_symbols = 0., 0
                epoch_tokens = dict.fromkeys(tokens_by_dataset, 0)
                epoch_lr = optimizer.param_groups[0]["lr"]
                step_iter = range(1, steps_per_epoch + 1)
                if progress and rank == 0 and tqdm is not None:
                    step_iter = tqdm(step_iter, total=steps_per_epoch,
                                     desc=f"epoch {epoch}/{epochs}", unit="batch", file=sys.stderr)
                for step in step_iter:
                    if global_step < warmup_steps:
                        for group in optimizer.param_groups:
                            group["lr"] = learning_rate * (global_step + 1) / warmup_steps
                    if step == 1:
                        epoch_lr = optimizer.param_groups[0]["lr"]
                    sample = sampler.sample_batch()
                    tensors = fastq_batch_to_tensors(sample.batch, device)
                    nats, symbols, grad_norm = distributed_train_step(
                        ddp, optimizer, tensors, microbatch_reads=microbatch_reads,
                        grad_clip=grad_clip, precision=precision)
                    global_step += 1
                    epoch_nats += nats
                    epoch_symbols += symbols
                    epoch_tokens[sample.dataset.accession] += symbols
                    if progress and rank == 0 and tqdm is not None and (
                            step % 10 == 0 or step == steps_per_epoch):
                        step_iter.set_postfix(train_bits=f"{epoch_nats / epoch_symbols / math.log(2):.4f}",
                                              lr=f"{optimizer.param_groups[0]['lr']:.3g}")
                validation = evaluate_distributed_validation(
                    model, sampler, max_reads_per_file=validation_max_reads_per_file,
                    microbatch_reads=validation_microbatch_reads, device=device)
                validation_bits = validation[SELECTION_METRIC]
                if not math.isfinite(validation_bits):
                    raise ValueError("validation macro BPQ must be finite")
                # Every rank sees the same reduced metric, scheduler and LR.
                stopped_early = control.step(validation_bits)
                total_tokens += epoch_symbols
                for accession, tokens in epoch_tokens.items():
                    tokens_by_dataset[accession] += tokens
                improved = validation_bits < best_bits
                if improved:
                    best_bits = validation_bits
                rank_states, fingerprints = [None] * world_size, [None] * world_size
                dist.all_gather_object(rank_states, _rng_state(device))
                dist.all_gather_object(fingerprints, _sampler_fingerprint(sampler))
                if len(set(fingerprints)) != 1:
                    raise ValueError("distributed rank samplers diverged")
                record = {
                    "epoch": epoch, "global_step": global_step,
                    "epoch_training_quality_tokens": epoch_symbols,
                    "training_quality_tokens": total_tokens,
                    "epoch_tokens_by_dataset": epoch_tokens,
                    "training_tokens_by_dataset": dict(tokens_by_dataset),
                    "train_loss": epoch_nats / epoch_symbols,
                    "train_bits_per_quality": epoch_nats / epoch_symbols / math.log(2),
                    "validation": validation, "learning_rate": epoch_lr,
                    "next_learning_rate": optimizer.param_groups[0]["lr"],
                    "last_gradient_norm": grad_norm,
                    "early_stopping_bad_epochs": control.bad_epochs,
                    "stopped_early": stopped_early,
                }
                payload = {
                    "epoch": epoch, "global_step": global_step,
                    "data_split": report["data_split"], "sampler_config": sampler.config_dict(),
                    "sampler_statistics": sampler.statistics_dict(),
                    "sampler_state": sampler.state_dict(), "selection_metric": SELECTION_METRIC,
                    "best_validation_bits_per_quality": best_bits, "validation_metrics": validation,
                    "training_state": {"version": DISTRIBUTED_STATE_VERSION, "kind": TRAINING_KIND,
                        "run_config": run_config, "training_quality_tokens": total_tokens,
                        "training_tokens_by_dataset": dict(tokens_by_dataset),
                        "control_state": control.state_dict(), "rank_rng_states": rank_states,
                        "sampler_fingerprints": fingerprints},
                }

                def save_epoch():
                    with (output_dir / "training_log.jsonl").open("a") as handle:
                        handle.write(json.dumps(record, sort_keys=True) + "\n")
                    save_checkpoint(output_dir / "last.pt", model=model, optimizer=optimizer, **payload)
                    if improved:
                        save_checkpoint(best_path, model=model, optimizer=optimizer, **payload)
                    write_json(output_dir / "sampling_statistics.json", sampler.statistics_dict())
                    write_json(output_dir / "training_summary.json", {
                        "start_epoch": start_epoch, "completed_epoch": epoch,
                        "requested_epochs": epochs, "global_step": global_step,
                        "training_quality_tokens": total_tokens,
                        "training_tokens_by_dataset": dict(tokens_by_dataset),
                        "training_corpus_quality_tokens": report["corpus"]["train_quality_tokens"],
                        "best_validation_bits_per_quality": best_bits, "model_parameters": parameters,
                        "world_size": world_size, "precision": run_config["precision"],
                        "final_learning_rate": optimizer.param_groups[0]["lr"],
                        "stopped_early": stopped_early})
                _primary_action(save_epoch, rank)
                if progress and rank == 0:
                    print(f"epoch={epoch} global_step={global_step} "
                          f"train_bits={record['train_bits_per_quality']:.6f} "
                          f"val_bits={validation_bits:.6f} best_val_bits={best_bits:.6f} "
                          f"next_lr={optimizer.param_groups[0]['lr']:.6g}",
                          file=sys.stderr, flush=True)
                if stopped_early:
                    break
            return best_path
