"""FP32 Transformer training with source-balanced, shuffled individual reads.

This entry point leaves the legacy single-file trainer and checkpoint wire
format unchanged. The training-state marker makes continuation explicit; a
legacy checkpoint can still be used by the codec but cannot silently resume
under a different sampling objective.
"""

import json
import math
import os
from pathlib import Path
import random
import shutil
import tempfile
from contextlib import ExitStack

import numpy as np
import torch

from .checkpoint import (CHECKPOINT_SCHEMA_VERSION, cache_schema,
                         load_training_checkpoint, save_training_checkpoint)
from .datasets import (EXPANDED_FAMILY_WEIGHTS, EXPANDED_TRAIN_DATASETS,
                       EXPANDED_UNSEEN_DATASETS)
from .evaluate import evaluate_reader_range, summarize_metric_group
from .mixed_read_loss import source_weighted_cross_entropy
from .mixed_read_sampling import MixedReadTrainingSampler
from .model import DirectQualityTransformer, fastq_batch_to_tensors, feature_schema
from .training_cache import DEFAULT_CACHE_DIR
from .training_control import TrainingControl, TrainingControlConfig

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

SELECTION_METRIC = "dataset_macro_bits_per_quality"
DEFAULT_CONTROL = TrainingControlConfig(lr_scheduler="plateau")
TRAINER = "source_weighted_mixed_read_transformer_v1"
TRAINING_STATE_VERSION = 2


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp",
                                            dir=path.parent)
    os.close(descriptor)
    try:
        Path(temporary).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _validate_model_config(config):
    optional = ("quality_prefix_features", "relative_position_bias", "base_cross_attention",
                "block_quality_features", "short_quality_features", "quality_history_cnn",
                "public_read_quality_state")
    if (any(getattr(config, name, False) for name in optional)
            or getattr(config, "base_encoder_type", "local_cnn") != "local_cnn"
            or getattr(config, "ffn_type", "gelu") != "gelu"):
        raise ValueError("mixed-read comparison requires the original Transformer features")


def _sampler(cache_dir, train_datasets, train_fraction, batch_reads, seed,
             steps_per_epoch, family_weights, preload_training_values):
    return MixedReadTrainingSampler(
        train_datasets, Path(cache_dir), train_fraction=train_fraction,
        batch_reads=batch_reads, seed=seed, steps_per_epoch=steps_per_epoch,
        family_weights=family_weights, feature_block_reads=0,
        preload_training_values=preload_training_values)


def inspect_sampler(sampler, model_config, unseen_datasets, validation_max_reads_per_file):
    """Check every explicit cache at both split boundaries without truncation."""
    _validate_model_config(model_config)
    if validation_max_reads_per_file < 0:
        raise ValueError("validation_max_reads_per_file must be nonnegative")
    entries = {}
    for dataset in sampler.datasets:
        reader = sampler.reader_for(dataset.accession)
        metadata = reader.metadata
        train_stop = sampler.train_stop_for(dataset.accession)
        for start, stop in ((0, min(sampler.batch_reads, train_stop)),
                            (train_stop, min(train_stop + sampler.batch_reads,
                                             metadata.read_count))):
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
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "trainer": TRAINER, "model_config": model_config.to_dict(),
        "feature_schema": feature_schema(model_config), "cache_schema": cache_schema(),
        "data_split": sampler.data_split_manifest(unseen_datasets),
        "sampler_config": sampler.config_dict(),
        "corpus": sampler.corpus_statistics_dict(), "datasets": entries,
        "preflight": "metadata and first training/validation cache batches; not a full cache rescan",
        "reads_truncated": False, "block_features": False,
    }


def preflight(*, model_config, cache_dir=DEFAULT_CACHE_DIR, train_fraction=.9,
              batch_reads=256, seed=20260909, steps_per_epoch=2000,
              validation_max_reads_per_file=5000,
              train_datasets=EXPANDED_TRAIN_DATASETS,
              unseen_datasets=EXPANDED_UNSEEN_DATASETS,
              family_weights=EXPANDED_FAMILY_WEIGHTS, preload_training_values=False):
    with _sampler(cache_dir, train_datasets, train_fraction, batch_reads, seed,
                  steps_per_epoch, family_weights, preload_training_values) as sampler:
        return inspect_sampler(sampler, model_config, unseen_datasets,
                               validation_max_reads_per_file)


def evaluate_validation(model, sampler, *, max_reads_per_file, device):
    metrics = []
    for dataset in sampler.datasets:
        reader = sampler.reader_for(dataset.accession)
        metrics.append(evaluate_reader_range(
            model, reader, dataset, category="validation",
            start=sampler.train_stop_for(dataset.accession),
            stop=reader.metadata.read_count, batch_reads=sampler.batch_reads,
            max_reads=max_reads_per_file, device=device))
    return {**summarize_metric_group(metrics),
            "per_dataset": {metric.accession: metric.to_dict() for metric in metrics}}


def resume_contract(run_config):
    return {key: value for key, value in run_config.items()
            if key not in ("epochs", "resume_from")}


def _rng_state(device):
    return {"python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}


def _restore_rng(state, device):
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"].cpu())
    if device.type == "cuda":
        if len(state["cuda_rng"]) != torch.cuda.device_count():
            raise ValueError("resume CUDA device count differs from saved RNG state")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng"]])


def _continuation_state(payload):
    state = payload.get("training_state")
    if (not isinstance(state, dict) or state.get("version") != TRAINING_STATE_VERSION
            or state.get("trainer") != TRAINER):
        raise ValueError("resume requires a mixed-read Transformer training state")
    return state


def run_training(*, model_config, output_dir, cache_dir=DEFAULT_CACHE_DIR,
                 init_mode="random", resume=None,
                 epochs=100, steps_per_epoch=2000, batch_reads=256,
                 train_fraction=.9, validation_max_reads_per_file=5000,
                 learning_rate=1e-3, weight_decay=1e-5, grad_clip=1.,
                 device=torch.device("cpu"), seed=20260909, progress=True,
                 control_config=DEFAULT_CONTROL, microbatch_reads=0,
                 train_datasets=EXPANDED_TRAIN_DATASETS,
                 unseen_datasets=EXPANDED_UNSEEN_DATASETS,
                 family_weights=EXPANDED_FAMILY_WEIGHTS,
                 preload_training_values=False, log_loss_curves=True,
                 loss_smoothing_window=200):
    """Train from scratch or continue an exact, completed-epoch checkpoint.

    Each microchunk uses the *whole batch's* per-source Q denominators, and
    chunk objectives are summed before the one optimizer step. Read length
    therefore changes neither source probabilities nor their objective weights.
    """
    output_dir, device = Path(output_dir), torch.device(device)
    _validate_model_config(model_config)
    if type(epochs) is not int or epochs <= 0 or type(steps_per_epoch) is not int or steps_per_epoch <= 0:
        raise ValueError("epochs and steps_per_epoch must be positive integers")
    if init_mode != "random":
        raise ValueError("mixed-read Transformer supports random initialization only")
    if type(microbatch_reads) is not int or not 0 <= microbatch_reads <= batch_reads:
        raise ValueError("microbatch_reads must be between 0 and batch_reads")
    if type(loss_smoothing_window) is not int or loss_smoothing_window <= 0:
        raise ValueError("loss_smoothing_window must be a positive integer")
    if (not all(math.isfinite(x) for x in (learning_rate, weight_decay, grad_clip))
            or learning_rate <= 0 or weight_decay < 0 or grad_clip <= 0):
        raise ValueError("invalid optimizer configuration")
    if control_config.min_lr > learning_rate and control_config.lr_scheduler == "plateau":
        raise ValueError("min_lr must not exceed learning_rate")
    if {d.accession for d in train_datasets} & {d.accession for d in unseen_datasets}:
        raise ValueError("training and unseen datasets overlap")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"{output_dir}: choose a new empty output directory")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    with ExitStack() as stack:
        sampler = stack.enter_context(_sampler(
            cache_dir, train_datasets, train_fraction, batch_reads, seed,
            steps_per_epoch, family_weights, preload_training_values))
        report = inspect_sampler(sampler, model_config, unseen_datasets,
                                 validation_max_reads_per_file)
        # Preserve the original model's TransformerEncoder cloned initialization.
        model = DirectQualityTransformer(model_config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                      weight_decay=weight_decay)
        run_config = {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "trainer": TRAINER, "model_config": model_config.to_dict(),
            "feature_schema": feature_schema(model_config), "cache_schema": cache_schema(),
            "data_split": report["data_split"], "sampler_config": sampler.config_dict(),
            "optimizer": {"name": "AdamW", "learning_rate": learning_rate,
                          "weight_decay": weight_decay, "grad_clip": grad_clip},
            "initialization": {"mode": "random", "transformer_layers": "torch_encoder_cloned"},
            "epochs": epochs, "steps_per_epoch": steps_per_epoch,
            "validation_max_reads_per_file": validation_max_reads_per_file,
            "validation_read_rule": "fixed_start_of_validation_suffix_contiguous",
            "validation_read_counts": {name: row["validation_reads_evaluated"]
                                       for name, row in report["datasets"].items()},
            "selection_metric": SELECTION_METRIC,
            "device": str(device), "seed": seed,
            "torch_cpu_threads": torch.get_num_threads(),
            "training_control": control_config.to_dict(),
            "microbatch_reads": microbatch_reads, "precision": "float32",
            "block_features": False,
            "loss_objective": {
                "strategy": "source_probability_weighted_mean_quality_cross_entropy",
                "source_probabilities": dict(sampler.source_probabilities),
                "padding_enters_loss": False},
        }
        if log_loss_curves:
            run_config["loss_logging"] = {
                "per_batch": True, "tensorboard": True,
                "curve_refresh": "each_completed_epoch_and_on_close",
                "smoothing_window": loss_smoothing_window}
        best_bits, global_step, start_epoch, total_tokens = float("inf"), 0, 1, 0
        tokens_by_dataset = {dataset.accession: 0 for dataset in train_datasets}
        reads_by_dataset = dict.fromkeys(tokens_by_dataset, 0)
        saved_control = best_source = loaded = None
        if resume is not None:
            # Checkpoint loading constructs a model and must not alter dropout RNG.
            with torch.random.fork_rng(devices=[]):
                loaded = load_training_checkpoint(resume)
            payload = loaded.payload
            state = _continuation_state(payload)
            if resume_contract(state["run_config"]) != resume_contract(run_config):
                raise ValueError("resume configuration differs from saved run")
            if payload.get("selection_metric") != SELECTION_METRIC:
                raise ValueError("resume selection metric mismatch")
            if payload.get("optimizer_state_dict") is None:
                raise ValueError("resume checkpoint lacks optimizer state")
            start_epoch = payload["epoch"] + 1
            if epochs < start_epoch:
                raise ValueError("epochs is a cumulative target and must exceed completed epoch")
            sampler.load_state_dict(payload["sampler_state"])
            if (sampler._epoch != payload["epoch"]
                    or sampler._schedule_cursor != steps_per_epoch
                    or sampler._total_samples != payload["global_step"]):
                raise ValueError("resume requires a completed epoch checkpoint")
            model.load_state_dict(loaded.model.state_dict(), strict=True)
            optimizer.load_state_dict(payload["optimizer_state_dict"])
            global_step = payload["global_step"]
            total_tokens = state["training_quality_tokens"]
            tokens_by_dataset = dict(state["training_tokens_by_dataset"])
            reads_by_dataset = dict(state["training_reads_by_dataset"])
            best_bits = payload["best_validation_bits_per_quality"]
            saved_control = state["control_state"]
            best_source = Path(resume).parent / "best.pt"
            with torch.random.fork_rng(devices=[]):
                previous_best = load_training_checkpoint(best_source).payload
            best_state = _continuation_state(previous_best)
            if (previous_best.get("selection_metric") != SELECTION_METRIC
                    or previous_best["validation_metrics"][SELECTION_METRIC] != best_bits
                    or resume_contract(best_state["run_config"])
                    != resume_contract(state["run_config"])):
                raise ValueError("resume requires its matching best.pt beside the checkpoint")
            run_config["resume_from"] = str(Path(resume).resolve())
        control = TrainingControl(optimizer, control_config, best_bits=best_bits)
        if saved_control is not None:
            control.load_state_dict(saved_control)
        if loaded is not None:
            _restore_rng(loaded.payload["training_state"], device)
        # All compatibility checks precede the first output artifact.
        output_dir.mkdir(parents=True, exist_ok=True)
        best_path = output_dir / "best.pt"
        if best_source is not None:
            shutil.copyfile(best_source, best_path)
        write_json(output_dir / "run_config.json", run_config)
        write_json(output_dir / "preflight.json", report)
        write_json(output_dir / "training_corpus.json", report["corpus"])
        logger = None
        if log_loss_curves:
            from .training_loss_logging import TrainingLossLogger
            logger = TrainingLossLogger(output_dir, smoothing_window=loss_smoothing_window,
                                        require_tensorboard=True)
            stack.callback(logger.close)
            if resume is not None:
                logger.import_history(Path(resume).parent, max_global_step=global_step)
        source_names = [dataset.accession for dataset in train_datasets]
        source_probabilities = torch.tensor(
            [sampler.source_probabilities[name] for name in source_names],
            dtype=torch.float32, device=device)
        if progress:
            print(f"training_files={len(train_datasets)} train_quality_tokens="
                  f"{report['corpus']['train_quality_tokens']} parameters="
                  f"{sum(p.numel() for p in model.parameters())} "
                  f"start_epoch={start_epoch} target_epoch={epochs}", flush=True)
        for epoch in range(start_epoch, epochs + 1):
            model.train()
            epoch_lr = optimizer.param_groups[0]["lr"]
            epoch_nats, epoch_symbols, epoch_objective = 0., 0, 0.
            epoch_tokens = dict.fromkeys(tokens_by_dataset, 0)
            epoch_reads = dict.fromkeys(tokens_by_dataset, 0)
            step_iter = range(1, steps_per_epoch + 1)
            if progress and tqdm is not None:
                step_iter = tqdm(step_iter, total=steps_per_epoch,
                                 desc=f"epoch {epoch}/{epochs}", unit="batch")
            for step in step_iter:
                sample = sampler.sample_batch()
                tensors = fastq_batch_to_tensors(sample.batch, device)
                symbols = int(tensors["active_mask"].sum().item())
                if not symbols:
                    raise ValueError("training batch contains no quality symbols")
                source_reads = {name: int(count) for name, count
                                in sample.source_read_counts.items()}
                source_symbols = {name: int(sample.batch.lengths[
                    sample.row_dataset_indices == index].sum())
                    for index, name in enumerate(source_names)}
                if any(count <= 0 for count in source_symbols.values()):
                    raise ValueError("each mixed-batch source must contribute quality symbols")
                row_sources = torch.as_tensor(sample.row_dataset_indices,
                                              dtype=torch.long, device=device)
                source_denominators = torch.tensor(
                    [source_symbols[name] for name in source_names],
                    dtype=torch.float32, device=device)
                source_nats = dict.fromkeys(source_names, 0.)
                optimizer.zero_grad(set_to_none=True)
                loss_value = 0.
                micro = microbatch_reads or sample.batch.read_count
                for begin in range(0, sample.batch.read_count, micro):
                    end = min(begin + micro, sample.batch.read_count)
                    chunk = {key: value[begin:end] for key, value in tensors.items()}
                    if not chunk["active_mask"].any():
                        continue
                    logits = model.forward_full(**chunk)
                    loss, chunk_source_nll = source_weighted_cross_entropy(
                        logits, chunk["qualities"], chunk["active_mask"],
                        row_sources[begin:end], source_probabilities, source_denominators)
                    if not torch.isfinite(loss):
                        raise ValueError("nonfinite training loss")
                    loss.backward()
                    loss_value += float(loss.detach().item())
                    for name, nats in zip(source_names, chunk_source_nll.detach().cpu().tolist()):
                        source_nats[name] += nats
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip,
                                               error_if_nonfinite=True)
                optimizer.step()
                global_step += 1
                pooled_nats = sum(source_nats.values())
                pooled_loss = pooled_nats / symbols
                epoch_nats += pooled_nats
                epoch_symbols += symbols
                epoch_objective += loss_value
                for name in source_names:
                    epoch_tokens[name] += source_symbols[name]
                    epoch_reads[name] += source_reads[name]
                if logger is not None:
                    logger.log_batch({
                        "global_step": global_step, "epoch": epoch, "batch_in_epoch": step,
                        "loss_nats": loss_value, "pooled_loss_nats": pooled_loss,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "read_count": sample.batch.read_count, "quality_symbols": symbols,
                        "source_read_counts": source_reads,
                        "source_quality_symbols": source_symbols,
                        "source_loss_nats": {name: source_nats[name] / source_symbols[name]
                                             for name in source_names}})
                if progress and tqdm is not None and (step % 10 == 0 or step == steps_per_epoch):
                    step_iter.set_postfix(train_bits=f"{epoch_nats / epoch_symbols / math.log(2):.4f}")
            validation = evaluate_validation(model, sampler,
                max_reads_per_file=validation_max_reads_per_file, device=device)
            validation_bits = validation[SELECTION_METRIC]
            if not math.isfinite(validation_bits):
                raise ValueError("validation macro BPQ must be finite")
            stopped_early = control.step(validation_bits)
            total_tokens += epoch_symbols
            for name in source_names:
                tokens_by_dataset[name] += epoch_tokens[name]
                reads_by_dataset[name] += epoch_reads[name]
            improved = validation_bits < best_bits
            if improved:
                best_bits = validation_bits
            record = {
                "epoch": epoch, "global_step": global_step,
                "epoch_training_quality_tokens": epoch_symbols,
                "training_quality_tokens": total_tokens,
                "epoch_tokens_by_dataset": epoch_tokens,
                "training_tokens_by_dataset": dict(tokens_by_dataset),
                "epoch_reads_by_dataset": epoch_reads,
                "training_reads_by_dataset": dict(reads_by_dataset),
                "train_loss": epoch_nats / epoch_symbols,
                "train_objective_loss": epoch_objective / steps_per_epoch,
                "train_bits_per_quality": epoch_nats / epoch_symbols / math.log(2),
                "validation": validation, "learning_rate": epoch_lr,
                "next_learning_rate": optimizer.param_groups[0]["lr"],
                "early_stopping_bad_epochs": control.bad_epochs,
                "stopped_early": stopped_early}
            with (output_dir / "training_log.jsonl").open("a") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            if logger is not None:
                logger.log_epoch(record)
            payload = {
                "epoch": epoch, "global_step": global_step,
                "data_split": report["data_split"], "sampler_config": sampler.config_dict(),
                "sampler_statistics": sampler.statistics_dict(),
                "sampler_state": sampler.state_dict(), "selection_metric": SELECTION_METRIC,
                "best_validation_bits_per_quality": best_bits, "validation_metrics": validation,
                "training_state": {
                    "version": TRAINING_STATE_VERSION, "trainer": TRAINER,
                    "run_config": run_config, "training_quality_tokens": total_tokens,
                    "training_tokens_by_dataset": dict(tokens_by_dataset),
                    "training_reads_by_dataset": dict(reads_by_dataset),
                    "control_state": control.state_dict(), **_rng_state(device)}}
            save_training_checkpoint(output_dir / "last.pt", model=model,
                                     optimizer=optimizer, **payload)
            if improved:
                save_training_checkpoint(best_path, model=model, optimizer=optimizer, **payload)
            write_json(output_dir / "sampling_statistics.json", sampler.statistics_dict())
            write_json(output_dir / "training_summary.json", {
                "trainer": TRAINER, "start_epoch": start_epoch, "completed_epoch": epoch,
                "requested_epochs": epochs, "global_step": global_step,
                "training_quality_tokens": total_tokens,
                "training_tokens_by_dataset": tokens_by_dataset,
                "training_reads_by_dataset": reads_by_dataset,
                "training_corpus_quality_tokens": report["corpus"]["train_quality_tokens"],
                "best_validation_bits_per_quality": best_bits,
                "model_parameters": sum(p.numel() for p in model.parameters()),
                "final_learning_rate": optimizer.param_groups[0]["lr"],
                "stopped_early": stopped_early})
            if progress:
                print(f"epoch={epoch} train_bits={record['train_bits_per_quality']:.6f} "
                      f"val_bits={validation_bits:.6f} best_val_bits={best_bits:.6f} "
                      f"lr={epoch_lr:.6g} next_lr={optimizer.param_groups[0]['lr']:.6g}", flush=True)
            if stopped_early:
                break
    return best_path
