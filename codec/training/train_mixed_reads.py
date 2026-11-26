"""Train the original four-layer Transformer with balanced, shuffled reads."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Optional, Sequence

import torch

from codec.data.datasets import EXPANDED_FAMILY_WEIGHTS, EXPANDED_TRAIN_DATASETS
from codec.diagnostics.evaluate import choose_device
from codec.training.mixed_read_training import preflight, run_training
from codec.models.transformer import DirectQualityModelConfig, DirectQualityTransformer, feature_schema
from codec.data.training_cache import DEFAULT_CACHE_DIR
from codec.training.control import TrainingControlConfig


MODEL_WIDTHS = {"2m": 256, "1p5m": 208, "1m": 168}


def model_config_for_size(size: str) -> DirectQualityModelConfig:
    if size not in MODEL_WIDTHS:
        raise ValueError(f"unknown Transformer size: {size}")
    width = MODEL_WIDTHS[size]
    return replace(DirectQualityModelConfig(), d_model=width, feedforward_dim=2 * width)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-size", choices=tuple(MODEL_WIDTHS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="new empty run directory, including for a resume")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps-per-epoch", type=int, default=2000)
    parser.add_argument("--batch-reads", type=int, default=256)
    parser.add_argument("--microbatch-reads", type=int, default=0)
    parser.add_argument("--train-fraction", type=float, default=.9)
    parser.add_argument("--validation-max-reads-per-file", type=int, default=5000)
    parser.add_argument("--learning-rate", type=float, default=.001)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--lr-scheduler", choices=("none", "plateau"), default="plateau")
    parser.add_argument("--lr-factor", type=float, default=.5)
    parser.add_argument("--lr-patience", type=int, default=2)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--preload-training-values", action="store_true",
                        help="read the fixed training prefixes into RAM; no data is copied on disk")
    parser.add_argument("--loss-smoothing-window", type=int, default=200)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--resume", type=Path, help="resume this same mixed-read configuration")
    parser.add_argument("--dry-run", action="store_true",
                        help="print configuration without opening caches or creating outputs")
    parser.add_argument("--preflight-only", action="store_true",
                        help="also inspect explicit cache metadata and split boundaries, then exit")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.cpu_threads <= 0:
            raise ValueError("cpu-threads must be positive")
        torch.set_num_threads(args.cpu_threads)
        config = model_config_for_size(args.model_size)
        control = TrainingControlConfig(
            lr_scheduler=args.lr_scheduler, lr_factor=args.lr_factor,
            lr_patience=args.lr_patience, min_lr=args.min_lr,
            min_delta=args.min_delta, early_stopping_patience=args.early_stopping_patience)
        parameter_count = sum(p.numel() for p in DirectQualityTransformer(config).parameters())
        description = {
            "model_size": args.model_size, "model_parameters": parameter_count,
            "model_config": config.to_dict(), "feature_schema": feature_schema(config),
            "training_set": "expanded20", "sampling_mode": "mixed_read",
            "training_accessions": [d.accession for d in EXPANDED_TRAIN_DATASETS],
            "family_weights": EXPANDED_FAMILY_WEIGHTS, "precision": "float32",
            "output_dir": str(args.output_dir.resolve()),
            "cache_dir": str(args.cache_dir.resolve()),
            "initialization": "strict_resume" if args.resume else "random",
            "resume_from": str(args.resume.resolve()) if args.resume else None,
            "epochs": args.epochs, "steps_per_epoch": args.steps_per_epoch,
            "target_updates": args.epochs * args.steps_per_epoch,
            "batch_reads": args.batch_reads, "microbatch_reads": args.microbatch_reads,
            "train_fraction": args.train_fraction,
            "validation_max_reads_per_file": args.validation_max_reads_per_file,
            "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip, "seed": args.seed,
            "training_control": control.to_dict(),
            "preload_training_values": args.preload_training_values,
            "loss_logging": {"per_batch": True, "per_epoch": True, "tensorboard": True,
                             "smoothing_window": args.loss_smoothing_window}}
        if args.preflight_only:
            description["cache_preflight"] = preflight(
                model_config=config, cache_dir=args.cache_dir,
                train_fraction=args.train_fraction, batch_reads=args.batch_reads,
                seed=args.seed, steps_per_epoch=args.steps_per_epoch,
                validation_max_reads_per_file=args.validation_max_reads_per_file,
                preload_training_values=args.preload_training_values)
        if args.dry_run or args.preflight_only:
            print(json.dumps(description, indent=2, sort_keys=True))
            return 0
        print(json.dumps(description, sort_keys=True), flush=True)
        best = run_training(
            model_config=config, output_dir=args.output_dir, cache_dir=args.cache_dir,
            resume=args.resume, epochs=args.epochs, steps_per_epoch=args.steps_per_epoch,
            batch_reads=args.batch_reads, microbatch_reads=args.microbatch_reads,
            train_fraction=args.train_fraction,
            validation_max_reads_per_file=args.validation_max_reads_per_file,
            learning_rate=args.learning_rate, weight_decay=args.weight_decay,
            grad_clip=args.grad_clip, device=choose_device(args.device), seed=args.seed,
            progress=not args.no_progress, control_config=control,
            preload_training_values=args.preload_training_values,
            log_loss_curves=True, loss_smoothing_window=args.loss_smoothing_window)
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(f"best checkpoint: {best}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
