"""Controlled expanded20 architecture and model-size pretraining experiments.

Each invocation allocates its own empty run directory, including when multiple
terminals start the same experiment concurrently. No scratch-file training or
read-state labels are used here.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Optional, Sequence

from .datasets import (
    EXPANDED_FAMILY_WEIGHTS,
    EXPANDED_TRAIN_DATASETS,
    EXPANDED_UNSEEN_DATASETS,
)
from .evaluate import choose_device
from .model import DirectQualityModelConfig, DirectQualityTransformer, feature_schema
from .train import run_training
from .training_cache import DEFAULT_CACHE_DIR
from .training_control import TrainingControlConfig


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUN_GROUP = REPOSITORY_ROOT / "codec/runs/architecture_ablation_20261020"
DEFAULT_SCALING_RUN_GROUP = REPOSITORY_ROOT / "codec/runs/model_scaling_20261021"
DEFAULT_CONDITIONING_RUN_GROUP = REPOSITORY_ROOT / "codec/runs/conditioning_architecture_20261022"
CONDITIONING_EXPERIMENT_CHANGES = {
    "relative_position": {"relative_position_bias": True},
    "base_encoder_decoder": {"base_cross_attention": True},
}
SCALING_EXPERIMENT_CHANGES = {
    "depth6": {"num_layers": 6},
    "depth8": {"num_layers": 8},
    "wide512": {"d_model": 512, "num_heads": 8, "feedforward_dim": 1024},
}
EXPERIMENT_CHANGES = {
    "ffn1024": {"feedforward_dim": 1024},
    # Keep 64 channels per attention head and the original FFN/model ratio.
    "wide384": {"d_model": 384, "num_heads": 6, "feedforward_dim": 768},
    "base_dilated": {"base_encoder_type": "dilated_cnn"},
    # 3*344 is close to the GELU baseline's 2*512 FFN matrix budget.
    "swiglu": {"ffn_type": "swiglu", "feedforward_dim": 344},
    "qprefix": {"quality_prefix_features": True},
    **SCALING_EXPERIMENT_CHANGES,
    **CONDITIONING_EXPERIMENT_CHANGES,
}


def experiment_config(name: str) -> DirectQualityModelConfig:
    try:
        changes = EXPERIMENT_CHANGES[name]
    except KeyError as exc:
        raise ValueError(f"unknown pretraining experiment: {name}") from exc
    return replace(DirectQualityModelConfig(), **changes)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=tuple(EXPERIMENT_CHANGES))
    parser.add_argument(
        "--run-group", type=Path,
        help="override the architecture- or model-scaling-specific run directory",
    )
    parser.add_argument("--cache-dir", type=Path, default=REPOSITORY_ROOT / DEFAULT_CACHE_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps-per-epoch", type=int, default=2000)
    parser.add_argument("--batch-reads", type=int, default=256)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--validation-max-reads-per-file", type=int, default=5000)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--lr-scheduler", choices=("none", "plateau"), default="plateau")
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--lr-patience", type=int, default=2)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--resume", type=Path,
                        help="restore a matching last.pt into a NEW run directory")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the configuration without allocating a run or training")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.run_group is None:
        if args.experiment in CONDITIONING_EXPERIMENT_CHANGES:
            args.run_group = DEFAULT_CONDITIONING_RUN_GROUP
        elif args.experiment in SCALING_EXPERIMENT_CHANGES:
            args.run_group = DEFAULT_SCALING_RUN_GROUP
        else:
            args.run_group = DEFAULT_RUN_GROUP
    try:
        config = experiment_config(args.experiment)
        control = TrainingControlConfig(
            lr_scheduler=args.lr_scheduler, lr_factor=args.lr_factor,
            lr_patience=args.lr_patience, min_lr=args.min_lr,
            min_delta=args.min_delta,
            early_stopping_patience=args.early_stopping_patience,
        )
        parameter_count = sum(p.numel() for p in DirectQualityTransformer(config).parameters())
        description = {
            "experiment": args.experiment,
            "model_config": config.to_dict(),
            "feature_schema": feature_schema(config),
            "model_parameters": parameter_count,
            "training_set": "expanded20",
            "training_accessions": [d.accession for d in EXPANDED_TRAIN_DATASETS],
            "family_weights": EXPANDED_FAMILY_WEIGHTS,
            "cache_dir": str(args.cache_dir.resolve()),
            "run_group": str(args.run_group.resolve()),
            "epochs": args.epochs,
            "steps_per_epoch": args.steps_per_epoch,
            "target_updates": args.epochs * args.steps_per_epoch,
            "batch_reads": args.batch_reads,
            "train_fraction": args.train_fraction,
            "validation_max_reads_per_file": args.validation_max_reads_per_file,
            "optimizer": {
                "name": "AdamW", "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay, "grad_clip": args.grad_clip,
            },
            "seed": args.seed,
            "training_control": control.to_dict(),
            "initialization": "resume" if args.resume else "random",
            "resume_from": str(args.resume.resolve()) if args.resume else None,
        }
        if args.dry_run:
            print(json.dumps(description, indent=2, sort_keys=True))
            return 0
        device = choose_device(args.device)
        missing = [
            str(d.cache_path(args.cache_dir))
            for d in EXPANDED_TRAIN_DATASETS
            if not d.cache_path(args.cache_dir).is_file()
        ]
        if missing:
            raise FileNotFoundError("missing training caches: " + ", ".join(missing))
        args.run_group.mkdir(parents=True, exist_ok=True)
        output_dir = Path(tempfile.mkdtemp(prefix=args.experiment + "_", dir=args.run_group))
        print(f"experiment={args.experiment} parameters={parameter_count:,} "
              f"target_updates={description['target_updates']:,}", flush=True)
        print(f"run directory: {output_dir.resolve()}", flush=True)
        best_path = run_training(
            cache_dir=args.cache_dir,
            output_dir=output_dir,
            model_config=config,
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            batch_reads=args.batch_reads,
            train_fraction=args.train_fraction,
            validation_max_reads_per_file=args.validation_max_reads_per_file,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            device=device,
            seed=args.seed,
            train_datasets=EXPANDED_TRAIN_DATASETS,
            unseen_datasets=EXPANDED_UNSEEN_DATASETS,
            family_weights=EXPANDED_FAMILY_WEIGHTS,
            progress=not args.no_progress,
            resume=args.resume,
            control_config=control,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(f"best checkpoint: {best_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
