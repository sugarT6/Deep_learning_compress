#!/usr/bin/env python3
"""Train the 0.5B quality Transformer with torchrun and strict FlashAttention."""

import argparse
import json
import os
from pathlib import Path

import torch

from .evaluate import choose_device
from .large_transformer import LargeQualityTransformer
from .large_transformer_training import (DEFAULT_CACHE_DIR, TRAINING_KIND, preflight,
                                         run_distributed_training, write_json)
from .model import DirectQualityModelConfig
from .training_control import TrainingControlConfig


DEFAULT_MODEL_CONFIG = Path(__file__).resolve().parent / "configs" / "transformer_500m.json"


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path,
                        help="new empty run directory; required unless --dry-run")
    parser.add_argument("--model-config", type=Path,
                        help="model JSON overrides; fresh default codec/configs/transformer_500m.json")
    parser.add_argument("--resume", type=Path,
                        help="completed-epoch last.pt with its matching best.pt beside it")
    for name in ("d-model", "num-heads", "num-layers", "feedforward-dim", "context-length"):
        parser.add_argument("--" + name, type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--attention-backend", choices=("flash", "flash2", "math"), default="flash",
                        help="flash/flash2 are strict CUDA BF16 with no fallback; math supports CPU tests")
    parser.add_argument("--attention-query-chunk-size", type=int, default=16,
                        help="bounded query chunk for exact sliding-window FlashAttention")
    parser.add_argument("--precision", choices=("bf16", "float32"), default="bf16")
    parser.add_argument("--activation-checkpointing", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--epochs", type=int, default=100,
                        help="cumulative target; each epoch contains --steps-per-epoch updates")
    parser.add_argument("--steps-per-epoch", type=int, default=2000)
    parser.add_argument("--batch-reads", type=int, default=256,
                        help="global sampler batch, shared across all ranks")
    parser.add_argument("--microbatch-reads", type=int, default=16,
                        help="per-rank gradient accumulation microbatch; 0 uses whole shard")
    parser.add_argument("--validation-microbatch-reads", type=int, default=16)
    parser.add_argument("--train-fraction", type=float, default=.9)
    parser.add_argument("--validation-max-reads-per-file", type=int, default=5000)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.)
    parser.add_argument("--lr-scheduler", choices=("none", "plateau"), default="plateau")
    parser.add_argument("--lr-factor", type=float, default=.5)
    parser.add_argument("--lr-patience", type=int, default=2)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="single-process data/config/count preflight; never trains")
    parser.add_argument("--preflight-output", type=Path)
    return parser


def resolve_config(args):
    if args.resume is not None:
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        if (not isinstance(payload, dict)
                or payload.get("training_state", {}).get("kind") != TRAINING_KIND):
            raise ValueError("--resume requires a large Transformer DDP checkpoint")
        values = dict(payload["model_config"])
    else:
        values = json.loads(DEFAULT_MODEL_CONFIG.read_text())
    if args.model_config is not None:
        overrides = json.loads(args.model_config.read_text())
        if not isinstance(overrides, dict):
            raise ValueError("--model-config must contain a JSON object")
        values.update(overrides)
    for name in ("d_model", "num_heads", "num_layers", "feedforward_dim",
                 "context_length", "dropout"):
        if getattr(args, name) is not None:
            values[name] = getattr(args, name)
    return DirectQualityModelConfig.from_dict(values)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cpu_threads != 1:
        parser.error("large Transformer training requires --cpu-threads 1")
    if args.resume is not None and args.dry_run:
        parser.error("--dry-run supports fresh initialization only")
    if args.dry_run and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        parser.error("--dry-run is a single-process preflight")
    if args.preflight_output is not None and not args.dry_run:
        parser.error("--preflight-output requires --dry-run")
    if not args.dry_run and args.output_dir is None:
        parser.error("--output-dir is required for training")
    torch.set_num_threads(1)
    try:
        config = resolve_config(args)
        if args.dry_run:
            report = preflight(model_config=config, cache_dir=args.cache_dir,
                train_fraction=args.train_fraction, batch_reads=args.batch_reads,
                seed=args.seed, steps_per_epoch=args.steps_per_epoch,
                validation_max_reads_per_file=args.validation_max_reads_per_file)
            # Meta tensors count the actual model without allocating 2GB of
            # parameter storage or initializing an unallocated GPU.
            with torch.device("meta"):
                model = LargeQualityTransformer(config,
                    attention_backend=args.attention_backend,
                    attention_query_chunk_size=args.attention_query_chunk_size)
            report.update(model_parameters=sum(p.numel() for p in model.parameters()),
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                attention_implementation=model.training_attention_metadata(),
                initialization={"mode": "random", "transformer_layers": "independent_pytorch_default"},
                training_started=False)
            if args.preflight_output is not None:
                if args.preflight_output.exists():
                    raise FileExistsError(args.preflight_output)
                write_json(args.preflight_output, report)
                print(f"preflight: {args.preflight_output}")
            else:
                print(json.dumps(report, indent=2, sort_keys=True))
            print(f"checked {report['corpus']['training_files']} training caches; "
                  f"train_quality_tokens={report['corpus']['train_quality_tokens']}; "
                  f"parameters={report['model_parameters']}; training_started=false")
            return 0
        best = run_distributed_training(model_config=config, output_dir=args.output_dir,
            cache_dir=args.cache_dir, resume=args.resume, epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch, batch_reads=args.batch_reads,
            microbatch_reads=args.microbatch_reads, train_fraction=args.train_fraction,
            validation_max_reads_per_file=args.validation_max_reads_per_file,
            validation_microbatch_reads=args.validation_microbatch_reads,
            learning_rate=args.learning_rate, warmup_steps=args.warmup_steps,
            weight_decay=args.weight_decay, grad_clip=args.grad_clip,
            attention_backend=args.attention_backend,
            attention_query_chunk_size=args.attention_query_chunk_size,
            precision=args.precision, activation_checkpointing=args.activation_checkpointing,
            device=choose_device(args.device), seed=args.seed, progress=not args.no_progress,
            control_config=TrainingControlConfig(lr_scheduler=args.lr_scheduler,
                lr_factor=args.lr_factor, lr_patience=args.lr_patience, min_lr=args.min_lr,
                min_delta=args.min_delta, early_stopping_patience=args.early_stopping_patience))
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"best checkpoint: {best}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
