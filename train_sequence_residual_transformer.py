#!/usr/bin/env python3
"""Train the q-hat-conditioned quality/residual Transformer on SRR*.h5 files."""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

from sequence_residual_transformer_model import (
    DEFAULT_BASE_CONV_KERNELS,
    DEFAULT_HISTORY_RUN_EMBED_DIM,
    DEFAULT_MER_STRIDE,
    DEFAULT_MER_VOCAB_SIZE,
    DEFAULT_PRIOR_FEATURE_MODE,
    DEFAULT_PREDICTION_TARGET,
    DEFAULT_QMER_KS,
    DEFAULT_RMER_KS,
    DIRECT_LOGITS,
    FULL_PRIOR,
    LOG_P0_PLUS_DELTA,
    OUTPUT_PARAMETERIZATIONS,
    PAD_TARGET,
    PREDICTION_TARGETS,
    PRIOR_FEATURE_MODES,
    QUALITY_TARGET,
    RESIDUAL_TARGET,
    ContiguousReadBatchSampler,
    ResidualTransformer,
    base_sidecar_path_for_h5,
    batch_to_torch,
    continuous_feature_dim,
    discover_h5_files,
    inspect_h5,
    inspect_base_sidecar,
    iter_read_batches,
    mask_invalid_residual_logits,
    prediction_output_dim,
    save_json,
)


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def fmt4(value: float) -> str:
    return f"{value:.4f}"


def parse_int_list(text: str) -> tuple[int, ...]:
    """Parse comma-separated positive integers, or an empty string."""

    text = text.strip()
    if not text:
        return tuple()
    values = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    if any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("all values must be positive integers")
    return values


@torch.no_grad()
def evaluate_model(
    model: ResidualTransformer,
    files: list[Path],
    train_fraction: float,
    split: str,
    batch_reads: int,
    max_reads_per_file: int | None,
    device: torch.device,
    qmer_ks: tuple[int, ...],
    rmer_ks: tuple[int, ...],
    mer_stride: int,
    qmer_vocab_size: int,
    rmer_vocab_size: int,
    base_sidecar_dir: Path,
    prior_feature_mode: str,
    prediction_target: str,
    history_run_features: bool,
) -> dict[str, float]:
    """Evaluate compression metrics on a deterministic read split."""

    # CrossEntropyLoss 的单位是 nat；最终压缩指标需要除以 ln(2) 转成 bit。
    # padding 位置 target = PAD_TARGET，不参与 loss。
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TARGET, reduction="sum")
    model.eval()

    total_nats = 0.0
    total_symbols = 0
    baseline_bits = 0.0
    zero_true_freq = 0

    for path in files:
        # 验证/测试按 read 顺序遍历，不随机采样，保证指标稳定可复现。
        for batch in iter_read_batches(
            path=path,
            train_fraction=train_fraction,
            split=split,
            batch_reads=batch_reads,
            max_reads=max_reads_per_file,
            base_sidecar_path=base_sidecar_path_for_h5(path, base_sidecar_dir),
            qmer_ks=qmer_ks,
            rmer_ks=rmer_ks,
            mer_stride=mer_stride,
            qmer_vocab_size=qmer_vocab_size,
            rmer_vocab_size=rmer_vocab_size,
            prior_feature_mode=prior_feature_mode,
            prediction_target=prediction_target,
            history_run_features=history_run_features,
        ):
            tensors = batch_to_torch(batch, device)
            # 模型输出当前实验的 95 个 quality 类别；旧 residual checkpoint
            # 仍可通过 prediction_target=residual 使用 189 类。
            logits = model(
                continuous=tensors["continuous"],
                q_hat=tensors["q_hat"],
                prev_q=tensors["prev_q"],
                prev_r=tensors["prev_r"],
                zero_residual_run=tensors["zero_residual_run"],
                same_quality_run=tensors["same_quality_run"],
                qmer_tokens=tensors["qmer_tokens"],
                rmer_tokens=tensors["rmer_tokens"],
                base_ids=tensors["base_ids"],
                lengths=tensors["lengths"],
            )
            if prediction_target == RESIDUAL_TARGET:
                logits = mask_invalid_residual_logits(
                    logits=logits,
                    q_hat=tensors["q_hat"],
                    valid_mask=tensors["valid_mask"],
                )
            loss = criterion(logits.reshape(-1, logits.shape[-1]), tensors["targets"].reshape(-1))

            # batch.baseline_bits 是同一批位置在原始 H5 P0(q_true) 下的 bit 数。
            total_nats += float(loss.item())
            total_symbols += batch.total_symbols
            baseline_bits += batch.baseline_bits
            zero_true_freq += batch.zero_true_freq

    if total_symbols == 0:
        raise ValueError("evaluation split produced zero symbols")

    # model_avg_bits_per_quality 是最终最重要的压缩评估指标。
    model_total_bits = total_nats / math.log(2.0)
    model_avg_bits = model_total_bits / total_symbols
    h5_avg_bits = baseline_bits / total_symbols
    return {
        "total_symbols": float(total_symbols),
        "model_total_bits": model_total_bits,
        "model_avg_bits_per_quality": model_avg_bits,
        "h5_baseline_total_bits": baseline_bits,
        "h5_baseline_avg_bits_per_quality": h5_avg_bits,
        "delta_bits": model_avg_bits - h5_avg_bits,
        "relative_improvement": (h5_avg_bits - model_avg_bits) / h5_avg_bits,
        "zero_true_freq": float(zero_true_freq),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a causal read-level Transformer that predicts quality distributions "
            "from H5 predictor features and decoded quality/residual history."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        default=[Path("h5")],
        help="HDF5 files or directories containing SRR*.h5 files; default: h5",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "runs/transformer_quality_4layer_qhatonly_qrmer234_"
            "baseconv357_b64_e15"
        ),
        help="directory for checkpoints, config, and train log",
    )
    parser.add_argument(
        "--base-sidecar-dir",
        type=Path,
        default=Path("base_sidecars"),
        help="directory produced by prepare_base_sidecars.py",
    )
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument(
        "--prediction-target",
        choices=PREDICTION_TARGETS,
        default=DEFAULT_PREDICTION_TARGET,
        help="predict the 95-class quality id (default) or historical 189-class residual",
    )
    parser.add_argument(
        "--prior-feature-mode",
        choices=PRIOR_FEATURE_MODES,
        default=DEFAULT_PRIOR_FEATURE_MODE,
        help=(
            "qhat_only uses the H5 matrix only to derive q_hat/residual; "
            "full_prior also feeds log P0_r and probability summaries to the model"
        ),
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--steps-per-epoch", type=int, default=1000)
    parser.add_argument(
        "--batch-reads",
        type=int,
        default=64,
        help="number of contiguous reads sampled per training batch",
    )
    parser.add_argument(
        "--eval-batch-reads",
        type=int,
        default=64,
        help="number of contiguous reads per validation batch",
    )
    parser.add_argument(
        "--eval-max-reads-per-file",
        type=int,
        default=5000,
        help="limit validation reads per file for speed; use 0 for full split",
    )
    parser.add_argument("--q-hat-embed-dim", type=int, default=16)
    parser.add_argument("--prev-q-embed-dim", type=int, default=16)
    parser.add_argument("--prev-r-embed-dim", type=int, default=32)
    parser.add_argument(
        "--history-run-embed-dim",
        type=int,
        default=DEFAULT_HISTORY_RUN_EMBED_DIM,
        help="embedding dimension for causal residual-zero and same-quality run buckets",
    )
    parser.add_argument(
        "--qmer-ks",
        type=parse_int_list,
        default=DEFAULT_QMER_KS,
        help="comma-separated Q-mer history windows; use '' to disable",
    )
    parser.add_argument(
        "--rmer-ks",
        type=parse_int_list,
        default=DEFAULT_RMER_KS,
        help="comma-separated residual-mer history windows; use '' to disable",
    )
    parser.add_argument("--mer-stride", type=int, default=DEFAULT_MER_STRIDE)
    parser.add_argument("--qmer-vocab-size", type=int, default=DEFAULT_MER_VOCAB_SIZE)
    parser.add_argument("--rmer-vocab-size", type=int, default=DEFAULT_MER_VOCAB_SIZE)
    parser.add_argument("--qmer-embed-dim", type=int, default=8)
    parser.add_argument("--rmer-embed-dim", type=int, default=8)
    parser.add_argument("--base-embed-dim", type=int, default=16)
    parser.add_argument(
        "--base-conv-kernels",
        type=parse_int_list,
        default=DEFAULT_BASE_CONV_KERNELS,
        help="comma-separated centered odd base Conv1D kernels",
    )
    parser.add_argument("--base-conv-channels", type=int, default=16)
    parser.add_argument("--base-context-dim", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--feedforward-dim", type=int, default=512)
    parser.add_argument(
        "--context-length",
        type=int,
        default=256,
        help="per-layer causal attention window including the current position",
    )
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--output-parameterization",
        choices=OUTPUT_PARAMETERIZATIONS,
        default=DIRECT_LOGITS,
        help=(
            "use the output head directly, or add its learned correction to "
            "the H5 log P0_r prior"
        ),
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--save-every-epoch", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not (0.0 < args.train_fraction < 1.0):
        raise SystemExit("--train-fraction must be between 0 and 1")
    if args.epochs <= 0 or args.steps_per_epoch <= 0:
        raise SystemExit("--epochs and --steps-per-epoch must be positive")
    if args.batch_reads <= 0 or args.eval_batch_reads <= 0:
        raise SystemExit("--batch-reads and --eval-batch-reads must be positive")
    if args.eval_max_reads_per_file is not None and args.eval_max_reads_per_file < 0:
        raise SystemExit("--eval-max-reads-per-file must be >= 0")
    if args.num_layers <= 0:
        raise SystemExit("--num-layers must be positive")
    if args.d_model <= 0 or args.num_heads <= 0 or args.d_model % args.num_heads != 0:
        raise SystemExit("--d-model must be positive and divisible by --num-heads")
    if args.feedforward_dim <= 0 or args.context_length <= 0:
        raise SystemExit("--feedforward-dim and --context-length must be positive")
    if args.mer_stride <= 0:
        raise SystemExit("--mer-stride must be positive")
    if args.history_run_embed_dim < 0:
        raise SystemExit("--history-run-embed-dim must be non-negative")
    if args.output_parameterization == LOG_P0_PLUS_DELTA and (
        args.prior_feature_mode != FULL_PRIOR
        or args.prediction_target != RESIDUAL_TARGET
    ):
        raise SystemExit(
            "--output-parameterization log_p0_plus_delta requires "
            "full_prior and residual target"
        )
    if args.qmer_vocab_size <= 0 or args.rmer_vocab_size <= 0:
        raise SystemExit("Q/R-mer vocabulary sizes must be positive")
    if args.qmer_embed_dim <= 0 or args.rmer_embed_dim <= 0:
        raise SystemExit("Q/R-mer embedding dimensions must be positive")
    if (
        args.base_embed_dim <= 0
        or args.base_conv_channels <= 0
        or args.base_context_dim <= 0
    ):
        raise SystemExit("base embedding, convolution, and context dimensions must be positive")
    if not args.base_conv_kernels or any(
        kernel <= 0 or kernel % 2 == 0 for kernel in args.base_conv_kernels
    ):
        raise SystemExit("--base-conv-kernels must contain positive odd integers")

    # 固定随机种子，方便比较不同模型/参数的实验结果。
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = choose_device(args.device)

    # 默认会读取 h5/ 下所有 SRR*.h5；也可以手动传入单个 H5 或目录。
    files = discover_h5_files(args.inputs)
    infos = [inspect_h5(path) for path in files]
    for info in infos:
        if info.alphabet_size != 95:
            raise SystemExit(f"{info.path}: expected alphabet size 95, got {info.alphabet_size}")
    base_infos = [
        inspect_base_sidecar(
            path,
            base_sidecar_path_for_h5(path, args.base_sidecar_dir),
        )
        for path in files
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    eval_limit = None if args.eval_max_reads_per_file == 0 else args.eval_max_reads_per_file
    selected_continuous_dim = continuous_feature_dim(args.prior_feature_mode)
    selected_output_dim = prediction_output_dim(args.prediction_target)

    # 保存完整配置，后续 predict 脚本会从 checkpoint 里恢复模型结构。
    if args.prediction_target == QUALITY_TARGET:
        model_type = f"causal_transformer_{args.prior_feature_mode}_direct_quality_base_conv"
    elif args.output_parameterization == LOG_P0_PLUS_DELTA:
        model_type = "causal_transformer_log_p0_delta_residual_base_conv"
    else:
        model_type = f"causal_transformer_{args.prior_feature_mode}_direct_residual_base_conv"
    config = {
        "model_type": model_type,
        "prediction_target": args.prediction_target,
        "target": (
            "quality_id_0_94"
            if args.prediction_target == QUALITY_TARGET
            else "residual_-94_94"
        ),
        "prior_feature_mode": args.prior_feature_mode,
        "output_parameterization": args.output_parameterization,
        "uses_qr_mer": bool(args.qmer_ks or args.rmer_ks),
        "uses_history_runs": args.history_run_embed_dim > 0,
        "uses_q_hat": True,
        "uses_residual_history": True,
        "uses_base_context": True,
        "base_context_is_bidirectional": True,
        "complete_base_read_available_before_quality": True,
        "body_length_is_quality_side_information": True,
        "base_sidecar_dir": str(args.base_sidecar_dir),
        "base_sidecar_files": {
            str(path): str(base_sidecar_path_for_h5(path, args.base_sidecar_dir))
            for path in files
        },
        "file_raw_bases": {
            str(path): base_info.base_count
            for path, base_info in zip(files, base_infos)
        },
        "input_files": [str(path) for path in files],
        "file_rows": {str(info.path): info.rows for info in infos},
        "file_reads": {str(info.path): info.read_count for info in infos},
        "file_empty_reads": {str(info.path): info.empty_read_count for info in infos},
        "read_length_summary": {
            str(info.path): {
                "min": info.min_read_len,
                "max": info.max_read_len,
                "mean": info.mean_read_len,
            }
            for info in infos
        },
        "train_fraction": args.train_fraction,
        "epochs": args.epochs,
        "steps_per_epoch": args.steps_per_epoch,
        "batch_reads": args.batch_reads,
        "eval_batch_reads": args.eval_batch_reads,
        "eval_max_reads_per_file": args.eval_max_reads_per_file,
        "continuous_dim": selected_continuous_dim,
        "q_hat_embed_dim": args.q_hat_embed_dim,
        "prev_q_embed_dim": args.prev_q_embed_dim,
        "prev_r_embed_dim": args.prev_r_embed_dim,
        "history_run_embed_dim": args.history_run_embed_dim,
        "qmer_ks": list(args.qmer_ks),
        "rmer_ks": list(args.rmer_ks),
        "mer_stride": args.mer_stride,
        "qmer_vocab_size": args.qmer_vocab_size,
        "rmer_vocab_size": args.rmer_vocab_size,
        "qmer_embed_dim": args.qmer_embed_dim,
        "rmer_embed_dim": args.rmer_embed_dim,
        "base_embed_dim": args.base_embed_dim,
        "base_conv_kernels": list(args.base_conv_kernels),
        "base_conv_channels": args.base_conv_channels,
        "base_context_dim": args.base_context_dim,
        "d_model": args.d_model,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "feedforward_dim": args.feedforward_dim,
        "context_length": args.context_length,
        "dropout": args.dropout,
        "output_dim": selected_output_dim,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "device": str(device),
        "seed": args.seed,
    }
    # 训练时随机抽取连续 read block，兼顾随机性和 HDF5 顺序读取效率。
    sampler = ContiguousReadBatchSampler(
        files=files,
        train_fraction=args.train_fraction,
        batch_reads=args.batch_reads,
        seed=args.seed,
        base_sidecar_dir=args.base_sidecar_dir,
        qmer_ks=args.qmer_ks,
        rmer_ks=args.rmer_ks,
        mer_stride=args.mer_stride,
        qmer_vocab_size=args.qmer_vocab_size,
        rmer_vocab_size=args.rmer_vocab_size,
        prior_feature_mode=args.prior_feature_mode,
        prediction_target=args.prediction_target,
        history_run_features=args.history_run_embed_dim > 0,
    )
    # Transformer 输入与阶段 3 Q/R-mer 版本使用相同的 causal features。
    model = ResidualTransformer(
        continuous_dim=selected_continuous_dim,
        q_hat_embed_dim=args.q_hat_embed_dim,
        prev_q_embed_dim=args.prev_q_embed_dim,
        prev_r_embed_dim=args.prev_r_embed_dim,
        history_run_embed_dim=args.history_run_embed_dim,
        qmer_ks=args.qmer_ks,
        rmer_ks=args.rmer_ks,
        qmer_vocab_size=args.qmer_vocab_size,
        rmer_vocab_size=args.rmer_vocab_size,
        qmer_embed_dim=args.qmer_embed_dim,
        rmer_embed_dim=args.rmer_embed_dim,
        base_embed_dim=args.base_embed_dim,
        base_conv_kernels=args.base_conv_kernels,
        base_conv_channels=args.base_conv_channels,
        base_context_dim=args.base_context_dim,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        feedforward_dim=args.feedforward_dim,
        context_length=args.context_length,
        dropout=args.dropout,
        output_dim=selected_output_dim,
        output_parameterization=args.output_parameterization,
    ).to(device)
    config["parameter_count"] = sum(parameter.numel() for parameter in model.parameters())
    save_json(args.output_dir / "config.json", config)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TARGET)

    log_path = args.output_dir / "train_log.csv"
    best_bits = float("inf")
    started = time.time()

    with log_path.open("w", encoding="utf-8", newline="") as log_file:
        writer = csv.DictWriter(
            log_file,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_bits_per_quality",
                "train_h5_baseline_bits",
                "val_model_avg_bits_per_quality",
                "val_h5_baseline_avg_bits_per_quality",
                "val_delta_bits",
                "val_relative_improvement",
                "val_total_symbols",
                "val_zero_true_freq",
                "gpu_peak_memory_bytes",
                "elapsed_seconds",
            ],
        )
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            model.train()
            running_nats = 0.0
            running_symbols = 0
            running_baseline_bits = 0.0

            step_iter = range(1, args.steps_per_epoch + 1)
            if tqdm is not None and not args.no_progress:
                step_iter = tqdm(
                    step_iter,
                    total=args.steps_per_epoch,
                    desc=f"epoch {epoch}/{args.epochs}",
                    unit="batch",
                    leave=True,
                )

            for step in step_iter:
                batch, _ = sampler.sample()
                tensors = batch_to_torch(batch, device)

                optimizer.zero_grad(set_to_none=True)
                # causal attention 只看当前及历史位置；prev_q/prev_r 已右移，
                # 因此当前位置输入不包含真实 quality 或 residual。
                logits = model(
                    continuous=tensors["continuous"],
                    q_hat=tensors["q_hat"],
                    prev_q=tensors["prev_q"],
                    prev_r=tensors["prev_r"],
                    zero_residual_run=tensors["zero_residual_run"],
                    same_quality_run=tensors["same_quality_run"],
                    qmer_tokens=tensors["qmer_tokens"],
                    rmer_tokens=tensors["rmer_tokens"],
                    base_ids=tensors["base_ids"],
                    lengths=tensors["lengths"],
                )
                # 直接质量目标的 95 类天然全合法；只有历史 residual 目标
                # 才需要按 q_hat 屏蔽物理非法 residual。
                if args.prediction_target == RESIDUAL_TARGET:
                    logits = mask_invalid_residual_logits(
                        logits=logits,
                        q_hat=tensors["q_hat"],
                        valid_mask=tensors["valid_mask"],
                    )
                # 展平成 [batch*seq_len, classes]；padding 由 ignore_index 跳过。
                loss = criterion(logits.reshape(-1, logits.shape[-1]), tensors["targets"].reshape(-1))
                loss.backward()
                # 梯度裁剪用于抑制训练早期的异常梯度尖峰。
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

                symbols = batch.total_symbols
                running_nats += float(loss.item()) * symbols
                running_symbols += symbols
                running_baseline_bits += batch.baseline_bits

                if tqdm is not None and not args.no_progress and step % 10 == 0:
                    train_bits = (running_nats / max(running_symbols, 1)) / math.log(2.0)
                    step_iter.set_postfix(train_bits=f"{train_bits:.4f}")

            train_avg_bits = (running_nats / running_symbols) / math.log(2.0)
            train_loss = running_nats / running_symbols
            train_baseline_bits = running_baseline_bits / running_symbols

            # 每个 epoch 后与 H5 baseline 比较 bits，而不是只看分类准确率。
            val = evaluate_model(
                model=model,
                files=files,
                train_fraction=args.train_fraction,
                split="test",
                batch_reads=args.eval_batch_reads,
                max_reads_per_file=eval_limit,
                device=device,
                qmer_ks=args.qmer_ks,
                rmer_ks=args.rmer_ks,
                mer_stride=args.mer_stride,
                qmer_vocab_size=args.qmer_vocab_size,
                rmer_vocab_size=args.rmer_vocab_size,
                base_sidecar_dir=args.base_sidecar_dir,
                prior_feature_mode=args.prior_feature_mode,
                prediction_target=args.prediction_target,
                history_run_features=args.history_run_embed_dim > 0,
            )
            elapsed = time.time() - started
            row = {
                "epoch": epoch,
                "train_loss": fmt4(train_loss),
                "train_bits_per_quality": fmt4(train_avg_bits),
                "train_h5_baseline_bits": fmt4(train_baseline_bits),
                "val_model_avg_bits_per_quality": fmt4(val["model_avg_bits_per_quality"]),
                "val_h5_baseline_avg_bits_per_quality": fmt4(val["h5_baseline_avg_bits_per_quality"]),
                "val_delta_bits": fmt4(val["delta_bits"]),
                "val_relative_improvement": fmt4(val["relative_improvement"]),
                "val_total_symbols": int(val["total_symbols"]),
                "val_zero_true_freq": int(val["zero_true_freq"]),
                "gpu_peak_memory_bytes": (
                    int(torch.cuda.max_memory_allocated(device))
                    if device.type == "cuda"
                    else 0
                ),
                "elapsed_seconds": fmt4(elapsed),
            }
            writer.writerow(row)
            log_file.flush()

            print(
                "epoch={epoch} loss={loss:.4f} train_bits={train_bits:.4f} "
                "val_bits={val_bits:.4f} h5_bits={h5_bits:.4f} "
                "delta={delta:.4f} rel_improve={rel:.4%} symbols={symbols}".format(
                    epoch=epoch,
                    loss=train_loss,
                    train_bits=train_avg_bits,
                    val_bits=val["model_avg_bits_per_quality"],
                    h5_bits=val["h5_baseline_avg_bits_per_quality"],
                    delta=val["delta_bits"],
                    rel=val["relative_improvement"],
                    symbols=int(val["total_symbols"]),
                ),
                flush=True,
            )

            # 保存最佳 checkpoint：以验证集 model_avg_bits_per_quality 最低为准。
            checkpoint = {
                "config": config,
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_metrics": val,
            }
            if args.save_every_epoch:
                torch.save(checkpoint, args.output_dir / f"checkpoint_epoch{epoch}.pt")
            if val["model_avg_bits_per_quality"] < best_bits:
                best_bits = val["model_avg_bits_per_quality"]
                torch.save(checkpoint, args.output_dir / "best.pt")

    print(f"wrote {args.output_dir / 'best.pt'}")
    print(f"wrote {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
