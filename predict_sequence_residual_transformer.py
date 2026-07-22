#!/usr/bin/env python3
"""Evaluate or sample predictions from a trained stage-4 Transformer model."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from sequence_residual_transformer_model import (
    ALPHABET_SIZE,
    PAD_TARGET,
    RESIDUAL_CLASSES,
    RESIDUAL_MIN,
    batch_to_torch,
    discover_h5_files,
    inspect_h5,
    iter_read_batches,
    load_checkpoint,
    mask_invalid_residual_logits,
)


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def fmt6(value: float) -> str:
    return f"{value:.6f}"


@torch.no_grad()
def evaluate_file(
    model: nn.Module,
    path: Path,
    train_fraction: float,
    split: str,
    batch_reads: int,
    max_reads: int | None,
    device: torch.device,
) -> dict[str, float | int | str]:
    # 预测阶段同样按压缩目标评估：真实 residual 在模型分布下的 -log2 概率。
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TARGET, reduction="sum")
    total_nats = 0.0
    total_symbols = 0
    baseline_bits = 0.0
    zero_true_freq = 0

    model.eval()
    for batch in iter_read_batches(
        path=path,
        train_fraction=train_fraction,
        split=split,
        batch_reads=batch_reads,
        max_reads=max_reads,
    ):
        tensors = batch_to_torch(batch, device)
        # 使用 teacher-forced 的 prev_q/prev_r 评估概率。
        # 对无损解码来说，前一位已经恢复，因此 encoder/decoder 也能得到同样历史。
        logits = model(
            continuous=tensors["continuous"],
            q_hat=tensors["q_hat"],
            prev_q=tensors["prev_q"],
            prev_r=tensors["prev_r"],
            lengths=tensors["lengths"],
        )
        # 预测/评估时也必须做同样的非法 residual mask，保持训练和解码一致。
        logits = mask_invalid_residual_logits(
            logits=logits,
            q_hat=tensors["q_hat"],
            valid_mask=tensors["valid_mask"],
        )
        loss = criterion(logits.reshape(-1, logits.shape[-1]), tensors["targets"].reshape(-1))

        total_nats += float(loss.item())
        total_symbols += batch.total_symbols
        baseline_bits += batch.baseline_bits
        zero_true_freq += batch.zero_true_freq

    if total_symbols == 0:
        raise ValueError(f"{path}: evaluation split produced zero symbols")

    # total_nats 转成 bit 后，再除以 symbol 数得到 avg_bits_per_quality。
    model_total_bits = total_nats / math.log(2.0)
    model_avg_bits = model_total_bits / total_symbols
    h5_avg_bits = baseline_bits / total_symbols
    return {
        "file": str(path),
        "total_symbols": total_symbols,
        "model_total_bits": model_total_bits,
        "model_avg_bits_per_quality": model_avg_bits,
        "h5_baseline_total_bits": baseline_bits,
        "h5_baseline_avg_bits_per_quality": h5_avg_bits,
        "delta_bits": model_avg_bits - h5_avg_bits,
        "relative_improvement": (h5_avg_bits - model_avg_bits) / h5_avg_bits,
        "zero_true_freq": zero_true_freq,
    }


@torch.no_grad()
def write_prediction_samples(
    model: nn.Module,
    path: Path,
    output_csv: Path,
    train_fraction: float,
    split: str,
    sample_reads: int,
    device: torch.device,
) -> None:
    """Write position-level examples from the first sample_reads reads."""

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    # 这个 CSV 是调试用的详细样例，包含真实值、预测值、模型概率和 H5 baseline。
    first_batch = next(
        iter_read_batches(
            path=path,
            train_fraction=train_fraction,
            split=split,
            batch_reads=sample_reads,
            max_reads=sample_reads,
        )
    )
    tensors = batch_to_torch(first_batch, device)
    logits = model(
        continuous=tensors["continuous"],
        q_hat=tensors["q_hat"],
        prev_q=tensors["prev_q"],
        prev_r=tensors["prev_r"],
        lengths=tensors["lengths"],
    )
    logits = mask_invalid_residual_logits(
        logits=logits,
        q_hat=tensors["q_hat"],
        valid_mask=tensors["valid_mask"],
    )
    probs = torch.softmax(logits, dim=-1).cpu().numpy()

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "read_in_sample",
                "position_in_read",
                "observed_id",
                "observed_char",
                "q_hat",
                "true_residual",
                "pred_residual",
                "pred_quality",
                "pred_quality_char",
                "pred_probability",
                "model_true_prob",
                "model_bits",
                "h5_true_prob",
                "h5_bits",
            ],
        )
        writer.writeheader()

        for read_idx, length in enumerate(first_batch.lengths):
            for pos in range(int(length)):
                target = int(first_batch.targets[read_idx, pos])
                true_residual = target + RESIDUAL_MIN
                q_hat = int(first_batch.q_hat[read_idx, pos])
                observed = q_hat + true_residual

                pred_class = int(np.argmax(probs[read_idx, pos]))
                pred_residual = pred_class + RESIDUAL_MIN
                pred_quality = int(np.clip(q_hat + pred_residual, 0, ALPHABET_SIZE - 1))
                pred_prob = float(probs[read_idx, pos, pred_class])
                model_true_prob = float(probs[read_idx, pos, target])

                # continuous 的前 189 维就是 log P0_r(r)，target 位置对应
                # H5 对真实质量值 q_true 的原始概率。
                h5_log_prob = float(first_batch.continuous[read_idx, pos, target])
                h5_true_prob = math.exp(h5_log_prob)

                writer.writerow(
                    {
                        "read_in_sample": read_idx,
                        "position_in_read": pos,
                        "observed_id": observed,
                        "observed_char": chr(observed + 33),
                        "q_hat": q_hat,
                        "true_residual": true_residual,
                        "pred_residual": pred_residual,
                        "pred_quality": pred_quality,
                        "pred_quality_char": chr(pred_quality + 33),
                        "pred_probability": fmt6(pred_prob),
                        "model_true_prob": fmt6(model_true_prob),
                        "model_bits": fmt6(-math.log2(max(model_true_prob, 1e-300))),
                        "h5_true_prob": fmt6(h5_true_prob),
                        "h5_bits": fmt6(-h5_log_prob / math.log(2.0)),
                    }
                )


@torch.no_grad()
def write_quality_probability_log(
    model: nn.Module,
    files: list[Path],
    output_log: Path,
    train_fraction: float,
    split: str,
    batch_reads: int,
    rows_to_write: int,
    device: torch.device,
) -> int:
    """Write compact predicted quality/probability rows.

    Output format per line:

    {pred_quality_value}{pred_quality_char}{predicted_probability}{true_quality_value}{true_quality_char}

    这里的 predicted_probability 是模型 argmax residual 对应的概率，也就是
    模型最想预测的质量值概率；后两列给出该位置的真实质量值，方便对照。
    """

    output_log.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    model.eval()

    with output_log.open("w", encoding="utf-8", newline="\n") as out:
        for path in files:
            if rows_written >= rows_to_write:
                break

            # 为了得到“预测部分结果日志”，这里从指定 split 的开头顺序取样。
            # 默认 rows_to_write=1000，避免日志过大。
            for batch in iter_read_batches(
                path=path,
                train_fraction=train_fraction,
                split=split,
                batch_reads=batch_reads,
                max_reads=None,
            ):
                tensors = batch_to_torch(batch, device)
                logits = model(
                    continuous=tensors["continuous"],
                    q_hat=tensors["q_hat"],
                    prev_q=tensors["prev_q"],
                    prev_r=tensors["prev_r"],
                    lengths=tensors["lengths"],
                )
                logits = mask_invalid_residual_logits(
                    logits=logits,
                    q_hat=tensors["q_hat"],
                    valid_mask=tensors["valid_mask"],
                )
                probs = torch.softmax(logits, dim=-1).cpu().numpy()

                for read_idx, length in enumerate(batch.lengths):
                    for pos in range(int(length)):
                        pred_class = int(np.argmax(probs[read_idx, pos]))
                        pred_residual = pred_class + RESIDUAL_MIN
                        q_hat = int(batch.q_hat[read_idx, pos])
                        quality_value = int(np.clip(q_hat + pred_residual, 0, ALPHABET_SIZE - 1))
                        quality_char = chr(quality_value + 33)
                        pred_prob = float(probs[read_idx, pos, pred_class])
                        true_class = int(batch.targets[read_idx, pos])
                        true_residual = true_class + RESIDUAL_MIN
                        true_quality_value = int(q_hat + true_residual)
                        true_quality_char = chr(true_quality_value + 33)

                        out.write(
                            f"{{{quality_value}}}{{{quality_char}}}{{{pred_prob:.6f}}}"
                            f"{{{true_quality_value}}}{{{true_quality_char}}}\n"
                        )
                        rows_written += 1
                        if rows_written >= rows_to_write:
                            return rows_written

    return rows_written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained causal Transformer residual model on H5 predictor files."
    )
    parser.add_argument("checkpoint", type=Path, help="path to best.pt")
    parser.add_argument(
        "inputs",
        nargs="*",
        default=[Path("h5")],
        help="HDF5 files or directories containing SRR*.h5 files; default: h5",
    )
    parser.add_argument("--split", choices=["test", "train", "all"], default="test")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--batch-reads", type=int, default=128)
    parser.add_argument("--max-reads-per-file", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("runs/transformer_residual/predict_metrics.csv"),
    )
    parser.add_argument(
        "--sample-predictions",
        type=Path,
        default=None,
        help="optional CSV path for position-level prediction examples",
    )
    parser.add_argument("--sample-reads", type=int, default=5)
    parser.add_argument(
        "--quality-prob-log",
        type=Path,
        default=None,
        help=(
            "optional compact prediction log path. When provided, writes rows formatted as "
            "{pred_quality_value}{pred_quality_char}{predicted_probability}"
            "{true_quality_value}{true_quality_char}."
        ),
    )
    parser.add_argument(
        "--quality-prob-log-rows",
        type=int,
        default=1000,
        help="number of compact prediction rows to write; default: 1000",
    )
    parser.add_argument(
        "--no-quality-prob-log",
        action="store_true",
        help="kept for backward compatibility; compact log is disabled unless --quality-prob-log is provided",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.batch_reads <= 0:
        raise SystemExit("--batch-reads must be positive")
    if args.max_reads_per_file is not None and args.max_reads_per_file <= 0:
        raise SystemExit("--max-reads-per-file must be positive")
    if args.sample_reads <= 0:
        raise SystemExit("--sample-reads must be positive")
    if args.quality_prob_log_rows <= 0:
        raise SystemExit("--quality-prob-log-rows must be positive")

    device = choose_device(args.device)
    files = discover_h5_files(args.inputs)
    for info in [inspect_h5(path) for path in files]:
        if info.alphabet_size != 95:
            raise SystemExit(f"{info.path}: expected alphabet size 95, got {info.alphabet_size}")

    # checkpoint 里保存了模型结构参数，所以预测时只需要传 best.pt。
    model, _ = load_checkpoint(args.checkpoint, device)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | int | str]] = []
    for path in files:
        # 逐文件评估，方便看模型是否在不同 SRR 数据上泛化。
        metric = evaluate_file(
            model=model,
            path=path,
            train_fraction=args.train_fraction,
            split=args.split,
            batch_reads=args.batch_reads,
            max_reads=args.max_reads_per_file,
            device=device,
        )
        rows.append(metric)
        print(
            f"{path.name}: model_bits={metric['model_avg_bits_per_quality']:.6f} "
            f"h5_bits={metric['h5_baseline_avg_bits_per_quality']:.6f} "
            f"delta={metric['delta_bits']:.6f} "
            f"rel_improve={metric['relative_improvement']:.4%} "
            f"symbols={metric['total_symbols']}",
            flush=True,
        )

    # 汇总 CSV 是正式指标文件，重点看 model_avg_bits_per_quality 是否低于 H5 baseline。
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "file",
            "total_symbols",
            "model_total_bits",
            "model_avg_bits_per_quality",
            "h5_baseline_total_bits",
            "h5_baseline_avg_bits_per_quality",
            "delta_bits",
            "relative_improvement",
            "zero_true_freq",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "file": row["file"],
                    "total_symbols": row["total_symbols"],
                    "model_total_bits": fmt6(float(row["model_total_bits"])),
                    "model_avg_bits_per_quality": fmt6(float(row["model_avg_bits_per_quality"])),
                    "h5_baseline_total_bits": fmt6(float(row["h5_baseline_total_bits"])),
                    "h5_baseline_avg_bits_per_quality": fmt6(
                        float(row["h5_baseline_avg_bits_per_quality"])
                    ),
                    "delta_bits": fmt6(float(row["delta_bits"])),
                    "relative_improvement": fmt6(float(row["relative_improvement"])),
                    "zero_true_freq": row["zero_true_freq"],
                }
            )

    if args.sample_predictions is not None:
        write_prediction_samples(
            model=model,
            path=files[0],
            output_csv=args.sample_predictions,
            train_fraction=args.train_fraction,
            split=args.split,
            sample_reads=args.sample_reads,
            device=device,
        )

    quality_log_rows = 0
    if args.quality_prob_log is not None and not args.no_quality_prob_log:
        quality_log_rows = write_quality_probability_log(
            model=model,
            files=files,
            output_log=args.quality_prob_log,
            train_fraction=args.train_fraction,
            split=args.split,
            batch_reads=args.batch_reads,
            rows_to_write=args.quality_prob_log_rows,
            device=device,
        )

    print(f"wrote {args.output_csv}")
    if args.sample_predictions is not None:
        print(f"wrote {args.sample_predictions}")
    if args.quality_prob_log is not None and not args.no_quality_prob_log:
        print(f"wrote {args.quality_prob_log} rows={quality_log_rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
