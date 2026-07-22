#!/usr/bin/env python3
"""Read-level causal Transformer residual model for FASTQ quality compression.

This module implements the shared pieces for stage 4:

* read-level HDF5 discovery and inspection;
* conversion from H5 quality probabilities P0(q) to residual probabilities
  P0_r(r), where r = q_true - q_hat;
* causal history tokens for previous quality and previous residual;
* padding/collation for variable-length reads;
* a lightweight causal Transformer that predicts P(r_i | current H5 features,
  current position, previous decoded quality, previous decoded residual).

The code intentionally does not read FASTQ sequence files. At this stage all
features are derived from the H5 predictor output and the already decoded
quality/residual history within each read.
"""

from __future__ import annotations

import json
import math
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import h5py
import numpy as np
import torch
from torch import nn


ALPHABET_SIZE = 95
RESIDUAL_MIN = -(ALPHABET_SIZE - 1)
RESIDUAL_MAX = ALPHABET_SIZE - 1
RESIDUAL_CLASSES = RESIDUAL_MAX - RESIDUAL_MIN + 1

Q_BOS_TOKEN = ALPHABET_SIZE
Q_TOKEN_COUNT = ALPHABET_SIZE + 1
R_BOS_TOKEN = RESIDUAL_CLASSES
R_TOKEN_COUNT = RESIDUAL_CLASSES + 1

DEFAULT_MER_KS = (2, 3, 4, 6, 8)
DEFAULT_MER_STRIDE = 1
DEFAULT_MER_VOCAB_SIZE = 4096

# Quality buckets: 0-9, 10-19, 20-24, 25-29, 30-34, 35-39, 40+.
Q_BUCKET_COUNT = 7
Q_MER_BOS_BUCKET = Q_BUCKET_COUNT
Q_MER_BASE = Q_BUCKET_COUNT + 1

# Residual buckets: <-4, -4, -3, -2, -1, 0, 1, 2, 3, 4, >4.
R_BUCKET_COUNT = 11
R_MER_BOS_BUCKET = R_BUCKET_COUNT
R_MER_BASE = R_BUCKET_COUNT + 1

# 质量值 alphabet 固定为 95，对应 Phred+33 后的 quality id: 0..94。
# residual = q_true - q_hat，因此 residual 范围是 -94..94，共 189 类。

# 连续输入特征包括：
#   189 维 log P0_r(r)：把 H5 原始质量分布 P0(q) 平移到 residual 空间；
#   max_prob：原始 H5 predictor 对 q_hat 的置信度；
#   entropy：P0(q) 的归一化熵，越大越接近均匀分布；
#   expected_q：P0(q) 下的质量值期望，归一化到 0..1；
#   rel_pos/read_len_norm：read 内位置和 read 长度信息。
CONTINUOUS_FEATURE_DIM = RESIDUAL_CLASSES + 5
PAD_TARGET = -100
EPS = 1e-12


@dataclass(frozen=True)
class H5SequenceInfo:
    path: Path
    rows: int
    alphabet_size: int
    read_count: int
    empty_read_count: int
    min_read_len: int
    max_read_len: int
    mean_read_len: float


@dataclass
class SequenceBatch:
    """A padded read-level batch.

    continuous:
        Float tensor input, shape [batch, max_len, CONTINUOUS_FEATURE_DIM].
    q_hat:
        Current H5 argmax quality token, shape [batch, max_len].
    prev_q:
        Previous decoded quality token. Position 0 uses Q_BOS_TOKEN.
    prev_r:
        Previous decoded residual-class token. Position 0 uses R_BOS_TOKEN.
    targets:
        True residual class for the current position, or PAD_TARGET on padding.
    valid_mask:
        True at real quality positions and false on padding.
    lengths:
        Original read lengths before padding.
    qmer_tokens / rmer_tokens:
        Causal Q/R-mer history tokens, shape [batch, max_len, num_windows].
    baseline_bits:
        Sum of -log2 P0(q_true) over real positions in this batch.
    zero_true_freq:
        Number of real positions where H5 assigned zero count to q_true.
    """

    continuous: np.ndarray
    q_hat: np.ndarray
    prev_q: np.ndarray
    prev_r: np.ndarray
    targets: np.ndarray
    valid_mask: np.ndarray
    lengths: np.ndarray
    qmer_tokens: np.ndarray
    rmer_tokens: np.ndarray
    baseline_bits: float
    zero_true_freq: int

    @property
    def total_symbols(self) -> int:
        return int(self.valid_mask.sum())


def discover_h5_files(paths: Iterable[str | Path]) -> list[Path]:
    """Return unique SRR*.h5 files from explicit files or directories."""

    files: list[Path] = []
    for item in paths:
        path = Path(item)
        # 如果输入是目录，只取 SRR 开头的 H5，避免误读其它中间文件。
        if path.is_dir():
            files.extend(sorted(path.glob("SRR*.h5")))
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(path)

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in files:
        resolved = path.resolve()
        if resolved not in seen:
            unique.append(path)
            seen.add(resolved)
    if not unique:
        raise ValueError("no HDF5 files found")
    return unique


def inspect_h5(path: Path) -> H5SequenceInfo:
    """Validate one H5 file and collect read-level shape metadata."""

    with h5py.File(path, "r") as handle:
        # 序列模型必须使用 read_offsets，按 read 组织独立上下文。
        # /observed 是真实 quality id，/freqs 是 H5 predictor 的频数表。
        for name in ("/observed", "/freqs", "/read_offsets"):
            if name not in handle:
                raise ValueError(f"{path}: missing required dataset {name}")

        observed = handle["/observed"]
        freqs = handle["/freqs"]
        offsets = np.asarray(handle["/read_offsets"][:], dtype=np.int64)

        if len(observed.shape) != 1 or len(freqs.shape) != 2:
            raise ValueError(f"{path}: unexpected /observed or /freqs rank")
        if observed.shape[0] != freqs.shape[0]:
            raise ValueError(f"{path}: /observed and /freqs row count mismatch")
        if offsets.ndim != 1 or offsets.size < 2:
            raise ValueError(f"{path}: /read_offsets must be a 1-D boundary array")
        if int(offsets[0]) != 0 or int(offsets[-1]) != int(observed.shape[0]):
            raise ValueError(f"{path}: /read_offsets does not match row count")

        # offsets 是 read 边界：[0, len(read0), len(read0)+len(read1), ...]
        read_lengths = np.diff(offsets)
        if np.any(read_lengths < 0):
            raise ValueError(f"{path}: /read_offsets must be non-decreasing")
        # 有些真实 H5 会包含空 read，也就是相邻两个 offset 相同。
        # 空 read 没有任何质量字符，不能提供训练样本；这里统计它们，
        # 但不再报错。真正构造 batch 时会自动跳过这些空 read。
        nonempty_lengths = read_lengths[read_lengths > 0]
        if nonempty_lengths.size == 0:
            raise ValueError(f"{path}: all reads are empty")

        return H5SequenceInfo(
            path=path,
            rows=int(observed.shape[0]),
            alphabet_size=int(freqs.shape[1]),
            read_count=int(offsets.size - 1),
            empty_read_count=int(np.count_nonzero(read_lengths == 0)),
            min_read_len=int(nonempty_lengths.min()),
            max_read_len=int(nonempty_lengths.max()),
            mean_read_len=float(nonempty_lengths.mean()),
        )


def split_reads(read_count: int, train_fraction: float) -> tuple[int, int]:
    """Split by read id, not by flat quality-position row."""

    # 按 read 切分可以避免同一条 read 的前半段在训练、后半段在测试，
    # 这样评估更符合 sequence model 的泛化目标。
    train_reads = max(1, min(read_count - 1, int(read_count * train_fraction)))
    return train_reads, read_count - train_reads


def _quality_probs(freqs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # H5 里存的是频数，不一定已经归一化；这里转成概率 P0(q)。
    freqs_f = freqs.astype(np.float32, copy=False)
    row_sum = freqs_f.sum(axis=1, keepdims=True)
    probs = freqs_f / np.maximum(row_sum, EPS)
    log_probs = np.log(np.maximum(probs, EPS))
    return freqs_f, probs, log_probs


def _residual_log_probs(probs: np.ndarray, q_hat: np.ndarray) -> np.ndarray:
    """Map P0(q) to residual-class order for each row.

    For a fixed row and residual r, q = q_hat + r. If q is outside [0, 94],
    the residual is physically impossible and receives only EPS as an input
    feature. The output logits are masked separately before the loss/softmax.
    """

    # 对每个 residual r，找到它对应的质量值 q = q_hat + r。
    # 合法 q 落在 0..94；非法 residual 只是占位，后面 logits 会再 mask。
    residual_values = np.arange(RESIDUAL_MIN, RESIDUAL_MAX + 1, dtype=np.int64)
    q_index = q_hat[:, None] + residual_values[None, :]
    valid = (q_index >= 0) & (q_index < ALPHABET_SIZE)
    clipped = np.clip(q_index, 0, ALPHABET_SIZE - 1)
    gathered = np.take_along_axis(probs, clipped, axis=1)
    gathered = np.where(valid, gathered, EPS)
    return np.log(np.maximum(gathered, EPS)).astype(np.float32, copy=False)


def normalize_mer_ks(values: Iterable[int] | None) -> tuple[int, ...]:
    """Normalize Q/R-mer window lengths."""

    if values is None:
        return tuple()
    return tuple(int(value) for value in values if int(value) > 0)


def quality_to_bucket(q: np.ndarray) -> np.ndarray:
    """Map quality ids 0..94 to seven coarse history buckets."""

    q_i = q.astype(np.int64, copy=False)
    buckets = np.empty_like(q_i, dtype=np.int64)
    buckets[q_i <= 9] = 0
    buckets[(q_i >= 10) & (q_i <= 19)] = 1
    buckets[(q_i >= 20) & (q_i <= 24)] = 2
    buckets[(q_i >= 25) & (q_i <= 29)] = 3
    buckets[(q_i >= 30) & (q_i <= 34)] = 4
    buckets[(q_i >= 35) & (q_i <= 39)] = 5
    buckets[q_i >= 40] = 6
    return buckets


def residual_to_bucket(residual: np.ndarray) -> np.ndarray:
    """Map raw residual values to eleven coarse history buckets."""

    r_i = residual.astype(np.int64, copy=False)
    buckets = np.empty_like(r_i, dtype=np.int64)
    buckets[r_i < -4] = 0
    buckets[r_i == -4] = 1
    buckets[r_i == -3] = 2
    buckets[r_i == -2] = 3
    buckets[r_i == -1] = 4
    buckets[r_i == 0] = 5
    buckets[r_i == 1] = 6
    buckets[r_i == 2] = 7
    buckets[r_i == 3] = 8
    buckets[r_i == 4] = 9
    buckets[r_i > 4] = 10
    return buckets


def build_mer_tokens_for_read(
    buckets: np.ndarray,
    ks: tuple[int, ...],
    stride: int,
    base: int,
    bos_bucket: int,
    vocab_size: int,
) -> np.ndarray:
    """Hash causal history windows into fixed-size token vocabularies."""

    length = int(buckets.shape[0])
    if not ks:
        return np.zeros((length, 0), dtype=np.int64)
    if stride <= 0:
        raise ValueError("mer stride must be positive")
    if vocab_size <= 0:
        raise ValueError("mer vocab size must be positive")

    tokens = np.zeros((length, len(ks)), dtype=np.int64)
    for pos in range(length):
        for mer_idx, k in enumerate(ks):
            code = 0
            for distance in range(k, 0, -1):
                hist_pos = pos - distance * stride
                bucket = int(buckets[hist_pos]) if hist_pos >= 0 else bos_bucket
                code = (code * base + bucket) % vocab_size
            tokens[pos, mer_idx] = code
    return tokens


def build_sequence_batch(
    freqs: np.ndarray,
    observed: np.ndarray,
    local_offsets: np.ndarray,
    qmer_ks: Iterable[int] | None = DEFAULT_MER_KS,
    rmer_ks: Iterable[int] | None = DEFAULT_MER_KS,
    mer_stride: int = DEFAULT_MER_STRIDE,
    qmer_vocab_size: int = DEFAULT_MER_VOCAB_SIZE,
    rmer_vocab_size: int = DEFAULT_MER_VOCAB_SIZE,
) -> SequenceBatch:
    """Build one padded batch from a contiguous group of reads.

    local_offsets contains read boundaries relative to freqs[0]. For example,
    if the batch has three reads, local_offsets has four entries:
    [0, len(read0), len(read0)+len(read1), total_rows].
    """

    if freqs.ndim != 2 or freqs.shape[1] != ALPHABET_SIZE:
        raise ValueError(f"expected freqs shape [N, {ALPHABET_SIZE}], got {freqs.shape}")
    if observed.ndim != 1 or observed.shape[0] != freqs.shape[0]:
        raise ValueError("observed must be a 1-D array with the same row count as freqs")
    if local_offsets.ndim != 1 or local_offsets.size < 2:
        raise ValueError("local_offsets must contain at least one read")

    # 真实数据里可能存在空 read：read_offsets 中相邻边界相同。
    # 空 read 没有 quality 字符，既没有 loss，也没有可用历史上下文。
    # 因此 batch 内只保留长度 > 0 的 read；freqs/observed 本身不需要改，
    # 因为空 read 对应 0 行数据。
    raw_read_lengths = np.diff(local_offsets).astype(np.int64)
    if np.any(raw_read_lengths < 0):
        raise ValueError("local_offsets must be non-decreasing")
    nonempty_read_lengths = raw_read_lengths[raw_read_lengths > 0]
    if nonempty_read_lengths.size == 0:
        raise ValueError("batch contains no non-empty reads")
    if nonempty_read_lengths.size != raw_read_lengths.size:
        local_offsets = np.concatenate(
            [
                np.asarray([0], dtype=np.int64),
                np.cumsum(nonempty_read_lengths, dtype=np.int64),
            ]
        )

    qmer_ks_t = normalize_mer_ks(qmer_ks)
    rmer_ks_t = normalize_mer_ks(rmer_ks)

    # 1. 从 H5 freqs 得到原始 predictor 概率分布 P0(q)。
    freqs_f, probs, log_probs = _quality_probs(freqs)
    observed_i = observed.astype(np.int64, copy=False)
    if np.any(observed_i < 0) or np.any(observed_i >= ALPHABET_SIZE):
        raise ValueError("observed quality id out of [0, 94]")

    # 2. q_hat 是 H5 predictor 的中心预测，即 P0(q) 的 argmax。
    #    我们不直接预测 quality，而是预测 q_true - q_hat。
    q_axis = np.arange(ALPHABET_SIZE, dtype=np.float32)
    q_hat = np.argmax(freqs_f, axis=1).astype(np.int64)
    max_prob = probs[np.arange(probs.shape[0]), q_hat].astype(np.float32)
    entropy = (-(probs * log_probs).sum(axis=1) / math.log(ALPHABET_SIZE)).astype(np.float32)
    expected_q = ((probs * q_axis[None, :]).sum(axis=1) / (ALPHABET_SIZE - 1)).astype(np.float32)

    # 3. 训练目标：真实 residual 类别。
    residual = observed_i - q_hat
    targets_flat = (residual - RESIDUAL_MIN).astype(np.int64)
    if np.any(targets_flat < 0) or np.any(targets_flat >= RESIDUAL_CLASSES):
        raise ValueError("residual target out of range")

    # 4. 把 P0(q) 变换到 residual 空间，作为当前位点的强 baseline 特征。
    log_p0_r = _residual_log_probs(probs, q_hat)
    read_lengths = np.diff(local_offsets).astype(np.int64)
    batch_size = int(read_lengths.size)
    max_len = int(read_lengths.max())

    # 5. 不同 read 长度不同，需要 padding 到 batch 内最长 read。
    #    padding 位置的 target 使用 PAD_TARGET，loss 会 ignore。
    continuous = np.zeros((batch_size, max_len, CONTINUOUS_FEATURE_DIM), dtype=np.float32)
    q_hat_tokens = np.zeros((batch_size, max_len), dtype=np.int64)
    prev_q_tokens = np.full((batch_size, max_len), Q_BOS_TOKEN, dtype=np.int64)
    prev_r_tokens = np.full((batch_size, max_len), R_BOS_TOKEN, dtype=np.int64)
    qmer_tokens = np.zeros((batch_size, max_len, len(qmer_ks_t)), dtype=np.int64)
    rmer_tokens = np.zeros((batch_size, max_len, len(rmer_ks_t)), dtype=np.int64)
    targets = np.full((batch_size, max_len), PAD_TARGET, dtype=np.int64)
    valid_mask = np.zeros((batch_size, max_len), dtype=bool)

    for read_idx, (start, end) in enumerate(zip(local_offsets[:-1], local_offsets[1:])):
        start_i = int(start)
        end_i = int(end)
        length = end_i - start_i

        # 位置特征只依赖 read 边界，不依赖当前/未来真实质量值，因此不会泄露信息。
        # denominator 写成 max(length - 1, 1)，保证 length=1 时也合法。
        rel_pos = np.arange(length, dtype=np.float32) / max(length - 1, 1)
        read_len_norm = np.full(length, min(length, 10_000) / 10_000.0, dtype=np.float32)

        continuous[read_idx, :length, :RESIDUAL_CLASSES] = log_p0_r[start_i:end_i]
        continuous[read_idx, :length, RESIDUAL_CLASSES] = max_prob[start_i:end_i]
        continuous[read_idx, :length, RESIDUAL_CLASSES + 1] = entropy[start_i:end_i]
        continuous[read_idx, :length, RESIDUAL_CLASSES + 2] = expected_q[start_i:end_i]
        continuous[read_idx, :length, RESIDUAL_CLASSES + 3] = rel_pos
        continuous[read_idx, :length, RESIDUAL_CLASSES + 4] = read_len_norm

        q_hat_tokens[read_idx, :length] = q_hat[start_i:end_i]
        targets[read_idx, :length] = targets_flat[start_i:end_i]
        valid_mask[read_idx, :length] = True

        # Tokens use only positions before the current one. Missing history at
        # the read start is represented by a dedicated BOS bucket.
        read_q_buckets = quality_to_bucket(observed_i[start_i:end_i])
        read_r_buckets = residual_to_bucket(residual[start_i:end_i])
        qmer_tokens[read_idx, :length, :] = build_mer_tokens_for_read(
            buckets=read_q_buckets,
            ks=qmer_ks_t,
            stride=mer_stride,
            base=Q_MER_BASE,
            bos_bucket=Q_MER_BOS_BUCKET,
            vocab_size=qmer_vocab_size,
        )
        rmer_tokens[read_idx, :length, :] = build_mer_tokens_for_read(
            buckets=read_r_buckets,
            ks=rmer_ks_t,
            stride=mer_stride,
            base=R_MER_BASE,
            bos_bucket=R_MER_BOS_BUCKET,
            vocab_size=rmer_vocab_size,
        )

        # teacher forcing 历史特征：
        #   训练时 prev_q/prev_r 使用真实的前一位；
        #   真正解码时，前一位已经被熵解码恢复，所以 encoder/decoder 也能一致得到。
        # 注意：当前位置 i 的输入不包含 q_true_i 或 r_i 本身。
        prev_q_tokens[read_idx, 0] = Q_BOS_TOKEN
        prev_r_tokens[read_idx, 0] = R_BOS_TOKEN
        if length > 1:
            prev_q_tokens[read_idx, 1:length] = observed_i[start_i : end_i - 1]
            prev_r_tokens[read_idx, 1:length] = targets_flat[start_i : end_i - 1]

    # H5 baseline bits 用于和神经模型 bits 对比：
    #   bits_i = -log2 P0(q_true_i)
    true_freq = freqs_f[np.arange(freqs_f.shape[0]), observed_i]
    row_sum = freqs_f.sum(axis=1)
    true_prob = true_freq / np.maximum(row_sum, EPS)
    baseline_bits = float((-np.log2(np.maximum(true_prob, EPS))).sum())
    zero_true_freq = int(np.count_nonzero(true_freq <= 0))

    return SequenceBatch(
        continuous=continuous,
        q_hat=q_hat_tokens,
        prev_q=prev_q_tokens,
        prev_r=prev_r_tokens,
        targets=targets,
        valid_mask=valid_mask,
        lengths=read_lengths,
        qmer_tokens=qmer_tokens,
        rmer_tokens=rmer_tokens,
        baseline_bits=baseline_bits,
        zero_true_freq=zero_true_freq,
    )


class ContiguousReadBatchSampler:
    """Randomly sample contiguous read blocks from the training split.

    A contiguous read block is much cheaper to fetch from HDF5 than many random
    single reads, while still giving stochastic training batches.
    """

    def __init__(
        self,
        files: list[Path],
        train_fraction: float,
        batch_reads: int,
        seed: int,
        qmer_ks: Iterable[int] | None = DEFAULT_MER_KS,
        rmer_ks: Iterable[int] | None = DEFAULT_MER_KS,
        mer_stride: int = DEFAULT_MER_STRIDE,
        qmer_vocab_size: int = DEFAULT_MER_VOCAB_SIZE,
        rmer_vocab_size: int = DEFAULT_MER_VOCAB_SIZE,
    ) -> None:
        self.files = files
        self.train_fraction = train_fraction
        self.batch_reads = batch_reads
        self.rng = random.Random(seed)
        self.qmer_ks = normalize_mer_ks(qmer_ks)
        self.rmer_ks = normalize_mer_ks(rmer_ks)
        self.mer_stride = int(mer_stride)
        self.qmer_vocab_size = int(qmer_vocab_size)
        self.rmer_vocab_size = int(rmer_vocab_size)
        self.infos = [inspect_h5(path) for path in files]
        self.train_reads = [split_reads(info.read_count, train_fraction)[0] for info in self.infos]

        weights = np.asarray(self.train_reads, dtype=np.float64)
        self.weights = (weights / weights.sum()).tolist()

    def sample(self) -> tuple[SequenceBatch, Path]:
        # 随机选择一个文件，再随机选择一段连续 read。
        # 连续读取比完全随机 read 更适合 HDF5，也能保持 batch 构造简单。
        for _ in range(100):
            file_index = self.rng.choices(range(len(self.files)), weights=self.weights, k=1)[0]
            path = self.files[file_index]
            train_read_count = self.train_reads[file_index]
            read_count = min(self.batch_reads, train_read_count)
            max_start = max(0, train_read_count - read_count)
            read_start = self.rng.randint(0, max_start) if max_start else 0
            try:
                return (
                    read_h5_read_range(
                        path,
                        read_start,
                        read_start + read_count,
                        qmer_ks=self.qmer_ks,
                        rmer_ks=self.rmer_ks,
                        mer_stride=self.mer_stride,
                        qmer_vocab_size=self.qmer_vocab_size,
                        rmer_vocab_size=self.rmer_vocab_size,
                    ),
                    path,
                )
            except ValueError as exc:
                if "no non-empty reads" not in str(exc):
                    raise

        raise RuntimeError("failed to sample a batch with non-empty reads after 100 attempts")


def read_h5_read_range(
    path: Path,
    read_start: int,
    read_stop: int,
    qmer_ks: Iterable[int] | None = DEFAULT_MER_KS,
    rmer_ks: Iterable[int] | None = DEFAULT_MER_KS,
    mer_stride: int = DEFAULT_MER_STRIDE,
    qmer_vocab_size: int = DEFAULT_MER_VOCAB_SIZE,
    rmer_vocab_size: int = DEFAULT_MER_VOCAB_SIZE,
) -> SequenceBatch:
    """Load a half-open read range [read_start, read_stop) from one H5 file."""

    if read_stop <= read_start:
        raise ValueError("read_stop must be greater than read_start")

    with h5py.File(path, "r") as handle:
        # 先读取 read 边界，再把全局 row 区间切出来。
        # local_offsets 会被平移到以当前 batch 的 row_start 为 0。
        offsets = np.asarray(handle["/read_offsets"][read_start : read_stop + 1], dtype=np.int64)
        if offsets.size != read_stop - read_start + 1:
            raise ValueError(f"{path}: requested read range is outside /read_offsets")

        row_start = int(offsets[0])
        row_stop = int(offsets[-1])
        local_offsets = offsets - row_start
        freqs = np.asarray(handle["/freqs"][row_start:row_stop, :])
        observed = np.asarray(handle["/observed"][row_start:row_stop])

    return build_sequence_batch(
        freqs=freqs,
        observed=observed,
        local_offsets=local_offsets,
        qmer_ks=qmer_ks,
        rmer_ks=rmer_ks,
        mer_stride=mer_stride,
        qmer_vocab_size=qmer_vocab_size,
        rmer_vocab_size=rmer_vocab_size,
    )


def iter_read_batches(
    path: Path,
    train_fraction: float,
    split: str,
    batch_reads: int,
    max_reads: int | None = None,
    qmer_ks: Iterable[int] | None = DEFAULT_MER_KS,
    rmer_ks: Iterable[int] | None = DEFAULT_MER_KS,
    mer_stride: int = DEFAULT_MER_STRIDE,
    qmer_vocab_size: int = DEFAULT_MER_VOCAB_SIZE,
    rmer_vocab_size: int = DEFAULT_MER_VOCAB_SIZE,
) -> Iterator[SequenceBatch]:
    """Iterate deterministic read batches for train/test/all evaluation."""

    # 评估时使用确定性的顺序 batch，保证每次计算出的 metrics 可复现。
    info = inspect_h5(path)
    train_reads, test_reads = split_reads(info.read_count, train_fraction)
    if split == "train":
        read_start, read_stop = 0, train_reads
    elif split == "test":
        read_start, read_stop = train_reads, train_reads + test_reads
    elif split == "all":
        read_start, read_stop = 0, info.read_count
    else:
        raise ValueError(f"unknown split: {split}")

    if max_reads is not None:
        read_stop = min(read_stop, read_start + max_reads)

    cursor = read_start
    while cursor < read_stop:
        end = min(cursor + batch_reads, read_stop)
        try:
            yield read_h5_read_range(
                path,
                cursor,
                end,
                qmer_ks=qmer_ks,
                rmer_ks=rmer_ks,
                mer_stride=mer_stride,
                qmer_vocab_size=qmer_vocab_size,
                rmer_vocab_size=rmer_vocab_size,
            )
        except ValueError as exc:
            # 顺序评估时，如果某个 batch 恰好全是空 read，就直接跳过。
            # 这些 read 没有质量字符，本来也不应该计入 total_symbols。
            if "no non-empty reads" not in str(exc):
                raise
        cursor = end


class ResidualTransformer(nn.Module):
    """Lightweight causal Transformer that directly predicts residual logits.

    Position ``i`` may attend to itself because its input contains only the
    current H5 predictor features plus history shifted by one position. Future
    tokens are hidden by a causal sliding-window mask. Q/R-mer embeddings add
    multi-scale summaries built exclusively from already decoded history.
    """

    def __init__(
        self,
        continuous_dim: int = CONTINUOUS_FEATURE_DIM,
        q_hat_embed_dim: int = 16,
        prev_q_embed_dim: int = 16,
        prev_r_embed_dim: int = 32,
        qmer_ks: Iterable[int] | None = DEFAULT_MER_KS,
        rmer_ks: Iterable[int] | None = DEFAULT_MER_KS,
        qmer_vocab_size: int = DEFAULT_MER_VOCAB_SIZE,
        rmer_vocab_size: int = DEFAULT_MER_VOCAB_SIZE,
        qmer_embed_dim: int = 8,
        rmer_embed_dim: int = 8,
        d_model: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        feedforward_dim: int = 512,
        context_length: int = 256,
        dropout: float = 0.1,
        output_dim: int = RESIDUAL_CLASSES,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        if d_model <= 0 or num_heads <= 0 or d_model % num_heads != 0:
            raise ValueError("d_model must be positive and divisible by num_heads")
        if feedforward_dim <= 0:
            raise ValueError("feedforward_dim must be positive")
        if context_length <= 0:
            raise ValueError("context_length must be positive")

        self.continuous_dim = int(continuous_dim)
        self.d_model = int(d_model)
        self.context_length = int(context_length)
        self.qmer_ks = normalize_mer_ks(qmer_ks)
        self.rmer_ks = normalize_mer_ks(rmer_ks)
        self.q_hat_embedding = nn.Embedding(ALPHABET_SIZE, q_hat_embed_dim)
        self.prev_q_embedding = nn.Embedding(Q_TOKEN_COUNT, prev_q_embed_dim)
        self.prev_r_embedding = nn.Embedding(R_TOKEN_COUNT, prev_r_embed_dim)
        self.qmer_embeddings = nn.ModuleList(
            [nn.Embedding(qmer_vocab_size, qmer_embed_dim) for _ in self.qmer_ks]
        )
        self.rmer_embeddings = nn.ModuleList(
            [nn.Embedding(rmer_vocab_size, rmer_embed_dim) for _ in self.rmer_ks]
        )

        combined_dim = (
            continuous_dim
            + q_hat_embed_dim
            + prev_q_embed_dim
            + prev_r_embed_dim
            + len(self.qmer_ks) * qmer_embed_dim
            + len(self.rmer_ks) * rmer_embed_dim
        )
        self.input_projection = nn.Sequential(
            nn.Linear(combined_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )
        self.output_head = nn.Linear(d_model, output_dim)

    @staticmethod
    def _position_encoding(x: torch.Tensor) -> torch.Tensor:
        """Return sinusoidal position encodings for the current sequence."""

        seq_len, d_model = x.shape[1], x.shape[2]
        position = torch.arange(seq_len, device=x.device, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, device=x.device, dtype=torch.float32)
            * (-math.log(10_000.0) / d_model)
        )
        encoding = torch.zeros((seq_len, d_model), device=x.device, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term[: encoding[:, 1::2].shape[1]])
        return encoding.to(dtype=x.dtype).unsqueeze(0)

    def _attention_mask(
        self,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Mask future positions and history outside the configured window."""

        query_pos = torch.arange(seq_len, device=device).unsqueeze(1)
        key_pos = torch.arange(seq_len, device=device).unsqueeze(0)
        distance = query_pos - key_pos
        return (distance < 0) | (distance >= self.context_length)

    def forward(
        self,
        continuous: torch.Tensor,
        q_hat: torch.Tensor,
        prev_q: torch.Tensor,
        prev_r: torch.Tensor,
        qmer_tokens: torch.Tensor | None = None,
        rmer_tokens: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pieces = [
            continuous,
            self.q_hat_embedding(q_hat),
            self.prev_q_embedding(prev_q),
            self.prev_r_embedding(prev_r),
        ]
        if self.qmer_embeddings:
            if qmer_tokens is None:
                raise ValueError("qmer_tokens are required by this checkpoint")
            for mer_idx, embedding in enumerate(self.qmer_embeddings):
                pieces.append(embedding(qmer_tokens[:, :, mer_idx]))
        if self.rmer_embeddings:
            if rmer_tokens is None:
                raise ValueError("rmer_tokens are required by this checkpoint")
            for mer_idx, embedding in enumerate(self.rmer_embeddings):
                pieces.append(embedding(rmer_tokens[:, :, mer_idx]))
        x = self.input_projection(torch.cat(pieces, dim=-1))
        x = x + self._position_encoding(x)

        if lengths is not None:
            if torch.any(lengths <= 0) or torch.any(lengths > x.shape[1]):
                raise ValueError("lengths must be in [1, sequence_length]")

        # PyTorch 2.0 canonicalizes a boolean Transformer mask before entering
        # its no-grad fast path and emits a spurious conversion warning. The
        # mask remains causal and correct; suppress only that exact warning.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Converting mask without torch.bool dtype to bool.*",
                category=UserWarning,
            )
            out = self.transformer(
                x,
                mask=self._attention_mask(x.shape[1], x.device),
            )
        return self.output_head(out)


def batch_to_torch(batch: SequenceBatch, device: torch.device) -> dict[str, torch.Tensor]:
    """Move a numpy SequenceBatch to torch tensors."""

    return {
        "continuous": torch.from_numpy(batch.continuous).to(device),
        "q_hat": torch.from_numpy(batch.q_hat).to(device),
        "prev_q": torch.from_numpy(batch.prev_q).to(device),
        "prev_r": torch.from_numpy(batch.prev_r).to(device),
        "targets": torch.from_numpy(batch.targets).to(device),
        "valid_mask": torch.from_numpy(batch.valid_mask).to(device),
        "lengths": torch.from_numpy(batch.lengths).to(device),
        "qmer_tokens": torch.from_numpy(batch.qmer_tokens).to(device),
        "rmer_tokens": torch.from_numpy(batch.rmer_tokens).to(device),
    }


def mask_invalid_residual_logits(
    logits: torch.Tensor,
    q_hat: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    fill_value: float = -1.0e9,
) -> torch.Tensor:
    """Mask residuals that decode to quality ids outside [0, 94]."""

    # 对每个 q_hat，只允许 q_hat + r 落在合法 quality id 0..94。
    # 这不是数据集统计 mask，而是物理合法性 mask，不会把合法但少见的字符置零。
    residual_values = torch.arange(
        RESIDUAL_MIN,
        RESIDUAL_MAX + 1,
        device=logits.device,
        dtype=torch.long,
    )
    quality_values = q_hat.long().unsqueeze(-1) + residual_values.view(1, 1, -1)
    valid = (quality_values >= 0) & (quality_values < ALPHABET_SIZE)
    if valid_mask is not None:
        valid = valid & valid_mask.bool().unsqueeze(-1)
    return logits.masked_fill(~valid, fill_value)


def save_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def load_checkpoint(path: Path, device: torch.device) -> tuple[ResidualTransformer, dict[str, object]]:
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint["config"]
    model = ResidualTransformer(
        continuous_dim=int(config["continuous_dim"]),
        q_hat_embed_dim=int(config["q_hat_embed_dim"]),
        prev_q_embed_dim=int(config["prev_q_embed_dim"]),
        prev_r_embed_dim=int(config["prev_r_embed_dim"]),
        qmer_ks=config.get("qmer_ks", []),
        rmer_ks=config.get("rmer_ks", []),
        qmer_vocab_size=int(config.get("qmer_vocab_size", DEFAULT_MER_VOCAB_SIZE)),
        rmer_vocab_size=int(config.get("rmer_vocab_size", DEFAULT_MER_VOCAB_SIZE)),
        qmer_embed_dim=int(config.get("qmer_embed_dim", 8)),
        rmer_embed_dim=int(config.get("rmer_embed_dim", 8)),
        d_model=int(config["d_model"]),
        num_heads=int(config["num_heads"]),
        num_layers=int(config["num_layers"]),
        feedforward_dim=int(config["feedforward_dim"]),
        context_length=int(config["context_length"]),
        dropout=float(config["dropout"]),
        output_dim=int(config["output_dim"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, config
