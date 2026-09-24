# Floor quantization and direct intervals

Opt-in flag: `--quantizer floor`. The default `largest-remainder` remains
unchanged for paired comparisons. This change does not modify the model,
side-stream gzip, symbol order, arithmetic renormalization, or single-stream
container structure. Old containers still decode.

## Probability protocol v2

For each position with 42 finite logits, CPU float64 computes, in order:

```text
w = exp(logits - max(logits))
p = w / sum_float64(w)
f = 1 + floor(p * float64(cap - 42))
T = sum_int64(f)
```

`cap` is `--quantization-total` (default 65536, allowed 42..2^30). Unlike v1,
it is an upper bound, not the actual denominator. The actual T is recomputed
per position on both codec ends. There is no sorting, remainder distribution
or forced sum of 65536. Every class retains positive frequency. Protocol v2
is only supported for pure-neural probabilities, not historical expert profiles.

Metadata records `probability_quantization.version=2`, `total=cap`, the same
42-class alphabet, Phred offset and `finite_cpu_float64` contract. Physical
container and range stream versions do not change. Old decoders reject v2;
the updated decoder reads v1 and v2. FP32/GPU softmax is not substituted.

The encoder works on chunks of at most 4096 positions and outputs only:
`low=sum(f[:symbol])`, `high=low+f[symbol]`, and `T`. It does not build a full
43-column CDF in production. The arithmetic encoder consumes these arrays,
using bulk `.tolist()` conversion instead of NumPy scalar indexing for every
symbol. It still updates a single arithmetic state sequentially. The decoder,
which does not yet know the symbol, builds the small per-cycle CDF batch from
the same frequencies and uses each row's actual T.

`--verify-cdf` additionally reconstructs full CDFs and checks direct intervals
against them, then compares full and step model CDFs. This is intentionally a
slow diagnostic for short files, not the full compression benchmark command.
Head adaptation and optional LoRA wire validation use the selected quantizer
and actual denominators. Existing exported heads may be reused with either
quantizer; no re-training is required for a controlled coding comparison.

The existing `cdf_quantization` timer now measures frequency/direct-interval
construction in floor mode; `range_encode` remains arithmetic encoding. This
preserves comparable report fields. No C++/JIT migration, multiprocessing or
side-stream optimization is included.

## Full compression command (8000 reads, 2000 head updates)

Run from the repository, select an available GPU, and use fresh output names:

```bash
CUDA_VISIBLE_DEVICES=3 python -m codec.encode \
  data/2nd/CNR0847462_1.head2M.fastq.gz \
  codec/output/CNR0847462_head8k_s2000_res64_cross16_floor_20260924.fqdc \
  codec/runs/direct_quality_balanced_b256_s40000_v1/best.pt \
  --device cuda --batch-reads 256 \
  --quantizer floor \
  --finetune-head --head-type residual --head-residual-dim 64 \
  --head-cross-layer --head-cross-dim 16 \
  --head-max-reads 8000 --head-max-symbols 1200000 \
  --head-steps 2000 --head-max-seconds 20 \
  --save-head-adapter codec/output/CNR0847462_head8k_s2000_res64_cross16_floor_20260924.adapter.json
```

LoRA is not enabled. Inspect `steps_completed` to confirm all 2000 updates fit
the cooperative time budget. Decode uses the updated code and base checkpoint;
the quantizer is read from the container, and the external adapter JSON is not
needed. Do not compare compressed size to an older 1000-step run as a clean
quantizer-only experiment. For that, reuse this exact adapter and encode with
`--quantizer largest-remainder` to a different output path, without fitting.

## Small CPU comparison

On 2026-09-24 all GPUs were occupied. A CPU-only diagnostic reused a previously
fitted res64+cross16 head (1000 updates), predicted 1024 real CNR0847462 reads,
then benchmarked identical logits (153600 qualities) in reverse-order pairs:

| Path | Quantization seconds | Range seconds | Stream bytes |
| --- | ---: | ---: | ---: |
| v1 full CDF + original range path | 0.596 | 0.577 | 42163 |
| v2 full CDF + interval range path | 0.259 | 0.433 | 42162 |
| v2 direct intervals + interval range path | 0.246 | 0.430 | 42162 |

The combined quantization/range time improved about 1.7x on this sample.
Bits/Q changed from 2.194714102 to 2.194658906. This is not a whole-file speed
or size claim, nor a GPU or 2000-update training measurement. Timings exclude
model prediction, I/O, fitting and finalization. Temporary arrays are bounded;
the comparison uses the original 256-read batches. Report location:
`codec/runs/floor_quantizer_20260924_v1/report.json` (ignored run output).

Reproduce with `python -m codec.benchmark_quantizer SOURCE CHECKPOINT ADAPTER NEW_OUTPUT_DIR`.
The diagnostic also performs a small real-model CPU roundtrip with full/step
integer-CDF verification. Unit tests cover variable read lengths, empty rows,
extreme/tied logits, cap boundaries, batch/chunk invariance, the scalar reference
bitstream, and adapter reuse. GPU roundtrip remains to be checked on an idle GPU.
