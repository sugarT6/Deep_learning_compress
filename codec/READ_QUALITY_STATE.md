# Read-level quality state

`--read-quality-state` adds a transmitted four-class read quality state to
the normal Transformer suffix adaptation pipeline. It works with the existing
base checkpoints, dense suffix updates, and multi-layer Q/K/V/O + FFN LoRA.
It requires `--finetune-head --finetune-last-layers N`; it cannot be combined
with the legacy `--head-lora` path. No new pretraining is required.

## Conditioning and fitting

The ordinary output-head warm-up runs first, unchanged. The joint stage samples
complete reads across the file, excludes the warm-up prefix, and splits them
75%/25% for training/validation as before. Only training reads determine three
nearest-rank quartiles of `floor(1000 * sum(Q) / read_length)`. The four states
are the number of thresholds less than or equal to that integer mean. Empty
reads always use state zero; repeated thresholds and unused classes are legal.
Neither Header fields nor platform/species labels enter the model.

At the input of **each adapted Transformer block**, the main hidden sequence is
conditioned by a learned per-state scale and shift:

```text
h = h * (1 + scale[layer, state]) + shift[layer, state]
h = TransformerBlock[layer](h, causal_mask)
```

Both tensors start at zero and train jointly with the selected suffix, final
LayerNorm, and output head. Earlier blocks remain frozen. For two adapted D=256
blocks this adds 4096 FP32 parameters / 16384 raw bytes. This is main-backbone
conditioning; the residual64/cross16 head is retained. Dropout stays disabled
during per-file fitting. The existing backbone learning rate and anchor apply
to the state tensors too.

The best validated **positive training step** is used, even when loss or net
size is worse than the initial head. No gain/cost fallback is added. Insufficient
samples, nonfinite values, or an inability to validate a trained state fail
explicitly. `--head-max-seconds 0` disables the adaptation deadline.

## Lossless wire protocol and accounting

Adapter schema **v9** wraps exactly one dense v7 or suffix-LoRA v8 adapter and
stores the thresholds and exact little-endian FP32 scale/shift tensors, with
length/shape/finite-value/SHA checks. Nested state envelopes are rejected.
Both ends restore the same adapter; decoding never repeats training. Applying
another adapter clears previous conditioning, and repeated application restores
from pristine base weights.

Physical container **v4** appends a fifth `read_state` section after the four
existing sections. There is one label per read, including empty and warm-up
reads. Labels use two bits, most significant first, packed continuously across
batch boundaries; unused final low bits must be zero. Stored length is
`ceil(read_count / 4)` bytes. The quality arithmetic stream still contains only
Q symbols and retains its existing protocol.

The encoder computes each complete read's state and predicts its original Q
values conditioned on that state. The decoder reads the state before predicting
the read's qualities, then checks that reconstructed qualities reproduce it.
All original Q values, FASTQ fields, line endings, and final-newline behavior are
preserved. Missing labels, bad padding, label/adapter mismatch, corrupt sections,
and mismatched base checkpoints are errors. Existing containers v1--v3 remain
readable; older codec versions cannot decode v4.

`quality_and_adapter_bytes` includes the quality range stream, entire v9 adapter,
state label bytes, and the additional state metadata/section descriptor. Reports
also expose `read_state_stream_bytes`, `read_state_metadata_bytes`, and
`read_state_stream_bits_per_quality`. The shared base checkpoint remains excluded
and must be reported separately. The exported adapter JSON is a reusable copy,
not an extra decoding dependency or a second storage charge.

For 2M reads of 150 bp, labels cost exactly 500000 bytes / 0.013333 BPQ, plus
state parameters and JSON framing. Validation BPQ is conditional Q code length
and excludes these costs; the final `quality_and_adapter_bytes` is the net metric.
The pre-encoding savings estimate includes the fixed label stream but excludes
its small JSON framing and is only diagnostic.

## CNR compression test

Choose a free GPU by adjusting `CUDA_VISIBLE_DEVICES`. This matches the verified
rank8/5000-step suffix-LoRA baseline and adds state conditioning. It creates a
unique output directory. Only stdout goes to JSON; training messages and the
`encode FASTQ` progress bar stay in the terminal. Build the native coder once
with `python -m codec.build_native` if it has not been built on this machine.

```bash
cd /home/share/huadjyin/home/s_qiuping1/tanglei/fastq_quality_residual
mkdir -p codec/output/time_baseline_nodsub
STATE_RUN=$(mktemp -d codec/output/time_baseline_nodsub/CNR_readstate4_lora8_XXXXXX)
CUDA_VISIBLE_DEVICES=0 /usr/bin/time -v -o "$STATE_RUN/time.txt" \
python -m codec.encode \
  data/2nd/CNR0847462_1.head2M.fastq.gz \
  "$STATE_RUN/CNR0847462.fqdc" \
  codec/runs/direct_quality_4layer_resume80k_20261014_sPfodw/best.pt \
  --device cuda --batch-reads 256 --quantizer floor --coder-backend native \
  --finetune-head --head-type residual --head-residual-dim 64 \
  --head-cross-layer --head-cross-dim 16 \
  --head-max-reads 8000 --head-max-symbols 1200000 --head-steps 2000 \
  --head-max-seconds 0 --finetune-last-layers 2 \
  --backbone-lora-rank 8 --read-quality-state \
  --backbone-steps 5000 --backbone-batch-reads 16 \
  --backbone-learning-rate 0.0003 --backbone-eval-every 200 \
  --backbone-patience 6 --backbone-max-reads 32000 \
  --backbone-max-symbols 4800000 \
  --save-head-adapter "$STATE_RUN/head.adapter.json" \
  > "$STATE_RUN/metrics.json"
echo "Results: $STATE_RUN"
```

Compare `quality_and_adapter_bytes` against **75472388** for the previous
rank8 run; the head-only reference is 76426168. Conditional BPQ improvement
alone does not establish net savings. `--verify-cdf` is for small correctness
samples and should not be added to the full 2M-read compression benchmark.

To decode with the same checkpoint and batch size:

```bash
CUDA_VISIBLE_DEVICES=0 python -m codec.decode \
  "$STATE_RUN/CNR0847462.fqdc" "$STATE_RUN/CNR0847462.restored.fastq" \
  codec/runs/direct_quality_4layer_resume80k_20261014_sPfodw/best.pt \
  --device cuda --batch-reads 256 > "$STATE_RUN/decode_metrics.json"
gzip -cd data/2nd/CNR0847462_1.head2M.fastq.gz | cmp - "$STATE_RUN/CNR0847462.restored.fastq"
```

Decoding is still cycle-wise and can take substantially longer than encoding.
