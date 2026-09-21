# Per-file output-head adaptation

This opt-in experiment freezes the base model and fits its `Linear(d_model, 42)`
output head on a bounded FASTQ prefix. It does not train on the whole target file,
change the shared checkpoint, or use/update online experts during optimization.
It changes the codec from no-lookahead streaming to prefix-pre-read adaptation.
The exact fitted head is transmitted and paid for in the container.

## Defaults and admission

- At most 4096 whole reads and 600000 active qualities; stop before a read that
  would exceed the quality cap. A read longer than 2048 also stops sampling.
  The parser may read one extra record to discover a cap and gzip can read ahead.
- Split selected reads at `floor(3*N/4)`: first 75% train, remaining 25% validation.
  Reads are never split between training and validation. Require at least four
  reads, with nonempty quality sets in both splits. Empty reads remain legal.
- Backbone stays in eval mode (dropout off), uses `no_grad`, and runs once per
  feature batch. Active FP32 hidden features are preallocated on the device;
  default maximum cache is about 586 MiB for d_model=256, plus labels and model
  working memory. Padding is never cached or trained on.
- Adam, learning rate 0.001, at most 50 updates, at most 8192 sampled quality
  positions per update, seed 20260921. Positions are sampled with replacement
  from training features only. Loss is neural cross entropy plus 0.01 times the
  sum of mean-squared deviations of weight and bias from the base head; gradient
  norm clips at 1. A local generator does not advance the global sampling RNG.
- Default 20-second **cooperative** budget starts before reading the prefix.
  Prefix/features/optimization use at most the first 70% before checks stop work;
  final scoring and serialization use the remainder. CUDA is synchronized for
  timing. An individual read, kernel or validation chunk cannot be preempted;
  this is not a hard real-time 20-second guarantee. Check actual elapsed time.
- Serialize FP32 head, restore these exact values, then compare base/adapted
  **neural-only quantized bits/Q** on validation. Admit only strict improvement
  over `--head-min-gain` (default 0). No periodic validation or search on unseen
  results. Timeout, insufficient data, nonfinite training or no gain retains the
  original model. Lossless input validation errors still fail the command.

These are uncalibrated engineering starting values, not parameters selected on
the nine unseen files. Choose any future global hyperparameters using the ten
training datasets' fixed validation regions. A prefix admission check does not
prove net whole-file savings, hybrid gains, or representativeness of later reads.
Evaluate a later, unused region and ultimately actual full-file results.

## Experts and encoding

`--finetune-head` or `--head-adapter FILE` defaults to **neural-only** unless an
explicit probability profile or `--adaptive-weights` is supplied. Old invocations
without these flags retain their existing hierarchical-prior default.
`--neural-only` is also available for the unadapted baseline.

In hybrid mode, all v3 experts remain enabled. Formal encoding reopens the FASTQ
at read 1, creates fresh zero-count expert state and initial weights, and updates
only after each completely encoded batch. Prefix fitting never preloads counts.
The decoder mirrors this timing. New probabilities get their own weight feedback;
do not reuse the old-model mixture weights. This experiment does not automatically
choose between pure and hybrid modes.

## Container and decoder contract

An accepted head uses physical container **version 3**, with the same four
sections as v1/v2. Old readers reject the physical version. New readers keep
support for v1 legacy and v2 priors. v1/v2 reject a `head_adapter` field; v3 requires
it and an explicit `probability_profile` (JSON null denotes neural-only).

`head_adapter` has format `direct-quality-output-head`, version 1, application
`replace_output_head`, dtype `little_endian_float32`, and layout
`weight_row_major_then_bias`. It records `[42,d_model]` and `[42]` shapes, raw byte
length, raw SHA-256, base checkpoint SHA-256 and strict base64 data. Unknown or
missing fields, bad shapes/length/hash/dtype/version, and NaN/Inf are rejected
before entropy decoding. No pickle is used for the adapter.

For the production d_model=256 model, raw head size is 43176 bytes. Base64 and
metadata make the actual container overhead approximately 58 KB; **raw FP32
parameter size is not the full wire cost**. No FP16 quantization in this version.
The encoder applies the same serialized values as the decoder. The base
checkpoint SHA check remains mandatory; the decoder never trains or needs source
FASTQ, cache or an external adapter JSON. Numerical runtime requirements and
full/step integer-CDF agreement still apply; transferring weights alone does not
guarantee portability across arbitrary GPU/PyTorch configurations.

## Four-way comparison

Run from the repository root. Choose new output paths for every experiment:

```bash
# A: original neural-only
CUDA_VISIBLE_DEVICES=3 python -m codec.encode \
  data/2nd/CNR0847462_1.head2M.fastq.gz codec/output/CNR0847462_headexp_A.fqdc \
  codec/runs/direct_quality_balanced_b256_s40000_v1/best.pt \
  --device cuda --batch-reads 256 --neural-only

# B: original v3 mixture (the existing nine-file batch already supplies this baseline)
CUDA_VISIBLE_DEVICES=3 python -m codec.encode \
  data/2nd/CNR0847462_1.head2M.fastq.gz codec/output/CNR0847462_headexp_B.fqdc \
  codec/runs/direct_quality_balanced_b256_s40000_v1/best.pt \
  --device cuda --batch-reads 256 --adaptive-weights

# C: fit once, pure neural encoding, export exact result for a paired D run
CUDA_VISIBLE_DEVICES=3 python -m codec.encode \
  data/2nd/CNR0847462_1.head2M.fastq.gz codec/output/CNR0847462_headexp_C.fqdc \
  codec/runs/direct_quality_balanced_b256_s40000_v1/best.pt \
  --device cuda --batch-reads 256 --finetune-head --head-max-seconds 20 \
  --save-head-adapter codec/output/CNR0847462_headexp.adapter.json

# D: exactly the C head plus v3 experts; no second fitting run
CUDA_VISIBLE_DEVICES=3 python -m codec.encode \
  data/2nd/CNR0847462_1.head2M.fastq.gz codec/output/CNR0847462_headexp_D.fqdc \
  codec/runs/direct_quality_balanced_b256_s40000_v1/best.pt \
  --device cuda --batch-reads 256 \
  --head-adapter codec/output/CNR0847462_headexp.adapter.json --adaptive-weights

# Independent fresh-fit hybrid command instead of paired D:
# use --finetune-head --adaptive-weights (and a fresh output path).

# Decoder needs only container + unchanged base checkpoint; no new flags.
CUDA_VISIBLE_DEVICES=3 python -m codec.decode \
  codec/output/CNR0847462_headexp_C.fqdc codec/output/CNR0847462_headexp_C.restored.fq \
  codec/runs/direct_quality_balanced_b256_s40000_v1/best.pt \
  --device cuda --batch-reads 256
```

The export is a JSON artifact with base SHA, source provenance, fitting report
and adapter (null if fitting was rejected). Reusing a null artifact retains the
base head, making paired tests possible even after rejection. Exports refuse
overwrites; the export can remain even if later compression fails. They are
experimental output files: do not commit them. Loading a head does not include
its earlier fitting time in the current command's wall time; when comparing
fresh-fit deployment latency, add the original C command's
`head_adaptation.total_stage_seconds` (including export). The artifact's
`original_report.seconds` records fitting before the export itself, so it does
not include export I/O. A different deadline-limited fresh
fit can complete a different number of steps; exported-head reuse avoids this.

## Statistics and interpretation

- `head_adaptation`: accepted/reason, configuration, actual read ranges/counts,
  quality counts, prefix SHA, steps, before/after neural validation quantized
  bits/Q, stage seconds, budget_exceeded, raw parameter bytes and wire overhead.
- `encoding_stage_seconds.head_adaptation` and `encode_seconds` include fitting,
  admission, and optional export. Regular model prediction time still measures
  formal quality encoding only, not feature extraction for training.
- `range_stream_bytes` remains only the quality range section for compatibility.
- `quality_and_adapter_bytes` adds the transmitted adapter JSON overhead (and
  the explicit null profile for adapted neural-only) to the quality section.
  Use this instead of raw range bytes when comparing compression savings.
  Full `output_bytes` includes every container header and side-stream byte.
- No diagnostic four-way search, full-file scan, automatic expert removal or
  whole-file net-savings guarantee runs inside prefix fitting.

Correctness checks cover frozen backbone, read-disjoint splits, deterministic
fixed-step fitting, shared causal features, accepted/rejected/timed-out adapters,
strict malformed metadata rejection, original checkpoint preservation, and
byte-exact pure/hybrid full/step-CDF round trips. CPU tiny tests do not establish
GPU fitting time or compression gains on real files.
