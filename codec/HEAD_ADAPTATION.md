# Per-file output-head adaptation

This opt-in experiment freezes the base model and fits its output head on a
bounded FASTQ prefix. The default is `Linear(d_model, 42)`; the opt-in residual
head adds a small GELU branch. It does not train on the whole target file,
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

## Optional nonlinear residual head

Use `--finetune-head --head-type residual --head-residual-dim 32` to fit

```text
z = W h + b + V GELU(U h + a) + c
```

Here `h` is the same frozen 256-dimensional feature. `U/a` map 256 to 32 and
`V/c` map 32 to 42; GELU uses `approximate="none"`. `W/b` start from the base
checkpoint, `U/a` use seeded initialization, and `V/c` start at zero, so the
initial function is exactly the original linear head. Initialization preserves
the global RNG state. Train all six head tensors on cached features; the
backbone and data features are unchanged. The existing anchor penalty applies
to each tensor's mean squared deviation from its initialization, including the
residual tensors. No experts, extra input features, LoRA, or Transformer updates.

`--head-type linear` remains the default and retains the original wire format.
Residual width defaults to 32 and must be in [1, 128]. Both modes use the same
prefix split, sampling, optimization, deadline and final-only admission test.
There is no periodic best-state selection or automatic residual-vs-linear
search: admission compares the final candidate to the original base head, not
to a separately fine-tuned linear head. Rejection returns the original head.

The extra computation affects both fitting and inference. A 20-second budget
is still cooperative, not a hard guarantee. Inspect `steps_completed` and
`budget_exceeded`; compare total time as well as quality-plus-adapter bytes.

## Optional strict-prefix Q features

Add `--head-history-features` to `--finetune-head --head-type residual`. This
changes only the residual input from `h` to `[h, s]`; the linear path still sees
the original 256-dimensional `h`, and the backbone/checkpoint/features remain
unchanged. There is no online expert, count table, or cross-read state.

`s` has eight FP32 scalars at zero-based cycle t, in this exact wire order:

1. Exact `q[t-2]/41`, or -1 when t < 2.
2. Exact `q[t-3]/41`, or -1 when t < 3.
3. `(q[t-1]-q[t-2])/41`, or zero when t < 2.
4. Mean of `q[:t]`, divided by 41 (zero for empty history).
5. Population standard deviation of `q[:t]`, divided by 41 (zero for empty).
6. Mean of the last min(t,8) past Q values, divided by 41 (zero for empty).
7. Feature 6 minus feature 4.
8. `min(t,8)/8`, an available-history marker.

Current/future qualities are excluded. Padding outputs are zero. Prefix sums
and sums of squares use int64, including the variance numerator `n*S2-S*S`,
before FP32 division and square root; this avoids floating-point scan order
differences between full and step inference. The exact implementation and
versioned schema live in `quality_history_features.py` and are shared by
training, encoding and decoding.

Features are computed once per prefix feature batch and cached with `h` for
all updates. At 1.2 million active qualities the eight added features use
38.4 MB of extra cache. For residual width 32, only 256 parameters are added:
total 20660 parameters / 82640 raw bytes. New input columns start at zero;
the seeded original residual h-weights/biases are preserved. All head tensors
are trained with the unchanged optimizer and anchor penalty. Deadline, split,
steps, and final-only admission remain unchanged. The gate still compares
against the unadapted base, not against an independently fitted residual head.

This is opt-in and is not assumed to improve compression. Both codec ends must
understand the new adapter schema; no external feature file is required.

## Experts and encoding

All production encoding is now **neural-only**, with or without head fitting.
`--neural-only` remains a harmless compatibility alias. `--adaptive-weights`,
`--probability-profile`, and `--prior-*` are rejected explicitly. The public
Python encoder no longer accepts `online_prior_config`. No expert state is
created and no expert fusion, count update or weight feedback is performed.
Formal encoding reopens the FASTQ at read 1 after prefix fitting.

The decoder retains the exact historical expert protocols, including mixed
adapted containers. Historical profile code and private encoder fixtures remain
for regression testing, not as production modes. To reproduce old four-way
experiments, use their original commit `071247c` in a separate checkout. Do not
resume old v3 batch runs under the new neural-only encoder.

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

The residual head uses **adapter version 2 inside physical container version
3**, not a new expert profile. Its layout is
`weight_bias_down_weight_down_bias_up_weight_up_bias`, all row-major FP32, with
`residual_dim`, `activation="gelu_exact"`, and explicit shapes for the four
extra tensors. All six tensors are serialized and restored before admission
scoring and formal inference. Old linear adapters remain readable; older
decoders without adapter-v2 support reject the new format. Update both ends.

**Adapter v3** adds the strict `history_features` schema (`causal_q_summary`,
version 1) and widens `down_weight_shape` to `[r,d_model+8]`. The tensor order,
activation, and physical container version 3 are unchanged. Unknown feature
orders, normalization, window sizes, or rules are rejected. Adapter versions
1/2 remain readable. This is unrelated to the retired four-expert profile v3;
`probability_profile` remains null for all current encodes.

For the production d_model=256 model, raw head size is 43176 bytes. Base64 and
metadata make the actual container overhead approximately 58 KB; **raw FP32
parameter size is not the full wire cost**. No FP16 quantization in this version.
With residual width 32, the head has 20404 parameters (81616 raw bytes), about
109 KB on the wire. The external exported `.adapter.json` is optional: all
parameters required for decoding already live inside the `.fqdc` metadata.
The encoder applies the same serialized values as the decoder. The base
checkpoint SHA check remains mandatory; the decoder never trains or needs source
FASTQ, cache or an external adapter JSON. Numerical runtime requirements and
full/step integer-CDF agreement still apply; transferring weights alone does not
guarantee portability across arbitrary GPU/PyTorch configurations.

## Current neural-only commands

Run from the repository root. Choose new output paths for every experiment:

```bash
# A: original neural-only
CUDA_VISIBLE_DEVICES=3 python -m codec.encode \
  data/2nd/CNR0847462_1.head2M.fastq.gz codec/output/CNR0847462_headexp_A.fqdc \
  codec/runs/direct_quality_balanced_b256_s40000_v1/best.pt \
  --device cuda --batch-reads 256 --neural-only

# C: fit 8000 prefix reads / up to 1000 updates, then pure neural encoding
CUDA_VISIBLE_DEVICES=3 python -m codec.encode \
  data/2nd/CNR0847462_1.head2M.fastq.gz codec/output/CNR0847462_headexp_C.fqdc \
  codec/runs/direct_quality_balanced_b256_s40000_v1/best.pt \
  --device cuda --batch-reads 256 --finetune-head --head-max-seconds 20 \
  --head-max-reads 8000 --head-max-symbols 1200000 --head-steps 1000 \
  --save-head-adapter codec/output/CNR0847462_headexp.adapter.json

# Reuse exactly the C head; no second fitting run
CUDA_VISIBLE_DEVICES=3 python -m codec.encode \
  data/2nd/CNR0847462_1.head2M.fastq.gz codec/output/CNR0847462_headexp_D.fqdc \
  codec/runs/direct_quality_balanced_b256_s40000_v1/best.pt \
  --device cuda --batch-reads 256 \
  --head-adapter codec/output/CNR0847462_headexp.adapter.json

# Decoder needs only container + unchanged base checkpoint; no new flags.
CUDA_VISIBLE_DEVICES=3 python -m codec.decode \
  codec/output/CNR0847462_headexp_C.fqdc codec/output/CNR0847462_headexp_C.restored.fq \
  codec/runs/direct_quality_balanced_b256_s40000_v1/best.pt \
  --device cuda --batch-reads 256
```

For the nonlinear experiment, keep the same 8000-read / 1000-update setup:

```bash
CUDA_VISIBLE_DEVICES=3 python -m codec.encode \
  data/2nd/CNR0847462_1.head2M.fastq.gz \
  codec/output/CNR0847462_head8k_s1000_res32_20260922.fqdc \
  codec/runs/direct_quality_balanced_b256_s40000_v1/best.pt \
  --device cuda --batch-reads 256 \
  --finetune-head --head-type residual --head-residual-dim 32 \
  --head-max-reads 8000 --head-max-symbols 1200000 \
  --head-steps 1000 --head-max-seconds 20 \
  --save-head-adapter codec/output/CNR0847462_head8k_s1000_res32_20260922.adapter.json

# No adapter JSON or head-type arguments needed at decode.
CUDA_VISIBLE_DEVICES=3 python -m codec.decode \
  codec/output/CNR0847462_head8k_s1000_res32_20260922.fqdc \
  codec/output/CNR0847462_head8k_s1000_res32_20260922.restored.fq \
  codec/runs/direct_quality_balanced_b256_s40000_v1/best.pt \
  --device cuda --batch-reads 256
```

For a fresh linear control, replace `--head-type residual --head-residual-dim
32` with `--head-type linear` and choose distinct output and adapter paths.
When reusing an exported residual head, supply only `--head-adapter FILE`;
the head type/width are restored from the artifact, not from training flags.

To test the added Q features without overwriting the previous experiment:

```bash
CUDA_VISIBLE_DEVICES=3 python -m codec.encode \
  data/2nd/CNR0847462_1.head2M.fastq.gz \
  codec/output/CNR0847462_head8k_s1000_res32_qhist_20260922.fqdc \
  codec/runs/direct_quality_balanced_b256_s40000_v1/best.pt \
  --device cuda --batch-reads 256 \
  --finetune-head --head-type residual --head-residual-dim 32 \
  --head-history-features \
  --head-max-reads 8000 --head-max-symbols 1200000 \
  --head-steps 1000 --head-max-seconds 20 \
  --save-head-adapter codec/output/CNR0847462_head8k_s1000_res32_qhist_20260922.adapter.json
```

Decode with the updated decoder and base checkpoint as above, using the new
container/output paths; no feature flag or external adapter is needed.

The 8000-read/1000-update command is the current single-file engineering test,
not a newly calibrated global default. The CLI defaults above remain unchanged.
`--head-max-symbols` caps the entire sampled prefix (training plus validation),
not the cumulative number of sampled positions across optimizer updates.

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
- No diagnostic four-way search, full-file scan or
  whole-file net-savings guarantee runs inside prefix fitting.

Correctness checks cover frozen backbone, read-disjoint splits, deterministic
fixed-step fitting, shared causal features, accepted/rejected/timed-out adapters,
strict malformed metadata rejection, original checkpoint preservation, and
byte-exact pure/hybrid full/step-CDF round trips. CPU tiny tests do not establish
GPU fitting time or compression gains on real files.

### Bounded GPU check (2026-09-22)

On an A100-PCIE-40GB, with the balanced-b256-s40000 base checkpoint and the
CNR0847462 8000-read prefix, residual width 32 completed 1000 updates in a total
adaptation time of 6.806 s (optimization 2.747 s). The read split was 6000/2000;
validation quantized bits/Q changed from 2.8610113614 to 2.2194215204. The earlier
linear 8000/1000 run scored 2.2534338031 on the same held-out prefix. This is
prefix evidence only, not a whole-file compression result or a timing guarantee.

A separate 257-read GPU round trip using the real checkpoint verified full/step
integer CDF agreement and byte-exact restoration, after the test-owned external
adapter export was removed. The embedded adapter overhead was 109443 bytes.
No full-file compression was launched for this check.

The eight-Q-feature extension, with the same prefix/seed/1000-update budget,
completed in 6.647 s (optimization 2.767 s), scoring 2.2195841372 bits/Q on the
same validation reads. This is slightly worse than 2.2194215204 without the
features, not evidence of improved whole-file compression. All 1000 updates
completed, no budget overrun; embedded adapter overhead was 111128 bytes.
The 257-read real-checkpoint GPU full/step CDF and byte-exact round trip passed
without an external adapter file. The implementation remains an opt-in
ablation; no hyperparameter search was performed against this file.
