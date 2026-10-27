# File-specific recurrent quality predictor

Adapter **v10** replaces the quality predictor with a private GRU while reusing
the SHA-bound public checkpoint's Q embeddings, Q-mer embeddings, base CNN and
position features. It uses physical FQDC **v3** with the existing four sections,
range coder and either probability quantizer. It requires the updated reader;
older readers reject this adapter version. The decoder never trains a model.

All GRU matrices and biases, final LayerNorm parameters and the 42-class output
head are embedded and charged to the file. The public Transformer's projection,
attention layers and head are not evaluated for this predictor. Header fields
are not model inputs. Each read starts with zero recurrent state; there is no
cross-read history or transmitted read-state label.

## Import and bounded verification

The importer converts the checked `research-private-quality-gru-v1` absolute-Q
packet into the standard adaptation artifact. It verifies the public checkpoint
SHA, tensor names, shapes, finite values, decompression bounds and parameter SHA.
It preserves every FP32 parameter value; it does not fit or round weights.
Choose a new output path. Relative paths below are examples from the repository
root; `sample32.fastq` should contain only a small verification sample.

```bash
python -m codec.quality_gru_adapter frozen.model.json public_base.pt frozen.adapter.json
python -m codec.encode sample32.fastq sample32.fqdc public_base.pt \
  --device cpu --batch-reads 32 --head-adapter frozen.adapter.json \
  --compress-metadata --quantizer floor --coder-backend native --verify-cdf
python -m codec.decode sample32.fqdc restored.fastq public_base.pt \
  --device cpu --batch-reads 32 --verify-cdf
cmp sample32.fastq restored.fastq
```

`--compress-metadata` charges the actual incremental compressed metadata length
as `head_adaptation.container_adapter_bytes`. The file-specific contribution is
`range_stream_bytes + container_adapter_bytes`; the standalone JSON artifact is
a duplicate, not another decoder requirement. Without metadata compression the
larger canonical JSON length is charged. See [ADAPTER_COMPACTION.md](ADAPTER_COMPACTION.md)
for the accounting rule. That module's precision conversion supports v8 only;
it does not modify v10 predictors.

## Inference contract

The protocol identifier is
`public_features_f16_f32_gru_rzn_cycle_contiguous_v1`. Assemble the same public
features as the Transformer before its input projection. Explicitly convert
them FP32 → FP16 → FP32. Optional public causal-prefix features, if configured,
are included and their dimension is validated against the checkpoint.

Both encoder and decoder use explicit FP32 per-cycle recurrence. For every
layer, with contiguous input `x` and previous hidden state `h`:

```text
ir, iz, inn = split3(linear(x, weight_ih, bias_ih))
hr, hz, hn  = split3(linear(h, weight_hh, bias_hh))
r = sigmoid(ir + hr)
z = sigmoid(iz + hz)
n = tanh(inn + r * hn)
h = contiguous((1 - z) * n + z * h)
```

The updated hidden state becomes the next layer's input. The final head is
`Linear42(LayerNorm(h, eps=1e-5))`. Do not substitute cuDNN's full-sequence GRU:
it has produced different logits and CDFs. Decoder-owned state caches known base
features and hidden states, advances once per cycle and rejects changed batches,
owners, skipped cycles or altered past prefixes. Only already decoded Q values
enter the input features. State resets at every coding batch.

Verified 32-read CPU FQDC roundtrips reproduce all 4800 CDF rows and original
FASTQ bytes. Mixed lengths, empty reads/files, CRLF, missing final newline, both
quantizers and both metadata representations are covered by unit checks.
Cross-CPU/CUDA inference equivalence is not established. Use a matching inference
environment at both ends and keep all existing integrity checks enabled.

## Parameter wire

Metadata has exactly these members: `format`, `version`, `application`,
`protocol`, `base_checkpoint_sha256`, `feature_dim`, `hidden_dim`,
`recurrent_layers`, `dtype`, `compression`, `parameter_bytes`, `sha256`,
`data_base64`. Format is `direct-quality-output-head`, version is integer 10,
application is `replace_quality_predictor`, dtype is `little_endian_float32`,
compression is `byteplanes_zlib`.

Tensor order for layer 0 through L−1 is `gru.weight_ih_lN`, `gru.weight_hh_lN`,
`gru.bias_ih_lN`, `gru.bias_hh_lN`, followed by `norm.weight`, `norm.bias`,
`head.weight`, `head.bias`. Matrix order and gate order follow the equations
above. Shapes are derived from public feature dimension, hidden width and layer
count. Concatenate row-major little-endian FP32 tensors; SHA-256 covers these raw
bytes. Reshape bytes to `[parameters, 4]`, transpose to four byte planes,
zlib-compress and base64-encode. Truncation, extra streams/trailing data, wrong
raw lengths, nonfinite parameters, unknown members and wrong checkpoint hashes
are rejected. Supported widths are 1–512 and layer counts 1–4, subject to a
32 MiB raw parameter bound and the container's stricter metadata size limits.

The frozen research candidate is two layers of width 128, with 183466 private
parameters. Its pre-import retained-mantissa precision is five bits, plus FP32
sign and exponent; this is not a five-bit total weight representation. Any
subsequent weight or precision change requires new selection and independent
evaluation. Production integration alone does not establish a new compression
gain or a complete-file measurement.
