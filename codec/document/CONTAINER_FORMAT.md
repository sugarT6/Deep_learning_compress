# Direct-quality FASTQ container format, versions 1–4

Adapter v10 adds a file-specific recurrent quality predictor inside physical
version 3. It carries all private GRU, normalization and head parameters and
reuses the exact SHA-bound public input features. Both codec ends use the same
explicit per-cycle recurrence. See [QUALITY_GRU.md](QUALITY_GRU.md) for its
strict payload, inference protocol, accounting and bounded verification commands.
The four-section container and quality-range format are unchanged.

Version 4 adds read quality states: an adapter v9, explicit `read_state`
metadata, and a fifth packed `read_state` section after `quality_range`:
one bit per read for two classes, or two bits for four classes. The class count,
bit width and protocol must agree with adapter v9. The two-class extension uses
new protocol identifiers; existing four-class files retain their original wire
representation and remain readable.
It requires neural-only probabilities and preserves the original quality range
protocol. See [READ_QUALITY_STATE.md](READ_QUALITY_STATE.md) for conditioning,
packing, checks, and complete storage accounting. Versions 1–3 keep their
existing four-section layout. The layout and legacy descriptions below apply
to versions 1–3 unless stated otherwise.

The direct-quality container reconstructs the decompressed FASTQ content byte
for byte. It does not attempt to reproduce the original gzip member bytes. Encoding
reads `.fastq`, `.fq`, `.fastq.gz`, or `.fq.gz` sequentially and never creates
or reads a training cache.

The optional output-head adaptation path pre-reads a bounded prefix before
formal encoding. Accepted adapters use version 3; see
[HEAD_ADAPTATION.md](HEAD_ADAPTATION.md) for the strict head wire format, fitting
budget and read-disjoint admission procedure. The base checkpoint stays external
and hash-checked; fitted head bytes are embedded in metadata and counted in size.

Current production encoding is neural-only: version 1 without an accepted head,
version 3 with an accepted head, or version 4 with read quality states. Adapted
neural containers use null `probability_profile`. All non-null
expert profiles below describe historical containers whose decoding remains
supported; they are no longer selectable by the production encoder.

## Top-level layout

All fixed-width integers are little-endian:

```text
fixed prefix (20 bytes)
stored metadata (canonical JSON, optionally zlib-compressed)
header_gzip section
base_gzip section
plus_gzip section
quality_range section
read_state section (version 4 only)
```

The fixed prefix is:

| Offset | Bytes | Meaning |
|---:|---:|---|
| 0 | 8 | magic `FQDC0001` |
| 8 | 2 | format version: 1 legacy, 2 prior, 3 transmitted head, 4 read states |
| 10 | 2 | flags: bit 0 means zlib-compressed metadata; all other bits must be zero |
| 12 | 4 | stored metadata length |
| 16 | 4 | CRC32 of the stored metadata bytes |

Metadata is UTF-8 JSON serialized with sorted keys and no insignificant
whitespace. The default flag value remains zero and preserves the existing wire
representation. `encode --compress-metadata` sets bit 0 and stores one zlib
stream containing exactly those canonical JSON bytes. The CRC covers the stored
stream, before decompression. Stored and decompressed metadata each have a 16 MiB
limit; truncated streams, concatenated streams, trailing data, noncanonical JSON,
and unknown flag bits are rejected. Existing readers that only accept zero flags
reject compressed metadata; the updated reader accepts both representations for
container versions 1–4. Adapter schemas and quality-range bytes are unchanged.

Section offsets are relative to the first byte after stored metadata, so
metadata length cannot recursively change its own offsets. The four sections
(five for version 4)
must be contiguous, ordered exactly as above, and consume the remainder of the
file. Each section entry records its relative offset, compressed/stored byte
length, logical uncompressed length, and CRC32. Truncation and trailing bytes
are errors.

For compressed metadata, adapter cost is the stored metadata length minus the
length obtained after removing `head_adapter` and its null `probability_profile`
member and recompressing with the same settings. Read-state framing is charged
after the adapter, as a second incremental length; packed label bytes are charged
separately. These increments sum to the actual additional stored bytes and avoid
charging the ordinary container prefix twice. The encoder reports the original
uncompressed adapter cost separately. See [ADAPTER_COMPACTION.md](ADAPTER_COMPACTION.md).

Metadata records at least:

- format magic/version and flags;
- source basename, input-file size (stored in the legacy
  `source_compressed_size` key), uncompressed FASTQ size, and uncompressed
  FASTQ SHA-256; for a plain FASTQ the input-file and uncompressed sizes are
  equal;
- read count, configured batch size (at most 256), batch count, last-batch read
  count, and quality symbol count;
- gzip side-stream schema and compression level;
- probability quantizer version, total frequency, Q0--Q41 alphabet, and
  Phred+33 offset;
- the complete causal-online-prior probability profile, including its version,
  batch update rule, cycle bin, smoothing/backoff strengths, fusion rule and
  weight, count dtype, and float contract;
- one range stream, coder version, and meaningful arithmetic payload bit count;
- complete model configuration, feature schema, checkpoint schema version, and
  SHA-256 of the exact checkpoint file;
- informational PyTorch version, inference device type, and float32 model
  inference dtype.

The decoder hashes the supplied checkpoint before model inference and rejects
a mismatch. The model configuration and feature schema loaded from that
checkpoint must also match the container.

Version 3 requires `head_adapter` with finite little-endian FP32 weight/bias,
base64 data, shape/length and SHA-256 checks, bound to the same base checkpoint.
It also requires `probability_profile`: null means adapted neural-only, otherwise
the complete existing prior profile is validated. Versions 1 and 2 reject any
`head_adapter` field. Old readers reject physical version 3. The section layout
is unchanged. After loading the base checkpoint, both codec ends replace its
output head with the exact stored parameters. The decoder never fits a model.

Current head-adapter schemas include: adapter v1 replaces the linear weight
and bias; adapter v2 replaces those plus a 256-to-r-to-42 residual MLP (the input
dimension follows the checkpoint). The residual activation is exact GELU;
the bounded width, all tensor shapes, ordering, and six FP32 tensors are
transmitted. Both use physical container v3. Unknown adapter versions,
activations or layouts are rejected, including by older decoders. External
adapter JSON exports are not decoding dependencies. See `HEAD_ADAPTATION.md`.

Adapter v4 adds a cross-layer branch to the residual head. Its source is the
third block output with per-position, channel-only, affine-free LayerNorm
(epsilon 1e-5). It requires at least four blocks, transmits four additional
cross-projection tensors and their shapes, and records source/normalization
rules and branch width. Physical container version remains 3. The normalization
is performed when extracting features on both codec ends, not learned during
decoding. Older adapters keep their original probability paths.

Adapter v3 used eight experimental Q-history features and is no longer
supported. Its feature construction and compatibility loader were removed
during the 2026-10-14 cleanup. Version 3 must not be reused for a new adapter
protocol. Physical container version 3 remains supported for current heads.

Adapter v5 was an unsuccessful eight-Q sequence-branch experiment and is no
longer supported. Use original commit `d73c8b8` for any experimental v5 files;
version 5 must never be reused for another protocol.

Adapter v6 requires exactly four Transformer blocks. Its envelope has
`application=replace_head_and_block4_qv`, a nested validated v4 `head`, bounded
integer `rank` (1..16), and `protocol=block4_qv_rank_major_f64_merge_v1`. Four
FP32 arrays are transmitted in Q-A [r,D], Q-B [D,r], V-A [r,D], V-B [D,r]
row-major order using SHA-256-protected base64. `parameter_bytes` counts these
arrays plus the nested head. Scaling is exactly 1. For each Q/V matrix, decoder
and encoder accumulate B[:,k]*A[k,:] in increasing rank order using CPU FP64,
add the original base matrix in FP64, then round once to FP32. K, all biases,
FFN and normalization weights remain unchanged. The merged model and head,
not an unmerged training graph, are used for wire validation and coding.
No additional inference branch or decode-time training is needed. The nested
head must be v4, preventing recursive envelopes. Physical container remains v3.

Adapter v7 uses `application=replace_head_and_transformer_suffix` and
`protocol=transformer_suffix_and_final_norm_xor32_byteplanes_zlib_v1`. Its
nonrecursive nested head is v1, v2 or v4. `layers` is the contiguous final
1..4 block indices, bound to the checkpoint's layer count and FFN width.
Canonical tensor descriptors enumerate each block's QKV projection, output
projection, two FFN matrices/biases, both normalizations, then final LayerNorm.

Treat every selected FP32 value as a little-endian uint32. XOR it with the
original checkpoint value; group delta byte 0 for all values, then bytes
1, 2, and 3. Zlib-compress this byte-plane stream and base64-encode it.
`raw_bytes` is the uncompressed byte-plane length. `delta_sha256` verifies
that stream, and `sha256` verifies the reconstructed FP32 parameter bytes.
The decompressor has a strict output bound and rejects trailing streams.
Reconstruction uses only integer XOR, discards no precision, and rejects
nonfinite weights before changing the model. `parameter_bytes` counts raw
selected parameters plus the nested head; actual added JSON bytes, including
compressed payload/base64/framing, are charged to quality-plus-adapter size.
The four physical sections and container version 3 are unchanged. Older
readers reject unknown adapter v7 rather than silently using base weights.

Adapter v8 uses `application=replace_head_and_suffix_lora` and
`protocol=transformer_suffix_qkvo_ffn_rank_major_f64_merge_v1`. Its nested head
is restricted to v1/v2/v4. The contiguous final 1..depth layer indices, layer
count and FFN width are bound to the checkpoint architecture; supported model
depth is at most 64. Rank is an integer in
1..16 and `scaling` is exactly integer 1. Canonical matrix descriptors list
packed QKV (three independent parts), attention output, linear1 and linear2
in each selected block. Each descriptor includes its base matrix shape.

For a matrix with P parts and shape [P*R,C], raw FP32 values appear as A
[P,rank,C] then B [P,R,rank]. After all matrices come final LayerNorm weight
and bias [D]. The raw bytes are grouped into four byte planes, zlib compressed
and base64 encoded. `sha256` verifies the uncompressed original factor/norm
bytes, and bounded decompression rejects oversized output and trailing data.
Nonfinite values, noncanonical matrix descriptors and recursive heads fail.

For each part, compute the sum of B[:,k]*A[k,:] in increasing k order on CPU
in FP64, add the pristine checkpoint matrix, and round once to FP32. Replace
final norm with the transmitted values. `merged_sha256` verifies concatenated
reconstructed matrix bytes in descriptor order, then norm weight/bias. All
checks complete before modifying model weights. Biases and within-block norms
are unchanged. Inference uses ordinary merged layers, not parametrizations.
`raw_bytes` counts factors/norm, `parameter_bytes` includes the nested head,
and actual JSON adapter bytes are charged to quality-plus-adapter size.
Physical container version 3 and its four sections remain unchanged.

Version 2 changes the probability protocol but not the four-section physical
layout. The default prior is specified in `ONLINE_PRIOR_FORMAT.md`; the optional
`causal_quality_mixture_v1` profile is specified in `FUSION_VALIDATION.md`.
The optional `causal_adaptive_mixture_v1` profile is specified in
`ADAPTIVE_WEIGHTS.md`; readers without it reject its unknown name.
Readers predating the mixture extension reject its unknown profile name.
Version-2 metadata
must contain a complete, strictly validated `probability_profile`. Legacy
version-1 containers contain no such field and remain decodable with the
neural-only probability path. A version-1 file that declares a profile, or a
version-2 file with a missing or invalid profile, is rejected before entropy
decoding. Older decoders reject version 2 instead of silently applying the
wrong CDF protocol.

## FASTQ gzip side streams

The three side streams are independent gzip members with `mtime=0`. Their
uncompressed data begins with a four-byte type/version magic:

```text
header: FQH1
base:   FQB1
plus:   FQP1
```

Header and base records use:

```text
uint64 field_length
uint8  line_ending_code
byte[field_length] field
```

Plus records use:

```text
uint64 plus_field_length
uint8  plus_line_ending_code
uint8  quality_line_ending_code
byte[plus_field_length] plus_field
```

Line-ending codes are `0=empty`, `1=LF`, and `2=CRLF`. The quality contents are
not duplicated; only their line ending is carried by the plus stream. Every
header, base, and plus byte is retained verbatim. Record boundaries come from
the length prefixes, not from searching for newline bytes.

The decoder reads exactly the container-declared number of entries from all
three streams and then requires gzip EOF and the declared uncompressed size.
Headers must begin with `@`, plus fields with `+`, and non-quality lines must
have an ending. Only the final quality line may omit its ending.

## Quality stream and batching

Probability quantizer version 2 is an opt-in no-sort floor protocol. For CPU
float64 softmax p, f[q]=1+floor(p[q]*(total-42)). Metadata `total` is a cap;
the actual arithmetic denominator is sum(f) at each position. Decoder computes
that sum rather than using the cap. The neural encoder can pass only selected
low/high/denominator intervals without materializing a full CDF. The range
stream remains v1; its integer interval update already supports variable
totals. Physical container versions and symbol order are unchanged. See
`FAST_QUANTIZATION.md` for operation order and tests. V1 quantization retains
its previous fixed-total largest-remainder rule. Older decoders reject v2.

The quality section is the standalone single range stream documented in
`RANGE_CODER_FORMAT.md`. The inner range-stream symbol count and meaningful bit
count must match the outer metadata. Its CRC is checked both as a container
section and by its own frame.

Reads are grouped exactly by the stored `batch_reads`. Within each batch,
quality symbols use cycle-major order and inactive variable-length positions
are skipped. Production encoding uses `forward_full`, while decoding uses
`forward_step` cycle by cycle. A slow `--verify-cdf` debug mode additionally
recomputes the opposite inference path and requires the two integer CDFs to be
identical. The online-prior tables remain frozen for a complete batch and are
updated only after every quality in that batch has been processed. This
cross-check is covered by tests but is disabled during normal compression and
decompression.

The current default and maximum are 256 reads. The batch dimension is not a
trained model parameter, so checkpoints trained with 64-read batches remain
valid. The exact batch size is stored because encoder and decoder must recreate
the same tensor grouping. Early version-1 implementations limited this value
to 64; current readers still accept those containers, while early readers will
reject new version-1 containers whose stored value exceeds 64. The binary
layout itself did not change.

The encoder computes the SHA-256 of the decompressed source records during its
single FASTQ pass. Decode writes to a temporary file, hashes the reconstructed
uncompressed records, and atomically publishes the output only after size and
SHA-256 match. A `.gz` decode output is a new deterministic gzip member whose
decompressed FASTQ bytes match; the gzip bytes need not match the source.

Versions 1 and 2 deliberately use one serial quality stream and recompute model
prefixes. In-memory range payload accumulation and incremental-model/KV-cache
work belong to Stage E, after correctness measurements are established.
