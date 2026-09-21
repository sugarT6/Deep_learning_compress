# Direct-quality FASTQ container format, versions 1–3

The direct-quality container reconstructs the decompressed FASTQ content byte
for byte. It does not attempt to reproduce the original gzip member bytes. Encoding
reads `.fastq`, `.fq`, `.fastq.gz`, or `.fq.gz` sequentially and never creates
or reads a training cache.

The optional output-head adaptation path pre-reads a bounded prefix before
formal encoding. Accepted adapters use version 3; see
[HEAD_ADAPTATION.md](HEAD_ADAPTATION.md) for the strict head wire format, fitting
budget and read-disjoint admission procedure. The base checkpoint stays external
and hash-checked; fitted head bytes are embedded in metadata and counted in size.

## Top-level layout

All fixed-width integers are little-endian:

```text
fixed prefix (20 bytes)
canonical JSON metadata
header_gzip section
base_gzip section
plus_gzip section
quality_range section
```

The fixed prefix is:

| Offset | Bytes | Meaning |
|---:|---:|---|
| 0 | 8 | magic `FQDC0001` |
| 8 | 2 | format version: 1 legacy, 2 prior, 3 transmitted head |
| 10 | 2 | flags, currently 0 |
| 12 | 4 | canonical JSON metadata length |
| 16 | 4 | CRC32 of the JSON metadata bytes |

Metadata is UTF-8 JSON serialized with sorted keys and no insignificant
whitespace. Section offsets are relative to the first byte after metadata, so
metadata length cannot recursively change its own offsets. The four sections
must be contiguous, ordered exactly as above, and consume the remainder of the
file. Each section entry records its relative offset, compressed/stored byte
length, logical uncompressed length, and CRC32. Truncation and trailing bytes
are errors.

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
