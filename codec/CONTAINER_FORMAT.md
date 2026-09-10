# Direct-quality FASTQ container format, version 1

The Stage D container reconstructs the decompressed FASTQ content byte for
byte. It does not attempt to reproduce the original gzip member bytes. Encoding
reads `.fastq.gz` or `.fq.gz` sequentially and never creates or reads a
training cache.

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
| 8 | 2 | format version, currently 1 |
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
- source basename, compressed size, decompressed size, and decompressed
  SHA-256;
- read count, configured batch size (at most 64), batch count, last-batch read
  count, and quality symbol count;
- gzip side-stream schema and compression level;
- probability quantizer version, total frequency, Q0--Q41 alphabet, and
  Phred+33 offset;
- one range stream, coder version, and meaningful arithmetic payload bit count;
- complete model configuration, feature schema, checkpoint schema version, and
  SHA-256 of the exact checkpoint file;
- informational PyTorch version, inference device type, and float32 model
  inference dtype.

The decoder hashes the supplied checkpoint before model inference and rejects
a mismatch. The model configuration and feature schema loaded from that
checkpoint must also match the container.

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
are skipped. Encoding uses `forward_full`; before changing the range state for
that batch, every active position is also recomputed with `forward_step` and
the two integer CDFs must be identical. Decoding uses `forward_step` cycle by
cycle and performs a post-batch `forward_full` integer-CDF check after all true
qualities have been recovered.

The encoder computes the SHA-256 of the decompressed source records during its
single FASTQ pass. Decode writes to a temporary file, hashes the reconstructed
uncompressed records, and atomically publishes the output only after size and
SHA-256 match. A `.gz` decode output is a new deterministic gzip member whose
decompressed FASTQ bytes match; the gzip bytes need not match the source.

Version 1 deliberately uses one serial quality stream and recomputes model
prefixes. In-memory range payload accumulation and incremental-model/KV-cache
work belong to Stage E, after correctness measurements are established.
