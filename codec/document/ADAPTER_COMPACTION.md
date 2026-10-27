# Saved adapter precision and metadata compression

`codec.compact_adapter` rounds a saved suffix-LoRA v8 predictor to fewer FP32
mantissa bits. This can improve compression of its existing byte-plane parameter
payload. It keeps the public base, rank, matrix layout, FP32 storage format and
canonical CPU FP64 factor-merge protocol. It validates the original artifact and
checkpoint SHA, then regenerates and validates all affected payload and merged
parameter checksums. The input artifact is preserved and output must be new.

For example, from the repository root:

```bash
python -m codec.compact_adapter saved.adapter.json public_base.pt compact.adapter.json \
  --factor-mantissa-bits 6 --head-mantissa-bits 8
python -m codec.encode sample.fastq sample.fqdc public_base.pt \
  --device cuda --head-adapter compact.adapter.json --compress-metadata \
  --quantizer floor --coder-backend native --verify-cdf
python -m codec.decode sample.fqdc restored.fastq public_base.pt --device cuda
```

The bit counts mean **retained mantissa bits**, in addition to FP32 sign and
exponent. They do not mean a six-bit total parameter representation. Defaults
are eight bits for factors/final normalization and all 23 bits for the output
head. Supported values are 2–23; 23/23 preserves the original adapter exactly.
Only v8 is supported; other adapter schemas are rejected explicitly.

Parameter rounding changes the predictor and can increase the quality code
length. Choose precision on a bounded selection sample, including the complete
adapter cost, then evaluate a fixed candidate on independent reads. The original
artifact's validation metrics are retained under `original_report`; they are not
measurements of the rounded predictor.

FASTQ compression remains lossless: encoder and decoder reconstruct the same
rounded predictor and range-code the original quality symbols. `--verify-cdf`
checks full/step probability agreement and is intended for small samples. Actual
decode and byte comparison are still required. Current GPU sample checks pass
when encode and decode use the same CUDA environment. CPU/CUDA FP32 inference
has produced different CDF rows with both original and rounded adapters;
cross-device interoperability has not been established. Parameter SHA validation
does not itself guarantee identical inference on different backends.

`--compress-metadata` is independent, lossless transport compression of the full
canonical JSON metadata, including the adapter. It is opt-in and requires the
updated decoder. Ordinary output-head FP32 data are base64 without internal
compression, so reducing head precision alone does not shrink that JSON member;
outer metadata compression can recover those repeated bits. See
[CONTAINER_FORMAT.md](CONTAINER_FORMAT.md) for flag, integrity and size checks.
Without the new option, encoding preserves the existing container layout.

`head_adaptation.container_adapter_bytes` and `quality_and_adapter_bytes` charge
the incremental compressed metadata cost when enabled. The report also retains
`container_adapter_bytes_uncompressed` and identifies
`metadata_encoding=canonical_json_zlib`. The standalone exported JSON is a
duplicate artifact and is not charged again. Shared public checkpoint cost is
still reported separately.
