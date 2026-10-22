# Encoder-only speed update

This update does not change model weights/architecture, prior configuration,
probability quantization, container metadata, range framing or the decoder.
For the same input, checkpoint, runtime and configuration, encoded bytes must
remain identical. Speed work alone does not improve compression ratio or
guarantee that a newly trained model beats fqzcomp on unseen files.

## Production changes

- `encode_fastpath.fuse_batch_logits` groups active positions by
  `(cycle_bin, prev_q)` and computes their float64 logit adjustments once.
  Original representative cycles preserve BOS validation. The lookup is local
  to each call, so there is no stale cache after the completed-batch update.
  The shared reference prior and decoder still use their original path.
- Neural-only softmax bits are disabled by default. Enable them with
  `--report-neural-only-bits` (Python: `report_neural_only_bits=True`). Both
  neural-only statistics are JSON `null` when disabled, not zero. This flag
  changes diagnostics only, not container bytes.
- Final quantized theoretical bits remain enabled. The encoder reads only
  `CDF[q+1] - CDF[q]` from already validated quantizer output, avoiding another
  full 42-class difference array and scan. Public validation remains intact.
- The range batch loop keeps arithmetic/bit state in local variables, avoids
  per-symbol/per-bit method calls and writes deferred identical bits in
  byte-sized chunks. The scalar public encoder remains the reference. There
  are no new dependencies or build steps; this is still a Python coder.
- JSON `encoding_stage_seconds` separates `prior_fusion`, `prior_update`,
  `neural_only_diagnostic`, `cdf_quantization`, `quantized_bits`, `range_encode`
  and other existing stages. `quality_entropy_coding_seconds` still aggregates
  the same broad scope. `cdf_quantization_and_transfer` now records only the
  optional full/step verification work; normal work has separate stage names.

Leave `--verify-cdf` off for formal timing. Keep all prior parameters fixed
between comparisons. Use a new output filename and the newly trained
checkpoint, not an overwritten old checkpoint. Old containers still require
their original checkpoint hash.

## Verification and bounded benchmark

139 repository tests passed, including unchanged decoder round-trips, legacy
containers, 63/64/65 and 255/256/257 reads, exact reference/fused logits and
CDF equality, multi-batch mixed scalar/batch range state, and identical
containers with diagnostics on/off and reference/optimized prior paths.

CPU synthetic benchmark against commit `80a711c`: NumPy seed 42, 256 x 150
active qualities sampled uniformly from Q0..Q41, normally distributed float32
logits, one previously observed batch, default prior parameters, five runs per
measurement, median wall time. No trained model or large file was used.

| Work | Before | After |
|---|---:|---:|
| Prior fusion (new includes grouping/index construction) | 108.987 ms | 11.317 ms |
| Range encoding and finalization | 560.380 ms | 245.926 ms |
| CPU probability/diagnostic/quantization/range pipeline | 865.864 ms | 428.336 ms |

The old pipeline includes neural-only diagnostics; the new one disables them.
Both include final quantized bits. Pipeline timing excludes neural prediction,
FASTQ parsing, prior count updates, side-stream gzip and container writing.
Synthetic symbol probabilities differ from real model outputs: these numbers
are not an end-to-end speed promise. Both benchmark pipelines produced equal
range bytes. Formal real-file timing is left to the user.
