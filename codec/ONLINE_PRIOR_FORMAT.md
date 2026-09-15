# Causal online quality prior, probability profile version 1

New version-2 FASTQ containers use the `causal_online_hierarchical_v1`
probability profile. The profile adapts a frozen neural checkpoint to quality
statistics already observed in the same file. It never scans ahead, reads a
training cache, or stores a target-file quality histogram in the container.

## State and update boundary

Every file starts with three zero-filled signed 64-bit count tables:

```text
C_global[q]                         shape [42]
C_prev[prev_q, q]                   shape [43, 42]
C_cycle[cycle_bin, prev_q, q]       shape [dynamic, 43, 42]
```

`prev_q=42` is BOS and is used only at cycle 0. At later cycles `prev_q` is
the preceding quality of the same read. `cycle_bin = cycle //
cycle_bin_width`. Only active Q0--Q41 symbols are counted; padding is never
counted.

The state visible to a batch contains only earlier, completely processed
batches. Encoder and decoder freeze the tables for all symbols in the current
batch, including all cycles, then update all three tables once after the whole
batch has been range-encoded or range-decoded. Within-batch decoded quality
history may select a `prev_q` row, but no current-batch symbol contributes to a
count until the batch is complete. The exact stored `batch_reads` is therefore
part of the probability contract.

## Hierarchical smoothing and backoff

All calculations below use finite CPU IEEE-754 float64. Let `U(q)=1/42` and
let the configured positive backoff strengths be `lambda_global`,
`lambda_prev`, and `lambda_cycle`:

```text
G(q) = (C_global[q] + lambda_global * U(q))
       / (sum_q C_global[q] + lambda_global)

T(q | prev) = (C_prev[prev,q] + lambda_prev * G(q))
              / (sum_q C_prev[prev,q] + lambda_prev)

B(q | bin,prev) = (C_cycle[bin,prev,q] + lambda_cycle * T(q | prev))
                  / (sum_q C_cycle[bin,prev,q] + lambda_cycle)
```

This is deterministic interpolated backoff in the fixed order
`cycle_bin+prev_q -> prev_q -> global_q -> uniform`. Positive strengths mean
all three count levels and uniform smoothing continue to participate even when
a more specific row has observations. The default `lambda_global=42` is
equivalent to adding one pseudocount to every quality.

## Neural fusion and integer quantization

Let neural logits be `z(q)`, the hierarchical prior be `B(q)`, uniform
`U(q)=1/42`, and the configured prior weight be `w`, strictly between zero and
one. The codec forms float64 scores

```text
s(q) = z(q) + w * log(B(q) / U(q))
```

and normalizes them implicitly in the existing stable-softmax quantizer. This
is multiplicative logit adjustment

```text
P_final(q) proportional to P_neural(q) * (B(q) / U(q)) ** w.
```

Because the uniform factor is constant across qualities, it does not change
normalization; writing it explicitly makes the neutral behavior clear. Before
any earlier batch has been observed, `B=U`, so the fused scores and integer
CDFs are exactly neural-only. The resulting scores go through probability quantization version 1: 42
positive integer frequencies, fixed `TOTAL`, largest-remainder allocation, and
quality-id tie breaking. Encoder `forward_full` and decoder `forward_step`
must produce identical final integer CDFs, not merely close floating-point
probabilities.

## Default configurable parameters

| Parameter | Default | Encode CLI |
|---|---:|---|
| cycle-bin width | 8 cycles | `--prior-cycle-bin-width` |
| global-to-uniform strength | 42.0 | `--prior-global-backoff-strength` |
| prev-Q-to-global strength | 42.0 | `--prior-prev-q-backoff-strength` |
| cycle-to-prev-Q strength | 42.0 | `--prior-cycle-backoff-strength` |
| prior-ratio logit-adjustment weight | 0.25 | `--prior-weight` |

These are fixed implementation defaults, not values selected on unseen data.
Any later parameter selection must use only validation portions of the ten
training datasets. A version-2 container stores and strictly validates the
profile name/version, all parameters, update granularity, backoff order,
fusion identifier, count dtype, and float contract.

Legacy version-1 containers have no `probability_profile` field and decode
with neural-only CDFs. Version 2 requires the complete online-prior profile;
missing, extra, or unsupported fields are errors.

Encoder optimization does not change this protocol: contexts may share a
precomputed float64 adjustment within one frozen batch. Final quantized bits
are always reported; neural-only softmax bits now require the optional
`--report-neural-only-bits` diagnostic flag and otherwise report `null`.
See `ENCODING_SPEED.md` for details.
