# Opt-in completed-batch adaptive mixture weights

This profile changes **only the weights** of the three existing experts:
neural, smoothed exact order-2, and smoothed cycle/previous-Q/run-length.
There are no new contexts, neural architecture changes, training steps or
unseen-data parameter searches. Original fixed-alpha mixture and legacy
profiles remain unchanged and decodable.

## Compression command

From the repository root, using a new output filename:

```bash
CUDA_VISIBLE_DEVICES=3 python -m codec.encode \
  data/2nd/CNR0847462_1.head2M.fastq.gz \
  codec/output/CNR0847462_1_adaptive_weights_v1.fqdc \
  codec/runs/direct_quality_balanced_b256_s40000_v1/best.pt \
  --device cuda --batch-reads 256 --adaptive-weights
```

No process substitution/profile file is needed. Do not add `--prior-weight`,
`--verify-cdf`, or `--report-neural-only-bits` for the formal timing run.
`--adaptive-weights` and `--probability-profile` are mutually exclusive;
nondefault legacy prior flags are rejected with `--adaptive-weights`.
Omitting the new flag preserves the old production default.

The updated decoder automatically reads the profile from the container and
requires the same checkpoint. No adaptive flag is needed for decoding.
Containers remain physically v2 but carry the new, strictly validated
`causal_adaptive_mixture_v1` profile. Older decoders reject the unknown name;
they cannot silently decode it as the fixed-alpha profile.

## Initial configuration (not tuned on unseen files)

| Setting | Default |
|---|---|
| Initial weights: neural / order2 / run | 0.50 / 0.25 / 0.25 |
| Neural temperature | 1.0 |
| Adaptation rate eta | 0.1 per completed batch |
| Minimum expert weight f | 0.01 |
| Responsibility integer total U | 2^24 |
| Cycle bin / context smoothing | unchanged: 8 / 42 |
| Original three hierarchy strengths | unchanged: 42 / 42 / 42 |
| Run bins, BOS, padding, count updates | unchanged |

`AdaptivePriorConfig` extends `MixturePriorConfig`; its `alpha` specifies
initial weights `(1-alpha, alpha/2, alpha/2)`, **not a fixed subsequent alpha**.
The context mode must be enriched. Advanced configuration can be serialized
with `to_profile_metadata()` and supplied through `--probability-profile`.
All configuration, expert order, rounding, cold-start and update rules are
stored in metadata and validated. Rate must be in (0,1], floor in (0,1/3), and
initial weights must respect the floor. These defaults are implementation
choices, not a claim of optimal compression or guaranteed improvement.

## Update rule

Within an entire batch, weights and all count tables are frozen. For expert
probabilities `p_e(q)`, final probabilities are:

```
P(q) = w_N*p_N(q) + w_O*p_O(q) + w_R*p_R(q)
```

As before, CPU float64 log probabilities pass through the existing 42-class
integer-CDF quantizer. When no earlier active symbol exists, use neural `z/T`
alone and **skip weight learning for that cold batch**; online uniform models
must not be penalized before they have any observations.

After a symbol y has been encoded/decoded, compute the posterior responsibility
of each expert using the frozen weights and its already computed probabilities:

```
r_e = w_e*p_e(y) / sum_j(w_j*p_j(y))
```

This is a damped online-EM mixture update, not hard winner selection or a
softmax over the entire batch's summed losses. It credits experts relative to
the current mixture, using the likelihood of the symbol actually observed.

Before any accumulation, quantize **each symbol's** three responsibilities:
`u_e=floor(U*r_e)`. Give all leftover units to the largest responsibility,
breaking ties in neural/order2/run order. The three units sum to U. Accumulate
int64 units; do not accumulate floating-point losses in different chunk orders.
This makes full-batch encoding and per-cycle decoding feedback reductions
exactly partition-independent when their expert probabilities agree.

Only after all active qualities in the batch have been encoded/decoded:

```
m_e = sum(u_e) / (number_of_scored_symbols * U)
v_e = (1-eta)*w_e + eta*m_e
a_e = max(v_e-f, 0)
w_next_e = f + (1-3*f)*a_e / sum_j(a_j)
```

The floor projection preserves a nonzero route back to each expert. Count
updates remain at the same completed-batch boundary. State/weights reset at
every FASTQ. No current-batch target can change its own CDF. Padding does not
contribute. Debug verification is pure: it neither captures probabilities nor
duplicates observations. Missing/duplicate feedback is rejected.

## Cost and observability

There is still one `forward_full` per encode batch and one `forward_step` per
decode cycle. No extra per-expert CDFs, 42-class softmaxes, or neural-only
diagnostic passes are added. The existing order2/run probabilities are exposed
separately and reused. Feedback works on just three selected probabilities per
symbol. Encoding retains the three probability matrices until range encoding
finishes; decoding retains only one cycle's matrices. This adds memory traffic
and some CPU work; it is not guaranteed to be free or faster.

JSON reports `online_adaptation.final_weights`, expert order and update count.
`encoding_stage_seconds.adaptive_weight_feedback` isolates feedback collection;
weight commitment and existing count updates are in `prior_update`. Final
weights are diagnostic output only, not future information embedded in the
container. Final quantized theoretical bits remain enabled.

## Bounded verification

The complete repository suite passes 155 tests; `git diff --check` passes.

Tests cover hand-calculated responsibilities/updates, exact integer feedback
partitioning and ties, frozen current-batch weights, full/step CDF and next-batch
state agreement, weight floors, cold start, reset, strict profile rejection,
legacy paths, deterministic output, and 0/1/63/64/65/255/256/257/513-read
round-trips (including production-size 256-read batches). A real trained
checkpoint with a 65-read synthetic FASTQ also passed byte-exact CPU round-trip
and debug CDF verification across three batches; two weight updates occurred.
No training or large/unseen file evaluation was run for this implementation.

A CPU synthetic benchmark (NumPy seed 42, 256x150 Q0..Q41 symbols, float32
normal logits, warmed count tables, five repeats, median) measured fixed-alpha
versus adaptive fusion + quantization + feedback/count updates at 319.497 ms
versus 335.452 ms. Feedback/count work alone was 13.164 versus 22.122 ms.
Model inference, range coding and file I/O were excluded. The roughly 5%
increase in this measured stage is **not** an end-to-end overhead estimate.
