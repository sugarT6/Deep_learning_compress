# Validation-only fusion study and opt-in mixture profile

The production default remains `causal_online_hierarchical_v1`. This study
does not train/update neural weights or alter the ten-train/nine-unseen split.
Candidate availability is not evidence of better unseen compression.

## Bounded evaluation

```bash
CUDA_VISIBLE_DEVICES=3 python -m codec.evaluate_fusion \
  codec/runs/direct_quality_balanced_b256_s40000_v1/best.pt \
  codec/runs/fusion_validation_w1024_s4096_v1 \
  --device cuda --batch-reads 256 --warmup-reads 1024 --score-reads 4096
```

Use a new output directory. The default cache directory is
`data/2nd/training_cache`. This evaluation-only tool reads caches; the actual
encoder/decoder still never read caches. It checks the checkpoint's exact ten
training accessions, disjoint ranges, cache read counts and source hashes.
There is deliberately no unseen/category/dataset selection CLI.

For each file, start at the validation suffix plus `--validation-offset`
(default zero), reset all counts, replay `warmup_reads` without scoring, then
score the next `score_reads`. Warmup must be a multiple of batch size. All
requested reads must fit inside validation; no silent truncation is allowed.
Each scored batch has one `forward_full`, with logits reused in memory by all
candidates and discarded after the batch. Warmup needs no model inference.
All candidate predictions finish before any current-batch counts are updated.
No range coder runs. No complete-file histogram or logits cache is written.

The fixed 12 candidates are neural-only, online hierarchy only, online enriched
only, current multiplicative fusion, hierarchy/enriched arithmetic mixtures
with alpha 0.25/0.5/0.75 at temperature 1, and enriched alpha 0.5 at temperatures
0.85/1.15. This is a bounded diagnostic set, not a broad search.

`fusion_report.json` records checkpoint hash, exact source-hashed windows,
batch/warmup/score rules, candidate profiles, and final integer-CDF bits:

- equal-file dataset macro (the selection criterion), symbol micro, family
  macro, worst dataset and per-file bits/Q;
- per-file bits and symbol counts grouped by current Q, previous Q, cycle
  bin (width 8), and previous-Q support bins (0, 1..15, 16..255, 256..4095,
  >=4096 prior observations).

The best deployable candidate is selected by dataset macro, with deterministic
candidate-order tie breaking. Pure online-only candidates are diagnostic, not
deployable profiles. `selected_profile.json` is provisional: this one window
does not establish generalization. JSON null selects legacy neural-only.
Do not blindly replace the default or use CNR0847462/any of the nine unseen
files to select parameters. Confirm on a separately fixed validation window
before any final unseen evaluation. Warm-start bits are not whole-file bits;
report the reset/warmup rule, and check cold-start behavior separately.

The tool does not launch fqzcomp. Compare matching lossless quality streams
separately if needed, verifying version/options and exact quality restoration.

## Opt-in deployment and compatibility

```bash
CUDA_VISIBLE_DEVICES=3 python -m codec.encode INPUT.fastq.gz NEW_OUTPUT.fqdc \
  codec/runs/direct_quality_balanced_b256_s40000_v1/best.pt \
  --device cuda --batch-reads 256 \
  --probability-profile codec/runs/fusion_validation_w1024_s4096_v1/selected_profile.json
```

This command is a template, not a recommendation to deploy the provisional
winner. Do not combine profile JSON with nondefault `--prior-*` values. No
profile argument preserves the old default. A selected JSON null explicitly
uses the existing container-v1 neural-only protocol. Mixtures use container v2
with a new strictly validated name, `causal_quality_mixture_v1`; existing
v2 decoders reject this unknown profile instead of applying wrong CDFs.
The updated decoder dispatches from container metadata and needs no extra
profile file/flag. Model architecture, checkpoint and range format are unchanged.
Legacy v1 and original hierarchical-v1 containers remain decodable.

## Exact mixture protocol

All counts are int64; all probability operations are CPU float64. Parameters
and fixed rules are serialized in the profile and checked for missing/unknown
fields, unsupported versions/modes, invalid temperatures or nonfinite values.
Defaults of the *opt-in candidate*, not production defaults:

| Parameter | Value |
|---|---|
| alpha | 0.5, strictly between 0 and 1 |
| temperature | 1.0, supported interval [0.25, 4] |
| context mode | enriched (or hierarchical) |
| richer context strength | 42 |
| cycle width / original three smoothing strengths | 8 / 42 / 42 / 42 |
| update boundary | completed batch |
| run cap | 16 previous matching qualities |

Let B be the original three-level hierarchical online distribution. Enriched
mode adds two count tables:

```
O[previous2, previous, q]          [43, 43, 42]
R[cycle_bin, previous, run_bin,q]  [dynamic, 43, 5, 42]
```

The previous two values are exact Q IDs, not Q-mer buckets. Missing history
uses BOS=42 (previous at cycle 0; previous2 at cycles 0 and 1). The run is the
number of equal qualities ending at cycle-1, capped at 16. Run bins are 0/1,
2..3, 4..7, 8..15 and 16. No current/future quality enters these features.
Only active symbols update tables; padding is excluded. State resets per file.

With smoothing strength s, the online expert is:

```
O_s(q) = (O[q] + s*B(q)) / (sum(O) + s)
R_s(q) = (R[q] + s*B(q)) / (sum(R) + s)
E(q)   = (O_s(q) + R_s(q)) / 2       # enriched
E(q)   = B(q)                        # hierarchical
N_T    = softmax(neural_logits / T)
P      = (1-alpha)*N_T + alpha*E
```

All three original levels still participate through B. The enriched table
totals do not depend on current-batch counts. Shared contexts may be computed
once and looked up. Before any active symbol has been observed, return z/T
directly (temperature-neural-only cold start). Otherwise pass `log(P)` through
the existing v1 42-class integer-CDF quantizer, including its stable ties and
minimum frequency. Quantization intentionally re-normalizes these log scores;
both encoder and decoder follow exactly this sequence.

The model still runs once per encode batch and once per decode cycle. The
decoder changes only to implement the new probability protocol, not to optimize
its network inference. This release does not introduce online neural training,
adaptive mixture weights, or delta-change contexts; fixed mixtures plus exact
order-2/run contexts are the first bounded experiment.

## Observed bounded validation result (not a deployment recommendation)

Using the user's new `direct_quality_balanced_b256_s40000_v1/best.pt`, CPU,
batch 256, 4096 unscored warmup reads + 512 scored reads per training file:
46,080 total validation reads, 5120 scored reads and 20 model forwards. All
ten files were included; no unseen file was accessed. Detailed artifacts live
under ignored `codec/runs/fusion_validation_w4096_s512_v1/`.

| Candidate | Dataset-macro quantized bits/Q |
|---|---:|
| Neural only | 1.33418944 |
| Current multiplicative fusion | 1.35508741 |
| Online hierarchy only | 1.87873733 |
| Online enriched only | 1.82629507 |
| Hierarchical mixture alpha .25, T=1 | 1.39059357 |
| Enriched mixture alpha .25, T=1 | 1.38623753 |
| Enriched mixture alpha .5, T=.85 | 1.47125280 |

None of the new deployable candidates beat neural-only or current fusion on
this window. Richer contexts improved the online expert, but it remained much
weaker than the trained model on these *seen-file* validation suffixes. This
does not show that neural-only will win on unseen files (the user's prior
unseen result showed an online-prior benefit). The provisional selected JSON
is null; do not automatically apply it to a full unseen file. The production
default, checkpoint selection rule and training configuration are unchanged.

A preceding plumbing smoke used 16 warmup + 16 scored reads per file at batch
16; it is far too small and uses different update boundaries, so it is not a
parameter-selection result. No additional training was started. The next
statistical question is file-shift robustness, not merely a larger alpha sweep
on validation suffixes of files already used in training.

Verification: 147 repository tests pass, including manual context/smoothing
checks, BOS/Q0/Q41/padding, no current-batch count leakage, full/step CDF
equality, evaluator/deployed candidate agreement, corrupt profile rejection,
legacy compatibility, deterministic containers and boundary round-trips.
The hermetic evaluator test checks exactly ten validation sources, warmup/score
ranges, one forward per scored batch, breakdown sums and macro selection.
