# Causal running-delta expert (adaptive v3)

Historical experiment: production encoding is now neural-only. Expert flags
below require the earlier commit `071247c`; old expert containers still decode.
See [current encoding commands](HEAD_ADAPTATION.md).

`--adaptive-weights` now selects `causal_adaptive_mixture_v3`, adding a fourth
expert to the existing neural / cycle-order2 / cycle-previous-Q-run mixture.
No model, training, checkpoint architecture, data split, or neural inference
count changes. Physical containers remain v2; older decoders reject this
unknown profile. The updated decoder still supports all earlier profiles.

## Definition and defaults

Inspired by [fqzcomp's quality model](https://github.com/jkbonfield/fqzcomp/blob/master/fqzcomp.c),
running delta measures cumulative **downward** quality changes, not signed net
change, cumulative absolute change, or number of changes. For zero-based cycle c:

```
D(c) = delta_initial + sum(max(q[j-1] - q[j], 0), j=1..c-1)
delta_bin(c) = min(D(c) // delta_bin_width, delta_max_bin)
```

Default initial=5, bin width=8, maximum bin=7. No BOS-to-first-quality drop;
cycles 0 and 1 both have D=5. Example Q=[30,30,20,25,0,41,0] gives prior D values
[5,5,5,15,15,40,40], bins [0,0,0,1,1,5,5]. The drop to current q[c] enters only
the next position's context. Delta resets at every read, independently of
file-level tables. Integer prefix scans use only prefix entries for each
prediction; future/current symbols cannot affect that position's delta.

The new table is `count[cycle_bin, prev2, prev, delta_bin, q]` with exact Q0..Q41
and BOS=42. Both previous values are BOS at cycle 0. Only active prefix qualities
are counted; padding is excluded. Let P_O be the unchanged v2 position-order2
expert, C the matching delta count vector, and s=context_strength (default 42):

```
P_D(q) = (C(q) + s*P_O(q)) / (sum(C) + s)
P(q) = w_N*P_N(q) + w_O*P_O(q) + w_R*P_R(q) + w_D*P_D(q)
```

Empty delta contexts borrow the position-order2 parent. Existing order2,
position-order2, hierarchy and run tables continue participating. Float64 CPU
arithmetic, int64 counters, and Stage C 42-class integer CDF quantization remain
unchanged. Predicting an entire batch never updates counts or weights; after
all active symbols finish coding, update all tables and weights. State resets
per FASTQ, with no pre-scan or runtime training-cache use.

Initial weights are [0.5, 1/6, 1/6, 1/6]; alpha=0.5 still specifies total initial
online weight. Temperature=1, eta=0.1, floor=0.01, original hierarchy strengths
42/42/42, cycle width=8. The existing integer responsibility update (total 2^24
per symbol, remainder to largest responsibility, first expert wins ties) is
generalized to four experts. Projection uses `1-4*floor` rather than `1-3*floor`.
Cold first batch stays neural-only and skips weight learning. Parameters are
implementation defaults, not tuned using CNR0847462 or other unseen files.

All defaults, expert order, delta definition, context/backoff rule, smoothing,
binning, numerical and completed-batch update contracts are strictly validated
in metadata. Missing, malformed, or mismatched v3 protocol fields reject decode.
Use `RunningDeltaPriorConfig` for advanced JSON profiles. For the preceding
v2 adaptive profile, serialize `AdaptivePriorConfig().to_profile_metadata()`;
for v1, use `AdaptivePriorConfig(context_mode="enriched")`. Earlier profiles'
CDFs are regression checked against pre-change frozen digests.

## Cost and command

There is still one encode `forward_full` per batch and one decode `forward_step`
per cycle, with only one final CDF. The fourth probability matrix, prefix scan,
count lookup and update add CPU/memory traffic. At default 8 delta bins the
dense table costs about 4.74 MiB per allocated cycle bin (about 90 MiB for
150-base reads). Tables allocate through the highest observed cycle bin only
and are not stored in containers. No speed or compression gain is guaranteed.

Run from the repository root with a new output filename:

```bash
CUDA_VISIBLE_DEVICES=3 python -m codec.encode \
  data/2nd/CNR0847462_1.head2M.fastq.gz \
  codec/output/CNR0847462_1_adaptive_running_delta_v3.fqdc \
  codec/runs/direct_quality_balanced_b256_s40000_v1/best.pt \
  --device cuda --batch-reads 256 --adaptive-weights
```

No `--prior-weight`, debug CDF verification, neural-only diagnostic, or profile
file is required. The decoder selects v3 from metadata; use the same exact
checkpoint file that produced the container. JSON reports four final weights
and experts, final quantized theoretical bits/Q, and existing stage timings.

Tests cover hand-calculated drops/bins/saturation, BOS, Q0/Q41, variable active
masks, smoothing/missing contexts, current/future target invariance, next-batch
effects, full versus step CDFs, integer feedback and per-batch tables/weights,
floor projection, strict metadata, legacy compatibility, deterministic streams,
and boundary-sized byte-exact FASTQ round-trips. No formal training, large-file
test, or unseen-file tuning is part of implementation verification.

Verification: all 168 repository tests passed (16.563 s, CPU numerical threads
limited to one), as did `git diff --check`. The actual encode/decode CLIs using
the real trained checkpoint passed a 5-read, 91-quality, batch-2 CPU smoke with
`--verify-cdf`, across three completed batches and multiple position/delta bins;
the decoded FASTQ matched byte-for-byte. Two weight updates occurred.
