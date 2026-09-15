# Dataset-balanced training and read coverage

The training CLI defaults to `--batch-reads 256`, accepting 1 through 256.
Batch size is an optimization/data-loading setting, not a model architecture
parameter. The fixed ten training files and nine unseen files are unchanged.

## Scheduling and file blocks

Sampler strategy `strict_dataset_balanced_shuffled_blocks`, version 2:

1. For `S` steps and `D` training files, give every file `S // D` slots.
2. Give the `S % D` extra slots to successive files in the fixed dataset order,
   starting at a persistent remainder cursor. Advance that cursor modulo `D`.
3. Shuffle the complete epoch schedule using its own seeded NumPy generator.
4. Each scheduled file supplies exactly one contiguous block from its train
   prefix. Block size is `batch_reads`; the final smaller block is retained.
5. Each file has a separate block permutation, round and cursor. Permutations
   use NumPy `SeedSequence([seed, dataset_index, round])`. Consume every block
   before incrementing the round and generating its next permutation.

Epoch boundaries never reset block cursors. At every completed epoch, cumulative
file slot counts differ by at most one; when `S` is divisible by ten, every
file gets exactly `S/10` slots in that epoch. Counts during a shuffled epoch
may differ by more than one. Each batch comes from a single file. Reads,
symbols, read lengths and platform families do not weight file selection.

For a 1,800,000-read prefix and batch 256, the blocks are 7031 full blocks and
one 64-read tail: 7032 optimizer steps per file. A partial block still receives
one optimizer update using mean loss over its active quality symbols.

Ten equal file weights imply the platform totals MGI/BGI 30%, Illumina 50%,
ABI SOLiD 10%, Ion Torrent 10%, given the unchanged 3/5/1/1 file split.

## Coverage accounting

| Configuration | Steps | Nominal read slots | Fraction of 18M training reads |
|---|---:|---:|---:|
| Historical run, batch 64 | 40,000 | 2,560,000 | 14.22% |
| New comparison, batch 256 | 40,000 | 10,240,000 | 56.89% |
| One complete pass, batch 256 | 70,320 | 18,001,920 | 100% actual coverage |

The historical sampler drew overlapping random ranges with replacement, so
14.22% was only an upper bound on unique coverage. The new strategy avoids
repeated reads until each file has exhausted its blocks. The 40k-step run has
4000 blocks per file, below the 7032-block pass length, so it has no repeated
training reads. Its actual count is `10,240,000 - 192*t`, where `t` is the
number of encountered tail blocks (0 through 10). The full pass reads exactly
18,000,000 actual reads, including all ten tails.

`sampling_statistics.json` is written only after training finishes. It records
dataset/family batch counts and proportions, total nominal slots, actual reads,
unique reads, train read count, and per-file coverage/round/cursor. Checkpoints
also save the same cumulative statistics after each completed epoch.

Changing batch 64 to 256 changes optimization dynamics and quadruples nominal
reads per step. The 40k-step command matches the old step count, not the old
read budget; it is not an isolated sampler-only ablation. Learning rate,
architecture and features are unchanged by default.

## Validation and best checkpoint

Only validation suffixes of the ten training files select `best.pt`.
`selection_metric = dataset_macro_bits_per_quality`: average the ten per-file
bits/Q values with equal weight. A strictly lower value replaces the best;
ties keep the earlier checkpoint. Nonfinite primary metrics fail explicitly.

Validation always reads contiguous records from the start of each validation
suffix, capped by `--validation-max-reads-per-file` (default 5000; 0 means the
whole suffix). The rule, limit and actual per-file read counts are saved in
`run_config.json`. Validation does not consume or advance sampler state.

Reports retain symbol micro, platform-family macro (mean of family symbol
micro values), per-file bits/Q and worst dataset. Epoch output remains one
progress bar and one summary line; no sampled dataset names are printed.
The nine unseen datasets never enter training, validation selection, stopping
or hyperparameter selection.

## Checkpoints and resuming

Checkpoints retain model configuration/weights, feature/cache schemas and
source-hashed data splits. New training checkpoints additionally contain:

- sampler version, full configuration and state: schedule RNG and cursor,
  extra-slot cursor, per-file block order/round/cursor and counts;
- the primary selection metric and complete validation report;
- optimizer state and versioned training state with run configuration and
  Python, NumPy, Torch CPU and CUDA RNG states.

The outer checkpoint schema remains 1 for codec inference compatibility.
Old checkpoints still load for inference, but cannot resume this sampler
because they lack its state. No online-prior or model-feature schema changes
are made by this training update.

`--resume /absolute/path/to/run/last.pt` resumes after its completed epoch.
Use a **new empty output directory** and keep the original matching `best.pt`
beside the source checkpoint. The previous best is copied to the new output
before continuation, so a run with no further improvement still has its best.
`--epochs` is the total target epoch, not the number of additional epochs.
Configuration, data hashes, batch size, steps per epoch and device must match;
only the target epoch count and output location may change. Sampler state can
also be serialized/restored mid-epoch via its public state methods; the CLI
currently writes checkpoints only at completed epochs. Reproducible model
updates require the same numerical runtime/hardware as well as RNG state.

## Formal commands (run manually)

Run these from the repository root. Choose new empty output directories; never
overwrite an old run. Both commands preserve the existing architecture and
optimizer defaults. They are documented commands, not part of automated tests.

40,000 steps, 20 epochs of 2000 steps, batch 256:

```bash
CUDA_VISIBLE_DEVICES=0 python -m codec.train \
  --cache-dir data/2nd/training_cache \
  --epochs 20 --steps-per-epoch 2000 --batch-reads 256 \
  --train-fraction 0.9 --validation-max-reads-per-file 5000 \
  --num-layers 4 \
  --output-dir codec/runs/direct_quality_balanced_b256_s40000_v1
```

One complete 18M-read pass: 20 epochs of 3516 steps = 70,320 steps. Rotating
the six extra slots per epoch gives exactly 7032 blocks per file after epoch 20:

```bash
CUDA_VISIBLE_DEVICES=0 python -m codec.train \
  --cache-dir data/2nd/training_cache \
  --epochs 20 --steps-per-epoch 3516 --batch-reads 256 \
  --train-fraction 0.9 --validation-max-reads-per-file 5000 \
  --num-layers 4 \
  --output-dir codec/runs/direct_quality_balanced_b256_fullpass_v1
```
