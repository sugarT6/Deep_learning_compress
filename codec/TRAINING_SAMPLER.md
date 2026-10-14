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
- optimizer state and versioned training state with run configuration,
  learning-rate/early-stopping state, and Python, NumPy, Torch CPU and CUDA RNG
  states.

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
the target epoch count and output location may change. Training-state-v1
checkpoints can enable the new controls when first resumed; patience counters
start fresh against the previous best metric, and the original optimizer LR
and sampler cursor are retained. Subsequent v2 resumes require identical
control configuration and restore their counters/scheduler exactly. Sampler state can
also be serialized/restored mid-epoch via its public state methods; the CLI
currently writes checkpoints only at completed epochs. Reproducible model
updates require the same numerical runtime/hardware as well as RNG state.

## Learning-rate reduction and stopping

Controls are opt-in; old commands keep constant LR and no early stopping.

- `--lr-scheduler plateau`: monitor the same validation macro BPQ used for
  checkpoint selection. With `--lr-patience 2`, tolerate two non-improving
  epochs and halve LR after the third, using `--lr-factor 0.5`.
- `--min-lr 1e-5`: the LR floor. The controls never use the nine comparison
  files to choose LR or stop training.
- `--min-delta 1e-4`: minimum absolute macro BPQ improvement for scheduling
  and optional stopping. Any strict BPQ improvement still updates `best.pt`.
- `--early-stopping-patience 0`: disabled, ensuring the requested step budget
  completes. A positive value stops after that many validations without a
  min-delta improvement. The final completed epoch always saves `last.pt`.

Each log row records LR used and LR for the next epoch, patience count, and
whether stopping occurred. `training_summary.json` records completed epoch,
global steps, parameter count, final LR and best validation macro BPQ.

## Current 80,000-step comparison (run manually)

Both models use the same ten training caches, seed, validation cap, batch 256,
and 2,000-step validation interval. Each completes a total of 80,000 updates
(40 logging epochs), covering the 18M-read training pool once and starting a
second pass. This budget does not by itself establish convergence.

Run commands sequentially on an available GPU; change GPU index 0 as needed.
`mktemp` creates a new empty run directory without overwriting checkpoints,
caches, or user output.

First, continue the existing 4-layer, 2,243,738-parameter model from its
40,000-step checkpoint for **another 40,000 updates**. `--epochs 40` is the
total target, so this runs epochs 21–40, not 40 extra epochs:

```bash
cd /home/share/huadjyin/home/s_qiuping1/tanglei/fastq_quality_residual
RUN4=$(mktemp -d codec/runs/direct_quality_4layer_resume80k_20261014_XXXXXX)
CUDA_VISIBLE_DEVICES=0 python -m codec.train \
  --cache-dir data/2nd/training_cache \
  --resume codec/runs/direct_quality_balanced_b256_s40000_v1/last.pt \
  --epochs 40 --steps-per-epoch 2000 --batch-reads 256 \
  --train-fraction 0.9 --validation-max-reads-per-file 5000 \
  --d-model 256 --num-heads 4 --num-layers 4 --feedforward-dim 512 \
  --context-length 256 --dropout 0.1 \
  --learning-rate 0.001 --weight-decay 1e-5 --grad-clip 1 \
  --lr-scheduler plateau --lr-factor 0.5 --lr-patience 2 \
  --min-lr 1e-5 --min-delta 1e-4 --early-stopping-patience 0 \
  --seed 20260909 --device cuda --output-dir "$RUN4"
```

Then train the chosen **6-layer, 3,297,946-parameter** model from scratch for
80,000 updates. Width 256, FFN 512, heads and features stay the same. Do not
pass the 4-layer checkpoint with `--resume`: the architectures differ.

```bash
cd /home/share/huadjyin/home/s_qiuping1/tanglei/fastq_quality_residual
RUN6=$(mktemp -d codec/runs/direct_quality_6layer_80k_20261014_XXXXXX)
CUDA_VISIBLE_DEVICES=0 python -m codec.train \
  --cache-dir data/2nd/training_cache \
  --epochs 40 --steps-per-epoch 2000 --batch-reads 256 \
  --train-fraction 0.9 --validation-max-reads-per-file 5000 \
  --d-model 256 --num-heads 4 --num-layers 6 --feedforward-dim 512 \
  --context-length 256 --dropout 0.1 \
  --learning-rate 0.001 --weight-decay 1e-5 --grad-clip 1 \
  --lr-scheduler plateau --lr-factor 0.5 --lr-patience 2 \
  --min-lr 1e-5 --min-delta 1e-4 --early-stopping-patience 0 \
  --seed 20260909 --device cuda --output-dir "$RUN6"
```

The old 4-layer run used constant LR for its first 40k updates; the new model
has plateau control from the start. This is a practical continued-training
comparison, not an isolated capacity ablation with identical LR histories.
If necessary after measuring results, train a fresh 4-layer control using
the 6-layer command with only `--num-layers 4` and a new directory.

Both backbones support existing residual/cross-layer head adaptation. The
cross branch still reads block 3; Q/V LoRA remains a four-layer-only experiment
and should not be enabled for the six-layer run. Old adapters are bound to
their original checkpoint SHA and must be fitted again for new weights.
