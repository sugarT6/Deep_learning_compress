# Four-layer Transformer: original vs quality-mode sampling

This experiment changes training-source probabilities, not the model, input
features, loss, or codec protocol. Both arms start from the same original
four-layer expanded20 checkpoint and reset their optimizer. It is ordinary
shared-model continuation, without LoRA, an inner adaptation loop, or new layers.

## The two data mixtures

`original` retains the nominal MGI/Illumina/SOLiD/Ion **3:5:1:1** batch-source
mixture, with equal files inside a family. Only the explicit expanded20 training
manifest is opened. Cache-directory globbing is never used.

`quality` mixes **50% original probabilities + 50% quality-mode probabilities**.
The mode component initially assigns equal mass to four quality groups and equal
files inside a group. A cap limits each final source probability to **2x its
original probability**. Any clipped mass is redistributed proportionally among
uncapped mode sources; the original 50% component is retained. This prevents a
single-file group from absorbing an arbitrarily large fraction of training.

These are a fixed diversity hypothesis, not DoReMi or weights optimized for
CNR0847462. Mode balance need not improve every file. All final weights are
recorded; do not infer compression gains from the profile alone.

The groups come only from the first 90% of the 20 training files. For each file,
4096 reads are taken from centered, disjoint chunks in 16 training-prefix strata.
The 68 profile features are:

- 42 Q frequencies, square-root transformed, distance-block weight 0.5;
- six within-read absolute-change frequencies: 0, 1, 2, 3–4, 5–9, >=10;
  square-root transformed, block weight 0.25;
- ten relative-cycle means (divided by 41) and ten Q<20 fractions,
  with block weight 0.25. Empty cycle bins use that source's global statistic.

Clustering is deterministic Lloyd clustering with farthest-first seeding. Every
training source is tried as the initial center; the smallest within-profile
squared error wins. No model loss, validation labels, target file, header, or
benchmark result selects the grouping. Identical profiles may produce fewer
than the requested four nonempty groups; the actual groups are reported.

## Fairness and accounting

Both arms use the same cumulative-deficit source-quota allocator and shuffled
schedule. The control preserves the old source proportions; it does not replay
the historical 200k trajectory. Within each source, both reuse the original
shuffled, nonoverlapping 256-read block sampler, including tail blocks.

Default budget: **10,000 optimizer updates x 32,768 loss-Q = 327,680,000 Q**.
Each update accumulates summed cross entropy divided by exactly 32,768. A batch
crossing an update boundary carries its unscored targets into the next update.
Loss targets are consumed in read-major order; preceding Q context, cycle,
length, and the active/padding mask remain intact. Unselected whole reads are
trimmed from that fragment's forward pass. Padding never enters the loss.

Thus both optimizer-update count and effective loss-Q match exactly. Total
forward context, wall time, source identities and unique-read coverage can
differ; these are recorded rather than described as matched. Sampler coverage
counts drawn reads, including an unfinished pending batch. The stream records
effective loss-Q per source and a hash of the actual loss targets.

Both arms keep the checkpoint's configured dropout during training, use AdamW
(LR cosine 1e-5 to 1e-6, weight decay 1e-5, clip 1), and run the full budget
without validation-driven LR changes or early stopping. This recipe is shared
between these two arms, not identical to the earlier adaptation-training A/B.

## Prepare once, then run in separate terminals

The current workspace already has
`codec/output/quality_sampling_ab_20261028/quality_modes.json` prepared from
training caches only. **Reuse it; do not rerun prepare at the same path.**
For a fresh checkout/cache, prepare a new profile path:

```bash
cd /home/share/huadjyin/home/s_qiuping1/tanglei/fastq_quality_residual
python -m codec.train_sampling_ab prepare \
  --cache-dir data/2nd/training_cache \
  --profile-out codec/output/quality_sampling_ab_20261028/quality_modes.json
```

The profile binds cache identities, sampled-read ranges, content hashes, groups
and final probabilities. Existing output is refused. Replacing a cache or
changing the train split requires a newly prepared profile, not silent reuse.

Use available GPU numbers; two jobs may run sequentially on one GPU. Each command
creates a unique run directory and leaves progress visible in the terminal.

```bash
cd /home/share/huadjyin/home/s_qiuping1/tanglei/fastq_quality_residual
CUDA_VISIBLE_DEVICES=0 python -m codec.train_sampling_ab original \
  --init-from codec/runs/direct_quality_4layer_expanded20_200k_S1pPAq/best.pt \
  --profile codec/output/quality_sampling_ab_20261028/quality_modes.json \
  --cache-dir data/2nd/training_cache --device cuda \
  --updates 10000 --quality-tokens-per-update 32768 --eval-every 2000 \
  --output-root codec/runs/quality_sampling_ab_20261028
```

```bash
cd /home/share/huadjyin/home/s_qiuping1/tanglei/fastq_quality_residual
CUDA_VISIBLE_DEVICES=1 python -m codec.train_sampling_ab quality \
  --init-from codec/runs/direct_quality_4layer_expanded20_200k_S1pPAq/best.pt \
  --profile codec/output/quality_sampling_ab_20261028/quality_modes.json \
  --cache-dir data/2nd/training_cache --device cuda \
  --updates 10000 --quality-tokens-per-update 32768 --eval-every 2000 \
  --output-root codec/runs/quality_sampling_ab_20261028
```

Add `--dry-run` to inspect configuration without loading a model/cache or using
a GPU. Interrupted runs can use `--resume RUN/last.pt` in place of `--init-from`,
with all original arguments unchanged, a new output directory, and matching
`best.pt` beside the old `last.pt`. Resume restores optimizer/RNG, source block
permutations, pending target offset and token counters. Budget extension, mode
switches and changed profile/configuration are rejected.

## Read the results

- **`last.pt` is the primary comparison**, after identical Q/update budgets.
- `best.pt` is the best positive evaluation point, an auxiliary source-validation
  checkpoint. Step-zero metrics are saved but never substituted for training.
- `training_summary.json`, `training_log.jsonl`, `sampling_statistics.json`,
  `quality_modes.json` and `run_config.json` record the evidence and recipe.
- Validation every 2000 updates uses the first 1024 reads of each training
  source's last 10%, identically in both arms. These are unadapted float BPQ,
  not CNR0847462 compression results or a new independent test.

```bash
python -m codec.train_sampling_ab compare /absolute/original_run /absolute/quality_run
```

This checks the recipe, complete token/update budget and reports source-level
BPQ differences. Final conclusions still require identical file adaptation on
predeclared development files, with quantized BPQ, model cost and intended file
scale reported separately. Existing consumed test1/2/3 must not be reused as new
independent tests. This command never runs complete-file compression.
