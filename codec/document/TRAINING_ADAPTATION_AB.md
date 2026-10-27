# Ordinary / adapter-aware continuation from the expanded20 shared model

`codec.train_adaptation_ab ordinary` and `adaptation` implement a small controlled
continuation experiment. They load the same shared checkpoint weights, reset
the optimizer, and train on the same 20-source manifest. This is weights-only
continuation, not a resume of the old 200k training schedule. No benchmark file
is used for gradient updates, validation or checkpoint selection.

## Common recipe

- Five logging epochs x 2000 = **10,000 additional shared updates**. AdamW,
  weight decay 1e-5, clip 1; shared LR follows a fixed cosine from 1e-5 to 1e-6.
  No validation-driven LR or early stopping changes the budget between modes.
- Preserve the loaded shared model architecture/configuration. Both modes train
  with dropout disabled at runtime (eval mode still permits gradients), matching
  file-fitting/inference and removing a dropout difference between branches.
- Keep platform **episode** weights MGI/Illumina/SOLiD/Ion = 3:5:1:1 and equal
  files within a family. Each episode comes from one training source only.
- Select 256 reads from eight strata across that file's first 90% (one random
  contiguous 32-read chunk per stratum). The original balanced sampler provides
  the file schedule and tail size; its contiguous anchor batch is not a model
  input. Chunks can repeat across episodes. Coverage is therefore not claimed
  to be unique; actual read identities are hashed and actual Q counts recorded.
- Deterministically split half the reads into support and half into query.
  Both modes use the identical actual batches and split with the same seed.
  Padding is excluded from all losses and counts. One-read tails are rejected.

## What A and B optimize

**A / ordinary:** a usual shared-model cross-entropy update using all 256 reads.

**B / adaptation:** independently copy the current shared model at every episode.
Fit only all-layer rank4 Q/K/V/O+FFN LoRA, final norm and a residual64/cross16 head
on support reads. Use Adam, backbone LR .0003, head LR .001, anchor .01 and
16 reads per step. Visit the 128 support reads **once**, giving eight inner
updates (fewer for a tail). No temporary file parameters or inner optimizer
state carry over to another episode.

Then freeze the fitted LoRA factors and extra head branches and compute query
gradients for all shared parameters. Map gradients of parametrized original
matrices to the original shared names. For the adapted norm and linear-head
weights, use the first-order identity approximation back to their shared
initialization. Do not differentiate through support optimizer steps or retain
their graphs. Apply these query gradients to the shared AdamW optimizer. All
temporary head/LoRA variables are discarded. The saved model remains a regular
codec checkpoint with the original linear head and parameter count.

This is a **first-order, short-inner-loop approximation** to adapter-aware
training. It does not unroll the production 2000-head + 10k-LoRA pipeline and
does not claim exact MAML gradients or codec-exact canonical LoRA merging in
its training-time proxy. The usual full file adapter is trained anew when
evaluating the resulting checkpoints.

Both modes expose the same Q tokens once per episode. A uses all of them for
the shared CE gradient; B uses support for the temporary variables and query
for the shared gradient. Inner/backward counters and elapsed training/validation
times are logged separately. **This is matched data exposure and shared update
count, not matched wall time, GPU work or shared-gradient sample count.**

## Validation and which checkpoint to compare

For each training source, select 1024 distinct reads from its final 10%, forming
four fixed episodes with disjoint support/query reads. No validation query is
used in its inner fit; no part of validation updates the shared model. Both
modes use identical fixed episodes, and validation has no effect on training
sampling RNG. Save unadapted and short-adapted float BPQ per file, plus their
20-file macro averages.

- `last.pt`: primary comparison after the same 10k additional shared updates.
- `best.pt`: best **positive-epoch** short-adapted validation macro; an auxiliary
  checkpoint, not proof of lower whole-file compressed size. Step zero is
  recorded in `initial_validation.json` and is never silently returned as best.
- `validation_indices.json`: exact support/query indices for auditing.
- `training_summary.json`, `training_log.jsonl`: data Q counts, inner and outer
  backward symbols, inner step counts, data-footprint hash, source counts and
  timings. Matching seeds/configuration should give equal actual data hashes.

The validation proxy measures few-step adaptation within known source files.
Final evaluation must include held-out files, the standard production file
adaptation budget, floor BPQ and actual quality-plus-adapter bytes. Faster early
adaptation alone is not evidence for a better final compression ratio.

## Commands (different terminals)

Change the visible GPU numbers to available devices. Use separate GPUs for
concurrent jobs, or run them sequentially on one GPU. Each invocation creates
a unique run directory under the specified root; progress stays in the terminal.

```bash
cd /home/share/huadjyin/home/s_qiuping1/tanglei/fastq_quality_residual
CUDA_VISIBLE_DEVICES=0 python -m codec.train_adaptation_ab ordinary \
  --init-from codec/runs/direct_quality_4layer_expanded20_200k_S1pPAq/best.pt \
  --cache-dir data/2nd/training_cache --device cuda \
  --epochs 5 --steps-per-epoch 2000 --batch-reads 256 --episode-blocks 8 \
  --inner-batch-reads 16 --lora-rank 4 \
  --learning-rate 0.00001 --min-learning-rate 0.000001 --seed 20261027 \
  --output-root codec/runs/adaptation_training_ab_20261027
```

```bash
cd /home/share/huadjyin/home/s_qiuping1/tanglei/fastq_quality_residual
CUDA_VISIBLE_DEVICES=1 python -m codec.train_adaptation_ab adaptation \
  --init-from codec/runs/direct_quality_4layer_expanded20_200k_S1pPAq/best.pt \
  --cache-dir data/2nd/training_cache --device cuda \
  --epochs 5 --steps-per-epoch 2000 --batch-reads 256 --episode-blocks 8 \
  --inner-batch-reads 16 --lora-rank 4 \
  --learning-rate 0.00001 --min-learning-rate 0.000001 --seed 20261027 \
  --output-root codec/runs/adaptation_training_ab_20261027
```

Add `--dry-run` to print the configuration without training or allocating a
run. To recover an interrupted A/B run, replace `--init-from` with `--resume`
pointing to its `last.pt`, keep **all original budget/configuration arguments**,
and use a new empty output directory (the CLI creates one). Keep matching
`best.pt` beside the old `last.pt`. Model, optimizer, sampler position, counters
and RNG are restored; the cosine budget cannot be extended through resume.
Do not pass an old ordinary-pretraining checkpoint to this `--resume` entry.
