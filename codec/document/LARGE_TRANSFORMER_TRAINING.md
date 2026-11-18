# Large quality Transformer training

`python -m codec.train_large_transformer` is a separate distributed trainer. It
leaves the existing single-device trainer, models and codec protocols unchanged.

The default configuration in `codec/configs/transformer_500m.json` has 27 layers,
width 1536, 24 attention heads and GELU FFN width 3072: **510,353,050 parameters**.
It retains the original 75 input features, 42 quality classes, dropout 0.1 and
256-position causal window. Each layer receives an independent draw from the
standard PyTorch initialization distributions. No residual64/cross16 adapter or
additional quality feature is installed during pretraining.

The training set is the explicit expanded20 manifest, with its existing split,
platform batch weights and shuffled block sampler. The default 100 logging
epochs contain 2000 optimizer updates each. An epoch is not a full corpus pass.
The global batch of 256 reads is shared across all ranks; each rank accumulates
microbatches of 16 reads. Gradients are weighted by the actual global active-Q
count, including unequal and empty tail shards.

## Attention and precision

`--attention-backend flash2` uses `flash-attn` 2.5.9.post1 with its native local window
`(255, 0)` and causal masking. This is the preferred backend for the full training
set, which includes reads of up to 386 positions. It requires CUDA BF16 and fails
if the dependency or its window API is unavailable.

The dependency-free default `--attention-backend flash` uses PyTorch SDPA with
only its FlashAttention backend enabled. For reads longer than 256, it batches
individual query windows after the causal prefix, retaining the exact mask.
`--attention-query-chunk-size` bounds those temporary windows. This path can be
substantially slower on long reads; measure it before using it for a full run.
Neither backend enlarges the window, truncates reads or silently falls back to
dense training attention.

Master weights, AdamW state and saved weights remain FP32. Activation checkpoint
recomputation preserves dropout RNG. Validation uses the original FP32 attention
and fixed validation prefix per source, reduced across ranks. The best checkpoint
minimizes the 20-source macro bits/Q. Saved state-dict names and model schemas
remain compatible with `codec.checkpoint.load_training_checkpoint` and the
ordinary compressor and decompressor; FlashAttention is a training execution
choice, not an extra decoding dependency.

## Run

From the repository root, check data/configuration without allocating the model:

```bash
python -m codec.train_large_transformer --dry-run \
  --cache-dir /path/to/training_cache --preflight-output /path/to/preflight.json
```

Install the optional native-window backend into an isolated directory using a
compiler compatible with the installed CUDA toolkit. Keep the existing PyTorch
installation and dependencies:

```bash
python -m pip install --no-build-isolation --no-deps \
  --target /path/to/isolated_flash2 -r codec/requirements-flash2.txt
```

Run one process per scheduler-allocated GPU, preserving `CUDA_VISIBLE_DEVICES`:

```bash
PYTHONPATH=/path/to/isolated_flash2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m torch.distributed.run --standalone --nproc_per_node=3 \
  -m codec.train_large_transformer --device cuda \
  --attention-backend flash2 \
  --cache-dir /path/to/training_cache --output-dir /path/to/new_run
```

Defaults are AdamW LR 0.0002, 2000 warmup updates, weight decay 0.00001, gradient
clip 1, validation-driven LR reduction by 0.5 with patience 2 and floor 0.00001,
and no early stopping. Progress is written to stderr; do not discard that stream.
For dsub, request one task with three GPUs, rather than three task replicas.

`best.pt` and `last.pt` include model/optimizer, sampler, learning-rate controls
and every rank's RNG state. Resume from a completed `last.pt` with its matching
`best.pt` beside it, the same world size and training configuration, a new output
directory, and a larger cumulative `--epochs` target. Changing execution or
sampling settings is rejected rather than silently changing the continuation.

Tests in `tests/test_large_transformer.py` and
`tests/test_large_transformer_training.py` cover local-mask values and gradients,
causality, checkpoint-compatible FP32 evaluation, activation recomputation,
three-rank global-gradient equivalence and resume continuity. A real A100
preflight is still required to establish fused kernel use and full-size memory.
