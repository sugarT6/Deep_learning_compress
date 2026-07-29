# Stage 4: causal Transformer residual model with base motifs

This repository contains the stage-4 FASTQ quality residual entropy model. It
keeps the stage-3 H5 feature construction, read split, training sampler,
physical residual mask, and bit-based evaluation, while replacing the
GRU/TCN sequence backbone with a lightweight causal Transformer. The current
model also uses the already-decoded full DNA read as side information through
a small local motif branch.

The model directly predicts a 189-class residual distribution:

```text
q_hat_i = argmax P0_i(q)
r_i = q_true_i - q_hat_i
P_model(r_i | H5 features at i, decoded history before i)
```

`log P0_r` is an input feature. It is not added to the output logits; the
Transformer output head directly produces the final residual logits.

## Input features

Each position uses exactly these features:

```text
189-dimensional log P0_r(r)
H5 max probability
normalized H5 entropy
normalized H5 expected quality
relative position in the read
normalized read length
q_hat embedding
previous decoded quality embedding
previous decoded residual embedding
Q-mer history embeddings for k = 2,3,4
Residual-mer history embeddings for k = 2,3,4
bidirectional local base context from complete DNA read
```

The first position uses BOS tokens for previous quality and residual. Q/R-mer
tokens use the same bucket definitions, causal history construction, stride-1
hashing, and default vocabulary size 4096 as the stage-3 Q/R-mer experiment.
No token includes the current or a future true quality/residual.

The quality decoder is assumed to have the complete DNA read before quality
decoding starts. Therefore the base branch may use both previous and future
bases without leaking a future quality. Current body length remains explicit
side information for this body-model experiment; its eventual coding cost is
not included in the reported body-quality bits.

## Default model

```text
feature concatenation
  (including three Q-mer and three residual-mer embeddings)
  (including Base Embedding + centered Conv1D kernels 3,5,7 -> 32 dims)
-> Linear + ReLU + Dropout
-> sinusoidal positional encoding
-> 4 causal Transformer encoder layers
-> Linear(d_model -> 189)
-> physical invalid-residual mask
-> softmax P(r_i)
```

Default dimensions:

```text
d_model = 256
num_heads = 4
num_layers = 4
feedforward_dim = 512
context_length = 256 positions, including the current position
dropout = 0.1
Q/R-mer embedding dimension = 8 per window
base embedding dimension = 16
base convolution channels = 16 per kernel
base context dimension = 32
training batch = 64 contiguous reads for the current experiment
```

Q/R-mer token construction is vectorized across all positions in each read.
It preserves the original causal stride-1 hash exactly while avoiding
per-position Python loops during H5 batch construction.

Within each Transformer layer, position `i` directly attends only to positions
`max(0, i-255)..i`. With stacked layers, information can propagate farther
indirectly through earlier token representations. The current token is safe
because it contains current H5 features and shifted history only; it does not
contain `q_true_i` or `r_i`. Reads are right-padded, so a real causal query can
never attend to a later padding token; padded query outputs are excluded from
loss and metrics. Context never crosses read boundaries.

## Data

Place local H5 predictor files under `h5/`, or pass files/directories as
positional arguments. Required datasets are:

```text
/observed
/freqs
/read_offsets
```

By default, each H5 file is split by read id: the first 80% of reads form the
training pool and the last 20% form validation/test. Empty reads are skipped.
The `h5/` and `fq/` directories are intentionally ignored by Git.

### Prepare base sidecars

Training never randomly seeks through gzip FASTQ. First create independent,
Git-ignored sidecars from `fq/`:

```bash
python prepare_base_sidecars.py \
  --fastq-dir fq \
  --output-dir base_sidecars
```

Each sidecar contains:

```text
/base_ids            flat uint8 complete-base stream
/base_read_offsets   boundaries for complete raw reads
/body_lengths        H5 body-quality lengths used for alignment validation
```

Base ids are `A=0,C=1,G=2,T=3,N=4,other=5`; token 6 is reserved for runtime
padding. Preprocessing verifies every read count, sequence/quality length,
H5 quality prefix, and trailing Q2 `#` suffix. It does not modify the original
quality-model H5 files.

## Training

Run from this repository directory:

```bash
CUDA_VISIBLE_DEVICES=0 python train_sequence_residual_transformer.py \
  --epochs 15 \
  --steps-per-epoch 2000 \
  --batch-reads 64 \
  --eval-batch-reads 64 \
  --eval-max-reads-per-file 5000 \
  --num-layers 4 \
  --qmer-ks 2,3,4 \
  --rmer-ks 2,3,4 \
  --base-sidecar-dir base_sidecars \
  --base-conv-kernels 3,5,7 \
  --output-dir runs/transformer_residual_4layer_qrmer_234_baseconv357_b64_e15
```

This run uses 64 reads and 2,000 optimizer updates per epoch. Relative to the
previous 128-read experiment at the same step count, it samples half as many
reads per epoch but preserves the same number of optimizer updates.

Small smoke run:

```bash
python train_sequence_residual_transformer.py \
  --epochs 1 \
  --steps-per-epoch 2 \
  --batch-reads 4 \
  --eval-batch-reads 4 \
  --eval-max-reads-per-file 8 \
  --base-sidecar-dir base_sidecars \
  --device cpu \
  --output-dir runs/baseconv_smoke \
  --no-progress
```

Training writes:

```text
runs/transformer_residual_4layer_qrmer_234_baseconv357_b64_e15/config.json
runs/transformer_residual_4layer_qrmer_234_baseconv357_b64_e15/train_log.csv
runs/transformer_residual_4layer_qrmer_234_baseconv357_b64_e15/best.pt
```

The best checkpoint is selected by the lowest validation
`model_avg_bits_per_quality`.

For a controlled no-mer ablation, pass both `--qmer-ks ''` and `--rmer-ks ''`
and use a separate output directory.

## Prediction and evaluation

```bash
python predict_sequence_residual_transformer.py \
  runs/transformer_residual_4layer_qrmer_234_baseconv357_b64_e15/best.pt \
  --batch-reads 64 \
  --base-sidecar-dir base_sidecars \
  --output-csv runs/transformer_residual_4layer_qrmer_234_baseconv357_b64_e15/predict_metrics.csv
```

The default split is the last 20% test reads. Use `--split all` to evaluate the
whole H5 file. Optional detailed outputs remain compatible with stage 3:

```bash
python predict_sequence_residual_transformer.py \
  runs/transformer_residual_4layer_qrmer_234_baseconv357_b64_e15/best.pt \
  --base-sidecar-dir base_sidecars \
  --sample-predictions runs/transformer_residual_4layer_qrmer_234_baseconv357_b64_e15/samples.csv \
  --quality-prob-log runs/transformer_residual_4layer_qrmer_234_baseconv357_b64_e15/predict_quality_prob.log \
  --quality-prob-log-rows 1000
```

The Q/R-mer and base-branch settings are restored from the checkpoint during
prediction. Old stage-4 checkpoints without a base branch remain loadable and
do not require sidecars.

## Metrics

The primary comparison fields are:

```text
model_avg_bits_per_quality
h5_baseline_avg_bits_per_quality
delta_bits = model_avg_bits - h5_baseline_avg_bits
relative_improvement = (h5_bits - model_bits) / h5_bits
```

For a fair stage-3 comparison, use the same H5 files, read split, number of
training steps, batch size, evaluation read limit, and test symbols. Argmax
accuracy is not the compression objective; the true residual probability is.

## Files

```text
sequence_residual_transformer_model.py  H5 data pipeline and Transformer model
prepare_base_sidecars.py                 FASTQ/base sidecar preprocessing
train_sequence_residual_transformer.py  training and validation
predict_sequence_residual_transformer.py evaluation and prediction logs
requirements.txt                         Python dependencies
```
