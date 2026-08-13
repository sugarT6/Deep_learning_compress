# Q-hat-conditioned direct-quality Transformer

This repository now tests direct 95-class quality prediction while retaining
the feature set that produced the best qhat-only residual result. Unlike
`../fastq_quality_direct`, this experiment still reads the quality-model H5,
uses `q_hat`, and retains decoded residual history and residual-mer features.

The current default target is the true quality id:

```text
q_hat_i = argmax P0_i(q)
r_j = q_j - q_hat_j, for already decoded positions j < i
P_model(q_true_i | q_hat_i, position, bases, decoded q/r history before i)
```

Three prior-feature modes are available:

```text
qhat_only (default): use the matrix only for q_hat and H5 baseline
compact_prior:       additionally input three compact distribution summaries
full_prior:          additionally input log P0_r and probability summaries
```

`compact_prior` adds the top-1 probability, the natural-log probability margin
between the top-1 and top-2 classes, and entropy normalized by `log(95)`. Its
five continuous inputs are those three summaries, relative position, and
normalized read length. This mode preserves the direct-quality target and all
causal history/base features while testing whether a small amount of prior
shape information recovers signal discarded by `qhat_only`.

`full_prior` reproduces the previous input. It normalizes H5 counts into
`P0(q)`, clamps probabilities to `1e-12`, and applies the natural logarithm
before mapping them into the 189 residual classes. `log_p0_plus_delta` remains
available only for historical `full_prior + residual` runs; direct 95-class
quality logits are the current default.

## Input features

The current `qhat_only` experiment uses:

```text
relative position in the read
normalized read length
q_hat embedding
previous decoded quality embedding
previous decoded residual embedding
Q-mer history embeddings for k = 2,3,4
Residual-mer history embeddings for k = 2,3,4
bidirectional local base context from complete DNA read
```

`compact_prior` adds top-1 confidence, top-1/top-2 log-probability margin, and
normalized H5 entropy. `full_prior` adds 189-dimensional `log P0_r(r)`, H5 max
probability, normalized H5 entropy, and normalized H5 expected quality.
Exact-lag support remains for historical checkpoint compatibility.

An optional file-level platform embedding is independent of the prior mode.
Training requires an explicit mapping for every input file; the resolved
basename-to-platform name and stable platform id are stored in the checkpoint,
so prediction restores them automatically. The current platform vocabulary is
`BGISEQ=0`, `Illumina=1`, and `IonTorrent=2`. The platform value must be
available as file/container metadata to both the encoder and decoder.

Missing history at the start of a read uses BOS tokens. Q/R-mer tokens use the
same bucket definitions, causal history construction, stride-1 hashing, and
default vocabulary size 4096 as the stage-3 Q/R-mer experiment. No token
includes the current or a future true quality/residual.

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
-> Linear(d_model -> 95)
-> softmax P(q_i)
```

All 95 output classes are legal quality ids, so the direct-quality target does
not use the old q_hat-dependent invalid-residual mask. Historical residual
checkpoints remain loadable and still apply that mask.

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
positional arguments. Directory discovery accepts standard SRA run accessions
(`SRR` or `ERR`) ending in `.qual_model.h5`; this excludes unrelated
HiFi/Nanopore files in the same directory. Required datasets are:

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

The current six-dataset run discovers the original five SRR files plus
`ERR2755197_1.block.fq.gz.qual_model.h5` from `h5/`. Use a new output directory
so the five-dataset best checkpoint remains an unchanged comparison baseline.

Run from this repository directory:

```bash
CUDA_VISIBLE_DEVICES=0 python train_sequence_residual_transformer.py \
  --epochs 15 \
  --steps-per-epoch 2000 \
  --batch-reads 64 \
  --eval-batch-reads 64 \
  --eval-max-reads-per-file 5000 \
  --num-layers 4 \
  --prediction-target quality \
  --prior-feature-mode qhat_only \
  --qmer-ks 2,3,4 \
  --rmer-ks 2,3,4 \
  --base-sidecar-dir base_sidecars \
  --base-conv-kernels 3,5,7 \
  --output-parameterization direct_logits \
  --output-dir runs/transformer_quality_4layer_qhatonly_qrmer234_baseconv357_b64_e15_srr5_err2755197
```

This run uses 64 reads and 2,000 optimizer updates per epoch. Relative to the
previous 128-read experiment at the same step count, it samples half as many
reads per epoch but preserves the same number of optimizer updates.

To test the three compact prior summaries against the current six-dataset
2,600-step baseline while keeping every other setting fixed:

```bash
CUDA_VISIBLE_DEVICES=0 python train_sequence_residual_transformer.py \
  h5 \
  --epochs 15 \
  --steps-per-epoch 2600 \
  --batch-reads 64 \
  --eval-batch-reads 64 \
  --eval-max-reads-per-file 5000 \
  --num-layers 4 \
  --prediction-target quality \
  --prior-feature-mode compact_prior \
  --qmer-ks 2,3,4 \
  --rmer-ks 2,3,4 \
  --base-sidecar-dir base_sidecars \
  --base-conv-kernels 3,5,7 \
  --output-parameterization direct_logits \
  --output-dir runs/transformer_quality_4layer_compactprior_qrmer234_baseconv357_b64_e15_s2600_srr5_err2755197
```

Then run the full held-out prediction:

```bash
CUDA_VISIBLE_DEVICES=0 python predict_sequence_residual_transformer.py \
  runs/transformer_quality_4layer_compactprior_qrmer234_baseconv357_b64_e15_s2600_srr5_err2755197/best.pt \
  h5 \
  --batch-reads 64 \
  --base-sidecar-dir base_sidecars \
  --output-csv runs/transformer_quality_4layer_compactprior_qrmer234_baseconv357_b64_e15_s2600_srr5_err2755197/predict_metrics.csv
```

To test only an 8-dimensional platform embedding on the `qhat_only` baseline:

```bash
CUDA_VISIBLE_DEVICES=0 python train_sequence_residual_transformer.py \
  h5 \
  --epochs 15 \
  --steps-per-epoch 2600 \
  --batch-reads 64 \
  --eval-batch-reads 64 \
  --eval-max-reads-per-file 5000 \
  --num-layers 4 \
  --prediction-target quality \
  --prior-feature-mode qhat_only \
  --platform-embed-dim 8 \
  --platform-map ERR2755197=BGISEQ,SRR1238539=IonTorrent,SRR3066199=Illumina,SRR5867380=IonTorrent,SRR622457=Illumina,SRR6691666=Illumina \
  --qmer-ks 2,3,4 \
  --rmer-ks 2,3,4 \
  --base-sidecar-dir base_sidecars \
  --base-conv-kernels 3,5,7 \
  --output-parameterization direct_logits \
  --output-dir runs/transformer_quality_4layer_qhatonly_platform8_qrmer234_baseconv357_b64_e15_s2600_srr5_err2755197
```

Prediction reads the mapping from the checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 python predict_sequence_residual_transformer.py \
  runs/transformer_quality_4layer_qhatonly_platform8_qrmer234_baseconv357_b64_e15_s2600_srr5_err2755197/best.pt \
  h5 \
  --batch-reads 64 \
  --base-sidecar-dir base_sidecars \
  --output-csv runs/transformer_quality_4layer_qhatonly_platform8_qrmer234_baseconv357_b64_e15_s2600_srr5_err2755197/predict_metrics.csv
```

To reproduce the previous qhat-only residual target with the same code, change
the target, output parameterization, and output directory:

```bash
CUDA_VISIBLE_DEVICES=0 python train_sequence_residual_transformer.py \
  --epochs 15 \
  --steps-per-epoch 2000 \
  --batch-reads 64 \
  --eval-batch-reads 64 \
  --eval-max-reads-per-file 5000 \
  --num-layers 4 \
  --prediction-target residual \
  --prior-feature-mode qhat_only \
  --qmer-ks 2,3,4 \
  --rmer-ks 2,3,4 \
  --base-sidecar-dir base_sidecars \
  --base-conv-kernels 3,5,7 \
  --output-parameterization direct_residual_logits \
  --output-dir runs/transformer_residual_4layer_qhatonly_qrmer234_baseconv357_b64_e15
```

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
runs/transformer_quality_4layer_qhatonly_qrmer234_baseconv357_b64_e15_srr5_err2755197/config.json
runs/transformer_quality_4layer_qhatonly_qrmer234_baseconv357_b64_e15_srr5_err2755197/train_log.csv
runs/transformer_quality_4layer_qhatonly_qrmer234_baseconv357_b64_e15_srr5_err2755197/best.pt
```

The training CSV and terminal show all floating-point values with four decimal
places. `train_loss` is the epoch mean cross-entropy in nats over valid quality
symbols; `train_bits_per_quality = train_loss / ln(2)`. The best checkpoint is
selected by the lowest validation `model_avg_bits_per_quality`.

For a controlled no-mer ablation, pass both `--qmer-ks ''` and `--rmer-ks ''`
and use a separate output directory.

## Prediction and evaluation

```bash
python predict_sequence_residual_transformer.py \
  runs/transformer_quality_4layer_qhatonly_qrmer234_baseconv357_b64_e15_srr5_err2755197/best.pt \
  --batch-reads 64 \
  --base-sidecar-dir base_sidecars \
  --output-csv runs/transformer_quality_4layer_qhatonly_qrmer234_baseconv357_b64_e15_srr5_err2755197/predict_metrics.csv
```

`--output-csv` normally accepts the full CSV filename. If an existing
directory is passed instead, the script writes `predict_metrics.csv` inside
that directory.

The default split is the last 20% test reads. Use `--split all` to evaluate the
whole H5 file. Optional detailed outputs remain compatible with stage 3:

```bash
python predict_sequence_residual_transformer.py \
  runs/transformer_quality_4layer_qhatonly_qrmer234_baseconv357_b64_e15_srr5_err2755197/best.pt \
  --base-sidecar-dir base_sidecars \
  --sample-predictions runs/transformer_quality_4layer_qhatonly_qrmer234_baseconv357_b64_e15_srr5_err2755197/samples.csv \
  --quality-prob-log runs/transformer_quality_4layer_qhatonly_qrmer234_baseconv357_b64_e15_srr5_err2755197/predict_quality_prob.log \
  --quality-prob-log-rows 1000
```

The prediction target, prior-feature, historical exact-lag,
output-parameterization, Q/R-mer, and base-branch settings are restored from
the checkpoint; no prediction-side switch is needed. Exact-lag support remains
only for loading the completed ablation checkpoint and is disabled in current
training. Old checkpoints without `prediction_target` are interpreted as
residual checkpoints; those without `output_parameterization` default to
`direct_residual_logits`. Old stage-4 checkpoints without a base branch remain
loadable and do not require sidecars. Prediction CSVs, optional sample outputs,
compact probability logs, and terminal floating-point metrics use four decimal
places. `predict_metrics.csv` records `elapsed_seconds` for each input file;
the terminal also prints each file's time and their summed total.

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
accuracy is not the compression objective; the true quality probability is.

## Files

```text
sequence_residual_transformer_model.py  H5 data pipeline and Transformer model
prepare_base_sidecars.py                 FASTQ/base sidecar preprocessing
train_sequence_residual_transformer.py  training and validation
predict_sequence_residual_transformer.py evaluation and prediction logs
requirements.txt                         Python dependencies
```
