# Fresh-sample symbol code-length diagnostic

`codec.diagnose_code_lengths` compares saved models without fitting parameters
or compressing the complete file. It reports where quality code length is spent
and where the saved Full model improves or loses relative to saved suffix LoRA.

The current command compares the unadapted expanded20 base, all-four-layer rank4
LoRA, and the completed Full diagnostic. The original prefix-only warm head was
not archived. The head embedded in the final LoRA artifact was jointly updated
with LoRA; it cannot stand in for that historical warm head. This tool does not
retrain or silently substitute a head-only baseline.

## Current CNR0847462 command

Run on an available GPU; replace `0` as appropriate. All model and source paths
are verified from the completed diagnostic manifest. No H5 is needed. Each call
creates a unique directory and keeps sampling/scoring progress on the terminal.

```bash
cd /home/share/huadjyin/home/s_qiuping1/tanglei/fastq_quality_residual
CUDA_VISIBLE_DEVICES=0 python -m codec.diagnose_code_lengths \
  codec/output/adaptation_diagnostic_20261021/CNR0847462_1_full_z7o0ba9m \
  --adapter codec/output/CNR0847462_last4_rank4_10k_somLvU/head.adapter.json \
  --device cuda --reads 8192 --blocks 8 --batch-reads 256 \
  --seed 20261027 --bootstrap-samples 1000 \
  --output-root codec/output/code_length_diagnostic_20261027
```

There is one sequential FASTQ scan to select the fresh sample, followed by
inference on that same sample for each saved model. It does not run 10k
adaptation steps or arithmetic coding. The final terminal line is the new
absolute output directory.

## Sample isolation and identity

- Exclude the entire original warm-up prefix and both reservoir train and
  validation reads. The latter participated in selecting the saved model step.
- Verify source size/mtime, base SHA, Full checkpoint identity, split hashes,
  and exact agreement of Full/LoRA reservoir reports. The adapter must be a v8
  suffix adapter from the same source and shared checkpoint, without read labels.
- Divide the whole file into eight equal read-index blocks, then sample 1024
  eligible reads uniformly without replacement inside each block. Sampling is
  deterministic and independent of model predictions. Short/empty/changed
  sources cannot silently produce a smaller sample.
- Header content is neither a feature nor a selection criterion. File read index
  is used only for sampling and analysis, never fed to the model.

## Outputs

| File | Contents |
|---|---|
| `summary.md` | Model BPQ, paired uncertainty, largest code-length contributions and Full gains by group. |
| `summary.json` | Full machine-readable metrics, provenance and interpretation limits. |
| `grouped_code_lengths.csv` | Every cycle, actual Q, transition category and file block, for all three models. Includes total bits, mean BPQ, sample bit share, and signed gain against LoRA. |
| `per_read.csv` | Read index, block, length, mean Q, large quality drops, all model code lengths and paired gains. |
| `per_symbol.npz` | Every active sampled Q: original read index, cycle, Q, signed Q change, groups, and float/floor bit lengths for each model. Read offsets allow further analysis without another model pass. |
| `sample_indices.npz`, `manifest.json` | Exact selected read indices, source/model identities, exclusions and sampling configuration. |

`read_index` is zero-based. Cycles and file blocks are one-based. Transition
labels are `first`, `unchanged`, `drop_1_to_4`, `drop_5_plus`, `rise_1_to_4`, and
`rise_5_plus`; change means current true Q minus previous true Q. True-label
groups are post-prediction analysis, not extra model inputs. Padding never
enters counts. NPZ rows are read-major rather than the codec's cycle-major wire
order; every active position still has its correct causal prediction.

Positive `gain_bits_vs_lora` / `gain_bpq_vs_lora` means fewer bits than saved
LoRA. Each dimension partitions all sampled symbols, so total group bits and
gains sum back to the model sample totals. Groups are overlapping across
dimensions; do not sum cycle and quality tables together.

Float lengths use CPU float64 softmax. Floor lengths use existing v2 selected
intervals with positive frequencies and the **actual row total**, not a fixed
65536 denominator. No coder framing or parameter storage is included in BPQ.
LoRA's actual container adapter bytes are recorded separately. Full `.pt` size
is a diagnostic file size, not a deployed adapter cost.

Bootstrap intervals resample whole reads in paired fashion within each file
block; they do not treat Q positions as independent samples. They do not
include training-seed variability and may understate cross-read dependence.
The aggregate is the sampled-symbol mean: equal block quotas after exclusions
do not guarantee a population-weighted whole-file estimate. This repeatedly
studied file is not an unseen-file test. Scores do not establish an entropy
limit, causal module contribution, or complete-file byte savings.
