# ABC 模型扩容预训练（2026-10-21）

三组均在 expanded20 混合训练集上从随机初始化开始训练，比较扩大深度或宽度的效果。
不载入现有预训练模型，也不走单文件从零训练路线。

| 版本 | 命令名 | 层数 | 隐藏维度 | attention heads | FFN 维度 | 基础模型参数量 |
|---|---|---:|---:|---:|---:|---:|
| 已有基线 | — | 4 | 256 | 4 | 512 | 2,243,738 |
| A | `depth6` | 6 | 256 | 4 | 512 | 3,297,946 |
| B | `depth8` | 8 | 256 | 4 | 512 | 4,352,154 |
| C | `wide512` | 4 | 512 | 8 | 1024 | 8,577,178 |

A、B 保持其余维度，形成 4→6→8 层对照；C 与已有的四层 D256、D384 模型形成宽度对照。
三组均沿用 GELU、局部碱基 CNN、context256、dropout0.1 和原输入特征，不加入 read 状态。

## 分别在三个终端启动

先激活此前使用的训练环境。三条命令各自完整，按实际空闲 GPU 修改
`CUDA_VISIBLE_DEVICES=0`；代码不会自动后台启动其他实验。

A：六层 D256 / FF512。

```bash
cd /home/share/huadjyin/home/s_qiuping1/tanglei/fastq_quality_residual
CUDA_VISIBLE_DEVICES=0 python -m codec.train_experiments depth6 --device cuda
```

B：八层 D256 / FF512。

```bash
cd /home/share/huadjyin/home/s_qiuping1/tanglei/fastq_quality_residual
CUDA_VISIBLE_DEVICES=0 python -m codec.train_experiments depth8 --device cuda
```

C：四层 D512 / FF1024。

```bash
cd /home/share/huadjyin/home/s_qiuping1/tanglei/fastq_quality_residual
CUDA_VISIBLE_DEVICES=0 python -m codec.train_experiments wide512 --device cuda
```

## 数据和训练控制

三组继承原 expanded20 训练设置：

- 直接读取原训练缓存目录
  `/home/share/huadjyin/home/s_qiuping1/tanglei/fastq_quality_residual/data/2nd/training_cache/`，
  明确使用原 10 源和新增 10 源，不复制数据，也不扫描加入其他文件。
- 每文件前 90% 为训练集、后 10% 为验证集；验证每源最多 5000 条 reads，
  按 20 源 macro BPQ 选择 `best.pt`，留出测试文件不进入训练集。
- 平台 batch 配比 3:5:1:1，组内文件均衡；batch256、seed20260909。
- 100 个 logging epoch × 2000 步 = 200,000 次更新；这里的 100 个 epoch
  **不是完整遍历整个训练数据池 100 遍**。
- AdamW，初始学习率 0.001、weight decay 1e-5、梯度裁剪 1；
  plateau 学习率调度 factor0.5、patience2、最低学习率 1e-5、min delta1e-4；
  不自动早停。

相同更新步数用于比较既定训练预算下的表现，不代表训练耗时相同。
如果某组到结束时验证 BPQ 仍明显下降，不能据此认定该规模已充分训练或收益已经封顶。

## 结果目录

这三组默认统一放在新的目录，原五组结构实验的默认目录保持不变：

```text
/home/share/huadjyin/home/s_qiuping1/tanglei/fastq_quality_residual/codec/runs/model_scaling_20261021/
  depth6_<随机后缀>/
  depth8_<随机后缀>/
  wide512_<随机后缀>/
```

每次启动都生成独立的新子目录，同一组同时或重复启动也不会覆盖旧结果。
终端开始时会打印完整路径、模型参数量和目标更新步数。各子目录沿用原训练输出：
`best.pt`、`last.pt`、`run_config.json`、`training_log.jsonl`、
`training_corpus.json`、`sampling_statistics.json`、`training_summary.json`。

可在正式启动前检查配置而不训练：

```bash
cd /home/share/huadjyin/home/s_qiuping1/tanglei/fastq_quality_residual
python -m codec.train_experiments depth6 --dry-run
```

训练完成后以各自 `best.pt` 比较压缩效果。LoRA 现已支持覆盖模型全部 Transformer 层，
包括六层、八层模型；适配器仍使用 v8 协议，dense v7 的最多四层限制保持不变。
最后仍以 `quality_and_adapter_bytes` 判断实际收益，扩大模型后适配器成本也需要计入。

## 全层 rank8 LoRA 压缩对照（2026-10-22）

下面三个命令可分别在独立终端执行，按空闲 GPU 修改 `CUDA_VISIBLE_DEVICES=0`。
同样测试 CNR0847462，保持原先输出头 warm-up 2000 步、主干适配上限 10000 步、
抽样、学习率和随机种子，不加入 read 状态；只把适配层数分别改成 6、8、4。
每次自动建立独立输出目录，不覆盖末两层对照结果。

“全层 LoRA”覆盖所有 Transformer 层的 Q/K/V/O 和两层 FFN，另训练最终
LayerNorm 和原输出头；不是 Full 全参数微调。原 cross-layer 输出头继续从
第 3 个 Transformer block 取特征，不随模型深度调整。
相同 rank 下增加适配层数会增加适配器预算，因此最终比较含实际适配器的
`quality_and_adapter_bytes`，原四层末两层 rank8 参考值为 **75,179,664 字节**。
原四层全层 rank4 的 **74,975,072 字节**可作补充参考，但 rank 和参数预算不同。

A：六层 D256，全部六层 rank8。

```bash
cd /home/share/huadjyin/home/s_qiuping1/tanglei/fastq_quality_residual
TEST_RUN=$(mktemp -d /home/share/huadjyin/home/s_qiuping1/tanglei/fastq_quality_residual/codec/output/CNR0847462_A_depth6_all6_rank8_10k_XXXXXX)

CUDA_VISIBLE_DEVICES=0 python -m codec.encode \
  /home/share/huadjyin/home/s_qiuping1/tanglei/fastq_quality_residual/data/2nd/CNR0847462_1.head2M.fastq.gz \
  "$TEST_RUN/CNR0847462.fqdc" \
  /home/share/huadjyin/home/s_qiuping1/tanglei/fastq_quality_residual/codec/runs/model_scaling_20261021/depth6_s_fnb3se/best.pt \
  --device cuda --batch-reads 256 --quantizer floor --coder-backend native \
  --finetune-head --head-type residual --head-residual-dim 64 \
  --head-cross-layer --head-cross-dim 16 \
  --head-max-reads 8000 --head-max-symbols 1200000 --head-steps 2000 \
  --head-max-seconds 0 --head-seed 20260921 \
  --finetune-last-layers 6 --backbone-lora-rank 8 \
  --backbone-steps 10000 --backbone-batch-reads 16 \
  --backbone-learning-rate 0.0003 --backbone-eval-every 200 \
  --backbone-patience 6 --backbone-max-reads 32000 --backbone-max-symbols 4800000 \
  --save-head-adapter "$TEST_RUN/head.adapter.json" \
  > "$TEST_RUN/metrics.json"

cat "$TEST_RUN/metrics.json"
```

B：八层 D256，全部八层 rank8。

```bash
cd /home/share/huadjyin/home/s_qiuping1/tanglei/fastq_quality_residual
TEST_RUN=$(mktemp -d /home/share/huadjyin/home/s_qiuping1/tanglei/fastq_quality_residual/codec/output/CNR0847462_B_depth8_all8_rank8_10k_XXXXXX)

CUDA_VISIBLE_DEVICES=0 python -m codec.encode \
  /home/share/huadjyin/home/s_qiuping1/tanglei/fastq_quality_residual/data/2nd/CNR0847462_1.head2M.fastq.gz \
  "$TEST_RUN/CNR0847462.fqdc" \
  /home/share/huadjyin/home/s_qiuping1/tanglei/fastq_quality_residual/codec/runs/model_scaling_20261021/depth8_13od_1x1/best.pt \
  --device cuda --batch-reads 256 --quantizer floor --coder-backend native \
  --finetune-head --head-type residual --head-residual-dim 64 \
  --head-cross-layer --head-cross-dim 16 \
  --head-max-reads 8000 --head-max-symbols 1200000 --head-steps 2000 \
  --head-max-seconds 0 --head-seed 20260921 \
  --finetune-last-layers 8 --backbone-lora-rank 8 \
  --backbone-steps 10000 --backbone-batch-reads 16 \
  --backbone-learning-rate 0.0003 --backbone-eval-every 200 \
  --backbone-patience 6 --backbone-max-reads 32000 --backbone-max-symbols 4800000 \
  --save-head-adapter "$TEST_RUN/head.adapter.json" \
  > "$TEST_RUN/metrics.json"

cat "$TEST_RUN/metrics.json"
```

C：四层 D512，全部四层 rank8。

```bash
cd /home/share/huadjyin/home/s_qiuping1/tanglei/fastq_quality_residual
TEST_RUN=$(mktemp -d /home/share/huadjyin/home/s_qiuping1/tanglei/fastq_quality_residual/codec/output/CNR0847462_C_wide512_all4_rank8_10k_XXXXXX)

CUDA_VISIBLE_DEVICES=0 python -m codec.encode \
  /home/share/huadjyin/home/s_qiuping1/tanglei/fastq_quality_residual/data/2nd/CNR0847462_1.head2M.fastq.gz \
  "$TEST_RUN/CNR0847462.fqdc" \
  /home/share/huadjyin/home/s_qiuping1/tanglei/fastq_quality_residual/codec/runs/model_scaling_20261021/wide512_khuefi70/best.pt \
  --device cuda --batch-reads 256 --quantizer floor --coder-backend native \
  --finetune-head --head-type residual --head-residual-dim 64 \
  --head-cross-layer --head-cross-dim 16 \
  --head-max-reads 8000 --head-max-symbols 1200000 --head-steps 2000 \
  --head-max-seconds 0 --head-seed 20260921 \
  --finetune-last-layers 4 --backbone-lora-rank 8 \
  --backbone-steps 10000 --backbone-batch-reads 16 \
  --backbone-learning-rate 0.0003 --backbone-eval-every 200 \
  --backbone-patience 6 --backbone-max-reads 32000 --backbone-max-symbols 4800000 \
  --save-head-adapter "$TEST_RUN/head.adapter.json" \
  > "$TEST_RUN/metrics.json"

cat "$TEST_RUN/metrics.json"
```
