# 最简单的全局 rank 编码

`--quality-coding rank` 保持预测模型不变，将真实 Q 转为它在模型预测中的零基名次，
用全文件的一张固定 rank 频率表进行 range 编码。没有置信度分组、在线更新或概率修正，
不训练新网络。频率表随压缩文件保存，解码不需要原 FASTQ。

排名依据现有量化后的神经概率（整数频数），降序排列；同频时 Q 值较小者在前。
这使排序使用与原 codec 相同的数值合同。全局表保存原始 42 项计数，
整数 CDF 每类保留 1 个频数单位，再按计数比例和最大余数分配剩余单位。
这是编码精度约定，不是另行拟合的概率校正。

模型仅推理一遍。真实 rank 先按原 batch/cycle-major 顺序暂存到输出目录下的临时文件，
统计结束后使用固定表编码；完成后删除暂存。CNR0847462 的 300M 个 Q 需要约 300MB
额外临时磁盘空间。终端会依次显示 `encode FASTQ` 和 `encode ranks` 进度。

## CNR0847462：40k 共享模型，仅微调输出头

以下命令使用截图对应的 40k 共享模型，仅微调残差输出头，不做 LoRA 或额外诊断。
GPU 0 可替换为空闲卡。每次自动创建新目录，JSON 写入 `metrics.json`，终端保留进度。

```bash
cd /home/share/huadjyin/home/s_qiuping1/tanglei/fastq_quality_residual
rank_run_dir=$(mktemp -d codec/output/CNR0847462_global_rank_XXXXXX)
CUDA_VISIBLE_DEVICES=0 /home/share/huadjyin/home/tanglei2/envs/scgpt/bin/python -m codec.encode \
  data/2nd/CNR0847462_1.head2M.fastq.gz \
  "$rank_run_dir/CNR0847462.rank.fqdc" \
  codec/runs/direct_quality_balanced_b256_s40000_v1/best.pt \
  --finetune-head --head-type residual --head-residual-dim 64 \
  --head-cross-layer --head-cross-dim 16 \
  --head-max-reads 8000 --head-max-symbols 1200000 \
  --head-steps 2000 --head-max-seconds 20 \
  --save-head-adapter "$rank_run_dir/head.adapter.json" \
  --quality-coding rank --quantizer floor --coder-backend native \
  --device cuda --batch-reads 256 \
  > "$rank_run_dir/metrics.json"
```

主要比较 `range_stream_bytes`，截图对应旧方案原整文件实测为 **76,259,037B**，
质量码流加适配器为 **76,445,533B**。
`rank_histogram` 是实际 42 项计数；`original_quantized_theoretical_bits` 是同次推理下原
神经分布的理论 bit 数；`quantized_theoretical_bits` 是固定 rank 表的理论 bit 数。
`quality_and_adapter_bytes` 包含 rank 码流、适配器及新增 rank 表的实际元数据增量。

## 解码

新版解码器自动识别 rank 容器版本 5。解码只需要 FQDC 和同一个 40k 共享 checkpoint，
适配器与 rank 表已经在容器中。

解码先得到 rank，再根据模型排序恢复 Q，把恢复的 Q 输入下一位置的模型历史。
现有 FASTQ SHA256 校验仍然执行。旧版 1–4 容器照常解码；旧解码器不支持版本 5。
该最简版本不组合旧 prior 或 read-state 标签。
