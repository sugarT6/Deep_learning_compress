# Full 诊断模型的分组回退评估

复用已完成的 `full.best.pt`，每组独立从它复制，不重新训练，不运行整文件编码。
用原来的2048条训练探针及8000条验证reads，重新计算浮点与floor量化BPQ。
原始Full基线也会重算，与原诊断结果核验一致后才继续评估。

| 输出候选名 | 恢复到公共预训练权重的部分 |
|---|---|
| `full` | 无；完整保留已训练Full状态。 |
| `revert_early_layers` | Transformer前两层的全部参数，包括块内norm。 |
| `revert_input_features` | prevQ/Qmer embedding、碱基编码器、输入投影。 |
| `revert_both` | 同时恢复上述两组，用于观察交互。 |

四组均保留Full的输出头、最后两层和最终Transformer norm。
索引、输入样本内容、来源文件身份和公共checkpoint SHA256均须匹配原诊断。
只扫描源FASTQ以重新取回原样本，不重新reservoir抽样，不复制数据。

```bash
cd /home/share/huadjyin/home/s_qiuping1/tanglei/fastq_quality_residual
CUDA_VISIBLE_DEVICES=0 python -m codec.ablate_adaptation \
  /home/share/huadjyin/home/s_qiuping1/tanglei/fastq_quality_residual/codec/output/adaptation_diagnostic_20261021/CNR0847462_1_full_z7o0ba9m \
  --device cuda --eval-batch-reads 256
```

程序自动在 `codec/output/adaptation_ablation_20261021/` 下创建唯一子目录，
终端显示取回样本和各组评分进度；结果为 `summary.json` 和每候选同名JSON。
`validation_quantized_delta_from_full`为相对Full的验证量化BPQ增量，正值表示退步。

这个评估测量的是当前Full解对相应参数更新的依赖。各部分已经共同适配，
回退后的损失可能包含参数间失配，不能把各组差值当作独立贡献相加，也不能
直接认定只训练该部分就能得到相同收益。最终采用的小适配器仍须实测整文件
`quality_and_adapter_bytes`。

同轮另一个任务是外层独立目录
`/home/share/huadjyin/home/s_qiuping1/tanglei/fastq_quality_scratch/` 中的随机初始化A实验，
其不限时间模式及停止规则见该目录README。它与本回退评估独立运行。
