# 文件适配诊断：固定训练探针与验证曲线

这个实验从当前扩充训练集的四层 `best.pt` 开始，比较后两层 rank8 LoRA
与全模型微调。目的在于判断现有适配是否还没训够、是否出现样本过拟合，
以及 LoRA 的参数约束是否限制拟合。它不是从随机权重开始训练。

诊断只处理适配样本，不执行整文件压缩。LoRA 分支额外导出兼容现有 codec 的 adapter，
full 分支只保存诊断权重，没有可直接部署的 adapter。
输出的 BPQ 是预测码长证据，不是最终 `quality_and_adapter_bytes`。
若全参数微调有效，还需要单独验证权重存储成本和实际整文件压缩收益。

## 保持相同的数据和初始化

- 基础模型：`codec/runs/direct_quality_4layer_expanded20_200k_S1pPAq/best.pt`。
- 两组共用 residual64/cross16 输出头预热：前 8,000 reads、2,000 步。
- 沿用整文件 reservoir 采样：32,000 reads，划分 24,000 训练、8,000 验证。
  预热头使用的前缀不进入这一 reservoir；验证 reads 不参加后续梯度更新。
- 从 24,000 条训练 reads 中固定选 2,048 条作为训练探针。
  它们参与训练，用于观察拟合程度，不是另一组留出验证集。
- 每 200 步，对同一训练探针和全部 8,000 条验证 reads 计算浮点 bits/Q。
  与每步变化的随机 mini-batch loss 相比，这两条曲线可以跨步比较。
- 初始状态和验证最优状态另对全部训练 reads、全部验证 reads 计算浮点及 floor 量化码长。
  训练过程按浮点验证 BPQ 选择最优正训练步，step 0 不参与最优步选择。
- 不加入 read 状态，不增加标签流；两组从相同的基础模型与预热头权重开始。

LoRA 组保留当前后两层 Q/K/V/O 与 FFN 的 rank8 适配方式；full 组放开全部模型参数，
包括输出头。默认主干学习率 LoRA 为 `0.0003`，full 为 `0.0001`，两组输出头均为 `0.001`，
分别可由 `--lora-lr`、`--full-lr`、`--head-lr` 设置。
这是两种适配方式的实用对照；两组可训练参数、学习率、每步耗时不同，
不能称为相同计算量或相同学习率的严格消融。

两组默认均保留 `--anchor-strength 0.01`，对应当前适配基线的参数锚定项。
日志分别记录 mini-batch 交叉熵和 anchor 分量，固定探针与验证 BPQ 不含 anchor。
相同 anchor 系数在 LoRA 因子和全参数位移上并不是相同的先验约束。
可用 `--anchor-strength 0` 检查关闭联合适配约束的效果；这不会关闭共享头预热阶段
独立的 `--warmup-anchor-strength 0.01`。如需两阶段都关闭，需同时设置两项为 0。

## 推荐命令：一次运行两组

将 GPU `0` 改为空闲卡编号。程序在 `--output-root` 内自动创建唯一的结果目录，
不会覆盖已有实验。终端保留实时进度。

```bash
cd /home/share/huadjyin/home/s_qiuping1/tanglei/fastq_quality_residual

CUDA_VISIBLE_DEVICES=0 python -m codec.diagnose_adaptation \
  data/2nd/CNR0847462_1.head2M.fastq.gz \
  codec/runs/direct_quality_4layer_expanded20_200k_S1pPAq/best.pt \
  --mode both --device cuda \
  --steps 10000 --eval-every 200 --batch-reads 16 \
  --output-root codec/output/adaptation_diagnostic_20261021
```

`both` 共用预热和样本准备，依次运行两种适配方式。
1 万步只是上限，不代表已经收敛。默认 `--patience 6` 保留验证早停，可能提前结束。
固定探针与完整验证增加了评估耗时，
不能将这个命令的耗时直接当作原压缩命令的适配耗时。

## 可选：不同终端分别运行

两个终端均可直接执行完整命令，无需共享 Shell 变量。GPU `0`、`1` 按需修改。
默认种子相同，采样和预热设置也相同；程序会记录实际采样与运行配置。
跨终端对照时可核对 manifest 中的预热权重 hash、样本内容 hash 和分组索引 hash。

LoRA：

```bash
cd /home/share/huadjyin/home/s_qiuping1/tanglei/fastq_quality_residual

CUDA_VISIBLE_DEVICES=0 python -m codec.diagnose_adaptation \
  data/2nd/CNR0847462_1.head2M.fastq.gz \
  codec/runs/direct_quality_4layer_expanded20_200k_S1pPAq/best.pt \
  --mode lora --device cuda \
  --steps 10000 --eval-every 200 --batch-reads 16 \
  --output-root codec/output/adaptation_diagnostic_20261021
```

全参数微调：

```bash
cd /home/share/huadjyin/home/s_qiuping1/tanglei/fastq_quality_residual

CUDA_VISIBLE_DEVICES=1 python -m codec.diagnose_adaptation \
  data/2nd/CNR0847462_1.head2M.fastq.gz \
  codec/runs/direct_quality_4layer_expanded20_200k_S1pPAq/best.pt \
  --mode full --device cuda \
  --steps 10000 --eval-every 200 --batch-reads 16 \
  --output-root codec/output/adaptation_diagnostic_20261021
```

## 输出文件

| 文件 | 用途 |
|---|---|
| `manifest.json` | 基础 checkpoint SHA256、输入文件身份、预热权重 hash、样本内容和分组 hash，以及全部配置。 |
| `split_indices.npz` | 固定训练、验证和训练探针的原文件 read 索引，可核对样本是否一致。 |
| `initial_metrics.json` | 共享预热头起点的全部训练集、验证集和训练探针评分，包含浮点与量化码长。 |
| `lora.history.jsonl`、`full.history.jsonl` | 每次评估的固定训练探针、验证 BPQ，以及 mini-batch 交叉熵和 anchor 等训练记录。 |
| `lora.summary.json`、`full.summary.json` | 各组实际步数、最佳步数、训练信息，以及恢复最佳状态后的全训练集、验证集评分。 |
| `summary.json` | 共同起点与已完成分支的汇总，首先查看这个文件。 |
| `lora.best.pt`、`full.best.pt` | 自定义诊断状态，不是正式 codec checkpoint，不能直接作为基础模型传给 encode 或训练恢复入口。 |
| `lora.adapter.json` | LoRA 组额外导出的正式 adapter artifact；本轮不使用它运行整文件压缩。 |

只运行一个模式时，另一模式的文件不会生成。LoRA 最终评分按正式 adapter 的参数合并方式
恢复模型，避免直接保留 LoRA 因子计算与正式合并推理之间的数值差异。

## 如何判断曲线

优先比较每 200 步记录的固定训练探针与验证 bits/Q，越低越好。
训练探针只覆盖训练集的一部分，不能当作全训练集的精确损失。

| 观察到的现象 | 可以支持的判断 |
|---|---|
| 训练与验证仍同时下降 | 当前预算下仍有优化空间，可先研究适配步数和学习率过程。 |
| 训练下降，验证停滞或变差 | 正在拟合训练样本特性，应优先检查适配样本代表性和过拟合。 |
| 两条曲线均趋平 | 当前训练设置进入平台期；单凭这一点不能区分优化限制、模型限制和可预测信息不足。 |

如果 full 明显降低训练和验证码长，而 LoRA 做不到，说明“当前 rank8 适配约束”
值得进一步研究，不代表必须扩大公共基础模型。若 full 只降低训练损失、没有降低验证损失，
则不能据此宣称最终压缩更好。

验证曲线参与最优步数选择，不是独立的最终测试结果；小样本诊断也不能证明质量数据的熵极限。
正式判断仍需保留最有希望的一组配置，执行原有整文件压缩并计算全部传输成本。
