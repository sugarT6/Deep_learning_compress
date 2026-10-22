# 相对位置与碱基 Encoder＋Q Decoder（2026-10-22）

两项分别训练、分别比较。均使用 expanded20 原有 20 个 H5 缓存做常规预训练，
不使用待压缩文件训练基础模型。

| 实验名 | 结构变化 | 基础模型参数量 |
|---|---|---:|
| 原基线 | 四层 D256、4 heads、FF512、局部碱基 CNN | 2,243,738 |
| `relative_position` | 每层增加可学习的逐 head 相对距离偏置 | 2,247,834 |
| `base_encoder_decoder` | 增加两层 D64 碱基 Encoder，后两层 Q block 增加 cross-attention | 2,642,954 |

## 训练方式

**两组都从随机初始化重新预训练。** 相对位置版可以设计为旧权重初始化，
但本次统一从头训练，保持与已有基线相同的数据和 20 万步预算，避免把结构效果
与额外续训混在一起。碱基 Encoder、cross-attention 需要在预训练中一起学。
入口不自动加载旧基线，`--resume` 只用于恢复本实验自身的 `last.pt`。

沿用原 expanded20 训练规则：

- 原 10 源＋新增 10 源，读取 `data/2nd/training_cache/`；不复制数据、不纳入留出测试源。
- 每源前 90% 训练、后 10% 验证，每源最多 5000 条验证 read；用 20 源 macro BPQ 选择 `best.pt`。
- 平台 batch 比例 3:5:1:1，组内文件均衡；batch256，seed20260909。
- 100 个 logging epoch × 每 epoch 2000 步＝200,000 次更新，不是完整遍历数据 100 次。
- AdamW：LR0.001、weight decay1e-5、clip1；plateau 乘 0.5、patience2、
  min LR1e-5、min delta1e-4；不自动早停。

## 两个终端分别运行

使用之前的训练环境。示例使用 GPU0 和 GPU1，按实际空闲卡修改。

显式相对位置：

```bash
cd /home/share/huadjyin/home/s_qiuping1/tanglei/fastq_quality_residual
CUDA_VISIBLE_DEVICES=0 python -m codec.train_experiments relative_position --device cuda
```

轻量碱基 Encoder＋Q Decoder：

```bash
cd /home/share/huadjyin/home/s_qiuping1/tanglei/fastq_quality_residual
CUDA_VISIBLE_DEVICES=1 python -m codec.train_experiments base_encoder_decoder --device cuda
```

每次调用自动创建独立空目录，不覆盖已有结果：

```text
/home/share/huadjyin/home/s_qiuping1/tanglei/fastq_quality_residual/codec/runs/conditioning_architecture_20261022/
  relative_position_<随机后缀>/
  base_encoder_decoder_<随机后缀>/
```

终端会输出实际路径。保存 `best.pt`、`last.pt`、`run_config.json`、训练日志、
采样统计和训练总结，训练完成后用各自 `best.pt` 测试压缩。
相同更新步数不代表两组训练用时相同。

检查配置但不训练：

```bash
python -m codec.train_experiments relative_position --dry-run
python -m codec.train_experiments base_encoder_decoder --dry-run
```

中断后使用同一实验名和训练配置，加 `--resume /absolute/path/to/last.pt`。
入口创建新子目录，恢复 optimizer、sampler 和 LR 状态，累计目标仍为 100 个 epoch。
旧目录中的匹配 `best.pt` 也须保留。

## 结构与压缩边界

相对位置采用 **learned relative attention bias**。每层每个 head 对过去距离
0～255 各学习一个偏置，添加到 self-attention 分数；初始化为零。
原来的三个位置／长度标量保留，未来 Q 和窗口外位置仍屏蔽。
共增加 4 层 × 4 heads × 256 距离＝4096 个参数。
推理路径显式执行该偏置，避开 PyTorch 2.0 会丢弃浮点 attention mask 的 MHA 快路径。

碱基 Encoder 独立读取完整碱基序列与三个位置／长度标量，不接收任何 Q。
使用两层双向 Transformer，D64、4 heads、FF128；padding 不参与有效位置的 attention。
Q 主干仍为四层 D256／FF512：前两层为原因果 self-attention block，后两层依次执行
因果 self-attention、读取碱基 memory 的 cross-attention、FFN，均为 pre-LN 残差。
原局部碱基 CNN 输入支路保留。整条碱基序列在解码前已知，允许双向编码。

两组都不增加 read 标签或其他辅助流。默认配置、旧 checkpoint 和已有实验继续兼容。
新 checkpoint 保存结构配置与 feature schema，解码时自动重建对应结构。

后续压缩对照沿用当前最佳 **全四层 rank4 LoRA＋residual/cross-layer 输出头**，
保持原来的采样和 10,000 步预算。新增的 relative bias、base memory 和 cross-attention
在单文件 LoRA 阶段冻结；LoRA 仍只覆盖原 self-attention／FFN 矩阵，适配器参数数量
与四层基线相同。压缩后的适配器字节数会随实际权重变化，应继续以
`quality_and_adapter_bytes` 判断净收益。共享的基础 checkpoint 仍需在解码端提供。

新结构支持普通输出头适配和 suffix LoRA。当前 dense suffix 格式没有保存新增层参数，
所以显式拒绝这两组的 dense suffix 训练；碱基 memory 版也拒绝会绕过 memory 的旧
`--head-lora` 路径。使用 `--finetune-last-layers 4 --backbone-lora-rank 4` 即可走现有
suffix LoRA 协议，无需新增容器格式。

常规 `codec.train` 也提供 `--relative-position-bias`／`--base-cross-attention`，
以及 `--base-memory-dim`、`--base-memory-heads`、`--base-memory-layers`、
`--base-cross-attention-layers` 供手动配置。本次两个 preset 不叠加，均使用 GELU。
