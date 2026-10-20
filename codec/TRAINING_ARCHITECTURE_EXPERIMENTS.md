# 四层基线的五组结构实验（2026-10-20）

正式 expanded20 预训练；不走单文件从零训练 A 路线。
以最近的四层 D256 / FF512 模型为参照，每组单独从随机初始化训练。

| 命令名 | 本组改动 | 基础模型参数量 |
|---|---|---:|
| 原基线（已有） | 四层，D256，4 heads，FF512，局部 base CNN | 2,243,738 |
| `ffn1024` | 只扩大 FFN 到 1024 | 3,294,362 |
| `wide384` | D384，6 heads（维持每头64维），FF768 | 4,886,170 |
| `base_dilated` | 独立双向碱基 CNN，7层 dilation 1/2/4/8/16/32/64 | 2,261,066 |
| `swiglu` | FFN 改为 SwiGLU，隐藏宽度344，近似保持 FFN 参数预算 | 2,252,634 |
| `qprefix` | 拼入严格因果的 Q 前缀均值、方差、Q<20比例、连续低Q长度 | 2,244,762 |

五组均为四层，context256、dropout0.1。除表中改动，其余模型设置相同。
SwiGLU 的基础参数量比原模型多约0.4%，适合比较门控结构；并非 FF512 的额外扩容组。
碱基 CNN 输出维度仍为32，感受野跨度255，可使用较远的未来碱基，未来质量值不可见。
每层卷积前后屏蔽 padding。Q 前缀统计在预测第t个Q时仅计算Q[:t]，解码器能自行
重建，**没有 read 状态标签流**。四项依次按41、41²、1、512归一化，空前缀为0。

训练数据、split和控制与现有 expanded20 20万步基线一致：

- 明确使用原10个及新增10个H5缓存，不扫描其他源，不纳入14个留出文件。
- 每文件前90%训练，后10%验证；每文件最多5000条验证reads，以20源macro BPQ选best。
- 平台batch比例3:5:1:1，组内文件均衡；batch256、相同seed20260909。
- 100个logging epoch × 2000步 = 200,000步；不是100次完整遍历数据池。
- AdamW，初始LR0.001、weight decay1e-5、clip1；plateau×0.5、patience2、
  min LR1e-5、min delta1e-4；不自动早停。

## 在不同终端分别启动

先激活此前的训练环境。每条命令独立完整；按实际空闲卡更换 `CUDA_VISIBLE_DEVICES=0`。
入口默认CUDA，下面显式写出。训练保留实时进度，不由代码后台启动其他实验。

```bash
cd /home/share/huadjyin/home/s_qiuping1/tanglei/fastq_quality_residual
CUDA_VISIBLE_DEVICES=0 python -m codec.train_experiments ffn1024 --device cuda
```

```bash
cd /home/share/huadjyin/home/s_qiuping1/tanglei/fastq_quality_residual
CUDA_VISIBLE_DEVICES=0 python -m codec.train_experiments wide384 --device cuda
```

```bash
cd /home/share/huadjyin/home/s_qiuping1/tanglei/fastq_quality_residual
CUDA_VISIBLE_DEVICES=0 python -m codec.train_experiments base_dilated --device cuda
```

```bash
cd /home/share/huadjyin/home/s_qiuping1/tanglei/fastq_quality_residual
CUDA_VISIBLE_DEVICES=0 python -m codec.train_experiments swiglu --device cuda
```

```bash
cd /home/share/huadjyin/home/s_qiuping1/tanglei/fastq_quality_residual
CUDA_VISIBLE_DEVICES=0 python -m codec.train_experiments qprefix --device cuda
```

上述命令自动使用原缓存绝对路径，不复制数据，不传 `--init-from`。
不同模型的每步耗时不同，相同20万步预算不代表相同运行时间或保证一晚完成。

## 结果目录和恢复

统一位于：

```text
/home/share/huadjyin/home/s_qiuping1/tanglei/fastq_quality_residual/codec/runs/architecture_ablation_20261020/
  ffn1024_<随机后缀>/
  wide384_<随机后缀>/
  base_dilated_<随机后缀>/
  swiglu_<随机后缀>/
  qprefix_<随机后缀>/
```

每次调用都会创建新空子目录；同实验重复启动也不会覆盖前次结果。
终端开始时输出完整路径和参数量。各目录都有 `best.pt`、`last.pt`、
`run_config.json`、`training_log.jsonl`、`training_corpus.json`、
`sampling_statistics.json` 和 `training_summary.json`，沿用现有训练流程。
完成后用各自 **best.pt** 做压缩测试，保持相同文件与微调预算，以
`quality_and_adapter_bytes` 判断净收益，不能只看range流大小。

五组都兼容现有压缩/解压入口及last2 rank8 LoRA；SwiGLU会同时适配gate权重，
额外gate因子真实计入adapter大小。其FF344下last2/rank8的未压缩suffix参数为
248,320 B，原FF512 GELU为231,424 B；最终adapter压缩大小仍由实际权重决定。

只检查配置而不训练：

```bash
python -m codec.train_experiments qprefix --dry-run
```

中断恢复保留相同实验名及所有训练控制配置，加 `--resume` 指向该实验的`last.pt`。
仍自动生成新子目录，不覆盖旧目录，累计训练目标仍是100个epoch；旧目录中的
匹配`best.pt`须保留。例如：

```bash
python -m codec.train_experiments qprefix --device cuda \
  --resume /absolute/path/to/qprefix_previous/last.pt
```

入口亦提供 `--epochs`、`--steps-per-epoch`、`--batch-reads`、`--cache-dir`、
`--run-group`等选项。做公平对照时五组保持相同设置。常规 `codec.train` 入口新增
`--base-encoder-type`、`--ffn-type`、`--quality-prefix-features`用于手动配置；旧默认值
及checkpoint保持兼容。
