# 以block质量特征继续训练expanded20模型

第一版每256条**原文件中连续的read**共享一份16字节质量描述。block按文件内read索引
固定划分，不跨文件，不把随机reservoir样本拼成相邻block。最后不足256条的尾块保留。
原来的碱基、历史Q、cycle和read长度输入保留；不使用Header、平台或accession embedding。

## 特征与训练

每条read按相对位置分4段，边界为`floor(s * read_length / 4)`。每段汇总全block的
均值Q、总体标准差、`Q<20`比例、`Q>=35`比例，共16个字段。均值/标准差按Q41归一，
各字段量化为uint8（最近值、恰好半数向上），空段为零；模型只接收16字节恢复的
float32值`byte/255`，不能直接使用未量化统计。

共享条件网络16→32→每层scale/shift，在四层Transformer之前施加FiLM。
输出层零初始化，启用时先保持原预测；新增68,128个参数，总参数2,311,866。
主干和条件网络一起训练，每个read仍根据自己的上下文预测，不要求整块使用相同分布。

训练`--batch-reads`必须等于`--block-reads`，当前训练/codec上限均为256。
推荐整块256条、`--microbatch-reads 32`：8个微批按有效Q数量加权累积梯度，
整块处理完更新一次optimizer。microbatch只影响显存，不改变block划分或每步Q预算。

按每源前90%训练、后10%验证，训练边界**向上**对齐到完整block；从旧checkpoint继续时
逐源核对SHA/read数量及训练范围，旧模型训练过的read不能被移到新验证集。
验证上限也向上完成最后一个block，实际read数量写入记录；不跨split计算特征。

## 训练命令

从已完成expanded20训练的四层`best.pt`继续，重置optimizer/采样/计步。
以下先继续10,000次更新（5×2000），不是再从头跑200k。仍使用显式20源清单和
MGI/Illumina/SOLiD/Ion的3:5:1:1 batch配比、组内文件均衡；未见文件不参与选模。

```bash
cd /home/share/huadjyin/home/s_qiuping1/tanglei/fastq_quality_residual
block_run_dir=$(mktemp -d codec/runs/block_quality_expanded20_XXXXXX)
CUDA_VISIBLE_DEVICES=0 /home/share/huadjyin/home/tanglei2/envs/scgpt/bin/python -m codec.train \
  --training-set expanded20 --cache-dir data/2nd/training_cache \
  --init-from codec/runs/direct_quality_4layer_expanded20_200k_S1pPAq/best.pt \
  --output-dir "$block_run_dir" \
  --block-quality-features --block-reads 256 --block-feature-mode features \
  --block-conditioning-dim 32 --batch-reads 256 --microbatch-reads 32 \
  --epochs 5 --steps-per-epoch 2000 \
  --learning-rate 0.0001 --weight-decay 0.00001 --grad-clip 1 \
  --lr-scheduler plateau --lr-factor 0.5 --lr-patience 2 --min-lr 0.00001 \
  --early-stopping-patience 0 --train-fraction 0.9 \
  --validation-max-reads-per-file 5120 --seed 20261104 --device cuda
```

GPU编号可替换。训练日志和进度保留在终端，目录中自动保存`best.pt`、`last.pt`、
`run_config.json`、`training_log.jsonl`和采样/语料统计。`--resume`支持在新空目录
恢复epoch checkpoint；其余配置需一致，`--epochs`为累计目标。

用于判断block特征贡献的独立constant对照（同初始化、网络、数据顺序和更新预算）：

```bash
cd /home/share/huadjyin/home/s_qiuping1/tanglei/fastq_quality_residual
control_run_dir=$(mktemp -d codec/runs/block_quality_constant_expanded20_XXXXXX)
CUDA_VISIBLE_DEVICES=0 /home/share/huadjyin/home/tanglei2/envs/scgpt/bin/python -m codec.train \
  --training-set expanded20 --cache-dir data/2nd/training_cache \
  --init-from codec/runs/direct_quality_4layer_expanded20_200k_S1pPAq/best.pt \
  --output-dir "$control_run_dir" \
  --block-quality-features --block-reads 256 --block-feature-mode constant \
  --block-conditioning-dim 32 --batch-reads 256 --microbatch-reads 32 \
  --epochs 5 --steps-per-epoch 2000 \
  --learning-rate 0.0001 --weight-decay 0.00001 --grad-clip 1 \
  --lr-scheduler plateau --lr-factor 0.5 --lr-patience 2 --min-lr 0.00001 \
  --early-stopping-patience 0 --train-fraction 0.9 \
  --validation-max-reads-per-file 5120 --seed 20261104 --device cuda
```

constant模式使用双方已知的全零描述，不读取未来Q得到模型输入，也不收描述流费用。
不增加block开关时，原训练命令、配置序列化、schema和模型参数保持原行为。

## 评价与实际编码

终端分别打印`val_Q_bpq`（纯Q浮点码长）、`val_Q_floor_bpq`（floor-v2整数概率码长）
和`val_Q_plus_side_bpq`。按20文件macro的**整数Q码长＋每块128bit**选择best及调节LR，
字段为`dataset_macro_quantized_Q_plus_block_side_bits_per_quality`；constant侧费为0。
验证指标不计容器固定metadata，实际大文件压缩时另计。150bp×256时16B约0.00333bit/Q。

模型训练完成后，`codec.encode`/`codec.decode`支持新增FQDC v6：先恢复每块描述，
再按已知bases、历史Q和描述恢复当前Q；描述独立流使用CRC、长度和解码后重算校验。
`block_feature_stream_bytes`和`block_feature_metadata_bytes`分别记录描述流及新增metadata，
两者加入`quality_and_adapter_bytes`。普通v1～5文件继续支持。

block模型直接使用神经概率编码，`--batch-reads`须与checkpoint相同；
支持仅输出头微调及导出/复用线性、residual、cross-layer头；主干及block条件网络冻结。
暂不组合rank编码、旧prior、read-state或suffix/LoRA适配。模型不接收免费未来Q。
特征提取是固定统计，当前训练的是条件网络和主干，还不是学习式hyperencoder。

## 仅输出头微调后压缩CNR0847462

以下使用已训练完成的block模型，沿用residual64/cross16头、2000步；不设秒数上限。
8000-read上限向下保留7936条完整block，按完整block划分5888训练/2048验证，
实际EOF尾块可保留。微调缓存使用原文件完整block描述的wire恢复值，与正式编码一致。
原输出头验证接受逻辑保留；报告`head_adaptation.accepted/steps_completed`记录实际使用情况。

```bash
cd /home/share/huadjyin/home/s_qiuping1/tanglei/fastq_quality_residual
block_head_dir=$(mktemp -d codec/output/CNR0847462_block_head_XXXXXX)
CUDA_VISIBLE_DEVICES=0 /home/share/huadjyin/home/tanglei2/envs/scgpt/bin/python -m codec.encode \
  data/2nd/CNR0847462_1.head2M.fastq.gz \
  "$block_head_dir/CNR0847462.block_head.fqdc" \
  codec/runs/block_quality_expanded20_y6qHS0/best.pt \
  --device cuda --batch-reads 256 \
  --quality-coding neural --quantizer floor --coder-backend native \
  --finetune-head --head-type residual --head-residual-dim 64 \
  --head-cross-layer --head-cross-dim 16 \
  --head-max-reads 8000 --head-max-symbols 1200000 \
  --head-steps 2000 --head-max-seconds 0 \
  --save-head-adapter "$block_head_dir/head.adapter.json" \
  > "$block_head_dir/metrics.json"
```

压缩进度和微调日志保留在终端。预测报告为同目录
`CNR0847462.block_head.predictions.txt`。比较`range_stream_bytes`和
`quality_and_adapter_bytes`，后者已计入头及block描述；2M条read需要7813块、125008B描述流。
