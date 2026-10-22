# 人类数据扩充训练（2026-10-19）

`--training-set expanded20` 明确读取原10个加选定新增10个缓存；不会扫描
目录后自动纳入其他文件。默认 `original10` 保留历史行为。

新增：ERR15766980、ERR15801889、SRR13142212、SRR33170689、ERR12916630、
ERR15158516、ERR11454184、ERR3989434、DRR917081、SRR29727531。
前四个属于MGI/BGI，后六个属于Illumina。

每文件前90%训练，后10%验证。现有缓存共3600万训练reads、
4,747,400,112个独立训练Q token。平台batch比例维持MGI/BGI 30%、
Illumina 50%、ABI SOLiD 10%、Ion Torrent 10%；组内文件均衡。
这些比例不是Q token比例：read长度不同，实际token比例会记录在日志中。
验证仍用20个训练文件各自后缀最多5000条reads，按文件macro bits/Q选best。

图中9个对照数据集和5个新留出项目（SRR30917699、SRR32083931、SRR29436245、
ERR16748054、SRR36237671）均不训练、不参与选best或学习率调整，也不需要其缓存。
新训练checkpoint的split记录14个留出文件；NovaSeq 6000和NovaSeq X Plus已出现在
训练池，旧两个文件的unseen-instrument标记在该split中相应取消。
历史original10 registry及其标记保持原样。

## 推荐命令：四层从头训练，先给20万步预算

在已安装torch/h5py/numpy/tqdm的原训练环境运行。按需更改空闲GPU编号。
保留实时进度；不自动启动训练。

```bash
cd /home/share/huadjyin/home/s_qiuping1/tanglei/fastq_quality_residual
EXPANDED_RUN=$(mktemp -d codec/runs/direct_quality_4layer_expanded20_200k_XXXXXX)
CUDA_VISIBLE_DEVICES=0 python -m codec.train \
  --training-set expanded20 --cache-dir data/2nd/training_cache \
  --output-dir "$EXPANDED_RUN" \
  --epochs 100 --steps-per-epoch 2000 --batch-reads 256 \
  --train-fraction 0.9 --validation-max-reads-per-file 5000 \
  --d-model 256 --num-heads 4 --num-layers 4 --feedforward-dim 512 \
  --context-length 256 --dropout 0.1 \
  --learning-rate 0.001 --weight-decay 1e-5 --grad-clip 1 \
  --lr-scheduler plateau --lr-factor 0.5 --lr-patience 2 \
  --min-lr 1e-5 --min-delta 1e-4 --early-stopping-patience 0 \
  --seed 20260909 --device cuda
```

每个logging epoch为2000步，不是遍历完整训练池。20万步是首轮预算，不代表已经
收敛或达到压缩率上限。以每文件180万训练reads、batch256计算，每文件需7032次
选择才完整覆盖。当前加权采样下，78个logging epoch覆盖全部Illumina文件，
83个覆盖全部MGI/BGI文件；100个epoch足够所有训练源至少完整覆盖一遍。
SOLiD和Ion会更早进入重复遍历。学习率由验证集plateau控制，不启用自动早停。

独立数据池Q token数不随训练重复遍数增加。日志中的累计训练Q token数则包含重复
消费的位置，并排除padding和验证数据。

## 输出与恢复

- `best.pt`：20文件验证macro最优权重，后续压缩使用这份checkpoint。
- `last.pt`：最近完整logging epoch，包含优化器、调度、采样与RNG状态。
- `training_corpus.json`：每文件及总体的训练/验证read和Q token数。
- `run_config.json`：清单、源文件SHA256、split、采样与模型/控制参数。
- `training_log.jsonl`：每轮与累计实际训练Q token、逐文件token计数、验证指标、学习率。
- `sampling_statistics.json`：结束时的逐文件实际read覆盖率与平台batch比例。
  相同统计也随每轮checkpoint保存。

中断后沿用上面命令的全部配置、指定新空目录，将原 `--output-dir` 替换为新目录，
并加 `--resume /absolute/path/to/expanded_run/last.pt`。保留原目录配套的`best.pt`。
`--epochs 100`表示累计目标100轮；例如完成30轮后恢复，只再跑70轮。
改变训练清单或采样权重会拒绝严格resume。

## 可选：从旧四层模型权重开始

如果优先考虑尽快改进现有压缩模型，可以在上述命令中加：

```bash
  --init-from codec/runs/direct_quality_4layer_resume80k_20261014_sPfodw/best.pt
```

`--init-from`只加载匹配架构的基础权重，重新建立优化器、采样、学习率调度和验证
最优值，新增训练步数从0计；不加载每文件adapter。来源checkpoint SHA256和原步数
写入run_config。该初始化训练生成的新checkpoint仍可正常严格resume，恢复时无需再传
init-from。init-from与resume互斥。

从头训练用于新的数据规模基线更容易解释；权重初始化包含原80k步的历史训练，
不能把两者直接当作相同总计算量的规模消融。若要隔离“扩数据”与“多训了更多步”，
还需用相同初始化、优化器重置和额外训练预算运行original10控制组。
