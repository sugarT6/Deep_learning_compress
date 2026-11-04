# 全文件预测准确率、概率区间和低概率位置报告

每次使用 `python -m codec.encode` 或 `encode_fastq()` 压缩，都会在压缩文件旁
自动保存预测报告，不需要增加命令行参数。例如输出 `CNR0847462.fqdc` 时，报告为
同目录下的 `CNR0847462.predictions.txt`。文件名由输出路径替换扩展名得到。

第一行统计整个输入文件中所有有效 Q 位置的 top-1 准确率，接下来七行给出名次占比。
以下只是格式示意，不是实验实测结果：

```text
prediction_accuracy=75.000000% correct=300000 total=400000
rank1 75.000000%
rank2 15.000000%
rank3 5.000000%
rank4 2.000000%
rank5 1.000000%
rank6 1.000000%
rank其它 1.000000%
```

`correct` 为模型最大概率类别等于真实 Q 的位置数，`total` 为有效质量值总数，
不包含 padding。空文件或全部 read 长度为零时，准确率和七项占比均为 `N/A`，两个计数为零。

名次按量化前的模型分布排序，使用从1开始的编号：`rank1` 为真实Q恰好排第一，
`rank2`～`rank6` 分别为恰好排第二至第六，`rank其它` 为排第七及以后。
七项是互斥占比，分母均为全文件的有效Q数；不是累计 Top-k 命中率，
也不局限于前1000个位置。`rank1` 占比与第一行准确率一致。
这里只改变报告的编号，压缩容器内部的rank索引和编码方式保持原样。

八行汇总之后不再打印前1000个位置的top-5，改为统计**全文件所有有效Q**。
概率指模型赋给真实Q的概率`p_true`，包括top1预测错误的位置；不是最大类别的概率，
也不是rank编码表中的概率。每0.1一档，左闭右开，最后一档包含1：

```text
[0.0,0.1) 12.345678%
...
[0.9,1.0] 3.456789%
```

以上数字仅示意格式。每行分母均为全文件Q总数，10档计数之和等于全文件Q数；
百分比保留6位小数，舍入后之和可能略有差异。概率为最终适配模型在全部42类上
计算的float64 softmax，位于rank转换、旧私有概率融合和整数CDF量化之前。
Q是数值Q0～Q41；同logit的rank仍按Q从小到大排列。

接着汇总`0.0 <= p_true <= 0.3`的总数及占全文件比例，并按read内位置打印分布。
例如以下1000Q、其中100个低概率Q的示意：

```text
low_probability=[0.0,0.3] count=100 total=1000 proportion=10.000000%
low_probability_positions (1-based; share=position_low/all_low; rate=position_low/position_total)
position1 20.000000% low=20 total=100 rate=20.000000%
```

- `position1`是read第1个字符，依次到最大read长度；CNR150bp时有150行。
- 位置后的第一个比例是该位置低概率Q数除以**全文件全部低概率Q数**。
- `low`是该位置低概率Q数，`total`是该位置全部有效Q数，`rate=low/total`。
  比较`rate`可以判断read开头是否更难，且支持不同read长度。
- 恰好0.3归`[0.3,0.4)`档，同时计入低概率；不先舍入概率再分类。
- 所有batch均累计，padding和空read不进入任何分母。分母为0写`N/A`；
  没有低概率时，位置占全部低概率的比例为`N/A`、位置内低概率率为0。
  全文件没有Q时，10档均为`N/A`，不打印位置行。

`metrics.json` 中另外增加三个字段：

| 字段 | 含义 |
|---|---|
| `prediction_top1_accuracy` | 全文件 top-1 准确率，数值范围0～1；没有Q时为null |
| `prediction_correct_symbols` | 全文件预测正确的Q数量 |
| `prediction_report_path` | 报告的绝对路径 |

报告复用压缩已有的前向结果，不额外运行模型。全文件新增42类softmax统计有CPU开销，
每次只处理至多8192行，累计计数的内存只随最大read长度增长，不保存逐Q概率或位置清单。
额外耗时分别记录为
`encoding_stage_seconds.prediction_diagnostic` 和 `prediction_report_write`。

报告是压缩文件外的诊断产物，不写入 `.fqdc`，不改变 CDF、容器格式或
`quality_and_adapter_bytes`。已有同名报告会在加载模型前拒绝覆盖；沿用 `mktemp -d`
为每次实验分配新目录即可。

准确率用于辅助观察。压缩码长还取决于真实 Q 获得的概率，准确率较高并不保证
码长更短。旧专家概率配置仅保留为私有回归测试入口，其报告仍统计神经模型，
不统计专家混合概率；公开压缩入口始终为纯神经预测。
