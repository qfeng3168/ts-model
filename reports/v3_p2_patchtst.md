# P2 PatchTST 式模型回测（A股日线 / 沪深300 / 未来5日收益）

评估口径与 P1 一致（walk-forward 按年测试、top20% 每5日调仓、单边成本 0.15%）；
模型差异：60日×6通道 patch Transformer，训练目标为日期内秩百分位；
训练集为测试年前两年起的扩张窗口，测试前一年单独留作早停验证（P1 无此划分）。
注：成分股为当前名单，存在幸存者偏差。

| model    | year   |   rank_ic |   dir_acc |          n |   total_return |   ann_return |   max_drawdown |   sharpe |   periods |
|:---------|:-------|----------:|----------:|-----------:|---------------:|-------------:|---------------:|---------:|----------:|
| patchtst | 2020   |    0.0588 |    0.5103 | 63388.0000 |       nan      |     nan      |       nan      | nan      |  nan      |
| patchtst | 2021   |   -0.0138 |    0.5168 | 66643.0000 |       nan      |     nan      |       nan      | nan      |  nan      |
| patchtst | 2022   |    0.0212 |    0.4984 | 69069.0000 |       nan      |     nan      |       nan      | nan      |  nan      |
| patchtst | 2023   |   -0.0035 |    0.5093 | 71332.0000 |       nan      |     nan      |       nan      | nan      |  nan      |
| patchtst | 2024   |    0.0357 |    0.5074 | 71795.0000 |       nan      |     nan      |       nan      | nan      |  nan      |
| patchtst | 2025   |    0.0158 |    0.5128 | 72296.0000 |       nan      |     nan      |       nan      | nan      |  nan      |
| patchtst | 2026   |    0.0156 |    0.4740 | 59301.0000 |       nan      |     nan      |       nan      | nan      |  nan      |
| patchtst | ALL    |  nan      |  nan      |   nan      |         0.8246 |       0.0951 |        -0.3235 |   0.5282 |  331.0000 |
