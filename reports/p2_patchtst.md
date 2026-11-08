# P2 PatchTST 式模型回测（A股日线 / 沪深300 / 未来5日收益）

评估口径与 P1 一致（walk-forward 按年测试、top20% 每5日调仓、单边成本 0.15%）；
模型差异：60日×6通道 patch Transformer，训练目标为日期内秩百分位；
训练集为测试年前两年起的扩张窗口，测试前一年单独留作早停验证（P1 无此划分）。
注：成分股为当前名单，存在幸存者偏差。

| model    | year   |   rank_ic |   dir_acc |          n |   total_return |   ann_return |   max_drawdown |   sharpe |   periods |
|:---------|:-------|----------:|----------:|-----------:|---------------:|-------------:|---------------:|---------:|----------:|
| patchtst | 2020   |    0.0583 |    0.5110 | 63388.0000 |       nan      |     nan      |       nan      | nan      |  nan      |
| patchtst | 2021   |   -0.0139 |    0.5171 | 66643.0000 |       nan      |     nan      |       nan      | nan      |  nan      |
| patchtst | 2022   |    0.0203 |    0.4987 | 69069.0000 |       nan      |     nan      |       nan      | nan      |  nan      |
| patchtst | 2023   |   -0.0041 |    0.5095 | 71332.0000 |       nan      |     nan      |       nan      | nan      |  nan      |
| patchtst | 2024   |    0.0351 |    0.5062 | 71795.0000 |       nan      |     nan      |       nan      | nan      |  nan      |
| patchtst | 2025   |    0.0160 |    0.5128 | 72296.0000 |       nan      |     nan      |       nan      | nan      |  nan      |
| patchtst | 2026   |    0.0157 |    0.4739 | 59301.0000 |       nan      |     nan      |       nan      | nan      |  nan      |
| patchtst | ALL    |  nan      |  nan      |   nan      |         0.7409 |       0.0874 |        -0.3301 |   0.4955 |  331.0000 |
