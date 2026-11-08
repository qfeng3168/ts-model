# P1 基线回测（A股日线 / 沪深300成分 / 未来5日收益）

口径：walk-forward 扩张窗口按年测试；组合每 5 日调仓 top20% 等权，单边成本 0.15%。
注：成分股为当前名单，存在幸存者偏差，v0 只验证全流程。

| model    | year   |   rank_ic |   dir_acc |          n |   total_return |   ann_return |   max_drawdown |   sharpe |   periods |
|:---------|:-------|----------:|----------:|-----------:|---------------:|-------------:|---------------:|---------:|----------:|
| momentum | 2020   |   -0.0044 |    0.4801 | 63388.0000 |       nan      |     nan      |       nan      | nan      |  nan      |
| momentum | 2021   |   -0.0082 |    0.4924 | 66643.0000 |       nan      |     nan      |       nan      | nan      |  nan      |
| momentum | 2022   |   -0.0293 |    0.4953 | 69069.0000 |       nan      |     nan      |       nan      | nan      |  nan      |
| momentum | 2023   |   -0.0153 |    0.4933 | 71332.0000 |       nan      |     nan      |       nan      | nan      |  nan      |
| momentum | 2024   |   -0.0073 |    0.4980 | 71795.0000 |       nan      |     nan      |       nan      | nan      |  nan      |
| momentum | 2025   |   -0.0117 |    0.4920 | 72296.0000 |       nan      |     nan      |       nan      | nan      |  nan      |
| momentum | 2026   |   -0.0029 |    0.5045 | 59301.0000 |       nan      |     nan      |       nan      | nan      |  nan      |
| momentum | ALL    |  nan      |  nan      |   nan      |         1.9225 |       0.1759 |        -0.3102 |   0.7400 |  331.0000 |
| lgbm     | 2020   |    0.0430 |    0.5119 | 63388.0000 |       nan      |     nan      |       nan      | nan      |  nan      |
| lgbm     | 2021   |    0.0402 |    0.4953 | 66643.0000 |       nan      |     nan      |       nan      | nan      |  nan      |
| lgbm     | 2022   |    0.0126 |    0.4674 | 69069.0000 |       nan      |     nan      |       nan      | nan      |  nan      |
| lgbm     | 2023   |    0.0203 |    0.4775 | 71332.0000 |       nan      |     nan      |       nan      | nan      |  nan      |
| lgbm     | 2024   |    0.0195 |    0.4986 | 71795.0000 |       nan      |     nan      |       nan      | nan      |  nan      |
| lgbm     | 2025   |   -0.0105 |    0.5038 | 72296.0000 |       nan      |     nan      |       nan      | nan      |  nan      |
| lgbm     | 2026   |    0.0293 |    0.4772 | 59301.0000 |       nan      |     nan      |       nan      | nan      |  nan      |
| lgbm     | ALL    |  nan      |  nan      |   nan      |         3.2051 |       0.2423 |        -0.2143 |   0.9840 |  331.0000 |
