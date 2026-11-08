# P1 基线回测（A股日线 / 沪深300成分 / 未来5日收益）

口径：walk-forward 扩张窗口按年测试；组合每 5 日调仓 top20% 等权，单边成本 0.15%。
注：成分股为当前名单，存在幸存者偏差，v0 只验证全流程。

| model    | year   |   rank_ic |   dir_acc |          n |   total_return |   ann_return |   max_drawdown |   sharpe |   periods |
|:---------|:-------|----------:|----------:|-----------:|---------------:|-------------:|---------------:|---------:|----------:|
| momentum | 2020   |   -0.0040 |    0.4801 | 63388.0000 |       nan      |     nan      |       nan      | nan      |  nan      |
| momentum | 2021   |   -0.0083 |    0.4924 | 66643.0000 |       nan      |     nan      |       nan      | nan      |  nan      |
| momentum | 2022   |   -0.0294 |    0.4953 | 69069.0000 |       nan      |     nan      |       nan      | nan      |  nan      |
| momentum | 2023   |   -0.0155 |    0.4932 | 71332.0000 |       nan      |     nan      |       nan      | nan      |  nan      |
| momentum | 2024   |   -0.0071 |    0.4980 | 71795.0000 |       nan      |     nan      |       nan      | nan      |  nan      |
| momentum | 2025   |   -0.0117 |    0.4921 | 72296.0000 |       nan      |     nan      |       nan      | nan      |  nan      |
| momentum | 2026   |   -0.0029 |    0.5045 | 59301.0000 |       nan      |     nan      |       nan      | nan      |  nan      |
| momentum | ALL    |  nan      |  nan      |   nan      |         1.9424 |       0.1771 |        -0.3093 |   0.7439 |  331.0000 |
| lgbm     | 2020   |    0.0408 |    0.5117 | 63388.0000 |       nan      |     nan      |       nan      | nan      |  nan      |
| lgbm     | 2021   |    0.0402 |    0.4968 | 66643.0000 |       nan      |     nan      |       nan      | nan      |  nan      |
| lgbm     | 2022   |    0.0137 |    0.4673 | 69069.0000 |       nan      |     nan      |       nan      | nan      |  nan      |
| lgbm     | 2023   |    0.0216 |    0.4786 | 71332.0000 |       nan      |     nan      |       nan      | nan      |  nan      |
| lgbm     | 2024   |    0.0241 |    0.5006 | 71795.0000 |       nan      |     nan      |       nan      | nan      |  nan      |
| lgbm     | 2025   |   -0.0143 |    0.5030 | 72296.0000 |       nan      |     nan      |       nan      | nan      |  nan      |
| lgbm     | 2026   |    0.0338 |    0.4766 | 59301.0000 |       nan      |     nan      |       nan      | nan      |  nan      |
| lgbm     | ALL    |  nan      |  nan      |   nan      |         3.4298 |       0.2521 |        -0.2272 |   1.0130 |  331.0000 |
