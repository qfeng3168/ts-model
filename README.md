# ts-model

自研时序模型实验：先以 A 股日线跑通「数据 → 特征 → 基线 → 无泄漏回测」全流程，再迭代自研架构。

## 口径（先钉死，再谈模型）

- 数据：沪深 300 成分股日线（前复权），2015-01 至今，源 AKShare（新浪）。
  成分股为**当前名单**，存在幸存者偏差——v0 只验证全流程，结论不可直接当策略用。
- 特征：14 个全因果特征（多周期收益、波动率、均线乖离、量比、换手、振幅等），只用 t 日及之前数据。
- 标签：未来 5 个交易日收益 `fwd_ret_5`（仅标签列使用未来信息）。
- 回测：walk-forward 扩张窗口、按年测试（2020–2026）；随机切分在时序上禁止。
- 评估：截面 RankIC、个股方向准确率、组合曲线（每 5 日调仓 top 20% 等权，单边成本 0.15%）。
- 基线：动量（ret_20）与 LightGBM。日线方向准确率能稳定 53–55% 已属真信号，跑出 70% 先查泄漏。

## 结构

- `src/fetch_data.py` — P0 数据管道：逐股拉取、断点落盘 `data/raw/`，合并 `data/daily.parquet`
- `src/features.py` — 特征与标签构建，产出 `data/dataset.parquet`
- `src/backtest.py` — P1 基线回测，产出 `reports/p1_baseline.md` 与 `reports/p1_curves.csv`
- `data/`、`reports/` 中的数据文件不入库（见 .gitignore 对 data/ 的排除，reports 仅留 md）

## 运行

```bash
python3 -m venv .venv && .venv/bin/pip install akshare lightgbm scikit-learn pyarrow pandas
.venv/bin/python src/fetch_data.py
.venv/bin/python src/features.py
.venv/bin/python src/backtest.py
```
