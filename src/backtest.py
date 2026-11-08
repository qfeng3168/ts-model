"""P1 回测：walk-forward（扩张窗口、按年测试），对比动量基线与 LightGBM。

组合口径：每 5 个交易日调仓一次，按预测值取截面 top 20% 等权持有 5 日，
单边成本 0.15%（佣金+滑点粗估），收益用标签 fwd_ret_5 近似持有期收益。
评估：每日截面 RankIC、方向准确率（预测高于截面中位数且实际为正的命中近似）、组合年化/回撤。
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from features import FEATURES

ROOT = Path(__file__).resolve().parent.parent
COST = 0.0015
TEST_YEARS = [2020, 2021, 2022, 2023, 2024, 2025, 2026]


def rank_ic(d: pd.DataFrame, pred_col: str) -> float:
    ics = []
    for _, g in d.groupby("date"):
        if len(g) >= 20:
            ic, _ = spearmanr(g[pred_col], g["fwd_ret_5"])
            if np.isfinite(ic):
                ics.append(ic)
    return float(np.mean(ics)) if ics else float("nan")


def portfolio(d: pd.DataFrame, pred_col: str) -> pd.Series:
    """每 5 个交易日调仓：top 20% 等权，返回调仓周期收益序列（已扣成本）。"""
    dates = sorted(d["date"].unique())
    rebal = dates[::5]
    rets = []
    for dt in rebal:
        g = d[d["date"] == dt]
        if len(g) < 20:
            continue
        k = max(1, int(len(g) * 0.2))
        top = g.nlargest(k, pred_col)
        rets.append(top["fwd_ret_5"].mean() - COST)
    return pd.Series(rets, index=pd.to_datetime(rebal[: len(rets)]))


def ann_stats(rets: pd.Series, periods_per_year: int = 50) -> dict:
    if len(rets) == 0:
        return {}
    cum = (1 + rets).cumprod()
    total = cum.iloc[-1] - 1
    years = len(rets) / periods_per_year
    ann = (1 + total) ** (1 / years) - 1 if years > 0 else float("nan")
    dd = (cum / cum.cummax() - 1).min()
    sharpe = rets.mean() / rets.std() * np.sqrt(periods_per_year) if rets.std() > 0 else float("nan")
    return {"total_return": float(total), "ann_return": float(ann),
            "max_drawdown": float(dd), "sharpe": float(sharpe), "periods": int(len(rets))}


def main() -> None:
    df = pd.read_parquet(ROOT / "data" / "dataset.parquet")
    df["year"] = df["date"].dt.year
    rows = []
    curves = {}
    for model_name in ["momentum", "lgbm"]:
        all_test = []
        for y in TEST_YEARS:
            train = df[df["year"] < y]
            test = df[df["year"] == y].copy()
            if len(test) == 0 or len(train) == 0:
                continue
            if model_name == "momentum":
                test["pred"] = test["ret_20"]
            else:
                m = lgb.LGBMRegressor(
                    n_estimators=300, learning_rate=0.05, num_leaves=31,
                    subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1,
                )
                m.fit(train[FEATURES], train["fwd_ret_5"])
                test["pred"] = m.predict(test[FEATURES])
            ic = rank_ic(test, "pred")
            # 方向准确率：预测为正且实际为正 / 预测为负且实际为负（个股层面）
            dir_acc = float(((test["pred"] > 0) == (test["fwd_ret_5"] > 0)).mean())
            rows.append({"model": model_name, "year": y, "rank_ic": ic,
                         "dir_acc": dir_acc, "n": int(len(test))})
            test["model"] = model_name
            all_test.append(test[["date", "code", "fwd_ret_5", "pred", "model"]])
        cat = pd.concat(all_test, ignore_index=True)
        rets = portfolio(cat, "pred")
        curves[model_name] = (1 + rets).cumprod()
        stats = ann_stats(rets)
        rows.append({"model": model_name, "year": "ALL", **stats})
    rep = pd.DataFrame(rows)
    out_md = ROOT / "reports" / "p1_baseline.md"
    lines = ["# P1 基线回测（A股日线 / 沪深300成分 / 未来5日收益）", "",
             "口径：walk-forward 扩张窗口按年测试；组合每 5 日调仓 top20% 等权，单边成本 0.15%。",
             "注：成分股为当前名单，存在幸存者偏差，v0 只验证全流程。", "",
             rep.to_markdown(index=False, floatfmt=".4f"), ""]
    out_md.write_text("\n".join(lines), encoding="utf-8")
    pd.DataFrame({k: v for k, v in curves.items()}).to_csv(ROOT / "reports" / "p1_curves.csv")
    print(rep.to_string(index=False))
    print(f"\nreport: {out_md}")


if __name__ == "__main__":
    main()
