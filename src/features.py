"""特征与目标构建：全部特征只用 t 日及之前的数据（因果），目标为未来 5 日收益（仅作标签）。"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
HORIZON = 5

FEATURES = [
    "ret_1", "ret_5", "ret_10", "ret_20", "ret_60",
    "vol_20", "vol_60", "ma5_gap", "ma20_gap", "ma60_gap",
    "vol_ratio_20", "turnover_ma20", "range_5", "mom_rev",
]


def build(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for code, g in df.groupby("code", sort=False):
        g = g.sort_values("date").copy()
        c = g["close"]
        ret1 = c.pct_change()
        g["ret_1"] = ret1
        g["ret_5"] = c.pct_change(5)
        g["ret_10"] = c.pct_change(10)
        g["ret_20"] = c.pct_change(20)
        g["ret_60"] = c.pct_change(60)
        g["vol_20"] = ret1.rolling(20).std()
        g["vol_60"] = ret1.rolling(60).std()
        g["ma5_gap"] = c / c.rolling(5).mean() - 1
        g["ma20_gap"] = c / c.rolling(20).mean() - 1
        g["ma60_gap"] = c / c.rolling(60).mean() - 1
        g["vol_ratio_20"] = g["volume"] / g["volume"].rolling(20).mean()
        g["turnover_ma20"] = g["turnover"].rolling(20).mean()
        g["range_5"] = (g["high"].rolling(5).max() / g["low"].rolling(5).min() - 1)
        g["mom_rev"] = g["ret_5"] - g["ret_20"] / 4.0
        # 标签：未来 5 日收益（shift 为负 = 向未来看，仅标签列，特征列禁止使用）
        g["fwd_ret_5"] = c.shift(-HORIZON) / c - 1
        out.append(g)
    df = pd.concat(out, ignore_index=True)
    df = df.dropna(subset=FEATURES + ["fwd_ret_5"]).reset_index(drop=True)
    return df


if __name__ == "__main__":
    daily = pd.read_parquet(ROOT / "data" / "daily.parquet")
    df = build(daily)
    out = ROOT / "data" / "dataset.parquet"
    df.to_parquet(out, index=False)
    print(f"dataset: rows={len(df)} stocks={df['code'].nunique()} "
          f"{df['date'].min().date()}..{df['date'].max().date()}")
