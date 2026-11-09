"""Head-focused training on the B feature set (P1+F10), same protocol.

The recurring failure mode: cross-sectional RankIC improves but the
top-20% portfolio does not. Three learners, identical data/splits:
  B_reg   plain regression on fwd_ret_5 (control; preds saved for blends)
  H_rank  LGBMRanker lambdarank, per-date groups, quintile labels,
          lambdarank_truncation_level=60 (only the tradable head counts)
  H_wreg  regression with 2x sample weight on the top/bottom return
          quintile of each training date

Writes reports/p3_head.md, reports/preds_b_reg.parquet.
"""
from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest import ann_stats, portfolio, rank_ic  # noqa: E402
from backtest_v3 import F10  # noqa: E402
from features import FEATURES  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FEATS = FEATURES + F10
PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=31,
              subsample=0.8, colsample_bytree=0.8, random_state=42,
              verbose=-1)


def quintile_labels(d: pd.DataFrame) -> np.ndarray:
    pct = d.groupby("date")["fwd_ret_5"].rank(pct=True)
    return np.minimum(4, (pct * 5).astype(int)).to_numpy()


def head_weights(d: pd.DataFrame) -> np.ndarray:
    pct = d.groupby("date")["fwd_ret_5"].rank(pct=True)
    return np.where((pct >= 0.8) | (pct <= 0.2), 2.0, 1.0)


def run(df: pd.DataFrame, name: str):
    rows, preds = [], []
    for y in range(2020, 2027):
        tr = df[df["year"] < y].sort_values("date")
        te = df[df["year"] == y].copy()
        if name == "H_rank":
            m = lgb.LGBMRanker(objective="lambdarank",
                               lambdarank_truncation_level=60, **PARAMS)
            m.fit(tr[FEATS], quintile_labels(tr),
                  group=tr.groupby("date").size().to_numpy())
        elif name == "H_wreg":
            m = lgb.LGBMRegressor(**PARAMS)
            m.fit(tr[FEATS], tr["fwd_ret_5"],
                  sample_weight=head_weights(tr))
        else:
            m = lgb.LGBMRegressor(**PARAMS)
            m.fit(tr[FEATS], tr["fwd_ret_5"])
        te["pred"] = m.predict(te[FEATS])
        rows.append({"model": name, "year": y,
                     "rank_ic": rank_ic(te, "pred"), "n": len(te)})
        preds.append(te[["date", "code", "fwd_ret_5", "pred"]])
        print(f"{name} {y}: ic={rows[-1]['rank_ic']:.4f}", flush=True)
    allp = pd.concat(preds, ignore_index=True)
    st = ann_stats(portfolio(allp, "pred"))
    rows.append({"model": name, "year": "ALL", "rank_ic": np.nan, **st})
    return pd.DataFrame(rows), allp


def main() -> None:
    df = pd.read_parquet(ROOT / "data/dataset_v3.parquet")
    df["year"] = df["date"].dt.year
    tables = []
    for name in ["B_reg", "H_rank", "H_wreg"]:
        t, preds = run(df, name)
        tables.append(t)
        if name == "B_reg":
            preds.to_parquet(ROOT / "reports/preds_b_reg.parquet",
                             index=False)
    res = pd.concat(tables, ignore_index=True)
    wide = res[res["year"] != "ALL"].pivot(index="year", columns="model",
                                           values="rank_ic")
    wide.loc["mean"] = wide.mean()
    allrows = res[res["year"] == "ALL"].set_index("model")
    lines = ["# P3 头部排序实验：为 top20% 而训", "",
             "B 特征集不变（P1 量价 + F10），只换学习目标；口径与 P1/P3 一致。",
             "B_reg 普通回归（对照）；H_rank 为 LambdaRank（按日分组、五分位标签、",
             "截断 60 名）；H_wreg 为头部加权回归（首尾五分位样本权重 ×2）。", "",
             "## 逐年 RankIC", "", wide.round(4).to_markdown(), "",
             "## 组合（全期拼接）", "",
             allrows[["total_return", "ann_return", "max_drawdown",
                      "sharpe"]].round(4).to_markdown(), ""]
    (ROOT / "reports/p3_head.md").write_text("\n".join(lines),
                                             encoding="utf-8")
    print(wide.round(4).to_string())
    print(allrows[["ann_return", "max_drawdown", "sharpe"]].round(4)
          .to_string())


if __name__ == "__main__":
    main()
