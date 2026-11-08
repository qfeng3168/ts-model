"""P3 ablation backtest: same walk-forward protocol as P1, three feature sets.

A = P1's 14 price/volume features (control, on the fixed data)
B = A + 13 F10 features (fundamentals, point-in-time by notice_date)
C = B + 5 XWLB features (industry policy intensity, strict 1-day lag)

Writes reports/p3_ablation.md and reports/p3_curves.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest import ann_stats, portfolio, rank_ic  # noqa: E402
from features import FEATURES  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
F10 = ["roe_wtd", "roa_wtd", "gross_margin", "net_margin", "total_rev_yoy_gr",
       "net_profit_yoy_gr", "net_profit_qoq_gr", "asset_liab_ratio",
       "days_since_notice", "ep_ttm", "bp", "cfp", "div_yield_ttm"]
XW = ["xwlb_grp", "xwlb_grp5", "xwlb_grp20", "xwlb_fin", "xwlb_items"]
VARIANTS = {"A_p1": FEATURES, "B_f10": FEATURES + F10,
            "C_f10_xwlb": FEATURES + F10 + XW}


def run(df: pd.DataFrame, feats: list[str], name: str):
    rows, preds, curves = [], [], {}
    for y in range(2020, 2027):
        tr = df[df["year"] < y]
        te = df[df["year"] == y].copy()
        m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05,
                              num_leaves=31, subsample=0.8,
                              colsample_bytree=0.8, random_state=42,
                              verbose=-1)
        m.fit(tr[feats], tr["fwd_ret_5"])
        te["pred"] = m.predict(te[feats])
        rets = portfolio(te, "pred")
        curves[y] = rets
        rows.append({"model": name, "year": y,
                     "rank_ic": rank_ic(te, "pred"),
                     "dir_acc": ((te["pred"] > 0)
                                 == (te["fwd_ret_5"] > 0)).mean(),
                     "n": len(te)})
        preds.append(te[["date", "code", "fwd_ret_5", "pred"]])
        print(f"{name} {y}: ic={rows[-1]['rank_ic']:.4f}", flush=True)
    allp = pd.concat(preds, ignore_index=True)
    all_rets = portfolio(allp, "pred")
    st = ann_stats(all_rets)
    rows.append({"model": name, "year": "ALL", "rank_ic": None,
                 "dir_acc": None, "n": None, **st})
    return pd.DataFrame(rows), all_rets


def main() -> None:
    df = pd.read_parquet(ROOT / "data/dataset_v3.parquet")
    df["year"] = df["date"].dt.year
    tables, curve_frames = [], []
    for name, feats in VARIANTS.items():
        t, rets = run(df, feats, name)
        tables.append(t)
        cf = rets.rename(name).to_frame()
        curve_frames.append(cf)
    res = pd.concat(tables, ignore_index=True)
    curves = pd.concat(curve_frames, axis=1)
    curves.to_csv(ROOT / "reports/p3_curves.csv")
    wide = res[res["year"] != "ALL"].pivot(index="year", columns="model",
                                           values="rank_ic")
    wide["mean"] = wide.mean()
    allrows = res[res["year"] == "ALL"].set_index("model")
    lines = ["# P3 消融回测：F10 基本面 + 新闻联播政策特征", "",
             "口径与 P1 完全一致（修复后数据、walk-forward 按年、top20% 每 5 日调仓、成本 0.15%）。",
             "A = P1 的 14 个量价特征；B = A + 13 个 F10 特征（按公告日时点对齐）；",
             "C = B + 5 个新闻联播特征（行业政策提及强度，严格滞后一日）。", "",
             "## 逐年 RankIC", "", wide.round(4).to_markdown(), "",
             "## 组合（全期拼接）", "",
             allrows[["total_return", "ann_return", "max_drawdown",
                      "sharpe"]].round(4).to_markdown(), ""]
    (ROOT / "reports/p3_ablation.md").write_text("\n".join(lines),
                                                 encoding="utf-8")
    print(wide.round(4).to_string())
    print(allrows[["ann_return", "max_drawdown", "sharpe"]].round(4)
          .to_string())
    print("report: reports/p3_ablation.md")


if __name__ == "__main__":
    main()
