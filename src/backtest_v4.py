"""Ablation round 2 on dataset_v4 (base = B: P1 features + F10).

D_sector = B + 4 industry momentum features
F_macro  = B + 12 macro features (gold/FX/copper, lagged 1 day)
G_events = B + 5 block-trade / dragon-tiger features (trailing 20d)
ALL      = B + everything above

Same walk-forward protocol. Writes reports/p4_ablation.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest import ann_stats, portfolio, rank_ic  # noqa: E402
from backtest_v3 import F10  # noqa: E402
from build_v4 import EVENTS, MACRO, SECTOR  # noqa: E402
from features import FEATURES  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BASE = FEATURES + F10
VARIANTS = {"D_sector": BASE + SECTOR, "F_macro": BASE + MACRO,
            "G_events": BASE + EVENTS,
            "ALL_v4": BASE + SECTOR + MACRO + EVENTS}


def run(df: pd.DataFrame, feats: list[str], name: str):
    rows, preds = [], []
    for y in range(2020, 2027):
        tr, te = df[df["year"] < y], df[df["year"] == y].copy()
        m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05,
                              num_leaves=31, subsample=0.8,
                              colsample_bytree=0.8, random_state=42,
                              verbose=-1)
        m.fit(tr[feats], tr["fwd_ret_5"])
        te["pred"] = m.predict(te[feats])
        rows.append({"model": name, "year": y,
                     "rank_ic": rank_ic(te, "pred"), "n": len(te)})
        preds.append(te[["date", "code", "fwd_ret_5", "pred"]])
        print(f"{name} {y}: ic={rows[-1]['rank_ic']:.4f}", flush=True)
    allp = pd.concat(preds, ignore_index=True)
    st = ann_stats(portfolio(allp, "pred"))
    rows.append({"model": name, "year": "ALL", "rank_ic": None, **st})
    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_parquet(ROOT / "data/dataset_v4.parquet")
    df["year"] = df["date"].dt.year
    res = pd.concat([run(df, f, n) for n, f in VARIANTS.items()],
                    ignore_index=True)
    wide = res[res["year"] != "ALL"].pivot(index="year", columns="model",
                                           values="rank_ic")
    wide.loc["mean"] = wide.mean()
    allrows = res[res["year"] == "ALL"].set_index("model")
    lines = ["# P4 消融：板块动量 / 宏观 / 大宗与龙虎榜", "",
             "基底为 B（P1 量价 + F10），口径与 P1/P3 一致。", "",
             "## 逐年 RankIC", "", wide.round(4).to_markdown(), "",
             "## 组合（全期拼接）", "",
             allrows[["total_return", "ann_return", "max_drawdown",
                      "sharpe"]].round(4).to_markdown(), ""]
    (ROOT / "reports/p4_ablation.md").write_text("\n".join(lines),
                                                 encoding="utf-8")
    print(wide.round(4).to_string())
    print(allrows[["ann_return", "max_drawdown", "sharpe"]].round(4)
          .to_string())


if __name__ == "__main__":
    main()
