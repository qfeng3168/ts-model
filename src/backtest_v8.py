"""P8: training-window sensitivity for the D6 config.

Expanding (all history) vs rolling 5y vs rolling 3y. Motivated by the
regime argument: macro/regime effects are time-local, so old-regime
samples may pollute training - this tests whether truncating history
helps or hurts on identical features and hyper-parameters.
"""
from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest import ann_stats, portfolio, rank_ic  # noqa: E402
from backtest_v6 import D5  # noqa: E402
from features_ta import TA  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
D6 = D5 + TA
WINDOWS = {"expanding": None, "roll5y": 5, "roll3y": 3}


def run(df, feats, name, win):
    rows, preds = [], []
    for y in range(2020, 2027):
        tr = df[df["year"] < y]
        if win is not None:
            tr = tr[tr["year"] >= y - win]
        te = df[df["year"] == y].copy()
        m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05,
                              num_leaves=31, subsample=0.8,
                              colsample_bytree=0.8, random_state=42,
                              verbose=-1)
        m.fit(tr[feats], tr["fwd_ret_5"])
        te["pred"] = m.predict(te[feats])
        rows.append({"model": name, "year": y,
                     "rank_ic": rank_ic(te, "pred"), "n": len(te)})
        preds.append(te[["date", "code", "fwd_ret_5", "pred"]])
        print(f"{name} {y}: ic={rows[-1]['rank_ic']:.4f} "
              f"train={len(tr)}", flush=True)
    allp = pd.concat(preds, ignore_index=True)
    allp.to_parquet(ROOT / f"reports/preds_win_{name}.parquet",
                    index=False)
    st = ann_stats(portfolio(allp, "pred"))
    rows.append({"model": name, "year": "ALL", "rank_ic": None, **st})
    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_parquet(ROOT / "data/dataset_v6.parquet")
    df["year"] = df["date"].dt.year
    res = pd.concat([run(df, D6, n, w) for n, w in WINDOWS.items()],
                    ignore_index=True)
    wide = res[res["year"] != "ALL"].pivot(index="year", columns="model",
                                           values="rank_ic")
    wide = wide.reindex(columns=list(WINDOWS))
    wide.loc["mean"] = wide.mean()
    allrows = res[res["year"] == "ALL"].set_index("model") \
        .reindex(list(WINDOWS))
    lines = ["# P8：训练窗口敏感性（D6 配置）", "",
             "扩张窗口 vs 滚动 5 年 vs 滚动 3 年，其余口径不变。", "",
             "## 逐年 RankIC", "", wide.round(4).to_markdown(), "",
             "## 组合（全期拼接）", "",
             allrows[["total_return", "ann_return", "max_drawdown",
                      "sharpe"]].round(4).to_markdown(), ""]
    (ROOT / "reports/p8_windows.md").write_text("\n".join(lines),
                                                encoding="utf-8")
    print(wide.round(4).to_string())
    print(allrows[["ann_return", "max_drawdown", "sharpe"]].round(4)
          .to_string())


if __name__ == "__main__":
    main()
