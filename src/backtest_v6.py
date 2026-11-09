"""P6-local ablations: D5 base, +TA pack, +concept pack, +both.

Predictions of every variant are saved immediately after it finishes
(reports/preds_<variant>.parquet) per the checkpoint rule.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import lightgbm as lgb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest import ann_stats, portfolio, rank_ic  # noqa: E402
from backtest_v3 import F10  # noqa: E402
from build_v4 import SECTOR  # noqa: E402
from features import FEATURES  # noqa: E402
from features_concept import CONCEPT  # noqa: E402
from features_ta import TA  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CMDTY = ["exp_cu_r5", "exp_cu_r20", "exp_au_r5", "exp_au_r20",
         "exp_oil_r5", "exp_oil_r20", "exp_fx_r20"]
SURP = ["sue", "ind_share"]
D5 = FEATURES + F10 + SECTOR + CMDTY + SURP
VARIANTS = {"D5_base": D5, "D6_ta": D5 + TA, "D7_concept": D5 + CONCEPT,
            "D8_ta_concept": D5 + TA + CONCEPT}


def run(df, feats, name):
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
    allp.to_parquet(ROOT / f"reports/preds_{name}.parquet", index=False)
    st = ann_stats(portfolio(allp, "pred"))
    rows.append({"model": name, "year": "ALL", "rank_ic": None, **st})
    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_parquet(ROOT / "data/dataset_v6.parquet")
    df["year"] = df["date"].dt.year
    res = pd.concat([run(df, f, n) for n, f in VARIANTS.items()],
                    ignore_index=True)
    wide = res[res["year"] != "ALL"].pivot(index="year", columns="model",
                                           values="rank_ic")
    wide = wide.reindex(columns=list(VARIANTS))
    wide.loc["mean"] = wide.mean()
    allrows = res[res["year"] == "ALL"].set_index("model") \
        .reindex(list(VARIANTS))
    lines = ["# P6（本地）消融：TA 技术指标包 / 概念包", "",
             "基底 D5 = P1+F10+板块+商品条件化+业绩超预期（P5 胜出）。", "",
             "## 逐年 RankIC", "", wide.round(4).to_markdown(), "",
             "## 组合（全期拼接）", "",
             allrows[["total_return", "ann_return", "max_drawdown",
                      "sharpe"]].round(4).to_markdown(), ""]
    (ROOT / "reports/p6_local_ablation.md").write_text("\n".join(lines),
                                                       encoding="utf-8")
    print(wide.round(4).to_string())
    print(allrows[["ann_return", "max_drawdown", "sharpe"]].round(4)
          .to_string())


if __name__ == "__main__":
    main()
