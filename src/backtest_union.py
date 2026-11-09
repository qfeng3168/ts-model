"""Survivorship-bias measurement: D6 (exact production config, 59
features) on the point-in-time HS300 universe vs the current-300
universe. Training and test rows are both restricted to membership
days in the union run. Same engine as backtest_v6.run().
"""
from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest import ann_stats, portfolio, rank_ic  # noqa: E402
from backtest_v3 import F10  # noqa: E402
from backtest_v6 import CMDTY, D5, SURP  # noqa: E402
from build_v4 import SECTOR  # noqa: E402
from features import FEATURES  # noqa: E402
from features_ta import TA  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FEATS = FEATURES + F10 + SECTOR + CMDTY + SURP + TA
assert len(FEATS) == 59, len(FEATS)


def run(df: pd.DataFrame, name: str) -> pd.DataFrame:
    rows, preds = [], []
    for y in range(2020, 2027):
        tr = df[df["year"] < y]
        te = df[df["year"] == y].copy()
        m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05,
                              num_leaves=31, subsample=0.8,
                              colsample_bytree=0.8, random_state=42,
                              verbose=-1)
        m.fit(tr[FEATS], tr["fwd_ret_5"])
        te["pred"] = m.predict(te[FEATS])
        rows.append({"model": name, "year": y,
                     "rank_ic": rank_ic(te, "pred"), "n": len(te)})
        preds.append(te[["date", "code", "fwd_ret_5", "pred"]])
        print(f"{name} {y}: ic={rows[-1]['rank_ic']:.4f} n={len(te)}",
              flush=True)
    allp = pd.concat(preds, ignore_index=True)
    allp.to_parquet(ROOT / f"reports/preds_{name}.parquet", index=False)
    st = ann_stats(portfolio(allp, "pred"))
    rows.append({"model": name, "year": "ALL", "rank_ic": None, **st})
    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_parquet(ROOT / "data/dataset_union.parquet")
    df["year"] = df["date"].dt.year
    mem = df[df["member"]].copy()
    print(f"union rows={len(df):,} member rows={len(mem):,} "
          f"member stocks={mem['code'].nunique()}", flush=True)
    res = run(mem, "D6_pit")
    print(res.to_string(), flush=True)
    res["year"] = res["year"].astype(str)
    res.to_parquet(ROOT / "reports/pit_d6_result.parquet", index=False)


if __name__ == "__main__":
    main()
