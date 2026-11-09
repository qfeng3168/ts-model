"""P10 caliber ladder: one fix at a time, ens5 everywhere (so seed
noise is not mistaken for a caliber effect).

  L0  old caliber (= backtest_union D6_pit, already have): no purge,
      F10 same-day, label close(t)->close(t+5), single seed.
  L1  + purge: 7 full trading days embargoed before each test year
      (covers both the 5-day and the 6-day label spans).
  L2  + F10 family effective +1 trading day (dataset_pit2a).
  L3  + execution label close(t+1)->close(t+6) (dataset_pit2).

Selection metric: mean RankIC 2020-2025. 2026 = frozen quasi-
holdout, reported only. Portfolio uses the legacy flat-cost
portfolio() for ladder comparability; the turnover-weighted cost
model lives in the robustness pack.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest import ann_stats, portfolio, rank_ic  # noqa: E402
from backtest_v3 import F10  # noqa: E402
from backtest_v6 import CMDTY, SURP  # noqa: E402
from build_v4 import SECTOR  # noqa: E402
from features import FEATURES  # noqa: E402
from features_ta import TA  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FEATS = FEATURES + F10 + SECTOR + CMDTY + SURP + TA
PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=31,
              subsample=0.8, colsample_bytree=0.8, verbose=-1)
SEEDS = (42, 7, 123, 2024, 555)
EMBARGO = 7  # full trading days dropped before the test year


def run(df, didx, name):
    rows, preds = [], []
    for y in range(2020, 2027):
        te = df[df["year"] == y].copy()
        first_idx = didx[te["date"].min()]
        tr = df[(df["year"] < y)
                & (df["date"].map(didx) <= first_idx - EMBARGO - 1)]
        ps = []
        for s in SEEDS:
            m = lgb.LGBMRegressor(random_state=s, **PARAMS)
            m.fit(tr[FEATS], tr["fwd_ret_5"])
            ps.append(m.predict(te[FEATS]))
        te["pred"] = np.mean(ps, axis=0)
        rows.append({"model": name, "year": y,
                     "rank_ic": rank_ic(te, "pred"), "n": len(te)})
        preds.append(te[["date", "code", "fwd_ret_5", "pred"]])
        print(f"{name} {y}: ic={rows[-1]['rank_ic']:.4f}", flush=True)
    allp = pd.concat(preds, ignore_index=True)
    allp.to_parquet(ROOT / f"reports/preds_{name}.parquet", index=False)
    st = ann_stats(portfolio(allp, "pred"))
    sel = np.mean([r["rank_ic"] for r in rows if r["year"] <= 2025])
    print(f"== {name}: IC20-25={sel:.4f} IC26={rows[-1]['rank_ic']:.4f} "
          f"ann={st['ann_return']*100:.1f}% sharpe={st['sharpe']:.2f} "
          f"mdd={st['max_drawdown']*100:.1f}%", flush=True)
    return rows


def load(fname):
    df = pd.read_parquet(ROOT / f"data/{fname}")
    df = df[df["member"]].copy()
    df["year"] = df["date"].dt.year
    dates = np.sort(df["date"].unique())
    return df, {d: i for i, d in enumerate(dates)}


def main() -> None:
    all_rows = []
    for name, fname in [("L1_purge", "dataset_union.parquet"),
                        ("L2_f10delay", "dataset_pit2a.parquet"),
                        ("L3_exec", "dataset_pit2.parquet")]:
        df, didx = load(fname)
        t0 = time.time()
        all_rows += run(df, didx, name)
        print(f"[{name} {time.time()-t0:.0f}s]", flush=True)
        pd.DataFrame(all_rows).to_csv(ROOT / "reports/pit2_ladder.csv",
                                      index=False)


if __name__ == "__main__":
    main()
