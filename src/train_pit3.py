"""L4 open-execution caliber: rank + raw labels on dataset_pit3,
ens5 + purge, same engine as the ladder. Selection on 2020-2025.
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
from train_pit2 import EMBARGO, FEATS, PARAMS, SEEDS, load  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def run(df, didx, name, rank_label):
    rows, preds = [], []
    for y in range(2020, 2027):
        te = df[df["year"] == y].copy()
        first_idx = didx[te["date"].min()]
        tr = df[(df["year"] < y)
                & (df["date"].map(didx) <= first_idx - EMBARGO - 1)]
        ytr = tr["fwd_ret_5"]
        if rank_label:
            ytr = tr.groupby("date")["fwd_ret_5"].rank(pct=True)
        ps = []
        for s in SEEDS:
            m = lgb.LGBMRegressor(random_state=s, **PARAMS)
            m.fit(tr[FEATS], ytr)
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


def main() -> None:
    df, didx = load("dataset_pit3.parquet")
    all_rows = []
    for name, rl in [("L4_rank", True), ("L4_raw", False)]:
        t0 = time.time()
        all_rows += run(df, didx, name, rl)
        print(f"[{name} {time.time()-t0:.0f}s]", flush=True)
        pd.DataFrame(all_rows).to_csv(ROOT / "reports/pit3_runs.csv",
                                      index=False)


if __name__ == "__main__":
    main()
