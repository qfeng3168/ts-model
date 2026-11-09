"""P10 cross-sectional rank-normalization variant: every feature
replaced by its per-date percentile rank within that day's member
universe, rank-label target, L3 caliber (dataset_pit2), ens5 +
purge. Tests whether monotone-invariant inputs generalize better
than raw feature scales. Selection on 2020-2025 as usual.
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


def main() -> None:
    df, didx = load("dataset_pit2.parquet")
    t0 = time.time()
    df[FEATS] = df.groupby("date")[FEATS].rank(pct=True)
    print(f"[rank transform {time.time()-t0:.0f}s]", flush=True)
    rows, preds = [], []
    for y in range(2020, 2027):
        te = df[df["year"] == y].copy()
        first_idx = didx[te["date"].min()]
        tr = df[(df["year"] < y)
                & (df["date"].map(didx) <= first_idx - EMBARGO - 1)]
        ytr = tr.groupby("date")["fwd_ret_5"].rank(pct=True)
        ps = []
        for s in SEEDS:
            m = lgb.LGBMRegressor(random_state=s, **PARAMS)
            m.fit(tr[FEATS], ytr)
            ps.append(m.predict(te[FEATS]))
        te["pred"] = np.mean(ps, axis=0)
        rows.append({"model": "ranknorm", "year": y,
                     "rank_ic": rank_ic(te, "pred"), "n": len(te)})
        preds.append(te[["date", "code", "fwd_ret_5", "pred"]])
        print(f"ranknorm {y}: ic={rows[-1]['rank_ic']:.4f}",
              flush=True)
    allp = pd.concat(preds, ignore_index=True)
    allp.to_parquet(ROOT / "reports/preds_ranknorm.parquet",
                    index=False)
    pd.DataFrame(rows).to_csv(ROOT / "reports/ranknorm.csv",
                              index=False)
    st = ann_stats(portfolio(allp, "pred"))
    sel = np.mean([r["rank_ic"] for r in rows if r["year"] <= 2025])
    print(f"== ranknorm: IC20-25={sel:.4f} "
          f"IC26={rows[-1]['rank_ic']:.4f} "
          f"ann={st['ann_return']*100:.1f}% sharpe={st['sharpe']:.2f} "
          f"mdd={st['max_drawdown']*100:.1f}%", flush=True)


if __name__ == "__main__":
    main()
