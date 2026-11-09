"""P10 model-layer variants on the PIT universe (member rows only).

- ens5: D6 config with 5 seeds, predictions averaged.
- ranklabel: target = per-date percentile rank of fwd_ret_5.
- drop_<group>: D6 minus one feature group (ablation under PIT).
Each variant saves preds immediately (checkpoint rule).
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
GROUPS = {"pv": FEATURES, "f10": F10, "sector": SECTOR, "cmdty": CMDTY,
          "surp": SURP, "ta": TA}
ALL = FEATURES + F10 + SECTOR + CMDTY + SURP + TA
PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=31,
              subsample=0.8, colsample_bytree=0.8, verbose=-1)


def run(df, feats, name, seeds=(42,), rank_label=False):
    rows, preds = [], []
    for y in range(2020, 2027):
        tr, te = df[df["year"] < y], df[df["year"] == y].copy()
        ytr = tr["fwd_ret_5"]
        if rank_label:
            ytr = tr.groupby("date")["fwd_ret_5"].rank(pct=True)
        ps = []
        for s in seeds:
            m = lgb.LGBMRegressor(random_state=s, **PARAMS)
            m.fit(tr[feats], ytr)
            ps.append(m.predict(te[feats]))
        te["pred"] = np.mean(ps, axis=0)
        rows.append({"model": name, "year": y,
                     "rank_ic": rank_ic(te, "pred"), "n": len(te)})
        preds.append(te[["date", "code", "fwd_ret_5", "pred"]])
        print(f"{name} {y}: ic={rows[-1]['rank_ic']:.4f}", flush=True)
    allp = pd.concat(preds, ignore_index=True)
    allp.to_parquet(ROOT / f"reports/preds_{name}.parquet", index=False)
    st = ann_stats(portfolio(allp, "pred"))
    mic = np.mean([r["rank_ic"] for r in rows])
    print(f"== {name}: meanIC={mic:.4f} ann={st['ann_return']*100:.1f}% "
          f"sharpe={st['sharpe']:.2f} mdd={st['max_drawdown']*100:.1f}%",
          flush=True)
    return {"model": name, "mean_ic": mic, **st}


def main() -> None:
    df = pd.read_parquet(ROOT / "data/dataset_union.parquet")
    df = df[df["member"]].copy()
    df["year"] = df["date"].dt.year
    print(f"member rows {len(df):,}", flush=True)
    jobs = [
        ("pit_ens5", ALL, dict(seeds=(42, 7, 123, 2024, 555))),
        ("pit_ranklabel", ALL, dict(rank_label=True)),
    ]
    for gname, gcols in GROUPS.items():
        feats = [c for c in ALL if c not in set(gcols)]
        jobs.append((f"pit_drop_{gname}", feats, {}))
    results = []
    for name, feats, kw in jobs:
        t0 = time.time()
        results.append(run(df, feats, name, **kw))
        print(f"[{name} done in {time.time()-t0:.0f}s]", flush=True)
        pd.DataFrame(results).to_csv(
            ROOT / "reports/pit_variants.csv", index=False)


if __name__ == "__main__":
    main()
