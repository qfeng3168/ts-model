"""P10 regularization scan on the L3 caliber (dataset_pit2, exec
label close(t+1)->close(t+6)), rank-label target, purge embargo.

Scan stage uses 3 seeds (42/7/123) to keep the grid affordable;
any config beating base by >0.002 IC20-25 gets an ens5 confirmation
before adoption. Selection metric: mean RankIC 2020-2025 only;
2026 is the frozen quasi-holdout, reported but never selected on.

Configs: min_child_samples / lambda_l2 / num_leaves variations,
one combined "strong" config, and an early-stopping variant
(n_estimators cap 1200, validation = year y-1, train purged before
it). Every config's predictions are saved immediately.
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
from train_pit2 import EMBARGO, FEATS, PARAMS, load  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SCAN_SEEDS = (42, 7, 123)

CONFIGS = [
    ("base", {}),
    ("mcs50", dict(min_child_samples=50)),
    ("mcs100", dict(min_child_samples=100)),
    ("mcs200", dict(min_child_samples=200)),
    ("l2_1", dict(lambda_l2=1.0)),
    ("l2_5", dict(lambda_l2=5.0)),
    ("l2_20", dict(lambda_l2=20.0)),
    ("leaves15", dict(num_leaves=15)),
    ("strong", dict(num_leaves=15, min_child_samples=100,
                    lambda_l2=5.0)),
]


def run(df, didx, name, overrides):
    params = {**PARAMS, **overrides}
    rows, preds = [], []
    for y in range(2020, 2027):
        te = df[df["year"] == y].copy()
        first_idx = didx[te["date"].min()]
        tr = df[(df["year"] < y)
                & (df["date"].map(didx) <= first_idx - EMBARGO - 1)]
        ytr = tr.groupby("date")["fwd_ret_5"].rank(pct=True)
        ps = []
        for s in SCAN_SEEDS:
            m = lgb.LGBMRegressor(random_state=s, **params)
            m.fit(tr[FEATS], ytr)
            ps.append(m.predict(te[FEATS]))
        te["pred"] = np.mean(ps, axis=0)
        rows.append({"model": name, "year": y,
                     "rank_ic": rank_ic(te, "pred"), "n": len(te)})
        preds.append(te[["date", "code", "fwd_ret_5", "pred"]])
        print(f"{name} {y}: ic={rows[-1]['rank_ic']:.4f}", flush=True)
    allp = pd.concat(preds, ignore_index=True)
    allp.to_parquet(ROOT / f"reports/preds_reg_{name}.parquet",
                    index=False)
    st = ann_stats(portfolio(allp, "pred"))
    sel = np.mean([r["rank_ic"] for r in rows if r["year"] <= 2025])
    print(f"== reg_{name}: IC20-25={sel:.4f} "
          f"IC26={rows[-1]['rank_ic']:.4f} "
          f"ann={st['ann_return']*100:.1f}% sharpe={st['sharpe']:.2f} "
          f"mdd={st['max_drawdown']*100:.1f}%", flush=True)
    return rows


def run_es(df, didx):
    name = "es"
    rows, preds = [], []
    for y in range(2020, 2027):
        te = df[df["year"] == y].copy()
        va = df[df["year"] == y - 1]
        va_first = didx[va["date"].min()]
        # core train: years before y-1, tail purged so no train
        # label window reaches into the validation year
        tr = df[(df["year"] < y - 1)
                & (df["date"].map(didx) <= va_first - EMBARGO - 1)]
        ytr = tr.groupby("date")["fwd_ret_5"].rank(pct=True)
        yva = va.groupby("date")["fwd_ret_5"].rank(pct=True)
        ps, best = [], []
        for s in SCAN_SEEDS:
            m = lgb.LGBMRegressor(random_state=s, n_estimators=1200,
                                  learning_rate=0.05, num_leaves=31,
                                  subsample=0.8, colsample_bytree=0.8,
                                  verbose=-1)
            m.fit(tr[FEATS], ytr, eval_set=[(va[FEATS], yva)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
            best.append(m.best_iteration_)
            ps.append(m.predict(te[FEATS]))
        te["pred"] = np.mean(ps, axis=0)
        rows.append({"model": name, "year": y,
                     "rank_ic": rank_ic(te, "pred"), "n": len(te)})
        preds.append(te[["date", "code", "fwd_ret_5", "pred"]])
        print(f"{name} {y}: ic={rows[-1]['rank_ic']:.4f} "
              f"best_iter={best}", flush=True)
    allp = pd.concat(preds, ignore_index=True)
    allp.to_parquet(ROOT / "reports/preds_reg_es.parquet", index=False)
    st = ann_stats(portfolio(allp, "pred"))
    sel = np.mean([r["rank_ic"] for r in rows if r["year"] <= 2025])
    print(f"== reg_es: IC20-25={sel:.4f} "
          f"IC26={rows[-1]['rank_ic']:.4f} "
          f"ann={st['ann_return']*100:.1f}% sharpe={st['sharpe']:.2f} "
          f"mdd={st['max_drawdown']*100:.1f}%", flush=True)
    return rows


def main() -> None:
    df, didx = load("dataset_pit2.parquet")
    all_rows = []
    out = ROOT / "reports/reg_scan.csv"
    for name, ov in CONFIGS:
        t0 = time.time()
        all_rows += run(df, didx, name, ov)
        print(f"[reg_{name} {time.time()-t0:.0f}s]", flush=True)
        pd.DataFrame(all_rows).to_csv(out, index=False)
    t0 = time.time()
    all_rows += run_es(df, didx)
    print(f"[reg_es {time.time()-t0:.0f}s]", flush=True)
    pd.DataFrame(all_rows).to_csv(out, index=False)


if __name__ == "__main__":
    main()
