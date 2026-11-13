"""Intraday v1 add-on test at L4 caliber (dataset_pit3, rank label,
ens5, 7d purge). Jobs: control L4_rank reproduction, +intraday,
intraday-only. Selection on 2020-2025 IC; 2026 quasi-holdout.

Caveat (inherent to the data): ticks start 2020-01, so for test year
2020 the intraday features are all-NaN in training; the real add-on
window is 2021+. Report IC21-25 alongside IC20-25.
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

INTRA = ["ofi_day", "ofi_am", "ofi_pm", "ofi_last30", "buy_share",
         "large_share", "avg_trade_size_rel", "tick_count_rel",
         "vol_conc_am", "vol_conc_last30", "vwap_dev", "amihud_tick",
         "range_tick", "auction_vol_share", "auction_gap"]


def run(df, didx, name, feats):
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
    sel = np.mean([r["rank_ic"] for r in rows if r["year"] <= 2025])
    sel21 = np.mean([r["rank_ic"] for r in rows if 2021 <= r["year"] <= 2025])
    print(f"== {name}: IC20-25={sel:.4f} IC21-25={sel21:.4f} "
          f"IC26={rows[-1]['rank_ic']:.4f} ann={st['ann_return']*100:.1f}% "
          f"sharpe={st['sharpe']:.2f} mdd={st['max_drawdown']*100:.1f}%",
          flush=True)
    return rows


def main() -> None:
    df, didx = load("dataset_pit3.parquet")
    intra = pd.read_parquet(ROOT / "data" / "intraday_v1.parquet")
    cols = ["date", "code"] + INTRA
    before = len(df)
    df = df.merge(intra[cols], on=["date", "code"], how="left")
    cov = df[(df["year"] >= 2020)].groupby("year")["ofi_day"].apply(
        lambda s: s.notna().mean())
    print(f"merged {before} rows; intraday coverage 2020+: "
          f"{cov.round(3).to_dict()}", flush=True)
    all_rows = []
    jobs = [("L4_rank", FEATS),
            ("L4_rank_intra", FEATS + INTRA),
            ("L4_intra_only", INTRA)]
    for name, feats in jobs:
        t0 = time.time()
        all_rows += run(df, didx, name, feats)
        print(f"[{name} {time.time()-t0:.0f}s]", flush=True)
        pd.DataFrame(all_rows).to_csv(ROOT / "reports/intraday_runs.csv",
                                      index=False)
    print("=== done ===", flush=True)


if __name__ == "__main__":
    main()
