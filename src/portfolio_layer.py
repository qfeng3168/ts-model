"""Portfolio layer on PIT D6 predictions: vol targeting + regime
switches. Rules are canonical/few (pre-registered here) to limit
overlay overfitting; all variants reported.

Base period returns replicate backtest.portfolio() exactly (top20%,
5-day rebalance, 0.15% cost). Exposures use only info <= rebalance
date. Exposure changes pay the same 0.15% on |delta|.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest import COST, ann_stats  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def period_returns(d: pd.DataFrame, pred_col: str = "pred"):
    dates = sorted(d["date"].unique())
    rebal = dates[::5]
    rows = []
    for dt in rebal:
        g = d[d["date"] == dt]
        if len(g) < 20:
            continue
        k = max(1, int(len(g) * 0.2))
        top = g.nlargest(k, pred_col)
        rows.append({"date": dt,
                     "ret": top["fwd_ret_5"].mean() - COST})
    return pd.DataFrame(rows).set_index("date")


def apply_exposure(pr: pd.DataFrame, exp: pd.Series) -> pd.Series:
    e = exp.reindex(pr.index).fillna(1.0).clip(0, 1)
    de = e.diff().abs().fillna(0.0)
    return e * pr["ret"] - de * COST


def main() -> None:
    preds = pd.read_parquet(ROOT / "reports/preds_D6_pit.parquet")
    pr = period_returns(preds)
    ix = pd.read_parquet(ROOT / "data/extra/index_hs300.parquet")
    ix["date"] = pd.to_datetime(ix["date"])
    ix = ix.sort_values("date").set_index("date")["close"]
    ix = ix[~ix.index.duplicated(keep="last")]
    qv = pd.read_parquet(ROOT / "data/extra/qvix_300etf.parquet")
    qdate = "date" if "date" in qv.columns else qv.columns[0]
    qcol = [c for c in qv.columns if c != qdate][0]
    qv = qv.rename(columns={qdate: "date", qcol: "qvix"})
    qv["date"] = pd.to_datetime(qv["date"])
    qv = qv.sort_values("date").set_index("date")["qvix"]
    qv = qv[~qv.index.duplicated(keep="last")]

    T = pr.index
    ma120 = ix.rolling(120).mean()
    trend = (ix > ma120).astype(float)
    trend_soft = trend.replace({1.0: 1.0, 0.0: 0.5})
    q80 = qv.rolling(250, min_periods=60).quantile(0.8)
    calm = (qv <= q80).astype(float).replace({1.0: 1.0, 0.0: 0.5})

    def at(s: pd.Series) -> pd.Series:
        return s.reindex(ix.index.union(T)).ffill().reindex(T)

    # vol targeting: target 15% ann; period vol from trailing 12 periods
    pvol = pr["ret"].rolling(12).std() * np.sqrt(50)
    vt = (0.15 / pvol).clip(0, 1).shift(1)

    variants = {
        "base (full)": pd.Series(1.0, index=T),
        "trend hard (0/1)": at(trend),
        "trend soft (0.5/1)": at(trend_soft),
        "qvix calm (0.5/1)": at(calm),
        "vol target 15%": vt,
        "vt + trend soft": vt * at(trend_soft),
        "vt + trend soft + qvix": vt * at(trend_soft) * at(calm),
    }
    out = []
    for name, e in variants.items():
        r = pr["ret"] if name.startswith("base") else apply_exposure(pr, e)
        st = ann_stats(r)
        out.append({"variant": name,
                    "ann%": round(st["ann_return"] * 100, 1),
                    "sharpe": round(st["sharpe"], 2),
                    "mdd%": round(st["max_drawdown"] * 100, 1),
                    "avg_exp": round(float(e.reindex(T).fillna(1).mean()), 2)})
        print(out[-1], flush=True)
    pd.DataFrame(out).to_csv(ROOT / "reports/pit_portfolio_layer.csv",
                             index=False)


if __name__ == "__main__":
    main()
