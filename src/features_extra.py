"""Macro factor features + per-stock block-trade / dragon-tiger features.

Macro (market-wide, from data/extra/): gold (SGE Au99.99 CNY, COMEX USD),
USD/CNY central parity, copper (SHFE, COMEX). For trading date t the
features use the latest observation with date < t (strict lag: foreign
sessions and the SGE afternoon session are not known at the A-share
close). Features per series: 5d and 20d log returns; plus gold 60d
z-score and the market fund-flow 5d mean (short history, mostly NaN).

Block trades (data/extra/dzjy.parquet, per stock, 2020+): trailing 20
trading days up to t-1: count, mean discount rate, summed turnover
share. Dragon-tiger list (data/lhb/lhb_summary.parquet): trailing 20
trading days up to t-1: appearances, mean net-buy share of turnover.
(The 上榜后N日 columns in the source are future returns - never used.)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EXTRA = ROOT / "data" / "extra"


def _series(df: pd.DataFrame, date_col: str, val_col: str) -> pd.DataFrame:
    s = df[[date_col, val_col]].copy()
    s.columns = ["date", "v"]
    s["date"] = pd.to_datetime(s["date"])
    s["v"] = pd.to_numeric(s["v"], errors="coerce")
    return s.dropna().sort_values("date").drop_duplicates("date")


def _lagged_feats(name: str, s: pd.DataFrame, tdates: np.ndarray,
                  out: dict) -> None:
    v = np.log(s["v"].to_numpy())
    d = s["date"].to_numpy()
    r5 = np.full(len(v), np.nan); r5[5:] = v[5:] - v[:-5]
    r20 = np.full(len(v), np.nan); r20[20:] = v[20:] - v[:-20]
    pos = np.searchsorted(d, tdates, side="left") - 1
    ok = pos >= 0
    idx = np.clip(pos, 0, None)
    out[f"{name}_r5"] = np.where(ok, r5[idx], np.nan)
    out[f"{name}_r20"] = np.where(ok, r20[idx], np.nan)
    if name == "gold_cny":
        mu = pd.Series(v).rolling(60).mean().to_numpy()
        sd = pd.Series(v).rolling(60).std().to_numpy()
        z = (v - mu) / sd
        out["gold_cny_z60"] = np.where(ok, z[idx], np.nan)


def build_macro(ds: pd.DataFrame) -> pd.DataFrame:
    tdates = np.sort(ds["date"].unique())
    out: dict[str, np.ndarray] = {"date": tdates}
    g = pd.read_parquet(EXTRA / "gold_sge_au9999.parquet")
    _lagged_feats("gold_cny", _series(g, "date", "close"), tdates, out)
    g = pd.read_parquet(EXTRA / "gold_gc_usd.parquet")
    _lagged_feats("gold_usd", _series(g, "date", "close"), tdates, out)
    fx = pd.read_parquet(EXTRA / "usdcny_mid_safe.parquet")
    _lagged_feats("usdcny", _series(fx, "日期", "美元"), tdates, out)
    cu = pd.read_parquet(EXTRA / "copper_cu0_shfe.parquet")
    _lagged_feats("copper_cny", _series(cu, "date", "close"), tdates, out)
    cu = pd.read_parquet(EXTRA / "copper_hg_usd.parquet")
    _lagged_feats("copper_usd", _series(cu, "date", "close"), tdates, out)
    ff = pd.read_parquet(EXTRA / "market_fund_flow.parquet")
    s = _series(ff, "日期", "主力净流入-净占比")
    s["v"] = s["v"].rolling(5).mean()
    s = s.dropna()
    pos = np.searchsorted(s["date"].to_numpy(), tdates, side="left") - 1
    out["mkt_flow5"] = np.where(pos >= 0, s["v"].to_numpy()[
        np.clip(pos, 0, None)], np.nan)
    m = pd.DataFrame(out)
    return ds[["date", "code"]].merge(m, on="date", how="left")


def _trailing(ds: pd.DataFrame, events: pd.DataFrame, prefix: str,
              cols: dict[str, str]) -> pd.DataFrame:
    """events: (code, edate, <value cols>). Trailing 20 trading days < t."""
    out = []
    ev_by_code = {c: g for c, g in events.groupby("code")}
    for code, g in ds.groupby("code"):
        dates = np.sort(g["date"].unique())
        rec: dict[str, np.ndarray] = {"date": dates, "code": code}
        ev = ev_by_code.get(code)
        if ev is None or len(ev) == 0:
            for new in cols:
                rec[new] = np.full(len(dates), np.nan)
            out.append(pd.DataFrame(rec))
            continue
        ev = ev.sort_values("edate")
        ed = ev["edate"].to_numpy()
        for new, src in cols.items():
            vals = ev[src].to_numpy(dtype=float)
            hi = np.searchsorted(ed, dates, side="left")
            lo = np.searchsorted(
                ed, dates - np.timedelta64(28, "D"), side="left")
            res = np.full(len(dates), np.nan)
            for i in range(len(dates)):
                w = vals[lo[i]:hi[i]]
                if len(w):
                    res[i] = w.sum() if new.endswith("_sum") else w.mean()
            rec[new] = res
        out.append(pd.DataFrame(rec))
    return pd.concat(out, ignore_index=True)


def build_events(ds: pd.DataFrame) -> pd.DataFrame:
    dz = pd.read_parquet(EXTRA / "dzjy.parquet")
    dz = dz.rename(columns={"交易日期": "edate", "证券代码": "code",
                            "折溢率": "disc", "成交额/流通市值": "amt_share"})
    dz["edate"] = pd.to_datetime(dz["edate"])
    dz["code"] = dz["code"].astype(str).str.zfill(6)
    dz["one"] = 1.0
    t1 = _trailing(ds, dz[["code", "edate", "disc", "amt_share", "one"]],
                   "dz", {"dz_disc_mean": "disc", "dz_amt_sum": "amt_share",
                          "dz_cnt_sum": "one"})
    lhb = pd.read_parquet(ROOT / "data/lhb/lhb_summary.parquet")
    lhb = lhb.rename(columns={"上榜日": "edate", "代码": "code",
                              "净买额占总成交比": "net_share"})
    lhb["edate"] = pd.to_datetime(lhb["edate"])
    lhb["code"] = lhb["code"].astype(str).str.zfill(6)
    lhb["one"] = 1.0
    t2 = _trailing(ds, lhb[["code", "edate", "net_share", "one"]],
                   "lhb", {"lhb_net_mean": "net_share",
                           "lhb_cnt_sum": "one"})
    return t1.merge(t2, on=["date", "code"], how="outer")
