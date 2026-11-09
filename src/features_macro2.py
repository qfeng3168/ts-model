"""Round-2 macro / market-structure / surprise features.

Daily series (lagged 1 trading day, same convention as features_extra):
  rate_shibor1w, rate_shibor_chg20, rate_cn10y, rate_cn10y_chg20,
  rate_spread_10y2y, rate_lpr1y
  mgn_bal_chg20 (SSE margin balance 20d % change), mgn_short_share
  nb_flow5, nb_flow20 (northbound net inflow sums; source series may
  degrade after the 2024-08 disclosure change - values used as-is)
  crude_r5, crude_r20
  idx_ret20, idx_vol20, idx_dd60 (HS300 regime), mkt_breadth5,
  mkt_limitup (share of universe with 1d return >= 9.8%)

Monthly series with fixed release lags (approximation of publication
dates; using the statistical month directly would be look-ahead):
  mon_pmi (usable from the 2nd of next month),
  mon_cpi_yoy, mon_ppi_yoy, mon_m2_yoy, mon_sf_yoy (16th of next month)

Per-stock (point-in-time by notice_date):
  sue - earnings surprise proxy: latest report's net-profit YoY minus
        the mean YoY of the same quarter over the prior 3 years,
        divided by their std (seasonal random-walk SUE).
  ind_share - latest report revenue / industry total revenue of the
        same report period (industry position proxy).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EXTRA = ROOT / "data" / "extra"


def _daily(df: pd.DataFrame, dcol: str, vcol: str) -> pd.DataFrame:
    s = df[[dcol, vcol]].copy()
    s.columns = ["date", "v"]
    s["date"] = pd.to_datetime(s["date"])
    s["v"] = pd.to_numeric(s["v"], errors="coerce")
    return s.dropna().sort_values("date").drop_duplicates("date")


def _asof_lag1(s: pd.DataFrame, tdates: np.ndarray,
               fn) -> np.ndarray:
    pos = np.searchsorted(s["date"].to_numpy(), tdates, side="left") - 1
    vals = fn(s["v"].to_numpy())
    return np.where(pos >= 0, vals[np.clip(pos, 0, None)], np.nan)


def _chg(k: int):
    def f(v: np.ndarray) -> np.ndarray:
        out = np.full(len(v), np.nan)
        out[k:] = v[k:] / v[:-k] - 1
        return out
    return f


def _rollsum(k: int):
    def f(v: np.ndarray) -> np.ndarray:
        return pd.Series(v).rolling(k).sum().to_numpy()
    return f


def build_macro2(ds: pd.DataFrame) -> pd.DataFrame:
    t = np.sort(ds["date"].unique())
    out: dict[str, np.ndarray] = {"date": t}
    sh = pd.read_parquet(EXTRA / "shibor.parquet")
    s = _daily(sh, "日期", "1W-定价")
    out["rate_shibor1w"] = _asof_lag1(s, t, lambda v: v)
    out["rate_shibor_chg20"] = _asof_lag1(s, t, _chg(20))
    by = pd.read_parquet(EXTRA / "bond_yield_zh_us.parquet")
    s = _daily(by, "日期", "中国国债收益率10年")
    out["rate_cn10y"] = _asof_lag1(s, t, lambda v: v)
    out["rate_cn10y_chg20"] = _asof_lag1(s, t, _chg(20))
    s2 = _daily(by, "日期", "中国国债收益率10年-2年")
    out["rate_spread_10y2y"] = _asof_lag1(s2, t, lambda v: v)
    lpr = pd.read_parquet(EXTRA / "lpr.parquet")
    s = _daily(lpr, "TRADE_DATE", "LPR1Y")
    out["rate_lpr1y"] = _asof_lag1(s, t, lambda v: v)
    mg = pd.read_parquet(EXTRA / "margin_sse.parquet")
    s = _daily(mg, "信用交易日期", "融资余额")
    out["mgn_bal_chg20"] = _asof_lag1(s, t, _chg(20))
    nb = pd.read_parquet(EXTRA / "hsgt_north.parquet")
    s = _daily(nb, "日期", "当日资金流入")
    out["nb_flow5"] = _asof_lag1(s, t, _rollsum(5))
    out["nb_flow20"] = _asof_lag1(s, t, _rollsum(20))
    cr = pd.read_parquet(EXTRA / "crude_sc.parquet")
    s = _daily(cr, "日期", "收盘价")
    lv = np.log(s["v"].to_numpy())
    s = s.assign(v=lv)
    out["crude_r5"] = _asof_lag1(s, t, _chg(5))
    out["crude_r20"] = _asof_lag1(s, t, _chg(20))
    ix = pd.read_parquet(EXTRA / "index_hs300.parquet")
    s = _daily(ix, "date", "close")
    out["idx_ret20"] = _asof_lag1(s, t, _chg(20))
    r1 = pd.Series(np.log(s["v"])).diff()
    vol = r1.rolling(20).std().to_numpy()
    out["idx_vol20"] = _asof_lag1(s.assign(v=vol).dropna(), t,
                                   lambda v: v)
    dd = (s["v"] / s["v"].rolling(60).max() - 1).to_numpy()
    out["idx_dd60"] = _asof_lag1(s.assign(v=dd).dropna(), t, lambda v: v)
    m = pd.DataFrame(out)
    res = ds[["date", "code"]].merge(m, on="date", how="left")
    # market breadth / limit-up share from the universe itself (at t,
    # causal: uses closes up to t like the stock's own features)
    d0 = ds[["date", "code", "ret_5"]].copy()
    daily = pd.read_parquet(ROOT / "data/daily.parquet",
                            columns=["date", "code", "close"])
    daily = daily.sort_values(["code", "date"])
    daily["r1"] = daily.groupby("code")["close"].pct_change()
    b = daily.groupby("date").agg(
        mkt_breadth5_src=("r1", "size"),
        lu=("r1", lambda s: (s >= 0.098).mean()))
    b2 = d0.groupby("date")["ret_5"].apply(lambda s: (s > 0).mean())
    br = pd.DataFrame({"mkt_breadth5": b2,
                       "mkt_limitup": b["lu"]}).reset_index()
    res = res.merge(br, on="date", how="left")
    return res


def _monthly(df: pd.DataFrame, mcol: str, vcol: str, lag_day: int,
             t: np.ndarray) -> np.ndarray:
    s = df[[mcol, vcol]].copy()
    s.columns = ["m", "v"]
    txt = s["m"].astype(str)
    if txt.str.fullmatch(r"\d{6}").all():
        s["m"] = pd.to_datetime(txt, format="%Y%m")
    else:
        ym = txt.str.extract(r"(\d{4})\D+(\d{1,2})")
        if ym[0].notna().all():
            s["m"] = pd.to_datetime(ym[0] + "-" + ym[1] + "-01")
        else:
            s["m"] = pd.to_datetime(s["m"])
    s["v"] = pd.to_numeric(s["v"], errors="coerce")
    s = s.dropna().sort_values("m")
    usable = (s["m"] + pd.offsets.MonthEnd(0)).to_numpy() \
        + np.timedelta64(lag_day, "D")
    pos = np.searchsorted(usable, t, side="right") - 1
    return np.where(pos >= 0, s["v"].to_numpy()[np.clip(pos, 0, None)],
                    np.nan)


def build_monthly(ds: pd.DataFrame) -> pd.DataFrame:
    t = np.sort(ds["date"].unique())
    out: dict[str, np.ndarray] = {"date": t}
    pmi = pd.read_parquet(EXTRA / "pmi.parquet")
    out["mon_pmi"] = _monthly(pmi, "月份", "制造业-指数", 2, t)
    cpi = pd.read_parquet(EXTRA / "cpi.parquet")
    out["mon_cpi_yoy"] = _monthly(cpi, "月份", "全国-同比增长", 16, t)
    ppi = pd.read_parquet(EXTRA / "ppi.parquet")
    out["mon_ppi_yoy"] = _monthly(ppi, "月份", "当月同比增长", 16, t)
    sf = pd.read_parquet(EXTRA / "social_finance.parquet")
    sf = sf.sort_values("月份")
    sf["yoy"] = sf["社会融资规模增量"] / sf["社会融资规模增量"].shift(12) - 1
    out["mon_sf_yoy"] = _monthly(sf, "月份", "yoy", 16, t)
    m = pd.DataFrame(out)
    return ds[["date", "code"]].merge(m, on="date", how="left")


def build_surprise(ds: pd.DataFrame) -> pd.DataFrame:
    ci = pd.read_parquet(ROOT / "data/f10/core_index.parquet")
    ci = ci[ci["notice_date"] > "1990-01-01"].copy()
    ci["q"] = ci["report_date"].dt.month
    ci = ci.sort_values(["stock_code", "report_date"])
    g = ci.groupby(["stock_code", "q"])["net_profit_yoy_gr"]
    hist_mean = g.transform(lambda s: s.shift(1).rolling(3).mean())
    hist_std = g.transform(lambda s: s.shift(1).rolling(3).std())
    ci["sue"] = (ci["net_profit_yoy_gr"] - hist_mean) / (hist_std + 20.0)
    ind = pd.read_parquet(ROOT / "data/f10/industry.parquet")
    grp = {c: i[:3] for c, i in zip(ind["code"], ind["industry"])}
    ci["ind"] = ci["stock_code"].map(grp)
    tot = ci.groupby(["report_date", "ind"])["total_rev"].transform("sum")
    ci["ind_share"] = ci["total_rev"] / tot
    out = []
    for code, gg in ds.groupby("code"):
        reps = ci[ci["stock_code"] == code].sort_values("notice_date")
        dates = np.sort(gg["date"].unique())
        if len(reps) == 0:
            out.append(pd.DataFrame(
                {"date": dates, "code": code, "sue": np.nan,
                 "ind_share": np.nan}))
            continue
        nd = reps["notice_date"].to_numpy()
        pos = np.searchsorted(nd, dates, side="right") - 1
        ok = pos >= 0
        idx = np.clip(pos, 0, None)
        out.append(pd.DataFrame(
            {"date": dates, "code": code,
             "sue": np.where(ok, reps["sue"].to_numpy()[idx], np.nan),
             "ind_share": np.where(ok, reps["ind_share"].to_numpy()[idx],
                                   np.nan)}))
    return pd.concat(out, ignore_index=True)
