"""Classic technical-indicator pack (all causal, computed per stock).

The P1 base set is hand-rolled returns/volatility only; the standard TA
zoo was never systematically included. This pack:
  MACD: ta_dif (EMA12-EMA26, /close), ta_dea (signal), ta_hist
  RSI14 (Wilder), KDJ (K/D/J, 9-3-3), Bollinger %B + bandwidth (20, 2)
  ATR14/close, OBV 20d slope (normalized), MFI14, CCI20, WR14,
  BIAS12, ADX14 (Wilder), last-day candle body/shadow ratios.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TA = ["ta_dif", "ta_dea", "ta_hist", "ta_rsi", "ta_k", "ta_d", "ta_j",
      "ta_boll_b", "ta_boll_bw", "ta_atr", "ta_obv_slope", "ta_mfi",
      "ta_cci", "ta_wr", "ta_bias12", "ta_adx", "ta_body", "ta_shadow_u",
      "ta_shadow_l"]


def _wilder(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def build_ta(ds: pd.DataFrame) -> pd.DataFrame:
    daily = pd.read_parquet(
        ROOT / "data/daily.parquet",
        columns=["date", "code", "open", "high", "low", "close",
                 "volume", "amount"])
    out = []
    for code, g in daily.groupby("code", sort=False):
        g = g.sort_values("date").copy()
        c, h, l, o, v = g["close"], g["high"], g["low"], g["open"], \
            g["volume"]
        ema12 = c.ewm(span=12, adjust=False).mean()
        ema26 = c.ewm(span=26, adjust=False).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9, adjust=False).mean()
        g["ta_dif"] = dif / c
        g["ta_dea"] = dea / c
        g["ta_hist"] = (dif - dea) / c
        d1 = c.diff()
        up = _wilder(d1.clip(lower=0), 14)
        dn = _wilder(-d1.clip(upper=0), 14)
        g["ta_rsi"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
        ll = l.rolling(9).min()
        hh = h.rolling(9).max()
        rsv = (c - ll) / (hh - ll).replace(0, np.nan) * 100
        k = rsv.ewm(alpha=1 / 3, adjust=False).mean()
        dd = k.ewm(alpha=1 / 3, adjust=False).mean()
        g["ta_k"], g["ta_d"], g["ta_j"] = k, dd, 3 * k - 2 * dd
        ma20, sd20 = c.rolling(20).mean(), c.rolling(20).std()
        g["ta_boll_b"] = (c - (ma20 - 2 * sd20)) / (4 * sd20)
        g["ta_boll_bw"] = 4 * sd20 / ma20
        pc = c.shift(1)
        tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()],
                       axis=1).max(axis=1)
        g["ta_atr"] = _wilder(tr, 14) / c
        obv = (np.sign(d1).fillna(0) * v).cumsum()
        g["ta_obv_slope"] = (obv - obv.shift(20)) / \
            (v.rolling(20).mean() * 20)
        tp = (h + l + c) / 3
        mf = tp * v
        pos = mf.where(tp > tp.shift(1), 0).rolling(14).sum()
        neg = mf.where(tp < tp.shift(1), 0).rolling(14).sum()
        g["ta_mfi"] = 100 - 100 / (1 + pos / neg.replace(0, np.nan))
        g["ta_cci"] = (tp - tp.rolling(20).mean()) / \
            (0.015 * tp.rolling(20).apply(
                lambda x: np.mean(np.abs(x - x.mean())), raw=True))
        g["ta_wr"] = (h.rolling(14).max() - c) / \
            (h.rolling(14).max() - l.rolling(14).min()).replace(0, np.nan) \
            * -100
        g["ta_bias12"] = c / c.rolling(12).mean() - 1
        up_dm = h.diff().clip(lower=0)
        dn_dm = -l.diff().clip(upper=0)
        up_dm = up_dm.where(up_dm > dn_dm, 0)
        dn_dm = dn_dm.where(dn_dm > up_dm, 0)
        atr_w = _wilder(tr, 14)
        pdi = 100 * _wilder(up_dm, 14) / atr_w
        mdi = 100 * _wilder(dn_dm, 14) / atr_w
        dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
        g["ta_adx"] = _wilder(dx, 14)
        rng = (h - l).replace(0, np.nan)
        g["ta_body"] = (c - o) / rng
        g["ta_shadow_u"] = (h - pd.concat([c, o], axis=1).max(axis=1)) / rng
        g["ta_shadow_l"] = (pd.concat([c, o], axis=1).min(axis=1) - l) / rng
        out.append(g[["date", "code"] + TA])
    ta = pd.concat(out, ignore_index=True)
    return ds[["date", "code"]].merge(ta, on=["date", "code"], how="left")
