"""Industry-conditioned commodity exposure features (P5/D2).

P4 showed raw market-wide macro levels hurt: a feature that is identical
for every stock on a day can only act through interactions. Here each
commodity's 5d/20d log-return is delivered ONLY to stocks in the exposed
industries (NaN elsewhere, so trees can split on exposure itself):

  copper/gold -> 有色金属 (CSRC C32 有色金属冶炼和压延加工业 + 采掘 B09)
  crude       -> 石油加工 C25 / 化学原料 C26 / 化学纤维 C28 / 橡胶塑料 C29
  fx (USDCNY) -> 出口敞口近似: 电子 C39 / 家电 C38 部分无法细分, 用制造业
                 大类 C 全行业给弱敞口 (值乘 0.5)
CSRC codes from data/f10/industry.parquet (e.g. 'C39计算机、通信...').
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

EXPOSURE = {
    "cu": ("copper_cu0_shfe", "close", ["C32", "B09"]),
    "au": ("gold_sge_au9999", "close", ["C32", "B09"]),
    "oil": ("crude_sc", "收盘价", ["C25", "C26", "C28", "C29"]),
}


def build_cmdty(ds: pd.DataFrame) -> pd.DataFrame:
    ind = pd.read_parquet(ROOT / "data/f10/industry.parquet")
    code2ind = dict(zip(ind["code"], ind["industry"].str[:3]))
    res = ds[["date", "code"]].copy()
    t = np.sort(res["date"].unique())
    tpos = {d: i for i, d in enumerate(t)}
    ti = res["date"].map(tpos).to_numpy()
    for key, (fname, col, prefixes) in EXPOSURE.items():
        raw = pd.read_parquet(ROOT / f"data/extra/{fname}.parquet")
        dcol = "日期" if "日期" in raw.columns else raw.columns[0]
        s = raw[[dcol, col]].copy()
        s.columns = ["date", "v"]
        s["date"] = pd.to_datetime(s["date"])
        s["v"] = pd.to_numeric(s["v"], errors="coerce")
        s = s.dropna().sort_values("date")
        pos = np.searchsorted(s["date"].to_numpy(), t,
                              side="left") - 1
        lv = np.log(s["v"].to_numpy())
        for k, name in ((5, "r5"), (20, "r20")):
            chg = np.full(len(s), np.nan)
            chg[k:] = lv[k:] - lv[:-k]
            vals = np.where(pos >= 0,
                            chg[np.clip(pos, 0, None)], np.nan)
            exposed = res["code"].map(code2ind).isin(prefixes).to_numpy()
            res[f"exp_{key}_{name}"] = np.where(exposed, vals[ti],
                                                np.nan)
    fx = pd.read_parquet(ROOT / "data/extra/usdcny_mid_safe.parquet")
    s = fx[["日期", "美元"]].copy()
    s.columns = ["date", "v"]
    s["date"] = pd.to_datetime(s["date"])
    s["v"] = pd.to_numeric(s["v"], errors="coerce")
    s = s.dropna().sort_values("date")
    pos = np.searchsorted(s["date"].to_numpy(), t, side="left") - 1
    lv = np.log(s["v"].to_numpy())
    chg = np.full(len(s), np.nan)
    chg[20:] = lv[20:] - lv[:-20]
    vals = np.where(pos >= 0, chg[np.clip(pos, 0, None)], np.nan)
    manuf = res["code"].map(code2ind).str.startswith("C").to_numpy()
    res["exp_fx_r20"] = np.where(manuf, vals[ti] * 0.5, np.nan)
    return res
