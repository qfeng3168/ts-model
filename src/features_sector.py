"""Sector (industry) momentum features from daily prices + CSRC industry.

For each stock-date: peer (same industry, ex-self) aggregates of the
features already in the dataset: mean ret_5, mean ret_20, breadth of
ret_5 (share of peers > 0), mean vol_ratio_20. All causal ( peers' values
at the same date t use closes up to t, same convention as own features).

Returns a frame keyed by (date, code) with columns ind_*.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def build_sector(ds: pd.DataFrame) -> pd.DataFrame:
    ind = pd.read_parquet(ROOT / "data/f10/industry.parquet")
    grp = {c: i[:3] for c, i in zip(ind["code"], ind["industry"])}
    d = ds[["date", "code", "ret_5", "ret_20", "vol_ratio_20"]].copy()
    d["ind"] = d["code"].map(grp)
    g = d.groupby(["date", "ind"])
    agg = g.agg(sum_r5=("ret_5", "sum"), sum_r20=("ret_20", "sum"),
                sum_vr=("vol_ratio_20", "sum"),
                sum_up=("ret_5", lambda s: (s > 0).sum()),
                n=("ret_5", "size")).reset_index()
    d = d.merge(agg, on=["date", "ind"], how="left")
    denom = (d["n"] - 1).clip(lower=1)
    d["ind_ret5"] = (d["sum_r5"] - d["ret_5"]) / denom
    d["ind_ret20"] = (d["sum_r20"] - d["ret_20"]) / denom
    d["ind_breadth5"] = (d["sum_up"] - (d["ret_5"] > 0).astype(float)) / denom
    d["ind_volr"] = (d["sum_vr"] - d["vol_ratio_20"]) / denom
    return d[["date", "code", "ind_ret5", "ind_ret20", "ind_breadth5",
              "ind_volr"]]
