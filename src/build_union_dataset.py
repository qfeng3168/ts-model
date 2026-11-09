"""Assemble dataset_union.parquet: D6's 59 features on the point-in-time
HS300 universe (682 stocks), plus a `member` flag per (date, code).

Faithfulness strategy: the original builder functions run unchanged;
only their file reads are routed to combined inputs (core_index +
core_index_ext, dividends + ext, industry + ext). Sector peers are
restricted to same-day index members. Membership is a step function
from the monthly baostock snapshots (latest snapshot <= t).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features  # noqa: E402
import features_sector  # noqa: E402
import features_ta  # noqa: E402
import features_v3  # noqa: E402
from features_cmdty import build_cmdty  # noqa: E402
from features_macro2 import build_surprise  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

_orig_read = pd.read_parquet


def _combined(name: str, ext_name: str) -> pd.DataFrame:
    base = _orig_read(DATA / "f10" / name)
    ext_p = DATA / "f10" / ext_name
    if ext_p.exists():
        ext = _orig_read(ext_p)
        return pd.concat([base, ext], ignore_index=True)
    print(f"WARN: {ext_name} missing, using base only", flush=True)
    return base


def routed(path, **kw):
    p = str(path)
    if p.endswith("f10/core_index.parquet"):
        return _combined("core_index.parquet", "core_index_ext.parquet")
    if p.endswith("f10/dividends.parquet"):
        return _combined("dividends.parquet", "dividends_ext.parquet")
    if p.endswith("f10/industry.parquet"):
        return _combined("industry.parquet", "industry_ext.parquet")
    if p.endswith("data/daily.parquet"):
        return _orig_read(DATA / "daily_v2.parquet", **kw)
    return _orig_read(path, **kw)


def membership(daily: pd.DataFrame) -> pd.DataFrame:
    snaps = _orig_read(DATA / "hs300_membership.parquet")
    S = np.sort(snaps["snap_date"].unique())
    codes = np.sort(snaps["code"].unique())
    cidx = {c: i for i, c in enumerate(codes)}
    sidx = {d: i for i, d in enumerate(S)}
    P = np.zeros((len(codes), len(S)), dtype=bool)
    P[snaps["code"].map(cidx), snaps["snap_date"].map(sidx)] = True
    T = np.sort(daily["date"].unique())
    pos = np.searchsorted(S, T, side="right") - 1
    M = np.zeros((len(codes), len(T)), dtype=bool)
    ok = pos >= 0
    M[:, ok] = P[:, pos[ok]]
    mem = pd.DataFrame(M.T, index=pd.Index(T, name="date"),
                       columns=codes)
    long = mem.stack().rename("member").reset_index()
    long.columns = ["date", "code", "member"]
    return long


def main() -> None:
    daily = _orig_read(DATA / "daily_v2.parquet")
    mem = membership(daily)
    base = features.build(daily)
    base = base.merge(mem, on=["date", "code"], how="left")
    base["member"] = base["member"].fillna(False)
    print(f"base: {len(base):,} rows, member rows="
          f"{base['member'].sum():,}", flush=True)

    pd.read_parquet = routed
    try:
        f10 = features_v3.build_f10(base, daily)
        div = features_v3.build_div(base, daily)
        ds = base.merge(f10, on=["date", "code"], how="left") \
                 .merge(div, on=["date", "code"], how="left")
        print("f10 merged", flush=True)
        sec = features_sector.build_sector(base[base["member"]])
        ds = ds.merge(sec, on=["date", "code"], how="left")
        print("sector (member peers) merged", flush=True)
        cmd = build_cmdty(base)
        ds = ds.merge(cmd, on=["date", "code"], how="left")
        print("cmdty merged", flush=True)
        sur = build_surprise(base)
        ds = ds.merge(sur, on=["date", "code"], how="left")
        print("surprise merged", flush=True)
        ta = features_ta.build_ta(base)
        ds = ds.merge(ta, on=["date", "code"], how="left")
        print("ta merged", flush=True)
    finally:
        pd.read_parquet = _orig_read
    ds.to_parquet(DATA / "dataset_union.parquet", index=False)
    print(f"dataset_union: {len(ds):,} rows x {ds.shape[1]} cols, "
          f"stocks={ds['code'].nunique()}", flush=True)


if __name__ == "__main__":
    main()
