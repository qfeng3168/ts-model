"""Fetch xdxr (除权除息) records and raw (unadjusted) daily bars via TDX/mootdx.

QUANTAXIS design point: keep the corporate-action records locally and do
adjustment locally, instead of trusting a vendor's adjusted series.
TDX servers proved reachable from this VM on 180.153.18.170 (2026-11-08);
other TDX servers reset or timed out, so keep the fallback list short.

Output: data/xdxr/{code}.parquet, data/daily_raw.parquet
Usage: python src/fetch_xdxr.py
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from mootdx.quotes import Quotes

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "xdxr"
SERVERS = [("180.153.18.170", 7709), ("119.147.212.81", 7709), ("218.6.170.47", 7709)]


def connect():
    last = None
    for srv in SERVERS:
        try:
            c = Quotes.factory(market="std", server=srv, timeout=8)
            c.xdxr(symbol="600519")  # protocol probe
            print(f"connected: {srv}", flush=True)
            return c
        except Exception as exc:
            last = exc
            print(f"server {srv} failed: {exc}", flush=True)
    raise RuntimeError(f"no TDX server reachable: {last}")


def fetch_raw_daily(client, code: str) -> pd.DataFrame:
    frames = []
    start = 0
    while True:
        df = client.bars(symbol=code, frequency=9, start=start, offset=800)
        if df is None or len(df) == 0:
            break
        frames.append(df)
        if len(df) < 800:
            break
        start += 800
    if not frames:
        return pd.DataFrame()
    d = pd.concat(frames)
    d["date"] = pd.to_datetime(d["datetime"]).dt.normalize()
    d = d[d["date"] >= "2014-06-01"]
    out = d[["date", "open", "high", "low", "close", "volume"]].copy()
    out["code"] = code
    return out.sort_values("date").drop_duplicates("date").reset_index(drop=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    codes = sorted(pd.read_parquet(
        ROOT / "data" / "daily.parquet", columns=["code"])["code"].unique())
    print(f"constituents: {len(codes)}", flush=True)
    client = connect()
    raw_frames, xf, rf = [], 0, 0
    for i, code in enumerate(codes, 1):
        xp = OUT / f"{code}.parquet"
        if not xp.exists():
            try:
                x = client.xdxr(symbol=code)
                if x is not None and len(x):
                    x = x.copy()
                    x["code"] = code
                    x.to_parquet(xp, index=False)
            except Exception as exc:
                print(f"[xdxr {code}] {exc}", flush=True)
                xf += 1
        try:
            raw_frames.append(fetch_raw_daily(client, code))
        except Exception as exc:
            print(f"[raw {code}] {exc}", flush=True)
            rf += 1
            try:
                client = connect()
            except Exception:
                time.sleep(5)
                client = connect()
        if i % 50 == 0 or i == len(codes):
            print(f"[{i}/{len(codes)}] xdxr_fail={xf} raw_fail={rf}", flush=True)
    raw = pd.concat(raw_frames, ignore_index=True)
    raw.to_parquet(ROOT / "data" / "daily_raw.parquet", index=False)
    print(f"DONE daily_raw: {len(raw):,} rows, stocks={raw['code'].nunique()}, "
          f"{raw['date'].min().date()}..{raw['date'].max().date()}", flush=True)


if __name__ == "__main__":
    main()
