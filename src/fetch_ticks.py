"""Fetch per-tick transactions （分笔成交） via TDX/mootdx and store locally.

TDX servers keep deep tick history (600519 verified back to 2024-11), but
retention is server-dependent and not contractual, so the operating model
is still accumulate-daily: this script is resume-safe and idempotent, and
the daily cron calls it for the just-closed trading day.

Output: data/ticks/{code}/{YYYYMMDD}.parquet
Usage: python src/fetch_ticks.py [--date 20261106] [--codes 600519,000001]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from mootdx.quotes import Quotes

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "ticks"
SERVERS = [("180.153.18.170", 7709), ("119.147.212.81", 7709), ("218.6.170.47", 7709)]
PAGE = 2000


def connect():
    last = None
    for srv in SERVERS:
        try:
            c = Quotes.factory(market="std", server=srv, timeout=8)
            c.xdxr(symbol="600519")
            print(f"connected: {srv}", flush=True)
            return c
        except Exception as exc:
            last = exc
            print(f"server {srv} failed: {exc}", flush=True)
    raise RuntimeError(f"no TDX server reachable: {last}")


def fetch_day(client, code: str, date: str) -> pd.DataFrame:
    frames = []
    start = 0
    while True:
        df = client.transactions(symbol=code, date=date, start=start, offset=PAGE)
        if df is None or len(df) == 0:
            break
        frames.append(df)
        if len(df) < PAGE:
            break
        start += PAGE
        if start > 40000:  # safety bound, a day never has this many ticks
            break
    if not frames:
        return pd.DataFrame()
    # pages arrive newest-first, but rows inside a page are chronological:
    # reverse the page order only, never the rows.
    d = pd.concat(frames[::-1], ignore_index=True)
    keep = [c for c in ["time", "price", "vol", "buyorsell", "volume"] if c in d.columns]
    d = d[keep].copy()
    d["date"] = date
    d["code"] = code
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="20261106")
    ap.add_argument("--codes", default="")
    args = ap.parse_args()

    if args.codes:
        codes = args.codes.split(",")
    else:
        codes = sorted(pd.read_parquet(
            ROOT / "data" / "daily.parquet", columns=["code"])["code"].unique())
    print(f"date={args.date} stocks={len(codes)}", flush=True)
    client = connect()
    ok = skip = fail = rows = 0
    for i, code in enumerate(codes, 1):
        out_dir = OUT / code
        out = out_dir / f"{args.date}.parquet"
        if out.exists():
            skip += 1
            continue
        try:
            df = fetch_day(client, code, args.date)
        except Exception as exc:
            print(f"[{code}] {exc}; reconnect + retry", flush=True)
            fail += 1
            try:
                client = connect()
                df = fetch_day(client, code, args.date)
                fail -= 1
            except Exception as exc2:
                print(f"[{code}] FAILED: {exc2}", flush=True)
                continue
        if df.empty:
            print(f"[{code}] empty (non-trading day or no retention)", flush=True)
            fail += 1
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        ok += 1
        rows += len(df)
        if i % 50 == 0 or i == len(codes):
            print(f"[{i}/{len(codes)}] ok={ok} skip={skip} fail={fail} "
                  f"rows={rows:,}", flush=True)
        time.sleep(0.05)
    print(f"DONE date={args.date} ok={ok} skip={skip} fail={fail} rows={rows:,}",
          flush=True)


if __name__ == "__main__":
    main()
