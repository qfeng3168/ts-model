"""Fetch long-history 5-minute bars from baostock (free, from 2020-01).

baostock is the only free source with multi-year 5-min history (Sina only
keeps ~2 months). Front-adjusted (adjustflag=2) to match the daily store.
Output: data/minute5_hist/raw/{code}.parquet, resume-safe (skips existing).

Usage: python src/fetch_min5_baostock.py [--start 2020-01-01] [--end 2026-11-06]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import baostock as bs
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "minute5_hist" / "raw"
FIELDS = "date,time,open,high,low,close,volume,amount"


def bs_code(code: str) -> str:
    return ("sh." if code.startswith("6") else "sz.") + code


def fetch_one(code: str, start: str, end: str) -> pd.DataFrame:
    frames = []
    year0, year1 = int(start[:4]), int(end[:4])
    for year in range(year0, year1 + 1):
        s = max(start, f"{year}-01-01")
        e = min(end, f"{year}-12-31")
        rs = bs.query_history_k_data_plus(
            bs_code(code), FIELDS, start_date=s, end_date=e,
            frequency="5", adjustflag="2")
        if rs.error_code != "0":
            raise RuntimeError(f"{code} {year}: {rs.error_msg}")
        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        if rows:
            frames.append(pd.DataFrame(rows, columns=FIELDS.split(",")))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["datetime"] = pd.to_datetime(df["time"].str[:14], format="%Y%m%d%H%M%S")
    for c in ["open", "high", "low", "close", "volume", "amount"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["code"] = code
    return df[["datetime", "open", "high", "low", "close", "volume", "amount", "code"]] \
        .dropna(subset=["close"]).sort_values("datetime").reset_index(drop=True)


def worker(task: tuple[list[str], str, str]) -> tuple[int, int, int]:
    """One process, one baostock session, a disjoint shard of codes."""
    codes, start, end = task
    lg = bs.login()
    assert lg.error_code == "0", lg.error_msg
    ok = fail = rows = 0
    try:
        for code in codes:
            out = OUT / f"{code}.parquet"
            if out.exists():
                continue
            try:
                df = fetch_one(code, start, end)
            except Exception as exc:
                print(f"[{code}] {exc}; re-login + retry", flush=True)
                try:
                    bs.logout()
                except Exception:
                    pass
                time.sleep(2)
                bs.login()
                try:
                    df = fetch_one(code, start, end)
                except Exception as exc2:
                    print(f"[{code}] FAILED: {exc2}", flush=True)
                    fail += 1
                    continue
            if df.empty:
                print(f"[{code}] empty", flush=True)
                fail += 1
                continue
            df.to_parquet(out, index=False)
            ok += 1
            rows += len(df)
            print(f"[{code}] ok rows={len(df):,}", flush=True)
    finally:
        bs.logout()
    return ok, fail, rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default="2026-11-06")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    codes = sorted(pd.read_parquet(
        ROOT / "data" / "daily.parquet", columns=["code"])["code"].unique())
    todo = [c for c in codes if not (OUT / f"{c}.parquet").exists()]
    print(f"constituents: {len(codes)}, todo: {len(todo)}, "
          f"workers: {args.workers}", flush=True)
    if not todo:
        print("DONE nothing to do", flush=True)
        return
    shards = [todo[i::args.workers] for i in range(args.workers)]
    from multiprocessing import Pool
    with Pool(args.workers) as pool:
        results = pool.map(worker, [(s, args.start, args.end) for s in shards])
    ok = sum(r[0] for r in results)
    fail = sum(r[1] for r in results)
    rows = sum(r[2] for r in results)
    print(f"DONE ok={ok} fail={fail} rows={rows:,}", flush=True)


if __name__ == "__main__":
    main()
