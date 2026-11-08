"""Session-aware resampling of A-share minute bars (QUANTAXIS-style design).

Store only the finest granularity (1-min) and derive coarser bars locally
instead of fetching every frequency. A-share sessions are 09:30-11:30 and
13:00-15:00; bars are end-labelled (09:35 = the bar covering 09:30-09:35).
Resampling never crosses the lunch break or the overnight gap: within each
session, consecutive 1-min bars are chunked from the session open.

Usage:
  python src/resample.py --code 600519 --minutes 5 \
      --src data/minute1/raw --dst data/minute5_local/raw
  python src/resample.py --selftest
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SESSIONS = [(9 * 60 + 30, 11 * 60 + 30), (13 * 60, 15 * 60)]  # minutes of day, [start, end]


def resample_bars(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """df: 1-min bars with columns datetime, open, high, low, close, volume, amount[, code]."""
    if df.empty:
        return df.copy()
    d = df.sort_values("datetime").copy()
    mins_all = d["datetime"].dt.hour * 60 + d["datetime"].dt.minute  # end labels 571..690, 781..900
    in_session = ((mins_all > 9 * 60 + 30) & (mins_all <= 11 * 60 + 30)) | \
                 ((mins_all > 13 * 60) & (mins_all <= 15 * 60))
    d = d[in_session].copy()
    if d.empty:
        return d
    mins_f = d["datetime"].dt.hour * 60 + d["datetime"].dt.minute
    sess_f = np.where(mins_f <= 11 * 60 + 30, 0, 1)
    start_f = np.where(sess_f == 0, 9 * 60 + 30, 13 * 60)
    pos_f = mins_f - start_f
    d["_day"] = d["datetime"].dt.normalize()
    d["_grp"] = list(zip(d["_day"], sess_f, (pos_f - 1) // minutes))
    agg = {"open": "first", "high": "max", "low": "min", "close": "last",
           "volume": "sum", "datetime": "last"}
    if "amount" in d.columns:
        agg["amount"] = "sum"
    if "code" in d.columns:
        agg["code"] = "first"
    out = d.groupby("_grp", sort=True).agg(agg).reset_index(drop=True)
    return out.sort_values("datetime").reset_index(drop=True)


def selftest() -> None:
    # one synthetic day: 240 one-min bars, price = bar index, volume = 1
    rows = []
    base = pd.Timestamp("2026-11-06")
    for start, _ in SESSIONS:
        for i in range(1, 121):
            t = base + pd.Timedelta(minutes=start + i)
            p = float(start + i)
            rows.append({"datetime": t, "open": p, "high": p, "low": p,
                         "close": p, "volume": 1.0, "amount": p, "code": "TEST"})
    df = pd.DataFrame(rows)
    out5 = resample_bars(df, 5)
    assert len(out5) == 48, len(out5)
    first = out5.iloc[0]
    assert first["datetime"] == base + pd.Timedelta(minutes=9 * 60 + 35)
    assert first["open"] == 571 and first["close"] == 575 and first["volume"] == 5
    last_am = out5.iloc[23]
    assert last_am["datetime"] == base + pd.Timedelta(minutes=11 * 60 + 30)
    first_pm = out5.iloc[24]
    assert first_pm["datetime"] == base + pd.Timedelta(minutes=13 * 60 + 5), first_pm["datetime"]
    out15 = resample_bars(df, 15)
    assert len(out15) == 16
    out60 = resample_bars(df, 60)
    assert len(out60) == 4 and out60.iloc[0]["volume"] == 60
    print("selftest OK: 240x1min -> 48x5min / 16x15min / 4x60min, lunch break respected")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--code")
    ap.add_argument("--minutes", type=int, default=5)
    ap.add_argument("--src", default="data/minute1/raw")
    ap.add_argument("--dst", default="data/minute5_local/raw")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    src = ROOT / args.src / f"{args.code}.parquet"
    dst_dir = ROOT / args.dst
    dst_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(src)
    if "datetime" not in df.columns and "ts" in df.columns:
        df = df.rename(columns={"ts": "datetime"})
    out = resample_bars(df, args.minutes)
    out.to_parquet(dst_dir / f"{args.code}.parquet", index=False)
    print(f"{args.code}: {len(df)} x 1min -> {len(out)} x {args.minutes}min")


if __name__ == "__main__":
    main()
