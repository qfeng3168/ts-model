"""Raw (unadjusted) daily bars from Sina via AKShare (adjust="").

Fallback for the TDX bars channel: the reachable TDX servers answer xdxr
but return empty for get_security_bars (verified 2026-11-09 on two
servers). Only close + date feed the factor math in adjust.py.

Output: data/daily_raw.parquet (date, open, high, low, close, volume, code)
"""
from __future__ import annotations

import time
from pathlib import Path

import akshare as ak
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "reports" / "fetch_raw_sina.log"


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def main() -> None:
    codes = sorted(pd.read_parquet(
        ROOT / "data" / "daily.parquet", columns=["code"])["code"].unique())
    frames, fail = [], []
    for i, code in enumerate(codes, 1):
        sym = ("sh" if code.startswith("6") else "sz") + code
        try:
            d = ak.stock_zh_a_daily(symbol=sym, start_date="20140601",
                                    end_date="20261106", adjust="")
            d = d[["date", "open", "high", "low", "close", "volume"]].copy()
            d["date"] = pd.to_datetime(d["date"]).dt.normalize()
            d["code"] = code
            frames.append(d)
        except Exception as exc:  # noqa: BLE001
            fail.append(code)
            log(f"[{i}/{len(codes)}] {code} FAIL {exc}")
        if i % 50 == 0:
            log(f"[{i}/{len(codes)}] ok={len(frames)} fail={len(fail)}")
    raw = pd.concat(frames, ignore_index=True)
    raw.to_parquet(ROOT / "data" / "daily_raw.parquet", index=False)
    log(f"DONE daily_raw: {len(raw):,} rows, stocks={raw['code'].nunique()}, "
        f"fail={fail}")


if __name__ == "__main__":
    main()
