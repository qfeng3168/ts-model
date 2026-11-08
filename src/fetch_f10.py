"""Fetch F10 fundamentals for HS300 constituents, point-in-time safe.

Primary: adata get_core_index -> 43 core indicators per report period with
BOTH report_date and notice_date (announcement date). Features may only be
joined on notice_date; joining on report_date would be a future function.

Secondary: AKShare dividend detail (carries its own disclosure dates).

Output: data/f10/core_index.parquet, data/f10/dividends.parquet.
Usage: python src/fetch_f10.py
"""
from __future__ import annotations

import time
from pathlib import Path

import adata
import akshare as ak
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "f10"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    codes = sorted(pd.read_parquet(
        ROOT / "data" / "daily.parquet", columns=["code"])["code"].unique())
    print(f"constituents: {len(codes)}", flush=True)

    core_frames, div_frames = [], []
    core_fail, div_fail = [], []
    for i, code in enumerate(codes, 1):
        try:
            df = adata.stock.finance.get_core_index(stock_code=code)
            if df is not None and len(df):
                core_frames.append(df)
            else:
                core_fail.append(code)
        except Exception as exc:
            print(f"[core {code}] {exc}", flush=True)
            core_fail.append(code)
        try:
            dv = ak.stock_history_dividend_detail(symbol=code, indicator="分红")
            if dv is not None and len(dv):
                dv = dv.copy()
                dv["code"] = code
                div_frames.append(dv)
        except Exception as exc:
            print(f"[div {code}] {exc}", flush=True)
            div_fail.append(code)
        if i % 50 == 0 or i == len(codes):
            print(f"[{i}/{len(codes)}] core_fail={len(core_fail)} "
                  f"div_fail={len(div_fail)}", flush=True)
        time.sleep(0.2)

    if core_frames:
        core = pd.concat(core_frames, ignore_index=True)
        core["report_date"] = pd.to_datetime(core["report_date"])
        core["notice_date"] = pd.to_datetime(core["notice_date"])
        core.to_parquet(OUT / "core_index.parquet", index=False)
        print(f"core_index: {len(core):,} rows, stocks={core['stock_code'].nunique()}, "
              f"notice {core['notice_date'].min().date()}..{core['notice_date'].max().date()}",
              flush=True)
    if div_frames:
        div = pd.concat(div_frames, ignore_index=True)
        div.to_parquet(OUT / "dividends.parquet", index=False)
        print(f"dividends: {len(div):,} rows, stocks={div['code'].nunique()}, "
              f"cols={list(div.columns)}", flush=True)
    print(f"DONE core_fail={core_fail[:10]}({len(core_fail)}) "
          f"div_fail={div_fail[:10]}({len(div_fail)})", flush=True)


if __name__ == "__main__":
    main()
