"""Fetch the macro/market-structure series named by the factor checklist.

Each series is independent: failures are logged, never fatal.
Output: data/extra/<name>.parquet + reports/fetch_macro.log summary.
Sources are AKShare endpoints on sina / chinabond / exchanges / SAFE /
NBS where possible (eastmoney endpoints are tried last; this IP gets
rate-limited by eastmoney).
"""
from __future__ import annotations

import time
import traceback
from pathlib import Path

import akshare as ak
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "extra"
LOG = ROOT / "reports" / "fetch_macro.log"


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def save(name: str, df: pd.DataFrame) -> None:
    if df is None or len(df) == 0:
        log(f"{name}: EMPTY")
        return
    df.to_parquet(OUT / f"{name}.parquet", index=False)
    log(f"{name}: {len(df):,} rows, cols={list(df.columns)[:6]}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [
        ("shibor", lambda: ak.macro_china_shibor_all()),
        ("lpr", lambda: ak.macro_china_lpr()),
        ("bond_yield_cn", lambda: ak.bond_china_yield(
            start_date="20150101", end_date="20261109")),
        ("cpi", lambda: ak.macro_china_cpi()),
        ("ppi", lambda: ak.macro_china_ppi()),
        ("pmi", lambda: ak.macro_china_pmi()),
        ("m2", lambda: ak.macro_china_m2_yearly()),
        ("social_finance", lambda: ak.macro_china_shrzgm()),
        ("margin_sse", lambda: ak.stock_margin_sse(
            start_date="20150101", end_date="20261109")),
        ("margin_szse", lambda: ak.stock_margin_szse(
            date="20261106")),
        ("hsgt_north", lambda: ak.stock_hsgt_hist_em(symbol="北向资金")),
        ("crude_sc", lambda: ak.futures_main_sina(
            symbol="SC0", start_date="20150101", end_date="20261109")),
        ("index_hs300", lambda: ak.stock_zh_index_daily(symbol="sh000300")),
        ("index_sse", lambda: ak.stock_zh_index_daily(symbol="sh000001")),
        ("restricted_release", lambda: ak.stock_restricted_release_queue_em(
            symbol="全部股票")),
        ("hold_change", lambda: ak.stock_hold_num_cninfo(date="20260331")),
    ]
    for name, fn in jobs:
        try:
            save(name, fn())
        except Exception:  # noqa: BLE001
            log(f"{name}: FAIL {traceback.format_exc(limit=1)[-200:]}")
        time.sleep(1)
    log("ALL DONE")


if __name__ == "__main__":
    main()
