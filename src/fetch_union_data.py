"""Fetch raw daily bars + xdxr for the HS300 union extras (382 codes
that were index members at some point but are not current members).

Raw bars: Sina via akshare (adjust=""), keeping amount/turnover.
Fallback for codes Sina no longer serves (delisted): baostock
unadjusted bars. xdxr: TDX via the existing fetch_xdxr.connect().
Checkpointed per code under data/union_raw/{code}.parquet and
data/xdxr/{code}.parquet (same store the 300 use).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import akshare as ak
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_xdxr import connect  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW_OUT = ROOT / "data" / "union_raw"
XDXR_OUT = ROOT / "data" / "xdxr"


def raw_sina(code: str) -> pd.DataFrame:
    sym = ("sh" if code.startswith("6") else "sz") + code
    d = ak.stock_zh_a_daily(symbol=sym, start_date="20140601",
                            end_date="20261106", adjust="")
    if d is None or len(d) == 0:
        return pd.DataFrame()
    d = d.copy()
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    keep = ["date", "open", "high", "low", "close", "volume", "amount",
            "turnover"]
    for c in keep:
        if c not in d.columns:
            d[c] = pd.NA
    d = d[keep]
    d["code"] = code
    return d


def raw_baostock(code: str) -> pd.DataFrame:
    import baostock as bs
    pre = "sh." if code.startswith("6") else "sz."
    rs = bs.query_history_k_data_plus(
        pre + code, "date,open,high,low,close,volume,amount",
        start_date="2014-06-01", end_date="2026-11-06",
        frequency="d", adjustflag="3")
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    if not rows:
        return pd.DataFrame()
    d = pd.DataFrame(rows, columns=rs.fields)
    for c in ["open", "high", "low", "close", "volume", "amount"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["date"] = pd.to_datetime(d["date"])
    d["turnover"] = pd.NA
    d["code"] = code
    return d[["date", "open", "high", "low", "close", "volume",
              "amount", "turnover", "code"]]


def main() -> None:
    RAW_OUT.mkdir(parents=True, exist_ok=True)
    union = pd.read_parquet(ROOT / "data/hs300_union.parquet")
    current = set(pd.read_parquet(ROOT / "data/daily.parquet",
                                  columns=["code"])["code"].unique())
    extras = sorted(set(union["code"]) - current)
    print(f"extras: {len(extras)}", flush=True)
    client = None
    import baostock as bs
    bs.login()
    fail_raw, fail_x = [], []
    for i, code in enumerate(extras, 1):
        rp = RAW_OUT / f"{code}.parquet"
        if not rp.exists():
            try:
                d = raw_sina(code)
                if len(d) == 0:
                    d = raw_baostock(code)
                if len(d):
                    d.to_parquet(rp, index=False)
                else:
                    fail_raw.append(code)
            except Exception as exc:
                try:
                    d = raw_baostock(code)
                    if len(d):
                        d.to_parquet(rp, index=False)
                    else:
                        fail_raw.append(code)
                except Exception:
                    fail_raw.append(code)
                    print(f"[raw {code}] {exc}", flush=True)
        xp = XDXR_OUT / f"{code}.parquet"
        if not xp.exists():
            try:
                if client is None:
                    client = connect()
                x = client.xdxr(symbol=code)
                if x is not None and len(x):
                    x = x.copy()
                    x["code"] = code
                    x.to_parquet(xp, index=False)
                else:
                    fail_x.append(code)
            except Exception as exc:
                fail_x.append(code)
                print(f"[xdxr {code}] {exc}", flush=True)
                client = None
        if i % 50 == 0 or i == len(extras):
            print(f"[{i}/{len(extras)}] raw_fail={len(fail_raw)} "
                  f"xdxr_fail={len(fail_x)}", flush=True)
        time.sleep(0.3)
    bs.logout()
    print(f"DONE raw_fail={fail_raw}")
    print(f"DONE xdxr_fail={fail_x}")


if __name__ == "__main__":
    main()
