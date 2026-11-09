"""F10 + dividends + industry for the HS300 union extras.

Same sources as fetch_f10.py (adata core index, akshare dividends),
plus baostock industry classification. Outputs land in separate
*_ext.parquet files so the production F10 files stay untouched;
the union build merges them.
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
    union = pd.read_parquet(ROOT / "data/hs300_union.parquet")
    current = set(pd.read_parquet(ROOT / "data/daily.parquet",
                                  columns=["code"])["code"].unique())
    have_raw = {p.stem for p in (ROOT / "data/union_raw").glob("*.parquet")}
    extras = sorted((set(union["code"]) - current) & have_raw)
    print(f"extras with price data: {len(extras)}", flush=True)
    core_frames, div_frames = [], []
    core_fail, div_fail = [], []
    for i, code in enumerate(extras, 1):
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
            dv = ak.stock_history_dividend_detail(symbol=code,
                                                  indicator="分红")
            if dv is not None and len(dv):
                dv = dv.copy()
                dv["code"] = code
                div_frames.append(dv)
        except Exception:
            div_fail.append(code)
        if i % 50 == 0 or i == len(extras):
            print(f"[{i}/{len(extras)}] core_fail={len(core_fail)} "
                  f"div_fail={len(div_fail)}", flush=True)
        time.sleep(0.2)
    if core_frames:
        core = pd.concat(core_frames, ignore_index=True)
        core["report_date"] = pd.to_datetime(core["report_date"])
        core["notice_date"] = pd.to_datetime(core["notice_date"])
        core.to_parquet(OUT / "core_index_ext.parquet", index=False)
        print(f"core_index_ext: {len(core):,} rows, "
              f"stocks={core['stock_code'].nunique()}", flush=True)
    if div_frames:
        div = pd.concat(div_frames, ignore_index=True)
        div.to_parquet(OUT / "dividends_ext.parquet", index=False)
        print(f"dividends_ext: {len(div):,} rows", flush=True)
    print(f"DONE core_fail={len(core_fail)} div_fail={len(div_fail)}",
          flush=True)

    import baostock as bs
    bs.login()
    rows = []
    for code in extras:
        pre = "sh." if code.startswith("6") else "sz."
        rs = bs.query_stock_industry(code=pre + code)
        while rs.next():
            rows.append(rs.get_row_data())
    bs.logout()
    if rows:
        ind = pd.DataFrame(rows, columns=rs.fields)
        ind = ind.rename(columns={"code": "bs_code"})
        ind["code"] = ind["bs_code"].str.split(".").str[1]
        ind.to_parquet(OUT / "industry_ext.parquet", index=False)
        print(f"industry_ext: {len(ind)} rows", flush=True)


if __name__ == "__main__":
    main()
