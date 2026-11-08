"""P0: 拉取沪深300成分股日线（前复权），逐股断点落盘，最后合并为 daily.parquet。

数据源：AKShare（东方财富）。成分股为当前名单——v0 已知幸存者偏差，报告中注明。
"""
import time
import sys
from pathlib import Path

import akshare as ak
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)
START = "20150101"
END = pd.Timestamp.today().strftime("%Y%m%d")


def _prefix(code: str) -> str:
    if code.startswith(("6", "9")):
        return "sh"
    if code.startswith(("4", "8")):
        return "bj"
    return "sz"


def fetch_one(code: str) -> pd.DataFrame | None:
    fp = RAW / f"{code}.parquet"
    if fp.exists():
        return pd.read_parquet(fp)
    for attempt in range(3):
        try:
            # 新浪源（东方财富 push2 对本 IP 限流，2026-11-08 实测第 2 只即断）
            df = ak.stock_zh_a_daily(
                symbol=f"{_prefix(code)}{code}", start_date=START, end_date=END, adjust="qfq"
            )
            if df is None or df.empty:
                return None
            df = df.reset_index() if "date" not in df.columns else df
            df["date"] = pd.to_datetime(df["date"])
            if "amount" not in df.columns:
                df["amount"] = df["volume"] * df["close"]
            if "turnover" not in df.columns:
                df["turnover"] = pd.NA
            df = df[["date", "open", "close", "high", "low", "volume", "amount", "turnover"]]
            df["code"] = code
            df.to_parquet(fp, index=False)
            return df
        except Exception as e:
            print(f"  {code} attempt {attempt+1} failed: {e}", flush=True)
            time.sleep(2 + attempt * 3)
    return None


def main() -> None:
    cons = ak.index_stock_cons_csindex(symbol="000300")
    codes = sorted(cons["成分券代码"].astype(str).str.zfill(6).unique())
    print(f"CSI300 constituents: {len(codes)}", flush=True)
    ok, fail = 0, []
    for i, code in enumerate(codes, 1):
        df = fetch_one(code)
        if df is None or df.empty:
            fail.append(code)
        else:
            ok += 1
        if i % 50 == 0:
            print(f"[{i}/{len(codes)}] ok={ok} fail={len(fail)}", flush=True)
        time.sleep(0.15)
    frames = [pd.read_parquet(p) for p in sorted(RAW.glob("*.parquet"))]
    all_df = pd.concat(frames, ignore_index=True).sort_values(["code", "date"])
    out = ROOT / "data" / "daily.parquet"
    all_df.to_parquet(out, index=False)
    print(f"saved {out}: rows={len(all_df)} stocks={all_df['code'].nunique()} "
          f"{all_df['date'].min().date()}..{all_df['date'].max().date()} fail={fail}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
