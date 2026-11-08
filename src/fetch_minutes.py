"""分钟线采集（新浪源，前复权）：1 分钟与 5 分钟双粒度，逐股合并落盘、按时间戳去重。

用法：fetch_minutes.py [1|5|all]（默认 all）
落盘：data/minute{period}/raw/{code}.parquet
首次运行拉回新浪上限约 1970 根（5 分钟约 2 个月、1 分钟约 9 天）；
每日收盘后重跑即增量补当天，历史深度逐日累积。
"""
import sys
import time
from pathlib import Path

import akshare as ak
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def _prefix(code: str) -> str:
    if code.startswith(("6", "9")):
        return "sh"
    if code.startswith(("4", "8")):
        return "bj"
    return "sz"


def fetch_one(code: str, period: str, raw_dir: Path) -> int:
    fp = raw_dir / f"{code}.parquet"
    try:
        df = ak.stock_zh_a_minute(symbol=f"{_prefix(code)}{code}", period=period, adjust="qfq")
    except Exception as e:
        print(f"  [{period}m] {code} failed: {e}", flush=True)
        return 0
    if df is None or df.empty:
        return 0
    df = df.rename(columns={"day": "ts"})
    df["ts"] = pd.to_datetime(df["ts"])
    df["code"] = code
    df = df[["ts", "open", "high", "low", "close", "volume", "code"]]
    if fp.exists():
        old = pd.read_parquet(fp)
        df = pd.concat([old, df], ignore_index=True)
    df = df.drop_duplicates(subset=["ts"]).sort_values("ts")
    df.to_parquet(fp, index=False)
    return len(df)


def run(period: str, codes: list[str]) -> None:
    raw_dir = ROOT / "data" / f"minute{period}" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    ok = 0
    for i, code in enumerate(codes, 1):
        if fetch_one(code, period, raw_dir) > 0:
            ok += 1
        if i % 100 == 0:
            print(f"[{period}m {i}/{len(codes)}] ok={ok}", flush=True)
        time.sleep(0.15)
    files = list(raw_dir.glob("*.parquet"))
    total = sum(len(pd.read_parquet(p, columns=["ts"])) for p in files)
    print(f"[{period}m] done: files={len(files)} total_bars={total}", flush=True)


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    periods = ["5", "1"] if which == "all" else [which]
    cons = ak.index_stock_cons_csindex(symbol="000300")
    codes = sorted(cons["成分券代码"].astype(str).str.zfill(6).unique())
    print(f"constituents: {len(codes)}, periods: {periods}", flush=True)
    for p in periods:
        run(p, codes)


if __name__ == "__main__":
    main()
